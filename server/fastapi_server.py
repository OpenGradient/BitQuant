from typing import List, Any, Dict, Tuple
import os
import json
import traceback
import logging
import asyncio
import re

from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import ValidationError
from langchain_core.runnables.config import RunnableConfig
from datadog import initialize, statsd
import aiohttp

from onchain.pools.protocol import ProtocolRegistry
from onchain.pools.solana.orca_protocol import OrcaProtocol
from onchain.pools.solana.save_protocol import SaveProtocol
from onchain.pools.solana.kamino_protocol import KaminoProtocol
from onchain.tokens.metadata import TokenMetadataRepo
from onchain.portfolio.solana_portfolio import PortfolioFetcher
from api.api_types import (
    AgentChatRequest,
    AgentMessage,
    AgentType,
    Portfolio,
    Message,
    TokenMetadata,
    SolanaVerifyRequest,
    Context,
    UserMessage,
    ProcessSwapRequest,
    ProcessSwapResponse,
)
from agent.agent_executors import (
    create_investor_executor,
    create_suggestions_model,
    create_analytics_executor,
)
from agent.prompts import (
    get_investor_agent_prompt,
    get_suggestions_prompt,
    get_analytics_prompt,
)
from agent.tools import (
    create_investor_agent_toolkit,
    create_analytics_agent_toolkit,
)
from langchain_openai import ChatOpenAI
from server.invitecode import InviteCodeManager
from server.activity_tracker import ActivityTracker
from server.utils import extract_patterns, convert_to_agent_msg
from server.dynamodb_helpers import DatabaseManager
from server.middleware import DatadogMetricsMiddleware
from server.swap_tracker import SwapTracker
from server.jup_validator import JUPValidator
from server.cow_validator import COWValidator

from . import service
from .auth import FirebaseIDTokenData, get_current_user

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(ROOT_DIR, "static")

# Initialize Datadog
initialize(
    api_key=os.environ.get("DD_API_KEY"),
    app_key=os.environ.get("DD_APP_KEY"),
    host_name=os.environ.get("DD_HOSTNAME", "localhost"),
)

# number of messages to send to agents
NUM_MESSAGES_TO_KEEP = 6

# API key for whitelist management
API_KEY = os.environ.get("WHITELIST_API_KEY")


def create_fastapi_app() -> FastAPI:
    """Create and configure the FastAPI application with routes."""
    app = FastAPI()

    # Configure CORS
    # FastAPI CORSMiddleware doesn't support regex patterns, so we list all allowed origins
    allowed_origins = [
        "https://bitquant.io",
        "https://www.bitquant.io",
        # Localhost ports for development
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:3002",
        "http://localhost:4000",
        "http://localhost:4200",
        "http://localhost:5000",
        "http://localhost:5173",
        "http://localhost:8000",
        "http://localhost:8080",
        "http://localhost:8081",
        "http://localhost:9000",
    ]
    
    # Vercel preview deployments pattern (will be checked in custom middleware)
    vercel_pattern = re.compile(r"^https://defi-chat-hub-git-[\w-]+-open-gradient\.vercel\.app$")
    
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add Datadog metrics middleware
    app.add_middleware(DatadogMetricsMiddleware)
    
    # Add custom CORS check for Vercel preview deployments
    # This middleware runs after CORSMiddleware to handle regex patterns
    @app.middleware("http")
    async def cors_vercel_middleware(request: Request, call_next):
        origin = request.headers.get("origin")
        if origin and vercel_pattern.match(origin):
            response = await call_next(request)
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Credentials"] = "true"
            response.headers["Access-Control-Allow-Methods"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "*"
            return response
        return await call_next(request)

    # Initialize DynamoDB session
    database_manager = DatabaseManager()

    # Initialize services with their dependencies
    activity_tracker = ActivityTracker(
        database_manager.table_context_factory("twoligma_activity")
    )
    invite_manager = InviteCodeManager(
        database_manager.table_context_factory("twoligma_invite_codes"),
        activity_tracker,
    )
    token_metadata_repo = TokenMetadataRepo(
        database_manager.table_context_factory("token_metadata_v2")
    )
    portfolio_fetcher = PortfolioFetcher(token_metadata_repo)
    swap_tracker = SwapTracker(
        database_manager.table_context_factory("twoligma_processed_swaps")
    )
    jup_validator = JUPValidator(token_metadata_repo)
    cow_validator = COWValidator(token_metadata_repo)

    # Store services in app state for access in routes
    app.state.activity_tracker = activity_tracker
    app.state.invite_manager = invite_manager
    app.state.token_metadata_repo = token_metadata_repo
    app.state.portfolio_fetcher = portfolio_fetcher
    app.state.swap_tracker = swap_tracker
    app.state.jup_validator = jup_validator
    app.state.cow_validator = cow_validator

    @app.on_event("shutdown")
    async def shutdown_event():
        await protocol_registry.shutdown()
        await token_metadata_repo.close()
        await portfolio_fetcher.close()
        await jup_validator.close()
        await cow_validator.close()

    # Initialize agents
    suggestions_model = create_suggestions_model()
    analytics_agent = create_analytics_executor(token_metadata_repo)
    investor_agent = create_investor_executor()

    # Initialize protocol registry
    protocol_registry = ProtocolRegistry(token_metadata_repo)
    protocol_registry.register_protocol(OrcaProtocol())
    protocol_registry.register_protocol(SaveProtocol())
    protocol_registry.register_protocol(KaminoProtocol())

    # Store agents in app state
    app.state.suggestions_model = suggestions_model
    app.state.analytics_agent = analytics_agent
    app.state.investor_agent = investor_agent
    app.state.protocol_registry = protocol_registry

    @app.on_event("startup")
    async def startup_event():
        await protocol_registry.initialize()

    # Exception handlers
    @app.exception_handler(ValidationError)
    async def validation_exception_handler(request: Request, exc: ValidationError):
        logging.error(f"400 Error: {str(exc)}")
        return JSONResponse(
            status_code=400,
            content={"error": str(exc)},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):
        error_traceback = traceback.format_exc()
        logging.error(f"500 Error: {str(exc)}")
        logging.error(f"Traceback: {error_traceback}")
        logging.error(f"Request Path: {request.url.path}")
        logging.error(f"Request Body: {await request.body()}")

        return JSONResponse(
            status_code=500,
            content={"error": str(exc)},
        )

    # API key dependency
    async def require_api_key(x_api_key: str = Header(None)):
        if not x_api_key or x_api_key != API_KEY:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")
        return x_api_key

    async def verify_captcha_token(captchaToken: str):
        secret_key = os.getenv("CLOUDFLARE_TURNSTILE_SECRET_KEY")
        if not secret_key:
            raise Exception(
                "CLOUDFLARE_TURNSTILE_SECRET_KEY environment variable is not set"
            )

        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://challenges.cloudflare.com/turnstile/v0/siteverify",
                data={"secret": secret_key, "response": captchaToken},
                headers={"content-type": "application/x-www-form-urlencoded"},
            ) as response:
                result = await response.json()
                if result.get("success"):
                    return True
                else:
                    logging.error(f"Captcha verification failed: {result}")
                    return False

    # Routes
    @app.post("/api/cloudflare/turnstile/v0/siteverify")
    async def verify_cloudflare_turnstile_token(request: Request):
        try:
            secret_key = os.getenv("CLOUDFLARE_TURNSTILE_SECRET_KEY")
            if not secret_key:
                raise Exception(
                    "CLOUDFLARE_TURNSTILE_SECRET_KEY environment variable is not set"
                )

            data = await request.json()
            token = data.get("token")

            if not token:
                raise HTTPException(status_code=400, detail="Missing token")

            # Make the request to Cloudflare Turnstile using aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://challenges.cloudflare.com/turnstile/v0/siteverify",
                    data={"secret": secret_key, "response": token},
                    headers={"content-type": "application/x-www-form-urlencoded"},
                ) as response:
                    result = await response.json()
                    status_code = 200 if result.get("success") else 400
                    return JSONResponse(content=result, status_code=status_code)

        except Exception as e:
            logging.error(f"Error verifying Cloudflare Turnstile token: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/api/verify/solana")
    async def verify_solana_signature(request: Request):
        try:
            request_data = await request.json()
            verify_request = SolanaVerifyRequest(**request_data)

            token = await asyncio.to_thread(
                service.verify_solana_signature, verify_request
            )
            return {"token": token}

        except ValidationError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            logging.error(f"Error verifying SIWX signature: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.get("/api/healthcheck")
    async def healthcheck():
        return {"status": "ok"}

    @app.get("/api/whitelisted")
    async def is_whitelisted():
        return {"allowed": True}

    @app.get("/api/portfolio")
    async def get_portfolio(
        address: str,
        user: FirebaseIDTokenData = Depends(get_current_user),
    ):
        if not address:
            raise HTTPException(status_code=400, detail="Address parameter is required")

        portfolio = await portfolio_fetcher.get_portfolio(address)
        return portfolio.model_dump()

    @app.get("/api/tokenlist")
    async def get_tokenlist():
        file_path = os.path.join(STATIC_DIR, "tokenlist.json")
        if not os.path.isfile(file_path):
            raise HTTPException(status_code=404, detail="Tokenlist file not found")
        return FileResponse(file_path)

    @app.post("/api/v2/agent/run")
    async def run_agent(
        request: Request,
        user: FirebaseIDTokenData = Depends(get_current_user),
    ):
        request_data = await request.json()
        agent_request = AgentChatRequest(**request_data)

        if not agent_request.captchaToken:
            raise HTTPException(status_code=400, detail="Captcha token is required")

        # Increment message count, return 429 if limit reached
        if not await activity_tracker.increment_message_count(
            agent_request.context.address
        ):
            statsd.increment("agent.message.daily_limit_reached")
            raise HTTPException(status_code=429, detail="Daily message limit reached")

        portfolio = await portfolio_fetcher.get_portfolio(
            wallet_address=agent_request.context.address
        )

        # Restrict agent usage to funded wallets
        if portfolio.total_value_usd < 1:
            return AgentMessage(
                message="Please use a funded Solana wallet (at least $1) to start using the agent",
                pools=[],
                tokens=[],
            )

        try:
            response = await handle_agent_chat_request(
                token_metadata_repo=token_metadata_repo,
                protocol_registry=protocol_registry,
                request=agent_request,
                portfolio=portfolio,
                investor_agent=investor_agent,
                analytics_agent=analytics_agent,
            )

            return (
                response.model_dump() if hasattr(response, "model_dump") else response
            )
        except Exception as e:
            logging.error(f"Error processing agent request: {e}")
            raise

    @app.post("/api/v2/agent/suggestions")
    async def run_suggestions(
        request: Request,
        user: FirebaseIDTokenData = Depends(get_current_user),
    ):
        request_data = await request.json()
        agent_request = AgentChatRequest(**request_data)

        if not agent_request.captchaToken:
            raise HTTPException(status_code=400, detail="Captcha token is required")
        # if not await verify_captcha_token(agent_request.captchaToken):
        #     raise HTTPException(status_code=429, detail="Invalid captcha token")

        # Check if user has reached daily message limit (without incrementing)
        stats = await activity_tracker.get_activity_stats(agent_request.context.address)
        if stats.daily_message_count >= stats.daily_message_limit:
            statsd.increment("agent.suggestions.daily_limit_reached")
            raise HTTPException(status_code=429, detail="Daily message limit reached")

        portfolio = await portfolio_fetcher.get_portfolio(
            wallet_address=agent_request.context.address
        )

        # Restrict agent usage to funded wallets
        if portfolio.total_value_usd <= 1:
            return {"suggestions": []}

        suggestions = await handle_suggestions_request(
            token_metadata_repo=token_metadata_repo,
            request=agent_request,
            portfolio=portfolio,
            suggestions_model=suggestions_model,
        )
        return {"suggestions": suggestions}

    # TODO: Re-enable this endpoint if needed
    async def generate_invite_code(
        request: Request,
        user: FirebaseIDTokenData = Depends(get_current_user),
    ):
        try:
            request_data = await request.json()
            if not request_data or "address" not in request_data:
                raise HTTPException(status_code=400, detail="Address is required")

            creator_address = request_data["address"]

            # Generate invite code
            invite_code = await invite_manager.generate_invite_code(creator_address)
            if not invite_code:
                raise HTTPException(
                    status_code=500, detail="Failed to generate invite code"
                )

            return {"invite_code": invite_code}
        except Exception as e:
            logging.error(f"Error generating invite code: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/api/invite/use")
    async def use_invite_code(request: Request):
        try:
            return {"status": "ended"}
        except Exception as e:
            logging.error(f"Error using invite code: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.get("/api/activity/stats")
    async def get_activity_stats(
        address: str,
        user: FirebaseIDTokenData = Depends(get_current_user),
    ):
        try:
            if not address:
                raise HTTPException(
                    status_code=400, detail="Address parameter is required"
                )

            stats = await activity_tracker.get_activity_stats(address)
            return stats
        except Exception as e:
            logging.error(
                f"Error getting activity stats: {e}\nTraceback:\n{traceback.format_exc()}"
            )
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/api/process_swap")
    async def process_swap(
        request: ProcessSwapRequest,
        user: FirebaseIDTokenData = Depends(get_current_user),
    ):
        """
        Process a JUP swap transaction and award points based on referral rewards.

        Expected request body:
        {
            "txid": "transaction_id",
            "address": "wallet_address",
        }

        Returns:
        {
            "success": bool,
            "points_awarded": int,
            "referral_reward": float,
            "message": str
        }
        """
        try:
            # Get services from app state
            swap_tracker: SwapTracker = app.state.swap_tracker
            jup_validator: JUPValidator = app.state.jup_validator
            cow_validator: COWValidator = app.state.cow_validator
            activity_tracker: ActivityTracker = app.state.activity_tracker

            # Check if this swap has already been processed
            if await swap_tracker.is_swap_processed(request.chain, request.txid):
                return ProcessSwapResponse(
                    success=False,
                    points_awarded=0,
                    referral_reward=0.0,
                    message="Transaction has already been processed",
                )

            # Validate the swap transaction
            if request.chain == "solana":
                validation_result = await jup_validator.validate_swap_transaction(
                    request.txid
                )
                logging.info(f"Validation result: {validation_result}")
                if not validation_result or not validation_result.get("valid"):
                    return ProcessSwapResponse(
                        success=False,
                        points_awarded=0,
                        referral_reward=0.0,
                        message="Invalid or non-JUP swap transaction",
                    )

                # Get the actual referral reward from the transaction
                referral_reward_usdc = validation_result.get(
                    "referral_reward_usdc", 0.0
                )

                # Calculate points to award
                points_awarded = jup_validator.calculate_points_from_reward(
                    referral_reward_usdc
                )
            elif request.chain in ["ethereum", "base"]:
                # Handle CoW protocol swaps
                validation_result = await cow_validator.validate_swap_order(
                    request.txid, request.chain
                )
                if not validation_result or not validation_result.get("valid"):
                    return ProcessSwapResponse(
                        success=False,
                        points_awarded=0,
                        referral_reward=0.0,
                        message="Invalid or non-CoW swap order",
                    )

                # Get the actual referral reward from the order in USD
                referral_reward_usdc = validation_result.get(
                    "referral_reward_usdc", 0.0
                )
                points_awarded = validation_result.get("points_awarded", 0)
            else:
                raise HTTPException(status_code=400, detail="Invalid chain")

            # Award points to the user
            if points_awarded > 0:
                await activity_tracker.award_swap_points(
                    request.address, points_awarded
                )

            # Mark the swap as processed
            await swap_tracker.mark_swap_processed(
                request.chain,
                request.txid,
                request.address,
                referral_reward_usdc,
                points_awarded,
            )

            return ProcessSwapResponse(
                success=True,
                points_awarded=points_awarded,
                referral_reward=referral_reward_usdc,
                message=f"Successfully processed swap and awarded {points_awarded} points",
            )

        except Exception as e:
            logging.error(
                f"Error processing swap: {e}\nTraceback:\n{traceback.format_exc()}"
            )
            raise HTTPException(status_code=500, detail="Internal server error")

    return app


async def handle_agent_chat_request(
    protocol_registry: ProtocolRegistry,
    request: AgentChatRequest,
    portfolio: Portfolio,
    token_metadata_repo: TokenMetadataRepo,
    investor_agent: any,
    analytics_agent: any,
) -> AgentMessage:
    if request.agent == AgentType.YIELD:
        return await handle_investor_chat_request(
            request, portfolio, investor_agent, protocol_registry
        )
    else:
        return await handle_analytics_chat_request(
            request, token_metadata_repo, portfolio, analytics_agent
        )


async def handle_investor_chat_request(
    request: AgentChatRequest,
    portfolio: Portfolio,
    investor_agent: any,
    protocol_registry: ProtocolRegistry,
) -> AgentMessage:
    """Handle requests for the investor agent."""
    # Emit metric for investor agent usage
    statsd.increment("agent.usage", tags=["agent_type:investor"])

    # Build investor agent system prompt
    investor_system_prompt = get_investor_agent_prompt(
        tokens=portfolio.holdings,
        poolDeposits=[],
    )

    # Prepare message history
    message_history = convert_to_agent_message_history(
        request.context.conversationHistory
    )

    # Create messages for investor agent
    investor_messages = [
        ("system", investor_system_prompt),
        *message_history,
        ("user", request.message.message),
    ]

    # Create config for the agent
    agent_config = RunnableConfig(
        configurable={
            "tokens": portfolio.holdings,
            "positions": [],
            "protocol_registry": protocol_registry,
        }
    )

    # Run investor agent
    investor_result = await run_main_agent(
        investor_agent, investor_messages, agent_config, protocol_registry
    )

    return AgentMessage(
        message=investor_result["content"],
        pools=investor_result["pools"],
    )


async def handle_suggestions_request(
    request: AgentChatRequest,
    portfolio: Portfolio,
    token_metadata_repo: TokenMetadataRepo,
    suggestions_model: ChatOpenAI,
) -> List[str]:
    # Get tools from agent config and format them
    tools = create_investor_agent_toolkit() + create_analytics_agent_toolkit(
        token_metadata_repo
    )
    tools_list = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])

    # Build suggestions system prompt
    suggestions_system_prompt = get_suggestions_prompt(
        conversation_history=request.context.conversationHistory,
        tokens=portfolio.holdings,
        tools=tools_list,
    )

    # Run suggestions model directly
    response = await suggestions_model.ainvoke(suggestions_system_prompt)
    content = response.content

    # Clean the content by removing markdown code block syntax if present
    if content.startswith("```json"):
        content = content[7:]  # Remove ```json
    if content.endswith("```"):
        content = content[:-3]  # Remove ```
    content = content.strip()

    try:
        # First try parsing as JSON
        suggestions = json.loads(content)
        if isinstance(suggestions, list):
            return suggestions
    except json.JSONDecodeError:
        # If JSON parsing fails, try parsing as string array
        try:
            # Remove any JSON-like syntax and split by comma
            cleaned = content.strip("[]")
            # Split by comma and remove quotes
            suggestions = [item.strip().strip("'\"") for item in cleaned.split(",")]
            return suggestions
        except Exception as e:
            logging.error(f"Error parsing suggestions string: {e}")
            return []

    return []


async def run_main_agent(
    agent: any,
    messages: List,
    config: RunnableConfig,
    protocol_registry: ProtocolRegistry,
) -> Dict[str, Any]:
    try:
        # Run agent directly
        result = await agent.ainvoke({"messages": messages}, config=config, debug=False)
        # Extract final state and last message
        last_message = result["messages"][-1]

        # Extract pool IDs and clean text
        cleaned_text, pool_ids = extract_patterns(last_message.content, "pool")

        # Get full pool objects for the extracted pool IDs
        pool_objects = protocol_registry.get_pools_by_ids(pool_ids)

        return {
            "content": cleaned_text,
            "pools": pool_objects,
            "messages": result["messages"],
        }
    except Exception as e:
        logging.error(f"Error running main agent: {e}")
        raise


def convert_to_agent_message_history(messages: List[Message]) -> List[Tuple[str, str]]:
    # Get the last NUM_MESSAGES_TO_KEEP messages
    recent_messages = messages[-NUM_MESSAGES_TO_KEEP:]

    # Convert all messages except the last one with truncation
    converted_messages = [
        convert_to_agent_msg(m, truncate=True) for m in recent_messages[:-1]
    ]

    # Convert the last message without truncation
    if recent_messages:
        converted_messages.append(
            convert_to_agent_msg(recent_messages[-1], truncate=False)
        )

    for _, message in converted_messages:
        if not message:
            logging.error(
                f"Empty message.\nOriginal: {messages}\nConverted: {converted_messages}"
            )

    return converted_messages


async def handle_analytics_chat_request(
    request: AgentChatRequest,
    token_metadata_repo: TokenMetadataRepo,
    portfolio: Portfolio,
    agent: any,
) -> AgentMessage:
    # Emit metric for analytics agent usage
    statsd.increment("agent.usage", tags=["agent_type:analytics"])

    # Build analytics agent system prompt
    analytics_system_prompt = get_analytics_prompt(
        tokens=portfolio.holdings,
    )

    message_history = convert_to_agent_message_history(
        request.context.conversationHistory
    )

    # Create messages for analytics agent
    analytics_messages = [
        ("system", analytics_system_prompt),
        *message_history,
        ("user", request.message.message),
    ]

    # Create config for the agent
    agent_config = RunnableConfig(
        configurable={
            "tokens": portfolio.holdings,
            "positions": [],
            "available_pools": [],
        }
    )

    return await run_analytics_agent(
        agent, token_metadata_repo, analytics_messages, agent_config
    )


async def run_analytics_agent(
    agent: any,
    token_metadata_repo: TokenMetadataRepo,
    messages: List,
    config: RunnableConfig,
) -> AgentMessage:
    try:
        # Run agent directly
        result = await agent.ainvoke({"messages": messages}, config=config, debug=False)

        # Extract final state and last message
        last_message = result["messages"][-1]

        cleaned_text, token_ids = extract_patterns(last_message.content, "token")
        cleaned_text, buy_token_ids = extract_patterns(
            cleaned_text, "swap", remove_pattern=False
        )

        # Deduplicate token ids
        buy_token_ids = list(set(buy_token_ids))
        token_ids = list(set(token_ids).difference(buy_token_ids))

        # Create list of async tasks for token metadata search
        token_metadata_tasks = [
            token_metadata_repo.search_token(
                parts[1] if len(parts) > 1 else parts[0],  # token part
                parts[0] if len(parts) > 1 else None,  # chain part, None if no colon
            )
            for token_id in token_ids + buy_token_ids
            for parts in [token_id.split(":", 1)]
        ]

        # Wait for all token metadata searches to complete
        token_metadata = await asyncio.gather(*token_metadata_tasks)

        api_token_metadata = [
            TokenMetadata(
                address=token.address,
                name=token.name,
                symbol=token.symbol,
                chain=token.chain,
                price_usd=token.price or 0.0,
                market_cap_usd=(
                    str(token.market_cap_usd) if token.market_cap_usd else None
                ),
                dex_pool_address=token.dex_pool_address,
                image_url=token.image_url,
                show_buy_widget=f"{token.chain}:{token.address}" in buy_token_ids,
            )
            for token in token_metadata
            if token is not None
        ]
        if any(token is None for token in token_metadata):
            logging.warning(f"Some token metadata is None: {token_ids + buy_token_ids}")

        return AgentMessage(
            message=cleaned_text,
            tokens=api_token_metadata,
            pools=[],
        )
    except Exception as e:
        logging.error(f"Error running analytics agent: {e}")
        raise

from typing import List, Any, Dict, Tuple
import os
import json
import traceback
import logging
import asyncio

from fastapi import FastAPI, Request, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import ValidationError
from langgraph.graph.graph import CompiledGraph
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
)
from agent.agent_executors import (
    create_investor_executor,
    create_suggestions_model,
    create_analytics_executor,
    create_routing_model,
)
from agent.prompts import (
    get_investor_agent_prompt,
    get_suggestions_prompt,
    get_analytics_prompt,
    get_router_prompt,
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
from agent.integrations.sentient.sentient_agent import BitQuantSentientAgent
from server.middleware import DatadogMetricsMiddleware

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
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "https://bitquant.io",
            "https://www.bitquant.io",
            r"^http://localhost:(3000|3001|3002|4000|4200|5000|5173|8000|8080|8081|9000)$",
            r"^https://defi-chat-hub-git-[\w-]+-open-gradient\.vercel\.app$",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add Datadog metrics middleware
    app.add_middleware(DatadogMetricsMiddleware)

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

    # Store services in app state for access in routes
    app.state.activity_tracker = activity_tracker
    app.state.invite_manager = invite_manager
    app.state.token_metadata_repo = token_metadata_repo
    app.state.portfolio_fetcher = portfolio_fetcher

    @app.on_event("shutdown")
    async def shutdown_event():
        await protocol_registry.shutdown()
        await token_metadata_repo.close()
        await portfolio_fetcher.close()

    # Initialize agents
    router_model = create_routing_model()
    suggestions_model = create_suggestions_model()
    analytics_agent = create_analytics_executor(token_metadata_repo)
    investor_agent = create_investor_executor()

    # Initialize protocol registry
    protocol_registry = ProtocolRegistry(token_metadata_repo)
    protocol_registry.register_protocol(OrcaProtocol())
    protocol_registry.register_protocol(SaveProtocol())
    protocol_registry.register_protocol(KaminoProtocol())

    # Store agents in app state
    app.state.router_model = router_model
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
        # if not await verify_captcha_token(agent_request.captchaToken):
        #     raise HTTPException(status_code=429, detail="Invalid captcha token")

        # Increment message count, return 429 if limit reached
        if not await activity_tracker.increment_message_count(
            agent_request.context.address
        ):
            statsd.increment("agent.message.daily_limit_reached")
            raise HTTPException(status_code=429, detail="Daily message limit reached")

        try:
            portfolio = await portfolio_fetcher.get_portfolio(
                wallet_address=agent_request.context.address
            )
            response = await handle_agent_chat_request(
                token_metadata_repo=token_metadata_repo,
                protocol_registry=protocol_registry,
                request=agent_request,
                portfolio=portfolio,
                investor_agent=investor_agent,
                analytics_agent=analytics_agent,
                router_model=router_model,
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

        portfolio = await portfolio_fetcher.get_portfolio(
            wallet_address=agent_request.context.address
        )
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

    # @app.post("/api/sentient/assist")
    async def sentient_assist(
        request: Request,
        user: FirebaseIDTokenData = Depends(get_current_user),
    ):
        """
        Proxy endpoint for Sentient Agent. Accepts JSON with 'session' and 'query',
        calls BitQuantSentientAgent.assist, and returns the output blocks as JSON.
        """
        try:
            data = await request.json()
            session_dict = data.get("session")
            query_dict = data.get("query")

            class SentientAssistSession:
                def __init__(self, processor_id, activity_id, request_id, interactions):
                    self.processor_id = processor_id
                    self.activity_id = activity_id
                    self.request_id = request_id
                    self.interactions = interactions or []

                def get_interactions(self):
                    return self.interactions

            class SentientAssistQuery:
                def __init__(self, id, prompt):
                    self.id = id
                    self.prompt = prompt

            session = SentientAssistSession(
                processor_id=session_dict.get("processor_id"),
                activity_id=session_dict.get("activity_id"),
                request_id=session_dict.get("request_id"),
                interactions=session_dict.get("interactions", []),
            )
            query = SentientAssistQuery(
                id=query_dict.get("id"),
                prompt=query_dict.get("prompt"),
            )

            class SentientFlaskResponseHandler:
                def __init__(self):
                    self.blocks = []

                async def emit_text_block(self, label, text):
                    self.blocks.append({"label": label, "text": text})

                async def complete(self):
                    pass

            agent = BitQuantSentientAgent()
            handler = SentientFlaskResponseHandler()

            await agent.assist(session, query, handler)
            return JSONResponse(content={"blocks": handler.blocks})
        except Exception as e:
            logging.exception("Error in sentient_assist endpoint")
            return JSONResponse(status_code=500, content={"error": str(e)})

    return app


async def handle_agent_chat_request(
    protocol_registry: ProtocolRegistry,
    request: AgentChatRequest,
    portfolio: Portfolio,
    token_metadata_repo: TokenMetadataRepo,
    investor_agent: CompiledGraph,
    analytics_agent: CompiledGraph,
    router_model: ChatOpenAI,
) -> AgentMessage:
    # If agent is explicitly specified, bypass router
    if request.agent is not None:
        if request.agent == AgentType.ANALYTICS:
            return await handle_analytics_chat_request(
                request, token_metadata_repo, portfolio, analytics_agent
            )
        elif request.agent == AgentType.INVESTOR:
            return await handle_investor_chat_request(
                request, portfolio, investor_agent, protocol_registry
            )
        else:
            raise ValueError(f"Invalid agent type specified: {request.agent}")

    # Otherwise use router to determine agent
    router_prompt = get_router_prompt(
        message_history=request.context.conversationHistory[-NUM_MESSAGES_TO_KEEP:],
        current_message=request.message.message,
    )

    router_response = await router_model.ainvoke(router_prompt)
    selected_agent = router_response.content.strip().lower()

    # Extract agent type from response if it contains additional text
    if "investor_agent" in selected_agent:
        selected_agent = "investor_agent"
    elif "analytics_agent" in selected_agent:
        selected_agent = "analytics_agent"
    else:
        # Default to analytics agent if no clear choice
        selected_agent = "analytics_agent"

    if selected_agent == AgentType.ANALYTICS:
        return await handle_analytics_chat_request(
            request, token_metadata_repo, portfolio, analytics_agent
        )
    elif selected_agent == AgentType.INVESTOR:
        return await handle_investor_chat_request(
            request, portfolio, investor_agent, protocol_registry
        )
    else:
        raise ValueError(f"Invalid agent selection from router: {selected_agent}")


async def handle_investor_chat_request(
    request: AgentChatRequest,
    portfolio: Portfolio,
    investor_agent: CompiledGraph,
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
    agent: CompiledGraph,
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
    agent: CompiledGraph,
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
    agent: CompiledGraph,
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

        # Create list of async tasks for token metadata search
        token_metadata_tasks = [
            token_metadata_repo.search_token(
                parts[1] if len(parts) > 1 else parts[0],  # token part
                parts[0] if len(parts) > 1 else None,  # chain part, None if no colon
            )
            for token_id in token_ids
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
                price_usd=str(token.price),
                market_cap_usd=(
                    str(token.market_cap_usd) if token.market_cap_usd else None
                ),
                dex_pool_address=token.dex_pool_address,
                image_url=token.image_url,
            )
            for token in token_metadata
            if token is not None
        ]

        return AgentMessage(
            message=cleaned_text,
            tokens=api_token_metadata,
            pools=[],
        )
    except Exception as e:
        logging.error(f"Error running analytics agent: {e}")
        raise

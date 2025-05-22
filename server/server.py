from typing import List, Any, Tuple, Dict
import os
import boto3.dynamodb
import boto3.dynamodb.table
import boto3.dynamodb.types
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pydantic import ValidationError
from langgraph.graph.graph import CompiledGraph
from langchain_core.runnables.config import RunnableConfig
import json
import traceback
import boto3
from datetime import datetime
from functools import wraps
from datadog import initialize, statsd
import logging
import requests

import boto3.data
from onchain.pools.protocol import ProtocolRegistry
from onchain.pools.solana.orca_protocol import OrcaProtocol
from onchain.pools.solana.save_protocol import SaveProtocol
from onchain.pools.solana.kamino_protocol import KaminoProtocol
from onchain.tokens.metadata import TokenMetadataRepo
from onchain.portfolio.solana_portfolio import PortfolioFetcher
from api.api_types import (
    AgentChatRequest,
    AgentMessage,
    Message,
    AgentType,
    Portfolio,
    FeedbackRequest,
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
from . import service
from .auth import protected_route
from agent.integrations.sentient.sentient_agent import BitQuantSentientAgent
import asyncio

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


def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key != API_KEY:
            return jsonify({"error": "Invalid or missing API key"}), 401
        return f(*args, **kwargs)

    return decorated_function


def create_flask_app() -> Flask:
    """Create and configure the Flask application with routes."""
    app = Flask(__name__)
    app.config["PROPAGATE_EXCEPTIONS"] = True

    CORS(
        app,
        origins=[
            "https://bitquant.io",
            "https://www.bitquant.io",
            r"^http://localhost:(3000|3001|3002|4000|4200|5000|5173|8000|8080|8081|9000)$",
            r"^https://defi-chat-hub-git-[\w-]+-open-gradient\.vercel\.app$",
        ],
    )

    # Initialize DynamoDB
    dynamodb = boto3.resource(
        "dynamodb",
        region_name=os.environ.get("AWS_REGION"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )

    tokens_table = dynamodb.Table("token_metadata_v2")
    feedback_table = dynamodb.Table("twoligma_feedback")
    invite_codes_table = dynamodb.Table("twoligma_invite_codes")
    activity_table = dynamodb.Table("twoligma_activity")

    # Services
    activity_tracker = ActivityTracker(activity_table)
    invite_manager = InviteCodeManager(invite_codes_table, activity_tracker)

    # Token data
    token_metadata_repo = TokenMetadataRepo(tokens_table)
    portfolio_fetcher = PortfolioFetcher(token_metadata_repo)

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
    protocol_registry.initialize()

    @app.errorhandler(ValidationError)
    def handle_validation_error(e):
        logging.error(f"400 Error: {str(e)}")
        return jsonify({"error": str(e)}), 400

    @app.errorhandler(Exception)
    def handle_generic_error(e):
        error_traceback = traceback.format_exc()
        logging.error(f"500 Error: {str(e)}")
        logging.error(f"Traceback: {error_traceback}")
        logging.error(f"Request Path: {request.path}")
        logging.error(f"Request Body: {request.get_data(as_text=True)}")
        statsd.increment("agent.message.unhandled_error")

        return jsonify({"error": str(e)}), 500

    @app.route("/api/cloudflare/turnstile/v0/siteverify", methods=["POST"])
    def verify_cloudflare_turnstile_token():
        try:
            secret_key = os.getenv("CLOUDFLARE_TURNSTILE_SECRET_KEY")
            if not secret_key:
                raise Exception(
                    "CLOUDFLARE_TURNSTILE_SECRET_KEY environment variable is not set"
                )

            data = request.get_json()
            token = data.get("token")

            if not token:
                return jsonify({"error": "Missing token"}), 400

            # Make the request to Cloudflare Turnstile
            response = requests.post(
                "https://challenges.cloudflare.com/turnstile/v0/siteverify",
                data={"secret": secret_key, "response": token},
                headers={"content-type": "application/x-www-form-urlencoded"},
            )

            result = response.json()
            status_code = 200 if result.get("success") else 400
            return jsonify(result), status_code

        except Exception as e:
            logging.error(f"Error verifying Cloudflare Turnstile token: {e}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/verify/solana", methods=["POST"])
    def verify_solana_signature():
        try:
            request_data = request.get_json()
            verify_request = SolanaVerifyRequest(**request_data)

            token = service.verify_solana_signature(verify_request)
            return jsonify({"token": token})

        except ValidationError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logging.error(f"Error verifying SIWX signature: {e}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/healthcheck", methods=["GET"])
    def healthcheck():
        return jsonify({"status": "ok"})

    @app.route("/api/portfolio", methods=["GET"])
    @protected_route
    def get_portfolio():
        address = request.args.get("address")
        if not address:
            return jsonify({"error": "Address parameter is required"}), 400

        portfolio = portfolio_fetcher.get_portfolio(address)
        return jsonify(portfolio.model_dump())

    @app.route("/api/whitelisted", methods=["GET"])
    def is_whitelisted():
        return jsonify({"allowed": True})

    @app.route("/api/whitelist/add", methods=["POST"])
    @require_api_key
    def add_to_whitelist():
        return jsonify({"status": "success"})

    @app.route("/api/tokenlist", methods=["GET"])
    def get_tokenlist():
        file_path = os.path.join(STATIC_DIR, "tokenlist.json")
        if not os.path.isfile(file_path):
            return jsonify({"error": "Tokenlist file not found"}), 404
        return send_from_directory(STATIC_DIR, "tokenlist.json")

    @app.route("/api/agent/run", methods=["POST"])
    @protected_route
    def run_agent():
        request_data = request.get_json()
        agent_request = AgentChatRequest(**request_data)

        statsd.increment("agent.message.received")

        # Increment message count, return 429 if limit reached
        if not activity_tracker.increment_message_count(
            agent_request.context.address, agent_request.context.miner_token
        ):
            statsd.increment("agent.message.daily_limit_reached")
            return jsonify({"error": "Daily message limit reached"}), 429

        try:
            portfolio = portfolio_fetcher.get_portfolio(agent_request.context.address)
            response = handle_agent_chat_request(
                token_metadata_repo=token_metadata_repo,
                protocol_registry=protocol_registry,
                request=agent_request,
                portfolio=portfolio,
                investor_agent=investor_agent,
                analytics_agent=analytics_agent,
                router_model=router_model,
            )

            return jsonify(
                response.model_dump() if hasattr(response, "model_dump") else response
            )
        except Exception as e:
            logging.error(f"Error processing agent request: {e}")
            statsd.increment("agent.message.server_error")
            raise

    @app.route("/api/agent/suggestions", methods=["POST"])
    @protected_route
    def run_suggestions():
        request_data = request.get_json()
        agent_request = AgentChatRequest(**request_data)

        portfolio = portfolio_fetcher.get_portfolio(agent_request.context.address)
        suggestions = handle_suggestions_request(
            token_metadata_repo=token_metadata_repo,
            request=agent_request,
            portfolio=portfolio,
            suggestions_model=suggestions_model,
        )
        return jsonify({"suggestions": suggestions})

    @app.route("/api/feedback", methods=["POST"])
    @protected_route
    def submit_feedback():
        try:
            request_data = request.get_json()
            feedback_request = FeedbackRequest(**request_data)

            timestamp = datetime.now()
            user_timestamp = f"{feedback_request.walletAddress}_{timestamp}"

            # Convert conversation history to string
            conversation_history_str = json.dumps(feedback_request.conversationHistory)

            feedback_item = {
                "id": user_timestamp,
                "wallet_address": feedback_request.walletAddress,
                "feedback": feedback_request.feedback,
                "share_history": feedback_request.shareHistory,
                "conversation_history": conversation_history_str,
            }

            feedback_table.put_item(Item=feedback_item)
            return jsonify({"status": "success"}), 200

        except ValidationError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            logging.error(f"Error submitting feedback: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/invite/generate", methods=["POST"])
    @protected_route
    def generate_invite_code():
        try:
            request_data = request.get_json()
            if not request_data or "address" not in request_data:
                return jsonify({"error": "Address is required"}), 400

            creator_address = request_data["address"]

            # Generate invite code
            invite_code = invite_manager.generate_invite_code(creator_address)
            if not invite_code:
                return jsonify({"error": "Failed to generate invite code"}), 500

            return jsonify({"invite_code": invite_code})
        except Exception as e:
            logging.error(f"Error generating invite code: {e}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/invite/use", methods=["POST"])
    def use_invite_code():
        try:
            request_data = request.get_json()
            if (
                not request_data
                or "code" not in request_data
                or "address" not in request_data
            ):
                return jsonify({"error": "Code and address are required"}), 400

            code = request_data["code"]
            user_address = request_data["address"]

            # Try to use the invite code
            if not invite_manager.use_invite_code(code, user_address):
                return jsonify({"error": "Invalid or already used invite code"}), 400

            return jsonify({"status": "success"})
        except Exception as e:
            logging.error(f"Error using invite code: {e}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/activity/stats", methods=["GET"])
    @protected_route
    def get_activity_stats():
        try:
            address = request.args.get("address")
            if not address:
                return jsonify({"error": "Address parameter is required"}), 400

            stats = activity_tracker.get_activity_stats(address)
            return jsonify(stats)
        except Exception as e:
            logging.error(f"Error getting activity stats: {e}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/sentient/assist", methods=["POST"])
    @protected_route
    def sentient_assist():
        """
        Proxy endpoint for Sentient Agent. Accepts JSON with 'session' and 'query',
        calls BitQuantSentientAgent.assist, and returns the output blocks as JSON.
        """
        data = request.get_json()
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
        try:
            asyncio.run(agent.assist(session, query, handler))
            return jsonify({"blocks": handler.blocks})
        except Exception as e:
            logging.exception("Error in sentient_assist endpoint")
            return jsonify({"error": str(e)}), 500

    return app


def handle_agent_chat_request(
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
            return handle_analytics_chat_request(
                request, token_metadata_repo, portfolio, analytics_agent
            )
        elif request.agent == AgentType.INVESTOR:
            return handle_investor_chat_request(
                request, portfolio, investor_agent, protocol_registry
            )
        else:
            raise ValueError(f"Invalid agent type specified: {request.agent}")

    # Otherwise use router to determine agent
    router_prompt = get_router_prompt(
        message_history=request.context.conversationHistory[-NUM_MESSAGES_TO_KEEP:],
        current_message=request.message.message,
    )

    router_response = router_model.invoke(router_prompt)
    selected_agent = router_response.content.strip().lower()

    if selected_agent == AgentType.ANALYTICS:
        return handle_analytics_chat_request(
            request, token_metadata_repo, portfolio, analytics_agent
        )
    elif selected_agent == AgentType.INVESTOR:
        return handle_investor_chat_request(
            request, portfolio, investor_agent, protocol_registry
        )
    else:
        raise ValueError(f"Invalid agent selection from router: {selected_agent}")


def handle_investor_chat_request(
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
    investor_result = run_main_agent(
        investor_agent, investor_messages, agent_config, protocol_registry
    )

    return AgentMessage(
        message=investor_result["content"],
        pools=investor_result["pools"],
    )


def handle_suggestions_request(
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
    response = suggestions_model.invoke(suggestions_system_prompt)
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


def run_main_agent(
    agent: CompiledGraph,
    messages: List,
    config: RunnableConfig,
    protocol_registry: ProtocolRegistry,
) -> Dict[str, Any]:
    try:
        # Run agent directly
        result = agent.invoke({"messages": messages}, config=config, debug=False)
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
        statsd.increment("agent.failure", tags=["agent_type:main"])
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


def handle_analytics_chat_request(
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

    return run_analytics_agent(
        agent, token_metadata_repo, analytics_messages, agent_config
    )


def run_analytics_agent(
    agent: CompiledGraph,
    token_metadata_repo: TokenMetadataRepo,
    messages: List,
    config: RunnableConfig,
) -> AgentMessage:
    try:
        # Run agent directly
        result = agent.invoke({"messages": messages}, config=config, debug=False)

        # Extract final state and last message
        last_message = result["messages"][-1]
        cleaned_text, token_ids = extract_patterns(last_message.content, "token")

        token_metadata = [
            token_metadata_repo.search_token(
                parts[1] if len(parts) > 1 else parts[0],  # token part
                parts[0] if len(parts) > 1 else None,  # chain part, None if no colon
            )
            for token_id in token_ids
            for parts in [token_id.split(":", 1)]
        ]
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
        statsd.increment("agent.failure", tags=["agent_type:analytics"])
        raise

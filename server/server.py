from typing import List, Any, Tuple, Dict, Set
import os
import boto3.dynamodb
import boto3.dynamodb.table
import boto3.dynamodb.types
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pydantic import ValidationError, BaseModel
from langgraph.graph.graph import CompiledGraph, RunnableConfig
import json
import logging
import traceback
import boto3
import re
from datetime import datetime
from functools import wraps

import boto3.data
from onchain.pools.protocol import ProtocolRegistry
from onchain.pools.solana.orca_protocol import OrcaProtocol
from onchain.pools.solana.save_protocol import SaveProtocol
from onchain.pools.solana.kamino_protocol import KaminoProtocol
from onchain.tokens.metadata import TokenMetadataRepo
from onchain.tokens.portfolio import PortfolioFetcher
from api.api_types import (
    AgentChatRequest,
    Pool,
    UserMessage,
    AgentMessage,
    Message,
    AgentType,
    Portfolio,
    FeedbackRequest,
    TokenMetadata,
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
from server.whitelist import TwoLigmaWhitelist
from server.invitecode import InviteCodeManager
from server.utils import extract_patterns, convert_to_agent_msg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(ROOT_DIR, "static")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# number of messages to send to agents
NUM_MESSAGES_TO_KEEP = 6

# API key for whitelist management
API_KEY = os.environ.get("WHITELIST_API_KEY")


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
    CORS(app)

    # Initialize DynamoDB
    dynamodb = boto3.resource(
        "dynamodb",
        region_name=os.environ.get("AWS_REGION"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )

    tokens_table = dynamodb.Table("sol_token_metadata")
    feedback_table = dynamodb.Table("twoligma_feedback")
    whitelist_table = dynamodb.Table("twoligma_whitelist")
    invite_codes_table = dynamodb.Table("twoligma_invite_codes")

    whitelist = TwoLigmaWhitelist(whitelist_table)
    invite_manager = InviteCodeManager(invite_codes_table)

    # Initialize agents
    router_model = create_routing_model()
    suggestions_model = create_suggestions_model()
    analytics_agent = create_analytics_executor()
    investor_agent = create_investor_executor()

    token_metadata_repo = TokenMetadataRepo(tokens_table)
    portfolio_fetcher = PortfolioFetcher(token_metadata_repo)

    # Initialize protocol registry
    protocol_registry = ProtocolRegistry(token_metadata_repo)
    protocol_registry.register_protocol(OrcaProtocol())
    protocol_registry.register_protocol(SaveProtocol())
    protocol_registry.register_protocol(KaminoProtocol())
    protocol_registry.initialize()

    # Set up error handlers for production environment
    if not app.config.get("TESTING"):

        @app.errorhandler(ValidationError)
        def handle_validation_error(e):
            return jsonify({"error": str(e)}), 400

        @app.errorhandler(Exception)
        def handle_generic_error(e):
            error_traceback = traceback.format_exc()
            logger.error(f"500 Error: {str(e)}")
            logger.error(f"Traceback: {error_traceback}")
            logger.error(f"Request Path: {request.path}")
            logger.error(f"Request Body: {request.get_data(as_text=True)}")
            return jsonify({"error": str(e)}), 500

    @app.route("/api/healthcheck", methods=["GET"])
    def healthcheck():
        return jsonify({"status": "ok"})

    @app.route("/api/portfolio", methods=["GET"])
    def get_portfolio():
        address = request.args.get("address")
        if not address:
            return jsonify({"error": "Address parameter is required"}), 400
        if not whitelist.is_allowed(address):
            return jsonify({"error": "Address is not whitelisted"}), 400

        portfolio = portfolio_fetcher.get_portfolio(address)
        return jsonify(portfolio.model_dump())

    @app.route("/api/whitelisted", methods=["GET"])
    def is_whitelisted():
        address = request.args.get("address")
        if not address:
            return jsonify({"error": "Address parameter is required"}), 400

        response = jsonify({"allowed": whitelist.is_allowed(address)})
        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.route("/api/whitelist", methods=["GET"])
    @require_api_key
    def get_whitelist():
        response = jsonify({"allowed": whitelist.get_allowed()})

        response.headers["Cache-Control"] = (
            "no-store, no-cache, must-revalidate, max-age=0"
        )
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.route("/api/whitelist/add", methods=["POST"])
    @require_api_key
    def add_to_whitelist():
        try:
            request_data = request.get_json()
            if not request_data or "address" not in request_data:
                return jsonify({"error": "Address is required"}), 400

            if whitelist.add(request_data["address"]):
                return jsonify({"status": "success"})
            return jsonify({"error": "Failed to add address"}), 500
        except Exception as e:
            logger.error(f"Error adding to whitelist: {e}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/tokenlist", methods=["GET"])
    def get_tokenlist():
        file_path = os.path.join(STATIC_DIR, "tokenlist.json")
        if not os.path.isfile(file_path):
            return jsonify({"error": "Tokenlist file not found"}), 404
        return send_from_directory(STATIC_DIR, "tokenlist.json")

    @app.route("/api/agent/run", methods=["POST"])
    def run_agent():
        request_data = request.get_json()
        agent_request = AgentChatRequest(**request_data)

        if not whitelist.is_allowed(agent_request.context.address):
            return jsonify({"error": "Address is not whitelisted"}), 400

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

    @app.route("/api/agent/suggestions", methods=["POST"])
    def run_suggestions():
        request_data = request.get_json()
        agent_request = AgentChatRequest(**request_data)

        if not whitelist.is_allowed(agent_request.context.address):
            return jsonify({"error": "Address is not whitelisted"}), 400

        portfolio = portfolio_fetcher.get_portfolio(agent_request.context.address)
        suggestions = handle_suggestions_request(
            request=agent_request,
            portfolio=portfolio,
            suggestions_model=suggestions_model,
        )
        return jsonify({"suggestions": suggestions})

    @app.route("/api/feedback", methods=["POST"])
    def submit_feedback():
        try:
            request_data = request.get_json()
            feedback_request = FeedbackRequest(**request_data)

            if not whitelist.is_allowed(feedback_request.walletAddress):
                return jsonify({"error": "Address is not whitelisted"}), 400

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
            logger.error(f"Error submitting feedback: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/invite/generate", methods=["POST"])
    def generate_invite_code():
        try:
            request_data = request.get_json()
            if not request_data or "address" not in request_data:
                return jsonify({"error": "Address is required"}), 400

            creator_address = request_data["address"]

            # Check if creator is whitelisted
            if not whitelist.is_allowed(creator_address):
                return (
                    jsonify(
                        {"error": "Only whitelisted users can generate invite codes"}
                    ),
                    403,
                )

            # Generate invite code
            invite_code = invite_manager.generate_invite_code(creator_address)
            if not invite_code:
                return jsonify({"error": "Failed to generate invite code"}), 500

            return jsonify({"invite_code": invite_code})
        except Exception as e:
            logger.error(f"Error generating invite code: {e}")
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

            # Check if user is already whitelisted
            if whitelist.is_allowed(user_address):
                return jsonify({"error": "User is already whitelisted"}), 400

            # Try to use the invite code
            if not invite_manager.use_invite_code(code, user_address):
                return jsonify({"error": "Invalid or already used invite code"}), 400

            # Add user to whitelist
            if not whitelist.add(user_address):
                return jsonify({"error": "Failed to whitelist user"}), 500

            return jsonify({"status": "success"})
        except Exception as e:
            logger.error(f"Error using invite code: {e}")
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/invite/stats", methods=["GET"])
    def get_invite_stats():
        try:
            address = request.args.get("address")
            if not address:
                return jsonify({"error": "Address parameter is required"}), 400

            # Check if user is whitelisted
            if not whitelist.is_allowed(address):
                return (
                    jsonify({"error": "Only whitelisted users can view invite stats"}),
                    403,
                )

            stats = invite_manager.get_invite_stats(address)
            return jsonify(stats)
        except Exception as e:
            logger.error(f"Error getting invite stats: {e}")
            return jsonify({"error": "Internal server error"}), 500

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
    suggestions_model: ChatOpenAI,
) -> List[str]:
    # Get tools from agent config and format them
    tools = create_investor_agent_toolkit() + create_analytics_agent_toolkit()
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
            logger.error(f"Error parsing suggestions string: {e}")
            return []

    return []


def run_main_agent(
    agent: CompiledGraph,
    messages: List,
    config: RunnableConfig,
    protocol_registry: ProtocolRegistry,
) -> Dict[str, Any]:
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
            logger.error(
                f"Empty message.\nOriginal: {messages}\nConverted: {converted_messages}"
            )

    return converted_messages


def handle_analytics_chat_request(
    request: AgentChatRequest,
    token_metadata_repo: TokenMetadataRepo,
    portfolio: Portfolio,
    agent: CompiledGraph,
) -> AgentMessage:

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
    # Run agent directly
    result = agent.invoke({"messages": messages}, config=config, debug=False)

    # Extract final state and last message
    last_message = result["messages"][-1]
    cleaned_text, token_addresses = extract_patterns(last_message.content, "token")

    token_metadata = [
        token_metadata_repo.get_token_metadata(token_address)
        for token_address in token_addresses
    ]
    api_token_metadata = [
        TokenMetadata(
            address=token.address,
            name=token.name,
            symbol=token.symbol,
            price_usd=str(token.price),
            market_cap_usd=str(token.market_cap_usd) if token.market_cap_usd else None,
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

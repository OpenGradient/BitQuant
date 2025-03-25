from typing import List, Any, Tuple, Dict
import os
import boto3.dynamodb
import boto3.dynamodb.table
import boto3.dynamodb.types
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pydantic import ValidationError
from langgraph.graph.graph import CompiledGraph, RunnableConfig
import json
import logging
import traceback
import boto3

import boto3.data
from defi.pools.protocol import ProtocolRegistry
from defi.pools.solana.orca_protocol import OrcaProtocol
from defi.pools.solana.save_protocol import SaveProtocol
from defi.pools.solana.kamino_protocol import KaminoProtocol
from tokens.metadata import TokenMetadataRepo
from tokens.portfolio import PortfolioFetcher
from api.api_types import (
    AgentChatRequest,
    Pool,
    UserMessage,
    AgentMessage,
    Message,
    AgentType,
    Portfolio,
)
from agent.agent_executors import (
    create_investor_executor,
    create_suggestions_executor,
    create_analytics_executor,
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

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(ROOT_DIR, "static")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

    # Initialize agents
    suggestions_agent = create_suggestions_executor()
    analytics_agent = create_analytics_executor()
    investor_agent = create_investor_executor()

    router_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

    # Initialize protocol registry
    protocol_registry = ProtocolRegistry()
    protocol_registry.register_protocol(OrcaProtocol())
    protocol_registry.register_protocol(SaveProtocol())
    protocol_registry.register_protocol(KaminoProtocol())
    protocol_registry.initialize()

    token_metadata_repo = TokenMetadataRepo(tokens_table)
    portfolio_fetcher = PortfolioFetcher(token_metadata_repo)

    # Load address whitelist
    whitelist_path = os.path.join(STATIC_DIR, "whitelist.json")
    with open(whitelist_path, "r") as f:
        whitelist = json.load(f)
        # create set for faster access
        whitelist["allowed"] = set(whitelist["allowed"])
        whitelist["banned"] = set(whitelist["banned"])

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

    def check_whitelist(address: str) -> bool:
        if address not in whitelist["allowed"]:
            return False
        if address in whitelist["banned"]:
            return False
        return True

    @app.route("/api/portfolio", methods=["GET"])
    def get_portfolio():
        address = request.args.get("address")
        if not address:
            return jsonify({"error": "Address parameter is required"}), 400
        if not check_whitelist(address):
            return jsonify({"error": "Address is not whitelisted"}), 400

        portfolio = portfolio_fetcher.get_portfolio(address)
        return jsonify(portfolio.model_dump())

    @app.route("/api/whitelisted", methods=["GET"])
    def is_whitelisted():
        address = request.args.get("address")

        if not address:
            return jsonify({"error": "Address parameter is required"}), 400

        return jsonify({"allowed": check_whitelist(address)})

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

        if not check_whitelist(agent_request.context.address):
            return jsonify({"error": "Address is not whitelisted"}), 400

        portfolio = portfolio_fetcher.get_portfolio(agent_request.context.address)

        # Use router to select appropriate agent
        response = handle_agent_chat_request(
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

        if not check_whitelist(agent_request.context.address):
            return jsonify({"error": "Address is not whitelisted"}), 400

        suggestions = handle_suggestions_request(agent_request, suggestions_agent)

        return jsonify({"suggestions": suggestions})

    return app


def handle_agent_chat_request(
    protocol_registry: ProtocolRegistry,
    request: AgentChatRequest,
    portfolio: Portfolio,
    investor_agent: CompiledGraph,
    analytics_agent: CompiledGraph,
    router_model: ChatOpenAI,
) -> AgentMessage:
    # If agent is explicitly specified, bypass router
    if request.agent is not None:
        if request.agent == AgentType.ANALYTICS:
            return handle_analytics_chat_request(request, portfolio, analytics_agent)
        elif request.agent == AgentType.INVESTOR:
            return handle_investor_chat_request(
                request, portfolio, investor_agent, protocol_registry
            )
        else:
            raise ValueError(f"Invalid agent type specified: {request.agent}")

    # Otherwise use router to determine agent
    router_prompt = get_router_prompt(
        message_history=request.context.conversationHistory[-10:],
        current_message=request.message.message,
    )

    router_response = router_model.invoke(router_prompt)
    selected_agent = router_response.content.strip().lower()

    if selected_agent == AgentType.ANALYTICS:
        return handle_analytics_chat_request(request, portfolio, analytics_agent)
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
    message_history = [
        convert_to_agent_msg(m) for m in request.context.conversationHistory[-10:]
    ]

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
    suggestions_agent: CompiledGraph,
) -> List[str]:
    # Get tools from agent config and format them
    tools = create_investor_agent_toolkit() + create_analytics_agent_toolkit()
    tools_list = "\n".join([f"- {tool.name}: {tool.description}" for tool in tools])

    # Build suggestions agent system prompt
    suggestions_system_prompt = get_suggestions_prompt(
        tokens=portfolio.holdings,
        tools=tools_list,
    )

    # Prepare message history (last 10 messages)
    message_history = [
        convert_to_agent_msg(m) for m in request.context.conversationHistory[-10:]
    ]

    # Create messages for suggestions agent
    suggestions_messages = [
        ("system", suggestions_system_prompt),
        *message_history,
    ]

    # Create config for the agent
    agent_config = RunnableConfig(
        configurable={
            "tokens": portfolio.holdings,
        }
    )

    # Run suggestions agent
    suggestions = run_suggestions_agent(
        suggestions_agent, suggestions_messages, agent_config
    )

    return suggestions


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

    try:
        # Parse the JSON response from the agent's content
        response_data = json.loads(last_message.content)
        print(f"Response data: {response_data}")

        # Get full pool objects for the returned pool IDs
        pool_objects = protocol_registry.get_pools_by_ids(response_data["pools"])

        return {
            "content": response_data["text"],
            "pools": pool_objects,
            "messages": result["messages"],
        }
    except json.JSONDecodeError as e:
        logger.error(
            f"Failed to parse agent response ({last_message.content}) as JSON: {e}"
        )
        # Fallback to treating the entire response as text if JSON parsing fails
        return {
            "content": last_message.content,
            "pools": [],
            "messages": result["messages"],
        }


def run_suggestions_agent(
    agent: CompiledGraph, messages: List, config: RunnableConfig
) -> List[str]:
    # Run agent directly
    result = agent.invoke({"messages": messages}, config=config)

    # Extract final message
    last_message = result["messages"][-1]

    try:
        string_list = last_message.content
        # Remove brackets and split by comma
        cleaned = string_list.strip("[]")
        # Split by comma and remove quotes
        python_list = [item.strip().strip("'\"") for item in cleaned.split(",")]

        return python_list
    except json.JSONDecodeError as e:
        print(f"Error parsing suggestions JSON: {e}")
        return []


def convert_to_agent_msg(message: Message) -> Tuple[str, str]:
    if isinstance(message, UserMessage):
        return ("user", message.message)
    elif isinstance(message, AgentMessage):
        return (
            "assistant",
            json.dumps(
                {
                    "text": message.message,
                    "pools": [pool.id for pool in message.pools],
                }
            ),
        )
    else:
        raise TypeError(f"Unexpected message type: {type(message)}")


def extract_pools(messages: List[Any]) -> List[Pool]:
    return [
        a
        for msg in messages
        if hasattr(msg, "artifact") and msg.artifact
        for a in msg.artifact
    ]


def handle_analytics_chat_request(
    request: AgentChatRequest,
    portfolio: Portfolio,
    agent: CompiledGraph,
) -> AgentMessage:

    # Build analytics agent system prompt
    analytics_system_prompt = get_analytics_prompt(
        protocol="Save",
        tokens=portfolio.holdings,
        poolDeposits=[],
    )

    # Prepare message history (last 10 messages)
    message_history = [
        convert_to_agent_msg(m) for m in request.context.conversationHistory[-10:]
    ]

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

    # Run analytics agent
    analytics_result = run_analytics_agent(agent, analytics_messages, agent_config)

    return AgentMessage(message=analytics_result["content"], pools=[])


def run_analytics_agent(
    agent: CompiledGraph, messages: List, config: RunnableConfig
) -> Dict[str, Any]:
    # Run agent directly
    result = agent.invoke({"messages": messages}, config=config, debug=False)

    # Extract final state and last message
    last_message = result["messages"][-1]

    return {"content": last_message.content, "messages": result["messages"]}

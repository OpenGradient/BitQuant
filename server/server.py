from typing import List, Any, Tuple, Dict
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pydantic import ValidationError
from langgraph.graph.graph import CompiledGraph, RunnableConfig
import json
import logging
import traceback

from defi.pools.protocol import ProtocolRegistry
from defi.pools.solana.orca_protocol import OrcaProtocol
from defi.pools.solana.save_protocol import SaveProtocol

from api.api_types import (
    AgentChatRequest,
    PoolQuery,
    Chain,
    Pool,
    UserMessage,
    AgentMessage,
    Context,
    Message,
)
from agent.agent_executor import (
    create_agent_executor,
    create_suggestions_executor,
    create_analytics_executor,
)
from agent.prompts import get_agent_prompt, get_suggestions_prompt, get_analytics_prompt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(ROOT_DIR, "static")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_flask_app(protocols: List[str]) -> Flask:
    """Create and configure the Flask application with routes."""
    app = Flask(__name__)
    app.config["PROPAGATE_EXCEPTIONS"] = True
    CORS(app)

    # Initialize agents
    suggestions_agent = create_suggestions_executor()
    analytics_agent = create_analytics_executor()

    def analytics_agent_runner(query: str, tokens: List, positions: List) -> str:
        return handle_analytics_chat_request(
            AgentChatRequest(
                message=UserMessage(message=query),
                context=Context(
                    conversationHistory=[], tokens=tokens, poolPositions=positions
                ),
            ),
            analytics_agent,
        ).message

    main_agent = create_agent_executor(analytics_agent_run_func=analytics_agent_runner)

    # Initialize protocol registry
    protocol_registry = ProtocolRegistry()
    protocol_registry.register_protocol(OrcaProtocol())
    protocol_registry.register_protocol(SaveProtocol())
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

        response = handle_agent_chat_request(
            protocol_registry, protocols, agent_request, main_agent
        )

        return jsonify(response.model_dump())

    @app.route("/api/agent/suggestions", methods=["POST"])
    def run_suggestions():
        request_data = request.get_json()
        agent_request = AgentChatRequest(**request_data)

        suggestions = handle_suggestions_request(
            protocol_registry, protocols, agent_request, suggestions_agent
        )

        return jsonify({"suggestions": suggestions})

    @app.route("/api/agent/analytics", methods=["POST"])
    def run_analytics():
        """Endpoint for the DeFiDataScientist agent"""
        request_data = request.get_json()
        agent_request = AgentChatRequest(**request_data)

        response = handle_analytics_chat_request(agent_request, analytics_agent)

        return jsonify(response.model_dump())

    return app


def handle_agent_chat_request(
    protocol_registry: ProtocolRegistry,
    protocols: List[str],
    request: AgentChatRequest,
    agent: CompiledGraph,
) -> AgentMessage:
    # Get compatible pools
    compatible_pools = protocol_registry.get_pools(
        PoolQuery(
            chain=Chain.SOLANA,
            protocols=protocols,
            tokens=[token.address for token in request.context.tokens],
        )
    )

    # Build main agent system prompt
    main_system_prompt = get_agent_prompt(
        tokens=request.context.tokens,
        poolDeposits=request.context.poolPositions,
        availablePools=compatible_pools,
    )

    # Prepare message history
    message_history = [
        convert_to_agent_msg(m) for m in request.context.conversationHistory
    ]

    # Create messages for main agent
    main_messages = [
        ("system", main_system_prompt),
        *message_history,
        ("user", request.message.message),
    ]

    # Create config for the agent
    agent_config = RunnableConfig(
        configurable={
            "tokens": request.context.tokens,
            "positions": request.context.poolPositions,
            "available_pools": compatible_pools,
        }
    )

    # Run main agent
    main_result = run_main_agent(agent, main_messages, agent_config)

    return AgentMessage(
        message=main_result["content"],
        pools=extract_pools(main_result["messages"]),
    )


def handle_suggestions_request(
    protocol_registry: ProtocolRegistry,
    protocols: List[str],
    request: AgentChatRequest,
    suggestions_agent: CompiledGraph,
) -> List[str]:
    # Get compatible pools
    compatible_pools = protocol_registry.get_pools(
        PoolQuery(
            chain=Chain.SOLANA,
            protocols=protocols,
            tokens=[token.address for token in request.context.tokens],
        )
    )

    # Build suggestions agent system prompt
    suggestions_system_prompt = get_suggestions_prompt(
        tokens=request.context.tokens,
        poolDeposits=request.context.poolPositions,
        availablePools=compatible_pools,
    )

    # Prepare message history
    message_history = [
        convert_to_agent_msg(m) for m in request.context.conversationHistory
    ]

    # Create messages for suggestions agent
    suggestions_messages = [
        ("system", suggestions_system_prompt),
        *message_history,
    ]

    # Create config for the agent
    agent_config = RunnableConfig(
        configurable={
            "tokens": request.context.tokens,
            "positions": request.context.poolPositions,
            "available_pools": compatible_pools,
        }
    )

    # Run suggestions agent
    suggestions = run_suggestions_agent(
        suggestions_agent, suggestions_messages, agent_config
    )

    return suggestions


def run_main_agent(
    agent: CompiledGraph, messages: List, config: RunnableConfig
) -> Dict[str, Any]:
    # Run agent directly
    result = agent.invoke({"messages": messages}, config=config, debug=False)

    # Extract final state and last message
    last_message = result["messages"][-1]

    return {"content": last_message.content, "messages": result["messages"]}


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
        return ("assistant", message.message)
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
    agent: CompiledGraph,
) -> AgentMessage:

    # Build analytics agent system prompt
    analytics_system_prompt = get_analytics_prompt(
        protocol="Save",
        tokens=request.context.tokens,
        poolDeposits=request.context.poolPositions,
        availablePools=[],
    )

    # Prepare message history
    message_history = [
        convert_to_agent_msg(m) for m in request.context.conversationHistory
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
            "tokens": request.context.tokens,
            "positions": request.context.poolPositions,
            "available_pools": [],
        }
    )

    # Run analytics agent
    analytics_result = run_analytics_agent(agent, analytics_messages, agent_config)

    return AgentMessage(
        message=analytics_result["content"],
        pools=extract_pools(analytics_result["messages"]),
    )


def run_analytics_agent(
    agent: CompiledGraph, messages: List, config: RunnableConfig
) -> Dict[str, Any]:
    # Run agent directly
    result = agent.invoke({"messages": messages}, config=config, debug=False)

    # Extract final state and last message
    last_message = result["messages"][-1]

    return {"content": last_message.content, "messages": result["messages"]}

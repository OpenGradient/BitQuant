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
from defi.pools.solana.kamino_protocol import KaminoProtocol

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
from agent.agent_executors import (
    create_agent_executor,
    create_suggestions_executor,
    create_analytics_executor,
)
from agent.prompts import (
    get_agent_prompt,
    get_suggestions_prompt,
    get_analytics_prompt,
    get_router_prompt,
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

    # Initialize agents
    suggestions_agent = create_suggestions_executor()
    analytics_agent = create_analytics_executor()
    main_agent = create_agent_executor()

    router_model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0)

    # Initialize protocol registry
    protocol_registry = ProtocolRegistry()
    protocol_registry.register_protocol(OrcaProtocol())
    protocol_registry.register_protocol(SaveProtocol())
    protocol_registry.register_protocol(KaminoProtocol())
    protocol_registry.initialize()

    # Load tokenlist
    tokenlist_path = os.path.join(STATIC_DIR, "tokenlist.json")
    with open(tokenlist_path, "r") as f:
        tokenlist = json.load(f)

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

        # Enhance tokens with symbols from tokenlist
        agent_request.context.enhance_tokens_with_symbols(tokenlist)

        # Get router prompt
        router_prompt = get_router_prompt(
            message_history=agent_request.context.conversationHistory,
            current_message=agent_request.message.message,
        )

        # Get router decision
        router_response = router_model.invoke(router_prompt)
        agent_type = router_response.content.strip().lower()
        logger.info(f"Router response: {router_response.content}")

        # Route to appropriate handler
        if agent_type == "analytics_agent":
            response = handle_analytics_chat_request(agent_request, analytics_agent)
        else:  # default to yield agent
            response = handle_agent_chat_request(
                protocol_registry, agent_request, main_agent
            )

        return jsonify(
            response.model_dump() if hasattr(response, "model_dump") else response
        )

    @app.route("/api/agent/suggestions", methods=["POST"])
    def run_suggestions():
        request_data = request.get_json()
        agent_request = AgentChatRequest(**request_data)

        suggestions = handle_suggestions_request(
            agent_request, suggestions_agent
        )

        return jsonify({"suggestions": suggestions})

    return app


def handle_agent_chat_request(
    protocol_registry: ProtocolRegistry,
    request: AgentChatRequest,
    agent: CompiledGraph,
) -> AgentMessage:
    # Build main agent system prompt
    main_system_prompt = get_agent_prompt(
        tokens=request.context.tokens,
        poolDeposits=request.context.poolPositions,
    )

    # Prepare message history (last 10 messages)
    message_history = [
        convert_to_agent_msg(m) for m in request.context.conversationHistory[-10:]
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
            "protocol_registry": protocol_registry,
        }
    )

    # Run main agent
    main_result = run_main_agent(agent, main_messages, agent_config, protocol_registry)

    return AgentMessage(
        message=main_result["content"],
        pools=main_result["pools"],
    )


def handle_suggestions_request(
    request: AgentChatRequest,
    suggestions_agent: CompiledGraph,
) -> List[str]:
    # Build suggestions agent system prompt
    suggestions_system_prompt = get_suggestions_prompt(
        tokens=request.context.tokens,
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
            "tokens": request.context.tokens,
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
            "messages": result["messages"]
        }
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse agent response ({last_message.content}) as JSON: {e}")
        # Fallback to treating the entire response as text if JSON parsing fails
        return {
            "content": last_message.content,
            "pools": [],
            "messages": result["messages"]
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
            "tokens": request.context.tokens,
            "positions": request.context.poolPositions,
            "available_pools": [],
        }
    )

    # Run analytics agent
    analytics_result = run_analytics_agent(agent, analytics_messages, agent_config)

    return AgentMessage(
        message=analytics_result["content"],
        pools=[]
    )


def run_analytics_agent(
    agent: CompiledGraph, messages: List, config: RunnableConfig
) -> Dict[str, Any]:
    # Run agent directly
    result = agent.invoke({"messages": messages}, config=config, debug=False)

    # Extract final state and last message
    last_message = result["messages"][-1]

    return {"content": last_message.content, "messages": result["messages"]}

from typing import Set, List, Any, Tuple, Dict, Optional
import os
import asyncio
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pydantic import ValidationError
from langgraph.graph.graph import CompiledGraph, RunnableConfig
import json
from functools import wraps

from defi.stats import DefiMetrics
from defi.types import (
    AgentChatRequest,
    AgentOutput,
    PoolQuery,
    Chain,
    Pool,
    Message,
)
from agent.agent_executor import create_agent_executor, create_suggestions_executor
from agent.prompts import get_agent_prompt, get_suggestions_prompt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(ROOT_DIR, "static")


# Utility to run async functions from sync context
def async_route(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper


def create_flask_app() -> Flask:
    """Create and configure the Flask application with routes."""
    app = Flask(__name__)
    CORS(app)

    # Initialize agents
    agent = create_agent_executor()
    suggestions_agent = create_suggestions_executor()

    # Initialize metrics service
    defi_metrics = DefiMetrics()
    defi_metrics.refresh_metrics()

    # Set up error handlers for production environment
    if not app.config.get("TESTING"):
        @app.errorhandler(ValidationError)
        def handle_validation_error(e):
            return jsonify({"error": str(e)}), 400

        @app.errorhandler(Exception)
        def handle_generic_error(e):
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
    @async_route
    async def run_agent():
        request_data = request.get_json()
        agent_request = AgentChatRequest(**request_data)

        response = await handle_agent_chat_request(
            defi_metrics, agent_request, agent, suggestions_agent
        )

        return jsonify(response.model_dump())

    return app


async def handle_agent_chat_request(
    defi_metrics: DefiMetrics, 
    request: AgentChatRequest, 
    agent: CompiledGraph, 
    suggestions_agent: CompiledGraph
) -> AgentOutput:
    # Get compatible pools
    compatible_pools = defi_metrics.get_pools(
        PoolQuery(
            chain=Chain.SOLANA,
            protocols=["save", "kamino-lend"],
            tokens=[token.address for token in request.context.tokens],
        )
    )

    # Build main agent system prompt
    main_system_prompt = get_agent_prompt(
        protocol="Save",
        tokens=request.context.tokens,
        poolDeposits=request.context.poolPositions,
        availablePools=compatible_pools,
    )
    
    # Build suggestions agent system prompt
    suggestions_system_prompt = get_suggestions_prompt(
        protocol="Save",
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
        ("user", request.userInput),
    ]
    
    # Create messages for suggestions agent
    suggestions_messages = [
        ("system", suggestions_system_prompt),
        *message_history,
        ("user", request.userInput),
    ]
    
    # Create common config for both agents
    agent_config = RunnableConfig(
        configurable={
            "tokens": request.context.tokens,
            "positions": request.context.poolPositions,
            "available_pools": compatible_pools,
        }
    )

    # Run both agents in parallel
    main_agent_task = asyncio.create_task(
        run_main_agent(agent, main_messages, agent_config)
    )
    
    suggestions_task = asyncio.create_task(
        run_suggestions_agent(
            suggestions_agent, 
            suggestions_messages, 
            agent_config
        )
    )
    
    # Wait for both tasks to complete
    main_result, suggestions = await asyncio.gather(main_agent_task, suggestions_task)
    
    return AgentOutput(
        message=main_result["content"],
        pools=extract_pools(main_result["messages"]),
        suggestions=suggestions
    )


async def run_main_agent(
    agent: CompiledGraph, 
    messages: List, 
    config: RunnableConfig
) -> Dict[str, Any]:
    # Convert synchronous stream to async
    events_iter = agent.stream(
        {"messages": messages},
        config=config,
        stream_mode="values",
        debug=False
    )
    
    # Collect all events
    all_events = []
    for event in events_iter:
        all_events.append(event)
    
    # Extract final state and last message
    final_state = all_events[-1]
    last_message = final_state["messages"][-1]
    
    return {
        "content": last_message.content,
        "messages": final_state["messages"]
    }


async def run_suggestions_agent(
    agent: CompiledGraph, 
    messages: List, 
    config: RunnableConfig
) -> List[str]:
    # Convert synchronous stream to async
    events_iter = agent.stream(
        {"messages": messages},
        config=config,
        stream_mode="values",
        debug=False
    )
    
    # Collect all events
    all_events = []
    for event in events_iter:
        all_events.append(event)
    
    # Extract final state and last message
    final_state = all_events[-1]
    last_message = final_state["messages"][-1]
    
    try:
        string_list = last_message.content
        # Remove brackets and split by comma
        cleaned = string_list.strip('[]')
        # Split by comma and remove quotes
        python_list = [item.strip().strip('\'"') for item in cleaned.split(',')]

        return python_list
    except json.JSONDecodeError as e:
        print(f"Error parsing suggestions JSON: {e}")
        return []


def convert_to_agent_msg(message: Message) -> Tuple[str, str]:
    if isinstance(message, str):
        return ("user", message)
    elif isinstance(message, AgentOutput):
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
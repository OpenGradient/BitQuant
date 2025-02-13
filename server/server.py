from typing import Set, List, Any, Tuple

from flask import Flask, request, jsonify
from pydantic import ValidationError
from langgraph.graph.graph import CompiledGraph, RunnableConfig

from plugins.types import (
    AgentChatRequest,
    AgentSuggestionRequest,
    AgentOutput,
    Action,
    Message,
)
from agent.agent_executor import create_agent_executor
from agent.prompts import get_agent_prompt
from plugins.registry import PoolRegistry


def create_flask_app(plugins: List[str] = ["navi"]) -> Flask:

    app = Flask(__name__)
    agent = create_agent_executor()
    pool_registry = PoolRegistry(plugins=set(plugins))

    if not app.config.get("TESTING"):

        @app.errorhandler(ValidationError)
        def handle_validation_error(e):
            return jsonify({"error": str(e)}), 400

        @app.errorhandler(Exception)
        def handle_generic_error(e):
            return jsonify({"error": str(e)}), 500

    @app.route("/api/agent/suggest", methods=["POST"])
    def generate_suggestion():
        request_data = request.get_json()
        suggestion_request = AgentSuggestionRequest(**request_data)

        response = handle_agent_chat_request(pool_registry, suggestion_request, agent)

        return jsonify(response.model_dump())

    @app.route("/api/agent/run", methods=["POST"])
    def run_agent():
        request_data = request.get_json()
        agent_request = AgentChatRequest(**request_data)

        response = handle_agent_chat_request(pool_registry, agent_request, agent)

        return jsonify(response.model_dump())

    return app


def handle_agent_chat_request(
    pool_registry: PoolRegistry, request: AgentChatRequest, agent: CompiledGraph
) -> AgentOutput:
    # Get compatible pools
    compatible_pools = pool_registry.get_compatible_pools(
        request.context.tokens, request.context.poolPositions
    )

    # Build system prompt
    system_prompt = get_agent_prompt(
        protocol="Navi",
        tokens=request.context.tokens,
        poolDeposits=request.context.poolPositions,
        availablePools=compatible_pools,
    )

    message_history = [
        convert_to_agent_msg(m) for m in request.context.conversationHistory
    ]

    messages = [
        ("system", system_prompt),
        *message_history,
        ("user", request.userInput),
    ]

    events = agent.stream(
        {"messages": messages},
        config=RunnableConfig(
            configurable={
                "tokens": request.context.tokens,
                "positions": request.context.poolPositions,
                "available_pools": compatible_pools,
            }
        ),
        stream_mode="values",
        debug=False,  # Set to True for debugging
    )

    all_events = list(events)
    final_state = all_events[-1]
    last_message = final_state["messages"][-1]
    print(last_message)

    return AgentOutput(
        message=last_message.content,
        recommendedActions=extract_recommendations(final_state["messages"]),
    )


def convert_to_agent_msg(message: Message) -> Tuple[str, str]:
    if isinstance(message, str):
        return ("user", message)
    elif isinstance(message, AgentOutput):
        return ("assistant", message.message)
    else:
        raise TypeError(f"Unexpected message type: {type(message)}")


def extract_recommendations(messages: List[Any]) -> List[Action]:
    return [
        a
        for msg in messages
        if hasattr(msg, "artifact") and msg.artifact
        for a in msg.artifact
    ]

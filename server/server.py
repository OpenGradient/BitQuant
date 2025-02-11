from typing import List, Any, Tuple
import json

from flask import Flask, request, jsonify
from pydantic import ValidationError
from langgraph.graph.graph import CompiledGraph

from plugins.types import AgentRequest, AgentOutput, Action, Message
from agent import create_agent_executor, get_agent_prompt


def create_flask_app() -> Flask:

    app = Flask(__name__)
    agent = create_agent_executor()

    if not app.config.get("TESTING"):
        @app.errorhandler(ValidationError)
        def handle_validation_error(e):
            return jsonify({"error": str(e)}), 400

        @app.errorhandler(Exception)
        def handle_generic_error(e):
            return jsonify({"error": str(e)}), 500

    @app.route("/api/agent/run", methods=["POST"])
    def run_agent():
        request_data = request.get_json()
        agent_request = AgentRequest(**request_data)

        response = handle_agent_request(agent_request, agent)

        return jsonify(response.model_dump())

    return app


def handle_agent_request(request: AgentRequest, agent: CompiledGraph) -> AgentOutput:
    # Build system prompt
    system_prompt = get_agent_prompt(
        tokens=request.context.tokens,
        poolDeposits=request.context.poolPositions,
        availablePools=request.context.availablePools,
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
        stream_mode="values",
        debug=False,  # Set to True for debugging
    )

    all_events = list(events)
    final_state = all_events[-1]

    return AgentOutput(
        message=final_state["messages"][-1].content,
        recommendedActions=extract_recommendations(final_state["messages"]),
    )


def convert_to_agent_msg(message: Message) -> Tuple[str, str]:
    if isinstance(message, str):
        return ("user", message)
    elif isinstance(message, AgentOutput):
        return ("assistant", message.model_dump_json())
    else:
        raise TypeError(f"Unexpected message type: {type(message)}")


def extract_recommendations(messages: List[Any]) -> List[Action]:
    return [msg.artifact for msg in messages if hasattr(msg, "artifact")]

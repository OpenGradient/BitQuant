from flask import Flask, request, jsonify
from pydantic import ValidationError
from langgraph.graph.graph import CompiledGraph

from apitypes.types import AgentRequest, AgentOutput
from agent import create_agent_executor, get_agent_prompt


def create_flask_app():

    app = Flask(__name__)
    agent = create_agent_executor()

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

    events = agent.stream(
        {"messages": [("system", system_prompt), ("user", request.userInput)]},
        stream_mode="values",
        debug=False,  # Set to True for debugging
    )

    answer = list(events)[-1]["messages"][-1].content
    response = AgentOutput(message=answer, recommendedAction=[])

    return response

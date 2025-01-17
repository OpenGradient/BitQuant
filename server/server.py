from flask import Flask, request, jsonify
from pydantic import ValidationError

from server.types import AgentRequest, AgentOutput
from agent import create_agent_executor


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

        response = handle_agent_request(agent_request)

        return jsonify(response.model_dump())

    return app


def handle_agent_request(request: AgentRequest) -> AgentOutput:
    response = AgentOutput(message="Placeholder response", recommendedAction=None)
    return response

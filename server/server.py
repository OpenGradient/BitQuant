from flask import Flask, request, jsonify

from server.types import AgentRequest, AgentOutput
from agent import create_agent_executor

def create_flask_app():

    app = Flask(__name__)
    agent = create_agent_executor()

    @app.route("/api/agent/run", methods=["POST"])
    def run_agent():
        try:
            # Parse and validate request
            request_data = request.get_json()
            agent_request = AgentRequest(**request_data)

            # Placeholder response
            response = AgentOutput(message="Placeholder response", recommendedAction=None)

            return jsonify(response.model_dump())

        except Exception as e:
            return jsonify({"error": str(e)}), 400
    
    return app

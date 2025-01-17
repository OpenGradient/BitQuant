from flask import Flask, request, jsonify

from server.types import AgentRequest, AgentOutput

app = Flask(__name__)


@app.route("/api/agent/run", methods=["POST"])
def run_agent():
    try:
        # Parse and validate request
        request_data = request.get_json()
        agent_request = AgentRequest(**request_data)

        # TODO: Implement agent logic here
        # This would include:
        # 1. Processing the conversation history
        # 2. Analyzing token holdings and pool data
        # 3. Making recommendations based on user input
        # 4. Generating appropriate actions or messages

        # Placeholder response
        response = AgentOutput(message="Placeholder response", recommendedAction=None)

        return jsonify(response.dict())

    except Exception as e:
        return jsonify({"error": str(e)}), 400

import requests
from typing import Dict, Any
import logging

TEST_CONTEXT = {
    "conversationHistory": [],
    "address": "AzoqqVNzidLSVDiAfZcLf9AsXCVM5cFNLuwjcERPX8JN",
}


def main():
    context = TEST_CONTEXT.copy()
    endpoint = "http://127.0.0.1:5000/api/agent/run"

    while True:
        # read input from command line
        message = {"type": "user", "message": input("\nUser: ")}

        # Format the request payload correctly
        payload = {"message": message, "context": context}

        # send to agent
        response = make_request(payload, endpoint)
        try:
            response.raise_for_status()
        except Exception as e:
            logging.error(f"HTTP Error: {e}")
            logging.error(f"Status Code: {response.status_code}")
            logging.error(f"Response Text: {response.text}")
            logging.error(f"Request URL: {response.request.url}")
            raise

        agent_output = response.json()
        answer = agent_output["message"]
        pools = agent_output.get("pools", [])
        tokens = agent_output.get("tokens", [])

        # print results
        print(f"Two-Ligma: {answer}")
        if pools:
            print(f"Pools: {pools}")
        if tokens:
            print(f"Tokens: {tokens}")

        # append to history
        context["conversationHistory"].append(message)
        context["conversationHistory"].append(agent_output)


def make_request(input_data: Dict[str, Any], endpoint: str) -> Any:
    return requests.post(
        endpoint,
        json=input_data,
        headers={"Content-Type": "application/json"},
    )


def extract_final_response(response_data):
    """Extract just the final response text from the analytics response"""
    # If it's a dictionary with 'messages' key
    if isinstance(response_data, dict) and "messages" in response_data:
        messages = response_data["messages"]
        # Find the last non-empty assistant message
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("content"):
                return msg["content"]
            # Handle case where it's an object with content attribute
            elif hasattr(msg, "content") and msg.content:
                return msg.content

    # Return the original if we can't parse it
    return response_data


if __name__ == "__main__":
    main()

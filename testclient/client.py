import requests
from typing import Dict, Any

TEST_CONTEXT = {
    "conversationHistory": [],
    "tokens": [
        {"amount": 100, "address": "So11111111111111111111111111111111111111112"},
        {"amount": 45333, "address": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So"},
        {"amount": 900, "address": "USDSwr9ApdHk5bvJKMjzff41FfuX8bSxdKcR81vTwcA"},
        {"amount": 5, "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"},
        {"amount": 105454, "address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"},
    ],
    "poolPositions": [],
}


def main():
    context = TEST_CONTEXT.copy()

    # Ask user which agent to use
    agent_type = input("Choose an agent (regular/analytics): ").strip().lower()

    # Set endpoint based on agent type
    if agent_type == "analytics":
        endpoint = "http://127.0.0.1:5000/api/agent/analytics"
    else:
        endpoint = "http://127.0.0.1:5000/api/agent/run"

    while True:
        # read input from command line
        message = {"type": "user", "message": input("\nUser: ")}

        # Format the request payload correctly
        payload = {"message": {"message": message}, "context": context}

        # send to agent
        response = make_request(payload, endpoint)
        response.raise_for_status()

        agent_output = response.json()
        answer = agent_output["message"]
        actions = agent_output.get("recommendedActions", [])

        # print results
        if agent_type == "analytics":
            response_json = response.json()
            if isinstance(response_json, dict) and "message" in response_json:
                print(f"Two-Ligma: {response_json['message']}")
            else:
                print(f"Two-Ligma: {response_json}")
        else:
            # Regular handling for other responses
            print(f"Two-Ligma: {response.json()['message']}")
        if actions:
            print(actions)

        # append to history
        context["conversationHistory"].append({"message": message})
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

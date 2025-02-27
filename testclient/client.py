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

    while True:
        # read input from command line
        message = input("\nUser: ")

        # send to agent
        response = make_request({"userInput": message, "context": context})
        response.raise_for_status()

        agent_output = response.json()
        answer = agent_output["message"]
        actions = agent_output["recommendedActions"]

        # print results
        print(f"Two-Ligma: {answer}")
        print(actions)

        # append to history
        context["conversationHistory"].append(message)
        context["conversationHistory"].append(agent_output)


def make_request(input_data: Dict[str, Any]) -> Any:
    return requests.post(
        "http://127.0.0.1:5000/api/agent/run",
        json=input_data,
        headers={"Content-Type": "application/json"},
    )


if __name__ == "__main__":
    main()

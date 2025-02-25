import requests
from typing import Dict, Any

TEST_CONTEXT = {
    "conversationHistory": [],
    "tokens": [
        {"amount": 100, "tokenSymbol": "SUI"},
        {"amount": 45333, "tokenSymbol": "USDC"},
        {"amount": 999, "tokenSymbol": "wUSDC"},
        {"amount": 900, "tokenSymbol": "suiUSDT"},
        {"amount": 5, "tokenSymbol": "vSUI"},
        {"amount": 105454, "tokenSymbol": "wUSDT"},
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

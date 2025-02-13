import requests
from typing import Dict, Any

TEST_CONTEXT = {
    "conversationHistory": [],
    "tokens": [
        {"amount": 100, "tokenSymbol": "SUI"},
        {"amount": 45333, "tokenSymbol": "USDC"},
        {"amount": 900, "tokenSymbol": "suiUSDT"},
        {"amount": 5, "tokenSymbol": "wUSDT"},
    ],
    "poolPositions": [
        {"poolId": "SUI-USDC", "depositedTokens": {"SUI": 5000, "USDC": 10000}}
    ],
    "availablePools": [
        {
            "id": "suiUSDT-USDC",
            "tokens": [
                {"symbol": "suiUSDT", "price": 3.45},
                {"symbol": "USDC", "price": 1},
            ],
            "protocol": "OG",
            "TVL": "$19.64M",
            "APRLastDay": 2.64,
            "APRLastWeek": 33.45,
            "APRLastMonth": 81.06,
        },
        {
            "id": "SUI-USDC",
            "tokens": [
                {"symbol": "SUI", "price": 3.45},
                {"symbol": "USDC", "price": 1},
            ],
            "protocol": "OG",
            "TVL": "$10.14M",
            "APRLastDay": 103.11,
            "APRLastWeek": 118.33,
            "APRLastMonth": 102.79,
        },
        {
            "id": "wUSDT-USDC",
            "tokens": [{"symbol": "wUSDT", "price": 1}, {"symbol": "USDC", "price": 1}],
            "protocol": "OG",
            "TVL": "$6.16M",
            "APRLastDay": 8.76,
            "APRLastWeek": 40.71,
            "APRLastMonth": 39.09,
        },
    ],
}


def main():
    context = TEST_CONTEXT.copy()

    while True:
        # read input from command line
        message = input("Enter your message: ")

        # send to agent
        response = make_request({"userInput": message, "context": context})
        response.raise_for_status()

        agent_output = response.json()
        answer = agent_output["message"]
        actions = agent_output["recommendedActions"]

        # print results
        print(f"Bot: {answer}")
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

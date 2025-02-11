TEST_CONTEXT = {
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
            "tokenSymbols": ["suiUSDT", "USDC"],
            "protocol": "OG",
            "TVL": "$19.64M",
            "APRLastDay": 2.64,
            "APRLastWeek": 33.45,
            "APRLastMonth": 81.06,
        },
        {
            "id": "SUI-USDC",
            "tokenSymbols": ["SUI", "USDC"],
            "protocol": "OG",
            "TVL": "$10.14M",
            "APRLastDay": 103.11,
            "APRLastWeek": 118.33,
            "APRLastMonth": 102.79,
        },
        {
            "id": "wUSDT-USDC",
            "tokenSymbols": ["wUSDT", "USDC"],
            "protocol": "OG",
            "TVL": "$6.16M",
            "APRLastDay": 8.76,
            "APRLastWeek": 40.71,
            "APRLastMonth": 39.09,
        },
    ],
}

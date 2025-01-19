import unittest
import json
from typing import Dict, Any
import logging

from dotenv import load_dotenv
from flask import Flask
from flask.testing import FlaskClient
from werkzeug.test import TestResponse

from server import create_flask_app

DEFAULT_CONTEXT = {
    "conversationHistory": [],
    "tokens": [
        {"amount": 100, "symbol": "SUI"},
        {"amount": 45333, "symbol": "USDC"},
        {"amount": 900, "symbol": "suiUSDT"},
        {"amount": 5, "symbol": "wUSDT"}
    ],
    "poolPositions": [{"poolName": "SUI-USDC", "amountDeposited": 5000}],
    "availablePools": [
        {
            "name": "suiUSDT-USDC",
            "TVL": "$19.64M",
            "APRLastDay": 2.64,
            "APRLastWeek": 33.45,
            "APRLastMonth": 81.06,
        },
        {
            "name": "SUI-USDC",
            "TVL": "$10.14M",
            "APRLastDay": 103.11,
            "APRLastWeek": 118.33,
            "APRLastMonth": 102.79,
        },
        {
            "name": "wUSDT-USDC",
            "TVL": "$6.16M",
            "APRLastDay": 8.76,
            "APRLastWeek": 40.71,
            "APRLastMonth": 39.09,
        }
    ]
}


class TestAgentAPI(unittest.TestCase):

    app: Flask
    client: FlaskClient

    @classmethod
    def setUpClass(cls):
        """Set up Flask test client once for all tests"""
        load_dotenv()

        cls.app = create_flask_app()
        cls.app.testing = True

        cls.client = cls.app.test_client()

    def setUp(self):
        """Set up test cases with different inputs and expected outputs"""
        self.test_cases = [
            {
                "input": {
                    "userInput": "i wanna withdraw everything",
                    "context": DEFAULT_CONTEXT,
                },
                "expected": {
                    "status_code": 200,
                    "content_checks": [
                        lambda x: isinstance(x, dict),
                    ],
                },
            },
            {
                "input": {
                    "userInput": "what's my balance?",
                    "context": DEFAULT_CONTEXT,
                },
                "expected": {
                    "status_code": 200,
                    "content_checks": [
                        lambda x: isinstance(x, dict),
                    ],
                },
            },
        ]

    def make_request(self, input_data: Dict[str, Any]) -> TestResponse:
        return self.client.post(
            "/api/agent/run",
            json=input_data,
            headers={"Content-Type": "application/json"},
        )

    def test_agent_responses(self):
        for test_case in self.test_cases:
            with self.subTest(input_data=test_case["input"]):
                # Make the request
                response = self.make_request(test_case["input"])

                # Check status code
                self.assertEqual(
                    response.status_code,
                    test_case["expected"]["status_code"],
                    f"Expected status code {test_case['expected']['status_code']}, got {response.status_code}. Content: {response.text}",
                )

                # Parse response content
                try:
                    content = json.loads(response.data.decode())
                except json.JSONDecodeError:
                    self.fail("Response is not valid JSON")

                # Run all content checks
                for check in test_case["expected"]["content_checks"]:
                    self.assertTrue(
                        check(content), f"Content check failed for response: {content}"
                    )

    def test_invalid_input(self):
        """Test the API with invalid input"""
        invalid_input = {
            "userInput": "test",
            # Missing context
        }

        response = self.make_request(invalid_input)
        self.assertEqual(response.status_code, 400)


if __name__ == "__main__":
    unittest.main()

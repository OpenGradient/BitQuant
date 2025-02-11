import unittest
import json
from typing import Dict, Any, Callable, List
from dataclasses import dataclass
from copy import deepcopy


from dotenv import load_dotenv
from flask import Flask
from flask.testing import FlaskClient
from werkzeug.test import TestResponse

from server import create_flask_app
from testutils.contex import TEST_CONTEXT


def context_with_msg_history(msg_history: List) -> Dict:
    context = deepcopy(TEST_CONTEXT)
    context["conversationHistory"] = msg_history
    return context


@dataclass
class ContentCheck:
    description: str
    check_func: Callable[[Dict], bool]


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
                    "userInput": "Make my tokens work for me",
                    "context": TEST_CONTEXT,
                },
                "expected": {
                    "status_code": 200,
                    "content_checks": [
                        ContentCheck(
                            "Response should be a dictionary",
                            lambda x: isinstance(x, dict),
                        ),
                        ContentCheck(
                            "Response message should mention deposit and SUI-USDC",
                            lambda x: "SUI-USDC" in x["message"],
                        ),
                        ContentCheck(
                            "Should have at least 1 recommended action",
                            lambda x: len(x["recommendedActions"]) >= 1,
                        ),
                        ContentCheck(
                            "Recommended action should be depositToPool",
                            lambda x: x["recommendedActions"][0]["type"]
                            == "depositToPool",
                        ),
                    ],
                },
            },
            {
                "input": {
                    "userInput": "allocate my USDC",
                    "context": TEST_CONTEXT,
                },
                "expected": {
                    "status_code": 200,
                    "content_checks": [
                        ContentCheck(
                            "Response should be a dictionary",
                            lambda x: isinstance(x, dict),
                        ),
                        ContentCheck(
                            "Should have at least 1 actions",
                            lambda x: len(x["recommendedActions"]) >= 1,
                        ),
                    ],
                },
            },
            {
                "input": {
                    "userInput": "spread my USDC evenly",
                    "context": TEST_CONTEXT,
                },
                "expected": {
                    "status_code": 200,
                    "content_checks": [
                        ContentCheck(
                            "Response should be a dictionary",
                            lambda x: isinstance(x, dict),
                        ),
                        ContentCheck(
                            "Should have 3 recommended actions",
                            lambda x: len(x["recommendedActions"]) == 3,
                        ),
                    ],
                },
            },
            {
                "input": {
                    "userInput": "i want to withdraw everything",
                    "context": TEST_CONTEXT,
                },
                "expected": {
                    "status_code": 200,
                    "content_checks": [
                        ContentCheck(
                            "Response should be a dictionary",
                            lambda x: isinstance(x, dict),
                        ),
                        ContentCheck(
                            "Should have exactly one recommended action",
                            lambda x: len(x["recommendedActions"]) == 2,
                        ),
                        ContentCheck(
                            "Recommended action should be withdrawFromPool",
                            lambda x: x["recommendedActions"][0]["type"]
                            == "withdrawFromPool",
                        ),
                    ],
                },
            },
            {
                "input": {
                    "userInput": "what positions do i have?",
                    "context": TEST_CONTEXT,
                },
                "expected": {
                    "status_code": 200,
                    "content_checks": [
                        ContentCheck(
                            "Response should be a dictionary",
                            lambda x: isinstance(x, dict),
                        ),
                        ContentCheck(
                            "Should have no recommended actions",
                            lambda x: len(x["recommendedActions"]) == 0,
                        ),
                    ],
                },
            },
            {
                "input": {
                    "userInput": "only deposit 20k",
                    "context": context_with_msg_history(
                        [
                            "allocate my USDC to highest yielding pool",
                            {
                                "message": "you can deposit your USDC to SUI-USDC",
                                "recommendedActions": [
                                    {
                                        "pool": "SUI-USDC",
                                        "tokens": {
                                            "USDC": 45333,
                                            "SUI": 100,
                                        }
                                    }
                                ],
                            },
                        ]
                    ),
                },
                "expected": {
                    "status_code": 200,
                    "content_checks": [
                        ContentCheck(
                            "Response should be a dictionary",
                            lambda x: isinstance(x, dict),
                        ),
                        ContentCheck(
                            "Should have at least 1 actions",
                            lambda x: len(x["recommendedActions"]) >= 1,
                        ),
                        ContentCheck(
                            "Should deposit 20k to pool",
                            lambda x: x["recommendedActions"][0]["tokens"]["USDC"] == 20000,
                        ),
                    ],
                },
            },
        ]

    def test_agent_responses(self):
        for test_case in self.test_cases:
            with self.subTest(input_data=test_case["input"]["userInput"]):
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

                # Run all content checks with descriptive error messages
                for check in test_case["expected"]["content_checks"]:
                    self.assertTrue(
                        check.check_func(content),
                        f"\nCheck failed: {check.description}\nResponse content: {json.dumps(content, indent=2)}",
                    )

    def make_request(self, input_data: Dict[str, Any]) -> TestResponse:
        return self.client.post(
            "/api/agent/run",
            json=input_data,
            headers={"Content-Type": "application/json"},
        )


if __name__ == "__main__":
    unittest.main()

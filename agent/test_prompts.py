import unittest

from .prompts import get_agent_prompt
from testutils.contex import TEST_CONTEXT

TEST_CONTEXT = {
    "conversationHistory": [],
    "tokens": [
        {"amount": 100, "address": "So11111111111111111111111111111111111111112"},
        {"amount": 45333, "address": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So"},
        {"amount": 900, "address": "USDSwr9ApdHk5bvJKMjzff41FfuX8bSxdKcR81vTwcA"},
        {"amount": 500, "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"},
        {"amount": 105454, "address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"},
    ],
    "poolPositions": [],
}


class TestPrompts(unittest.TestCase):
    def test_prompt(self):
        prompt = get_agent_prompt(
            tokens=TEST_CONTEXT["tokens"],
            poolDeposits=TEST_CONTEXT["poolPositions"],
        )

        self.assertTrue(len(prompt) >= 100)

        print(prompt)

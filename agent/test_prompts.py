import unittest

from .prompts import get_agent_prompt
from testutils.contex import TEST_CONTEXT


class TestAgentAPI(unittest.TestCase):
    def test_prompt(self):
        prompt = get_agent_prompt(
            protocol="OpenGradient",
            tokens=TEST_CONTEXT["tokens"],
            poolDeposits=TEST_CONTEXT["poolPositions"],
            availablePools=TEST_CONTEXT["availablePools"],
        )

        self.assertTrue(len(prompt) >= 100)

        print(prompt)

from typing import List
import logging

import jinja2

from api.api_types import WalletTokenHolding, Pool, WalletPoolPosition

env = jinja2.Environment(loader=jinja2.FileSystemLoader("templates/"))

agent_template = env.get_template("agent.jinja2")
suggestions_template = env.get_template("suggestions.jinja2")


def get_agent_prompt(
    tokens: List[WalletTokenHolding],
    poolDeposits: List[WalletPoolPosition],
    availablePools: List[Pool],
) -> str:
    agent_prompt = agent_template.render(
        tokens=tokens,
        poolDeposits=poolDeposits,
        availablePools=availablePools,
    )

    return agent_prompt


def get_suggestions_prompt(
    tokens: List[WalletTokenHolding],
    poolDeposits: List[WalletPoolPosition],
    availablePools: List[Pool],
) -> str:
    agent_prompt = suggestions_template.render(
        tokens=tokens,
        poolDeposits=poolDeposits,
        availablePools=availablePools,
    )

    return agent_prompt

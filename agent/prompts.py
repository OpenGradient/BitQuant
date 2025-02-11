from typing import List
import logging

import jinja2

from plugins.types import WalletTokenHolding, Pool, WalletPoolPosition

env = jinja2.Environment(loader=jinja2.FileSystemLoader("templates/"))
template = env.get_template("prompt.jinja2")


def get_agent_prompt(
    protocol: str,
    tokens: List[WalletTokenHolding],
    poolDeposits: List[WalletPoolPosition],
    availablePools: List[Pool],
) -> str:
    agent_prompt = template.render(
        protocolName=protocol,
        tokens=tokens,
        poolDeposits=poolDeposits, 
        availablePools=availablePools
    )
    logging.debug("Built prompt:\n=======\n%s\n=======", agent_prompt)

    return agent_prompt

from typing import List
import logging

import jinja2

from plugins.types import TokenBalance, Pool, PoolPosition

env = jinja2.Environment(loader=jinja2.FileSystemLoader("templates/"))
template = env.get_template("prompt.jinja2")


def get_agent_prompt(
    tokens: List[TokenBalance],
    poolDeposits: List[PoolPosition],
    availablePools: List[Pool],
) -> str:
    agent_prompt = template.render(
        tokens=tokens, poolDeposits=poolDeposits, availablePools=availablePools
    )
    logging.debug("Built prompt:\n=======\n%s\n=======", agent_prompt)

    return agent_prompt

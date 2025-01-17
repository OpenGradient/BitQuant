from typing import List

import jinja2

from bftypes.types import TokenBalance, Pool, PoolPosition

env = jinja2.Environment(loader=jinja2.FileSystemLoader("templates/"))


def get_agent_prompt(
    tokens: List[TokenBalance],
    poolDeposits: List[PoolPosition],
    availablePools: List[Pool],
) -> str:
    template = env.get_template("prompt.jinja2")
    agent_prompt = template.render()

    return agent_prompt

import os

import opengradient as og
from langgraph.prebuilt import create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel

from agent.tools import create_investor_agent_toolkit, create_analytics_agent_toolkit
from onchain.tokens.metadata import TokenMetadataRepo
from server import config


##
# OpenRouter LLM Configuration
##

# Select model based on configuration
SUGGESTIONS_MODEL = og.TEE_LLM.GEMINI_2_5_FLASH
REASONING_MODEL = og.TEE_LLM.GEMINI_2_5_PRO

opengradient_client = og.LLM(
    private_key=config.WALLET_PRIV_KEY,
    rpc_url=config.OG_RPC_URL,
)


def create_suggestions_model() -> BaseChatModel:
    return og.agents.langchain_adapter(
        client=opengradient_client,
        model_cid=SUGGESTIONS_MODEL,
        temperature=0.3,
        max_tokens=1000,
    )


def create_investor_executor() -> any:
    model = og.agents.langchain_adapter(
        client=opengradient_client,
        model_cid=REASONING_MODEL,
        temperature=0.1,
        max_tokens=16384,
    )

    agent_executor = create_react_agent(
        model=model, tools=create_investor_agent_toolkit()
    )

    return agent_executor


def create_analytics_executor(
    token_metadata_repo: TokenMetadataRepo,
    extra_tools: list = None,
) -> any:
    model = og.agents.langchain_adapter(
        client=opengradient_client,
        model_cid=REASONING_MODEL,
        temperature=0.1,
        max_tokens=16384,
    )

    tools = create_analytics_agent_toolkit(token_metadata_repo)
    if extra_tools:
        tools.extend(extra_tools)

    analytics_executor = create_react_agent(
        model=model,
        tools=tools,
    )

    return analytics_executor

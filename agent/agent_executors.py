import os
import httpx
import logging

from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models.chat_models import BaseChatModel

from agent.tools import create_investor_agent_toolkit, create_analytics_agent_toolkit
from onchain.tokens.metadata import TokenMetadataRepo
from server import config
from web3 import Web3

from x402v2 import x402Client as x402Clientv2
from x402v2.http.clients import x402HttpxClient as x402HttpxClientv2
from x402v2.mechanisms.evm import EthAccountSigner as EthAccountSignerv2
from x402v2.mechanisms.evm.exact.register import (
    register_exact_evm_client as register_exact_evm_clientv2,
)
from x402v2.mechanisms.evm.upto.register import (
    register_upto_evm_client as register_upto_evm_clientv2,
)

logging.getLogger("x402.httpx").setLevel(logging.DEBUG)

WEB3_CONFIG = Web3(Web3.HTTPProvider(config.OG_RPC_URL))
WALLET_ACCOUNT = WEB3_CONFIG.eth.account.from_key(config.WALLET_PRIV_KEY)
BASE_TESTNET_NETWORK = "eip155:84532"

x402_client = x402Clientv2()
register_exact_evm_clientv2(
    x402_client,
    EthAccountSignerv2(WALLET_ACCOUNT),
    networks=[BASE_TESTNET_NETWORK],
)
register_upto_evm_clientv2(
    x402_client,
    EthAccountSignerv2(WALLET_ACCOUNT),
    networks=[BASE_TESTNET_NETWORK],
)


TIMEOUT = httpx.Timeout(
    timeout=90.0,
    connect=15.0,
    read=15.0,
    write=30.0,
    pool=10.0,
)

LIMITS = httpx.Limits(
    max_keepalive_connections=100,
    max_connections=500,
    keepalive_expiry=60 * 20,  # 20 minutes
)


##
# OpenRouter LLM Configuration
##

GOOGLE_GEMINI_20_FLASH_MODEL = (
    "gemini-2.0-flash"  # $0.1/M input tokens; $0.4/M output tokens
)
LLAMA_3_1_405B_MODEL = (
    "meta-llama/llama-3.1-405b-instruct"  # $0.8/M input tokens; $0.8/M output tokens
)
DEEPSEEK_CHAT_V3_MODEL = (
    "deepseek/deepseek-chat-v3-0324"  # $0.27/M input tokens; $1.1/M output tokens
)
GROK_MODEL = "x-ai/grok-2-1212"  # $2/M input tokens; $10/M output tokens

x402_http_client = x402HttpxClientv2(
    x402_client,
    timeout=TIMEOUT,
    limits=LIMITS,
    http2=False,
    follow_redirects=False,
)

# Select model based on configuration
SUGGESTIONS_MODEL = GOOGLE_GEMINI_20_FLASH_MODEL
REASONING_MODEL = GOOGLE_GEMINI_20_FLASH_MODEL
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/"
API_KEY = os.getenv("GEMINI_API_KEY")


def create_suggestions_model() -> BaseChatModel:
    return ChatOpenAI(
        model=SUGGESTIONS_MODEL,
        temperature=0.3,
        max_tokens=1000,
        api_key=config.DUMMY_X402_API_KEY,
        http_async_client=x402_http_client,
        stream_usage=True,
        streaming=True,
        base_url=config.LLM_SERVER_URL,
    )


def create_investor_executor() -> any:
    openai_model = ChatOpenAI(
        model=REASONING_MODEL,
        temperature=0.0,
        api_key=config.DUMMY_X402_API_KEY,
        http_async_client=x402_http_client,
        stream_usage=True,
        streaming=True,
        base_url=config.LLM_SERVER_URL,
        max_tokens=4096,
    )

    agent_executor = create_react_agent(
        model=openai_model, tools=create_investor_agent_toolkit()
    )

    return agent_executor


def create_analytics_executor(token_metadata_repo: TokenMetadataRepo) -> any:
    openai_model = ChatOpenAI(
        model=REASONING_MODEL,
        temperature=0.0,
        max_tokens=4096,
        api_key=config.DUMMY_X402_API_KEY,
        http_async_client=x402_http_client,
        stream_usage=True,
        streaming=True,
        base_url=config.LLM_SERVER_URL,
    )

    analytics_executor = create_react_agent(
        model=openai_model,
        tools=create_analytics_agent_toolkit(token_metadata_repo),
    )

    return analytics_executor

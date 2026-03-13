"""
Local CLI for chatting with the BitQuant analytics agent, bypassing the server.

Usage:
    python3.13 cli.py
    python3.13 cli.py --agent investor
"""

import asyncio
import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

# Set a dummy wallet key if not present so agent_executors module-level code doesn't crash
if not os.environ.get("WALLET_PRIV_KEY"):
    os.environ["WALLET_PRIV_KEY"] = "0x" + "00" * 32

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig

from agent.agent_executors import create_analytics_executor, create_investor_executor
from agent.prompts import get_analytics_prompt, get_investor_agent_prompt
from onchain.tokens.metadata import TokenMetadataRepo
from onchain.chains import parse_token_id
from server.utils import extract_patterns
from server.dynamodb_helpers import DatabaseManager

logging.basicConfig(level=logging.WARNING)


def create_token_metadata_repo() -> TokenMetadataRepo:
    db = DatabaseManager()
    return TokenMetadataRepo(db.table_context_factory("token_metadata_v2"))


async def run_analytics_turn(agent, token_metadata_repo, messages, config):
    """Run one turn of the analytics agent and post-process token/swap patterns."""
    result = await agent.ainvoke({"messages": messages}, config=config)
    last_message = result["messages"][-1]

    cleaned_text, token_ids = extract_patterns(last_message.content, "token")
    cleaned_text, buy_token_ids = extract_patterns(
        cleaned_text, "swap", remove_pattern=False
    )

    buy_token_ids = list(set(buy_token_ids))
    token_ids = list(set(token_ids).difference(buy_token_ids))

    # Resolve token metadata
    tokens = []
    for tid in token_ids + buy_token_ids:
        try:
            chain, address = parse_token_id(tid)
            meta = await token_metadata_repo.search_token(address, chain)
            if meta:
                label = "swap" if tid in buy_token_ids else "token"
                tokens.append(
                    f"  {label}: {meta.name} ({meta.symbol}) on {meta.chain} — ${meta.price or '?'}"
                )
        except ValueError:
            pass

    return cleaned_text, tokens


async def run_investor_turn(agent, messages, config):
    """Run one turn of the investor agent."""
    result = await agent.ainvoke({"messages": messages}, config=config)
    last_message = result["messages"][-1]
    return last_message.content, []


async def main():
    parser = argparse.ArgumentParser(description="Chat with BitQuant locally")
    parser.add_argument(
        "--agent",
        choices=["analytics", "investor"],
        default="analytics",
        help="Which agent to chat with (default: analytics)",
    )
    args = parser.parse_args()

    token_metadata_repo = create_token_metadata_repo()

    if args.agent == "analytics":
        agent = create_analytics_executor(token_metadata_repo)
        system_prompt = get_analytics_prompt(tokens=[])
        run_turn = lambda msgs, cfg: run_analytics_turn(
            agent, token_metadata_repo, msgs, cfg
        )
    else:
        agent = create_investor_executor()
        system_prompt = get_investor_agent_prompt(tokens=[], poolDeposits=[])
        run_turn = lambda msgs, cfg: run_investor_turn(agent, msgs, cfg)

    config = RunnableConfig(
        configurable={
            "tokens": [],
            "positions": [],
            "available_pools": [],
        }
    )

    conversation = [SystemMessage(content=system_prompt)]

    print(f"BitQuant CLI — {args.agent} agent")
    print("Type 'quit' or Ctrl+C to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        conversation.append(HumanMessage(content=user_input))

        try:
            response_text, tokens = await run_turn(conversation, config)
        except Exception as e:
            logging.error(f"Agent error: {e}", exc_info=True)
            print(f"\nError: {e}\n")
            # Remove the failed user message so conversation stays clean
            conversation.pop()
            continue

        print(f"\nBitQuant: {response_text}")
        if tokens:
            print("\nTokens mentioned:")
            for t in tokens:
                print(t)
        print()

    await token_metadata_repo.close()


if __name__ == "__main__":
    asyncio.run(main())

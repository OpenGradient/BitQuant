# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BitQuant is an AI agent framework for quantitative DeFi analytics and trading on Solana. It provides a natural language interface to portfolio analysis, token research, and yield optimization through a multi-agent system built with LangGraph and FastAPI.

## Common Commands

```bash
make install        # Install dependencies
make run           # Start FastAPI server on port 8000
make test          # Run all tests (unittest discover)
make integration-test  # Run server endpoint tests
make check         # Format with black + type check with mypy
make format        # Format with ruff
make chat          # Interactive test client
make docker        # Build Docker image
```

To run a single test file:
```bash
python3.13 -m pytest path/to/test_file.py -v
```

## Architecture

### Request Flow
```
User Query → FastAPI (/api/v2/agent/run) → Firebase Auth → Agent Selection (client-specified, defaults to analytics)
    ↓
    ├→ Analytics Agent: portfolio analysis, token research, market trends
    └→ Investor Agent: DeFi opportunity finding, yield optimization
    ↓
Agent executes tools → LLM inference → Post-process response → Return AgentMessage
```

### Key Components

- **agent/**: Agent orchestration - `agent_executors.py` creates LangGraph ReAct agents, `tools.py` defines agent tools, `prompts.py` loads Jinja2 templates
- **server/**: FastAPI app in `fastapi_server.py`, auth, validation, activity tracking
- **onchain/**: Blockchain data layer - `pools/` for DeFi protocol abstraction, `tokens/` for metadata, `portfolio/` for wallet analysis, `analytics/` for metrics
- **templates/**: Jinja2 prompt templates for agents (`analyst_agent.jinja2`, `investor_agent.jinja2`)
- **api/api_types.py**: All Pydantic models (Token, Pool, Portfolio, Message types)

### Protocol System
The `ProtocolRegistry` in `onchain/pools/protocol.py` manages DeFi protocol implementations. Each protocol (Orca, Kamino, Save) implements `get_pools()` returning Pool objects. Pools are cached and refreshed every 10 minutes.

### Tool System
Agent tools are `@tool` decorated async functions in `agent/tools.py`. Tools receive `RunnableConfig` for context access. Usage is tracked via `@track_tool_usage` decorator in `agent/telemetry.py`.

## Adding New Features

**New Agent**: Create executor function in `agent/agent_executors.py`, add prompt template in `templates/`, update router logic

**New Tool**: Add `@tool` decorated function in `agent/tools.py` or appropriate `onchain/` module, include in agent's tool list

**New Protocol**: Implement `Protocol` base class in `onchain/pools/solana/`, register in `main.py`

**API Changes**: Modify endpoints in `server/fastapi_server.py`, update types in `api/api_types.py`

## Environment

Requires `.env` with:
- Firebase credentials (FIREBASE_PROJECT_ID, etc.)
- SOLANA_RPC_URL
- LLM keys (OPENROUTER_API_KEY, GEMINI_API_KEY)
- COINGECKO_API_KEY
- AWS credentials for DynamoDB

Set `ENVIRONMENT=development` for local.

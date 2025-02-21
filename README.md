# OpenGradient DeFAI Agent

OpenGradient's DeFAI Agent is an AI system that helps users manage their DeFi operations through natural language commands. It exposes a REST API that turns user inputs like "deposit 20k" or "optimize my portfolio" into concrete DeFi actions.

## Structure

- `agent`: contains the agent logic
- `plugins`: plugins for integrating with different DeFi protocols
- `server`: contains Flask server that exposes API to interact with agent
- `strategies`: contains different strategies for portfolio allocation and optimization to be used by the agent
- `templates`: LLM prompt templates for agent

## Setup

1. `make venv`
2. `make install`
3. Create `.env` file and add `OG_PRIVATE_KEY` set to your OpenGradient private key
4. `make run`
5. (Optional) Run `make sample` for a sample query

## Testing

To run all tests, run `make test`

## Deployment

1. `make docker`
2. `make prod`

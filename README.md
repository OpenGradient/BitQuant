# OpenGradient DeFAI Agent

## Structure

- `agent`: contains the agent logic
- `plugins`: plugins for integrating with different DeFi protocols
- `server`: contains Flask server that exposes API to interact with agent
- `strategies`: contains different strategies for portfolio allocation and optimization to be used by the agent
- `templates`: LLM prompt templates for agent

## Setup

1. `make install`
2. Create `.env` file and add `OG_PRIVATE_KEY` set to your OpenGradient private key
3. `make run`
4. (Optional) Run `make sample` for a sample query

## Testing

To run all tests, run `make test` 

# Bluefin Agent

## Structure

- `agent`: contains the agent logic
- `api_types`: API type definitions
- `server`: contains Flask server that exposes API to interact with agent
- `templates`: LLM prompt templates for agent

## Setup

1. `make install`
2. Create `.env` file and add `PRIVATE_KEY` set to your OpenGradient private key
3. `make run`
4. (Optional) Run `make test` for a sample query

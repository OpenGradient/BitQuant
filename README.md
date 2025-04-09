# OpenQuant by OpenGradient

OpenQuant is an open-source AI agent framework for building AI Quant agents for quantitative DeFi including ML-powered analytics, trading, portfolio management, and more all through a natural language interface. It exposes a REST API that turns user inputs like "deposit 20k" or "optimize my portfolio" into concrete DeFi actions.

## Structure

- `agent`: contains the agent logic and tool definitions
- `api`: Server API input/output types
- `onchain`: Contains all classes that pull data about tokens, pools, etc
- `server`: contains Flask server that exposes API to interact with agent
- `templates`: LLM prompt templates for agent

## Setup

1. `make venv`
2. Activate virtual environment - `source venv/bin/activate`
3. `make install`
4. Create `.env` file and add `OG_PRIVATE_KEY` set to your OpenGradient private key
5. `make run`
6. (Optional) Run `make sample` for a sample query

## Testing

To run all tests, run `make test`

## Deployment

1. `make docker`
2. `make prod`

# OpenGradient + LangChain Agent

Combines OpenGradient's verifiable inference and advanced ML tools with LangChain.

## Structure

- The agent's tools are defined in `tools.py`
- The agent's prompts and instructions are in `prompts.py`
- The agent itself is defined (using the tools and prompts) in `agent.py`
- The main function for executing the agent is in `main.py`

## Agent

The agent is equipped by the following tools:

- Wikipedia search tool for looking up articles
- OpenGradient AlphaSense tool for Sui price prediction.

## Setup

1. Dependencies are defined in `requirements.txt`, run `pip install -r requirements.txt` to install all dependencies. We recommend using virtualenv.
2. You'll need an OpenGradient token in order to run inferences. You can use the `opengradient` CLI to create it, read more on our [official docs](https://docs.opengradient.ai/developers/sdk/#credentials-setup).
3. Make a copy of the `.env.sh.template` file in the root folder and rename to `.env.sh`. Fill in with your OpenGradient token as well as Twitter API keys.

## Execute

To execute the agent, simply run `python main.py <your-instruction-here>`. For example, `python main.py "what is the size of NYC"`

Next, you can add new tools, prompts, user input etc to customize the agent's behaviour.

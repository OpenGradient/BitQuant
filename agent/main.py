import os
import sys

import opengradient as og
from dotenv import load_dotenv
from agent import create_agent_executor

# Load environment variables from .env file
load_dotenv("../.env.sh")

# og.init(private_key=os.environ.get('PRIVATE_KEY'), email=None, password=None)

# # Check argument
# if len(sys.argv) != 2:
#     print("Usage: python main.py 'your prompt here'")
#     sys.exit(1)
# user_prompt = sys.argv[1]

agent = create_agent_executor()

rep = agent.nodes['agent']

print(help(rep))
exit(1)

# Execute agent
events = agent.stream(
    {"messages": [
        ("user", user_prompt)
    ]},
    stream_mode="values",
    debug=False # Set to True for debugging
)

# Print reasoning and result
for event in events:
    event["messages"][-1].pretty_print()

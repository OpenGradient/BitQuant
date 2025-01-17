from typing import List

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import BaseTool
from opengradient.mltools import create_og_model_tool, ToolType

# Define the tools the agent can use
def create_agent_toolkit() -> List[BaseTool]:
    tools = []

    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
    tools.append(wikipedia)

    # Add Spot Forecast model
    tools.append(create_og_model_tool(
        tool_type=ToolType.LANGCHAIN,
        model_cid="QmY1RjD3s4XPbSeKi5TqMwbxegumenZ49t2q7TrK7Xdga4",
        tool_name="SuiSpotForecast",
        input_getter=lambda: {"open_high_low_close": fetch_sui_price_candles()},
        output_formatter=lambda out: f"The predicted price change is: {out['destandardized_prediction'][0] * 100}%",
        tool_description="Runs an ML model to forecast the price of SUI 30 minutes from now. Requires no input."
    ))

    return tools

def fetch_sui_price_candles():
    # Should replace with real-time data from CEX/DEX
    return [
        [1,2,3,4],
        [1,2,3,4],
        [1,2,3,4],
        [1,2,3,4],
        [1,2,3,4],
        [1,2,3,4],
        [1,2,3,4],
        [1,2,3,4],
        [1,2,3,4],
        [1,2,3,4]
    ]

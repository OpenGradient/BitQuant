"""
OpenGradient AlphaSense tools for the TwoLigma agent.
"""
import os
import logging
import json
import traceback
from typing import Dict, Any, List

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

import opengradient as og
from opengradient.alphasense import create_run_model_tool, create_read_workflow_tool
from opengradient.alphasense.types import ToolType
from opengradient.types import InferenceResult  # Import from correct location

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('alphasense_tools')

# Input schemas for tools
class ReadWorkflowInput(BaseModel):
    """Schema for read workflow tool input."""
    workflow_contract_address: str = Field(
        description="The address of the workflow contract to read results from"
    )

class RunModelInput(BaseModel):
    """Schema for run model tool input."""
    model_cid: str = Field(
        description="The CID of the OpenGradient model to run"
    )
    model_input: Dict[str, Any] = Field(
        description="The input data to provide to the model"
    )

# SDK initialization flag
_sdk_initialized = False

def _ensure_sdk_initialized():
    """Initialize the OpenGradient SDK if it hasn't been already."""
    global _sdk_initialized
    if not _sdk_initialized:
        try:
            private_key = os.environ.get("OG_PRIVATE_KEY")
            if not private_key:
                raise ValueError("OG_PRIVATE_KEY environment variable must be set")

            email = os.environ.get("OG_EMAIL")
            if not email:
                raise ValueError("OG_EMAIL environment variable must be set")
            
            password = os.environ.get("OG_PASSWORD")
            if not password:
                raise ValueError("OG_PASSWORD environment variable must be set")
                
            logger.debug(f"Initializing OpenGradient SDK with email: {email}")
            og.init(private_key=private_key, email=email, password=password)
            _sdk_initialized = True
            logger.info("OpenGradient SDK initialized successfully.")
        except Exception as e:
            error_msg = f"Error initializing OpenGradient SDK: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            raise

def initialize_og_sdk():
    """
    Public function to initialize the OpenGradient SDK.
    This is a wrapper around the private _ensure_sdk_initialized function.
    """
    _ensure_sdk_initialized()

def format_model_output(inference_result: Any) -> str:
    """Format the output from model inference"""
    logger.debug(f"Formatting model output: {inference_result}")
    
    # Handle the case where inference_result is a tuple (tx_hash, result)
    if isinstance(inference_result, tuple) and len(inference_result) == 2:
        _, result = inference_result
        return str(result)
    
    # Handle the case where inference_result has a model_output attribute
    if hasattr(inference_result, 'model_output'):
        return str(inference_result.model_output)
    
    # Default case
    return str(inference_result)

def format_workflow_output(result: Any) -> str:
    """Format the output from workflow results"""
    logger.debug(f"Formatting workflow result: {json.dumps(result, default=str)}")
    return str(result)

def get_model_input(**kwargs):
    """Process model input from agent"""
    model_input = kwargs.get('model_input')
    logger.debug(f"Processing model input: {json.dumps(model_input, default=str)}")
    
    # If the input is already a dictionary, return it as is
    if isinstance(model_input, dict):
        return model_input
        
    # If the input is a list (OHLC data), wrap it with a 'data' key
    if isinstance(model_input, list):
        logger.debug("Converting list input to dict with 'data' key")
        return {"data": model_input}
        
    # Default case
    return model_input

def create_alphasense_read_workflow_tool() -> BaseTool:
    """
    Creates a tool for reading results from an AlphaSense workflow.
    
    Returns:
        BaseTool: A LangChain tool for reading AlphaSense workflow results
    """
    # Ensure the SDK is initialized
    _ensure_sdk_initialized()
    
    # Create the tool using the SDK function - now with the required workflow_contract_address
    return create_read_workflow_tool(
        tool_type=ToolType.LANGCHAIN,
        workflow_contract_address="0x71178d64eeF1D4b167bdF393407CB3a70F87a338",  # Default address
        tool_name="read_alphasense_workflow",
        tool_description="Reads results from an AlphaSense workflow execution",
        output_formatter=format_workflow_output
    )

def create_alphasense_run_model_tool() -> BaseTool:
    """
    Creates a tool for running an AlphaSense model.
    
    Returns:
        BaseTool: A LangChain tool for running AlphaSense models
    """
    # Ensure the SDK is initialized
    _ensure_sdk_initialized()
    
    # Create the tool using the SDK function
    return create_run_model_tool(
        tool_type=ToolType.LANGCHAIN,
        model_cid="QmRhcpDXfYCKsimTmJYrAVM4Bbvck59Zb2onj3MHv9Kw5N",  # Default model CID
        tool_name="run_alphasense_model",
        model_input_provider=get_model_input,
        model_output_formatter=format_model_output,
        tool_input_schema=RunModelInput,
        tool_description="Executes an AlphaSense ML model with the given inputs",
        inference_mode=og.InferenceMode.VANILLA
    )

def create_alphasense_toolkit() -> List[BaseTool]:
    """
    Creates and returns a list of AlphaSense-related tools.
    
    Returns:
        List[BaseTool]: A list containing AlphaSense tools
    """
    return [
        create_alphasense_read_workflow_tool(),
        create_alphasense_run_model_tool(),
    ] 
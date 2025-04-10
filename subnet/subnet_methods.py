import requests
from typing import Dict, Any, List
import logging

from subnet.api_types import QuantQuery, QuantResponse


def make_request(input_data: Dict[str, Any], endpoint: str) -> requests.Response:
    """Make a POST request to the specified endpoint"""
    return requests.post(
        endpoint,
        json=input_data,
        headers={"Content-Type": "application/json"},
    )

def subnet_evaluation(quant_query: QuantQuery, quant_response: QuantResponse) -> float:
    """
    Evaluate the subnet miner query based on the provided QuantQuery and QuantResponse.
    
    Args:
        quant_query (QuantQuery): The query object containing the query string and metadata.
        quant_response (QuantResponse): The response object containing the agent's response.
        
    Returns:
        float: A score representing the evaluation of the query and response.
    """
    # TODO: Evaluation logic based on quant_query and quant_response
    return 1.0  

def subnet_query(quant_query: QuantQuery) -> QuantResponse:
    """
    TODO:
    1. Metadata contains TEE -> return a remote attestation
    2. Remove whitelist check
    3. Handle wallet_address = None (can default to something)
    4. Create signature of sorts for quant_response
    
    Make a request to the agent with the provided QuantQuery and return a QuantResponse.
    
    Args:
        quant_query: A QuantQuery object containing the query, userID, and metadata
        
    Returns:
        A QuantResponse object containing the agent's response, or None if the request failed
    """
    print(quant_query)
    # Create context with the provided wallet address
    context = {
        "conversationHistory": [],
        "address": quant_query.userID,
    }
    
    # Set the endpoint
    endpoint = "http://127.0.0.1:5000/api/agent/run"
    
    # Format the message
    message = {"type": "user", "message": quant_query.query}
    
    # Prepare the payload
    payload = {
        "message": message, 
        "context": context
    }
    
    
    logging.info(f"Making request to {endpoint}")
    logging.info(f"Query: {quant_query.query}")
    logging.info(f"Wallet address: {quant_query.userID}")
    logging.info(f"Metadata: {quant_query.metadata}")
    
    # Send request to agent
    try:
        response = make_request(payload, endpoint)
        response.raise_for_status()
    except Exception as e:
        logging.error(f"HTTP Error: {e}")
        if hasattr(response, 'status_code'):
            logging.error(f"Status Code: {response.status_code}")
        if hasattr(response, 'text'):
            logging.error(f"Response Text: {response.text}")
        if hasattr(response, 'request') and hasattr(response.request, 'url'):
            logging.error(f"Request URL: {response.request.url}")
        return None
    
    # Parse the response
    agent_output = response.json()
    
    # Create and return a QuantResponse
    quant_response = QuantResponse(
        response=agent_output.get("message", "No message found in response"),
        signature=b'',  
        proofs=[],      
        metadata={}
    )
    
    return quant_response
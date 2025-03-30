import requests
from typing import Dict, Any, List
import logging
import sys
import os

# Add parent directory to path to import protocol
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from protocol import QuantQuery, QuantResponse

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    
    # Add metadata if provided
    if quant_query.metadata:
        payload["metadata"] = quant_query.metadata
    
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
        metadata=quant_query.metadata
    )
    
    return quant_response

"""
if __name__ == "__main__":
    # Sample values for subnet_query
    wallet_address = "AmZ68oe9jMSqBVyJWieHs6UR1vkyRZnSTyCKobUfT4Pb" 
    query = "What's the total value locked in Uniswap?"
    metadata = ["defi", "uniswap", "tvl"]
    
    # Create a QuantQuery object
    quant_query = QuantQuery(
        query=query,
        userID=wallet_address,
        metadata=metadata
    )
    
    print(f"\nTesting subnet_query function with:")
    print(f"Query: {quant_query.query}")
    print(f"UserID: {quant_query.userID}")
    print(f"Metadata: {quant_query.metadata}")
    
    # Call subnet_query with the QuantQuery object
    result = subnet_query(quant_query)
    
    if result:
        # Print the results
        print(f"\nResults:")
        print(f"Response: {result.response}")
        print(f"Signature: {result.signature}")
        print(f"Proofs: {result.proofs}")
        print(f"Metadata: {result.metadata}")
        
        logging.info("Subnet query test completed successfully")
    else:
        logging.error("Subnet query test failed") 
"""
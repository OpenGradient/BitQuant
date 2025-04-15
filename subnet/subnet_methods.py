import requests
from typing import Dict, Any, List
import logging
import os
import json
import re

import jinja2
from langchain_openai import ChatOpenAI
import os
import logging
import json
import re

from .api_types import QuantQuery, QuantResponse

evaluation_model = None

env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.dirname(os.path.abspath(__file__))))
LLM_MODEL = "google/gemini-2.0-flash-001"
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.getenv("OPENROUTER_API_KEY")

def create_evaluation_model() -> ChatOpenAI:

    return ChatOpenAI(
        model=LLM_MODEL,
        temperature=0.0,
        openai_api_base=BASE_URL,
        openai_api_key=API_KEY,
        request_timeout=120.0,
    )


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
    global evaluation_model
    if evaluation_model is None:
        evaluation_model = create_evaluation_model()

    try:
        template = env.get_template("evaluation_prompt.txt")
        prompt = template.render(
            user_prompt=quant_query.query,
            agent_answer="No response provided" if quant_response is None else quant_response.response)

        # Format messages properly for ChatOpenAI
        messages = [{"role": "user", "content": prompt}]
        response = evaluation_model.invoke(messages)

        # Parse the response
        answer = response.content if hasattr(response, "content") else response["content"]

        # Find ```json{{...}}``` in the answer
        match = re.search(r"```json\s*({.*})\s*```", answer, re.DOTALL)
        if not match:
            logging.error(f"Could not find JSON in model response: {answer}")
            return 0.0
        json_str = match.group(1)
        score = json.loads(json_str)["score"]
        # Normalize the score to be between 0 and 1
        return float(score) / 50
    except Exception as e:
        logging.error(f"subnet_evaluation error: {e}")
        return 0.0


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
    payload = {"message": message, "context": context}

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
        if hasattr(response, "status_code"):
            logging.error(f"Status Code: {response.status_code}")
        if hasattr(response, "text"):
            logging.error(f"Response Text: {response.text}")
        if hasattr(response, "request") and hasattr(response.request, "url"):
            logging.error(f"Request URL: {response.request.url}")
        return None

    # Parse the response
    agent_output = response.json()

    # Create and return a QuantResponse
    quant_response = QuantResponse(
        response=agent_output.get("message", "No message found in response"),
        signature=b"",
        proofs=[],
        metadata={},
    )

    return quant_response

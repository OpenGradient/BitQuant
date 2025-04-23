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
import opengradient as og
from opengradient import LlmInferenceMode

from .api_types import QuantQuery, QuantResponse

evaluation_model = None

env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(os.path.dirname(os.path.abspath(__file__)))
)
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


import time


def subnet_evaluation(quant_query: QuantQuery, quant_response: QuantResponse) -> float:
    """
    Evaluate the subnet miner query based on the provided QuantQuery and QuantResponse, with up to 3 retries on failure.

    Args:
        quant_query (QuantQuery): The query object containing the query string and metadata.
        quant_response (QuantResponse): The response object containing the agent's response.

    Returns:
        float: A score representing the evaluation of the query and response.
    """
    global evaluation_model
    if evaluation_model is None:
        evaluation_model = create_evaluation_model()

    retries = 3
    delay = 3.0
    last_exception = None
    for attempt in range(1, retries + 1):
        try:
            template = env.get_template("evaluation_prompt.txt")
            prompt = template.render(
                user_prompt=quant_query.query,
                agent_answer=(
                    "No response provided"
                    if quant_response is None
                    else quant_response.response
                ),
            )

            # Format messages properly for ChatOpenAI
            messages = [{"role": "user", "content": prompt}]
            response = evaluation_model.invoke(messages)

            # Parse the response
            answer = (
                response.content
                if hasattr(response, "content")
                else response["content"]
            )

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
            last_exception = e
            logging.error(f"subnet_evaluation attempt {attempt} failed: {e}")
            if attempt < retries:
                time.sleep(delay)
    logging.error(
        f"subnet_evaluation failed after {retries} attempts: {last_exception}"
    )
    return 0.0


def subnet_query(quant_query: QuantQuery) -> QuantResponse:
    """
    Make a request to the agent with the provided QuantQuery and return a QuantResponse.
    If TEE is enabled (via env or quant_query.metadata), use OG SDK (llm_chat). Otherwise, use REST agent as before.
    """
    use_tee = os.getenv("USE_OG_TEE", "").lower() in ("true")
    if use_tee:
        try:
            og.init(
                private_key=os.environ["OG_PRIVATE_KEY"],
                email=os.environ["OG_EMAIL"],
                password=os.environ["OG_PASSWORD"]
            )
            # Use the query string as prompt
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant.", "name": "HAL"},
                {"role": "user", "content": quant_query.query}
            ]
            model_cid = og.LLM.LLAMA_3_2_3B_INSTRUCT
            result = og.llm_chat(
                model_cid=model_cid,
                messages=messages
            )
            answer = result.chat_output['content']
            quant_response = QuantResponse(
                response=answer,
                signature=b"",
                proofs=[],
                metadata={"tee": True},
            )
            return quant_response
        except Exception as tee_e:
            logging.error(f"TEE/OG SDK query failed: {tee_e}")
            return QuantResponse(response="TEE/OG SDK error", signature=b"", proofs=[], metadata={"tee_error": str(tee_e)})

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

import requests
from typing import Dict, Any, Optional
import logging
import os
import json
import re
import jinja2
import time
from collections import OrderedDict

import opengradient as og
from langchain_openai import ChatOpenAI
from datadog import statsd

from subnet.api_types import QuantQuery, QuantResponse
from server.config import USE_TEE

evaluation_model = None

# Replay protection cache
REPLAY_CACHE_SIZE = 10_000  # Maximum number of entries to keep in cache
replay_cache = OrderedDict()  # OrderedDict to maintain insertion order for LRU behavior

env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(os.path.dirname(os.path.abspath(__file__)))
)
LLM_MODEL = "google/gemini-2.5-flash-lite"
BASE_URL = "https://openrouter.ai/api/v1"
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Evaluation constants
MAX_SCORE = 50.0
MAX_RETRIES = 3
RETRY_DELAY = 3.0


def _check_replay_attack(miner_id: str, response_content: str, query_hash: str) -> bool:
    """
    Check if this miner is attempting a replay attack by returning the same response
    for different queries.
    
    Args:
        miner_id: The ID of the miner (from userID)
        response_content: The response content from the miner
        query_hash: Hash of the current query
        
    Returns:
        True if replay attack detected, False otherwise
    """
    global replay_cache
    
    # Create a cache key for this miner's response
    response_hash = hash(response_content)
    cache_key = f"{miner_id}:{response_hash}"
    
    # Check if we've seen this exact response from this miner before
    if cache_key in replay_cache:
        cached_query_hash = replay_cache[cache_key]
        # If the query hash is different, it's a replay attack
        if cached_query_hash != query_hash:
            logging.warning(f"Replay attack detected: Miner {miner_id} returned same response for different queries")
            statsd.increment("subnet.replay_attack.detected")
            return True
    
    # Store this response in cache
    replay_cache[cache_key] = query_hash
    
    # Maintain cache size limit (LRU behavior)
    if len(replay_cache) > REPLAY_CACHE_SIZE:
        replay_cache.popitem(last=False)  # Remove oldest item
    
    return False


def _hash_query(query: str) -> str:
    """
    Create a simple hash of the query string for replay detection.
    
    Args:
        query: The query string
        
    Returns:
        Hash string of the query
    """
    return str(hash(query))


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


def _extract_score_from_response(response_content: str) -> Optional[float]:
    """
    Extract and validate the score from the LLM response.
    
    Args:
        response_content: The raw response content from the LLM
        
    Returns:
        The extracted score as a float, or None if extraction fails
    """
    # Find JSON block in the response
    match = re.search(r"```json\s*({.*})\s*```", response_content, re.DOTALL)
    if not match:
        logging.error(f"Could not find JSON in model response: {response_content}")
        return None

    json_str = match.group(1)
    try:
        parsed_json = json.loads(json_str.strip())
        score = parsed_json["score"]
        
        # Validate score is within expected range (0-50)
        if not isinstance(score, (int, float)) or score < 0 or score > MAX_SCORE:
            logging.error(f"Invalid score value: {score}. Expected range: 0-{MAX_SCORE}")
            return None
            
        return float(score)
        
    except (json.JSONDecodeError, KeyError) as e:
        logging.error(f"Failed to parse score from JSON: {e}. JSON string: {json_str}")
        return None


def _get_evaluation_model() -> ChatOpenAI:
    """Get or create the evaluation model singleton."""
    global evaluation_model
    if evaluation_model is None:
        evaluation_model = create_evaluation_model()
    return evaluation_model


def _invoke_evaluation_model(prompt: str) -> Optional[str]:
    """
    Invoke the evaluation model with the given prompt.
    
    Args:
        prompt: The evaluation prompt to send to the model
        
    Returns:
        The model's response content, or None if the request fails
    """
    try:
        model = _get_evaluation_model()
        messages = [{"role": "user", "content": prompt}]
        response = model.invoke(messages)
        
        # Handle different response formats
        if hasattr(response, "content"):
            return response.content
        elif isinstance(response, dict) and "content" in response:
            return response["content"]
        else:
            logging.error(f"Unexpected response format: {type(response)}")
            return None
            
    except Exception as e:
        logging.error(f"Failed to invoke evaluation model: {e}")
        return None


def subnet_evaluation(quant_query: QuantQuery, quant_response: QuantResponse) -> float:
    """
    Evaluate the subnet miner query based on the provided QuantQuery and QuantResponse.
    
    The evaluation uses a 5-criteria scoring system where each criterion is scored 0-10, 
    resulting in a maximum possible score of 50. The final score is normalized to 0-1 range.

    Args:
        quant_query: The query object containing the query string and metadata
        quant_response: The response object containing the agent's response

    Returns:
        A normalized score between 0 and 1 representing the evaluation quality
    """
    # Check for replay attack first
    miner_id = quant_response.metadata.get("miner_id", "unknown")
    agent_answer = quant_response.response if quant_response else "No response provided"
    query_hash = _hash_query(quant_query.query)
    
    if _check_replay_attack(miner_id, agent_answer, query_hash):
        logging.warning(f"Returning score 0 due to replay attack from miner {miner_id}")
        statsd.increment("subnet.evaluation.replay_penalty")
        return 0.0
    
    # Prepare the evaluation prompt
    template = env.get_template("evaluation_prompt.txt")
    
    prompt = template.render(
        user_prompt=quant_query.query,
        agent_answer=agent_answer,
    )

    # Retry logic for evaluation
    for attempt in range(MAX_RETRIES):
        try:
            # Get model response
            response_content = _invoke_evaluation_model(prompt)
            if response_content is None:
                continue
                
            logging.info(f"Evaluation Answer: {response_content}")
            
            # Extract and validate score
            score = _extract_score_from_response(response_content)
            if score is not None:
                logging.info(f"Raw Score: {score}")
                # Normalize the score to be between 0 and 1
                normalized_score = score / MAX_SCORE
                statsd.increment("subnet.evaluation.success")
                return normalized_score
                
        except Exception as e:
            logging.error(f"subnet_evaluation attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)

    logging.error(f"subnet_evaluation failed after {MAX_RETRIES} attempts")
    statsd.increment("subnet.evaluation.error")
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
    if USE_TEE:
        try:
            og.init(
                private_key=os.environ["OG_PRIVATE_KEY"],
                email=os.environ["OG_EMAIL"],
                password=os.environ["OG_PASSWORD"],
            )
            # Use the query string as prompt
            messages = [{"role": "user", "content": quant_query.query}]
            model_cid = og.TEE_LLM.META_LLAMA_3_1_70B_INSTRUCT
            result = og.llm_chat(
                model_cid=model_cid,
                messages=messages,
                inference_mode=og.LlmInferenceMode.TEE,
            )
            answer = result.chat_output["content"]
            quant_response = QuantResponse(
                response=answer,
                signature=b"",
                proofs=[],
                metadata={"tee": True},
            )
            return quant_response
        except Exception as tee_e:
            logging.error(f"TEE/OG SDK query failed: {tee_e}")
            return QuantResponse(
                response="TEE/OG SDK error",
                signature=b"",
                proofs=[],
                metadata={"tee_error": str(tee_e)},
            )

    print(quant_query)
    # Create context with the provided wallet address
    context = {
        "conversationHistory": [],
        "address": quant_query.userID,
    }

    # Set the endpoint
    endpoint = "http://127.0.0.1:5000/api/v2/agent/run"

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

import requests
from typing import Dict, Any
import logging
import os
import json
import re
import jinja2
from pydantic import BaseModel, Field
from collections import OrderedDict

import opengradient as og
from google import genai
from datadog import statsd

from subnet.api_types import QuantQuery, QuantResponse

# Replay protection cache
REPLAY_CACHE_SIZE = 10_000  # Maximum number of entries to keep in cache
replay_cache = OrderedDict()  # OrderedDict to maintain insertion order for LRU behavior

env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(os.path.dirname(os.path.abspath(__file__)))
)
LLM_MODEL = "gemini-2.5-flash-lite"
API_KEY = os.getenv("GEMINI_API_KEY")
SUBNET_PROMPT_INJECTION_PATTERNS = os.getenv("SUBNET_PROMPT_INJECTION_PATTERNS", "").split(",")

# Evaluation constants
MAX_SCORE = 50.0
MAX_RETRIES = 3
RETRY_DELAY = 3.0

client = genai.Client(api_key=API_KEY)


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
    miner_id = quant_response.metadata.get("miner_id")

    if not miner_id:
        logging.error("No miner ID found in metadata")
        return 0.0
    else:
        miner_id = miner_id.lower()
    
    agent_answer = quant_response.response if quant_response else "No response provided"

    # Cap the agent answer at 4000 characters
    agent_answer = agent_answer[:4000]

    query_hash = str(hash(quant_query.query))

    # Check for prompt injection attempts
    if _detect_prompt_injection(agent_answer):
        logging.warning(f"Prompt injection detected for miner {miner_id}, returning score 0")
        statsd.increment("subnet.evaluation.prompt_injection_penalty")
        return 0.0
    
    if _check_replay_attack(miner_id, agent_answer, query_hash):
        statsd.increment("subnet.evaluation.replay_penalty")
        return 0.0
    
    # Prepare the evaluation prompt
    template = env.get_template("evaluation_prompt.txt")
    
    prompt = template.render(
        user_prompt=quant_query.query,
        agent_answer=agent_answer,
    )

    try:
        # Get model response
        response_content = _invoke_evaluation_model(prompt)
        
        # Extract score from the JSON response
        score = _extract_score_from_response(response_content)

        # Normalize the score to be between 0 and 1
        normalized_score = float(score) / MAX_SCORE
        if normalized_score == 1.0:
            logging.info(f"Normalized score is 1.0 for answer: {agent_answer}")

        statsd.increment("subnet.evaluation.success")
        return normalized_score
    except Exception:
        logging.exception("Subnet evaluation failed")
        statsd.increment("subnet.evaluation.error")
        return 0.0


def subnet_query(quant_query: QuantQuery) -> QuantResponse:
    """
    Make a request to the agent with the provided QuantQuery and return a QuantResponse.
    Args:
        quant_query: A QuantQuery object containing the query, userID, and metadata
    Returns:
        A QuantResponse object containing the agent's response, or None if the request failed
    """
    from server.config import USE_TEE

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
        response = _make_request(payload, endpoint)
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


def _make_request(input_data: Dict[str, Any], endpoint: str) -> requests.Response:
    """Make a POST request to the specified endpoint"""
    return requests.post(
        endpoint,
        json=input_data,
        headers={"Content-Type": "application/json"},
    )


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


def _extract_score_from_response(response_content: str) -> float:
    """
    Extract and validate the score from the LLM response.
    
    Args:
        response_content: The raw response content from the LLM
        
    Returns:
        The extracted score as a float, or None if extraction fails
    """
    # With JSON mode, the response should be valid JSON directly
    # First try to parse the entire response as JSON
    try:
        parsed_json = json.loads(response_content.strip())
        
        # Check if the parsed JSON contains a score field
        if "score" not in parsed_json:
            raise ValueError(f"No 'score' field found in JSON: {response_content}")
            
        score = parsed_json["score"]
        
        # Validate score is within expected range (0-50)
        if not isinstance(score, (int, float)) or score < 0 or score > MAX_SCORE:
            raise ValueError(f"Invalid score value: {score}. Expected range: 0-{MAX_SCORE}")
            
        return float(score)
        
    except json.JSONDecodeError:
        # Fallback to regex-based extraction for backward compatibility
        # First try to find JSON block in markdown code blocks
        match = re.search(r"```json\s*({.*})\s*```", response_content, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            # If no markdown code block, try to find plain JSON object starting with {"score"
            match = re.search(r'\{[^}]*"score"[^}]*\}', response_content, re.DOTALL)
            if match:
                json_str = match.group(0)
            else:
                # Try to find any JSON object that might contain a score field
                # This handles cases where the response is just plain JSON like {"score": 45}
                match = re.search(r'\{[^}]*\}', response_content.strip())
                if match:
                    json_str = match.group(0)
                else:
                    raise ValueError("No JSON found in response")

        parsed_json = json.loads(json_str.strip())
        
        # Check if the parsed JSON contains a score field
        if "score" not in parsed_json:
            raise ValueError(f"No 'score' field found in JSON: {json_str}")
            
        score = parsed_json["score"]
        
        # Validate score is within expected range (0-50)
        if not isinstance(score, (int, float)) or score < 0 or score > MAX_SCORE:
            raise ValueError(f"Invalid score value: {score}. Expected range: 0-{MAX_SCORE}")
            
        return float(score)
            
def _detect_prompt_injection(response_content: str) -> bool:
    """
    Detect prompt injection attempts by looking for scoring instructions or JSON blocks
    in the agent response. Legitimate DeFi agent responses should never contain
    scoring instructions or evaluation-related JSON.
    
    Args:
        response_content: The agent's response content
        
    Returns:
        True if prompt injection is detected, False otherwise
    """
    if not response_content:
        return False
        
    response_lower = response_content.lower()
    
    # Check for scoring-related patterns
    for pattern in SUBNET_PROMPT_INJECTION_PATTERNS:
        if re.search(pattern, response_lower, re.IGNORECASE):
            logging.warning(f"Prompt injection detected - scoring pattern found: {pattern}")
            statsd.increment("subnet.prompt_injection.detected")
            return True
    
    # Check for JSON blocks that might be scoring attempts
    json_blocks = re.findall(r'```json\s*(\{.*?\})\s*```', response_content, re.DOTALL | re.IGNORECASE)
    for json_block in json_blocks:
        try:
            parsed = json.loads(json_block)
            if isinstance(parsed, dict) and "score" in parsed:
                logging.warning(f"Prompt injection detected - JSON block with score: {json_block}")
                statsd.increment("subnet.prompt_injection.detected")
                return True
        except json.JSONDecodeError:
            continue
    
    return False

def _invoke_evaluation_model(prompt: str) -> str:
    """
    Invoke the evaluation model with the given prompt.
    
    Args:
        prompt: The evaluation prompt to send to the model
        
    Returns:
        The model's response content as a string, or None if the request fails
    """
    from google.genai import types

    class Scoring(BaseModel):
        score: int = Field(..., description="The final score of the response")

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type='application/json',
            response_schema=Scoring,
            max_output_tokens=200,
            temperature=0.0,
        )
    )

    return response.text
            

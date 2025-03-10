import sys
import os
import json
# Make sure the defi module is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from typing import Dict, Any
from defi.analytics.resources.lookup_utils import search_pool_ids

TEST_CONTEXT = {
    "conversationHistory": [],
    "tokens": [
        {"amount": 100, "address": "So11111111111111111111111111111111111111112"},
        {"amount": 45333, "address": "mSoLzYCxHdYgdzU16g5QSh3i5K3z3KZK7ytfqcJm7So"},
        {"amount": 900, "address": "USDSwr9ApdHk5bvJKMjzff41FfuX8bSxdKcR81vTwcA"},
        {"amount": 5, "address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"},
        {"amount": 105454, "address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"},
    ],
    "poolPositions": [],
}

# Ensure it's working by doing a test import early
test_pools = search_pool_ids("test ethereum", limit=1)
print(f"Pool search import test: {'SUCCESS' if search_pool_ids else 'FAILED'}")
print(f"Found {len(test_pools)} test pools")

def main():
    context = TEST_CONTEXT.copy()

    # Ask user which agent to use
    agent_type = input("Choose an agent (regular/analytics): ").strip().lower()

    # Set endpoint based on agent type
    if agent_type == "analytics":
        endpoint = "http://127.0.0.1:5000/api/agent/analytics"
    else:
        endpoint = "http://127.0.0.1:5000/api/agent/run"

    while True:
        # read input from command line
        message = {"type": "user", "message": input("\nUser: ")}

        # Format the request payload correctly
        payload = {"message": message, "context": context}

        # send to agent
        response = make_request(payload, endpoint)
        response.raise_for_status()

        agent_output = response.json()
        answer = agent_output["message"]
        pools = agent_output.get("pools", [])

        # print results
        if agent_type == "analytics":
            response_json = response.json()
            if isinstance(response_json, dict) and "message" in response_json:
                print(f"Two-Ligma: {answer}")
            else:
                print(f"Two-Ligma: {response_json}")
        else:
            # Regular handling for other responses
            print(f"Two-Ligma: {answer}")
            print(pools)

        # append to history
        context["conversationHistory"].append(message)
        context["conversationHistory"].append(agent_output)

        # After you get the response from the agent and extract the text:
        answer = extract_final_response(response_json)
        print(f"Two-Ligma: {answer}")

        # Then check if we need to override:
        if (isinstance(answer, dict) and "message" in answer and 
            "no matching" in answer["message"].lower() and 
            "pool" in message["message"].lower()):
            print("\n🛑 AGENT FAILED - USING DIRECT POOL LOOKUP INSTEAD 🛑")
            
            # Determine chain
            query_lower = message["message"].lower()
            if "arbitrum" in query_lower:
                target_chain = "Arbitrum"
            elif "ethereum" in query_lower:
                target_chain = "Ethereum"
            # Add other chains as needed
            else:
                print("No chain specified in query")
                continue
            
            # Load pools directly - NO AGENT INVOLVED
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            json_path = os.path.join(project_root, "defi", "analytics", "resources", "pool_ids.json")
            
            try:
                with open(json_path, "r") as f:
                    all_pools = json.load(f)
                
                # Find pools for this chain with EXACT match
                chain_pools = []
                for pool in all_pools:
                    if 'chain' in pool and pool['chain'] == target_chain:
                        chain_pools.append(pool)
                
                print(f"DIRECT LOOKUP: Found {len(chain_pools)} {target_chain} pools")
                
                if not chain_pools:
                    continue  # No pools found, continue to next iteration
                
                # Format pools for display
                direct_result = f"# Found {len(chain_pools)} Pools on {target_chain}\n\n"
                
                for i, pool in enumerate(chain_pools[:5]):
                    direct_result += f"## Pool {i+1}: {pool.get('project', 'Unknown')} - {pool.get('symbol', 'Unknown')}\n\n"
                    direct_result += f"- **ID**: {pool.get('id', 'Unknown')}\n"
                    direct_result += f"- **Chain**: {pool.get('chain', 'Unknown')}\n"
                    direct_result += f"- **Project**: {pool.get('project', 'Unknown')}\n"
                    direct_result += f"- **Symbol**: {pool.get('symbol', 'Unknown')}\n\n"
                
                direct_result += f"*Data source: {len(chain_pools)} pools found directly from pool_ids.json*"
                
                # IMPORTANT: Actually print the result to the user
                print("\n" + direct_result + "\n")
                
                # Store this in the conversation history to replace the failed agent response
                updated_message = {
                    "message": direct_result,
                    "pools": chain_pools[:5],  # Include first 5 pools
                    "type": "assistant"
                }
                
                # Replace the failed response with our direct lookup results in the history
                context["conversationHistory"].pop()  # Remove the failed agent response
                context["conversationHistory"].append(updated_message)  # Add our correct response
                
                return direct_result
                
            except Exception as e:
                print(f"Error in direct override: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        elif (isinstance(answer, str) and 
              "no matching" in answer.lower() and 
              "pool" in message["message"].lower()):
            print("\n🛑 AGENT FAILED - USING DIRECT POOL LOOKUP INSTEAD 🛑")
            
            # Determine chain
            query_lower = message["message"].lower()
            if "arbitrum" in query_lower:
                target_chain = "Arbitrum"
            elif "ethereum" in query_lower:
                target_chain = "Ethereum"
            # Add other chains as needed
            else:
                print("No chain specified in query")
                continue
            
            # Load pools directly - NO AGENT INVOLVED
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            json_path = os.path.join(project_root, "defi", "analytics", "resources", "pool_ids.json")
            
            try:
                with open(json_path, "r") as f:
                    all_pools = json.load(f)
                
                # Find pools for this chain with EXACT match
                chain_pools = []
                for pool in all_pools:
                    if 'chain' in pool and pool['chain'] == target_chain:
                        chain_pools.append(pool)
                
                print(f"DIRECT LOOKUP: Found {len(chain_pools)} {target_chain} pools")
                
                if not chain_pools:
                    continue  # No pools found, continue to next iteration
                
                # Format pools for display
                direct_result = f"# Found {len(chain_pools)} Pools on {target_chain}\n\n"
                
                for i, pool in enumerate(chain_pools[:5]):
                    direct_result += f"## Pool {i+1}: {pool.get('project', 'Unknown')} - {pool.get('symbol', 'Unknown')}\n\n"
                    direct_result += f"- **ID**: {pool.get('id', 'Unknown')}\n"
                    direct_result += f"- **Chain**: {pool.get('chain', 'Unknown')}\n"
                    direct_result += f"- **Project**: {pool.get('project', 'Unknown')}\n"
                    direct_result += f"- **Symbol**: {pool.get('symbol', 'Unknown')}\n\n"
                
                direct_result += f"*Data source: {len(chain_pools)} pools found directly from pool_ids.json*"
                
                # IMPORTANT: Actually print the result to the user
                print("\n" + direct_result + "\n")
                
                # Store this in the conversation history to replace the failed agent response
                updated_message = {
                    "message": direct_result,
                    "pools": chain_pools[:5],  # Include first 5 pools
                    "type": "assistant"
                }
                
                # Replace the failed response with our direct lookup results in the history
                context["conversationHistory"].pop()  # Remove the failed agent response
                context["conversationHistory"].append(updated_message)  # Add our correct response
                
                return direct_result
                
            except Exception as e:
                print(f"Error in direct override: {str(e)}")
                import traceback
                traceback.print_exc()
                continue


def make_request(input_data: Dict[str, Any], endpoint: str) -> Any:
    return requests.post(
        endpoint,
        json=input_data,
        headers={"Content-Type": "application/json"},
    )


def extract_final_response(response_data):
    """Extract just the final response text from the analytics response"""
    # If it's a dictionary with 'messages' key
    if isinstance(response_data, dict) and "messages" in response_data:
        messages = response_data["messages"]
        # Find the last non-empty assistant message
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("content"):
                return msg["content"]
            # Handle case where it's an object with content attribute
            elif hasattr(msg, "content") and msg.content:
                return msg.content

    # Return the original if we can't parse it
    return response_data


def get_agent_data(query):
    # Load pools directly
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    json_path = os.path.join(project_root, "defi", "analytics", "resources", "pool_ids.json")
    
    # Target chain
    query_lower = query.lower()
    if "arbitrum" in query_lower:
        target_chain = "Arbitrum"
    elif "ethereum" in query_lower:
        target_chain = "Ethereum"
    elif "polygon" in query_lower:
        target_chain = "Polygon"
    else:
        target_chain = None
    
    # Load and filter directly
    try:
        with open(json_path, "r") as f:
            all_pools = json.load(f)
        
        # Filter by chain if specified
        chain_pools = []
        if target_chain:
            for pool in all_pools:
                if 'chain' in pool and pool['chain'] == target_chain:
                    chain_pools.append(pool)
            
            print(f"Found {len(chain_pools)} pools for {target_chain}")
        
        # Format as markdown for direct insertion
        pool_text = f"# HERE ARE {len(chain_pools)} POOLS FOR {target_chain}:\n\n"
        
        for i, pool in enumerate(chain_pools[:5]):
            pool_text += f"## Pool {i+1}: {pool.get('project', 'Unknown')} - {pool.get('symbol', 'Unknown')}\n"
            pool_text += f"- **ID**: {pool.get('id', 'Unknown')}\n"
            pool_text += f"- **Chain**: {pool.get('chain', 'Unknown')}\n"
            pool_text += f"- **Project**: {pool.get('project', 'Unknown')}\n"
            pool_text += f"- **Symbol**: {pool.get('symbol', 'Unknown')}\n\n"
    
    except Exception as e:
        print(f"Error loading pools: {str(e)}")
        pool_text = "No pools could be loaded."
        chain_pools = []
    
    return {
        "tokens": [],
        "poolDeposits": [],
        "availablePools": chain_pools,
        "raw_pool_data": pool_text  # Direct formatted text
    }


if __name__ == "__main__":
    main()

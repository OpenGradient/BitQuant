from typing import Set, List, Any, Tuple, Dict, Optional
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from pydantic import ValidationError
from langgraph.graph.graph import CompiledGraph, RunnableConfig
import json
from functools import wraps
from langchain_core.messages import SystemMessage, HumanMessage

from defi.stats import DefiMetrics
from defi.types import (
    AgentChatRequest,
    PoolQuery,
    Chain,
    Pool,
    UserMessage,
    AgentMessage,
    Message,
)
from agent.agent_executor import create_agent_executor, create_suggestions_executor, create_analytics_executor
from agent.prompts import get_agent_prompt, get_suggestions_prompt, get_analytics_prompt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(ROOT_DIR, "static")


def create_flask_app() -> Flask:
    """Create and configure the Flask application with routes.""" 
    app = Flask(__name__)
    CORS(app)

    # Initialize agents
    agent = create_agent_executor()
    suggestions_agent = create_suggestions_executor()
    analytics_agent = create_analytics_executor()
    app.config['PROPAGATE_EXCEPTIONS'] = True

    # Initialize metrics service
    defi_metrics = DefiMetrics()
    defi_metrics.refresh_metrics()

    # Set up error handlers for production environment
    if not app.config.get("TESTING"):

        @app.errorhandler(ValidationError)
        def handle_validation_error(e):
            return jsonify({"error": str(e)}), 400

        @app.errorhandler(Exception)
        def handle_generic_error(e):
            return jsonify({"error": str(e)}), 500

    @app.route("/api/healthcheck", methods=["GET"])
    def healthcheck():
        return jsonify({"status": "ok"})

    @app.route("/api/tokenlist", methods=["GET"])
    def get_tokenlist():
        file_path = os.path.join(STATIC_DIR, "tokenlist.json")

        if not os.path.isfile(file_path):
            return jsonify({"error": "Tokenlist file not found"}), 404

        return send_from_directory(STATIC_DIR, "tokenlist.json")

    @app.route("/api/agent/run", methods=["POST"])
    def run_agent():
        request_data = request.get_json()
        agent_request = AgentChatRequest(**request_data)

        response = handle_agent_chat_request(defi_metrics, agent_request, agent)

        return jsonify(response.model_dump())

    @app.route("/api/agent/suggestions", methods=["POST"])
    def run_suggestions():
        request_data = request.get_json()
        agent_request = AgentChatRequest(**request_data)

        suggestions = handle_suggestions_request(
            defi_metrics, agent_request, suggestions_agent
        )

        return jsonify({"suggestions": suggestions})

    @app.route("/api/agent/analytics", methods=["POST"])
    def run_analytics():
        """Endpoint for the DeFiDataScientist agent"""
        request_data = request.get_json()
        agent_request = AgentChatRequest(**request_data)

        response = handle_analytics_chat_request(defi_metrics, agent_request, analytics_agent)

        return jsonify(response.model_dump())

    return app


def handle_agent_chat_request(
    defi_metrics: DefiMetrics,
    request: AgentChatRequest,
    agent: CompiledGraph,
) -> AgentMessage:
    # Get compatible pools
    compatible_pools = defi_metrics.get_pools(
        PoolQuery(
            chain=Chain.SOLANA,
            protocols=["save", "kamino-lend"],
            tokens=[token.address for token in request.context.tokens],
        )
    )

    # Build main agent system prompt
    main_system_prompt = get_agent_prompt(
        protocol="Save",
        tokens=request.context.tokens,
        poolDeposits=request.context.poolPositions,
        availablePools=compatible_pools,
    )

    # Prepare message history
    message_history = [
        convert_to_agent_msg(m) for m in request.context.conversationHistory
    ]

    # Create messages for main agent
    main_messages = [
        ("system", main_system_prompt),
        *message_history,
        ("user", request.message.message),
    ]

    # Create config for the agent
    agent_config = RunnableConfig(
        configurable={
            "tokens": request.context.tokens,
            "positions": request.context.poolPositions,
            "available_pools": compatible_pools,
        }
    )

    # Run main agent
    main_result = run_main_agent(agent, main_messages, agent_config)

    return AgentMessage(
        message=main_result["output"],
        pools=extract_pools(main_result["messages"]),
    )


def handle_suggestions_request(
    defi_metrics: DefiMetrics,
    request: AgentChatRequest,
    suggestions_agent: CompiledGraph,
) -> List[str]:
    # Get compatible pools
    compatible_pools = defi_metrics.get_pools(
        PoolQuery(
            chain=Chain.SOLANA,
            protocols=["save", "kamino-lend"],
            tokens=[token.address for token in request.context.tokens],
        )
    )

    # Build suggestions agent system prompt
    suggestions_system_prompt = get_suggestions_prompt(
        protocol="Save",
        tokens=request.context.tokens,
        poolDeposits=request.context.poolPositions,
        availablePools=compatible_pools,
    )

    # Prepare message history
    message_history = [
        convert_to_agent_msg(m) for m in request.context.conversationHistory
    ]

    # Create messages for suggestions agent
    suggestions_messages = [
        ("system", suggestions_system_prompt),
        *message_history,
    ]

    # Create config for the agent
    agent_config = RunnableConfig(
        configurable={
            "tokens": request.context.tokens,
            "positions": request.context.poolPositions,
            "available_pools": compatible_pools,
        }
    )

    # Run suggestions agent
    suggestions = run_suggestions_agent(
        suggestions_agent, suggestions_messages, agent_config
    )

    return suggestions


def run_main_agent(
    agent: CompiledGraph, messages: List, config: RunnableConfig
) -> Dict[str, Any]:
    """Run the main agent and return the result as a dictionary."""
    try:
        result = agent.invoke({"messages": messages}, config)
        
        # Extract content regardless of result type
        output_text = None
        
        # Handle dictionary responses
        if isinstance(result, dict):
            if "output" in result and result["output"]:
                output_text = result["output"]
            elif "content" in result and result["content"]:
                output_text = result["content"]
            elif "messages" in result and result["messages"]:
                for msg in reversed(result["messages"]):
                    if msg.get("role") == "assistant" and msg.get("content"):
                        output_text = msg["content"]
                        break
        # Handle object with content attribute (like LangChain messages)
        elif hasattr(result, "content") and result.content:
            output_text = result.content
        # Handle string or other types
        elif result is not None:
            output_text = str(result)
            
        if not output_text:
            output_text = "No response was generated."
            
        return {"output": output_text}
    
    except Exception as e:
        print(f"Error in run_main_agent: {str(e)}")
        return {"output": f"I encountered an error processing your request: {str(e)}"}


def run_suggestions_agent(
    agent: CompiledGraph, messages: List, config: RunnableConfig
) -> List[str]:
    # Run agent directly
    result = agent.invoke({"messages": messages}, config=config)

    # Extract final message
    last_message = result["messages"][-1]

    try:
        string_list = last_message.content
        # Remove brackets and split by comma
        cleaned = string_list.strip("[]")
        # Split by comma and remove quotes
        python_list = [item.strip().strip("'\"") for item in cleaned.split(",")]

        return python_list
    except json.JSONDecodeError as e:
        print(f"Error parsing suggestions JSON: {e}")
        return []


def convert_to_agent_msg(message: Message) -> Tuple[str, str]:
    if isinstance(message, UserMessage):
        return ("user", message.message)
    elif isinstance(message, AgentMessage):
        return ("assistant", message.message)
    else:
        raise TypeError(f"Unexpected message type: {type(message)}")


def extract_pools(messages: List[Any]) -> List[Pool]:
    return [
        a
        for msg in messages
        if hasattr(msg, "artifact") and msg.artifact
        for a in msg.artifact
    ]


def run_analytics_agent(
    agent: CompiledGraph, messages: List, config: RunnableConfig
) -> Dict[str, Any]:
    """Run the analytics agent and return the result as a dictionary."""
    try:
        result = agent.invoke({"messages": messages}, config)
        
        # Debug info
        print(f"ANALYTICS RESULT TYPE: {type(result)}")
        
        # Extract content regardless of result type
        output_text = None
        
        # Handle LangGraph's AddableValuesDict
        if hasattr(result, "get") and callable(result.get):
            # Get the 'messages' list
            agent_messages = result.get("messages", [])
            
            # Find the last assistant message with content
            if agent_messages:
                for msg in reversed(agent_messages):
                    if hasattr(msg, "type") and msg.type == "assistant" and hasattr(msg, "content") and msg.content:
                        output_text = msg.content
                        break
                    if hasattr(msg, "role") and msg.role == "assistant" and hasattr(msg, "content") and msg.content:
                        output_text = msg.content
                        break
        
        # If we still don't have output, try fallback methods
        if not output_text:
            # Handle object with content attribute (like LangChain messages)
            if hasattr(result, "content") and result.content:
                output_text = result.content
            # Handle string or other types
            elif result is not None:
                output_text = str(result)
        
        if not output_text:
            output_text = "No response was generated."
            
        return {"output": output_text}
    
    except Exception as e:
        print(f"Error in run_analytics_agent: {str(e)}")
        return {"output": f"I encountered an error processing your request: {str(e)}"}


def handle_analytics_chat_request(
    defi_metrics: DefiMetrics,
    request: AgentChatRequest,
    agent: CompiledGraph,
) -> AgentMessage:
    # Extract the actual message text
    message_text = request.message
    
    # If request.message is a dictionary with a 'message' key, extract that
    if isinstance(message_text, dict) and 'message' in message_text:
        message_text = message_text['message']
    
    # Create system message with analytics prompt
    system_message = SystemMessage(
        content=get_analytics_prompt(question=message_text)
    )
    
    # Add the user's message - make sure it's a string
    user_message = HumanMessage(content=str(message_text))
    
    # Create the message history
    messages = [system_message, user_message]
    
    # Run the analytics agent
    config = RunnableConfig(callbacks=[])
    
    try:
        result = agent.invoke({"messages": messages}, config)
        
        # Debug info
        print(f"ANALYTICS RESULT TYPE: {type(result)}")
        
        # Extract the final text answer ONLY
        final_answer = None
        
        # Handle LangGraph's result with message history
        if hasattr(result, "get") and callable(result.get):
            messages_list = result.get("messages", [])
            
            # Find the last AIMessage with content
            for msg in reversed(messages_list):
                # Look for content in various message formats
                if hasattr(msg, "content") and msg.content:
                    final_answer = msg.content
                    break
                    
        # If we couldn't extract from messages, use fallbacks
        if not final_answer and hasattr(result, "content") and result.content:
            final_answer = result.content
            
        # Last resort - convert to string
        if not final_answer:
            final_answer = "I couldn't analyze that. Please try again."
        
        # Return ONLY the final answer text
        return AgentMessage(
            message=final_answer,  # Just the text, not the whole object
            pools=[],
        )
        
    except Exception as e:
        print(f"Analytics agent error: {str(e)}")
        return AgentMessage(
            message=f"Sorry, I encountered an error while analyzing your request: {str(e)}",
            pools=[],
        )

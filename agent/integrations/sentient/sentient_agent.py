import os
import logging
from sentient_agent_framework import AbstractAgent, DefaultServer, Session, Query, ResponseHandler
from typing import Optional

import requests

class BitQuantSentientAgent(AbstractAgent):
    def __init__(self, name: str = "BitQuant Sentient Agent"):
        super().__init__(name)
        # URL of the running BitQuant production server
        self.server_url = os.environ.get("BITQUANT_SERVER_URL")

    async def assist(self, session: Session, query: Query, response_handler: ResponseHandler):
        """
        Forward the Sentient chat query to the BitQuant production server and stream the response.
        """
        await response_handler.emit_text_block(
            "INFO", f"BitQuantSentientAgent received: {query.prompt}"
        )
        try:
            # TODO: Extract real wallet address and conversation history from session/query if available
            address = os.environ.get("BITQUANT_DEFAULT_WALLET_ADDRESS", "demo_wallet_address")
            payload = {
                "context": {
                    "address": address,
                    "conversationHistory": [],  # Optionally reconstruct from Sentient session
                },
                "message": {
                    "type": "user",
                    "message": query.prompt,
                },
                "agent": None,
            }
            # Add BitQuant API authentication header
            api_header = os.environ.get("SKIP_TOKEN_AUTH_HEADER")
            api_key = os.environ.get("SKIP_TOKEN_AUTH_KEY")
            headers = {api_header: api_key} if api_header and api_key else {}
            resp = requests.post(f"{self.server_url}/api/agent/run", json=payload, headers=headers)
            resp.raise_for_status()
            response_json = resp.json()
            response_text = response_json.get("message", "[No response from BitQuant server]")
            await response_handler.emit_text_block("RESPONSE", response_text)
        except Exception as e:
            logger.error(f"Error in BitQuantSentientAgent.assist: {e}", exc_info=True)
            await response_handler.emit_text_block("ERROR", f"BitQuant error: {e}")
        await response_handler.complete()


def start_sentient_chat_agent(agent: Optional[AbstractAgent] = None):
    """
    Start the Sentient Agent Framework server with the BitQuantSentientAgent.
    """
    if agent is None:
        agent = BitQuantSentientAgent()
    server = DefaultServer(agent)
    logger.info("Starting Sentient Agent server")
    server.run()

if __name__ == "__main__":
    start_sentient_chat_agent()

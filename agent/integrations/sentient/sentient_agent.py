import os
import logging
from sentient_agent_framework import (
    AbstractAgent,
    DefaultServer,
    Session,
    Query,
    ResponseHandler,
)
from typing import Optional

import requests


class BitQuantSentientAgent(AbstractAgent):
    def __init__(self, name: str = "BitQuant Sentient Agent"):
        super().__init__(name)
        self.server_url = os.environ.get("BITQUANT_SERVER_URL")

    async def assist(
        self, session: Session, query: Query, response_handler: ResponseHandler
    ):
        """
        Forward the Sentient chat query to the BitQuant production server and stream the response.
        """
        try:
            conversation_history = []
            for interaction in session.get_interactions():
                try:
                    user_content = interaction.request.content
                    conversation_history.append(
                        {"role": "user", "content": str(user_content)}
                    )
                except AttributeError:
                    pass
                for resp in getattr(interaction, "responses", []):
                    try:
                        agent_content = resp.event
                        conversation_history.append(
                            {"role": "agent", "content": str(agent_content)}
                        )
                    except AttributeError:
                        pass

            payload = {
                "context": {
                    "address": os.environ.get("BITQUANT_SENTIENT_WALLET_ADDRESS"),
                    "conversationHistory": conversation_history,
                },
                "message": {
                    "type": "user",
                    "message": query.prompt,
                },
            }
            api_header = os.environ.get("SKIP_TOKEN_AUTH_HEADER")
            api_key = os.environ.get("SKIP_TOKEN_AUTH_KEY")
            headers = {api_header: api_key} if api_header and api_key else {}
            try:
                resp = requests.post(
                    f"{self.server_url}/api/agent/run", json=payload, headers=headers
                )
                resp.raise_for_status()
                response_json = resp.json()
                response_text = response_json.get(
                    "message", "[No response from BitQuant server]"
                )
                await response_handler.emit_text_block("RESPONSE", response_text)
            except requests.exceptions.HTTPError as e:
                error_content = (
                    resp.content.decode(errors="replace")
                    if resp is not None
                    else str(e)
                )
                await response_handler.emit_text_block(
                    "ERROR", f"BitQuant error: {error_content}"
                )
                return
        except Exception as e:
            await response_handler.emit_text_block("ERROR", f"BitQuant error: {e}")
        await response_handler.complete()


def start_sentient_chat_agent(agent: Optional[AbstractAgent] = None):
    if agent is None:
        agent = BitQuantSentientAgent()
    server = DefaultServer(agent)
    logging.info("Starting Sentient Agent server")
    server.run()


if __name__ == "__main__":
    start_sentient_chat_agent()

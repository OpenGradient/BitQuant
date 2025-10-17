from __future__ import annotations
from datetime import timedelta, datetime
from typing import Optional
import os
import random
import json
import google.genai as genai
from pydantic import BaseModel, Field
import bittensor as bt
import asyncio


GENERATION_PROMPT = """Create a list of {question_count} questions that sound like they're being asked by a real person with varying levels of knowledge. The questions should be about common crypto topics. Some should be professionally formatted and others lazy (e.g., no capitalization, missing punctuation). The questions should cover the following topics, with an example for each:
* Portfolio Analytics: (e.g., "how can i evaluate my portfolio risk")
* Yield & Earning: (e.g., "what's the best way to earn yield on my ETH")
* Trading & Popularity: (e.g., "what are the trading tokens on solana?")
* Investment Advice: (e.g., "should i buy BONK?")
* Token Risk & Security: (e.g., "who are the top holders of pepe?")"""


_questioner: Optional[Questioner] = None
_questioner_lock = asyncio.Lock()


async def get_questioner() -> Questioner:
    global _questioner
    if _questioner is None:
        async with _questioner_lock:
            if _questioner is None:
                _questioner = Questioner(
                    api_key=os.getenv("GEMINI_API_KEY", ""),
                    question_count=42,
                    llm_model="gemini-2.5-flash-lite",
                    temperature=1.0,
                    question_lifetime=timedelta(days=7),
                )
    return _questioner


class Questioner:

    def __init__(
        self,
        api_key: str,
        question_lifetime: timedelta = timedelta(days=7),
        initial_questions: Optional[list[str]] = None,
        question_count: int = 42,
        llm_model: str = "gemini-2.5-flash-lite",
        temperature: float = 1.0,
    ):
        self.question_lifetime = question_lifetime
        self.questions: list[str] = initial_questions or []
        self.last_question_update = datetime.now()

        self.genai_client = genai.Client(api_key=api_key)
        self.llm_model: str = llm_model
        self.question_count: int = question_count
        self.temperature: float = temperature

        self._update_lock = asyncio.Lock()

    async def choose_question(self) -> str:
        await self.maybe_update_questions()
        return random.choice(self.questions)
    

    async def maybe_update_questions(self):
        if not self.is_update_due():
            return
        
        async with self._update_lock:
            if self.is_update_due():
                self.questions = await self.generate_exactly_n_questions(n=self.question_count)
                self.last_question_update = datetime.now()

    def is_update_due(self) -> bool:
        return not self.questions or self.is_update_time()

    def is_update_time(self) -> bool:
        return datetime.now() - self.last_question_update > self.question_lifetime

    async def generate_exactly_n_questions(self, n: int) -> list[str]:
        new_questions = []
        while len(new_questions) < n:
            qs = await self.generate_about_n_questions(n=n - len(new_questions))
            print(f"{len(qs) = }")
            new_questions.extend(qs)
        return new_questions[:n]

    async def generate_about_n_questions(self, n: int) -> list[str]:
        class Questions(BaseModel):
            questions: list[str] = Field(..., description="List of crypto questions")

        prompt = GENERATION_PROMPT.format(question_count=n)

        response = await self.genai_client.aio.models.generate_content(
            model=self.llm_model,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Questions,
                temperature=self.temperature,
            ),
        )

        if response_text := response.text:
            data = json.loads(response_text)
            return data["questions"]
        else:
            bt.logging.warning("No questions were generated!")
            return []

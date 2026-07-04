from __future__ import annotations

from collections.abc import AsyncIterator

from openai import AsyncOpenAI


class OpenAIChatService:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required")
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    async def stream(self, system_prompt: str, user_prompt: str) -> AsyncIterator[str]:
        stream = await self.client.chat.completions.create(
            model=self.model,
            temperature=0.2,
            stream=True,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        async for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta

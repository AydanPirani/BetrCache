import os
from enum import Enum
from dataclasses import dataclass
import httpx
import asyncio


class Provider(Enum):
    OPENAI = "openai"
    OPENROUTER = "openrouter"


@dataclass
class GPTOptions:
    model: str
    provider: Provider
    api_key: str
    prefix: str


@dataclass
class EmbeddingOptions:
    model: str
    provider: Provider
    api_key: str


async def get_gpt_response(prompt: str, options: GPTOptions) -> str:
    url = (
        "https://api.openai.com/v1/chat/completions"
        if options.provider == Provider.OPENAI
        else "https://openrouter.ai/api/v1/chat/completions"
    )
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {options.api_key}",
    }
    payload = {
        "model": options.model,
        "messages": [{"role": "user", "content": options.prefix + prompt}],
    }
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


async def get_embedding(input: str, options: EmbeddingOptions) -> list[float]:
    if options.provider == Provider.OPENROUTER:
        raise NotImplementedError("OpenRouter embeddings not supported")
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {options.api_key}",
    }
    payload = {"model": options.model, "input": input}
    async with httpx.AsyncClient() as client:
        resp = await client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
    return data["data"][0]["embedding"]

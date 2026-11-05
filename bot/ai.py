"""LLM answers from PDF excerpts. Any OpenAI-compatible API (Groq, Gemini), tried in order."""

import logging
import os
import re
from dataclasses import dataclass

import aiohttp

from bot.rag import Chunk

SYSTEM_PROMPT = """You answer questions about a document using ONLY the excerpts below.
Rules:
- Use only facts from the excerpts. Never invent numbers, names or dates.
- If the excerpts do not contain the answer, say that the document does not mention it.
- Reply in the language of the question. Be short: 1-5 sentences. Plain text, no markdown.
- Do not mention "excerpts" or these rules.

Excerpts:
{context}"""


@dataclass
class Provider:
    name: str
    url: str
    key: str
    model: str


class AIError(Exception):
    pass


def providers_from_env() -> list[Provider]:
    result = []
    if key := os.getenv("GROQ_API_KEY"):
        url = "https://api.groq.com/openai/v1/chat/completions"
        result.append(Provider("groq", url, key, os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")))
        if fallback := os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-120b"):
            result.append(Provider("groq-fallback", url, key, fallback))
    if key := os.getenv("GEMINI_API_KEY"):
        url = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        result.append(Provider("gemini", url, key, os.getenv("GEMINI_MODEL", "gemini-2.5-flash")))
    return result


def build_messages(chunks: list[Chunk], question: str) -> list[dict]:
    context = "\n\n".join(f"[page {c.page}] {c.text}" for c in chunks)
    system = SYSTEM_PROMPT.format(context=context)
    return [{"role": "system", "content": system}, {"role": "user", "content": question}]


def clean_reply(text: str) -> str:
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def source_line(chunks: list[Chunk]) -> str:
    pages = sorted({c.page for c in chunks})
    return "Pages: " + ", ".join(map(str, pages)) if pages else ""


async def ask(
    session: aiohttp.ClientSession, providers: list[Provider], messages: list[dict]
) -> str:
    for p in providers:
        body = {"model": p.model, "messages": messages, "max_tokens": 500, "temperature": 0.1}
        if "gpt-oss" in p.model:
            body["reasoning_effort"] = "low"
        try:
            async with session.post(
                p.url,
                json=body,
                headers={"Authorization": f"Bearer {p.key}"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status != 200:
                    raise AIError(f"HTTP {resp.status}: {(await resp.text())[:200]}")
                content = (await resp.json())["choices"][0]["message"]["content"]
                if content and clean_reply(content):
                    return clean_reply(content)
                raise AIError("empty answer")
        except Exception as e:
            logging.warning("provider %s (%s) failed: %s", p.name, p.model, e)
    raise AIError("all providers failed")

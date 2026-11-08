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
- End your reply with one last line in the exact format `PAGES: 2, 5` listing only the page numbers (from the [page N] markers) of excerpts whose facts you actually used, or `PAGES: none` if the document does not contain the answer.

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


def split_pages(reply: str, chunks: list[Chunk]) -> tuple[str, list[int]]:
    lines = reply.splitlines()
    raw = None
    for i in range(len(lines) - 1, -1, -1):
        if m := re.match(r"^\s*pages\s*:\s*(.*)$", lines[i], flags=re.I):
            raw = m.group(1).strip()
            lines.pop(i)
            break
    answer = "\n".join(lines).strip()
    fallback = [chunks[0].page] if chunks else []
    if raw is None:
        return answer, fallback
    if re.fullmatch(r"none\.?", raw, flags=re.I):
        return answer, []
    nums = [int(n) for n in re.findall(r"\d+", raw)]
    if not nums:
        return answer, fallback
    valid = {c.page for c in chunks}
    return answer, [n for n in dict.fromkeys(nums) if n in valid]


def source_line(pages: list[int]) -> str:
    unique = sorted(set(pages))
    return "Pages: " + ", ".join(map(str, unique)) if unique else ""


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

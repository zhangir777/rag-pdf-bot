"""Telegram bot: send a PDF, then ask questions about it."""

import asyncio
import logging
import os
import tempfile
import time
from pathlib import Path

import aiohttp
import sentry_sdk
from aiogram import Bot, Dispatcher, F, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message
from dotenv import load_dotenv

from bot import ai, rag

MAX_FILE_MB = 10
router = Router()
indexes: dict[int, tuple[str, rag.Index]] = {}  # user id -> (file name, index)

START_TEXT = (
    "Hi! Send me a PDF (up to 10 MB) and ask any question about it. "
    "I answer from the document and show the pages.\n\n/clear forgets the document."
)


@router.message(CommandStart())
async def start(message: Message):
    await message.answer(START_TEXT)


@router.message(Command("clear"))
async def clear(message: Message):
    indexes.pop(message.from_user.id, None)
    await message.answer("Done. Send a new PDF.")


@router.message(F.document)
async def document(message: Message, bot: Bot):
    doc = message.document
    if not (doc.file_name or "").lower().endswith(".pdf"):
        await message.answer("Please send a PDF file.")
        return
    if doc.file_size > MAX_FILE_MB * 1024 * 1024:
        await message.answer(f"The file is too big. Maximum is {MAX_FILE_MB} MB.")
        return
    await message.answer("Reading the document...")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "doc.pdf"
        await bot.download(doc, destination=path)
        try:
            pages = await asyncio.to_thread(rag.extract_pages, path)
        except Exception:
            logging.exception("cannot read pdf")
            await message.answer("I could not read this PDF.")
            return
    if not pages:
        await message.answer("There is no text in this PDF (maybe it is a scan).")
        return
    chunks = rag.chunk_pages(pages)
    indexes[message.from_user.id] = (doc.file_name, rag.Index(chunks))
    await message.answer(
        f"Ready: {doc.file_name}, {len(pages)} pages with text. Ask your question."
    )


@router.message(F.text)
async def question(message: Message, session: aiohttp.ClientSession, providers: list):
    entry = indexes.get(message.from_user.id)
    if not entry:
        await message.answer("Send me a PDF first.")
        return
    chunks = entry[1].search(message.text)
    if not chunks:
        await message.answer("I did not find anything about this in the document.")
        return
    try:
        answer = await ai.ask(session, providers, ai.build_messages(chunks, message.text))
    except ai.AIError:
        await message.answer("The AI is busy right now. Please try again in a minute.")
        return
    await message.answer(f"{answer}\n\n{ai.source_line(chunks)}")


async def heartbeat(session: aiohttp.ClientSession) -> None:
    """Send an 'up' metric to Datadog every minute (only if DD_API_KEY is set)."""
    key = os.environ.get("DD_API_KEY")
    if not key:
        return
    url = f"https://api.{os.environ.get('DD_SITE', 'datadoghq.com')}/api/v1/series"
    while True:
        body = {
            "series": [
                {
                    "metric": "bots.heroku.up",
                    "points": [[int(time.time()), 1]],
                    "type": "gauge",
                    "tags": ["service:rag-pdf-bot"],
                }
            ]
        }
        try:
            await session.post(url, json=body, headers={"DD-API-KEY": key})
        except aiohttp.ClientError:
            logging.warning("datadog heartbeat failed")
        await asyncio.sleep(60)


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    if dsn := os.environ.get("SENTRY_DSN"):  # optional error tracking
        sentry_sdk.init(dsn=dsn, environment="production", server_name="heroku-rag")
        sentry_sdk.set_tag("bot", "rag")
    providers = ai.providers_from_env()
    if not providers:
        raise SystemExit("Set GROQ_API_KEY or GEMINI_API_KEY in .env")
    bot = Bot(os.environ["BOT_TOKEN"])
    dp = Dispatcher()
    dp.include_router(router)
    async with aiohttp.ClientSession() as session:
        beat = asyncio.create_task(heartbeat(session))
        try:
            await dp.start_polling(bot, session=session, providers=providers)
        finally:
            beat.cancel()


if __name__ == "__main__":
    asyncio.run(main())

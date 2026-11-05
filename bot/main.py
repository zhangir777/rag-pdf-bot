"""Telegram bot: send a PDF, then ask questions about it."""

import asyncio
import logging
import os
import tempfile
from pathlib import Path

import aiohttp
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


async def main() -> None:
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    providers = ai.providers_from_env()
    if not providers:
        raise SystemExit("Set GROQ_API_KEY or GEMINI_API_KEY in .env")
    bot = Bot(os.environ["BOT_TOKEN"])
    dp = Dispatcher()
    dp.include_router(router)
    async with aiohttp.ClientSession() as session:
        await dp.start_polling(bot, session=session, providers=providers)


if __name__ == "__main__":
    asyncio.run(main())

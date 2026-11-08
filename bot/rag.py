"""PDF text extraction, chunking and BM25 search. No external search libraries."""

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader


@dataclass
class Chunk:
    page: int
    text: str


def extract_pages(path: str | Path) -> list[tuple[int, str]]:
    """Return [(page_number, text)] for pages that have text."""
    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        text = re.sub(r"\s+", " ", page.extract_text() or "").strip()
        if text:
            pages.append((number, text))
    return pages


def chunk_pages(pages: list[tuple[int, str]], size: int = 800, overlap: int = 100) -> list[Chunk]:
    """Split every page into pieces of about `size` characters, cutting at word borders."""
    chunks = []
    for number, text in pages:
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):
                space = text.rfind(" ", start, end)
                end = space if space > start else end
            chunks.append(Chunk(number, text[start:end].strip()))
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def tokenize(text: str) -> list[str]:
    """Lowercase words cut to 5 letters: a cheap stemmer that works for Russian and English."""
    return [w[:5] for w in re.findall(r"\w+", text.lower()) if len(w) > 1]


class Index:
    """BM25 index over chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(c.text)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = (sum(self.lengths) / len(chunks)) if chunks else 0
        df = Counter(term for d in self.docs for term in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, k: int = 4) -> list[Chunk]:
        terms = tokenize(query)
        scored = []
        for i, doc in enumerate(self.docs):
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if tf:
                    norm = (
                        tf
                        * (self.k1 + 1)
                        / (tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg))
                    )
                    score += self.idf[term] * norm
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True)
        cutoff = scored[0][0] * 0.4 if scored else 0
        return [self.chunks[i] for s, i in scored[:k] if s >= cutoff]

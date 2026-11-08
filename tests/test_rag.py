from bot.ai import build_messages, clean_reply, source_line, split_pages
from bot.rag import Chunk, Index, chunk_pages, tokenize

PAGES = [
    (1, "The company was founded in 2015 in Almaty. Our office is on Abay street."),
    (2, "Delivery takes three days. Refunds are possible within 14 days after purchase."),
    (3, "Компания работает по будням с девяти до шести. Выходной суббота и воскресенье."),
]


def test_chunking_keeps_page_and_size():
    chunks = chunk_pages([(7, "word " * 500)], size=200, overlap=40)
    assert len(chunks) > 5
    assert all(c.page == 7 and 0 < len(c.text) <= 200 for c in chunks)


def test_search_finds_right_page():
    index = Index(chunk_pages(PAGES))
    assert index.search("how long does delivery take")[0].page == 2
    assert index.search("when was the company founded")[0].page == 1


def test_search_handles_russian_word_forms():
    index = Index(chunk_pages(PAGES))
    assert index.search("в какие дни работает компания?")[0].page == 3


def test_search_no_match_and_empty_index():
    assert Index(chunk_pages(PAGES)).search("zzzz qqqq") == []
    assert Index([]).search("anything") == []


def test_tokenize_stems():
    assert tokenize("Delivery deliveries") == ["deliv", "deliv"]


def test_prompt_reply_and_sources():
    chunks = Index(chunk_pages(PAGES)).search("refunds")
    messages = build_messages(chunks, "refunds?")
    assert "[page 2]" in messages[0]["content"] and messages[1]["content"] == "refunds?"
    assert clean_reply("<think>hmm</think> Yes, 14 days.") == "Yes, 14 days."
    assert source_line([c.page for c in chunks]) == "Pages: 2"
    assert source_line([5, 2, 5]) == "Pages: 2, 5"
    assert source_line([]) == ""


def test_split_pages():
    chunks = [Chunk(2, "a"), Chunk(5, "b"), Chunk(8, "c")]
    assert split_pages("Answer here.\nPAGES: 2, 5", chunks) == ("Answer here.", [2, 5])
    assert split_pages("First\npages: 2\nLast\nPages: 5", chunks) == ("First\npages: 2\nLast", [5])
    assert split_pages("Not mentioned.\nPAGES: none", chunks) == ("Not mentioned.", [])
    assert split_pages("Answer.\nPAGES: ???", chunks) == ("Answer.", [2])
    assert split_pages("Answer without pages line.", chunks) == ("Answer without pages line.", [2])
    assert split_pages("Answer.\nPAGES: 5, 99", chunks) == ("Answer.", [5])
    assert split_pages("Answer.\nPAGES: 99", chunks) == ("Answer.", [])


def test_search_drops_low_score_chunks():
    pages = [
        (1, "alpha beta gamma delta epsilon"),
        (2, "alpha other words here"),
    ]
    results = Index(chunk_pages(pages)).search("alpha beta gamma delta epsilon")
    assert [c.page for c in results] == [1]

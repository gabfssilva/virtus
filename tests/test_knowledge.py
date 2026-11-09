import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from tests.scripted import Scripted, calls
from tests.test_pdf import document
from virtus import AssistantMessage, Embedder, Store, ToolResult, agent, knowledge, memory
from virtus.provider import Kind


def write(root: Path, name: str, text: str) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / name).write_text(text)


def describe_store():
    def it_titles_documents_from_frontmatter_heading_or_file_name(tmp_path: Path):
        write(tmp_path, "a.md", "---\ntitle: 'From frontmatter'\n---\n# Heading\nbody")
        write(tmp_path, "b.md", "# From heading\nbody")
        write(tmp_path, "c.txt", "# not a heading in txt\nbody")
        write(tmp_path / "nested", "d.md", "body only")
        write(tmp_path, "ignored.py", "x = 1")

        documents = Store(tmp_path).documents()

        assert [(d.id, d.title) for d in documents] == [
            ("a.md", "From frontmatter"),
            ("b.md", "From heading"),
            ("c.txt", "c"),
            ("nested/d.md", "d"),
        ]
        assert [d.content for d in documents] == [
            "# Heading\nbody",
            "body",
            "# not a heading in txt\nbody",
            "body only",
        ]

    def it_accepts_files_and_directories_as_roots(tmp_path: Path):
        write(tmp_path / "docs", "guide.md", "guide")
        write(tmp_path, "manual.txt", "manual")

        store = Store(tmp_path / "docs", tmp_path / "manual.txt")

        assert [d.id for d in store.documents()] == ["guide.md", "manual.txt"]

    async def it_ranks_by_bm25_and_drops_non_matches(tmp_path: Path):
        write(tmp_path, "cats.md", "Cats purr. Cats sleep. A cat is small.")
        write(tmp_path, "dogs.md", "Dogs bark and a dog fetches.")
        write(tmp_path, "both.md", "Cats and dogs can live together.")

        assert [d.id for d in await Store(tmp_path).search("cats")] == ["cats.md", "both.md"]
        assert await Store(tmp_path).search("elephants") == []

    async def it_ranks_by_cosine_with_an_embedder_and_keeps_the_vectors_in_an_index(tmp_path: Path):
        write(tmp_path, "cats.md", "# Felines\nThey purr.")
        write(tmp_path, "dogs.md", "# Canines\nThey bark.")
        embedder = Counting()
        store = Store(tmp_path, embedder=embedder)

        assert [d.id for d in await store.search("kitten", limit=1)] == ["cats.md"]
        assert [d.id for d in await store.search("puppy")] == ["dogs.md", "cats.md"]
        assert embedder.calls == [
            (["kitten"], "query"),
            (["Felines\n\nThey purr.", "Canines\n\nThey bark."], "document"),
            (["puppy"], "query"),
        ]
        index = json.loads((tmp_path / ".embeddings.json").read_text())
        assert sorted(index.values()) == [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]

        write(tmp_path, "dogs.md", "# Canines\nThey bark and fetch.")
        (tmp_path / "cats.md").unlink()
        await store.search("puppy")

        assert embedder.calls[-2:] == [(["puppy"], "query"), (["Canines\n\nThey bark and fetch."], "document")]
        assert list(json.loads((tmp_path / ".embeddings.json").read_text()).values()) == [[0.0, 1.0, 0.0]]

    async def it_takes_the_index_name_and_the_bm25_constants(tmp_path: Path):
        write(tmp_path, "cats.md", "# Cats\nCats purr.")
        write(tmp_path, "long.md", "# Long\n" + "cats " * 3 + "filler " * 100)

        await Store(tmp_path, embedder=Counting(), index="vectors.json").search("cats")

        assert (tmp_path / "vectors.json").exists() and not (tmp_path / ".embeddings.json").exists()
        assert [p.id for p in await Store(tmp_path).search("cats")] == ["cats.md", "long.md"]
        assert [p.id for p in await Store(tmp_path, b=0).search("cats")] == ["long.md", "cats.md"]

    async def it_embeds_everything_again_when_the_vectors_change_size(tmp_path: Path):
        write(tmp_path, "cats.md", "They purr.")
        (tmp_path / ".embeddings.json").write_text(json.dumps({"stale": [1.0, 0.0]}))
        embedder = Counting()

        await Store(tmp_path, embedder=embedder).search("kitten")

        assert embedder.calls[1] == (["cats\n\nThey purr."], "document")
        assert list(json.loads((tmp_path / ".embeddings.json").read_text()).values()) == [[1.0, 0.0, 0.0]]


def describe_passages():
    def it_cuts_long_documents_at_paragraphs_with_overlapping_words(tmp_path: Path):
        paragraphs = [f"Paragraph {i} " + "word " * 120 for i in range(8)]
        write(tmp_path, "long.md", "# Long\n" + "\n\n".join(paragraphs))
        write(tmp_path, "short.md", "# Short\nbody")

        passages = Store(tmp_path).passages()

        assert [(p.id, p.number, p.total) for p in passages] == [
            ("long.md#1", 1, 4),
            ("long.md#2", 2, 4),
            ("long.md#3", 3, 4),
            ("long.md#4", 4, 4),
            ("short.md", 1, 1),
        ]
        first, second = passages[0].content, passages[1].content
        assert first.endswith("word") and len(first) <= 2_000
        assert first[-200:].split()[-3:] == second.split()[:3]
        assert second.split("\n\n")[1].startswith("Paragraph")

    def it_takes_the_passage_size_and_overlap(tmp_path: Path):
        write(tmp_path, "long.md", "# Long\n" + "word " * 200)

        passages = Store(tmp_path, passage=300, overlap=0).passages()

        assert len(passages) == 4 and all(len(p.content) <= 300 for p in passages)
        assert sum(len(p.content.split()) for p in passages) == 200
        with pytest.raises(ValueError, match="overlap"):
            Store(tmp_path, passage=300, overlap=150)

    async def it_searches_and_reads_a_long_document_a_passage_at_a_time(tmp_path: Path):
        filler = "\n\n".join("Background " + "text " * 100 for _ in range(6))
        write(tmp_path, "manual.md", f"# Manual\n{filler}\n\nRefunds take five business days.\n\n{filler}")
        hit = (await Store(tmp_path).search("refunds"))[0]
        provider = Scripted(
            calls(("docs_read", {"id": "manual.md"})),
            calls(("docs_read", {"id": hit.id})),
            AssistantMessage("five days"),
        )

        @agent(provider=provider, capabilities=[knowledge(tmp_path, name="docs", max_tokens=100)])
        async def bot(question: str) -> str: ...

        await bot("refunds?")

        total = hit.total
        first, found = provider.requests[1].messages[-1].content, provider.requests[2].messages[-1].content
        assert first.startswith("# Manual\n\nBackground") and first.endswith(
            f"[passage 1 of {total}; next is manual.md#2]"
        )
        assert "Refunds take five business days." in found and f"[passage {hit.number} of {total}" in found

    async def it_reads_pdfs_by_page_and_embeds_each_passage(tmp_path: Path):
        pages = ["\n".join(["Cats purr " * 8] * 30), "\n".join(["Dogs bark " * 8] * 30)]
        (tmp_path / "pets.pdf").write_bytes(document(*pages))
        embedder = Counting()
        store = Store(tmp_path, embedder=embedder)

        [pets] = store.documents()

        assert (pets.id, pets.title) == ("pets.pdf", "pets")
        assert pets.content.startswith("[page 1]\nCats purr") and "[page 2]\nDogs bark" in pets.content
        [hit] = await store.search("puppy", limit=1)
        assert "Dogs bark" in hit.content and hit.id.startswith("pets.pdf#")
        assert len(embedder.calls[1][0]) == len(store.passages()) > 1


def describe_knowledge():
    async def it_injects_everything_while_it_fits(tmp_path: Path):
        write(tmp_path, "faq.md", "# FAQ\nRefunds take 5 days.")
        provider = Scripted(AssistantMessage("ok"))

        @agent(provider=provider, system="base", capabilities=[knowledge(tmp_path)])
        async def bot(question: str) -> str: ...

        await bot("?")

        (request,) = provider.requests
        assert request.system.startswith("base\n\n## knowledge\n\n")
        assert "### FAQ (faq.md)\nRefunds take 5 days." in request.system
        assert request.tools == ()

    async def it_injects_the_index_and_a_read_tool_when_the_content_is_too_big(tmp_path: Path):
        write(tmp_path, "faq.md", "# FAQ\n" + "Refunds take 5 days. " * 50)
        write(tmp_path, "terms.md", "# Terms\n" + "Be nice. " * 50)
        provider = Scripted(calls(("docs_read", {"id": "faq.md"})), AssistantMessage("5 days"))

        @agent(provider=provider, capabilities=[knowledge(tmp_path, name="docs", max_tokens=100)])
        async def bot(question: str) -> str: ...

        assert await bot("refunds?") == "5 days"

        first, second = provider.requests
        assert "- faq.md: FAQ\n- terms.md: Terms" in first.system
        assert [t.name for t in first.tools] == ["docs_read"]
        assert second.messages[-1].content == "# FAQ\n\n" + ("Refunds take 5 days. " * 50).strip()

    async def it_searches_by_meaning_with_an_embedder(tmp_path: Path):
        for i in range(40):
            write(tmp_path, f"topic-{i:02}.md", f"# Topic {i}\nSomething about subject number {i}.")
        write(tmp_path, "cats.md", "# Felines\nThey purr.")
        provider = Scripted(calls(("knowledge_search", {"query": "kitten"})), AssistantMessage("ok"))

        @agent(provider=provider, capabilities=[knowledge(tmp_path, max_tokens=50, embedder=Counting())])
        async def bot(question: str) -> str: ...

        await bot("?")

        first, second = provider.requests
        assert first.tools[0].description == "Search knowledge by meaning; returns ids to read."
        assert second.messages[-1].content.startswith("- cats.md: Felines\n  They purr.")

    async def it_takes_the_intro_estimate_hits_and_snippet(tmp_path: Path):
        for i in range(3):
            write(tmp_path, f"refunds-{i}.md", f"# Refunds {i}\nRefunds take {i} business days.")
        provider = Scripted(calls(("knowledge_search", {"query": "refunds"})), AssistantMessage("ok"))
        consulting = knowledge(tmp_path, intro="Policies.", max_tokens=10, chars_per_token=1, hits=2, snippet=7)

        @agent(provider=provider, capabilities=[consulting])
        async def bot(question: str) -> str: ...

        await bot("?")

        first, second = provider.requests
        assert first.system.startswith("## knowledge\n\nPolicies.\n\n3 entries.")
        assert second.messages[-1].content.count("\n  Refunds…") == 2

    async def it_offers_search_when_even_the_index_is_too_big(tmp_path: Path):
        for i in range(40):
            write(tmp_path, f"topic-{i:02}.md", f"# Topic {i}\nSomething about subject number {i}.")
        write(tmp_path, "refunds.md", "# Refund policy\nRefunds take five business days.")
        provider = Scripted(calls(("knowledge_search", {"query": "refund policy"})), AssistantMessage("ok"))

        @agent(provider=provider, capabilities=[knowledge(tmp_path, max_tokens=50)])
        async def bot(question: str) -> str: ...

        await bot("?")

        first, second = provider.requests
        assert "41 entries" in first.system
        assert [t.name for t in first.tools] == ["knowledge_search", "knowledge_read"]
        assert second.messages[-1].content.startswith("- refunds.md: Refund policy\n  Refunds take five business days.")


def describe_memory():
    async def it_persists_notes_across_agents_on_the_same_path(tmp_path: Path):
        remember = calls(("memory_remember", {"title": "User's name", "content": "Ana"}))
        writer = Scripted(remember, AssistantMessage("noted"))

        @agent(provider=writer, capabilities=[memory(tmp_path / "notes")])
        async def first(message: str) -> str: ...

        assert await first("I'm Ana") == "noted"
        assert writer.requests[1].messages[-1].content == "saved as user-s-name.md"
        assert (tmp_path / "notes" / "user-s-name.md").read_text().startswith("---\ntitle: User's name\ncreated: ")

        reader = Scripted(AssistantMessage("Ana"))

        @agent(provider=reader, capabilities=[memory(tmp_path / "notes")])
        async def second(message: str) -> str: ...

        assert await second("who am I?") == "Ana"
        assert "### User's name (user-s-name.md)\nAna" in reader.requests[0].system
        assert [t.name for t in reader.requests[0].tools] == ["memory_remember", "memory_forget"]

    async def it_forgets_by_id_and_reports_unknown_ids(tmp_path: Path):
        write(tmp_path, "old.md", "---\ntitle: Old\n---\nstale")
        provider = Scripted(
            calls(("memory_forget", {"id": "old.md"}), ("memory_forget", {"id": "missing.md"})), AssistantMessage("ok")
        )

        @agent(provider=provider, capabilities=[memory(tmp_path)])
        async def bot(message: str) -> str: ...

        await bot("clean up")

        deleted, missing = provider.requests[1].messages[-2:]
        assert isinstance(deleted, ToolResult) and isinstance(missing, ToolResult)
        assert (deleted.content, deleted.is_error) == ("deleted old.md", False)
        assert missing.is_error and "missing.md" in missing.content
        assert not (tmp_path / "old.md").exists()

    async def it_starts_empty(tmp_path: Path):
        provider = Scripted(AssistantMessage("ok"))

        @agent(provider=provider, capabilities=[memory(tmp_path / "none")])
        async def bot(message: str) -> str: ...

        await bot("hi")

        assert "(nothing yet)" in provider.requests[0].system


class Counting(Embedder):
    """Three axes: cats, dogs and everything else; records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[str], Kind]] = []

    async def embed(self, texts: Sequence[str], kind: Kind) -> list[list[float]]:
        self.calls.append((list(texts), kind))
        return [self.vector(text) for text in texts]

    @staticmethod
    def vector(text: str) -> list[float]:
        lowered = text.lower()
        if any(word in lowered for word in ("cat", "feline", "kitten", "purr")):
            return [1.0, 0.0, 0.0]
        if any(word in lowered for word in ("dog", "canine", "puppy", "bark")):
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


@pytest.mark.parametrize("arguments", [{"id": "../escape.md"}])
def it_never_resolves_ids_outside_the_store(tmp_path: Path, arguments: dict[str, str]):
    (tmp_path / "escape.md").write_text("secret")
    write(tmp_path / "store", "inside.md", "ok")

    assert Store(tmp_path / "store").read(json.loads(json.dumps(arguments))["id"]) is None

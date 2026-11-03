import json
from pathlib import Path

import pytest

from tests.scripted import Scripted, calls
from virtus import AssistantMessage, Store, ToolResult, agent, knowledge, memory


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

    def it_ranks_by_bm25_and_drops_non_matches(tmp_path: Path):
        write(tmp_path, "cats.md", "Cats purr. Cats sleep. A cat is small.")
        write(tmp_path, "dogs.md", "Dogs bark and a dog fetches.")
        write(tmp_path, "both.md", "Cats and dogs can live together.")

        assert [d.id for d in Store(tmp_path).search("cats")] == ["cats.md", "both.md"]
        assert Store(tmp_path).search("elephants") == []


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


@pytest.mark.parametrize("arguments", [{"id": "../escape.md"}])
def it_never_resolves_ids_outside_the_store(tmp_path: Path, arguments: dict[str, str]):
    (tmp_path / "escape.md").write_text("secret")
    write(tmp_path / "store", "inside.md", "ok")

    assert Store(tmp_path / "store").read(json.loads(json.dumps(arguments))["id"]) is None

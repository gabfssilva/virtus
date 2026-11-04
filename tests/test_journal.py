from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from tests.scripted import Scripted, calls
from virtus import (
    AssistantMessage,
    Delta,
    Image,
    JsonJournal,
    Provider,
    Request,
    ToolCall,
    ToolResult,
    UserMessage,
    agent,
    on_input,
    tool,
)

PICTURE = Image(b"\x89PNG", "image/png")
HISTORY = (
    UserMessage("What is this?", (PICTURE,)),
    AssistantMessage("Checking.", (ToolCall("c1", "zoom", '{"factor": 2}'),)),
    ToolResult("c1", "zoom", "[image]", images=(PICTURE,)),
    ToolResult("c2", "time", "boom", is_error=True),
    AssistantMessage("A red square."),
)


def describe_json_journal():
    def it_round_trips_every_kind_of_message(tmp_path: Path):
        journal = JsonJournal(tmp_path)

        journal.write(("ana",), HISTORY)

        assert journal.read(("ana",)) == HISTORY

    def it_returns_none_for_an_unknown_thread(tmp_path: Path):
        assert JsonJournal(tmp_path).read(("nobody",)) is None

    def it_keeps_children_under_their_parent_and_deletes_them_together(tmp_path: Path):
        journal = JsonJournal(tmp_path)
        journal.write(("ana",), HISTORY[:1])
        journal.write(("ana", "researcher", "bees"), HISTORY[:1])
        journal.write(("bia",), HISTORY[:1])

        assert (tmp_path / "ana" / "researcher" / "bees.json").exists()

        journal.delete(("ana",))

        assert journal.read(("ana",)) is None
        assert journal.read(("ana", "researcher", "bees")) is None
        assert journal.read(("bia",)) == HISTORY[:1]

    def it_keeps_every_path_inside_the_directory(tmp_path: Path):
        journal = JsonJournal(tmp_path / "threads")

        journal.write(("..", "../escape", "a/b"), HISTORY[:1])

        assert journal.read(("..", "../escape", "a/b")) == HISTORY[:1]
        assert all(file.is_relative_to(tmp_path / "threads") for file in tmp_path.rglob("*.json"))


def describe_persistent_threads():
    async def it_recovers_a_thread_by_id_in_a_new_agent(tmp_path: Path):
        journal = JsonJournal(tmp_path)
        await _chat(Scripted(AssistantMessage("Hi Ana")), journal).thread("u1")("I'm Ana")

        provider = Scripted(AssistantMessage("Your name is Ana"))
        thread = _chat(provider, journal).thread("u1")

        assert thread.history == (UserMessage("I'm Ana"), AssistantMessage("Hi Ana"))
        assert await thread("What's my name?") == "Your name is Ana"
        assert provider.requests[0].messages == (
            UserMessage("I'm Ana"),
            AssistantMessage("Hi Ana"),
            UserMessage("What's my name?"),
        )

    async def it_writes_the_history_as_rewritten_by_middleware(tmp_path: Path):
        @on_input
        def redact(message: UserMessage) -> UserMessage:
            return UserMessage(message.content.replace("1234", "****"))

        journal = JsonJournal(tmp_path)

        @agent(provider=Scripted(AssistantMessage("noted")), capabilities=[redact], journal=journal)
        async def chat(message: str) -> str: ...

        await chat.thread("u1")("card 1234")

        assert journal.read(("u1",)) == (UserMessage("card ****"), AssistantMessage("noted"))
        assert "1234" not in (tmp_path / "u1.json").read_text()

    async def it_does_not_write_one_off_calls(tmp_path: Path):
        journal = JsonJournal(tmp_path)

        assert await _chat(Scripted(AssistantMessage("hi")), journal)("hello") == "hi"
        assert list(tmp_path.rglob("*.json")) == []

    async def it_deletes_the_thread_on_dispose(tmp_path: Path):
        journal = JsonJournal(tmp_path)
        thread = _chat(Scripted(AssistantMessage("hi")), journal).thread("u1")
        await thread("hello")

        thread.dispose()

        assert journal.read(("u1",)) is None

    async def it_persists_the_threads_opened_by_delegation(tmp_path: Path):
        journal = JsonJournal(tmp_path)
        first = _lead(Scripted(AssistantMessage("bees dance")), _delegating("bees"), journal)
        await first.thread("u1")("go")

        research = Scripted(AssistantMessage("wasps sting"))
        second = _lead(research, _delegating("wasps"), journal)
        await second.thread("u1")("more")

        assert research.requests[0].messages == (
            UserMessage("Research bees"),
            AssistantMessage("bees dance"),
            UserMessage("Research wasps"),
        )
        assert journal.read(("u1", "researcher", "insects")) == (
            *research.requests[0].messages,
            AssistantMessage("wasps sting"),
        )


def describe_resume():
    async def it_continues_a_run_interrupted_between_steps(tmp_path: Path):
        journal = JsonJournal(tmp_path)
        interrupted = _chat(Interrupting(calls(("lookup", {}))), journal)

        with pytest.raises(ConnectionError):
            await interrupted.thread("u1")("answer?")

        provider = Scripted(AssistantMessage("It is 42"))
        thread = _chat(provider, journal).thread("u1")

        assert thread.history == (UserMessage("answer?"), calls(("lookup", {})), ToolResult("call-0", "lookup", "42"))
        assert await thread.resume() == "It is 42"
        assert provider.requests[0].messages == thread.history[:3]
        assert journal.read(("u1",)) == thread.history

    async def it_replays_the_last_answer_without_calling_the_provider(tmp_path: Path):
        journal = JsonJournal(tmp_path)
        await _chat(Scripted(AssistantMessage("Hi Ana")), journal).thread("u1")("I'm Ana")

        provider = Scripted()

        @agent(provider=provider, journal=journal)
        async def chat(message: str) -> AsyncIterator[str]: ...

        assert [text async for text in chat.thread("u1").resume()] == ["Hi Ana"]
        assert provider.requests == []

    async def it_refuses_an_empty_thread():
        with pytest.raises(RuntimeError, match="nothing to resume"):
            await _chat(Scripted(), None).thread("u1").resume()


class Interrupting(Provider):
    """Answers the first request, then fails as if the process had died."""

    def __init__(self, first: AssistantMessage) -> None:
        self.replies = [first]

    async def complete(self, request: Request) -> AsyncIterator[Delta | AssistantMessage]:
        if not self.replies:
            raise ConnectionError("interrupted")
        yield self.replies.pop()


def _chat(provider: Provider, journal: JsonJournal | None):
    @tool
    def lookup() -> str:
        return "42"

    @agent(provider=provider, capabilities=[lookup], journal=journal)
    async def chat(message: str) -> str: ...

    return chat


def _lead(research: Provider, provider: Provider, journal: JsonJournal):
    @agent(provider=research)
    async def researcher(topic: str) -> str:
        """Research {topic}"""
        ...

    @agent(provider=provider, capabilities=[researcher], journal=journal)
    async def lead(message: str) -> str: ...

    return lead


def _delegating(topic: str) -> Scripted:
    """A lead that sends `topic` to the researcher on the thread named "insects", then finishes."""
    return Scripted(calls(("researcher", {"topic": topic, "thread": "insects"})), AssistantMessage("done"))

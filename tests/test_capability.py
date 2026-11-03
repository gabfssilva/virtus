from collections.abc import AsyncIterator
from itertools import count

import pytest

from tests.scripted import Scripted, calls
from virtus import (
    AssistantMessage,
    Capabilities,
    Event,
    Message,
    Next,
    Request,
    ToolResult,
    UserMessage,
    agent,
    context,
    on_input,
    on_message,
    tool,
)


def describe_context():
    async def it_appends_to_the_system_prompt_on_every_call_in_list_order():
        calls_made = count(1)

        @context
        def first() -> str:
            return f"call {next(calls_made)}"

        @context
        async def second() -> str:
            return "second"

        @tool
        def noop() -> str:
            return "done"

        provider = Scripted(calls(("noop", {})), AssistantMessage("ok"))

        @agent(provider=provider, system="base", capabilities=[first, second, noop])
        async def bot(question: str) -> str: ...

        await bot("?")

        assert [r.system for r in provider.requests] == [
            "base\n\ncall 1\n\nsecond",
            "base\n\ncall 2\n\nsecond",
        ]


def describe_on_input():
    async def it_rewrites_the_input_once_and_keeps_the_rewrite_in_the_history():
        @on_input
        def redact(message: UserMessage) -> UserMessage:
            return UserMessage(message.content.replace("secret", "[redacted]"))

        @tool
        def noop() -> str:
            return "done"

        provider = Scripted(calls(("noop", {})), AssistantMessage("ok"))

        @agent(provider=provider, capabilities=[redact, noop])
        async def bot(text: str) -> str: ...

        await bot("my secret")

        assert [r.messages[0] for r in provider.requests] == [UserMessage("my [redacted]")] * 2

    async def it_stops_the_run_before_the_provider_when_it_raises():
        @on_input
        def reject(message: UserMessage) -> UserMessage:
            raise PermissionError("off topic")

        provider = Scripted(AssistantMessage("never"))

        @agent(provider=provider, capabilities=[reject])
        async def bot(text: str) -> str: ...

        with pytest.raises(PermissionError):
            await bot("hi")

        assert provider.requests == []


def describe_on_message():
    async def it_rewrites_replies_and_tool_results():
        @on_message
        def upper(message: Message) -> Message:
            match message:
                case AssistantMessage(content=content, tool_calls=tool_calls):
                    return AssistantMessage(content.upper(), tool_calls)
                case ToolResult() as result:
                    return ToolResult(result.call_id, result.name, result.content.upper(), result.is_error)
                case _:
                    return message

        @tool
        def fetch() -> str:
            return "data"

        provider = Scripted(calls(("fetch", {})), AssistantMessage("final"))

        @agent(provider=provider, capabilities=[upper, fetch])
        async def bot(text: str) -> str: ...

        assert await bot("go") == "FINAL"
        assert provider.requests[1].messages[-1] == ToolResult("call-0", "fetch", "DATA")

    async def it_prevents_tool_calls_from_running_when_it_raises():
        executed: list[str] = []

        @tool
        def delete_file(path: str) -> str:
            executed.append(path)
            return "deleted"

        @on_message
        def deny(message: Message) -> Message:
            if isinstance(message, AssistantMessage) and message.tool_calls:
                raise PermissionError("needs approval")
            return message

        provider = Scripted(calls(("delete_file", {"path": "/etc"})))

        @agent(provider=provider, capabilities=[deny, delete_file])
        async def bot(text: str) -> str: ...

        with pytest.raises(PermissionError):
            await bot("clean up")

        assert executed == []

    async def it_rejects_a_different_kind_of_message():
        @on_message
        def wrong(message: Message) -> Message:
            return UserMessage("not a reply")

        @agent(provider=Scripted(AssistantMessage("hi")), capabilities=[wrong])
        async def bot(text: str) -> str: ...

        with pytest.raises(TypeError, match="AssistantMessage"):
            await bot("hi")


def describe_raw_filters():
    async def it_can_answer_without_calling_the_provider():
        async def cached(request: Request, next: Next) -> AsyncIterator[Event]:
            yield AssistantMessage("from cache")

        provider = Scripted()

        @agent(provider=provider, capabilities=[cached])
        async def bot(text: str) -> str: ...

        assert await bot("hi") == "from cache"
        assert provider.requests == []


def describe_capabilities():
    async def it_collects_member_tools_and_context_bound_to_the_instance():
        class Notes(Capabilities):
            def __init__(self) -> None:
                self.saved: list[str] = []

            @tool
            def save(self, note: str) -> str:
                """Save a note."""
                self.saved.append(note)
                return "saved"

            @context
            def summary(self) -> str:
                return f"{len(self.saved)} notes"

        class TaggedNotes(Notes):
            @tool
            def tag(self, label: str) -> str:
                return label

        notes = TaggedNotes()
        provider = Scripted(calls(("tagged_notes__save", {"note": "buy milk"})), AssistantMessage("ok"))

        @agent(provider=provider, capabilities=[notes])
        async def bot(text: str) -> str: ...

        await bot("remember milk")

        assert notes.saved == ["buy milk"]
        assert [t.name for t in provider.requests[0].tools] == ["tagged_notes__save", "tagged_notes__tag"]
        assert [t.description for t in provider.requests[0].tools] == ["Save a note.", ""]
        assert [r.system for r in provider.requests] == ["0 notes", "1 notes"]

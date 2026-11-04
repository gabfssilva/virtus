import asyncio
import json
from collections.abc import AsyncIterator

import pytest

from tests.scripted import Scripted, calls
from virtus import (
    AssistantMessage,
    Capability,
    Delta,
    Provider,
    Request,
    ToolCall,
    ToolResult,
    UserMessage,
    agent,
    on_input,
    tool,
)


def describe_thread():
    async def it_sends_previous_turns_with_each_new_message():
        provider = Scripted(AssistantMessage("Hi Ana"), AssistantMessage("Your name is Ana"))

        @agent(provider=provider)
        async def chat(message: str) -> str: ...

        thread = chat.thread()
        await thread("I'm Ana")

        assert await thread("What's my name?") == "Your name is Ana"
        assert provider.requests[1].messages == (
            UserMessage("I'm Ana"),
            AssistantMessage("Hi Ana"),
            UserMessage("What's my name?"),
        )

    async def it_keeps_tool_calls_and_results_in_the_history():
        @tool
        def lookup() -> str:
            return "42"

        provider = Scripted(calls(("lookup", {})), AssistantMessage("It is 42"))

        @agent(provider=provider, capabilities=[lookup])
        async def chat(message: str) -> str: ...

        thread = chat.thread()
        await thread("answer?")

        assert thread.history == (
            UserMessage("answer?"),
            calls(("lookup", {})),
            ToolResult("call-0", "lookup", "42"),
            AssistantMessage("It is 42"),
        )

    async def it_records_streamed_turns():
        @agent(provider=Scripted(AssistantMessage("streamed")))
        async def chat(message: str) -> AsyncIterator[str]: ...

        thread = chat.thread()
        async for _ in thread("hi"):
            pass

        assert thread.history == (UserMessage("hi"), AssistantMessage("streamed"))

    async def it_keeps_rewrites_made_by_filters():
        @on_input
        def redact(message: UserMessage) -> UserMessage:
            return UserMessage(message.content.replace("1234", "****"))

        provider = Scripted(AssistantMessage("noted"), AssistantMessage("ok"))

        @agent(provider=provider, capabilities=[redact])
        async def chat(message: str) -> str: ...

        thread = chat.thread()
        await thread("card 1234")
        await thread("thanks")

        assert provider.requests[1].messages[0] == UserMessage("card ****")

    async def it_leaves_the_history_unchanged_when_a_turn_is_rejected():
        @on_input
        def reject_secrets(message: UserMessage) -> UserMessage:
            if "secret" in message.content:
                raise PermissionError("no secrets")
            return message

        @agent(provider=Scripted(AssistantMessage("hello")), capabilities=[reject_secrets])
        async def chat(message: str) -> str: ...

        thread = chat.thread()
        await thread("hi")

        with pytest.raises(PermissionError):
            await thread("my secret")

        assert thread.history == (UserMessage("hi"), AssistantMessage("hello"))

    async def it_runs_concurrent_calls_one_at_a_time():
        class Slow(Provider):
            def __init__(self) -> None:
                self.requests: list[Request] = []

            async def complete(self, request: Request) -> AsyncIterator[Delta | AssistantMessage]:
                self.requests.append(request)
                await asyncio.sleep(0.01)
                yield AssistantMessage(f"reply {len(self.requests)}")

        provider = Slow()

        @agent(provider=provider)
        async def chat(message: str) -> str: ...

        thread = chat.thread()
        await asyncio.gather(thread("first"), thread("second"))

        assert provider.requests[1].messages == (
            UserMessage("first"),
            AssistantMessage("reply 1"),
            UserMessage("second"),
        )


def describe_agent_thread():
    async def it_is_the_same_conversation_for_the_same_id():
        provider = Scripted(AssistantMessage("Hi Ana"), AssistantMessage("Your name is Ana"))

        @agent(provider=provider)
        async def chat(message: str) -> str: ...

        await chat.thread("user-42")("I'm Ana")
        await chat.thread("user-42")("What's my name?")

        assert chat.thread("user-42").id == "user-42"
        assert provider.requests[1].messages == (
            UserMessage("I'm Ana"),
            AssistantMessage("Hi Ana"),
            UserMessage("What's my name?"),
        )

    def it_creates_a_new_thread_when_no_id_is_given():
        @agent(provider=Scripted())
        async def chat(message: str) -> str: ...

        first, second = chat.thread(), chat.thread()

        assert first.id != second.id
        assert chat.thread(first.id).path == first.path

    async def it_runs_calls_on_one_conversation_one_at_a_time():
        @agent(provider=Scripted(AssistantMessage("one"), AssistantMessage("two")))
        async def chat(message: str) -> str: ...

        await asyncio.gather(chat.thread("user-42")("first"), chat.thread("user-42")("second"))

        assert chat.thread("user-42").history == (
            UserMessage("first"),
            AssistantMessage("one"),
            UserMessage("second"),
            AssistantMessage("two"),
        )

    async def it_starts_over_after_dispose():
        provider = Scripted(AssistantMessage("one"), AssistantMessage("two"))

        @agent(provider=provider)
        async def chat(message: str) -> str: ...

        thread = chat.thread("user-42")
        await thread("first")
        thread.dispose()

        assert thread.history == ()
        await chat.thread("user-42")("second")
        assert provider.requests[1].messages == (UserMessage("second"),)

    def it_rejects_a_parameter_named_thread():
        with pytest.raises(TypeError, match="reserved"):

            @agent(provider=Scripted())
            async def chat(thread: str) -> str: ...


def describe_delegation():
    async def it_makes_one_off_requests_without_a_thread_name():
        research = Scripted(AssistantMessage("bees dance"), AssistantMessage("wasps sting"))
        lead = _lead_for(
            _researcher_with(research),
            Scripted(_delegation({"topic": "bees"}), _delegation({"topic": "wasps"}), AssistantMessage("done")),
        )

        await lead("go")

        assert [r.messages for r in research.requests] == [
            (UserMessage("Research bees"),),
            (UserMessage("Research wasps"),),
        ]

    async def it_continues_the_conversation_with_the_same_name():
        research = Scripted(AssistantMessage("bees dance"), AssistantMessage("they sting less"))
        lead = _lead_for(
            _researcher_with(research),
            Scripted(
                _delegation({"topic": "bees", "thread": "insects"}),
                _delegation({"topic": "compared to wasps", "thread": "insects"}),
                AssistantMessage("done"),
            ),
        )

        await lead("go")

        assert research.requests[1].messages == (
            UserMessage("Research bees"),
            AssistantMessage("bees dance"),
            UserMessage("Research compared to wasps"),
        )

    async def it_keeps_different_names_apart():
        research = Scripted(AssistantMessage("a"), AssistantMessage("b"))
        lead = _lead_for(
            _researcher_with(research),
            Scripted(
                _delegation({"topic": "bees", "thread": "one"}, {"topic": "wasps", "thread": "two"}),
                AssistantMessage("done"),
            ),
        )

        await lead("go")

        assert {r.messages for r in research.requests} == {
            (UserMessage("Research bees"),),
            (UserMessage("Research wasps"),),
        }

    async def it_scopes_threads_to_the_calling_conversation():
        research = Scripted(AssistantMessage("for ana"), AssistantMessage("for bia"))
        researcher = _researcher_with(research)
        lead = _lead_for(
            researcher,
            Scripted(
                _delegation({"topic": "bees", "thread": "x"}),
                AssistantMessage("done"),
                _delegation({"topic": "wasps", "thread": "x"}),
                AssistantMessage("done"),
            ),
        )

        await lead.thread("ana")("go")
        await lead.thread("bia")("go")

        assert research.requests[1].messages == (UserMessage("Research wasps"),)
        assert lead.thread("ana").children == (("researcher", "x"),)
        assert lead.thread("ana").child(("researcher", "x"), researcher).id == "ana/researcher/x"

    async def it_tells_the_model_which_conversations_are_open():
        lead_provider = Scripted(_delegation({"topic": "bees", "thread": "insects"}), AssistantMessage("done"))
        lead = _lead_for(_researcher_with(Scripted(AssistantMessage("ok"))), lead_provider)

        await lead.thread()("go")

        assert lead_provider.requests[0].system == ""
        assert lead_provider.requests[1].system == "Open conversations with researcher: insects"

    async def it_disposes_child_threads_with_the_parent():
        researcher = _researcher_with(Scripted(AssistantMessage("ok")))
        lead = _lead_for(
            researcher,
            Scripted(_delegation({"topic": "bees", "thread": "insects"}), AssistantMessage("done")),
        )
        parent = lead.thread()
        await parent("go")
        child = parent.child(("researcher", "insects"), researcher)
        assert child.history != ()

        parent.dispose()

        assert parent.children == ()
        assert child.history == ()

    async def it_shares_a_thread_passed_as_a_capability():
        research = Scripted(AssistantMessage("first"), AssistantMessage("second"))
        shared = _researcher_with(research).thread("knowledge-base")
        lead = _lead_for(
            shared,
            Scripted(
                _delegation({"topic": "bees"}),
                AssistantMessage("done"),
                _delegation({"topic": "wasps"}),
                AssistantMessage("done"),
            ),
        )

        await lead.thread("ana")("go")
        await lead.thread("bia")("go")

        assert shared.history == (
            UserMessage("Research bees"),
            AssistantMessage("first"),
            UserMessage("Research wasps"),
            AssistantMessage("second"),
        )


def _lead_for(researcher: Capability, provider: Scripted):
    @agent(provider=provider, capabilities=[researcher])
    async def lead(message: str) -> str: ...

    return lead


def _delegation(*requested: dict[str, object]) -> AssistantMessage:
    return AssistantMessage(
        tool_calls=tuple(ToolCall(f"call-{i}", "researcher", json.dumps(args)) for i, args in enumerate(requested))
    )


def _researcher_with(provider: Scripted):
    @agent(provider=provider)
    async def researcher(topic: str) -> str:
        """Research {topic}"""
        ...

    return researcher

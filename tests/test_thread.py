import asyncio
import json
from collections.abc import AsyncIterator

import pytest

from tests.scripted import Scripted, calls
from virtus import (
    AssistantMessage,
    Capability,
    Delta,
    Image,
    Provider,
    Request,
    ToolResult,
    Usage,
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

    async def it_sums_the_usage_of_its_provider_calls():
        provider = Scripted(AssistantMessage("a", usage=Usage(10, 2)), AssistantMessage("b", usage=Usage(30, 4)))

        @agent(provider=provider)
        async def chat(message: str) -> str: ...

        thread = chat.thread("ana")
        await thread("one")
        await thread("two")

        assert thread.usage == Usage(40, 6)

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



def describe_dispatch():
    async def it_makes_one_off_requests_without_a_conversation_name():
        research = Scripted(AssistantMessage("bees dance"), AssistantMessage("wasps sting"))
        lead = _lead_for(
            _researcher_with(research),
            Scripted(_dispatch({"topic": "bees"}, "researcher"), _dispatch({"topic": "wasps"}, "researcher"), _done()),
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
                _dispatch({"topic": "bees"}, "researcher:insects"),
                _dispatch({"topic": "compared to wasps"}, "researcher:insects"),
                _done(),
            ),
        )

        await lead("go")

        assert research.requests[1].messages == (
            UserMessage("Research bees"),
            AssistantMessage("bees dance"),
            UserMessage("Research compared to wasps"),
        )

    async def it_delivers_one_message_to_several_addresses_and_keys_the_replies():
        research = Scripted(AssistantMessage("a"), AssistantMessage("b"))
        lead_provider = Scripted(_dispatch({"topic": "insects"}, "researcher:one", "researcher:two"), _done())
        lead = _lead_for(_researcher_with(research), lead_provider)

        await lead.thread("ana")("go")

        assert [r.messages for r in research.requests] == [(UserMessage("Research insects"),)] * 2
        assert lead.thread("ana").children == (("researcher", "one"), ("researcher", "two"))
        result = lead_provider.requests[1].messages[-1]
        assert isinstance(result, ToolResult)
        assert json.loads(result.content) == {"researcher:one": "a", "researcher:two": "b"}

    async def it_scopes_conversations_to_the_calling_thread():
        research = Scripted(AssistantMessage("for ana"), AssistantMessage("for bia"))
        researcher = _researcher_with(research)
        lead = _lead_for(
            researcher,
            Scripted(
                _dispatch({"topic": "bees"}, "researcher:x"),
                _done(),
                _dispatch({"topic": "wasps"}, "researcher:x"),
                _done(),
            ),
        )

        await lead.thread("ana")("go")
        await lead.thread("bia")("go")

        assert research.requests[1].messages == (UserMessage("Research wasps"),)
        assert lead.thread("ana").children == (("researcher", "x"),)
        child = lead.thread("ana").child(("researcher", "x"), researcher)
        assert (child.id, child.path) == ("x", ("lead", "ana", "researcher", "x"))

    async def it_lists_the_agents_and_their_open_conversations_in_the_system_prompt():
        lead_provider = Scripted(_dispatch({"topic": "bees"}, "researcher:insects"), _done())
        lead = _lead_for(_researcher_with(Scripted(AssistantMessage("ok"))), lead_provider)

        await lead.thread()("go")

        listing = "Agents you can `dispatch` to:\n- researcher: Research {topic}"
        assert lead_provider.requests[0].system == listing
        assert lead_provider.requests[1].system == f"{listing} Open conversations: insects"

    async def it_is_one_tool_however_many_agents_there_are():
        lead_provider = Scripted(_dispatch({"topic": "hello"}, "researcher", "writer"), _done())

        @agent(provider=Scripted(AssistantMessage("written")))
        async def writer(topic: str) -> str:
            """Write about {topic}"""
            ...

        researcher = _researcher_with(Scripted(AssistantMessage("ok")))

        @agent(provider=lead_provider, system="Lead.", capabilities=[researcher, writer])
        async def lead(message: str) -> str: ...

        await lead("go")

        assert [t.name for t in lead_provider.requests[0].tools] == ["dispatch"]
        assert lead_provider.requests[0].system == (
            "Lead.\n\nAgents you can `dispatch` to:\n"
            "- researcher: Research {topic}\n"
            "- writer: Write about {topic}"
        )
        [dispatch] = lead_provider.requests[0].tools
        topic = {"properties": {"topic": {"title": "Topic", "type": "string"}}, "required": ["topic"], "type": "object"}
        assert dispatch.schema["$defs"] == {
            "researcher": {**topic, "title": "researcher"},
            "writer": {**topic, "title": "writer"},
        }
        result = lead_provider.requests[1].messages[-1]
        assert isinstance(result, ToolResult)
        assert json.loads(result.content) == {"researcher": "ok", "writer": "written"}

    async def it_rejects_arguments_the_template_does_not_take():
        lead_provider = Scripted(_dispatch({"subject": "bees"}, "researcher"), _done())
        lead = _lead_for(_researcher_with(Scripted()), lead_provider)

        await lead("go")

        result = lead_provider.requests[1].messages[-1]
        assert isinstance(result, ToolResult)
        assert result.is_error
        assert "topic" in result.content

    async def it_reports_a_failing_address_without_losing_the_others():
        research = Scripted(AssistantMessage("a"))  # the second address finds no reply and fails
        lead_provider = Scripted(_dispatch({"topic": "insects"}, "researcher:one", "researcher:two"), _done())
        lead = _lead_for(_researcher_with(research), lead_provider)

        await lead("go")

        result = lead_provider.requests[1].messages[-1]
        assert isinstance(result, ToolResult)
        assert not result.is_error
        replies = json.loads(result.content)
        assert replies["researcher:one"] == "a"
        assert replies["researcher:two"]["error"].startswith("IndexError")

    async def it_refuses_agents_with_media_parameters():
        @agent(provider=Scripted())
        async def describer(picture: Image, question: str = "what is it?") -> str:
            """{question} {picture}"""
            ...

        with pytest.raises(TypeError, match="describer: can't be dispatched to; the model can't fill picture"):

            @agent(provider=Scripted(), capabilities=[describer])
            async def lead(message: str) -> str: ...

    async def it_rejects_addresses_of_unknown_agents():
        lead_provider = Scripted(_dispatch({"topic": "hello"}, "nobody"), _done())
        lead = _lead_for(_researcher_with(Scripted()), lead_provider)

        await lead("go")

        result = lead_provider.requests[1].messages[-1]
        assert isinstance(result, ToolResult)
        assert result.is_error
        assert "unknown agents: nobody; known: researcher" in result.content

    async def it_disposes_child_threads_with_the_parent():
        researcher = _researcher_with(Scripted(AssistantMessage("ok")))
        lead = _lead_for(researcher, Scripted(_dispatch({"topic": "bees"}, "researcher:insects"), _done()))
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
                _dispatch({"topic": "bees"}, "researcher"),
                _done(),
                _dispatch({"topic": "wasps"}, "researcher"),
                _done(),
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


def _dispatch(arguments: dict[str, object], *to: str) -> AssistantMessage:
    return calls(("dispatch", {"to": list(to), "arguments": arguments}))


def _done() -> AssistantMessage:
    return AssistantMessage("done")


def _researcher_with(provider: Scripted):
    @agent(provider=provider)
    async def researcher(topic: str) -> str:
        """Research {topic}"""
        ...

    return researcher

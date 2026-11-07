import json

from tests.scripted import Scripted, calls
from virtus import AssistantMessage, Capabilities, ToolResult, agent, context, lazy, tool
from virtus.lazy import HEADER


@tool
def clock() -> str:
    """The time now.

    Always UTC.
    """
    return "12:00"


@tool
def weather(city: str) -> str:
    """The weather in a city."""
    return f"sunny in {city}"


def describe_lazy():
    async def it_lists_the_tools_and_offers_them_once_loaded():
        provider = Scripted(
            calls(("load_tools", {"names": ["weather"]})),
            calls(("weather", {"city": "Lisbon"})),
            AssistantMessage("Sunny."),
        )

        @agent(provider=provider, system="Be brief.", capabilities=[lazy(clock, weather)])
        async def assistant(message: str) -> str: ...

        assert await assistant.thread("u1")("Weather in Lisbon?") == "Sunny."

        first, second, third = provider.requests
        assert [t.name for t in first.tools] == ["load_tools"]
        assert first.system == f"Be brief.\n\n{HEADER}\n- clock: The time now.\n- weather: The weather in a city."
        names = {"items": {"type": "string"}, "title": "Names", "type": "array"}
        assert first.tools[0].schema["properties"] == {"names": names}
        loaded = ToolResult("call-0", "load_tools", "Loaded weather; available from your next step on.")
        assert second.messages[-1] == loaded
        assert [t.name for t in second.tools] == ["load_tools", "weather"]
        assert second.system == f"Be brief.\n\n{HEADER}\n- clock: The time now."
        assert third.messages[-1] == ToolResult("call-0", "weather", "sunny in Lisbon")

    async def it_keeps_what_was_loaded_for_the_rest_of_the_conversation_only():
        replies = calls(("load_tools", {"names": ["clock"]})), AssistantMessage("ok"), AssistantMessage("ok")
        provider = Scripted(*replies, AssistantMessage("ok"))

        @agent(provider=provider, capabilities=[lazy(clock)])
        async def assistant(message: str) -> str: ...

        await assistant.thread("u1")("load the clock")
        await assistant.thread("u1")("again")
        await assistant.thread("u2")("fresh")

        assert [[t.name for t in r.tools] for r in provider.requests[1:]] == [
            ["load_tools", "clock"],
            ["load_tools", "clock"],
            ["load_tools"],
        ]

    async def it_rejects_names_outside_the_catalog():
        provider = Scripted(calls(("load_tools", {"names": ["clock", "oracle"]})), AssistantMessage("ok"))

        @agent(provider=provider, capabilities=[lazy(clock, weather)])
        async def assistant(message: str) -> str: ...

        await assistant("go")

        result = provider.requests[1].messages[-1]
        assert isinstance(result, ToolResult)
        assert result.is_error
        assert result.content == "ValueError: unknown tools: oracle; known: clock, weather"
        assert [t.name for t in provider.requests[1].tools] == ["load_tools"]

    async def it_takes_any_capability_and_keeps_what_it_adds_to_the_prompt():
        class Notes(Capabilities):
            @context
            def count(self) -> str:
                return "You have 2 notes."

            @tool
            def add(self, text: str) -> str:
                """Add a note."""
                return f"added {text!r}"

        load, add = calls(("load_tools", {"names": ["notes__add"]})), calls(("notes__add", {"text": "milk"}))
        provider = Scripted(load, add, AssistantMessage("ok"))

        @agent(provider=provider, capabilities=[clock, lazy(Notes())])
        async def assistant(message: str) -> str: ...

        await assistant("go")

        first, second, third = provider.requests
        assert [t.name for t in first.tools] == ["clock", "load_tools"]
        assert first.system == f"You have 2 notes.\n\n{HEADER}\n- notes__add: Add a note."
        assert [t.name for t in second.tools] == ["clock", "load_tools", "notes__add"]
        assert second.system == "You have 2 notes."
        assert json.loads(json.dumps(third.messages[-1].content)) == "added 'milk'"

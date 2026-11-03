import asyncio
from typing import Annotated

import pytest
from pydantic import Field

from liteagent import AssistantMessage, ToolCall, ToolResult, agent, tool
from tests.scripted import Scripted, calls


def describe_tool():
    def it_describes_parameters_from_the_signature():
        @tool
        def search(query: Annotated[str, Field(description="What to look for")], limit: int = 5) -> list[str]:
            """Search the catalog."""
            return []

        assert search.name == "search"
        assert search.description == "Search the catalog."
        assert search.schema["required"] == ["query"]
        assert search.schema["properties"] == {
            "query": {"type": "string", "title": "Query", "description": "What to look for"},
            "limit": {"type": "integer", "title": "Limit", "default": 5},
        }

    def it_requires_annotated_parameters():
        def untyped(value):
            return value

        with pytest.raises(TypeError, match="value"):
            tool(untyped)

    def describe_execute():
        async def it_runs_sync_and_async_functions():
            @tool
            def double(n: int) -> int:
                return n * 2

            @tool
            async def shout(text: str) -> str:
                return text.upper()

            assert await double.execute(ToolCall("1", "double", '{"n": 4}')) == ToolResult("1", "double", "8")
            assert await shout.execute(ToolCall("2", "shout", '{"text": "hi"}')) == ToolResult("2", "shout", "HI")

        async def it_reports_failures_as_error_results():
            @tool
            def explode() -> str:
                raise ValueError("boom")

            result = await explode.execute(ToolCall("1", "explode", "{}"))

            assert result == ToolResult("1", "explode", "ValueError: boom", is_error=True)

        async def it_reports_invalid_arguments_as_error_results():
            @tool
            def double(n: int) -> int:
                return n * 2

            result = await double.execute(ToolCall("1", "double", '{"n": "four"}'))

            assert result.is_error
            assert result.content.startswith("ValidationError")


def describe_tool_calls_in_an_agent():
    async def it_sends_the_results_back_to_the_model():
        @tool
        def weather(city: str) -> str:
            return f"Sunny in {city}"

        provider = Scripted(calls(("weather", {"city": "Paris"})), AssistantMessage("It's sunny"))

        @agent(provider=provider, capabilities=[weather])
        async def bot(question: str) -> str: ...

        assert await bot("Paris?") == "It's sunny"
        assert provider.requests[1].messages[-1] == ToolResult("call-0", "weather", "Sunny in Paris")

    async def it_runs_the_calls_of_a_step_concurrently():
        barrier = asyncio.Barrier(2)

        @tool
        async def wait(name: str) -> str:
            async with asyncio.timeout(1):
                await barrier.wait()
            return name

        provider = Scripted(calls(("wait", {"name": "a"}), ("wait", {"name": "b"})), AssistantMessage("both"))

        @agent(provider=provider, capabilities=[wait])
        async def bot(question: str) -> str: ...

        assert await bot("go") == "both"
        assert [m.content for m in provider.requests[1].messages[-2:]] == ["a", "b"]

    async def it_answers_unknown_tools_with_an_error():
        provider = Scripted(calls(("missing", {})), AssistantMessage("sorry"))

        @agent(provider=provider)
        async def bot(question: str) -> str: ...

        await bot("?")

        assert provider.requests[1].messages[-1] == ToolResult("call-0", "missing", "Unknown tool: missing", True)

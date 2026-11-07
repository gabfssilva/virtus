import json
import sys
from pathlib import Path

import pytest

from tests.scripted import Scripted, calls
from virtus import AssistantMessage, Http, Stdio, Tool, ToolCall, agent, mcp, secrets
from virtus.servers import _spec

SERVER = f"{sys.executable} {Path(__file__).with_name('mcp_server.py')}"


async def _run(tool: Tool, **arguments: object):
    return await tool.execute(ToolCall("c", tool.name, json.dumps(arguments)))


def describe_mcp():
    async def it_offers_the_servers_tools_with_their_schemas_under_the_prefix():
        provider = Scripted(calls(("calc__add", {"a": 2, "b": 3})), AssistantMessage("5"))

        async with mcp(calc=SERVER) as tools:

            @agent(provider=provider, capabilities=[tools])
            async def chat(message: str) -> str: ...

            assert await chat("add") == "5"

        request = provider.requests[0]
        names = ["calc__add", "calc__shout", "calc__whoami", "calc__fail", "calc__dot"]
        assert [t.name for t in request.tools] == names
        add = request.tools[0]
        assert add.description == "Add two integers."
        assert add.schema == {
            "properties": {"a": {"title": "A", "type": "integer"}, "b": {"title": "B", "type": "integer"}},
            "required": ["a", "b"],
            "title": "addArguments",
            "type": "object",
        }
        assert provider.requests[1].messages[-1].content == "5"

    async def it_returns_the_servers_errors_as_error_results():
        async with mcp(calc=SERVER) as tools:
            [fail] = [t for t in tools.tools if t.name == "calc__fail"]
            [add] = [t for t in tools.tools if t.name == "calc__add"]

            failed = await _run(fail)
            invalid = await _run(add, a="x")

        assert (failed.is_error, failed.content) == (True, "Error executing tool fail")
        assert invalid.is_error and invalid.content.startswith("Error executing tool add: 2 validation errors")

    async def it_attaches_images_the_server_returns():
        async with mcp(calc=SERVER) as tools:
            [dot] = [t for t in tools.tools if t.name == "calc__dot"]

            result = await _run(dot)

        assert result.content == "[image]"
        [image] = result.images
        assert (image.media_type, image.data[:8]) == ("image/png", b"\x89PNG\r\n\x1a\n")

    async def it_passes_the_environment_to_a_stdio_server():
        async with mcp(a=Stdio(SERVER, env={"TESTER": "ana"}), b=SERVER) as tools:
            by_name = {t.name: t for t in tools.tools}

            assert (await _run(by_name["a__whoami"])).content == "ana"
            assert (await _run(by_name["b__whoami"])).content == "nobody"

    async def it_lists_every_page_of_tools(monkeypatch: pytest.MonkeyPatch):
        from mcp import ClientSession, types

        asked: list[str | None] = []

        async def list_tools(self: ClientSession, *, params: types.PaginatedRequestParams | None = None):
            asked.append(params.cursor if params else None)
            if params is None:
                return types.ListToolsResult(tools=[types.Tool(name="first", inputSchema={})], nextCursor="p2")
            return types.ListToolsResult(tools=[types.Tool(name="second", inputSchema={})])

        monkeypatch.setattr(ClientSession, "list_tools", list_tools)

        async with mcp(calc=SERVER) as tools:
            names = [t.name for t in tools.tools]

        assert (asked, names) == ([None, "p2"], ["calc__first", "calc__second"])

    def it_reads_a_string_as_a_command_or_a_url():
        assert _spec("npx -y server .") == Stdio("npx -y server .")
        assert _spec("https://host/mcp") == Http("https://host/mcp")
        http = Http("https://host/mcp", headers={"Authorization": "x"})
        assert _spec(http) == http

    async def it_takes_secrets_like_any_other_tool():
        provider = Scripted(calls(("calc__shout", {"text": "{{secrets.word}}"})), AssistantMessage("ok"))

        async with mcp(calc=SERVER) as tools:

            @agent(provider=provider, capabilities=[tools, secrets(word=("ana", "a name"))])
            async def chat(message: str) -> str: ...

            await chat("shout")

        assert provider.requests[1].messages[-1].content == "ANA"

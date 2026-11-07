import json
from pathlib import Path

from tests.scripted import Scripted, calls
from virtus import AssistantMessage, Tool, ToolCall, agent, workspace


async def _tools(root: Path) -> dict[str, Tool]:
    """The workspace tools as the provider sees them."""
    provider = Scripted(AssistantMessage("ok"))

    @agent(provider=provider, capabilities=[workspace(root)])
    async def chat(message: str) -> str: ...

    await chat("go")
    return {tool.name: tool for tool in provider.requests[0].tools}


async def _run(tool: Tool, **arguments: object) -> str:
    result = await tool.execute(ToolCall("c", tool.name, json.dumps(arguments)))
    return result.content


def describe_workspace():
    async def it_offers_the_five_tools_and_says_so(tmp_path: Path):
        provider = Scripted(AssistantMessage("ok"))

        @agent(provider=provider, system="Be brief.", capabilities=[workspace(tmp_path, name="repo")])
        async def chat(message: str) -> str: ...

        await chat("go")

        request = provider.requests[0]
        assert [t.name for t in request.tools] == ["repo_read", "repo_write", "repo_edit", "repo_list", "repo_search"]
        assert request.system == (
            "Be brief.\n\n"
            "Your repo is a folder of files you reach with the repo_* tools; paths are relative to its root."
        )

    async def it_writes_reads_and_edits_files(tmp_path: Path):
        tools = await _tools(tmp_path)

        written = await _run(tools["workspace_write"], path="notes/a.txt", content="hello world")
        assert written == "wrote 11 chars to notes/a.txt"
        assert await _run(tools["workspace_read"], path="notes/a.txt") == "hello world"
        assert await _run(tools["workspace_edit"], path="notes/a.txt", old="world", new="there") == "edited notes/a.txt"
        assert (tmp_path / "notes" / "a.txt").read_text() == "hello there"

    async def it_requires_the_edited_text_to_occur_once(tmp_path: Path):
        (tmp_path / "f.txt").write_text("a a")
        tools = await _tools(tmp_path)

        assert "occurs 2 times" in await _run(tools["workspace_edit"], path="f.txt", old="a", new="b")
        assert "not found" in await _run(tools["workspace_edit"], path="f.txt", old="z", new="b")
        assert (tmp_path / "f.txt").read_text() == "a a"

    async def it_lists_folders_first_and_skips_hidden_entries(tmp_path: Path):
        (tmp_path / "src").mkdir()
        (tmp_path / ".git").mkdir()
        (tmp_path / "b.txt").write_text("")
        (tmp_path / "a.txt").write_text("")
        tools = await _tools(tmp_path)

        assert await _run(tools["workspace_list"]) == "src/\na.txt\nb.txt"
        assert await _run(tools["workspace_list"], path="src") == "(empty)"

    async def it_searches_text_files_recursively(tmp_path: Path):
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "a.py").write_text("def alpha():\n    return 1\n")
        (tmp_path / "b.txt").write_text("alpha and beta\n")
        (tmp_path / "blob.bin").write_bytes(b"\xff\xfe alpha")
        tools = await _tools(tmp_path)

        hits = await _run(tools["workspace_search"], pattern=r"alpha\b")
        assert hits == "b.txt:1: alpha and beta\nsrc/a.py:1: def alpha():"
        assert await _run(tools["workspace_search"], pattern="gamma") == "No matches."

    async def it_refuses_paths_outside_the_root(tmp_path: Path):
        tools = await _tools(tmp_path / "inner")

        result = await _run(tools["workspace_read"], path="../secret.txt")

        assert result == "PermissionError: '../secret.txt' is outside the workspace"

    async def it_truncates_long_files(tmp_path: Path):
        (tmp_path / "big.txt").write_text("x" * 30)
        provider = Scripted(calls(("workspace_read", {"path": "big.txt"})), AssistantMessage("ok"))

        @agent(provider=provider, capabilities=[workspace(tmp_path, max_chars=10)])
        async def chat(message: str) -> str: ...

        await chat("go")

        assert provider.requests[1].messages[-1].content == "x" * 10 + "\n… [truncated; 30 chars in total]"

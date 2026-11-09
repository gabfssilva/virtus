"""`mcp(name=server, ...)` connects to MCP servers for the block and offers their tools under `name__*`.

Here the server is the reference filesystem server, started with `npx` over stdio; a URL would be
reached over streamable HTTP, and `Stdio(...)` / `Http(...)` add environment variables and headers.
"""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from virtus import agent, mcp, observe
from virtus.providers import ChatCompletions

# Chat Completions only takes function tools from gpt-6 with reasoning off; the Responses API has no such limit.
model = ChatCompletions(reasoning_effort="none")


async def main() -> None:
    with TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "notes.md").write_text("# Notes\n\n- virtus: agents as functions\n- mcp: tools over a protocol\n")

        async with mcp(fs=f"npx -y @modelcontextprotocol/server-filesystem {root}") as tools:

            @agent(
                provider=model,
                system="Use the fs tools for anything about files. Answer in one line.",
                capabilities=[tools, observe()],
            )
            async def clerk(request: str) -> str: ...

            print(await clerk(f"Which files are in {root}, and what does notes.md say about mcp?"))


asyncio.run(main())

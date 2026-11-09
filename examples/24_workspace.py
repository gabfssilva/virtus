"""`workspace(path)` gives the agent a folder: read, write, edit, list and search, nothing outside it."""

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

from virtus import agent, observe, workspace
from virtus.providers import ChatCompletions

# Chat Completions only takes function tools from gpt-6 with reasoning off; the Responses API has no such limit.
model = ChatCompletions(reasoning_effort="none")


async def main() -> None:
    with TemporaryDirectory() as folder:
        root = Path(folder)
        (root / "notes").mkdir()
        (root / "notes" / "todo.md").write_text("# Todo\n\n- [ ] buy milk\n- [ ] call Ana\n- [x] pay rent\n")
        (root / "README.md").write_text("# Errands\n\nThe list lives in notes/todo.md.\n")

        @agent(
            provider=model,
            system="You manage the files in your workspace. Make the requested change, then confirm it in one line.",
            capabilities=[workspace(root), observe()],
        )
        async def clerk(request: str) -> str: ...

        print(await clerk("Mark 'call Ana' as done in the todo list and add 'water the plants' at the end."))
        print((root / "notes" / "todo.md").read_text())


asyncio.run(main())

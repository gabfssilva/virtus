"""Agents used as capabilities are reached through one `dispatch` tool, with the agent's own arguments:
`researcher` for a one-off request, `researcher:subject` to keep talking to it, several addresses to fan out."""

import asyncio

from virtus import agent, observe
from virtus.providers import OpenAI

# Chat Completions only takes function tools from gpt-6 with reasoning off; the Responses API has no such limit.
model = OpenAI(reasoning_effort="none")


@agent(provider=model, system="You are a researcher. Answer in two or three sentences.")
async def researcher(topic: str) -> str:
    """Research {topic}."""
    ...


@agent(
    provider=model,
    system="Dispatch research to the researcher. Keep one named conversation per subject so follow-ups have context.",
    capabilities=[researcher, observe()],
)
async def lead(request: str) -> str: ...


async def main() -> None:
    thread = lead.thread("project")
    print(await thread("Find out how honeybees communicate."))
    print(await thread("Ask the same researcher how that compares to ants."))

    for name, title in thread.children:
        child = thread.child((name, title), researcher)
        print(f"\n{name} thread {title!r}: {len(child.history)} messages")


asyncio.run(main())

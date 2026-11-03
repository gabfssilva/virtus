"""An agent used as a capability becomes a tool. The caller may name a thread to keep talking to it."""

import asyncio

from virtus import agent
from virtus.providers import OpenAI


@agent(provider=OpenAI(), system="You are a researcher. Answer in two or three sentences.")
async def researcher(topic: str) -> str:
    """Research {topic}."""
    ...


@agent(
    provider=OpenAI(),
    system="Delegate research to the researcher tool. Keep one named thread per subject so follow-ups have context.",
    capabilities=[researcher],
)
async def lead(request: str) -> str: ...


async def main() -> None:
    thread = lead.thread("project")
    print(await thread("Find out how honeybees communicate."))
    print(await thread("Ask the same researcher how that compares to ants."))

    for (name, title), child in thread.children.items():
        print(f"\n{name} thread {title!r}: {len(child.history)} messages")


asyncio.run(main())

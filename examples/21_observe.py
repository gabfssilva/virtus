"""`observe()` draws each run as a tree on stderr while it happens: provider calls, tool calls, the streamed answer
and timings. Add it to any agent; nothing else changes."""

import asyncio

from virtus import agent, observe, tool
from virtus.providers import OpenAI
from virtus.tools import websearch


@tool
def population(city: str) -> int:
    """Population of a city, latest census."""
    return {"lisbon": 545_796, "porto": 231_962}.get(city.lower(), 0)


@agent(
    provider=OpenAI("gpt-6-luna", reasoning_effort="none"),
    system="Answer in one sentence. Use the tools.",
    capabilities=[population, websearch, observe()],
)
async def assistant(question: str) -> str: ...


async def main() -> None:
    answer = await assistant("How many people live in Lisbon, and what is its current mayor's name?")
    print(answer)


asyncio.run(main())

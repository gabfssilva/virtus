"""A raw middleware wraps each step: it can time it, stop the run, or swap the provider for one call."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import replace
from time import perf_counter

from virtus import Event, Middleware, Next, Provider, Request, agent, tool
from virtus.providers import OpenAI


async def timing(request: Request, next: Next) -> AsyncIterator[Event]:
    started = perf_counter()
    async for event in next(request):
        yield event
    print(f"  [step {request.iteration} took {perf_counter() - started:.2f}s]")


def max_steps(limit: int) -> Middleware:
    async def enforce(request: Request, next: Next) -> AsyncIterator[Event]:
        if request.iteration >= limit:
            raise RuntimeError(f"gave up after {limit} steps")
        async for event in next(request):
            yield event

    return enforce


def escalate_to(provider: Provider, after: int) -> Middleware:
    """Use a stronger model once the cheap one has spent `after` rounds calling tools."""

    async def swap(request: Request, next: Next) -> AsyncIterator[Event]:
        step = request if request.iteration < after else replace(request, provider=provider)
        async for event in next(step):
            yield event

    return swap


@tool
def search(query: str) -> str:
    """Search the company wiki by page title."""
    if "fiscal" in query.lower():
        return "Fiscal year 2025 results: revenue 12.4M EUR, up 8% from 2024."
    return "No results. Closest titles: 'Fiscal year 2025 results', 'Fiscal year 2024 results'."


@agent(
    provider=OpenAI(),
    system="Answer from the wiki. If a search misses, follow its hint and search again instead of asking the user.",
    capabilities=[timing, max_steps(4), escalate_to(OpenAI("gpt-4.1"), after=2), search],
)
async def analyst(question: str) -> str: ...


async def main() -> None:
    print(await analyst("What was the revenue in 2025?"))


asyncio.run(main())

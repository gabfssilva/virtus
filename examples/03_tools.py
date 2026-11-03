"""Functions become tools; the model calls them until it can answer. Sync tools run in a thread."""

import asyncio
from datetime import UTC, datetime

from virtus import agent, tool
from virtus.providers import OpenAI


@tool
def now() -> str:
    """The current date and time, in UTC."""
    return datetime.now(UTC).isoformat(timespec="seconds")


@tool
async def days_until(date: str) -> int:
    """Days from today until an ISO date such as 2026-12-25."""
    return (datetime.fromisoformat(date).date() - datetime.now(UTC).date()).days


@agent(provider=OpenAI(), capabilities=[now, days_until])
async def assistant(question: str) -> str: ...


async def main() -> None:
    print(await assistant("How many days are left until the next Christmas?"))


asyncio.run(main())

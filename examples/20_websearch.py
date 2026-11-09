"""`websearch` from `virtus.tools` lets the agent look things up on the web: pages by default, or news, images,
videos and books through `kind`, within a `period`. Needs the `websearch` extra; no key."""

import asyncio

from virtus import agent, observe
from virtus.providers import ChatCompletions
from virtus.tools import websearch

# Chat Completions only takes function tools from gpt-6 with reasoning off; the Responses API has no such limit.
model = ChatCompletions("gpt-6.1-terra", reasoning_effort="none")


@agent(provider=model, system="Search before answering; cite your source.", capabilities=[observe(), websearch()])
async def assistant(question: str) -> str: ...


async def main() -> None:
    print(await assistant("What does asyncio.TaskGroup do, in one sentence?"))
    print(await assistant("One piece of Python news from the last week, with its date."))


asyncio.run(main())

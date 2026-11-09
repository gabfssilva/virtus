"""`lazy(...)` offers tools on demand: the model sees the tool names and finds what it needs with `tool_search`,
so a big toolbox doesn't travel with every request. Anything that goes in `capabilities=` goes in."""

import asyncio

from virtus import agent, lazy, observe
from virtus.providers import Responses
from virtus.tools import arxiv, calculator, http, openalex, wikipedia

# The Responses API loads the tools a search finds natively; Chat Completions would add them to the request.
model = Responses()


@agent(
    provider=model,
    system="Load only the tools the question needs, then answer in one sentence.",
    capabilities=[lazy(wikipedia(), calculator(), http(), arxiv(), openalex()), observe()],
)
async def assistant(question: str) -> str: ...


async def main() -> None:
    question = "How many years passed between the completion of the Eiffel Tower and the first Moon landing?"
    print(await assistant(question))


asyncio.run(main())

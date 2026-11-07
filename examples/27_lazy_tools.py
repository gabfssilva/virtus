"""`lazy(...)` offers tools on demand: the model sees a one-line catalog and loads what it needs with
`load_tools`, so a big toolbox doesn't travel with every request. Anything that goes in `capabilities=` goes in."""

import asyncio

from virtus import agent, lazy, observe
from virtus.providers import OpenAI
from virtus.tools import arxiv, calculator, http, openalex, wikipedia

# Chat Completions only takes function tools from gpt-6 with reasoning off; the Responses API has no such limit.
model = OpenAI(reasoning_effort="none")


@agent(
    provider=model,
    system="Load only the tools the question needs, then answer in one sentence.",
    capabilities=[lazy(wikipedia, calculator, http, arxiv, openalex), observe()],
)
async def assistant(question: str) -> str: ...


async def main() -> None:
    question = "How many years passed between the completion of the Eiffel Tower and the first Moon landing?"
    print(await assistant(question))


asyncio.run(main())

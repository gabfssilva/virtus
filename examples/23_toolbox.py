"""Three ready-made tools, all stdlib and keyless: `wikipedia` for facts, `calculator` for exact arithmetic,
`http` for any endpoint, with JSON reindented and HTML reduced to text."""

import asyncio

from virtus import agent, observe
from virtus.providers import OpenAI
from virtus.tools import calculator, http, wikipedia

# Chat Completions only takes function tools from gpt-6 with reasoning off; the Responses API has no such limit.
model = OpenAI(reasoning_effort="none")


@agent(
    provider=model,
    system="Look facts up before answering and do every calculation with the calculator. Answer in one paragraph.",
    capabilities=[wikipedia, calculator, http, observe()],
)
async def assistant(question: str) -> str: ...


async def main() -> None:
    print(
        await assistant(
            "How many years passed between the completion of the Eiffel Tower and the first Moon landing? "
            "Also, what is the slideshow title at https://httpbin.org/json?"
        )
    )


asyncio.run(main())

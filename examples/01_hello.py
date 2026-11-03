"""The smallest agent: the docstring is the prompt, the parameters fill it in, the answer is text."""

import asyncio

from virtus import agent
from virtus.providers import OpenAI


@agent(provider=OpenAI())
async def haiku(topic: str) -> str:
    """Write a haiku about {topic}."""
    ...


async def main() -> None:
    print(await haiku("the sea"))


asyncio.run(main())

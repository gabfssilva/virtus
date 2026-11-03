"""`AsyncIterator[str]` streams the answer's text; `AsyncIterator[Event]` streams everything filters see,
including tool call arguments as the model writes them."""

import asyncio
from collections.abc import AsyncIterator

from virtus import Event, TextDelta, ToolCallDelta, ToolResult, agent, tool
from virtus.providers import OpenAI


@tool
def population(city: str) -> int:
    """Population of a city."""
    return {"tokyo": 37_400_000, "delhi": 32_900_000, "shanghai": 29_200_000}.get(city.lower(), 0)


@agent(provider=OpenAI())
async def story(topic: str) -> AsyncIterator[str]:
    """Tell a three-sentence story about {topic}."""
    ...


@agent(provider=OpenAI(), capabilities=[population])
async def compare(first: str, second: str) -> AsyncIterator[Event]:
    """Compare the populations of {first} and {second} in two sentences."""
    ...


async def main() -> None:
    async for text in story("a lighthouse"):
        print(text, end="", flush=True)
    print("\n")

    async for event in compare("Tokyo", "Delhi"):
        match event:
            case TextDelta(content=text):
                print(text, end="", flush=True)
            case ToolCallDelta(name=name, arguments=arguments):
                print(f"[{name} <- {arguments!r}]")
            case ToolResult(name=name, content=content):
                print(f"[{name} -> {content}]")
    print()


asyncio.run(main())

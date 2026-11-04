"""A thread keeps the conversation between calls. The same id is the same conversation."""

import asyncio

from virtus import agent
from virtus.providers import OpenAI


@agent(provider=OpenAI(), system="Answer in one short sentence.")
async def chat(message: str) -> str: ...


async def main() -> None:
    thread = chat.thread("user-42")
    print(await thread("My name is Ana and I live in Lisbon."))
    print(await thread("Where do I live?"))

    same = chat.thread("user-42")
    print(await same("And what is my name?"))

    print(f"\n{len(thread.history)} messages in the thread")

    thread.dispose()
    print(await chat.thread("user-42")("What is my name?"))


asyncio.run(main())

"""`compaction` summarizes older messages once the history grows past a token budget."""

import asyncio

from virtus import UserMessage, agent, compaction
from virtus.providers import ChatCompletions


@agent(provider=ChatCompletions(), system="Reply in one sentence.", capabilities=[compaction(max_tokens=150, keep=2)])
async def chat(message: str) -> str: ...


async def main() -> None:
    thread = chat.thread()
    for message in (
        "I'm planning a trip to Japan in April.",
        "I want to see cherry blossoms in Kyoto and eat ramen in Fukuoka.",
        "My budget is around 3000 euros for two weeks.",
        "I don't like crowded places.",
        "Where should I go first, and how much money did I say I have?",
    ):
        print(f"> {message}")
        print(await thread(message))

    print(f"\n{len(thread.history)} messages kept:")
    for message in thread.history:
        match message:
            case UserMessage(content=content) if content.startswith("Summary"):
                print(f"  {content[:120]}...")
            case _:
                print(f"  {type(message).__name__}")


asyncio.run(main())

"""`on_input` rewrites or rejects what the user sends; `on_message` sees every complete message."""

import asyncio
import re

from virtus import AssistantMessage, Message, UserMessage, agent, on_input, on_message
from virtus.providers import OpenAI

CARD = re.compile(r"\b\d(?:[ -]?\d){12,15}\b")


@on_input
def redact_cards(message: UserMessage) -> UserMessage:
    return UserMessage(CARD.sub("[card number]", message.content))


@on_input
def refuse_off_topic(message: UserMessage) -> UserMessage:
    if "weather" in message.content.lower():
        raise PermissionError("this assistant does not discuss the weather")
    return message


@on_message
def log(message: Message) -> Message:
    if isinstance(message, AssistantMessage):
        print(f"  [assistant replied with {len(message.content)} chars]")
    return message


@agent(provider=OpenAI(), capabilities=[redact_cards, refuse_off_topic, log])
async def support(message: str) -> str: ...


async def main() -> None:
    thread = support.thread()
    print(await thread("My card 4111 1111 1111 1111 was charged twice. Repeat the card number I gave you."))
    print(thread.history[0])

    try:
        await thread("What's the weather like?")
    except PermissionError as error:
        print(f"rejected: {error}")


asyncio.run(main())

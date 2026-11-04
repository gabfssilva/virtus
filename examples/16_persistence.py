"""A journal keeps threads on disk: a new process picks up the conversation, and an interrupted run resumes."""

import asyncio
import tempfile

from virtus import JsonJournal, agent
from virtus.providers import OpenAI


async def main() -> None:
    with tempfile.TemporaryDirectory() as folder:

        @agent(provider=OpenAI(), system="Answer in one short sentence.", journal=JsonJournal(folder))
        async def assistant(message: str) -> str: ...

        first = assistant.thread("user-42")
        print(await first("My name is Ana and I live in Lisbon."))

        # A copy of the agent stands in for a new process: same journal, same thread id, history from disk.
        second = assistant.replace().thread("user-42")
        print(f"\nloaded {len(second.history)} messages from disk")
        print(await second("Where do I live?"))
        print(await second.resume())  # nothing pending: the last answer, without a call


asyncio.run(main())

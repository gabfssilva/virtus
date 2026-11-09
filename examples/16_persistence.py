"""A journal keeps threads on disk: a new process picks up the conversation, and an interrupted run resumes."""

import asyncio
import tempfile

from virtus import JsonJournal, agent
from virtus.providers import ChatCompletions


async def main() -> None:
    with tempfile.TemporaryDirectory() as folder:

        @agent(provider=ChatCompletions(), system="Answer in one short sentence.", journal=JsonJournal(folder))
        async def assistant(message: str) -> str: ...

        first = assistant.thread("user-42")
        print(await first("My name is Ana and I live in Lisbon."))

        # Any handle to the id reads the history from the journal, as a new process would.
        second = assistant.thread("user-42")
        print(f"\nloaded {len(second.history)} messages from disk")
        print(await second("Where do I live?"))
        print(await second.resume())  # nothing pending: the last answer, without a call


asyncio.run(main())

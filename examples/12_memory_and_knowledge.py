"""`knowledge` gives the agent documents to consult; `memory` lets it keep notes between conversations.

Both scale retrieval to the volume: inline while it fits, then an index plus `read`, then `search`.
"""

import asyncio
import tempfile
from pathlib import Path

from virtus import agent, knowledge, memory
from virtus.providers import ChatCompletions


async def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        docs = Path(folder, "docs")
        docs.mkdir()
        (docs / "refunds.md").write_text("# Refund policy\n\nRefunds are issued within 5 business days of the request.")
        (docs / "shipping.md").write_text("# Shipping\n\nOrders ship in 48 hours; express shipping costs 9 EUR.")

        @agent(provider=ChatCompletions(), capabilities=[knowledge(docs), memory(Path(folder, "notes"))])
        async def support(message: str) -> str: ...

        print(await support("Hi, I'm Marina. How long do refunds take?"))
        print(await support("Please remember that I prefer express shipping."))

        print("\nnotes on disk:")
        for note in Path(folder, "notes").glob("*.md"):
            print(f"--- {note.name}\n{note.read_text()}")

        print(await support("What do you know about me?"))


asyncio.run(main())

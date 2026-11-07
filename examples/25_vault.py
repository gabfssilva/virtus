"""`secrets(...)` lets the agent use a token it never sees: it writes `{{secrets.token}}` in a tool argument, the
value goes in just before the tool runs, and whatever the tool echoes back comes out redacted."""

import asyncio

from virtus import agent, observe, secrets
from virtus.providers import OpenAI
from virtus.tools import http

# Chat Completions only takes function tools from gpt-6 with reasoning off; the Responses API has no such limit.
model = OpenAI(reasoning_effort="none")


@agent(
    provider=model,
    system="Use the http tool. Report the status and what the server said, in one line.",
    capabilities=[http, secrets(token=("hunter2-but-longer", "bearer token for httpbin.org")), observe()],
)
async def client(request: str) -> str: ...


async def main() -> None:
    print(await client("Call https://httpbin.org/bearer with the bearer token and tell me if it was accepted."))


asyncio.run(main())

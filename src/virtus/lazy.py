"""Tools offered on demand, so a large toolbox doesn't fill every request; see `lazy`."""

from collections.abc import AsyncIterator
from dataclasses import replace
from functools import reduce

from virtus.capability import flatten
from virtus.message import Event
from virtus.middleware import Capability, Middleware, Next
from virtus.request import Request
from virtus.tool import tool

HEADER = "Tools you can load with `load_tools` (name: what it does):"


def lazy(*capabilities: Capability | Middleware) -> Middleware:
    """Offer these capabilities' tools on demand instead of on every request.

    The model sees a catalog in the system prompt, one line per tool, and a `load_tools` tool; a loaded
    tool joins the request's tools from the next step on, for the rest of that conversation. Anything
    that goes in `capabilities=` goes here: tools, `Capabilities` classes, `workspace(...)`, MCP servers.
    """
    chain = flatten(capabilities)
    loaded: dict[str, set[str]] = {}

    async def lazily(request: Request, next: Next) -> AsyncIterator[Event]:
        wanted = loaded.setdefault(request.thread.id, set())
        outer = request.tools

        async def offer(inner: Request) -> AsyncIterator[Event]:
            added = [t for t in inner.tools if all(t is not o for o in outer)]
            catalog = [t for t in added if t.name not in wanted]

            def load_tools(names: list[str]) -> str:
                """Load tools from the catalog; they are available from your next step on."""
                known = {t.name for t in added}
                if unknown := [name for name in names if name not in known]:
                    raise ValueError(f"unknown tools: {', '.join(unknown)}; known: {', '.join(sorted(known))}")
                wanted.update(names)
                return f"Loaded {', '.join(names)}; available from your next step on."

            lines = "\n".join(f"- {t.name}: {t.description.partition(chr(10))[0]}" for t in catalog)
            note = f"{HEADER}\n{lines}" if catalog else ""
            system = "\n\n".join(part for part in (inner.system, note) if part)
            tools = (*outer, tool(load_tools), *(t for t in added if t.name in wanted))
            async for event in next(replace(inner, system=system, tools=tools)):
                yield event

        def link(inner: Next, middleware: Middleware) -> Next:
            return lambda request: middleware(request, inner)

        innermost: Next = offer
        step = reduce(link, reversed(chain), innermost)
        async for event in step(request):
            yield event

    return lazily

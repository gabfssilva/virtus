"""The one tool through which an agent talks to other agents.

Every agent (or thread) given as a capability adds itself as a target of the same `dispatch` tool
instead of becoming a tool of its own, so the model learns one tool however many agents it can
reach. The `arguments` are those of the addressed agent's template, which formats the message as a
direct call would; the targets and their open conversations are listed in the system prompt.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, replace
from functools import reduce
from operator import or_
from typing import TYPE_CHECKING

from pydantic import Field, TypeAdapter, create_model

from virtus.message import Event
from virtus.middleware import Middleware, Next
from virtus.request import Request
from virtus.tool import Tool

if TYPE_CHECKING:
    from virtus.agent import Agent
    from virtus.thread import Conversation


@dataclass(frozen=True)
class Target:
    """An agent the model can address: `conversation` gives the thread behind `name` or `name:thread`."""

    agent: Agent[..., object]
    conversation: Callable[[Conversation, str | None], Conversation]


@dataclass(frozen=True)
class Dispatch(Tool):
    targets: tuple[Target, ...]

    @classmethod
    def to(cls, targets: tuple[Target, ...], parent: Conversation) -> Dispatch:
        by_name = {target.agent.name: target for target in targets}

        async def dispatch(to: list[str], arguments: dict[str, object]) -> dict[str, object]:
            if unknown := [address for address in to if address.partition(":")[0] not in by_name]:
                raise ValueError(f"unknown agents: {', '.join(unknown)}; known: {', '.join(by_name)}")

            async def deliver(address: str) -> object:
                name, _, thread = address.partition(":")
                target = by_name[name]
                values = dict(target.agent.parameters.model_validate(arguments))
                return await target.conversation(parent, thread or None).send(target.agent.input(**values))

            replies = await asyncio.gather(*(deliver(address) for address in to))
            return dict(zip(to, replies, strict=True))

        return cls("dispatch", DESCRIPTION, PARAMETERS, dispatch, targets)

    @property
    def schema(self) -> Mapping[str, object]:
        """The parameters schema with `arguments` spelled out as one of the targets' templates."""
        union = reduce(or_, (target.agent.parameters for target in self.targets))
        arguments = TypeAdapter[object](union).json_schema()
        schema = PARAMETERS.model_json_schema()
        defs = {**schema.get("$defs", {}), **arguments.pop("$defs", {})}
        properties = {**schema["properties"], "arguments": {**schema["properties"]["arguments"], **arguments}}
        return {**schema, "properties": properties, **({"$defs": defs} if defs else {})}


def dispatching(target: Target) -> Middleware:
    """Middleware adding `target` to the request's `dispatch` tool, creating the tool if it is the first."""

    async def add(request: Request, next: Next) -> AsyncIterator[Event]:
        parent = request.thread
        found = next_dispatch(request)
        tool = Dispatch.to((*(found.targets if found else ()), target), parent)
        tools = (*(t for t in request.tools if t is not found), tool)

        name = target.agent.name
        open = [thread for agent, thread in parent.children if agent == name]
        line = f"- {name}: {target.agent.description}" + (f" Open conversations: {', '.join(open)}" if open else "")
        if found:
            system = f"{request.system}\n{line}"
        else:
            system = "\n\n".join(part for part in (request.system, f"{HEADER}\n{line}") if part)

        async for event in next(replace(request, system=system, tools=tools)):
            yield event

    return add


def next_dispatch(request: Request) -> Dispatch | None:
    return next((tool for tool in request.tools if isinstance(tool, Dispatch)), None)


HEADER = "Agents you can `dispatch` to:"

DESCRIPTION = (
    "Send a request to one or more agents and get their replies, keyed by address. An address is an agent "
    "name for a one-off request, or `agent:conversation` to continue (or start) a named conversation with "
    "that agent, which then remembers earlier requests. `arguments` are the addressed agent's; one call "
    "sends the same arguments to every address, at once."
)

PARAMETERS = create_model(
    "dispatch",
    to=(list[str], Field(description="Addresses: `agent` or `agent:conversation`.", min_length=1)),
    arguments=(dict[str, object], Field(description="The arguments of the addressed agent.")),
)

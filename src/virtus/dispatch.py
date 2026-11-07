"""The one tool through which an agent talks to other agents.

Every agent (or thread) given as a capability adds itself as a target of the same `dispatch` tool
instead of becoming a tool of its own, so the model learns one tool however many agents it can
reach. The `arguments` are those of the addressed agent's template, which formats the message as a
direct call would; the targets and their open conversations are listed in the system prompt.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass, replace
from functools import reduce
from operator import or_
from typing import TYPE_CHECKING, get_args

from pydantic import Field, TypeAdapter, create_model

from virtus.message import Audio, Event, Image
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
class Failed:
    """What a failing address answers with, so the other addresses' replies still come back."""

    error: str

    def as_reply(self) -> dict[str, str]:
        return {"error": self.error}


@dataclass(frozen=True)
class Dispatch(Tool):
    targets: tuple[Target, ...]

    @classmethod
    def to(cls, targets: tuple[Target, ...], parent: Conversation) -> Dispatch:
        by_name = {target.agent.name: target for target in targets}

        async def dispatch(to: list[str], arguments: dict[str, object]) -> dict[str, object]:
            if unknown := [address for address in to if address.partition(":")[0] not in by_name]:
                raise ValueError(f"unknown agents: {', '.join(unknown)}; known: {', '.join(by_name)}")

            async def deliver(address: str) -> object | Failed:
                """The agent's reply, or `Failed`, so one address failing doesn't lose the others' replies."""
                name, _, thread = address.partition(":")
                target = by_name[name]
                try:
                    values = dict(target.agent.parameters.model_validate(arguments))
                    return await target.conversation(parent, thread or None).send(target.agent.input(**values))
                except Exception as error:
                    return Failed(f"{type(error).__name__}: {error}")

            replies = dict(zip(to, await asyncio.gather(*(deliver(address) for address in to)), strict=True))
            failed = {address: reply for address, reply in replies.items() if isinstance(reply, Failed)}
            if failed and len(failed) == len(replies):
                raise ValueError("\n".join(f"{address}: {reply.error}" for address, reply in failed.items()))
            return {address: r.as_reply() if isinstance(r, Failed) else r for address, r in replies.items()}

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
    """Middleware adding `target` to the request's `dispatch` tool, creating the tool if it is the first.

    An agent with an `Image` or `Audio` parameter is refused: the model has no way to pass media.
    """
    fields = target.agent.parameters.model_fields.items()
    if media := [name for name, field in fields if _media(field.annotation)]:
        names = ", ".join(media)
        raise TypeError(f"{target.agent.name}: can't be dispatched to; the model can't fill {names} with media")

    async def add(request: Request, next: Next) -> AsyncIterator[Event]:
        parent = request.thread
        found = next_dispatch(request)
        tool = Dispatch.to((*(found.targets if found else ()), target), parent)
        tools = (*(t for t in request.tools if t is not found), tool)

        name = target.agent.name
        open = [thread for agent, thread in parent.children if agent == name and not ONE_OFF.fullmatch(thread)]
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


def _media(annotation: object) -> bool:
    return annotation in (Image, Audio) or any(_media(inner) for inner in get_args(annotation))


HEADER = "Agents you can `dispatch` to:"

ONE_OFF = re.compile(r"[0-9a-f]{32}")
"""The generated name of a one-off request's thread, kept in the caller's journal but not offered to the model."""

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

"""Secrets the agent uses without seeing them.

The model writes `{{secrets.name}}` wherever a tool argument needs the value; the placeholder is
replaced in the arguments just before the tool runs, and the value is redacted back to the placeholder
in whatever the tool returns. The agent only ever learns that the secret exists.
"""

import json
import re
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, replace

from virtus.message import Event, ToolCall, ToolResult
from virtus.middleware import Middleware, Next
from virtus.request import Request
from virtus.tool import Tool

PLACEHOLDER = re.compile(r"\{\{\s*secrets\.([A-Za-z_][\w-]*)\s*\}\}")

INTRO = "Secrets you can use as placeholders in tool arguments; their values are never shown to you:"


def secrets(intro: str = INTRO, /, **values: str | tuple[str, str]) -> Middleware:
    """Make these secrets usable as `{{secrets.name}}` in the arguments of any tool before it in the chain.

    A value may come with a description: `secrets(github=(token, "GitHub token"))`. Put `secrets()`
    after the tools it guards; tools added later in the chain are not covered. `intro` opens the list of
    secrets in the system prompt; it is positional, so that any name is free for a secret.
    """
    described = {name: (value, "") if isinstance(value, str) else value for name, value in values.items()}
    vault = {name: value for name, (value, _) in described.items()}
    lines = (f"- {{{{secrets.{name}}}}}" + (f": {about}" if about else "") for name, (_, about) in described.items())
    note = "\n".join((intro, *lines))

    async def guard(request: Request, next: Next) -> AsyncIterator[Event]:
        system = "\n\n".join(part for part in (request.system, note) if part)
        tools = tuple(Guarded.over(tool, vault) for tool in request.tools)
        deferred = tuple(Guarded.over(tool, vault) for tool in request.deferred)
        async for event in next(replace(request, system=system, tools=tools, deferred=deferred)):
            yield event

    return guard


@dataclass(frozen=True)
class Guarded(Tool):
    """A tool whose arguments get the secrets filled in and whose results get them redacted."""

    inner: Tool
    vault: Mapping[str, str]

    @classmethod
    def over(cls, tool: Tool, vault: Mapping[str, str]) -> "Guarded":
        return cls(tool.name, tool.description, tool.parameters, tool.function, tool, vault)

    @property
    def schema(self) -> Mapping[str, object]:
        return self.inner.schema

    async def execute(self, call: ToolCall) -> ToolResult:
        def fill(match: re.Match[str]) -> str:
            if (name := match.group(1)) not in self.vault:
                raise KeyError(name)
            return json.dumps(self.vault[name])[1:-1]  # escaped for the JSON string it sits in

        try:
            arguments = PLACEHOLDER.sub(fill, call.arguments or "")
        except KeyError as error:
            known = ", ".join(self.vault) or "none"
            return ToolResult(call.id, call.name, f"unknown secret {error.args[0]!r}; known: {known}", is_error=True)

        result = await self.inner.execute(replace(call, arguments=arguments))
        content = result.content
        for name, value in self.vault.items():
            if value:
                content = content.replace(value, f"{{{{secrets.{name}}}}}")
        return replace(result, content=content)

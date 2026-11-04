import asyncio
import inspect
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from types import MethodType
from typing import get_type_hints

from pydantic import BaseModel, create_model
from pydantic_core import to_json

from virtus.message import Audio, Event, Image, ToolCall, ToolResult, placeholder
from virtus.middleware import Capability, Middleware, Next
from virtus.request import Request


@dataclass(frozen=True)
class Tool(Capability):
    name: str
    description: str
    parameters: type[BaseModel]
    function: Callable[..., object]

    def __get__(self, instance: object | None, owner: type | None = None) -> "Tool":
        return self if instance is None else replace(self, function=MethodType(self.function, instance))

    def middleware(self) -> tuple[Middleware, ...]:
        async def provide(request: Request, next: Next) -> AsyncIterator[Event]:
            async for event in next(replace(request, tools=(*request.tools, self))):
                yield event

        return (provide,)

    @property
    def schema(self) -> Mapping[str, object]:
        return self.parameters.model_json_schema()

    async def execute(self, call: ToolCall) -> ToolResult:
        """Run the tool; any failure becomes an error result so the model can react to it.

        A returned `Image` or `Audio`, alone or in a sequence, is attached to the result and reads as
        `[image]` or `[audio]` in its text.
        """
        try:
            arguments = dict(self.parameters.model_validate_json(call.arguments or "{}"))
            value = await invoke(self.function, arguments)
        except Exception as error:
            return ToolResult(call.id, call.name, f"{type(error).__name__}: {error}", is_error=True)

        images, audio = Image.found_in(value), Audio.found_in(value)
        match value:
            case str():
                return ToolResult(call.id, call.name, value)
            case Image() | Audio():
                return ToolResult(call.id, call.name, placeholder(value) or "", images=images, audio=audio)
            case Sequence() if images or audio:
                text = to_json([placeholder(v) or v for v in value]).decode()
                return ToolResult(call.id, call.name, text, images=images, audio=audio)
            case _:
                return ToolResult(call.id, call.name, to_json(value).decode())


def tool(function: Callable[..., object], *, name: str | None = None, description: str | None = None) -> Tool:
    """Turn a function, sync or async, into a tool described by its signature and docstring."""
    return Tool(
        name=name or function.__name__,
        description=description or inspect.getdoc(function) or "",
        parameters=parameters_model(function),
        function=function,
    )


def parameters_model(function: Callable[..., object]) -> type[BaseModel]:
    """Build the arguments model from a signature, skipping a leading `self` so methods can be tools."""
    hints = get_type_hints(function, include_extras=True)
    parameters = list(inspect.signature(function).parameters.values())
    parameters = parameters[1:] if parameters and parameters[0].name == "self" else parameters

    if missing := [p.name for p in parameters if p.name not in hints]:
        raise TypeError(f"{function.__name__}: parameters need type annotations: {', '.join(missing)}")

    fields = {p.name: (hints[p.name], ... if p.default is p.empty else p.default) for p in parameters}
    # The stubs check unpacked field definitions against create_model's own `__config__`-style keywords.
    return create_model(function.__name__, **fields)  # pyrefly: ignore[no-matching-overload]


async def invoke(function: Callable[..., object], arguments: Mapping[str, object]) -> object:
    """Await async callables; run sync ones in a thread so they don't block the event loop."""
    if inspect.iscoroutinefunction(function):
        return await function(**arguments)
    return await asyncio.to_thread(function, **arguments)

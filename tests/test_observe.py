import importlib
import io
from itertools import count

import pytest

from tests.scripted import Scripted, calls
from virtus import AssistantMessage, agent, observe, tool


@pytest.fixture
def ticking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every look at the clock is one second later than the previous one."""
    clock = count()
    # `virtus.observe` names the function once the package is imported; the module is reached by name.
    monkeypatch.setattr(importlib.import_module("virtus.observe"), "perf_counter", lambda: float(next(clock)))


def describe_observe():
    async def it_draws_the_run_as_a_tree(ticking: None):
        @tool
        def clock() -> str:
            return "12:00"

        out = io.StringIO()

        @agent(
            provider=Scripted(calls(("clock", {})), AssistantMessage("It is noon.")),
            capabilities=[clock, observe(out, color=False)],
        )
        async def assistant(message: str) -> str: ...

        assert await assistant.thread("u1")("What time is it?") == "It is noon."
        assert out.getvalue() == (
            "▶ u1 · What time is it?\n"
            "├─ Scripted · 1 msg · tools: clock\n"
            "│  1.0s\n"
            "│  ⚙ clock()\n"
            "│  ↳ 12:00 · 1.0s\n"
            "├─ Scripted · 3 msgs · tools: clock\n"
            "│  It is noon.\n"
            "└─ ✓ 11 chars · 2.0s · first token 1.0s · 8.0s total\n"
        )

    async def it_marks_tool_errors_and_shows_arguments(ticking: None):
        @tool
        def divide(a: int, b: int) -> float:
            """Divide."""
            return a / b

        out = io.StringIO()

        @agent(
            provider=Scripted(calls(("divide", {"a": 1, "b": 0})), AssistantMessage("Cannot.")),
            capabilities=[divide, observe(out, color=False)],
        )
        async def assistant(message: str) -> str: ...

        await assistant.thread("u1")("1/0")
        lines = out.getvalue().splitlines()

        assert lines[3] == "│  ⚙ divide(a=1, b=0)"
        assert lines[4] == "│  ✗ divide ZeroDivisionError: division by zero · 1.0s"

    async def it_closes_the_tree_when_the_step_fails(ticking: None):
        out = io.StringIO()

        @agent(provider=Scripted(), capabilities=[observe(out, color=False)])
        async def assistant(message: str) -> str: ...

        with pytest.raises(IndexError):
            await assistant.thread("u1")("hi")

        assert out.getvalue().splitlines()[-1] == "└─ ✗ IndexError: pop from empty list"

    async def it_colors_a_terminal(ticking: None):
        out = io.StringIO()

        @agent(provider=Scripted(AssistantMessage("ok")), capabilities=[observe(out, color=True)])
        async def assistant(message: str) -> str: ...

        await assistant("hi")

        assert "└─ \x1b[32m✓\x1b[0m \x1b[2m2 chars" in out.getvalue()

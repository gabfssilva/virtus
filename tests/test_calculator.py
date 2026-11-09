import json

import pytest

from virtus import ToolCall, ToolResult
from virtus.tools import calculator


async def _calc(expression: str) -> ToolResult:
    return await calculator().execute(ToolCall("c", "calculator", json.dumps({"expression": expression})))


def describe_calculator():
    @pytest.mark.parametrize(
        ("expression", "answer"),
        [
            ("2 + 3 * 4", "14"),
            ("(2 + 3) * 4", "20"),
            ("10 / 4", "2.5"),
            ("10 / 2", "5"),
            ("7 // 2 + 7 % 2", "4"),
            ("2 ** 10", "1024"),
            ("-3 ** 2", "-9"),
            ("sqrt(16) + abs(-2)", "6"),
            ("round(pi, 2)", "3.14"),
            ("max(1, 2, 3) + min(4, 5)", "7"),
            ("factorial(5)", "120"),
            ("2 ** 0.5", "1.4142135623730951"),
        ],
    )
    async def it_evaluates_arithmetic(expression: str, answer: str):
        assert (await _calc(expression)).content == answer

    @pytest.mark.parametrize(
        "expression",
        ["__import__('os')", "x + 1", "(1).real", "[1, 2]", "True + 1", "1 if 2 else 3", "lambda: 1", "sqrt(x=4)"],
    )
    async def it_refuses_anything_but_arithmetic(expression: str):
        result = await _calc(expression)
        assert result.is_error, expression

    async def it_bounds_exponents():
        assert (await _calc("2 ** 100000")).is_error
        assert (await _calc("(10 ** 1000) ** 5000")).is_error

    async def it_takes_its_functions_constants_and_bounds():
        tool = calculator(functions={"double": lambda x: 2 * x}, constants={"answer": 42}, max_exponent=3)

        async def run(expression: str) -> ToolResult:
            return await tool.execute(ToolCall("c", "calculator", json.dumps({"expression": expression})))

        assert (await run("double(answer)")).content == "84"
        assert (await run("sqrt(4)")).is_error and (await run("2 ** 4")).is_error
        assert "double" in tool.description and "answer" in tool.description and "sqrt" not in tool.description

    async def it_reports_math_errors():
        assert "ZeroDivisionError" in (await _calc("1 / 0")).content
        assert "ValueError" in (await _calc("sqrt(-1)")).content

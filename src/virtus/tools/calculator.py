import ast
import math
import operator
from collections.abc import Callable, Mapping

from virtus.tool import Tool, tool

type Number = int | float

OPERATORS: Mapping[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

FUNCTIONS: Mapping[str, Callable[..., object]] = {
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "atan2": math.atan2,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "floor": math.floor,
    "ceil": math.ceil,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "hypot": math.hypot,
    "degrees": math.degrees,
    "radians": math.radians,
}

CONSTANTS: Mapping[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}


def calculator(
    *,
    functions: Mapping[str, Callable[..., object]] = FUNCTIONS,
    constants: Mapping[str, float] = CONSTANTS,
    max_exponent: int = 10_000,
    max_bits: int = 1_000_000,
) -> Tool:
    """The `calculator` tool over `functions` and `constants`. A power is refused past `max_exponent`, or when
    an integer result would take more than `max_bits`, so an expression cannot stall the event loop."""

    def evaluate(node: ast.expr) -> Number:
        match node:
            case ast.Constant(value=bool()):
                raise ValueError("booleans are not numbers")
            case ast.Constant(value=int() | float() as value):
                return value
            case ast.UnaryOp(op=ast.USub(), operand=operand):
                return -evaluate(operand)
            case ast.UnaryOp(op=ast.UAdd(), operand=operand):
                return evaluate(operand)
            case ast.BinOp(left=left, op=ast.Pow(), right=right):
                return power(evaluate(left), evaluate(right))
            case ast.BinOp(left=left, op=op, right=right) if type(op) in OPERATORS:
                return OPERATORS[type(op)](evaluate(left), evaluate(right))
            case ast.Name(id=name) if name in constants:
                return constants[name]
            case ast.Call(func=ast.Name(id=name), args=args, keywords=[]) if name in functions:
                return _number(functions[name](*(evaluate(a) for a in args)))
            case ast.Name(id=name) | ast.Call(func=ast.Name(id=name)):
                raise ValueError(f"unknown name {name!r}")
            case _:
                raise ValueError(f"unsupported syntax: {type(node).__name__}")

    def power(base: Number, exponent: Number) -> Number:
        huge = isinstance(base, int) and isinstance(exponent, int) and base.bit_length() * exponent > max_bits
        if abs(exponent) > max_exponent or huge:
            raise ValueError("exponent too large")
        return operator.pow(base, exponent)

    def calculator(expression: str) -> str:
        value = evaluate(ast.parse(expression.strip(), mode="eval").body)
        if isinstance(value, float) and value.is_integer() and abs(value) < 2**53:
            return str(int(value))
        return str(value)

    description = (
        "Evaluate an arithmetic expression exactly: numbers, + - * / // % **, parentheses, the functions "
        f"{', '.join(functions)}, and the constants {', '.join(constants)}."
    )
    return tool(calculator, description=description)


def _number(value: object) -> Number:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"not a number: {value!r}")
    return value

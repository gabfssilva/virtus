"""Three tools ported from the old library: a calculator, a sandboxed file system and Wikipedia search.

Each is a few lines on top of `@tool` or `Capabilities`; the library ships none of them.
"""

import ast
import asyncio
import operator
import tempfile
from collections.abc import Callable
from pathlib import Path

import httpx2 as httpx

from virtus import Capabilities, agent, tool
from virtus.providers import OpenAI

OPERATORS: dict[type[ast.operator] | type[ast.unaryop], Callable[..., float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
}


@tool
def calculator(expression: str) -> float:
    """Evaluate an arithmetic expression with + - * / ** % and parentheses."""

    def evaluate(node: ast.expr) -> float:
        match node:
            case ast.Constant(value=int() | float() as value):
                return value
            case ast.BinOp(left=left, op=op, right=right):
                return OPERATORS[type(op)](evaluate(left), evaluate(right))
            case ast.UnaryOp(op=op, operand=operand):
                return OPERATORS[type(op)](evaluate(operand))
            case _:
                raise ValueError(f"unsupported syntax: {ast.dump(node)}")

    return evaluate(ast.parse(expression, mode="eval").body)


class Files(Capabilities):
    """Read and write files under one folder; paths that escape it are refused."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    @tool
    def list(self) -> list[str]:
        """List every file in the folder."""
        return sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*") if p.is_file())

    @tool
    def read(self, path: str) -> str:
        """Read a file."""
        return self._inside(path).read_text()

    @tool
    def write(self, path: str, content: str) -> str:
        """Create or overwrite a file."""
        file = self._inside(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text(content)
        return f"wrote {len(content)} characters to {path}"

    def _inside(self, path: str) -> Path:
        resolved = (self.root / path).resolve()
        if not resolved.is_relative_to(self.root):
            raise PermissionError(f"{path} is outside the folder")
        return resolved


@tool
async def wikipedia(query: str, limit: int = 3) -> list[dict[str, str]]:
    """Search English Wikipedia and return titles with one-line descriptions."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            "https://en.wikipedia.org/w/rest.php/v1/search/page",
            params={"q": query, "limit": limit},
            headers={"User-Agent": "virtus-example/0.1"},
        )
        response.raise_for_status()
    return [
        {"title": page["title"], "description": page.get("description") or ""} for page in response.json()["pages"]
    ]


async def main() -> None:
    with tempfile.TemporaryDirectory() as folder:
        files = Files(Path(folder))

        @agent(provider=OpenAI(), capabilities=[calculator, files, wikipedia])
        async def assistant(task: str) -> str: ...

        print(await assistant("What is (17 * 23) ** 2 / 7? Use the calculator."))
        print(await assistant("Look up the Voyager 1 probe on Wikipedia and save a two-line note to voyager.md."))
        print(await assistant("Which files exist now, and what do they say?"))


asyncio.run(main())

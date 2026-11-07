"""A small MCP server the tests start over stdio: `python tests/mcp_server.py`."""

import os

from mcp.server.mcpserver import Image, MCPServer

server = MCPServer("calc")


@server.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@server.tool()
def shout(text: str) -> str:
    """Upper-case the text."""
    return text.upper()


@server.tool()
def whoami() -> str:
    """The TESTER variable of the server's environment."""
    return os.environ.get("TESTER", "nobody")


@server.tool()
def fail() -> str:
    """Always fails."""
    raise RuntimeError("as requested")


@server.tool()
def dot() -> Image:
    """A one-pixel PNG."""
    return Image(data=PNG, format="png")


PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c4890000000d4944415478da63f8cfc0f01f0005"
    "0002017e5e9e6c0000000049454e44ae426082"
)

if __name__ == "__main__":
    server.run()

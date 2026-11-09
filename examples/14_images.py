"""An `Image` parameter is attached to the user message; the prompt refers to it as `[image]`."""

import asyncio
import struct
import zlib

from virtus import Image, agent
from virtus.providers import ChatCompletions


def solid_png(red: int, green: int, blue: int, size: int = 128) -> Image:
    """A single-colour PNG built by hand, so the example needs no image file or library."""

    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body))

    row = b"\x00" + bytes((red, green, blue)) * size
    header = chunk(b"IHDR", struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0))
    data = chunk(b"IDAT", zlib.compress(row * size))
    return Image(b"\x89PNG\r\n\x1a\n" + header + data + chunk(b"IEND", b""), "image/png")


@agent(provider=ChatCompletions())
async def colour(picture: Image) -> str:
    """What colour is this image? Answer with one word."""
    ...


async def main() -> None:
    print(await colour(solid_png(220, 20, 20)))
    print(await colour(solid_png(20, 20, 220)))


asyncio.run(main())

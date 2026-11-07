"""An agent declared `-> Image` answers with a picture. Needs a model that draws.

Reached here through OpenRouter, so this example reads `OPENROUTER_API_KEY`.
"""

import asyncio
import tempfile
from pathlib import Path

from virtus import Image, agent
from virtus.providers import openrouter

painter = openrouter("google/gemini-3.1-flash-lite-image")


@agent(provider=painter)
async def draw(subject: str) -> Image:
    """Draw {subject}. A small, simple illustration."""
    ...


async def main() -> None:
    picture = await draw("a red bicycle leaning on a white wall")
    path = Path(tempfile.gettempdir()) / f"virtus_draw.{picture.media_type.removeprefix('image/')}"
    path.write_bytes(picture.data)
    print(f"{len(picture.data)} bytes of {picture.media_type} written to {path}")


asyncio.run(main())

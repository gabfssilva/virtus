"""An agent declared `-> Audio` answers out loud; an `Audio` parameter is heard. Needs an audio model.

OpenAI's audio models are reached here through OpenRouter, so this example reads `OPENROUTER_API_KEY`.
"""

import asyncio
import os
import tempfile
from pathlib import Path

from virtus import Audio, agent
from virtus.providers import OpenAI

voice = OpenAI("openai/gpt-audio-mini", api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1")


@agent(provider=voice, system="Answer in one short sentence.")
async def speak(text: str) -> Audio:
    """Say this out loud: {text}"""
    ...


@agent(provider=voice)
async def listen(recording: Audio) -> str:
    """Repeat exactly what is said in {recording}, nothing else."""
    ...


async def main() -> None:
    clip = await speak("The quick brown fox jumps over the lazy dog.")
    path = Path(tempfile.gettempdir()) / "virtus_speak.wav"
    path.write_bytes(clip.data)
    print(f"{len(clip.data)} bytes of {clip.media_type} written to {path}")

    print(await listen(clip))


asyncio.run(main())

"""A spoken conversation over the Realtime API: frames in, the reply streamed back, tools in between.

The question is synthesized with OpenAI's text-to-speech so the example needs no microphone. Needs the
`realtime` extra and a realtime model on the key; `OPENAI_BASE_URL` points it at a local server such as oMLX.
"""

import asyncio
import tempfile
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI

from virtus import AssistantMessage, AudioDelta, TextDelta, ToolResult, agent, tool
from virtus.providers import OpenAI

FRAME = 24_000 * 2 // 10  # 100 ms of 24 kHz 16-bit mono PCM


@tool
def clock() -> str:
    """The current time, to the minute."""
    return datetime.now().strftime("%H:%M")


@agent(provider=OpenAI("gpt-realtime-2.1-mini"), system="Answer in one short sentence.", capabilities=[clock])
async def assistant(message: str) -> str: ...


async def main() -> None:
    speech = await AsyncOpenAI().audio.speech.create(
        model="tts-1-hd-1106", voice="alloy", input="Hi! What time is it right now?", response_format="pcm"
    )
    question = await speech.aread()

    async with assistant.thread().live() as call:
        for start in range(0, len(question), FRAME):
            await call.hear(question[start : start + FRAME])

        heard: list[bytes] = []
        async for event in call.turn():
            match event:
                case TextDelta(content=text):
                    print(text, end="", flush=True)
                case AudioDelta(data=pcm):
                    heard.append(pcm)
                case ToolResult(name=name, content=content):
                    print(f"\n[{name} -> {content}]")
                case AssistantMessage(audio=audio) if audio:
                    path = Path(tempfile.gettempdir()) / "virtus_live.wav"
                    path.write_bytes(audio.data)
                    print(f"\n{len(heard)} audio frames, {len(audio.data)} bytes written to {path}")


asyncio.run(main())

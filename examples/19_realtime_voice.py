"""Talk to an agent, hands free: speak, pause, hear the reply, talk over it to interrupt. Tools work mid-call.

Needs the `realtime` extra, `sounddevice` for the microphone and speaker, and a realtime model:
`gpt-realtime-2.1-mini` by default, or another as the first argument (`OPENAI_BASE_URL` points at oMLX).
"""

import asyncio
import contextlib
import sys
from collections.abc import AsyncIterator, Buffer
from datetime import datetime

import sounddevice

from virtus import AssistantMessage, AudioDelta, Interrupted, TextDelta, ToolResult, agent, tool
from virtus.providers import OpenAI

RATE = 24_000


@tool
def clock() -> str:
    """The current time, to the minute."""
    return datetime.now().strftime("%H:%M")


@agent(
    provider=OpenAI(sys.argv[1] if len(sys.argv) > 1 else "gpt-realtime-2.1-mini"),
    system="You are talking out loud. Answer in one or two short sentences.",
    capabilities=[clock],
)
async def assistant(message: str) -> str: ...


async def main() -> None:
    loop = asyncio.get_running_loop()
    frames: asyncio.Queue[bytes] = asyncio.Queue()

    def captured(indata: Buffer, count: int, time: object, status: sounddevice.CallbackFlags) -> None:
        loop.call_soon_threadsafe(frames.put_nowait, bytes(indata))

    async def microphone() -> AsyncIterator[bytes]:
        while True:
            yield await frames.get()

    stream = sounddevice.RawInputStream(
        samplerate=RATE, channels=1, dtype="int16", blocksize=RATE // 10, callback=captured
    )
    speaker = sounddevice.RawOutputStream(samplerate=RATE, channels=1, dtype="int16")

    with stream, speaker:
        print("Speak when ready; Ctrl-C to quit.")
        async with assistant.thread().live() as call:
            async for event in call.converse(microphone()):
                match event:
                    case TextDelta(content=text):
                        print(text, end="", flush=True)
                    case AudioDelta(data=pcm):
                        with contextlib.suppress(sounddevice.PortAudioError):  # aborted by an interruption
                            await asyncio.to_thread(speaker.write, pcm)
                    case Interrupted():
                        speaker.abort()
                        speaker.start()
                        print("\n(interrupted)")
                    case ToolResult(name=name, content=content):
                        print(f"\n[{name} -> {content}]")
                    case AssistantMessage():
                        print()


with contextlib.suppress(KeyboardInterrupt):
    asyncio.run(main())

"""Talk to an agent: Enter to start speaking, Enter again to send; the reply plays back. Tools work mid-call.

Needs the `realtime` extra, `sounddevice` for the microphone and speaker, and a realtime model:
`gpt-realtime-2.1-mini` by default, or another as the first argument (`OPENAI_BASE_URL` points at oMLX).
"""

import asyncio
import sys
import threading
from collections.abc import Buffer
from datetime import datetime

import sounddevice

from virtus import AudioDelta, Live, TextDelta, ToolResult, agent, tool
from virtus.providers import OpenAI

RATE = 24_000
FRAME = RATE // 10  # samples in 100 ms


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
    talking = threading.Event()

    def captured(indata: Buffer, count: int, time: object, status: sounddevice.CallbackFlags) -> None:
        if talking.is_set():
            loop.call_soon_threadsafe(frames.put_nowait, bytes(indata))

    async def forward(call: Live) -> None:
        while pcm := await frames.get():
            await call.hear(pcm)

    microphone = sounddevice.RawInputStream(
        samplerate=RATE, channels=1, dtype="int16", blocksize=FRAME, callback=captured
    )
    speaker = sounddevice.RawOutputStream(samplerate=RATE, channels=1, dtype="int16")

    with microphone, speaker:
        async with assistant.thread().live() as call:
            while await asyncio.to_thread(input, "\nEnter to talk, q to quit: ") != "q":
                talking.set()
                pump = asyncio.create_task(forward(call))
                await asyncio.to_thread(input, "... speak, then Enter to send ")
                talking.clear()
                frames.put_nowait(b"")
                await pump

                async for event in call.turn():
                    match event:
                        case TextDelta(content=text):
                            print(text, end="", flush=True)
                        case AudioDelta(data=pcm):
                            await asyncio.to_thread(speaker.write, pcm)
                        case ToolResult(name=name, content=content):
                            print(f"\n[{name} -> {content}]")
                print()


asyncio.run(main())

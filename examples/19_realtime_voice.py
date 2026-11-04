"""Talk to an agent, hands free: speak, pause, hear the reply. Tools work mid-call.

The microphone stays open; a simple level detector starts a turn when you speak and ends it after
800 ms of quiet. Needs the `realtime` extra, `sounddevice` for the microphone and speaker, and a realtime
model: `gpt-realtime-2.1-mini` by default, or another as the first argument (`OPENAI_BASE_URL` points at oMLX).
"""

import asyncio
import contextlib
import sys
from array import array
from collections import deque
from collections.abc import Buffer
from datetime import datetime
from math import sqrt

import sounddevice

from virtus import AudioDelta, Live, TextDelta, ToolResult, agent, tool
from virtus.providers import OpenAI

RATE = 24_000
FRAME = RATE // 10  # samples in 100 ms
QUIET_FRAMES = 8  # 800 ms of quiet ends the turn
PREROLL = 3  # frames kept from before the detector fired, so the first syllable is not clipped


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


def level(pcm: bytes) -> float:
    samples = array("h", pcm)
    return sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0


async def main() -> None:
    loop = asyncio.get_running_loop()
    frames: asyncio.Queue[bytes] = asyncio.Queue()

    def captured(indata: Buffer, count: int, time: object, status: sounddevice.CallbackFlags) -> None:
        loop.call_soon_threadsafe(frames.put_nowait, bytes(indata))

    async def calibrate() -> float:
        """A second of room noise sets the threshold for speech."""
        ambient = [level(await frames.get()) for _ in range(10)]
        return max(3 * sum(ambient) / len(ambient), 300.0)

    async def listen(call: Live, threshold: float) -> None:
        """Stream the microphone into the turn from the first loud frame until the room goes quiet."""
        recent: deque[bytes] = deque(maxlen=PREROLL)
        quiet, speaking = 0, False
        while True:
            pcm = await frames.get()
            loud = level(pcm) > threshold
            if not speaking:
                recent.append(pcm)
                if loud:
                    speaking = True
                    print("listening...", flush=True)
                    for frame in recent:
                        await call.hear(frame)
            else:
                await call.hear(pcm)
                quiet = 0 if loud else quiet + 1
                if quiet >= QUIET_FRAMES:
                    return

    microphone = sounddevice.RawInputStream(
        samplerate=RATE, channels=1, dtype="int16", blocksize=FRAME, callback=captured
    )
    speaker = sounddevice.RawOutputStream(samplerate=RATE, channels=1, dtype="int16")

    with microphone, speaker:
        threshold = await calibrate()
        print(f"Speak when ready; Ctrl-C to quit. (speech threshold {threshold:.0f})")
        async with assistant.thread().live() as call:
            while True:
                await listen(call, threshold)
                async for event in call.turn():
                    match event:
                        case TextDelta(content=text):
                            print(text, end="", flush=True)
                        case AudioDelta(data=pcm):
                            await asyncio.to_thread(speaker.write, pcm)
                        case ToolResult(name=name, content=content):
                            print(f"\n[{name} -> {content}]")
                print()
                while not frames.empty():  # drop what the microphone heard of the reply
                    frames.get_nowait()


with contextlib.suppress(KeyboardInterrupt):
    asyncio.run(main())

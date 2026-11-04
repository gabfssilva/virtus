"""Spoken conversations on a thread; see `Thread.live`.

Audio is 24 kHz, 16-bit, mono PCM. The caller owns the microphone and the speaker: frames come in through
`hear` or `converse`, and the `AudioDelta`s that come out are for it to play.
"""

import asyncio
from array import array
from collections import deque
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from math import sqrt

from virtus.message import Audio, AudioDelta, Event, Image, UserMessage
from virtus.provider import Session

RATE = 24_000

type Detector = Callable[[bytes], bool]
"""Whether a frame of PCM is speech."""

type _Frames = asyncio.Queue[bytes | None]
"""Microphone frames; `None` once they run out."""


@dataclass(frozen=True)
class Interrupted:
    """The user spoke over the reply: stop playing it. `heard_ms` of it had been handed out by then."""

    heard_ms: int


class Live:
    """A spoken conversation on a thread.

    `hear` streams the user's speech and `turn` answers it: every turn is a user message carrying the audio
    heard since the last one, and any image shown, so it lands in the history and the journal like a typed
    message, and the middleware chain runs on it as on any step. `converse` drives both from a microphone.
    """

    def __init__(self, session: Session, run: Callable[[UserMessage], AsyncIterator[Event]]) -> None:
        self._session = session
        self._run = run
        self._heard: list[bytes] = []
        self._shown: list[Image] = []

    async def hear(self, pcm: bytes) -> None:
        """Send a frame of the user's speech."""
        self._heard.append(pcm)
        await self._session.hear(pcm)

    async def show(self, image: Image) -> None:
        self._shown.append(image)
        await self._session.show(image)

    async def turn(self) -> AsyncIterator[Event]:
        """End the user's turn and stream the reply: deltas, tool results and the assistant messages."""
        message = UserMessage("[audio]", tuple(self._shown), (Audio.pcm16(b"".join(self._heard)),))
        self._heard.clear()
        self._shown.clear()
        await self._session.commit()
        async for event in self._run(message):
            yield event

    async def interrupt(self, heard_ms: int) -> None:
        """Stop the reply in progress; `heard_ms` is how much of it the user heard before talking over it."""
        await self._session.interrupt(heard_ms)

    async def converse(
        self,
        frames: AsyncIterator[bytes],
        *,
        speech: Detector | None = None,
        quiet_ms: int = 800,
        barge_in_ms: int = 300,
    ) -> AsyncIterator[Event | Interrupted]:
        """Talk hands free: a turn starts when `speech` is detected, ends after `quiet_ms` of silence, and
        speech lasting `barge_in_ms` during the reply interrupts it and opens the next turn.

        The 300 ms before the detector fires are sent too, so the first syllable is not clipped. Frames
        heard during the reply that are not speech are dropped: with open speakers that is the reply
        itself, which `Level` tells from the room only by volume; headphones avoid the problem.
        """
        detect = speech or Level()
        queue: _Frames = asyncio.Queue()

        async def pump() -> None:
            async for pcm in frames:
                queue.put_nowait(pcm)
            queue.put_nowait(None)

        pumping = asyncio.create_task(pump())
        try:
            opening: list[bytes] = []
            while True:
                if not await self._listen(queue, detect, opening, quiet_ms):
                    return
                opening = []
                async for event in self._reply(queue, detect, barge_in_ms, opening):
                    yield event
        finally:
            pumping.cancel()

    async def _listen(self, queue: _Frames, detect: Detector, opening: list[bytes], quiet_ms: int) -> bool:
        """Hear one turn of speech; `False` when the frames ran out first."""
        recent: deque[bytes] = deque()
        for pcm in opening:
            await self.hear(pcm)
        speaking, quiet = bool(opening), 0
        while quiet < quiet_ms:
            pcm = await queue.get()
            if pcm is None:
                return False
            loud = detect(pcm)
            if speaking:
                await self.hear(pcm)
                quiet = 0 if loud else quiet + _ms(pcm)
            elif loud:
                speaking = True
                for frame in (*recent, pcm):
                    await self.hear(frame)
            else:
                recent.append(pcm)
                while sum(map(_ms, recent)) > 300:
                    recent.popleft()
        return True

    async def _reply(
        self, queue: _Frames, detect: Detector, barge_in_ms: int, opening: list[bytes]
    ) -> AsyncIterator[Event | Interrupted]:
        """Stream the reply while watching the frames; speech that interrupts it is collected in `opening`."""
        reply = self.turn()
        heard, loud, interrupted = 0, 0, False
        next_event = asyncio.ensure_future(anext(reply))
        next_frame = asyncio.ensure_future(queue.get())
        try:
            while True:
                done, _ = await asyncio.wait({next_event, next_frame}, return_when=asyncio.FIRST_COMPLETED)
                if next_event in done:
                    try:
                        event = next_event.result()
                    except StopAsyncIteration:
                        return
                    if isinstance(event, AudioDelta):
                        heard += _ms(event.data)
                    if not (interrupted and isinstance(event, AudioDelta)):
                        yield event
                    next_event = asyncio.ensure_future(anext(reply))
                if next_frame in done:
                    pcm = next_frame.result()
                    if pcm is None:
                        queue.put_nowait(None)  # the end of the frames is for `_listen` to see too
                    else:
                        loud = loud + _ms(pcm) if detect(pcm) else 0
                        opening[:] = [*opening, pcm] if loud else []
                        if loud >= barge_in_ms:
                            interrupted = True
                            await self.interrupt(heard)
                            yield Interrupted(heard)
                    # Once interrupted, the frames are the next turn's: leave them in the queue for `_listen`.
                    stop = pcm is None or interrupted
                    next_frame = asyncio.ensure_future(_never() if stop else queue.get())
        finally:
            next_frame.cancel()


class Level:
    """Speech is sound louder than the room. The first second of frames calibrates: speech is three times
    the ambient level, and never less than `floor`. A `threshold` skips the calibration."""

    def __init__(self, threshold: float | None = None, floor: float = 300.0) -> None:
        self.threshold = threshold
        self.floor = floor
        self._ambient: list[float] = []
        self._calibrated_ms = 0

    def __call__(self, pcm: bytes) -> bool:
        samples = array("h", pcm)
        level = sqrt(sum(s * s for s in samples) / len(samples)) if samples else 0.0
        if self.threshold is None:
            self._ambient.append(level)
            self._calibrated_ms += _ms(pcm)
            if self._calibrated_ms >= 1000:
                self.threshold = max(3 * sum(self._ambient) / len(self._ambient), self.floor)
            return False
        return level > self.threshold


def _ms(pcm: bytes) -> int:
    return len(pcm) * 1000 // (2 * RATE)


async def _never() -> None:
    await asyncio.Event().wait()

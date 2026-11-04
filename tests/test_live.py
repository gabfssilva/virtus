import asyncio
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager

import pytest

from tests.scripted import Scripted, calls
from virtus import (
    AssistantMessage,
    Audio,
    AudioDelta,
    Delta,
    Event,
    Image,
    Interrupted,
    Level,
    LiveProvider,
    Next,
    Provider,
    Request,
    Session,
    TextDelta,
    ToolResult,
    UserMessage,
    agent,
    context,
    tool,
)
from virtus.tool import Tool

FRAME = b"\x00\x01" * 240


def describe_live():
    async def it_answers_each_turn_through_the_steps_and_records_it():
        voice = Voice(calls(("clock", {})), AssistantMessage("It is noon", audio=Audio.pcm16(b"\x02\x03")))

        @tool
        def clock() -> str:
            return "12:00"

        @agent(provider=voice, capabilities=[clock])
        async def assistant(message: str) -> str: ...

        thread = assistant.thread("u1")
        async with thread.live() as call:
            await call.hear(FRAME)
            await call.hear(FRAME)
            events = [e async for e in call.turn()]

        heard = UserMessage("[audio]", audio=(Audio.pcm16(FRAME * 2),))
        assert voice.heard == [FRAME, FRAME]
        assert voice.commits == 1
        assert voice.delivered == [ToolResult("call-0", "clock", "12:00")]
        assert [t for (_, t) in voice.turns] == [("clock",), ("clock",)]
        assert events[-1] == AssistantMessage("It is noon", audio=Audio.pcm16(b"\x02\x03"))
        assert AudioDelta(b"\x02\x03") in events
        assert thread.history == (heard, calls(("clock", {})), ToolResult("call-0", "clock", "12:00"), events[-1])

    async def it_runs_the_middleware_on_every_turn():
        voice = Voice(AssistantMessage("one"), AssistantMessage("two"))
        seen: list[int] = []
        requests: list[Request] = []

        @context
        def counter() -> str:
            seen.append(len(seen))
            return f"Turn {len(seen)}."

        async def record(request: Request, next: Next) -> AsyncIterator[Event]:
            requests.append(request)
            async for event in next(request):
                yield event

        @agent(provider=voice, system="Be brief.", capabilities=[counter, record])
        async def assistant(message: str) -> str: ...

        async with assistant.thread().live() as call:
            await call.hear(FRAME)
            [e async for e in call.turn()]
            await call.show(Image(b"\x89PNG", "image/png"))
            [e async for e in call.turn()]

        assert [s for (s, _) in voice.turns] == ["Be brief.\n\nTurn 1.", "Be brief.\n\nTurn 2."]
        assert voice.shown == [Image(b"\x89PNG", "image/png")]
        assert requests[1].messages[2] == UserMessage("[audio]", (Image(b"\x89PNG", "image/png"),), (Audio.pcm16(b""),))
        assert requests[0].media == ("audio",)

    async def it_forwards_interruptions():
        voice = Voice()

        @agent(provider=voice)
        async def assistant(message: str) -> str: ...

        async with assistant.thread().live() as call:
            await call.interrupt(1_200)

        assert voice.interrupted == [1_200]

    async def it_refuses_a_provider_without_a_realtime_connection():
        @agent(provider=Scripted())
        async def assistant(message: str) -> str: ...

        with pytest.raises(TypeError, match="realtime"):
            async with assistant.thread().live():
                pass


def describe_converse():
    QUIET, LOUD = b"\x00\x00" * 2400, b"\x10\x00" * 2400  # 100 ms frames
    _speech = LOUD.__eq__

    async def _frames(*batches: Sequence[bytes]) -> AsyncIterator[bytes]:
        """Each batch is fed at once; a pause lets the reply in progress run between batches."""
        for batch in batches:
            for frame in batch:
                yield frame
            await asyncio.sleep(0.05)

    async def it_turns_speech_bounded_by_silence_into_turns():
        voice = Voice(AssistantMessage("one", audio=Audio.pcm16(b"\x01\x02")), AssistantMessage("two"))

        @agent(provider=voice)
        async def assistant(message: str) -> str: ...

        thread = assistant.thread()
        async with thread.live() as call:
            heard = _frames([QUIET] * 5 + [LOUD] * 3 + [QUIET] * 8, [LOUD] * 2 + [QUIET] * 8)
            events = [e async for e in call.converse(heard, speech=_speech)]

        assert voice.commits == 2
        assert voice.heard[:14] == [QUIET] * 3 + [LOUD] * 3 + [QUIET] * 8  # 300 ms of pre-roll, speech, the quiet
        assert voice.heard[14:] == [LOUD] * 2 + [QUIET] * 8
        assert [e for e in events if isinstance(e, AssistantMessage)] == voice.replies_sent
        assert [type(m).__name__ for m in thread.history] == ["UserMessage"] * 1 + ["AssistantMessage"] + [
            "UserMessage",
            "AssistantMessage",
        ]

    async def it_interrupts_the_reply_when_spoken_over_and_opens_the_next_turn():
        voice = Voice(AssistantMessage("a long story", audio=Audio.pcm16(b"\x01\x02" * 2400)), AssistantMessage("ok"))
        voice.slow = True

        @agent(provider=voice)
        async def assistant(message: str) -> str: ...

        async with assistant.thread().live() as call:
            heard = _frames([LOUD] * 2 + [QUIET] * 8, [LOUD] * 4, [QUIET] * 8)
            events = [e async for e in call.converse(heard, speech=_speech)]

        assert voice.interrupted == [100]
        assert Interrupted(100) in events
        assert voice.heard[10:] == [LOUD] * 4 + [QUIET] * 8  # the interrupting speech opens the next turn
        assert voice.commits == 2

    def it_calibrates_on_the_first_second_of_room_noise():
        level = Level()
        room = (b"\x05\x00" * 2400, b"\x05\x00" * 2400)

        assert [level(frame) for frame in room * 5] == [False] * 10
        assert level.threshold == 300.0
        assert level(b"\x00\x02" * 2400)  # 512 > 300
        assert not level(b"\x10\x00" * 2400)  # 16


class Voice(Provider, LiveProvider, Session):
    """A fake realtime connection: replies in order and records what the thread sent it."""

    def __init__(self, *replies: AssistantMessage) -> None:
        self.replies = list(replies)
        self.replies_sent: list[AssistantMessage] = []
        self.slow = False
        self.heard: list[bytes] = []
        self.shown: list[Image] = []
        self.commits = 0
        self.turns: list[tuple[str, tuple[str, ...]]] = []
        self.delivered: list[ToolResult] = []
        self.interrupted: list[int] = []

    async def complete(self, request: Request) -> AsyncIterator[Delta | AssistantMessage]:
        raise AssertionError("a live thread never calls complete")
        yield

    @asynccontextmanager
    async def connect(self) -> AsyncIterator[Session]:
        yield self

    async def hear(self, pcm: bytes) -> None:
        self.heard.append(pcm)

    async def show(self, image: Image) -> None:
        self.shown.append(image)

    async def commit(self) -> None:
        self.commits += 1

    async def respond(self, system: str, tools: Sequence[Tool]) -> AsyncIterator[Delta | AssistantMessage]:
        self.turns.append((system, tuple(t.name for t in tools)))
        reply = self.replies.pop(0)
        self.replies_sent.append(reply)
        if reply.content:
            yield TextDelta(reply.content)
        if reply.audio:
            yield AudioDelta(reply.audio.data[44:])  # the PCM, as a server streams it
            if self.slow:
                await asyncio.sleep(0.2)
        yield reply

    async def deliver(self, result: ToolResult) -> None:
        self.delivered.append(result)

    async def interrupt(self, heard_ms: int) -> None:
        self.interrupted.append(heard_ms)

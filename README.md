# virtus

A small async framework for LLM agents. An agent is a function: its docstring is the prompt, its
parameters fill the prompt in, and its return annotation decides what you get back. Everything else,
tools, memory, guardrails, compaction, delegation, is a *capability* passed in a list.

```bash
uv add virtus                 # OpenAI-compatible providers included
uv add "virtus[anthropic]"    # plus the Anthropic provider
```

Requires Python 3.12. Not yet stable; the API may change between minor versions.

## Hello

```python
import asyncio

from virtus import agent
from virtus.providers import OpenAI


@agent(provider=OpenAI())
async def haiku(topic: str) -> str:
    """Write a haiku about {topic}."""
    ...


async def main() -> None:
    print(await haiku("the sea"))


asyncio.run(main())
```

`OpenAI()` reads `OPENAI_API_KEY`; `base_url=` points it at OpenRouter, DeepSeek, Ollama or any
compatible server. `Anthropic()` reads `ANTHROPIC_API_KEY`.
`agent.replace(provider=..., system=..., capabilities=..., journal=...)` is a copy of an agent with
those settings changed; `agent.extend(tool, ...)` is a copy with more capabilities.

## What a call returns

The return annotation picks the mode. Anything that isn't `str` is requested from the model as
structured output and validated with pydantic.

```python
async def hello() -> Person: ...                   # await hello()              -> Person
async def hello() -> AsyncIterator[str]: ...       # async for text in hello()  -> the answer as it streams
async def hello() -> AsyncIterator[Movie]: ...     # async for m in hello()     -> each Movie as it is completed
async def hello() -> Snapshot[Profile]: ...        # async for p in hello()     -> one Profile, refilled as it grows
async def hello() -> AsyncIterator[Event]: ...     # async for e in hello()     -> raw events: deltas, messages, tool results
```

```python
class Movie(BaseModel):
    title: str
    year: int
    director: str


@agent(provider=OpenAI())
async def recommend(genre: str, count: int = 3) -> list[Movie]:
    """Recommend {count} classic {genre} movies."""
    ...
```

## Tools

Any function, sync or async, becomes a tool. Its signature is the schema and its docstring the
description. Tool calls in one reply run in parallel; exceptions go back to the model as errors.

```python
@tool
def population(city: str) -> int:
    """Population of a city."""
    return {"tokyo": 37_400_000, "delhi": 32_900_000}.get(city.lower(), 0)


@agent(provider=OpenAI(), capabilities=[population])
async def compare(first: str, second: str) -> str:
    """Compare the populations of {first} and {second} in two sentences."""
    ...
```

Related tools can share state in a `Capabilities` class. Tool names get the class name as a prefix
(`todo__add`), and `@context` members add text to the system prompt on every call.

```python
class Todo(Capabilities):
    def __init__(self) -> None:
        self.items: dict[str, bool] = {}

    @context
    def summary(self) -> str:
        return f"The user has {len(self.items)} tasks."

    @tool
    def add(self, task: str) -> str:
        """Add a task to the list."""
        self.items[task] = False
        return f"added {task!r}"
```

## Threads

A thread is a conversation kept across calls. Its history lives in the agent's journal, in memory by
default. Calls on one conversation run one at a time.

```python
thread = assistant.thread("user-42")      # the conversation with this id, new or under way
await thread("I need to buy milk.")
await thread("What's left?")
thread.history                            # the messages as the provider saw them
thread.dispose()                          # forget it
```

With a `JsonJournal`, threads outlive the process: every completed step is written to disk,
`thread(id)` reads it back, and `resume()` finishes a run that was interrupted (or repeats the last
answer).

```python
@agent(provider=OpenAI(), journal=JsonJournal("./threads"))
async def assistant(message: str) -> str: ...

thread = assistant.thread("user-42")      # same history as last time
await thread.resume()                     # pick up where the crash left it
```

## Delegation

An agent in another agent's capabilities becomes a tool. The calling model may name a thread to
keep talking to the same sub-agent; those threads belong to the caller's thread.

```python
@agent(provider=OpenAI(), system="You are a researcher.")
async def researcher(topic: str) -> str:
    """Research {topic}."""
    ...


@agent(provider=OpenAI(), capabilities=[researcher])
async def lead(request: str) -> str: ...
```

## Memory and knowledge

`knowledge(...)` gives the agent read-only documents (`.md`, `.txt`, directories of them);
`memory(...)` gives it a folder of markdown notes it writes and reads back in later conversations.
Both scale retrieval to the volume: inline while it fits, then an index plus `read`, then BM25
`search`. No extra dependencies.

```python
@agent(provider=OpenAI(), capabilities=[knowledge("docs/"), memory("./notes")])
async def support(message: str) -> str: ...
```

## Hooks and middleware

`@on_input` rewrites or rejects each new user message; `@on_message` sees every complete message.
Rewrites become the history, so a redaction holds for the rest of the conversation.

```python
@on_input
def redact_cards(message: UserMessage) -> UserMessage:
    return UserMessage(CARD.sub("[card number]", message.content))
```

Under all of these is one primitive: a middleware wrapping each step (one provider call plus the tool
calls it requested). Raw middleware can time, retry, stop the run or swap the provider.

```python
async def timing(request: Request, next: Next) -> AsyncIterator[Event]:
    started = perf_counter()
    async for event in next(request):
        yield event
    print(f"step {request.iteration} took {perf_counter() - started:.2f}s")
```

`compaction(max_tokens=32_000, keep=8)` is a middleware that summarizes older messages with the same
provider once the conversation grows past the limit.

## Images and audio

An `Image` or `Audio` parameter is attached to the user message and sent to the model. A tool that
returns one, alone or in a list, sends it back the same way. An agent declared `-> Image` or
`-> Audio` asks the model to draw or speak its answer; that needs a model with the modality, such as
`openai/gpt-audio-mini` or Gemini's image models, and is not available on the Anthropic provider.

```python
@agent(provider=OpenAI())
async def colour(picture: Image) -> str:
    """What colour is this image? Answer with one word."""
    ...


await colour(Image.read("photo.png"))


@tool
def screenshot() -> Image:
    """Capture the screen."""
    return Image.read(capture())


@agent(provider=OpenAI("gpt-audio-mini", voice="echo"))
async def speak(text: str) -> Audio:
    """Say this out loud: {text}"""
    ...
```

## Voice

With a realtime model, a thread can be talked to. The caller owns the microphone and the speaker:
it streams the user's speech as 24 kHz 16-bit mono PCM and plays the audio that comes back. Each
turn runs the same middleware and tools as a typed message and lands in the history the same way.
Needs the `realtime` extra (`uv add "virtus[realtime]"`); works against OpenAI's `gpt-realtime`
models and oMLX's `/v1/realtime`.

```python
@agent(provider=OpenAI("gpt-realtime-2.1-mini"), capabilities=[clock])
async def assistant(message: str) -> str: ...


async with assistant.thread("user-42").live() as call:
    async for event in call.converse(microphone()):   # AsyncIterator[bytes] of PCM frames
        match event:
            case AudioDelta(data=pcm): speaker.write(pcm)
            case Interrupted(): speaker.abort()         # the user talked over the reply
            case TextDelta(content=text): ...           # the transcript
```

`converse` starts a turn when it detects speech, ends it after 800 ms of quiet and interrupts a reply
the user talks over; `speech=` swaps the level-based detector for your own. Underneath are
`call.hear(frame)`, `call.turn()` and `call.interrupt(heard_ms)`, for callers that decide the turns.

## Examples

`examples/` has one runnable script per feature, numbered in the order above. Each needs an
`OPENAI_API_KEY`; `13_providers.py` also needs an `ANTHROPIC_API_KEY`, `17_audio.py` and
`18_image_generation.py` reach their models through OpenRouter with an `OPENROUTER_API_KEY`, and
`19_realtime_voice.py` needs the `realtime` extra, `sounddevice` (in the dev group) for the microphone and
speaker, and a realtime model.

## Development

```bash
uv sync --all-extras
uv run ruff check . && uv run pyrefly check && uv run pytest -q    # offline suite
OPENAI_API_KEY="$OPENAI_API_KEY" uv run pytest -q -m integration   # real API
```

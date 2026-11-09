# virtus

A small library for building agents. Virtus allows you to describe agents as regular functions.

```bash
uv add virtus
```

Python 3.12. The API is still subject to change.

## Start with a function

```python
import asyncio

from virtus import agent
from virtus.providers import ChatCompletions


@agent(provider=ChatCompletions())
async def researcher(question: str) -> str:
    """Research this question: {question}"""
    ...


async def main() -> None:
    print(await researcher("What is retrieval-augmented generation?"))


asyncio.run(main())
```

With `OPENAI_API_KEY` set, this sends the request to OpenAI and returns the answer as a string.
The function is a template: its docstring supplies the message, `{question}` is filled from
the argument, and the return type tells Virtus what to give back. `ChatCompletions` handles
the model API. You can set a system prompt with `system=` on the decorator.

## Return the findings as data

```python
from pydantic import BaseModel


class Report(BaseModel):
    summary: str
    findings: list[str]


@agent(provider=ChatCompletions())
async def review(question: str) -> Report:
    """Explain this research question: {question}"""
    ...
```

`await review(...)` returns a `Report`, with fields you can use in your application.
Virtus requests structured JSON from the model and validates it with Pydantic. Other
structured types, including lists, work the same way.

## Give the researcher a search tool

To ground the answer in papers, give the researcher a search tool:

```python
from virtus.tools import arxiv


@agent(provider=ChatCompletions(), capabilities=[arxiv()])
async def searching_researcher(question: str) -> str:
    """Research this question: {question}"""
    ...
```

Now the model can search arXiv before answering. Virtus executes the function and
sends the result back, then calls the model again. That repeats until the model answers
without requesting another tool. `arxiv()` returns a tool built with `tool(...)` from a function: its
signature supplies the tool schema and its docstring describes what it does. Its arguments, such as
`arxiv(count=10)`, set the defaults and how it reaches the network. You can turn your own sync or async
functions into tools with the `@tool` decorator. Calls requested together run
in parallel, and exceptions are returned to the model as tool errors.

The `capabilities` list is where you give the agent tools and other behavior. Here it contains
just the search tool.

## Follow the request through middleware

Before giving the researcher more to do, we can watch those steps:

```python
from collections.abc import AsyncIterator
from time import perf_counter

from virtus import Event, Next, Request


async def timing(request: Request, next: Next) -> AsyncIterator[Event]:
    started = perf_counter()
    async for event in next(request):
        yield event
    print(f"Step {request.iteration}: {perf_counter() - started:.2f}s")


@agent(provider=ChatCompletions(), capabilities=[arxiv(), timing])
async def traced_researcher(question: str) -> str:
    """Research this question: {question}"""
    ...
```

If the model calls `arxiv`, `timing` runs around that provider call and the tool's
execution. It runs again around the next provider call, when the model sees the papers and
writes its answer. Each of these is one step.

This function is middleware. It receives a `Request`, passes it to `next`, and yields the
events that come back. Code before `next` can inspect or change the request; code around
the event stream can inspect or change the response. It can also stop a run, retry a step,
or choose a different provider.

The search tool enters the run through middleware too. Its middleware adds the tool to
the request before passing it on. Written out for this example, that looks like:

```python
from dataclasses import replace


async def add_search(request: Request, next: Next) -> AsyncIterator[Event]:
    request = replace(request, tools=(*request.tools, arxiv()))
    async for event in next(request):
        yield event


@agent(provider=ChatCompletions(), capabilities=[add_search, timing])
async def manual_researcher(question: str) -> str:
    """Research this question: {question}"""
    ...
```

`manual_researcher` makes the same tool available as `traced_researcher`. In both cases the tool is
added first, so the timing middleware receives a request that already contains it. The
`capabilities` list forms a chain in the order given, outermost first, and events travel back
through that chain.

All capabilities use this mechanism. Tools add tool definitions, context adds to the system
prompt, memory supplies documents and tools, and compaction rewrites the history. Even an
agent passed to another agent becomes middleware that offers delegation. A capability is
an object whose `middleware()` method supplies one or more middleware functions; plain
middleware functions can go in the same list.

This is also where you customize execution. For example, checking `request.iteration` before
calling `next` lets you impose a step limit. Replacing `request.provider` lets you switch
models during a run. The [middleware example](examples/08_middleware.py) shows both.

For everyday tracing, [`observe()`](examples/21_observe.py) provides a live tree on stderr
with provider calls, tool arguments and results, streamed answers, timing and token usage.
Like our timer, it sees the request at its position in the chain.

## Build on what you have already read

The researcher can find papers, but you may already have a collection you want it to consult.
Add papers saved as PDF, Markdown or text, and somewhere to keep notes between research sessions:

```python
from virtus import knowledge, memory


@agent(
    provider=ChatCompletions(),
    capabilities=[arxiv(), knowledge("./papers"), memory("./notes")],
)
async def informed_researcher(question: str) -> str:
    """Research this question: {question}"""
    ...
```

The next request passes through the tool middleware, then through `knowledge` and `memory`.
The model receives the saved papers and any notes that already exist. It also gets tools
to write or delete notes, so findings from one question can inform the next.

Both stores work with PDF, Markdown and text files. A PDF is read as its pages' text, with OCR
on scanned pages (the `pdf` extra). While their content fits `max_tokens`, it goes in the prompt.
Beyond that they supply an index and a read tool, then search and read tools when the index
becomes too large. Long documents are cut into passages of up to `passage=2_000` characters that overlap by `overlap=200`:
search returns passages (`paper.pdf#7`), and reading a long document gives one passage at a time
with a pointer to the next. Search uses BM25 by default, matching words in the query against the
passages. If a question uses different terms from the papers, embeddings let
you search by meaning:

```python
from virtus.providers import Embeddings


@agent(
    provider=ChatCompletions(),
    capabilities=[knowledge("./papers", embedder=Embeddings("text-embedding-3-small"))],
)
async def librarian(question: str) -> str:
    """Answer from the saved papers: {question}"""
    ...
```

When the collection needs search, the embedder turns the query and passages into vectors,
and the store ranks them by cosine similarity. Passage vectors are cached in `.embeddings.json`
beside the source files, so unchanged passages are reused across searches. The query is
embedded on each search. `memory` accepts the same `embedder=` option.

`Embeddings` uses OpenAI's `/v1/embeddings` API or a compatible `base_url`. `dimensions=` can
shorten vectors on models that support it; `input_type=True` sends Voyage's query/document
hint. You can also provide your own `Embedder`. For multiple collections, `name=` separates
their tools; `Store` exposes loading, reading and async search directly.
See [memory and knowledge](examples/12_memory_and_knowledge.py).

If the researcher should organize files or write a report to disk, add
[`workspace(path)`](examples/24_workspace.py). It offers read, write, exact edit, list and
regex search tools confined to that directory.

For context held in your application, a `@context` function can return text to add to each
step's system prompt. When that context and several tools share state, put them in a
[`Capabilities` class](examples/06_capabilities_class.py): its decorated methods share `self`,
and tool names get a prefix such as `notebook__save`.

## Follow up on a question

```python
async def investigate() -> None:
    thread = informed_researcher.thread("rag-review")
    print(await thread("Find papers on retrieval-augmented generation."))
    print(await thread("Which of these evaluate factual accuracy?"))
```

A direct call starts a fresh conversation. Here, both calls belong to the same thread, so
the second includes the first exchange. Each turn still goes through the middleware chain,
and the messages that reach the provider become the thread's history. If middleware redacts
or summarizes a message, later turns retain that change.

For input validation or redaction, `@on_input` wraps a function that rewrites a new
`UserMessage` or rejects it by raising. `@on_message` processes complete assistant messages
and tool results. These are middleware adapters too, and both accept sync or async functions.
See [hooks](examples/07_hooks.py).

To return to the investigation after restarting the process, give the researcher a disk-backed journal:

```python
from virtus import JsonJournal


@agent(
    provider=ChatCompletions(),
    capabilities=[arxiv(), knowledge("./papers"), memory("./notes")],
    journal=JsonJournal("./threads"),
)
async def persistent_researcher(question: str) -> str:
    """Research this question: {question}"""
    ...


thread = persistent_researcher.thread("rag-review")
```

`MemoryJournal` is the default. `JsonJournal` saves each completed step under
`<directory>/<agent>/<id>.json`; reopening the thread restores its history, and
`await thread.resume()` continues an interrupted run. Journals persist messages; state held
by a capability instance needs its own storage. See [persistence](examples/16_persistence.py).

Calls on one thread run one at a time. You can inspect `thread.history` and `thread.usage`,
or delete the conversation and its children with `thread.dispose()`. Each `AssistantMessage`
also carries the usage reported for that provider call. When research leads to long exchanges,
[`compaction(max_tokens=32_000, keep=8)`](examples/10_compaction.py) summarizes older messages
with the same provider while retaining recent ones.

## Stream the answer

To display the answer as it arrives, use a streaming return annotation:

| Annotation | What you iterate over |
| --- | --- |
| `AsyncIterator[str]` | Text chunks. |
| `AsyncIterator[Report]` | Completed items in a list. |
| `Snapshot[Report]` | A growing report; unfinished fields may be `None`. |
| `AsyncIterator[Event]` | Raw deltas, assistant messages and tool results. |

`AsyncIterator` comes from `collections.abc`; `Snapshot` and `Event` come from `virtus`.
These all consume the events coming back through the same middleware chain. See
[streaming](examples/04_streaming.py) and [structured streaming](examples/11_structured_streaming.py).

## Bring in other tools and agents

To search beyond arXiv, add `openalex` for published papers or `websearch` for the wider web.
`pdf` reads a paper from a URL or a local path, a page range at a time, so the researcher can go past the
abstract (see [reading PDFs](#read-the-papers-themselves)). `virtus.tools` also includes `calculator`,
`http` and `wikipedia`. Web search covers pages, news, images, videos and books with date filters; it and
`pdf` are the ones that need an extra. Each is a function that returns the tool, and its arguments set
the defaults the model sees and everything else: `pdf(max_chars=50_000, reader=Reader(scale=3))`,
`websearch(backend="google, bing")`, and `net=Net(timeout=60, retries=3)` on those that go online.
See [the toolbox](examples/23_toolbox.py).

To use tools from an MCP server, open `async with mcp(name=server) as tools:` and add `tools`
to the agent's capabilities. Servers can be stdio commands, streamable HTTP URLs,
`Stdio(command, env=...)` or `Http(url, headers=..., auth=...)`. Their tools are named
`name__tool`; connection credentials stay outside the model's context. The
[MCP example](examples/26_mcp.py) shows the connection lifecycle.

As that collection grows, [`lazy(*capabilities)`](examples/27_lazy_tools.py) offers tool search
so the model can discover what it needs. Discovered tools stay available in the conversation.
Anthropic and Responses use native deferred loading; Chat Completions adds discovered tools
to subsequent requests. If a tool needs a token, add [`secrets(**values)`](examples/25_vault.py)
after the tools it guards. It substitutes `{{secrets.name}}` before execution and redacts
the value from the result.

A second agent can review the findings before you use them:

```python
@agent(provider=ChatCompletions())
async def critic(report: str) -> str:
    """Review this research report for unsupported claims: {report}"""
    ...


@agent(provider=ChatCompletions(), capabilities=[arxiv(), critic])
async def team(question: str) -> str:
    """Research this question and ask the critic to review your findings: {question}"""
    ...
```

Adding the critic gives the researcher a `dispatch(to, arguments)` tool. It can address `critic`
for a one-off review or `critic:topic` to keep a conversation, using the critic's function arguments.
Multiple addresses run in parallel, with replies or errors keyed by address. Child conversations
live under the caller's thread in its journal; you can also pass an existing thread as a
capability. Dispatch accepts text and structured arguments, but cannot pass media parameters.
See [delegation](examples/09_delegation.py) and [a research team](examples/22_academic_researcher.py).

## Reuse an agent

Once an agent is configured, you can build variants without repeating its definition:

```python
from virtus import observe
from virtus.providers import Responses


observed_researcher = searching_researcher.extend(observe())
responses_researcher = searching_researcher.replace(provider=Responses())
```

`extend` appends capabilities; `replace` changes settings such as `provider`, `system`,
`capabilities` or `journal`. Both return a new agent, leaving the original unchanged, and keep
its journal unless you replace it. Here, one variant adds tracing and the other uses Responses
with the same prompt and tools.

## Change the model or the transport

Providers accept model names, credentials and generation settings; the built-in choices are:

| From `virtus.providers` | API and credentials |
| --- | --- |
| `ChatCompletions(model, ...)` | OpenAI or a compatible `base_url`; `OPENAI_API_KEY`. |
| `Responses(model, ...)` | OpenAI Responses API; `OPENAI_API_KEY`. |
| `Anthropic(model, ...)` | Anthropic Messages API; `ANTHROPIC_API_KEY`. |
| `openrouter(model)` / `omlx(model)` | Chat Completions presets; `OPENROUTER_API_KEY` / optional `OMLX_API_KEY`. |
| `Embeddings(model, ...)` | OpenAI-compatible `/v1/embeddings`; `base_url`, `dimensions`, optional `input_type=True` for Voyage. |

To supply your own integrations, implement the protocols:
[`Provider.complete(request)`](src/virtus/provider.py) streams deltas followed by one
`AssistantMessage`, `Embedder.embed(texts, kind)` produces vectors, and
[`Journal`](src/virtus/journal.py) stores history. `LiveProvider` and `Session` define realtime
connections. These supply the model and storage used by a run; capabilities supply the
middleware around its steps.

## Ask about a figure

```python
from virtus import Image


@agent(provider=ChatCompletions())
async def explain_figure(figure: Image, question: str) -> str:
    """Answer {question} about this figure: {figure}"""
    ...


async def inspect_figure() -> None:
    print(await explain_figure(Image.read("figure.png"), "What do the axes represent?"))
```

The image is attached to the user message; `{figure}` appears as `[image]` in the prompt.
The text question and the image reach the model together. All three chat providers accept
image input, and tools can return `Image` values too. See [images](examples/14_images.py).

The return annotation works in the other direction as well. With a model that generates
images, the researcher can turn an explanation into a diagram:

```python
from virtus.providers import openrouter


@agent(provider=openrouter("google/gemini-3.1-flash-lite-image"))
async def illustrate(description: str) -> Image:
    """Draw a diagram of this process: {description}"""
    ...
```

`await illustrate(...)` returns an `Image` with `data` and `media_type` fields. This example
uses OpenRouter and needs `OPENROUTER_API_KEY`. Image generation goes through `ChatCompletions`
with a model that supports it; see [image generation](examples/18_image_generation.py).

## Read the papers themselves

A `File` holds a PDF. Like an image, it can be an argument or a tool's return value:

```python
from virtus import File


@agent(provider=Responses())
async def summarize_paper(paper: File) -> str:
    """Summarize the method in {paper}."""
    ...


async def read_paper() -> None:
    print(await summarize_paper(File.read("paper.pdf")))
```

Anthropic, Responses and Chat Completions send the file itself, and the model reads its layout and
figures as well as its text. A `File` can also carry `text`, which goes in its place to a server whose
models read no files (`omlx()`, or `ChatCompletions(..., files=False)`) and to the Realtime API. A file
with no text is refused there.

The `pdf` tool returns the requested pages as a `File` along with their text. The text comes from the
PDF's text layer, or from OCR (RapidOCR) on scanned pages, so the fallback works on scans too.

## Listen to a talk, or hear the summary

```python
from virtus import Audio


@agent(provider=ChatCompletions("gpt-audio-mini"))
async def summarize_talk(recording: Audio) -> str:
    """Summarize the main findings and limitations discussed in {recording}."""
    ...


@agent(provider=ChatCompletions("gpt-audio-mini", voice="alloy"))
async def read_aloud(summary: str) -> Audio:
    """Read this summary aloud: {summary}"""
    ...


async def spoken_summary() -> Audio:
    summary = await summarize_talk(Audio.read("talk.wav"))
    return await read_aloud(summary)
```

The first call attaches the recording to the message and returns text. The second asks for
a spoken answer, returned as an `Audio` value containing WAV bytes in `data`. Audio input
and output require an audio-capable model through `ChatCompletions`; tools can return audio
too. See [audio](examples/17_audio.py) for saving and reusing a generated clip.

## Discuss the findings in realtime

For a conversation where you can speak, hear the answer and interrupt it, open a live thread:

```python
from collections.abc import AsyncIterator, Callable

from virtus import AudioDelta, Interrupted, TextDelta


@agent(provider=ChatCompletions("gpt-realtime-2.1-mini"), capabilities=[arxiv()])
async def voice_researcher(question: str) -> str:
    """Discuss this research question: {question}"""
    ...


async def discuss(
    frames: AsyncIterator[bytes],
    play: Callable[[bytes], None],
    stop: Callable[[], None],
) -> None:
    async with voice_researcher.thread("discussion").live() as call:
        async for event in call.converse(frames):
            match event:
                case AudioDelta(data=pcm):
                    play(pcm)
                case Interrupted():
                    stop()
                case TextDelta(content=text):
                    print(text, end="", flush=True)
```

Your application supplies microphone frames and the functions that play or stop the speaker.
Audio in both directions is 24 kHz, 16-bit mono PCM. `converse` detects speech and pauses,
and emits `Interrupted` when the user talks over a reply so you can stop playback.

Each turn runs through the same middleware and tools as a typed conversation, and is saved
in the thread's history. This needs the `realtime` extra and a realtime model on OpenAI or
oMLX. Pass `speech=` for your own speech detector, or control turns directly with `hear`,
`turn` and `interrupt`; `show` adds an image. The [realtime voice example](examples/19_realtime_voice.py)
includes microphone and speaker setup.

## Development

Install the extras you use:

```bash
uv add "virtus[anthropic]"
uv add "virtus[mcp]"
uv add "virtus[websearch]"
uv add "virtus[pdf]"
uv add "virtus[realtime]"
```

Runnable scripts are in [examples/](examples/). To run the offline checks:

```bash
uv sync --all-extras
uv run ruff check .
uv run pyrefly check
uv run pytest -q
```

## License

[MIT](LICENSE.md).

"""Tools offered on demand, so a large toolbox doesn't fill every request; see `lazy`."""

import math
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial, reduce
from typing import Annotated

from pydantic import BaseModel, Field, create_model

from virtus.capability import flatten
from virtus.message import Event, ToolCall, ToolResult
from virtus.middleware import Capability, Middleware, Next
from virtus.request import Request
from virtus.tool import Tool

SEARCH = "tool_search"
"""The search tool's name; the Responses provider turns calls to it into its own `tool_search_call` items."""

HEADER = f"Tools you can find with `{SEARCH}`:"

DESCRIPTION = (
    "Find tools by what they do; names, descriptions and parameters are searched. "
    "The tools found are loaded and usable from your next step on."
)

STOPWORDS = frozenset(
    ["the", "and", "for", "with", "from", "that", "this", "into", "when", "what", "how", "you", "your", "its", "are"]
)


@dataclass(frozen=True)
class Ranking:
    """How `find` scores tools: BM25 with `k1` and `b` over words of `shortest` characters or more that aren't
    `stopwords`, keeping only tools scoring at least `cutoff` times the best."""

    k1: float = 1.5
    b: float = 0.75
    cutoff: float = 0.5
    shortest: int = 3
    stopwords: frozenset[str] = STOPWORDS


def lazy(
    *capabilities: Capability | Middleware,
    header: str = HEADER,
    description: str = DESCRIPTION,
    limit: int = 5,
    ranking: Ranking = Ranking(),
) -> Middleware:
    """Offer these capabilities' tools on demand instead of on every request.

    The tools go in `Request.deferred`: the model sees their names in the system prompt and a `tool_search`
    tool, and what a search finds is usable from the next step on, for the rest of the conversation. The
    Messages API loads deferred tools natively, so the request prefix stays cacheable; Chat Completions
    gets the loaded ones added. Anything that goes in `capabilities=` goes here: tools, `Capabilities`
    classes, `workspace(...)`, MCP servers.

    `header` precedes the names in the system prompt, `description` is the search tool's, `limit` the
    default for how many tools a search loads, and `ranking` how it scores them.
    """
    chain = flatten(capabilities)
    field = Field(limit, description="How many matching tools to load.", ge=1)
    query = create_model("Query", __base__=Query, limit=(int, field))

    async def lazily(request: Request, next: Next) -> AsyncIterator[Event]:
        outer = request.tools

        async def defer(inner: Request) -> AsyncIterator[Event]:
            added = tuple(t for t in inner.tools if all(t is not o for o in outer))
            note = f"{header} {', '.join(t.name for t in added)}" if added else ""
            system = "\n\n".join(part for part in (inner.system, note) if part)
            search = Search.over(added, description=description, query=query, ranking=ranking)
            tools = (*outer, search) if added else outer
            deferred = (*inner.deferred, *added)
            async for event in next(replace(inner, system=system, tools=tools, deferred=deferred)):
                yield event

        def link(inner: Next, middleware: Middleware) -> Next:
            return lambda request: middleware(request, inner)

        innermost: Next = defer
        step = reduce(link, reversed(chain), innermost)
        async for event in step(request):
            yield event

    return lazily


class Query(BaseModel):
    query: Annotated[str, Field(description="What the tool should do, in a few words.")]
    limit: Annotated[int, Field(description="How many matching tools to load.", ge=1)] = 5


@dataclass(frozen=True)
class Search(Tool):
    """The `tool_search` tool over a catalog of deferred tools: a hit loads the tool from the next step on."""

    catalog: tuple[Tool, ...]
    query: type[Query]
    ranking: Ranking

    @classmethod
    def over(
        cls,
        catalog: tuple[Tool, ...],
        *,
        description: str = DESCRIPTION,
        query: type[Query] = Query,
        ranking: Ranking = Ranking(),
    ) -> "Search":
        return cls(SEARCH, description, query, partial(find, catalog, ranking=ranking), catalog, query, ranking)

    async def execute(self, call: ToolCall) -> ToolResult:
        try:
            query = self.query.model_validate_json(call.arguments or "{}")
            found = find(self.catalog, query.query, query.limit, ranking=self.ranking)
        except Exception as error:
            return ToolResult(call.id, call.name, f"{type(error).__name__}: {error}", is_error=True)

        if not found:
            names = ", ".join(t.name for t in self.catalog)
            return ToolResult(call.id, call.name, f"No tools match {query.query!r}. The tools are: {names}")
        lines = "\n".join(f"- {t.name}: {t.description.partition(chr(10))[0]}" for t in found)
        text = f"Loaded, usable from your next step on:\n{lines}"
        return ToolResult(call.id, call.name, text, tools=tuple(t.name for t in found))


def find(catalog: Sequence[Tool], query: str, limit: int = 5, *, ranking: Ranking = Ranking()) -> list[Tool]:
    """The tools matching `query` best, by BM25 over name, description and parameter names and descriptions.

    Tools scoring under `ranking.cutoff` of the best match are left out: a loaded tool stays for the whole
    conversation, so a word shared in passing shouldn't load it.
    """
    k1, b = ranking.k1, ranking.b
    documents = [_terms(t, ranking) for t in catalog]
    if not documents:
        return []
    average = sum(map(len, documents)) / len(documents)

    def weight(term: str) -> float:
        containing = sum(term in document for document in documents)
        return math.log(1 + (len(documents) - containing + 0.5) / (containing + 0.5))

    def score(document: list[str]) -> float:
        length = k1 * (1 - b + b * len(document) / average)
        counts = ((term, document.count(term)) for term in set(_tokens(query, ranking)))
        return sum(weight(term) * count * (k1 + 1) / (count + length) for term, count in counts if count)

    ranked = sorted(zip(map(score, documents), catalog, strict=True), key=lambda pair: pair[0], reverse=True)
    best = ranked[0][0]
    return [tool for points, tool in ranked if points > 0 and points >= best * ranking.cutoff][:limit]


def _terms(tool: Tool, ranking: Ranking) -> list[str]:
    properties = tool.schema.get("properties")
    fields = properties.items() if isinstance(properties, Mapping) else ()
    about = (f"{name} {value.get('description', '')}" if isinstance(value, Mapping) else name for name, value in fields)
    return _tokens(" ".join((tool.name, tool.description, *about)), ranking)


def _tokens(text: str, ranking: Ranking) -> list[str]:
    """Function words and short ones are left out: those would match about everything."""
    words = re.findall(rf"[a-z0-9]{{{ranking.shortest},}}", text.lower())
    return [word for word in words if word not in ranking.stopwords]

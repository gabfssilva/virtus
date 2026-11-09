"""A small research team: a scout that searches arXiv and OpenAlex, a reader that takes notes on each paper, a
researcher that writes the report with both as tools, and a critic that reviews it until it approves.

`arxiv` and `openalex` come from `virtus.tools` and need no key (OpenAlex shares a daily budget per IP without one;
`OPENALEX_API_KEY` lifts that). Every agent carries `observe()`, so the whole team's work scrolls by on stderr.
"""

import asyncio
import json
import sys
from collections.abc import AsyncIterator
from dataclasses import replace

from pydantic import BaseModel

from virtus import Event, Middleware, Next, Request, agent, observe
from virtus.providers import openrouter
from virtus.tools import Paper, arxiv, openalex

model = openrouter("mercury-3")


def answer_after(steps: int) -> Middleware:
    """Take the tools away once the model has spent `steps` rounds on them, so the next call has to answer."""

    async def cap(request: Request, next: Next) -> AsyncIterator[Event]:
        async for event in next(request if request.iteration < steps else replace(request, tools=())):
            yield event

    return cap


class Notes(BaseModel):
    title: str
    claims: list[str]
    method: str
    limitations: list[str]


class Finding(BaseModel):
    claim: str
    source: str
    """Title of the paper that supports it."""


class Report(BaseModel):
    title: str
    summary: str
    findings: list[Finding]
    open_questions: list[str]
    sources: list[str]


class Critique(BaseModel):
    approved: bool
    issues: list[str]


@agent(
    provider=model,
    system="Search both arXiv and OpenAlex. Return the four most relevant papers, no duplicates, with their summaries.",
    capabilities=[arxiv(), openalex(), answer_after(4), observe()],
)
async def scout(topic: str) -> list[Paper]:
    """Find papers on {topic}."""
    ...


@agent(provider=model, system="Read carefully; never add claims the text does not make.", capabilities=[observe()])
async def reader(title: str, summary: str) -> Notes:
    """Take notes on this paper. Title: {title}. Abstract: {summary}"""
    ...


@agent(
    provider=model,
    system=(
        "You are a demanding reviewer. Approve when every finding names its paper, the summary is specific and the "
        "sources carry a link. Judge the report on what it says; do not ask for more searching."
    ),
    capabilities=[observe()],
)
async def critic(report: str) -> Critique:
    """Review this research report: {report}"""
    ...


@agent(
    provider=model,
    system=(
        "You lead a literature review. Dispatch to scout to find papers, then to reader for each of the three most "
        "relevant, then write the report. Every finding must name the paper that supports it; sources carry the "
        "paper's url."
    ),
    capabilities=[observe(), scout, reader, answer_after(8)],
)
async def researcher(message: str) -> Report: ...


async def main() -> None:
    question = sys.argv[1] if len(sys.argv) > 1 else "How are large language models used for tabular data imputation?"
    team = researcher.thread("review")

    report = await team(question)
    for _ in range(2):
        critique = await critic(report.model_dump_json())
        if critique.approved:
            break
        report = await team(f"The reviewer did not approve. Issues: {json.dumps(critique.issues)}. Revise the report.")

    print(f"\n# {report.title}\n\n{report.summary}\n")
    for finding in report.findings:
        print(f"- {finding.claim}  [{finding.source}]")
    print("\nOpen questions: " + "; ".join(report.open_questions))
    print("\nSources:\n" + "\n".join(f"- {s}" for s in report.sources))


asyncio.run(main())

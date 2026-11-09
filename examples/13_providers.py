"""Providers are interchangeable: the same agent runs on Chat Completions, the Responses API or Anthropic.

Anthropic needs the extra: `uv add "virtus[anthropic]"` and an `ANTHROPIC_API_KEY`.
"""

import asyncio

from virtus import agent, tool
from virtus.providers import Anthropic, ChatCompletions, Responses


@tool
def exchange_rate(currency: str) -> float:
    """Euros per unit of a currency."""
    return {"USD": 0.92, "GBP": 1.17, "BRL": 0.16}.get(currency.upper(), 1.0)


@agent(provider=ChatCompletions(), capabilities=[exchange_rate])
async def convert(amount: float, currency: str) -> str:
    """How many euros are {amount} {currency}? Answer in one sentence."""
    ...


async def main() -> None:
    others = (convert.replace(provider=Responses()), convert.replace(provider=Anthropic(effort="low")))
    for converter in (convert, *others):
        print(f"{converter.provider!r}: {await converter(100, 'BRL')}")


asyncio.run(main())

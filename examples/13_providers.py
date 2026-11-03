"""Providers are interchangeable: the same agent runs on OpenAI or Anthropic.

Anthropic needs the extra: `uv add "virtus[anthropic]"` and an `ANTHROPIC_API_KEY`.
"""

import asyncio

from virtus import Provider, agent, tool
from virtus.providers import Anthropic, OpenAI


@tool
def exchange_rate(currency: str) -> float:
    """Euros per unit of a currency."""
    return {"USD": 0.92, "GBP": 1.17, "BRL": 0.16}.get(currency.upper(), 1.0)


def converter(provider: Provider):
    @agent(provider=provider, capabilities=[exchange_rate])
    async def convert(amount: float, currency: str) -> str:
        """How many euros are {amount} {currency}? Answer in one sentence."""
        ...

    return convert


async def main() -> None:
    for provider in (OpenAI(), Anthropic(effort="low")):
        print(f"{provider!r}: {await converter(provider)(100, 'BRL')}")


asyncio.run(main())

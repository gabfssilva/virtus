"""Any return annotation other than `str` is requested from the model as structured output."""

import asyncio

from pydantic import BaseModel

from virtus import agent
from virtus.providers import ChatCompletions


class Movie(BaseModel):
    title: str
    year: int
    director: str


@agent(provider=ChatCompletions())
async def recommend(genre: str, count: int = 3) -> list[Movie]:
    """Recommend {count} classic {genre} movies."""
    ...


async def main() -> None:
    for movie in await recommend("science fiction"):
        print(movie)


asyncio.run(main())

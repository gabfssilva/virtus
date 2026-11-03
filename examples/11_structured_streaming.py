"""`AsyncIterator[T]` yields each item of a list as it completes; `Snapshot[T]` yields one object as it fills in."""

import asyncio
from collections.abc import AsyncIterator

from pydantic import BaseModel

from virtus import Snapshot, agent
from virtus.providers import OpenAI


class Movie(BaseModel):
    title: str
    year: int


class Profile(BaseModel):
    name: str
    occupation: str
    favorite_movies: list[Movie]


@agent(provider=OpenAI())
async def recommend(genre: str) -> AsyncIterator[Movie]:
    """Recommend five {genre} movies."""
    ...


@agent(provider=OpenAI())
async def profile(description: str) -> Snapshot[Profile]:
    """Build a profile from: {description}"""
    ...


async def main() -> None:
    async for movie in recommend("heist"):
        print(f"{movie.year}  {movie.title}")
    print()

    async for snapshot in profile("Gabriel, 34, software engineer, loves Heat, Inception and Ocean's Eleven"):
        print(snapshot.model_dump())


asyncio.run(main())

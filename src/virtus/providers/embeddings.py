from collections.abc import Sequence
from functools import cached_property

from openai import AsyncOpenAI, omit

from virtus.provider import Embedder, Kind


class Embeddings(Embedder):
    """`/v1/embeddings`, OpenAI's and any compatible server: Gemini's OpenAI endpoint, Voyage, oMLX, Ollama.

    `dimensions` shortens the vectors on models that allow it. `input_type=True` sends the kind of text as
    Voyage's `input_type`, which its models want to tell queries from documents; OpenAI's take no such
    hint and reject the field. Vectors are cached by whoever stores them (`knowledge()` does), not here.
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 600,
        max_retries: int = 2,
        dimensions: int | None = None,
        input_type: bool = False,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.input_type = input_type
        self._api_key = api_key
        self._base_url = base_url
        self._timeout = timeout
        self._max_retries = max_retries

    async def embed(self, texts: Sequence[str], kind: Kind) -> list[list[float]]:
        if not texts:
            return []
        response = await self._client.embeddings.create(
            model=self.model,
            input=list(texts),
            dimensions=omit if self.dimensions is None else self.dimensions,
            extra_body={"input_type": kind} if self.input_type else None,
        )
        return [item.embedding for item in sorted(response.data, key=lambda item: item.index)]

    @cached_property
    def _client(self) -> AsyncOpenAI:
        """Created on first use, so building an agent doesn't require credentials."""
        return AsyncOpenAI(
            api_key=self._api_key, base_url=self._base_url, timeout=self._timeout, max_retries=self._max_retries
        )

    def __repr__(self) -> str:
        return f"Embeddings({self.model!r})"

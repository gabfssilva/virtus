from types import SimpleNamespace

from openai import omit
from openai.types import CreateEmbeddingResponse, Embedding
from openai.types.create_embedding_response import Usage

from virtus.providers import Embeddings


def describe_embeddings():
    async def it_returns_one_vector_per_text_in_input_order():
        provider = Embeddings("text-embedding-3-small", dimensions=2)
        client = Replaying(
            [
                Embedding(embedding=[0.0, 1.0], index=1, object="embedding"),
                Embedding(embedding=[1.0, 0.0], index=0, object="embedding"),
            ]
        )
        provider.__dict__["_client"] = client

        assert await provider.embed(("a", "b"), "document") == [[1.0, 0.0], [0.0, 1.0]]
        assert client.calls == [
            {"model": "text-embedding-3-small", "input": ["a", "b"], "dimensions": 2, "extra_body": None}
        ]

    async def it_sends_the_kind_as_voyage_input_type_when_asked_and_skips_empty_input():
        provider = Embeddings("voyage-4", base_url="https://ai.mongodb.com/v1", input_type=True)
        client = Replaying([Embedding(embedding=[1.0], index=0, object="embedding")])
        provider.__dict__["_client"] = client

        assert await provider.embed([], "query") == []
        assert await provider.embed(["q"], "query") == [[1.0]]
        assert client.calls == [
            {"model": "voyage-4", "input": ["q"], "dimensions": omit, "extra_body": {"input_type": "query"}}
        ]


class Replaying:
    """Stands in for `AsyncOpenAI`: `embeddings.create` records its arguments and answers with the given items."""

    def __init__(self, data: list[Embedding]) -> None:
        self.embeddings = SimpleNamespace(create=self.create)
        self.data = data
        self.calls: list[dict[str, object]] = []

    async def create(self, **arguments: object) -> CreateEmbeddingResponse:
        self.calls.append(arguments)
        usage = Usage(prompt_tokens=1, total_tokens=1)
        return CreateEmbeddingResponse(data=self.data, model="test", object="list", usage=usage)

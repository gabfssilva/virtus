"""HTTP for the ready-made tools: stdlib only, a timeout, a descriptive user agent, a retry when rate limited."""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.request import Request, urlopen

AGENT = "virtus (https://github.com/gabfssilva/virtus)"


@dataclass(frozen=True)
class Response:
    status: int
    headers: Mapping[str, str]
    """Header names in lower case."""
    body: bytes


@dataclass(frozen=True)
class Net:
    """How the tools reach the network: `timeout` in seconds, `retries` after a 429 or a timeout, each after
    `pause` seconds (arXiv wants three between requests), and the `agent` sent as the user agent."""

    agent: str = AGENT
    timeout: float = 20
    retries: int = 1
    pause: float = 3

    def request(
        self, url: str, *, method: str = "GET", headers: Mapping[str, str] | None = None, body: bytes | None = None
    ) -> Response:
        """Any status is a response, not an error."""
        prepared = Request(url, data=body, headers={"User-Agent": self.agent, **(headers or {})}, method=method)
        try:
            with urlopen(prepared, timeout=self.timeout) as response:
                return Response(response.status, _lower(response.headers.items()), response.read())
        except HTTPError as error:
            return Response(error.code, _lower(error.headers.items()), error.read())

    def get(self, url: str, *, headers: Mapping[str, str] | None = None) -> bytes:
        """A successful body, retried on 429 or a timeout."""
        prepared = Request(url, headers={"User-Agent": self.agent, **(headers or {})})
        for attempt in range(self.retries + 1):
            try:
                with urlopen(prepared, timeout=self.timeout) as response:
                    return response.read()
            except (HTTPError, TimeoutError) as error:
                if attempt == self.retries or (isinstance(error, HTTPError) and error.code != 429):
                    raise
                time.sleep(self.pause)
        raise AssertionError("unreachable")


def clip(text: str, max_chars: int) -> str:
    """The text cut to `max_chars`, saying so and how long it was."""
    return text if len(text) <= max_chars else f"{text[:max_chars]}\n… [truncated; {len(text)} chars in total]"


def _lower(items: list[tuple[str, str]]) -> Mapping[str, str]:
    return {name.lower(): value for name, value in items}

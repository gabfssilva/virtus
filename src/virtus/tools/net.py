"""HTTP for the ready-made tools: stdlib only, a fixed timeout, a descriptive user agent."""

import time
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.request import Request, urlopen

AGENT = "virtus (https://github.com/gabfssilva/virtus)"
TIMEOUT = 20


@dataclass(frozen=True)
class Response:
    status: int
    headers: Mapping[str, str]
    """Header names in lower case."""
    body: bytes


def request(
    url: str, *, method: str = "GET", headers: Mapping[str, str] | None = None, body: bytes | None = None
) -> Response:
    """Any status is a response, not an error."""
    prepared = Request(url, data=body, headers={"User-Agent": AGENT, **(headers or {})}, method=method)
    try:
        with urlopen(prepared, timeout=TIMEOUT) as response:
            return Response(response.status, _lower(response.headers.items()), response.read())
    except HTTPError as error:
        return Response(error.code, _lower(error.headers.items()), error.read())


def get(url: str, *, headers: Mapping[str, str] | None = None) -> bytes:
    """A successful body; one retry after a pause on 429 or a timeout, as arXiv wants three seconds between requests."""
    prepared = Request(url, headers={"User-Agent": AGENT, **(headers or {})})
    for attempt in (1, 2):
        try:
            with urlopen(prepared, timeout=TIMEOUT) as response:
                return response.read()
        except (HTTPError, TimeoutError) as error:
            if attempt == 2 or (isinstance(error, HTTPError) and error.code != 429):
                raise
            time.sleep(3)
    raise AssertionError("unreachable")


def _lower(items: list[tuple[str, str]]) -> Mapping[str, str]:
    return {name.lower(): value for name, value in items}

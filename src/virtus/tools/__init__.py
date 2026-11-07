"""Ready-made tools; none imports its backend until it runs, so each extra is only needed by those who use it."""

from virtus.tools.calculator import calculator
from virtus.tools.http import http
from virtus.tools.papers import Paper, arxiv, openalex
from virtus.tools.websearch import websearch
from virtus.tools.wikipedia import Article, wikipedia

__all__ = ["Article", "Paper", "arxiv", "calculator", "http", "openalex", "websearch", "wikipedia"]

"""Ready-made tools; none imports its backend until it runs, so each extra is only needed by those who use it."""

from virtus.pdfs import Reader
from virtus.tools.calculator import calculator
from virtus.tools.http import http
from virtus.tools.net import Net
from virtus.tools.papers import Paper, arxiv, openalex
from virtus.tools.pdf import pdf
from virtus.tools.websearch import websearch
from virtus.tools.wikipedia import Article, wikipedia

__all__ = [
    "Article",
    "Net",
    "Paper",
    "Reader",
    "arxiv",
    "calculator",
    "http",
    "openalex",
    "pdf",
    "websearch",
    "wikipedia",
]

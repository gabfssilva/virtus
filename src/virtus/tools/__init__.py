"""Ready-made tools; none imports its backend until it runs, so each extra is only needed by those who use it."""

from virtus.tools.websearch import websearch

__all__ = ["websearch"]

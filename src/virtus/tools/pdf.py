from io import BytesIO
from pathlib import Path
from urllib.parse import urlparse

from virtus.message import File
from virtus.pdfs import Reader
from virtus.tool import Tool, tool
from virtus.tools.net import Net, clip


def pdf(*, net: Net = Net(), reader: Reader = Reader(), max_chars: int = 20_000) -> Tool:
    """The `pdf` tool, fetching through `net` and reading pages with `reader`; `max_chars` is how much text it
    returns unless the model asks for more."""

    def pdf(
        source: str, first_page: int = 1, last_page: int | None = None, max_chars: int = max_chars
    ) -> list[File | str] | str:
        """Read a PDF from a URL or a local path, from `first_page` to `last_page` (1-based, inclusive; the end by
        default). Scanned pages are read with OCR. Read a long document a range of pages at a time."""
        import pypdfium2 as pdfium  # the `pdf` extra

        remote = urlparse(source).scheme in ("http", "https")
        document = pdfium.PdfDocument(net.get(source) if remote else Path(source).expanduser().read_bytes())
        total = len(document)
        last = min(last_page or total, total)
        if not 1 <= first_page <= last:
            return f"No pages {first_page}-{last_page or total}; the document has {total}."

        indices = list(range(first_page - 1, last))
        text = "\n\n".join(f"--- page {i + 1} of {total} ---\n{reader.text(document[i])}" for i in indices)
        pages = pdfium.PdfDocument.new()
        pages.import_pages(document, indices)
        buffer = BytesIO()
        pages.save(buffer)
        name = Path(urlparse(source).path if remote else source).name or "document.pdf"
        return [
            File(buffer.getvalue(), name=name, text=clip(text, max_chars)),
            f"Pages {first_page}-{last} of {total}.",
        ]

    return tool(pdf)

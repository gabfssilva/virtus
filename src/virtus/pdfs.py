"""Text out of PDFs, for the `pdf` tool and `knowledge()`: the text layer, or OCR where a page has none.

Both need the `pdf` extra; pypdfium2 and RapidOCR are imported on first use.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pypdfium2 import PdfPage
    from rapidocr import RapidOCR

type Setting = str | int | float | bool


@dataclass(frozen=True)
class Reader:
    """Reads pages: a scanned one is rendered at `scale` times its size for RapidOCR, built with `ocr` as its
    `params` (e.g. `{"Det.limit_side_len": 1280}`)."""

    scale: float = 2
    ocr: Mapping[str, Setting] = field(default_factory=lambda: {"Global.log_level": "critical"})

    def pages(self, data: bytes) -> list[str]:
        """The text of every page, in order."""
        import pypdfium2 as pdfium

        document = pdfium.PdfDocument(data)
        return [self.text(document[i]) for i in range(len(document))]

    def text(self, page: "PdfPage") -> str:
        """The page's text layer, or what OCR reads off the rendered page when it has none."""
        from rapidocr.utils.output import RapidOCROutput

        if text := page.get_textpage().get_text_bounded().replace("\r\n", "\n").strip():
            return text
        match _ocr(tuple(sorted(self.ocr.items())))(page.render(scale=self.scale).to_numpy()):
            case RapidOCROutput(txts=lines):
                return "\n".join(lines or ())
            case _:
                return ""


@cache
def _ocr(params: tuple[tuple[str, Setting], ...]) -> "RapidOCR":
    """Loading the models takes seconds, so an engine is built once per settings, on the first scanned page."""
    from rapidocr import RapidOCR

    return RapidOCR(params=dict(params))

import json
from io import BytesIO
from pathlib import Path

import pypdfium2 as pdfium
import pytest
from PIL import Image, ImageDraw, ImageFont

from virtus import File, ToolCall, ToolResult
from virtus.tools import pdf
from virtus.tools.net import Net


def describe_pdf():
    async def it_reads_a_url_as_a_file_of_the_pages_with_their_text(monkeypatch: pytest.MonkeyPatch):
        fetched: list[str] = []
        monkeypatch.setattr(Net, "get", lambda self, url: fetched.append(url) or document("Attention", "Results"))

        result = await read({"source": "https://arxiv.org/pdf/1706.03762.pdf"})

        assert fetched == ["https://arxiv.org/pdf/1706.03762.pdf"]
        assert result.content == '["[file]","Pages 1-2 of 2."]'
        [file] = result.files
        assert (file.name, pages(file)) == ("1706.03762.pdf", 2)
        assert file.text == "--- page 1 of 2 ---\nAttention\n\n--- page 2 of 2 ---\nResults"

    async def it_reads_a_local_path_and_keeps_only_the_range(tmp_path: Path):
        (tmp_path / "notes.pdf").write_bytes(document("One", "Two", "Three"))

        result = await read({"source": str(tmp_path / "notes.pdf"), "first_page": 2, "last_page": 9})

        [file] = result.files
        assert (file.name, pages(file), result.content) == ("notes.pdf", 2, '["[file]","Pages 2-3 of 3."]')
        assert file.text == "--- page 2 of 3 ---\nTwo\n\n--- page 3 of 3 ---\nThree"

    async def it_clips_the_text_and_says_when_the_range_is_outside_the_document(tmp_path: Path):
        (tmp_path / "one.pdf").write_bytes(document("One"))
        source = str(tmp_path / "one.pdf")

        assert (await read({"source": source, "max_chars": 10})).files[0].text == (
            "--- page 1\n… [truncated; 23 chars in total]"
        )
        assert (await read({"source": source, "first_page": 4})).content == "No pages 4-1; the document has 1."

    async def it_reads_scanned_pages_with_ocr(tmp_path: Path):
        picture = Image.new("RGB", (1240, 300), "white")
        ImageDraw.Draw(picture).text(
            (60, 100), "Missing values in tables", fill="black", font=ImageFont.load_default(48)
        )
        picture.save(tmp_path / "scan.pdf")

        [file] = (await read({"source": str(tmp_path / "scan.pdf")})).files

        assert file.text == "--- page 1 of 1 ---\nMissing values in tables"


async def read(arguments: dict[str, object]) -> ToolResult:
    return await pdf().execute(ToolCall("c", "pdf", json.dumps(arguments)))


def pages(file: File) -> int:
    return len(pdfium.PdfDocument(BytesIO(file.data)))


def document(*pages: str) -> bytes:
    """A PDF with a page of Helvetica text per argument, a line per `\\n`."""
    count = len(pages)
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        f"<< /Type /Pages /Kids [{' '.join(f'{4 + 2 * i} 0 R' for i in range(count))}] /Count {count} >>",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    for i, text in enumerate(pages):
        lines = " T* ".join(f"({line}) Tj" for line in text.split("\n"))
        stream = f"BT /F1 10 Tf 12 TL 40 760 Td {lines} ET"
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {5 + 2 * i} 0 R >>"
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    body, offsets = "%PDF-1.4\n", []
    for number, content in enumerate(objects, 1):
        offsets.append(len(body))
        body += f"{number} 0 obj\n{content}\nendobj\n"
    xref = "".join(f"{offset:010d} 00000 n \n" for offset in offsets)
    trailer = f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{len(body)}\n%%EOF"
    return f"{body}xref\n0 {len(objects) + 1}\n0000000000 65535 f \n{xref}{trailer}".encode()

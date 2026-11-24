"""
Document parsing for uploads: turns PDF, DOCX, TXT and Markdown files into
a stream of (page_number, text) pairs.

Pages are yielded one at a time so a caller can show progress and keep
memory flat even for documents that run to thousands of pages. PDFs are
parsed with PyMuPDF, which is fast enough for a 1,000-page file to take a
few seconds. Formats without real pages (DOCX, TXT, MD) are split into
pseudo-pages of roughly PSEUDO_PAGE_CHARS characters so citations still
point somewhere useful.
"""
import io
import os
import re
from typing import Iterator

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt", ".md")
PSEUDO_PAGE_CHARS = 3000


def _clean(text: str) -> str:
    text = text.replace("\x00", "")
    # Re-join words hyphenated across line breaks, then collapse whitespace.
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _paginate(text: str) -> Iterator[tuple[int, str]]:
    """Split a page-less document into ~PSEUDO_PAGE_CHARS chunks, breaking
    on paragraph boundaries where possible."""
    paragraphs = re.split(r"\n\s*\n", text)
    page, buf, size = 1, [], 0
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if size and size + len(para) > PSEUDO_PAGE_CHARS:
            yield page, _clean("\n\n".join(buf))
            page, buf, size = page + 1, [], 0
        buf.append(para)
        size += len(para) + 2
    if buf:
        yield page, _clean("\n\n".join(buf))


def count_pages(data: bytes, filename: str) -> int:
    """Total pages, for progress bars. Cheap for every format."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        import pymupdf as fitz  # PyMuPDF
        with fitz.open(stream=data, filetype="pdf") as doc:
            return doc.page_count
    return sum(1 for _ in iter_pages(data, filename))


def iter_pages(data: bytes, filename: str) -> Iterator[tuple[int, str]]:
    """Yield (page_number, text) for every non-empty page of the file."""
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        import pymupdf as fitz  # PyMuPDF
        with fitz.open(stream=data, filetype="pdf") as doc:
            for i, page in enumerate(doc, start=1):
                text = _clean(page.get_text("text"))
                if text:
                    yield i, text
    elif ext == ".docx":
        import docx  # python-docx
        document = docx.Document(io.BytesIO(data))
        parts = [p.text for p in document.paragraphs]
        for table in document.tables:
            for row in table.rows:
                parts.append(" | ".join(cell.text for cell in row.cells))
        yield from _paginate("\n\n".join(parts))
    elif ext in (".txt", ".md"):
        yield from _paginate(data.decode("utf-8", errors="replace"))
    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )


def load_directory(directory: str) -> dict[str, list[tuple[int, str]]]:
    """{filename: [(page, text), ...]} for every supported file in a folder."""
    out = {}
    if not os.path.isdir(directory):
        return out
    for filename in sorted(os.listdir(directory)):
        if filename.lower().endswith(SUPPORTED_EXTENSIONS):
            with open(os.path.join(directory, filename), "rb") as f:
                pages = list(iter_pages(f.read(), filename))
            if pages:
                out[filename] = pages
    return out

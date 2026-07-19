"""
PDF text extraction. Uses pypdf only — no external services.
"""
from io import BytesIO
from pypdf import PdfReader


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF given as bytes. Returns '' on failure."""
    try:
        reader = PdfReader(BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            pages.append(page_text)
        return "\n".join(pages).strip()
    except Exception:
        return ""

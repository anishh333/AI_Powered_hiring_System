"""
PDF text extraction using PyMuPDF (fitz) for fast and robust local parsing.
"""
import fitz
import logfire


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF given as bytes. Returns '' on failure."""
    try:
        with logfire.span("Extracting text from PDF"):
            # Open PDF directly from binary memory stream using PyMuPDF
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            pages = []
            for page in doc:
                page_text = page.get_text() or ""
                pages.append(page_text)
            return "\n".join(pages).strip()
    except Exception as e:
        logfire.error("Failed to extract text from PDF: {error}", error=str(e))
        return ""

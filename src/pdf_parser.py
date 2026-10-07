import pymupdf


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF resume.
    """

    text = []

    with pymupdf.open(pdf_path) as document:

        for page in document:
            page_text = page.get_text("text")

            if page_text:
                text.append(page_text)

    return "\n".join(text).strip()
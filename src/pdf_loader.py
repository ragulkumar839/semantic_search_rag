import re
from pypdf import PdfReader


def clean_text(text):
    if not text:
        return ""
    # Remove control characters and non-printable noise
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', ' ', text)
    # Normalize broken pipe/url header noise like ||sirkali_Mayiladurai...
    text = re.sub(r'\|+', ' ', text)
    # Normalize repeated spaces and blank lines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def load_pdf(file_path):
    reader = PdfReader(file_path)
    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text()
        if raw_text:
            sanitized = clean_text(raw_text)
            if sanitized:
                pages.append({
                    "page": page_number,
                    "text": sanitized
                })

    return pages
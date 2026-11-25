from PyPDF2 import PdfReader


def extract_text(file_path):
    """Return lower-cased text from a .pdf (text layer) or .txt file, else ''."""
    path = file_path.lower()

    if path.endswith(".pdf"):
        reader = PdfReader(file_path)
        return "".join(page.extract_text() or "" for page in reader.pages).lower()

    if path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().lower()

    return ""

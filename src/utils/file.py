from pathlib import Path
from chainlit.element import File
from pypdf import PdfReader


def read_uploaded_file(file: File) -> str:
    if not file.path:
        raise ValueError("Uploaded file has no path")

    path = Path(file.path)
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md", ".py", ".js", ".ts", ".json", ".csv"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text)

    return f"[Unsupported file type: {suffix}]"
import json
from pathlib import Path
from chainlit.element import File
from pypdf import PdfReader
import logging

logger = logging.getLogger(__name__)


def read_uploaded_file(file: File) -> str:
    if not file.path:
        raise ValueError("Uploaded file has no path")

    original_name = file.name
    suffix = Path(original_name).suffix.lower()
    
    # Keep the path for the actual reading
    storage_path = Path(file.path)

    logger.info(f"Processing {original_name} (Detected suffix: {suffix})")

    if suffix in {".txt", ".md", ".py", ".js", ".ts", ".json", ".csv", ".tex"}:
        return storage_path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".ipynb":
        try:
            content = json.loads(storage_path.read_text(encoding="utf-8"))
            extracted_text = []
            for cell in content.get("cells", []):
                cell_type = cell.get("cell_type", "unknown")
                source = "".join(cell.get("source", []))
                extracted_text.append(f"[{cell_type.upper()} CELL]\n{source}")
            return "\n\n".join(extracted_text)
        except Exception as e:
            return f"[Error parsing Notebook: {e}]"

    if suffix == ".pdf":
        reader = PdfReader(str(storage_path))
        text = []
        for page in reader.pages:
            text.append(page.extract_text() or "")
        return "\n".join(text)

    return f"[Unsupported file type: {suffix}]"

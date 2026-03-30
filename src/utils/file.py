import csv
import io
import json
import logging
from pathlib import Path

from chainlit.element import File
from pypdf import PdfReader

logger = logging.getLogger(__name__)

CSV_FULL_TEXT_LIMIT = 200_000  # bytes/chars threshold for including raw CSV text


def _infer_column_type(values: list[str]) -> str:
    non_empty = [v.strip() for v in values if str(v).strip()]
    if not non_empty:
        return "empty"

    def is_int(x: str) -> bool:
        try:
            int(x)
            return True
        except Exception:
            return False

    def is_float(x: str) -> bool:
        try:
            float(x)
            return True
        except Exception:
            return False

    lowered = {v.lower() for v in non_empty}
    if lowered.issubset({"true", "false", "yes", "no", "0", "1"}):
        return "boolean-like"
    if all(is_int(v) for v in non_empty):
        return "integer"
    if all(is_float(v) for v in non_empty):
        return "float"
    return "text"


def _parse_csv_file(storage_path: Path) -> str:
    raw_text = storage_path.read_text(encoding="utf-8", errors="ignore")

    try:
        sample = raw_text[:8192]
        dialect = csv.Sniffer().sniff(sample)
    except Exception:
        dialect = csv.excel

    reader = csv.reader(io.StringIO(raw_text), dialect)
    rows = list(reader)

    if not rows:
        return "[Empty CSV file]"

    header = rows[0]
    data_rows = rows[1:]
    row_count = len(data_rows)
    col_count = len(header)

    preview_rows = data_rows[:5]

    column_summaries = []
    for i, col_name in enumerate(header):
        sample_values = []
        for row in data_rows[:50]:
            if i < len(row):
                sample_values.append(row[i])
        inferred = _infer_column_type(sample_values)
        column_summaries.append(f"- {col_name}: {inferred}")

    parts = [
        "[CSV SUMMARY]",
        f"File: {storage_path.name}",
        f"Columns ({col_count}): {', '.join(header)}",
        f"Rows: {row_count}",
        "",
        "[COLUMN TYPES]",
        "\n".join(column_summaries),
        "",
        "[FIRST 5 ROWS]",
    ]

    for idx, row in enumerate(preview_rows, start=1):
        mapped = {
            header[i]: row[i] if i < len(row) else ""
            for i in range(len(header))
        }
        parts.append(f"Row {idx}: {json.dumps(mapped, ensure_ascii=False)}")

    # Include raw full CSV only when manageable
    if len(raw_text) <= CSV_FULL_TEXT_LIMIT:
        parts.extend([
            "",
            "[FULL CSV CONTENT]",
            raw_text
        ])
    else:
        parts.extend([
            "",
            f"[FULL CSV OMITTED: raw content length {len(raw_text)} chars exceeds "
            f"CSV_FULL_TEXT_LIMIT={CSV_FULL_TEXT_LIMIT}]"
        ])

    return "\n".join(parts)


def read_uploaded_file(file: File) -> str:
    if not file.path:
        raise ValueError("Uploaded file has no path")

    original_name = file.name
    suffix = Path(original_name).suffix.lower()
    storage_path = Path(file.path)

    logger.info("Processing %s (Detected suffix: %s)", original_name, suffix)

    if suffix in {".txt", ".md", ".py", ".js", ".ts", ".tex"}:
        return storage_path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".json":
        try:
            obj = json.loads(storage_path.read_text(encoding="utf-8", errors="ignore"))
            return json.dumps(obj, indent=2, ensure_ascii=False)
        except Exception:
            return storage_path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".csv":
        return _parse_csv_file(storage_path)

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
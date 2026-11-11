import re
from pathlib import Path

import fitz

from qdrant_service import qdrant_service


def extract_pdf_chunks(file_bytes: bytes, filename: str) -> list[dict]:
    if not file_bytes:
        raise ValueError("The PDF is empty.")
    try:
        document = fitz.open(stream=file_bytes, filetype="pdf")
    except Exception as error:
        raise ValueError("The uploaded file is not a readable PDF.") from error

    chunks: list[dict] = []
    chunk_number = 1
    try:
        for page_number, page in enumerate(document, start=1):
            text = re.sub(r"\s+", " ", page.get_text("text")).strip()
            for start in range(0, len(text), 1000):
                part = text[start:start + 1000]
                if not part:
                    continue
                chunks.append({
                    "id": f"{Path(filename).stem}-p{page_number:03d}-c{chunk_number:03d}",
                    "text": part,
                    "filename": filename,
                    "page": page_number,
                })
                chunk_number += 1
    finally:
        document.close()
    if not chunks:
        raise ValueError("No selectable text was found in the PDF.")
    return chunks


def document_id(session_id: str, file_bytes: bytes) -> str:
    return qdrant_service.document_id(session_id, file_bytes)


def index_chunks(chunks: list[dict], session_id: str, doc_id: str, source_type: str) -> int:
    return qdrant_service.upsert_document(
        chunks, session_id, doc_id, source_type=source_type
    )

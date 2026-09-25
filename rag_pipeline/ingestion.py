from __future__ import annotations

import os
os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
from typing import Dict, List, Optional

import pymupdf4llm
from langchain_community.document_loaders import PyMuPDFLoader


DEFAULT_PROSE_CHUNK_SIZE = 350
DEFAULT_PROSE_CHUNK_OVERLAP = 40


class DocumentIngestion:

    def __init__(
        self,
        prose_chunk_size: int = DEFAULT_PROSE_CHUNK_SIZE,
        prose_chunk_overlap: int = DEFAULT_PROSE_CHUNK_OVERLAP,
    ) -> None:
        self.prose_chunk_size = prose_chunk_size
        self.prose_chunk_overlap = prose_chunk_overlap

    def load_document_pages(self, pdf_path: str) -> list:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"[Ingestion] File not found: {pdf_path}")
        loader = PyMuPDFLoader(pdf_path)
        pages = loader.load()
        print(f"[Ingestion] Loaded {len(pages)} page(s) from '{os.path.basename(pdf_path)}'.")
        return pages

    def _split_prose_text(self, text: str) -> List[str]:
        max_size = self.prose_chunk_size
        overlap = self.prose_chunk_overlap
        if len(text) <= max_size:
            return [text]
        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(start + max_size, len(text))
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end == len(text):
                break
            start += max_size - overlap
        return chunks

    def extract_chunks(self, pdf_paths: List[str], doc_ids: Optional[Dict[str, str]] = None) -> List[Dict]:
        chunks: List[Dict] = []
        doc_ids = doc_ids or {}

        for path in pdf_paths:
            if not os.path.exists(path):
                print(f"[Ingestion] Skipping missing file: '{path}'")
                continue
            
            doc_id = doc_ids.get(path)
            doc_name = os.path.basename(path).replace(".pdf", "")
            print(f"[Ingestion] Extracting chunks from '{doc_name}'...")
            md_text = pymupdf4llm.to_markdown(path, use_ocr=True)
            lines = md_text.split("\n")

            current_header = doc_name
            prose_buffer: List[str] = []
            table_headers: List[str] = []
            table_rows: List[List[str]] = []
            table_counter = 0
            in_table = False

            def flush_table() -> None:
                nonlocal table_headers, table_rows, in_table, table_counter
                if not table_rows:
                    table_headers = []
                    table_rows = []
                    in_table = False
                    return

                table_counter += 1
                table_id = f"{doc_name}_tbl{table_counter}"

                for row in table_rows:
                    row_fields: List[str] = []
                    for col_idx, cell_val in enumerate(row):
                        if cell_val and cell_val != "•":
                            clean_val = cell_val.replace("**", "").replace("•", "").strip()
                            h_name = (
                                table_headers[col_idx].replace("**", "").strip()
                                if col_idx < len(table_headers)
                                else f"Col{col_idx + 1}"
                            )
                            if clean_val:
                                row_fields.append(f"{h_name}: {clean_val}")
                    if row_fields:
                        meta = {
                            "source": path,
                            "type": "child_table_row",
                            "parent_id": table_id,
                            "header": current_header,
                        }
                        if doc_id: meta["doc_id"] = doc_id
                        chunks.append({
                            "text": (
                                f"[{doc_name} > {current_header} > Table Item]\n"
                                + " | ".join(row_fields)
                            ),
                            "metadata": meta,
                        })

                col_summary = ", ".join(h.replace("**", "").strip() for h in table_headers if h)
                summary_lines = [
                    f"[{doc_name} > {current_header} > Full Table Overview]",
                    f"Summary: Contains {len(table_rows)} total rows across columns [{col_summary}].",
                    "All Items Listed:",
                ]
                for r in table_rows:
                    cleaned = [
                        c.strip().replace("<br>", " ").replace("**", "").replace("•", "").strip()
                        for c in r if c != "•"
                    ]
                    summary_lines.append(" - " + " | ".join(c for c in cleaned if c))

                meta = {
                    "source": path,
                    "type": "parent_table_summary",
                    "table_id": table_id,
                    "header": current_header,
                }
                if doc_id: meta["doc_id"] = doc_id
                chunks.append({
                    "text": "\n".join(summary_lines),
                    "metadata": meta,
                })

                table_headers = []
                table_rows = []
                in_table = False

            def flush_prose() -> None:
                nonlocal prose_buffer
                if not prose_buffer:
                    return
                full_prose = " ".join(prose_buffer).strip()
                for sub in self._split_prose_text(full_prose):
                    meta = {
                        "source": path,
                        "type": "prose",
                        "header": current_header,
                    }
                    if doc_id: meta["doc_id"] = doc_id
                    chunks.append({
                        "text": f"[{doc_name} > {current_header}]\n{sub.strip()}",
                        "metadata": meta,
                    })
                prose_buffer = []

            for line in lines:
                line_str = line.strip()

                if line_str.startswith("#"):
                    flush_table()
                    flush_prose()
                    current_header = line_str.lstrip("#").strip()
                    continue

                if line_str.startswith("|") and line_str.endswith("|"):
                    if not in_table:
                        flush_prose()
                    in_table = True
                    cells = [c.strip().replace("<br>", " ") for c in line_str.split("|")[1:-1]]
                    if all(set(c).issubset({"-", " ", ":"}) for c in cells if c):
                        continue
                    if not table_headers:
                        table_headers = cells
                    else:
                        table_rows.append(cells)
                    continue

                if in_table:
                    flush_table()
                if line_str:
                    prose_buffer.append(line_str)

            flush_table()
            flush_prose()

            file_chunks = [c for c in chunks if c["metadata"].get("source") == path]
            if not file_chunks and md_text.strip():
                print(f"[Ingestion] Fallback chunk added for '{doc_name}'.")
                meta = {
                    "source": path,
                    "type": "prose",
                    "header": current_header or doc_name,
                }
                if doc_id: meta["doc_id"] = doc_id
                chunks.append({
                    "text": f"[{doc_name}]\n{md_text.strip()}",
                    "metadata": meta,
                })

        return chunks

    def __repr__(self) -> str:
        return (
            f"DocumentIngestion("
            f"prose_chunk_size={self.prose_chunk_size}, "
            f"prose_chunk_overlap={self.prose_chunk_overlap})"
        )

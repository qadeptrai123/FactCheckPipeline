"""
Chunk corpus text using LangChain fixed-length token splitting and write CSV output.

Usage:
    python chunk_fixed_size.py

Output CSV columns:
    chunk_id, modality, row_id, chunk_index,
    text, image_path, text_chunk_id,
    source, url, title, author, date, corpus_id
"""

import csv
import os
import sys
from pathlib import Path
from typing import Dict, Iterator, List

from langchain_text_splitters import TokenTextSplitter
from tqdm.auto import tqdm


ENV_PATH = Path(__file__).resolve().parent.parent / ".env"


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file(ENV_PATH)

FIXED_SIZE_CHUNK_SIZE = int(os.environ.get("FIXED_SIZE_CHUNK_SIZE", "256"))
FIXED_SIZE_OVERLAP = int(os.environ.get("FIXED_SIZE_OVERLAP", "40"))
FIXED_SIZE_ENCODING = os.environ.get("FIXED_SIZE_ENCODING", "cl100k_base")
FIXED_SIZE_SAMPLE = int(os.environ["FIXED_SIZE_SAMPLE"]) if os.environ.get("FIXED_SIZE_SAMPLE") else None
CORPUS_PATH = Path(os.environ.get("FIXED_SIZE_CORPUS_PATH", "D:/RAG-DB/FinalDataset/final_corpus.csv"))
OUTPUT_PATH = Path(os.environ.get("FIXED_SIZE_OUTPUT", "D:/RAG-DB/chunking_scripts/chunks_fixed_size.csv"))


def read_corpus(path: Path) -> Iterator[Dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def count_rows(path: Path) -> int:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)


def build_splitter() -> TokenTextSplitter:
    return TokenTextSplitter(
        encoding_name=FIXED_SIZE_ENCODING,
        chunk_size=FIXED_SIZE_CHUNK_SIZE,
        chunk_overlap=FIXED_SIZE_OVERLAP,
    )


def chunk_text(splitter: TokenTextSplitter, text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    chunks = splitter.split_text(text)
    return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]


def make_chunk_rows(row: Dict[str, str], splitter: TokenTextSplitter) -> List[Dict[str, str]]:
    row_id = row.get("id", "0") or "0"
    chunks = chunk_text(splitter, row.get("content", ""))

    return [
        {
            "chunk_id": f"row_{row_id}_text_{chunk_index}",
            "modality": "text",
            "row_id": row_id,
            "chunk_index": str(chunk_index),
            "text": text,
            "image_path": "",
            "text_chunk_id": "",
            "source": row.get("source", ""),
            "url": row.get("url", ""),
            "title": row.get("title", ""),
            "author": row.get("author", ""),
            "date": row.get("date", ""),
            "corpus_id": row_id,
        }
        for chunk_index, text in enumerate(chunks)
    ]


def write_chunks(chunks: List[Dict[str, str]], output_path: Path) -> None:
    fieldnames = [
        "chunk_id", "modality", "row_id", "chunk_index",
        "text", "image_path", "text_chunk_id",
        "source", "url", "title", "author", "date",
        "corpus_id",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(chunks)


def run() -> None:
    splitter = build_splitter()
    all_chunks: List[Dict[str, str]] = []
    errors: List[str] = []

    total_rows = count_rows(CORPUS_PATH)
    limit = FIXED_SIZE_SAMPLE if FIXED_SIZE_SAMPLE else total_rows
    pbar = tqdm(
        total=limit,
        desc="[fixed_size] Chunking rows",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        dynamic_ncols=True,
        file=sys.stdout,
        leave=True,
    )

    for idx, row in enumerate(read_corpus(CORPUS_PATH)):
        if FIXED_SIZE_SAMPLE and idx >= FIXED_SIZE_SAMPLE:
            break

        row_id = row.get("id", "0") or "0"
        try:
            all_chunks.extend(make_chunk_rows(row, splitter))
        except Exception as e:
            errors.append(f"row_{row_id}: {e}")

        pbar.update(1)

    pbar.close()
    write_chunks(all_chunks, OUTPUT_PATH)

    print("\n[fixed_size] Done.", flush=True)
    print(f"  Total chunks : {len(all_chunks)}", flush=True)
    print(f"  Text chunks  : {len(all_chunks)}", flush=True)
    print("  Image chunks : 0", flush=True)
    print(f"  Errors       : {len(errors)}", flush=True)
    print(f"  Output file  : {OUTPUT_PATH}", flush=True)
    if errors:
        for error in errors[:10]:
            print(f"    ! {error}", flush=True)


if __name__ == "__main__":
    run()

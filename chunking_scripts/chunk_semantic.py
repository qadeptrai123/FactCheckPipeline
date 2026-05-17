"""
Chunk corpus text using LangChain SemanticChunker and write CSV output.

Usage:
    python chunk_semantic.py

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

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm


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

SEMANTIC_MODEL = os.environ.get("SEMANTIC_MODEL")
SEMANTIC_THRESHOLD = float(os.environ.get("SEMANTIC_THRESHOLD", "85"))
SEMANTIC_BREAKPOINT_TYPE = os.environ.get("SEMANTIC_BREAKPOINT_TYPE", "percentile")
SEMANTIC_SAMPLE = int(os.environ["SEMANTIC_SAMPLE"]) if os.environ.get("SEMANTIC_SAMPLE") else None
CORPUS_PATH = Path(os.environ.get("SEMANTIC_CORPUS_PATH", "D:/RAG-DB/FinalDataset/final_corpus.csv"))
OUTPUT_PATH = Path(os.environ.get("SEMANTIC_OUTPUT", "D:/RAG-DB/chunking_scripts/chunks_semantic.csv"))


def read_corpus(path: Path) -> Iterator[Dict[str, str]]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def count_rows(path: Path) -> int:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)


def build_chunker() -> SemanticChunker:
    if not SEMANTIC_MODEL:
        raise RuntimeError("Missing SEMANTIC_MODEL environment variable.")

    embeddings = HuggingFaceEmbeddings(model_name=SEMANTIC_MODEL)
    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
        breakpoint_threshold_amount=SEMANTIC_THRESHOLD,
    )


def chunk_text(chunker: SemanticChunker, text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []

    try:
        chunks = chunker.split_text(text)
    except Exception:
        chunks = [text]

    return [chunk.strip() for chunk in chunks if chunk and chunk.strip()]


def make_chunk_rows(row: Dict[str, str], chunker: SemanticChunker) -> List[Dict[str, str]]:
    row_id = row.get("id", "0") or "0"
    chunks = chunk_text(chunker, row.get("content", ""))

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
    chunker = build_chunker()
    all_chunks: List[Dict[str, str]] = []
    errors: List[str] = []

    total_rows = count_rows(CORPUS_PATH)
    limit = SEMANTIC_SAMPLE if SEMANTIC_SAMPLE else total_rows
    pbar = tqdm(
        total=limit,
        desc="[semantic] Chunking rows",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        dynamic_ncols=True,
        file=sys.stdout,
        leave=True,
    )

    for idx, row in enumerate(read_corpus(CORPUS_PATH)):
        if SEMANTIC_SAMPLE and idx >= SEMANTIC_SAMPLE:
            break

        row_id = row.get("id", "0") or "0"
        try:
            all_chunks.extend(make_chunk_rows(row, chunker))
        except Exception as e:
            errors.append(f"row_{row_id}: {e}")

        pbar.update(1)

    pbar.close()
    write_chunks(all_chunks, OUTPUT_PATH)

    print("\n[semantic] Done.", flush=True)
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

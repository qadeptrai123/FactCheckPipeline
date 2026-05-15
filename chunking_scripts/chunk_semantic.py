"""
Chunk corpus text using Semantic strategy and write CSV output.

Uses LangChain's SemanticChunker with bkai-foundation-models/vietnamese-bi-encoder
for dynamic semantic splitting, following the approach from:
https://www.lancedb.com/blog/chunking-techniques-with-langchain-and-llamaindex

Usage:
    python chunk_semantic.py
    python chunk_semantic.py --sample 20
    python chunk_semantic.py --threshold 90

Output CSV columns:
    chunk_id, modality, row_id, chunk_index,
    text, image_path, text_chunk_id,
    source, url, title, author, date, corpus_id
"""
import argparse
import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
from tqdm import tqdm

from src.domain.chunk import Chunk
from src.domain.corpus_row import CorpusRow
from src.chunking.semantic import SemanticChunkingStrategy

CORPUS_PATH = Path("D:/RAG-DB/FinalDataset/final_corpus.csv")
DEFAULT_OUTPUT = Path("D:/RAG-DB/chunking_scripts/chunks_semantic.csv")


def read_corpus(path: Path) -> List[CorpusRow]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield CorpusRow(
                id=int(row.get("id", 0)),
                source=row.get("source", ""),
                url=row.get("url", ""),
                title=row.get("title", ""),
                author=row.get("author", ""),
                date=row.get("date", ""),
                content=row.get("content", ""),
                media=row.get("media", ""),
            )


def chunk_to_csv(chunks: List[Chunk], output_path: Path):
    # corpus_id replaces 'content' — use it to retrieve full text from final_corpus.csv
    fieldnames = [
        "chunk_id", "modality", "row_id", "chunk_index",
        "text", "image_path", "text_chunk_id",
        "source", "url", "title", "author", "date",
        "corpus_id",
    ]  # no "embedding" — chunking output only, embedding handled at upsert time
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for chunk in chunks:
            payload = chunk.to_qdrant_payload()
            writer.writerow({k: payload.get(k, "") for k in fieldnames})


def run(strategy_name: str, threshold: float,
        breakpoint_type: str,
        output_path: Path, sample: int = None):
    strategy = SemanticChunkingStrategy(
        threshold=threshold,
        breakpoint_threshold_type=breakpoint_type,
    )

    all_chunks: List[Chunk] = []
    errors: List[str] = []

    total_rows = 0
    with open(CORPUS_PATH, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        total_rows = sum(1 for _ in reader) - 1

    limit = sample if sample else total_rows
    pbar = tqdm(total=limit, desc=f"[{strategy_name}] Chunking rows",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")

    for idx, row in enumerate(read_corpus(CORPUS_PATH)):
        if sample and idx >= sample:
            break

        try:
            metadata = {
                "row_id": row.id,
                "source": row.source,
                "url": row.url,
                "title": row.title,
                "author": row.author,
                "date": row.date,
                "content": row.content,
            }

            text_chunks = strategy.chunk_no_embed(row.content, metadata)
            all_chunks.extend(text_chunks)

        except Exception as e:
            errors.append(f"row_{row.id}: {e}")

        pbar.update(1)

    pbar.close()

    chunk_to_csv(all_chunks, output_path)

    text_count = sum(1 for c in all_chunks if c.modality == "text")
    img_count = sum(1 for c in all_chunks if c.modality == "image")
    print(f"\n[{strategy_name}] Done.")
    print(f"  Total chunks : {len(all_chunks)}")
    print(f"  Text chunks  : {text_count}")
    print(f"  Image chunks: {img_count}")
    print(f"  Errors      : {len(errors)}")
    print(f"  Output file : {output_path}")
    if errors:
        for e in errors[:10]:
            print(f"    ! {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk corpus with Semantic strategy")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only first N rows (for testing)")
    parser.add_argument("--threshold", type=float, default=85,
                        help="Breakpoint threshold amount (default: 85 for percentile mode)")
    parser.add_argument("--breakpoint-type", type=str, default="percentile",
                        choices=["percentile", "standard_deviation",
                                 "interquartile", "gradient"],
                        help="Breakpoint detection method (default: percentile)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    run(
        strategy_name="semantic",
        threshold=args.threshold,
        breakpoint_type=args.breakpoint_type,
        output_path=Path(args.output),
        sample=args.sample,
    )

"""
Chunk corpus text/images using Fixed-Size strategy and write CSV output.

Usage:
    python chunk_fixed_size.py              # full corpus
    python chunk_fixed_size.py --sample 20  # first 20 rows for testing
    python chunk_fixed_size.py --output my_chunks.csv

Output CSV columns:
    chunk_id, modality, row_id, chunk_index,
    text, image_path, text_chunk_id,
    source, url, title, author, date, content
"""
import argparse
import sys
from pathlib import Path
from typing import List

# ── project root so we can import src/ without pip install ──────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

import csv
from tqdm import tqdm

from src.domain.chunk import Chunk
from src.domain.corpus_row import CorpusRow
from src.chunking.fixed_size import FixedSizeChunkingStrategy
from src.paths import resolve_media_path

CORPUS_PATH = Path("D:/RAG-DB/FinalDataset/final_corpus.csv")
DEFAULT_OUTPUT = Path("D:/RAG-DB/chunking_scripts/chunks_fixed_size.csv")


def read_corpus(path: Path) -> List[CorpusRow]:
    """Read final_corpus.csv and yield CorpusRow objects."""
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
    """Write chunks to a flat CSV."""
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


def run(strategy_name: str, chunk_size: int, overlap: int,
        output_path: Path, sample: int = None):
    strategy = FixedSizeChunkingStrategy(chunk_size=chunk_size, overlap=overlap)

    all_chunks: List[Chunk] = []
    errors: List[str] = []

    # Count rows for progress bar
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
            text_start = len(all_chunks)
            all_chunks.extend(text_chunks)

            first_text_id = text_chunks[0].chunk_id if text_chunks else None
            # for img_idx, img_path_str in enumerate(row.media_paths):
            #     resolved = resolve_media_path(img_path_str)
            #     if resolved is None:
            #         errors.append(f"row_{row.id}: media not found: {img_path_str}")
            #         continue
            #     img_chunk = Chunk(
            #         chunk_id=f"row_{row.id}_img_{img_idx}",
            #         text=None,
            #         image_path=str(resolved),
            #         row_id=row.id,
            #         chunk_index=img_idx,
            #         modality="image",
            #         text_chunk_id=first_text_id,
            #         metadata=metadata,
            #     )
            #     all_chunks.append(img_chunk)

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
    print(f"  Errors       : {len(errors)}")
    print(f"  Output file  : {output_path}")
    if errors:
        for e in errors[:10]:
            print(f"    ! {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk corpus with Fixed-Size strategy")
    parser.add_argument("--sample", type=int, default=None,
                        help="Process only first N rows (for testing)")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="Max tokens per chunk (default: 256)")
    parser.add_argument("--overlap", type=int, default=40,
                        help="Token overlap between chunks (default: 40)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT),
                        help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    args = parser.parse_args()

    run(
        strategy_name="fixed_size",
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        output_path=Path(args.output),
        sample=args.sample,
    )

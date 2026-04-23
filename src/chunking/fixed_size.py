"""Fixed-size token-based chunking using tiktoken sliding window.

No embedder — token windowing is purely structural, no semantic decisions needed.
"""

from typing import Any, Dict, List

import tiktoken

from src.domain.chunk import Chunk


class FixedSizeChunkingStrategy:
    def __init__(self, chunk_size: int = 256, overlap: int = 40):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.enc = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Full chunk — no embedding needed for fixed-size strategy."""
        return self.chunk_no_embed(text, metadata)

    def chunk_no_embed(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Slide a token window over the text — no embedding."""
        tokens = self.enc.encode(text)
        if not tokens:
            return []

        step = self.chunk_size - self.overlap
        raw_chunks: List[str] = []
        idx = 0
        id_prefix = metadata.get("_id_prefix", "row")

        while idx < len(tokens):
            window_tokens = tokens[idx : idx + self.chunk_size]
            chunk_text = self.enc.decode(window_tokens)
            raw_chunks.append(chunk_text)

            if idx + self.chunk_size >= len(tokens):
                break
            idx += step

        return [
            Chunk(
                chunk_id=f"{id_prefix}_text_{i}",
                text=t,
                row_id=metadata.get("row_id", 0),
                chunk_index=i,
                modality="text",
                metadata={**metadata},
            )
            for i, t in enumerate(raw_chunks)
        ]

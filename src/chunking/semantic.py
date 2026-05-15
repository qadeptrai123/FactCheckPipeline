"""Semantic chunking using LangChain SemanticChunker + HuggingFace embeddings.

Follows the semantic splitting approach from:
https://www.lancedb.com/blog/chunking-techniques-with-langchain-and-llamaindex

Blog reference (LangChain – Semantic Splitting):
    >>> from langchain_experimental.text_splitter import SemanticChunker
    >>> from langchain_openai.embeddings import OpenAIEmbeddings
    >>> text_splitter = SemanticChunker(OpenAIEmbeddings())
    >>> docs = text_splitter.create_documents([state_of_the_union])

This implementation uses bkai-foundation-models/vietnamese-bi-encoder
(HuggingFaceEmbeddings) instead of OpenAIEmbeddings so it works offline
and is tuned for Vietnamese text.
"""

from typing import Any, Dict, List

from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

from src.domain.chunk import Chunk

# ── Global embedder cache ─────────────────────────────────────────────────────
_EMBEDDER: HuggingFaceEmbeddings | None = None


def _get_embedder(
    model: str = "bkai-foundation-models/vietnamese-bi-encoder",
) -> HuggingFaceEmbeddings:
    """Return a cached HuggingFaceEmbeddings instance (loaded once)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = HuggingFaceEmbeddings(model_name=model)
    return _EMBEDDER


# ── Main strategy class ────────────────────────────────────────────────────────


class SemanticChunkingStrategy:
    """Semantic chunking: LangChain SemanticChunker + bkai Vietnamese embedder.

    Strategy (handled internally by SemanticChunker):
      1. Split text into sentences.
      2. Embed every sentence with bkai-foundation-models/vietnamese-bi-encoder.
      3. Compute distances between consecutive sentence embeddings.
      4. Detect breakpoints where distance exceeds a threshold:
         - "percentile":          top-N% largest distance drops → new chunk
         - "standard_deviation":  distance > N standard deviations
         - "interquartile":       distance beyond IQR
         - "gradient":            gradient-based change detection

    Attributes:
        breakpoint_threshold_amount: Sensitivity parameter for the chosen method.
            For "percentile" mode, this is the percentile value (0–100).
            Higher value → fewer, larger chunks.  Default 85.
        breakpoint_threshold_type: One of "percentile", "standard_deviation",
            "interquartile", "gradient".  Default "percentile".
        embedder_model: HuggingFace model name for HuggingFaceEmbeddings.
            Default: bkai-foundation-models/vietnamese-bi-encoder (768d).
    """

    def __init__(
        self,
        threshold: float = 85,
        breakpoint_threshold_type: str = "percentile",
        embedder_model: str = "bkai-foundation-models/vietnamese-bi-encoder",
    ):
        self.threshold = threshold
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.embedder_model = embedder_model

    def _build_chunker(self) -> SemanticChunker:
        """Build a SemanticChunker instance with cached embedder."""
        embedder = _get_embedder(self.embedder_model)
        return SemanticChunker(
            embeddings=embedder,
            breakpoint_threshold_type=self.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.threshold,
        )

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Full chunk with embedding — single row."""
        chunks = self.chunk_no_embed(text, metadata)
        if not chunks:
            return []
        embedder = _get_embedder(self.embedder_model)
        embs = embedder.embed_documents([c.text for c in chunks])
        for c, emb in zip(chunks, embs):
            c.embedding = emb
        return chunks

    def chunk_no_embed(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Split into semantically coherent chunks — no embedding output."""
        if not text or not text.strip():
            return []

        chunker = self._build_chunker()

        try:
            texts = chunker.split_text(text)
        except Exception:
            # Fallback: if semantic chunking fails (e.g., very short text
            # with < 2 sentences), return the entire text as a single chunk
            texts = [text.strip()]

        if not texts:
            return []

        id_prefix = metadata.get("_id_prefix", "row")

        return [
            Chunk(
                chunk_id=f"{id_prefix}_text_{i}",
                text=t,
                row_id=metadata.get("row_id", 0),
                chunk_index=i,
                modality="text",
                metadata={**metadata},
            )
            for i, t in enumerate(texts)
        ]

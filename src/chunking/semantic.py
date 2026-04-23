"""LangChain + bkai-foundation-models/vietnamese-bi-encoder semantic chunking."""
import re
from typing import Any, Dict, List

from langchain_huggingface import HuggingFaceEmbeddings

from src.domain.chunk import Chunk
from src.chunking.section import LangChainParagraphStrategy

# ── Global embedder cache ─────────────────────────────────────────────────────
_EMBEDDER: HuggingFaceEmbeddings | None = None


def _get_embedder(
    model: str = "bkai-foundation-models/vietnamese-bi-encoder",
) -> HuggingFaceEmbeddings:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = HuggingFaceEmbeddings(model_name=model)
    return _EMBEDDER


# ── Sentence splitting ─────────────────────────────────────────────────────────
# Multilingual: ASCII (.!?) and CJK full-width (。？！) sentence terminators
_SENT_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+|(?<=[。？！])\s+",
    re.UNICODE,
)


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using multilingual sentence-boundary regex.

    Covers ASCII (.!?) and CJK full-width (。？！) punctuation followed by
    whitespace.  Returns empty list for empty/whitespace-only input.
    """
    if not text.strip():
        return []
    raw = _SENT_SPLIT_RE.split(text)
    return [s.strip() for s in raw if s.strip()]


# ── Semantic boundary detection ───────────────────────────────────────────────


def _cosine_sim(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length embedding vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b + 1e-9)


def _find_semantic_boundaries(
    sentences: List[str],
    threshold: float,
    embedder: HuggingFaceEmbeddings,
) -> List[int]:
    """Return start indices of each semantically coherent chunk.

    Algorithm (one inference pass per row):
      1. Batch-embed all sentences with bkai-foundation-models/vietnamese-bi-encoder.
      2. Compare each consecutive pair (sᵢ, sᵢ₊₁) by cosine similarity.
      3. If sim < threshold → boundary after sᵢ.

    Returns:
        Sorted list of chunk start indices, e.g. [0, 3, 7] means:
        chunk 0 = sentences[0..2], chunk 1 = sentences[3..6], chunk 2 = sentences[7:].
    """
    if len(sentences) <= 2:
        return [0]

    # Batch-embed all sentences (single inference call per row)
    embs: List[List[float]] = embedder.embed_documents(sentences)

    boundaries = [0]
    for i in range(len(sentences) - 1):
        sim = _cosine_sim(embs[i], embs[i + 1])
        if sim < threshold:
            boundaries.append(i + 1)

    return boundaries


# ── Main strategy class ────────────────────────────────────────────────────────


class SemanticChunkingStrategy:
    """Semantic chunking: LangChain + bkai Vietnamese embedder.

    Strategy:
      1. Split text into sentences (multilingual regex via ``re.split``).
      2. Batch-embed all sentences with bkai-foundation-models/vietnamese-bi-encoder.
      3. Detect boundaries between consecutive sentence pairs whose cosine
         similarity falls below ``threshold``.
      4. Fall back to LangChainParagraphStrategy for texts with ≤2 sentences.

    Attributes:
        threshold: Cosine-similarity threshold; below this between consecutive
                  sentences → new chunk boundary.  Default 0.55.
        embedder_model: HuggingFace model name for HuggingFaceEmbeddings.
                        Default: bkai-foundation-models/vietnamese-bi-encoder (768d).
        fallback_chunk_size: Character budget for paragraph fallback (≤2 sentences).
        fallback_overlap:   Character overlap for paragraph fallback.
    """

    def __init__(
        self,
        threshold: float = 0.55,
        embedder_model: str = "bkai-foundation-models/vietnamese-bi-encoder",
        fallback_chunk_size: int = 400,
        fallback_overlap: int = 40,
    ):
        self.threshold = threshold
        self.embedder_model = embedder_model
        self._fallback = LangChainParagraphStrategy(
            chunk_size=fallback_chunk_size,
            chunk_overlap=fallback_overlap,
        )

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Full chunk with embedding — single row."""
        chunks = self.chunk_no_embed(text, metadata)
        if not chunks:
            return []
        embs = self._embed_batch([c.text for c in chunks])
        for c, emb in zip(chunks, embs):
            c.embedding = emb
        return chunks

    def chunk_no_embed(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Split into semantically coherent chunks — no embedding (for batch)."""
        sentences = _split_sentences(text)
        n = len(sentences)

        if n <= 2:
            return self._fallback.chunk_no_embed(text, metadata)

        id_prefix = metadata.get("_id_prefix", "row")
        embedder = _get_embedder(self.embedder_model)
        embs: List[List[float]] = embedder.embed_documents(sentences)
        boundaries = _find_semantic_boundaries(sentences, self.threshold, embedder)

        if boundaries[-1] != n:
            boundaries.append(n)

        chunks: List[Chunk] = []
        for chunk_idx, start in enumerate(boundaries[:-1]):
            end = boundaries[chunk_idx + 1]
            chunk_text = " ".join(sentences[start:end])
            chunk_emb: List[float]
            if end - start == 1:
                chunk_emb = embs[start]
            else:
                chunk_emb = [sum(v) / len(v) for v in zip(*embs[start:end])]

            chunks.append(
                Chunk(
                    chunk_id=f"{id_prefix}_text_{chunk_idx}",
                    text=chunk_text,
                    row_id=metadata.get("row_id", 0),
                    chunk_index=chunk_idx,
                    modality="text",
                    metadata={**metadata},
                    embedding=chunk_emb,
                )
            )
        return chunks

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Batch-embed arbitrary text list in one inference pass."""
        return _get_embedder(self.embedder_model).embed_documents(texts)

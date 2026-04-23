"""LangChain RecursiveCharacterTextSplitter + LlamaIndex SentenceSplitter chunking.

No embeddings — both strategies are purely structural (separators / sentence boundaries).
The semantic strategy handles embedding + semantic splitting when needed.
"""
from typing import Any, Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter as _LangChainSplitter
from llama_index.core.node_parser import SentenceSplitter as _LlamaSplitter

from src.domain.chunk import Chunk

# ── LangChain RecursiveCharacterTextSplitter ──────────────────────────────────

VIETNAMESE_SEPARATORS = [
    "\n\n", "\n", " ",
    "\u3002",   # ideographic full stop (used in some Vietnamese texts)
    ".", ",", "",
]


def build_langchain_splitter(
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> _LangChainSplitter:
    """LangChain RecursiveCharacterTextSplitter tuned for Vietnamese."""
    return _LangChainSplitter(
        separators=VIETNAMESE_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )


class LangChainParagraphStrategy:
    """LangChain RecursiveCharacterTextSplitter wrapped to return Chunk objects."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.splitter = build_langchain_splitter(chunk_size, chunk_overlap)

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Full chunk — no embedding needed for paragraph strategy."""
        return self.chunk_no_embed(text, metadata)

    def chunk_no_embed(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Split text on paragraph/line/word boundaries — no embedding."""
        texts = self.splitter.split_text(text)
        if not texts:
            return []
        id_prefix = metadata.get("_id_prefix", "row")
        return [
            Chunk(
                chunk_id=f"{id_prefix}_text_{idx}",
                text=t,
                row_id=metadata.get("row_id", 0),
                chunk_index=idx,
                modality="text",
                metadata={**metadata},
            )
            for idx, t in enumerate(texts)
        ]


# ── LlamaIndex SentenceSplitter ───────────────────────────────────────────────

_LLAMA_SEPARATORS = r"[。.??!]+"  # multilingual sentence boundaries


# Minimal metadata keys to pass to LlamaIndex Document (exclude large fields)
_LLAMA_META_KEYS = frozenset({"source", "url", "title", "author", "date", "row_id"})


def build_llama_splitter(
    chunk_size: int = 1024,
    chunk_overlap: int = 100,
) -> _LlamaSplitter:
    """LlamaIndex SentenceSplitter tuned for RAG.

    chunk_size=1024: larger than paragraph because sentence chunks
    are bounded by sentence boundaries, not by size alone.
    """
    return _LlamaSplitter(
        separator=" ",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        paragraph_separator="\n\n",
        secondary_chunking_regex=_LLAMA_SEPARATORS,
        include_metadata=True,
        include_prev_next_rel=True,
    )


class LlamaSentenceStrategy:
    """LlamaIndex SentenceSplitter wrapped to return Chunk objects."""

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 100):
        self.splitter = build_llama_splitter(chunk_size, chunk_overlap)

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Use get_nodes_from_documents() to get TextNode objects, then map to Chunk."""
        from llama_index.core.schema import Document

        # Pass only small metadata keys to LlamaIndex to avoid chunk_size warnings.
        # The full content is restored from metadata below after splitting.
        llama_meta = {k: v for k, v in metadata.items()
                      if k in _LLAMA_META_KEYS and k != "content"}

        doc = Document(text=text, metadata=llama_meta)
        nodes = self.splitter.get_nodes_from_documents([doc])

        id_prefix = metadata.get("_id_prefix", "row")
        chunks: List[Chunk] = []

        for idx, node in enumerate(nodes):
            # prev/next relationships from LlamaIndex TextNode
            prev_id = getattr(node.prev_node, "node_id", None)
            next_id = getattr(node.next_node, "node_id", None)

            chunks.append(
                Chunk(
                    chunk_id=f"{id_prefix}_text_{idx}",
                    text=node.get_text(),
                    row_id=metadata.get("row_id", 0),
                    chunk_index=idx,
                    modality="text",
                    metadata={
                        **metadata,
                        "node_id": node.node_id,
                        "prev_chunk_id": prev_id or "",
                        "next_chunk_id": next_id or "",
                        # character-level positions for provenance
                        "start_char_idx": getattr(node, "start_char_idx", 0),
                        "end_char_idx": getattr(node, "end_char_idx", 0),
                    },
                )
            )
        return chunks
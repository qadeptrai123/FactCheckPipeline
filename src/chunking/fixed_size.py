"""Fixed-size token-based chunking using LangChain CharacterTextSplitter + tiktoken.

Follows the token-splitting approach from:
https://www.lancedb.com/blog/chunking-techniques-with-langchain-and-llamaindex

Uses CharacterTextSplitter.from_tiktoken_encoder to count by **tokens**
(not characters), producing consistent token-sized chunks with overlap.

Blog reference (LangChain – Token Splitting using Tiktoken):
    >>> from langchain_text_splitters import CharacterTextSplitter
    >>> text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    ...     chunk_size=100, chunk_overlap=0)
    >>> texts = text_splitter.split_text(state_of_the_union)
"""

from typing import Any, Dict, List

from langchain_text_splitters import CharacterTextSplitter

from src.domain.chunk import Chunk


class FixedSizeChunkingStrategy:
    """Fixed-size chunking: LangChain CharacterTextSplitter + tiktoken encoder.

    Token-based splitting: chunk_size and overlap are measured in **tokens**,
    not characters, ensuring consistent chunk sizes for downstream LLM processing.

    Attributes:
        chunk_size: Maximum number of tokens per chunk (default 256).
        overlap:    Number of overlapping tokens between consecutive chunks (default 40).
    """

    def __init__(self, chunk_size: int = 256, overlap: int = 40):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.splitter = CharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=chunk_size,
            chunk_overlap=overlap,
        )

    def chunk(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Full chunk — no embedding needed for fixed-size strategy."""
        return self.chunk_no_embed(text, metadata)

    def chunk_no_embed(self, text: str, metadata: Dict[str, Any]) -> List[Chunk]:
        """Split text into fixed-size token chunks using LangChain — no embedding."""
        if not text or not text.strip():
            return []

        texts = self.splitter.split_text(text)
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

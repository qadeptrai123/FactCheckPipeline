"""Domain model for a single chunk produced by a chunking strategy."""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    chunk_id: str           # unique identifier, e.g. "row_1_text_0"
    text: Optional[str]     # the chunk text (None for image-only chunks)
    image_path: Optional[str] = ""   # resolved image path (None for text chunks)
    row_id: int = 0
    chunk_index: int = 0
    modality: str = "text"  # "text" or "image"
    text_chunk_id: Optional[str] = None  # for image chunks: ID of parent text chunk
    metadata: Dict[str, Any] = field(default_factory=dict)
    # 768d embedding vector (bkai-foundation-models/vietnamese-bi-encoder)
    embedding: Optional[List[float]] = None

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Return the Qdrant point payload dict.

        Payload structure:
          - root-level: chunk_id, text, image_path, row_id, chunk_index, modality,
                        text_chunk_id, source, url, title, author, date, corpus_id,
                        embedding (768d list or None)
          - metadata:   full original source document (content), row_id, etc.

        Use corpus_id to retrieve the full original text from final_corpus.csv.
        """
        return {
            "chunk_id":      self.chunk_id,
            "text":          self.text or "",
            "image_path":    self.image_path or "",
            "row_id":        self.row_id,
            "chunk_index":   self.chunk_index,
            "modality":      self.modality,
            "text_chunk_id": self.text_chunk_id or "",
            "source":        self.metadata.get("source", ""),
            "url":           self.metadata.get("url", ""),
            "title":         self.metadata.get("title", ""),
            "author":        self.metadata.get("author", ""),
            "date":          self.metadata.get("date", ""),
            "corpus_id":     self.metadata.get("row_id", 0),
            # 768d embedding vector (bkai-foundation-models/vietnamese-bi-encoder)
            "embedding":     self.embedding,
            "metadata": {
                "source":  self.metadata.get("source", ""),
                "url":     self.metadata.get("url", ""),
                "title":   self.metadata.get("title", ""),
                "author":  self.metadata.get("author", ""),
                "date":    self.metadata.get("date", ""),
                "content": self.metadata.get("content", ""),
            },
        }

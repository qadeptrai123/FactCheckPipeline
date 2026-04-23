"""Path resolution utilities for media files."""
from pathlib import Path


def resolve_media_path(relative_path: str) -> Path | None:
    """Resolve a relative media path to an absolute path.

    Checks two roots:
      1. D:/RAG-DB/FinalDataset/
      2. D:/RAG-DB/media/

    Returns the resolved Path or None if the file does not exist.
    """
    rel = Path(relative_path.strip())

    for root in [
        Path("D:/RAG-DB/FinalDataset"),
        Path("D:/RAG-DB/media"),
    ]:
        candidate = root / rel
        if candidate.exists():
            return candidate

    return None

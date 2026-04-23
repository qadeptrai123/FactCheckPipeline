"""Domain model for a single row in the corpus CSV."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class CorpusRow:
    id: int
    source: str
    url: str
    title: str
    author: str
    date: str
    content: str
    media: str = ""

    @property
    def media_paths(self) -> List[str]:
        """Parse the media column into individual file paths.

        Format: single path or multiple paths joined by ' | '.
        Example: 'media/img.jpg'  OR  'media/img1.jpg | media/img2.jpg'
        """
        if not self.media.strip():
            return []
        return [p.strip() for p in self.media.split("|")]

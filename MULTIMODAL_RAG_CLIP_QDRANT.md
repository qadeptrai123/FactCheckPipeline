# Multimodal RAG with Qdrant — CLIP-Multilingual (Shared Vector Space)

> **Goal**: Store Vietnamese news articles + their images in Qdrant, enable cross-modal search
> (text→image, image→text, text→text, image→image) using a single shared 512d CLIP embedding space.

---

## Table of Contents

1. [Architecture](#1-architecture)
2. [Prerequisites](#2-prerequisites)
3. [Collection Design](#3-collection-design)
4. [Embedding Pipeline](#4-embedding-pipeline)
5. [Upsert Workflow](#5-upsert-workflow)
6. [Query / Search Strategies](#6-query--search-strategies)
7. [Payload Schema Reference](#7-payload-schema-reference)
8. [Full Notebook Code](#8-full-notebook-code)
9. [Performance & Production Tips](#9-performance--production-tips)

---

## 1. Architecture

### The core idea: one model, one space, two modalities

```
sentence-transformers/clip-ViT-B-32-multilingual-v1
├── Text Encoder  → 512d vector  (supports Vietnamese + 50 other languages)
└── Image Encoder → 512d vector  (CLIP ViT-B/32)
                        ↓
              Shared 512d hypersphere ← dot product = cross-modal similarity
```

| Query type | Search field | Returns | How it works |
|---|---|---|---|
| Text → Text | `text_vector` | Articles, paragraphs | CLIP text encoder embeds query; matched against article text embeddings |
| Text → Image | `text_vector` | Images (with captions) | Same query vector matches `text_vector` of image chunks (captions/OCR) |
| Image → Image | `image_vector` | Similar images | Query image embedded with CLIP image encoder; matched against stored image vectors |
| Image → Text | `image_vector` | Articles describing similar images | Image query matched against article images; associated text retrieved via payload |

### Why not bkai?

| | `bkai-foundation-models/vietnamese-bi-encoder` | `clip-ViT-B-32-multilingual-v1` |
|---|---|---|
| Output dim | 768d | 512d |
| Space | Text-only | **Shared text + image** |
| Image encoding | ❌ None | ✅ CLIP image encoder |
| Cross-modal (text↔image) | ❌ Impossible | ✅ Native |
| Languages | Vietnamese (specialized) | 50+ languages incl. Vietnamese |

bkai produces better-quality *text-only* Vietnamese embeddings. CLIP-multilingual produces embeddings that *bridge text and image*. For a true multimodal RAG, the shared space is worth the tradeoff.

---

## 2. Prerequisites

### Python packages

```bash
pip install qdrant-client torch torchvision
pip install sentence-transformers
pip install transformers pillow pandas tqdm
```

### Models downloaded

| Model | Size | Notes |
|---|---|---|
| `sentence-transformers/clip-ViT-B-32-multilingual-v1` | ~600 MB | Used for **both** text and image encoding |
| `sentence-transformers/clip-ViT-B-32` | ~600 MB | Fallback if multilingual model unavailable |

### Qdrant server

```bash
# Docker
docker run -d --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant

# Or connect to existing
QDRANT_URL = "http://localhost:6333"
```

---

## 3. Collection Design

### Collection: `multimodal_news`

One collection holds everything. Two named vectors:

| Named vector | Dimension | Distance | Purpose |
|---|---|---|---|
| `text_vector` | 512 | Cosine | Text embeddings (article text, image captions) |
| `image_vector` | 512 | Cosine | Image embeddings (article images, figure images) |

### HNSW index config

```python
from qdrant_client.models import (
    VectorParams, Distance,
    HnswConfigDiff, QuantizationConfigDiff,
    OptimizersConfigDiff,
)

client.create_collection(
    collection_name="multimodal_news",
    vectors_config={
        "text_vector":  VectorParams(size=512, distance=Distance.COSINE),
        "image_vector": VectorParams(size=512, distance=Distance.COSINE),
    },
    # Fast HNSW build
    hnsw_config=HnswConfigDiff(
        m=16,          # connections per node (higher = better recall, more RAM)
        ef_construct=128,  # build quality (higher = slower build, better recall)
    ),
    # Reduce memory footprint after build
    optimizer_config=OptimizersConfigDiff(
        indexing_threshold=20000,  # start indexing after 20k points
        memmap_threshold=50000,
    ),
)
```

### When to create vs. recreate

```python
if client.collection_exists("multimodal_news"):
    print("[SKIP] 'multimodal_news' already exists")
else:
    client.create_collection(...)
```

---

## 4. Embedding Pipeline

### One model for everything

```python
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Single model — both text and image encoders
CLIP_MODEL = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
IMG_BATCH   = 16   # tune to your GPU VRAM (8 GB → batch 16 safe)
TEXT_BATCH  = 32   # text is lighter than images

print("Loading CLIP-multilingual...")
model = SentenceTransformer(CLIP_MODEL).to(device)
print(f"Embedding dimension: {model.get_embedding_dimension()}")
```

### L2-normalization utility

```python
def normalize(embs: np.ndarray) -> list[list[float]]:
    """L2-normalize embeddings. Required for cosine similarity correctness."""
    norms = (embs ** 2).sum(axis=1, keepdims=True) ** 0.5
    return (embs / (norms + 1e-9)).tolist()
```

### Text embedding

```python
def embed_text(texts: list[str], batch_size: int = TEXT_BATCH) -> list[list[float]]:
    """CLIP text encoder → 512d L2-normalised vectors."""
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, convert_to_numpy=True,
                            show_progress_bar=False, batch_size=batch_size)
        all_embs.append(embs)
        torch.cuda.empty_cache()
    return normalize(np.vstack(all_embs))
```

### Image embedding

```python
import os

def load_valid_image(path: str) -> Image.Image | None:
    """Load image, return None if file missing or corrupted."""
    if not (os.path.exists(path) and os.path.isfile(path)):
        return None
    try:
        with Image.open(path) as img:
            img.verify()
        return Image.open(path).convert("RGB")
    except Exception:
        return None

def embed_images(image_paths: list[str], batch_size: int = IMG_BATCH) -> list[list[float]]:
    """CLIP image encoder → 512d L2-normalised vectors."""
    all_embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [load_valid_image(p) for p in batch_paths]
        valid_images = [img for img in images if img is not None]
        if valid_images:
            embs = model.encode(valid_images, convert_to_numpy=True,
                                show_progress_bar=False, batch_size=batch_size)
            all_embs.append(embs)
        torch.cuda.empty_cache()

    if not all_embs:
        return []
    return normalize(np.vstack(all_embs))
```

---

## 5. Upsert Workflow

### CSV structure expected

Your chunking scripts should produce CSVs with this schema:

| Column | Description | Example |
|---|---|---|
| `chunk_id` | Unique chunk identifier | `art_001_003` |
| `modality` | `text` or `image` | `text` / `image` |
| `row_id` | Source article index | `1` |
| `chunk_index` | Order within article | `3` |
| `text` | Text content (articles) or caption (images) | `Tòa án nhân dân tỉnh...` |
| `image_path` | Path to image file (images only) | `D:/RAG-DB/media/img_001.jpg` |
| `source` | Article/website source | `vneconomy.vn` |
| `url` | Source URL | `https://...` |
| `title` | Article title | `Tuyên án vụ lừa đảo` |
| `author` | Author name | `Nguyễn Văn A` |
| `date` | Publication date | `2024-01-15` |
| `corpus_id` | Dataset split ID | `1` |
| `text_chunk_id` | For images: ID of parent text chunk | `art_001_003` |

### Embedding strategy

| Chunk type | `text_vector` | `image_vector` | Notes |
|---|---|---|---|
| Article text | ✅ Article text embedding | ✅ **Same** article text embedding | Image search on text chunks retrieves *visually similar articles* |
| Article image | ✅ Image caption/OCR text embedding | ✅ Image embedding | Text query matches captions; image query matches visuals |

> **Key design decision**: Text chunks are embedded with the **text encoder**, not the image encoder. But they are stored in **both** `text_vector` and `image_vector` so that image queries can also surface relevant text chunks (via the text-as-image trick).

### Upsert code

```python
import uuid
import csv
from qdrant_client.models import PointStruct

BATCH_SIZE = 256

def build_payload(row: dict) -> dict:
    return {
        "chunk_id":     row["chunk_id"],
        "modality":     row["modality"],
        "row_id":       int(row["row_id"]),
        "chunk_index":  int(row["chunk_index"]),
        "text":         row["text"],
        "image_path":   row.get("image_path", ""),
        "text_chunk_id": row.get("text_chunk_id", ""),
        "source":       row["source"],
        "url":          row["url"],
        "title":        row["title"],
        "author":       row.get("author", ""),
        "date":         row.get("date", ""),
        "corpus_id":    int(row["corpus_id"]),
    }

def upsert_chunks(csv_path: str, collection: str = "multimodal_news"):
    """Load CSV, embed both modalities, batch upsert to Qdrant."""

    # ── Load CSV ──────────────────────────────────────────────────────────────
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    text_rows = [r for r in rows if r["modality"] == "text"]
    img_rows  = [r for r in rows if r["modality"] == "image"]

    print(f"[{collection}] {len(text_rows)} text chunks, {len(img_rows)} image chunks")

    # ── Text chunks: embed text, store in BOTH vectors ────────────────────────
    if text_rows:
        print(f"[{collection}] Embedding {len(text_rows)} text chunks...")
        text_embs = embed_text([r["text"] for r in text_rows])

        text_points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"txt_{r['row_id']}_{r['chunk_index']}")),
                vector={
                    "text_vector":  emb,   # CLIP text encoding
                    "image_vector": emb,   # Same — enables image→text retrieval
                },
                payload=build_payload(r),
            )
            for r, emb in zip(text_rows, text_embs)
        ]

        print(f"[{collection}] Upserting {len(text_points)} text points...")
        for i in range(0, len(text_points), BATCH_SIZE):
            client.upsert(collection_name=collection, points=text_points[i:i+BATCH_SIZE])
        print(f"[{collection}] Text upsert done.")

    # ── Image chunks: embed image AND caption, store in BOTH vectors ──────────
    if img_rows:
        # Image embeddings (CLIP image encoder)
        print(f"[{collection}] Embedding {len(img_rows)} image chunks...")
        img_paths = [r["image_path"] for r in img_rows]
        img_embs   = embed_images(img_paths)

        # Caption/OCR text embeddings (CLIP text encoder)
        # Use caption if available, else OCR text, else empty string
        captions = [
            r.get("caption", "") or r.get("text", "") or ""
            for r in img_rows
        ]
        caption_embs = embed_text(captions) if any(c for c in captions) else [[0.0]*512] * len(img_rows)

        img_points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"img_{r['row_id']}_{r['chunk_index']}")),
                vector={
                    "text_vector":  cap_emb,   # Caption/OCR text (text encoder)
                    "image_vector": img_emb,   # Image pixels (image encoder)
                },
                payload=build_payload(r),
            )
            for r, img_emb, cap_emb in zip(img_rows, img_embs, caption_embs)
        ]

        print(f"[{collection}] Upserting {len(img_points)} image points...")
        for i in range(0, len(img_points), BATCH_SIZE):
            client.upsert(collection_name=collection, points=img_points[i:i+BATCH_SIZE])
        print(f"[{collection}] Image upsert done.")


# ── Run upsert for each chunking strategy ────────────────────────────────────
CHUNK_CSV = {
    "fixed_size": "D:/RAG-DB/chunking_scripts/chunks_fixed_size.csv",
    "paragraph":  "D:/RAG-DB/chunking_scripts/chunks_paragraph.csv",
    "semantic":   "D:/RAG-DB/chunking_scripts/chunks_semantic.csv",
}

for strat, csv_path in CHUNK_CSV.items():
    print(f"\n{'='*50}")
    print(f"  Strategy: {strat}")
    print(f"{'='*50}")
    upsert_chunks(csv_path, collection=f"multimodal_news_{strat}")
```

---

## 6. Query / Search Strategies

### Imports for filtering

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue, Range
```

### 6.1 Text → Text (semantic article search)

```python
def search_text(query: str, collection: str, top_k: int = 10,
                modality: str | None = "text",
                source: str | None = None) -> list:
    """
    Semantic search on text_vector.
    Optionally filter by modality (text/image) or source domain.
    """
    emb = embed_text([query])[0]

    # Build filter
    must_conditions = []
    if modality:
        must_conditions.append(
            FieldCondition(key="modality", match=MatchValue(value=modality))
        )
    if source:
        must_conditions.append(
            FieldCondition(key="source", match=MatchValue(value=source))
        )

    query_filter = Filter(must=must_conditions) if must_conditions else None

    results = client.query_points(
        collection_name=collection,
        query=emb,
        using="text_vector",
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        query_filter=query_filter,
    )
    return results.points

# Example
results = search_text("đường dây lừa đảo qua mạng xã hội", "multimodal_news_fixed_size", top_k=5)
for r in results:
    print(f"[{r.score:.4f}] {r.payload['title'][:60]}")
    print(f"  {str(r.payload.get('text',''))[:120]}...")
    print()
```

### 6.2 Text → Images (find images matching a text description)

```python
def search_images_by_text(query: str, collection: str, top_k: int = 10) -> list:
    """
    Text query on text_vector, but ONLY image modality.
    Returns image chunks whose captions/OCR match the query.
    """
    emb = embed_text([query])[0]

    results = client.query_points(
        collection_name=collection,
        query=emb,
        using="text_vector",
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        query_filter=Filter(
            must=[FieldCondition(key="modality", match=MatchValue(value="image"))]
        ),
    )
    return results.points

# Example: find images related to "tai nạn giao thông"
img_results = search_images_by_text(
    "hiện trường vụ tai nạn giao thông",
    "multimodal_news_fixed_size",
    top_k=5
)
for r in img_results:
    print(f"[{r.score:.4f}] {r.payload['image_path']}")
    print(f"  Caption: {r.payload.get('text', '')[:100]}")
```

### 6.3 Image → Images (find visually similar images)

```python
def search_similar_images(image_path: str, collection: str, top_k: int = 10) -> list:
    """
    Image query on image_vector.
    Finds visually similar images from the corpus.
    """
    emb = embed_images([image_path])[0]

    results = client.query_points(
        collection_name=collection,
        query=emb,
        using="image_vector",
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    return results.points

# Example
similar = search_similar_images(
    "D:/RAG-DB/media/accident_001.jpg",
    "multimodal_news_fixed_size",
    top_k=5
)
for r in similar:
    print(f"[{r.score:.4f}] {r.payload['image_path']}")
    print(f"  Article: {r.payload.get('title', '')}")
```

### 6.4 Image → Text (find articles describing visually similar content)

```python
def search_articles_by_image(image_path: str, collection: str, top_k: int = 10) -> list:
    """
    Image query on image_vector, but returns text chunks.
    Finds articles whose embedded images are visually similar to the query image.
    Useful for: 'find articles about this type of scene/event'.
    """
    emb = embed_images([image_path])[0]

    results = client.query_points(
        collection_name=collection,
        query=emb,
        using="image_vector",
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        query_filter=Filter(
            must=[FieldCondition(key="modality", match=MatchValue(value="text"))]
        ),
    )
    return results.points

# Example
articles = search_articles_by_image(
    "D:/RAG-DB/media/protest_scene.jpg",
    "multimodal_news_fixed_size",
    top_k=5
)
for r in articles:
    print(f"[{r.score:.4f}] {r.payload['title']}")
    print(f"  {str(r.payload.get('text',''))[:120]}...")
```

### 6.5 Hybrid: Text query with both modalities, ranked together

```python
def search_hybrid_text(query: str, collection: str, top_k: int = 10) -> list:
    """
    Text query on text_vector, NO modality filter.
    Returns both text chunks and image chunks (captions) ranked by text similarity.
    Great for: 'show me everything about topic X'.
    """
    emb = embed_text([query])[0]

    results = client.query_points(
        collection_name=collection,
        query=emb,
        using="text_vector",
        limit=top_k,
        with_payload=True,
        with_vectors=False,
    )
    return results.points

# Example
all_results = search_hybrid_text("ô tô điện VinFast", "multimodal_news_fixed_size", top_k=10)
for r in all_results:
    print(f"[{r.score:.4f}] [{r.payload['modality']}] {r.payload.get('title', r.payload.get('image_path',''))}")
```

---

## 7. Payload Schema Reference

Every point in Qdrant carries this payload for filtering and retrieval:

| Field | Type | Filterable | Description |
|---|---|---|---|
| `chunk_id` | string | ✅ | Unique chunk ID (e.g., `art_001_003`) |
| `modality` | string | ✅ | `text` or `image` |
| `row_id` | int | ✅ | Source article index |
| `chunk_index` | int | ✅ | Order within article |
| `text` | string | ❌ | Text content or caption |
| `image_path` | string | ✅ | Path to image file (images only) |
| `text_chunk_id` | string | ✅ | Parent text chunk ID (images only) |
| `source` | string | ✅ | Domain (e.g., `vneconomy.vn`) |
| `url` | string | ❌ | Source URL |
| `title` | string | ✅ | Article title |
| `author` | string | ✅ | Author name |
| `date` | string | ✅ | Publication date (`YYYY-MM-DD`) |
| `corpus_id` | int | ✅ | Dataset split ID |

### Example filtered query

```python
# Find text chunks about economics from vneconomy.vn in 2024
results = client.query_points(
    collection_name="multimodal_news_fixed_size",
    query=embed_text(["chính sách tiền tệ"])[0],
    using="text_vector",
    limit=10,
    with_payload=True,
    query_filter=Filter(
        must=[
            FieldCondition(key="modality",   match=MatchValue(value="text")),
            FieldCondition(key="source",     match=MatchValue(value="vneconomy.vn")),
            FieldCondition(key="date",       match=MatchValue(value="2024")),
        ]
    ),
)
```

---

## 8. Full Notebook Code

### Cell 1 — Imports & Config

```python
import csv
import uuid
import os

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue,
    HnswConfigDiff, OptimizersConfigDiff,
)

# ── Config ──────────────────────────────────────────────────────────────────
QDRANT_URL  = "http://localhost:6333"
COLLECTION  = "multimodal_news_fixed_size"   # or _paragraph, _semantic
CHUNK_CSV   = "D:/RAG-DB/chunking_scripts/chunks_fixed_size.csv"

CLIP_MODEL  = "sentence-transformers/clip-ViT-B-32-multilingual-v1"
VEC_DIM     = 512
BATCH_SIZE  = 256
IMG_BATCH   = 16
TEXT_BATCH  = 32
```

### Cell 2 — Connect & Create Collection

```python
client = QdrantClient(url=QDRANT_URL)
info   = client.info()
print(f"Qdrant {info.version} | collections: {[c.name for c in client.get_collections().collections]}")

if not client.collection_exists(COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "text_vector":  VectorParams(size=VEC_DIM, distance=Distance.COSINE),
            "image_vector": VectorParams(size=VEC_DIM, distance=Distance.COSINE),
        },
        hnsw_config=HnswConfigDiff(m=16, ef_construct=128),
        optimizer_config=OptimizersConfigDiff(indexing_threshold=20000),
    )
    print(f"[CREATED] '{COLLECTION}'")
else:
    print(f"[SKIP] '{COLLECTION}' already exists")
```

### Cell 3 — Load Model

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

print("Loading CLIP-multilingual...")
model = SentenceTransformer(CLIP_MODEL).to(device)
print(f"Embedding dimension: {model.get_embedding_dimension()}")
```

### Cell 4 — Embedding Functions

```python
def normalize(embs: np.ndarray) -> list[list[float]]:
    norms = (embs ** 2).sum(axis=1, keepdims=True) ** 0.5
    return (embs / (norms + 1e-9)).tolist()

def load_valid_image(path: str) -> Image.Image | None:
    if not (os.path.exists(path) and os.path.isfile(path)):
        return None
    try:
        with Image.open(path) as img:
            img.verify()
        return Image.open(path).convert("RGB")
    except Exception:
        return None

def embed_text(texts: list[str], batch_size: int = TEXT_BATCH) -> list[list[float]]:
    all_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, convert_to_numpy=True,
                            show_progress_bar=False, batch_size=batch_size)
        all_embs.append(embs)
        torch.cuda.empty_cache()
    return normalize(np.vstack(all_embs))

def embed_images(image_paths: list[str], batch_size: int = IMG_BATCH) -> list[list[float]]:
    all_embs = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        images = [load_valid_image(p) for p in batch_paths]
        valid  = [img for img in images if img is not None]
        if valid:
            embs = model.encode(valid, convert_to_numpy=True,
                                show_progress_bar=False, batch_size=batch_size)
            all_embs.append(embs)
        torch.cuda.empty_cache()
    if not all_embs:
        return []
    return normalize(np.vstack(all_embs))
```

### Cell 5 — Upsert

```python
def build_payload(row: dict) -> dict:
    return {
        "chunk_id":     row["chunk_id"],
        "modality":     row["modality"],
        "row_id":       int(row["row_id"]),
        "chunk_index":  int(row["chunk_index"]),
        "text":         row["text"],
        "image_path":   row.get("image_path", ""),
        "text_chunk_id": row.get("text_chunk_id", ""),
        "source":       row["source"],
        "url":          row["url"],
        "title":        row["title"],
        "author":       row.get("author", ""),
        "date":         row.get("date", ""),
        "corpus_id":    int(row["corpus_id"]),
    }

def upsert_all(csv_path: str, collection: str):
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    text_rows = [r for r in rows if r["modality"] == "text"]
    img_rows  = [r for r in rows if r["modality"] == "image"]
    print(f"[{collection}] {len(text_rows)} text, {len(img_rows)} image")

    # Text chunks
    if text_rows:
        print(f"  Embedding text...")
        text_embs = embed_text([r["text"] for r in text_rows])
        text_points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"txt_{r['row_id']}_{r['chunk_index']}")),
                vector={"text_vector": emb, "image_vector": emb},
                payload=build_payload(r),
            )
            for r, emb in zip(text_rows, text_embs)
        ]
        for i in range(0, len(text_points), BATCH_SIZE):
            client.upsert(collection_name=collection, points=text_points[i:i+BATCH_SIZE])
        print(f"  Text upsert done ({len(text_points)} points).")

    # Image chunks
    if img_rows:
        print(f"  Embedding images...")
        img_embs = embed_images([r["image_path"] for r in img_rows])

        captions     = [r.get("caption", "") or r.get("text", "") for r in img_rows]
        caption_embs = embed_text(captions) if any(c for c in captions) else [[0.0]*VEC_DIM]*len(img_rows)

        img_points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"img_{r['row_id']}_{r['chunk_index']}")),
                vector={"text_vector": cap_emb, "image_vector": img_emb},
                payload=build_payload(r),
            )
            for r, img_emb, cap_emb in zip(img_rows, img_embs, caption_embs)
        ]
        for i in range(0, len(img_points), BATCH_SIZE):
            client.upsert(collection_name=collection, points=img_points[i:i+BATCH_SIZE])
        print(f"  Image upsert done ({len(img_points)} points).")

# Run
upsert_all(CHUNK_CSV, COLLECTION)
```

### Cell 6 — Search Functions

```python
def search(
    query: str = None,
    image_path: str = None,
    collection: str = COLLECTION,
    vector_field: str = "text_vector",
    top_k: int = 10,
    modality: str | None = None,
    source: str | None = None,
) -> list:
    """
    Unified search. Pass either `query` (text) or `image_path`, not both.
    `vector_field` selects which named vector to search: "text_vector" or "image_vector".
    `modality` filters: "text", "image", or None (both).
    """
    assert query or image_path, "Provide either query or image_path"

    if query:
        emb = embed_text([query])[0]
    else:
        emb = embed_images([image_path])[0]

    must = []
    if modality:
        must.append(FieldCondition(key="modality", match=MatchValue(value=modality)))
    if source:
        must.append(FieldCondition(key="source", match=MatchValue(value=source)))

    query_filter = Filter(must=must) if must else None

    results = client.query_points(
        collection_name=collection,
        query=emb,
        using=vector_field,
        limit=top_k,
        with_payload=True,
        with_vectors=False,
        query_filter=query_filter,
    )
    return results.points

# Convenience wrappers
def search_text(query, collection=COLLECTION, top_k=10, modality="text"):
    return search(query=query, vector_field="text_vector",
                  collection=collection, top_k=top_k, modality=modality)

def search_image(image_path, collection=COLLECTION, top_k=10, modality="image"):
    return search(image_path=image_path, vector_field="image_vector",
                  collection=collection, top_k=top_k, modality=modality)
```

### Cell 7 — Example Queries

```python
# 1) Text → Articles
print("─── Text → Articles ───")
results = search_text("đường dây lừa đảo qua mạng", top_k=5)
for r in results:
    print(f"[{r.score:.4f}] {r.payload['title'][:60]}")
    print(f"  {str(r.payload.get('text',''))[:100]}...\n")

# 2) Text → Images (find image captions matching description)
print("─── Text → Images ───")
img_results = search_text("hiện trường tai nạn giao thông", modality="image", top_k=5)
for r in img_results:
    print(f"[{r.score:.4f}] {r.payload['image_path']}")
    print(f"  Caption: {r.payload.get('text','')[:80]}\n")

# 3) Image → Similar Images
print("─── Image → Similar Images ───")
similar = search_image("D:/RAG-DB/media/accident_001.jpg", top_k=5)
for r in similar:
    print(f"[{r.score:.4f}] {r.payload['image_path']}")
    print(f"  Source: {r.payload['title']}\n")

# 4) Image → Articles (find articles about this type of visual content)
print("─── Image → Articles ───")
articles = search(image_path="D:/RAG-DB/media/protest.jpg",
                  vector_field="image_vector", modality="text", top_k=5)
for r in articles:
    print(f"[{r.score:.4f}] {r.payload['title']}")
    print(f"  {str(r.payload.get('text',''))[:100]}...\n")
```

### Cell 8 — Collection Stats

```python
info = client.get_collection(COLLECTION)
print(f"{COLLECTION}: {info.points_count} total points")

# Count by modality (requires scroll or count API)
from qdrant_client.models import Filter, FieldCondition, MatchValue
for modality in ["text", "image"]:
    count_result = client.count(
        collection_name=COLLECTION,
        count_filter=Filter(must=[FieldCondition(key="modality", match=MatchValue(value=modality))]),
        exact=True,
    )
    print(f"  {modality}: {count_result.count}")
```

---

## 9. Performance & Production Tips

### GPU memory tuning

| Batch size | VRAM usage (est.) | GPU |
|---|---|---|
| `IMG_BATCH = 8` | ~4 GB | Any 8 GB GPU |
| `IMG_BATCH = 16` | ~7 GB | RTX 3080+, A100 |
| `IMG_BATCH = 32` | ~12+ GB | A100 40GB, multi-GPU |

```python
# Monitor GPU usage during embedding
import torch
print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.1f} GB")
```

### CPU fallback

```python
device = torch.device("cpu")  # force CPU if GPU OOM
# Reduce batch size significantly for CPU
TEXT_BATCH = 4
IMG_BATCH  = 2
```

### Qdrant quantization (reduce RAM)

After initial indexing, apply product quantization to halve memory:

```python
from qdrant_client.models import QuantizationConfigDiff, ScalarQuantization, ScalarType

client.update_collection(
    collection_name=COLLECTION,
    quantization_config=QuantizationConfigDiff(
        scalar=ScalarQuantization(
            scalar=ScalarType.FLOAT16,   # 2 bytes per dimension vs 4
            quantile=0.99,
            always_ram=True,              # keep in RAM for speed
        )
    ),
)
```

> **Warning**: Quantization can slightly reduce recall. Test with your specific data before enabling in production.

### Async batch upload for large corpora

```python
import asyncio
from qdrant_client import AsyncQdrantClient

async def upsert_async(csv_path: str, collection: str):
    async_client = AsyncQdrantClient(url=QDRANT_URL)
    # ... load and embed same as sync version ...
    await async_client.upsert(collection_name=collection, points=text_points)
    await async_client.upsert(collection_name=collection, points=img_points)

asyncio.run(upsert_async(CHUNK_CSV, COLLECTION))
```

### Incremental upsert (new articles only)

```python
def upsert_incremental(new_csv_path: str, collection: str):
    """
    Only upsert chunks whose chunk_id is NOT already in Qdrant.
    Run this on a schedule to ingest new articles without re-processing everything.
    """
    existing_ids = set()
    for p in client.scroll(collection_name=collection, limit=10000, offset=0)[0]:
        existing_ids.add(str(p.id))

    with open(new_csv_path, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["chunk_id"] not in existing_ids]

    # Only embed + upsert new rows
    # ...
```

### Monitoring during upsert

```python
from tqdm.auto import tqdm

for i in tqdm(range(0, len(text_points), BATCH_SIZE)):
    batch = text_points[i:i+BATCH_SIZE]
    client.upsert(collection_name=COLLECTION, points=batch)
    torch.cuda.empty_cache()  # between batches
```

### Query performance tuning

```python
# Increase HNSW search width for higher recall (slower)
results = client.query_points(
    collection_name=COLLECTION,
    query=emb,
    using="text_vector",
    limit=top_k,
    search_params=HnswConfigDiff(ef=256),  # higher = better recall, slower
)

# Or exact search (very slow on large collections, but perfect recall)
results = client.query_points(
    collection_name=COLLECTION,
    query=emb,
    using="text_vector",
    limit=top_k,
    search_params=HnswConfigDiff(exact=True),
)
```

---

## Summary: Query Cheat Sheet

| What you want | Method call | `vector_field` | `modality` |
|---|---|---|---|
| Find articles by topic | `search_text(query)` | `text_vector` | `text` |
| Find images by description | `search_text(query)` | `text_vector` | `image` |
| Find similar images | `search_image(path)` | `image_vector` | `image` |
| Find articles about this scene | `search(image_path=path)` | `image_vector` | `text` |
| Everything about a topic | `search_text(query)` | `text_vector` | `None` |
| Everything visually similar | `search_image(path)` | `image_vector` | `None` |

---

*Document version: 2026-04-13 | CLIP: `sentence-transformers/clip-ViT-B-32-multilingual-v1` | Qdrant: 1.17.x | Python client: `qdrant-client` v1.x*

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
QDRANT_URL = "http://localhost:6333"
COLLECTION = "semantic"
IMAGE_VARIANT = "clip_finetuned"
USE_RERANKER = True
REQUIRE_CUDA = True

TEXT_VECTOR = "text_vector"
SPARSE_VECTOR = "sparse"
IMAGE_VECTOR = "image_vector_finetuned"
BKVEC_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
IMG_MODEL_FINETUNED = PROJECT_ROOT / "models" / "clip-vit-b32-finetuned-final-final" / "best"
CROSS_ENCODER_MODEL = "namdp-ptit/ViRanker"
CROSS_ENCODER_MAX_LENGTH = 512

CANDIDATES_PER_QUERY = 10
MAX_TEXT_QUERIES = 6
MAX_VISUAL_QUERIES = 6
FINAL_TOP_K = 3
RRF_K = 60
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
BRANCH_WEIGHTS = {"text_dense": 1.00, "text_sparse": 1.00, "image_clip_finetuned": 1.00}


def normalize_text(text: Any) -> str:
    text = str(text or "").lower().strip()
    text = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in text if not unicodedata.combining(ch))


def safe_json(value: Any, default: Any) -> Any:
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def dedupe(items: list[Any], max_items: int | None = None) -> list[str]:
    out, seen = [], set()
    for item in items:
        text = str(item or "").strip()
        key = normalize_text(text)
        if text and key not in seen:
            seen.add(key)
            out.append(text)
    return out[:max_items] if max_items else out


def build_query_pack(refined: dict[str, Any]) -> dict[str, Any]:
    search_queries = safe_json(refined.get("search_queries"), {})
    claim_atoms = safe_json(refined.get("claim_atoms"), [])
    visual_observations = safe_json(refined.get("visual_observations"), [])
    retrieval_focus = safe_json(refined.get("retrieval_focus"), {})
    verification_targets = safe_json(refined.get("verification_targets"), [])

    atom_queries = []
    for atom in claim_atoms if isinstance(claim_atoms, list) else []:
        if isinstance(atom, dict):
            atom_queries.extend(atom.get("retrieval_queries", []))

    visual_terms = []
    for obs in visual_observations if isinstance(visual_observations, list) else []:
        if isinstance(obs, dict):
            visual_terms.append(obs.get("text", ""))
            visual_terms.extend(obs.get("visible_evidence", []))

    text_queries = dedupe(
        [
            refined.get("primary_retrieval_query", ""),
            refined.get("normalized_claim", ""),
            *search_queries.get("semantic", []),
            *atom_queries,
            *verification_targets,
        ],
        MAX_TEXT_QUERIES,
    )
    keyword_query = " ".join(dedupe([*search_queries.get("keywords", []), *verification_targets]))
    visual_queries = dedupe([*search_queries.get("visual", []), *visual_terms], MAX_VISUAL_QUERIES)
    if retrieval_focus.get("cross_modal", False):
        visual_queries = dedupe([*visual_queries, refined.get("primary_retrieval_query", "")], MAX_VISUAL_QUERIES)
    return {"text_queries": text_queries, "keyword_query": keyword_query, "visual_queries": visual_queries}


def sparse_vector(text: str) -> Any:
    from qdrant_client.models import SparseVector

    counts = {}
    for token in TOKEN_RE.findall(str(text or "").lower()):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(digest, "big") % 2_147_483_647
        counts[idx] = counts.get(idx, 0) + 1
    indices = sorted(counts)
    values = [1.0 + math.log(counts[idx]) for idx in indices]
    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    return SparseVector(indices=indices, values=[v / norm for v in values])


class DocumentRetriever:
    def __init__(self) -> None:
        import torch
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=QDRANT_URL)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if REQUIRE_CUDA and self.device != "cuda":
            raise RuntimeError("CUDA is required for local retrieval models, but torch.cuda.is_available() is False")
        self.device_name = torch.cuda.get_device_name(0) if self.device == "cuda" else "cpu"
        self.bkai_model = None
        self.clip_finetuned_model = None
        self.cross_encoder = None

    def diagnostics(self) -> dict[str, Any]:
        collections = [collection.name for collection in self.client.get_collections().collections]
        return {
            "qdrant_url": QDRANT_URL,
            "qdrant_collections": collections,
            "device": self.device,
            "device_name": self.device_name,
            "require_cuda": REQUIRE_CUDA,
            "bkai_model_loaded": self.bkai_model is not None,
            "clip_finetuned_model_loaded": self.clip_finetuned_model is not None,
            "cross_encoder_loaded": self.cross_encoder is not None,
            "collection": COLLECTION,
            "image_variant": IMAGE_VARIANT,
            "use_reranker": USE_RERANKER,
        }

    def preload_models(self) -> None:
        self._embed_text_bkai(["kiểm tra tải mô hình"])
        self._embed_text_clip(["kiểm tra tải mô hình ảnh"])
        if USE_RERANKER:
            self._get_cross_encoder()

    def retrieve(self, refined: dict[str, Any]) -> dict[str, Any]:
        ranked_lanes = self._retrieve_lanes(refined)
        text_items = ranked_lanes["text"][:FINAL_TOP_K]
        image_items = ranked_lanes["image"][:FINAL_TOP_K]
        top_urls = self._dedupe_urls(
            [item["payload"].get("url", "") for item in text_items]
            + [item["payload"].get("url", "") for item in image_items[:1]]
        )
        evidence_for_judge = (
            [self._serialize_item(item, rank, "text") for rank, item in enumerate(text_items, start=1)]
            + [self._serialize_item(item, 1, "image") for item in image_items[:1]]
        )
        return {
            "experiment_id": "gemini-2.5-flash__semantic__clip_finetuned__reranker_1",
            "collection": COLLECTION,
            "image_variant": IMAGE_VARIANT,
            "use_reranker": USE_RERANKER,
            "top_urls": top_urls,
            "evidence_for_judge": evidence_for_judge,
            "text_results": [self._serialize_item(item, rank, "text") for rank, item in enumerate(text_items, start=1)],
            "image_results": [self._serialize_item(item, rank, "image") for rank, item in enumerate(image_items, start=1)],
        }

    def _retrieve_lanes(self, refined: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        branches = self._generate_candidates(refined)
        text_branches = {name: records for name, records in branches.items() if name.startswith("text_")}
        image_branches = {name: records for name, records in branches.items() if name.startswith("image_")}
        return {
            "text": self._rerank_items(self._weighted_rrf(text_branches), refined, USE_RERANKER),
            "image": self._rerank_items(self._weighted_rrf(image_branches), refined, False),
        }

    def _generate_candidates(self, refined: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
        pack = build_query_pack(refined)
        branches: dict[str, list[dict[str, Any]]] = {}

        text_hits = []
        if pack["text_queries"]:
            for q, v in zip(pack["text_queries"], self._embed_text_bkai(pack["text_queries"])):
                text_hits.extend({"point": p, "query": q} for p in self._query_qdrant(v, TEXT_VECTOR, "text", CANDIDATES_PER_QUERY))
        branches["text_dense"] = text_hits

        sparse_hits = []
        if pack["keyword_query"].strip():
            sparse_hits = [
                {"point": p, "query": pack["keyword_query"]}
                for p in self._query_qdrant(sparse_vector(pack["keyword_query"]), SPARSE_VECTOR, "text", CANDIDATES_PER_QUERY * 2)
            ]
        branches["text_sparse"] = sparse_hits

        image_hits = []
        if pack["visual_queries"]:
            for q, v in zip(pack["visual_queries"], self._embed_text_clip(pack["visual_queries"])):
                image_hits.extend({"point": p, "query": q} for p in self._query_qdrant(v, IMAGE_VECTOR, "image", CANDIDATES_PER_QUERY))
        branches["image_clip_finetuned"] = image_hits
        return branches

    def _query_qdrant(self, query: Any, vector_name: str, modality: str, limit: int):
        from qdrant_client import models

        return self.client.query_points(
            collection_name=COLLECTION,
            query=query,
            using=vector_name,
            query_filter=models.Filter(must=[models.FieldCondition(key="modality", match=models.MatchValue(value=modality))]),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        ).points

    def _weighted_rrf(self, branches: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}
        for branch, records in branches.items():
            seen = set()
            for rank, record in enumerate(records, start=1):
                point = record["point"]
                key = str(point.id)
                if key in seen:
                    continue
                seen.add(key)
                item = fused.setdefault(
                    key,
                    {"point_id": key, "payload": point.payload or {}, "rrf_score": 0.0, "branches": [], "best_qdrant_score": float(point.score)},
                )
                item["rrf_score"] += BRANCH_WEIGHTS.get(branch, 1.0) / (RRF_K + rank)
                item["best_qdrant_score"] = max(item["best_qdrant_score"], float(point.score))
                item["branches"].append({"branch": branch, "rank": rank, "score": float(point.score), "query": record.get("query", "")})
        return sorted(fused.values(), key=lambda x: x["rrf_score"], reverse=True)

    def _rerank_items(self, items: list[dict[str, Any]], refined: dict[str, Any], use_reranker: bool) -> list[dict[str, Any]]:
        query = str(refined.get("primary_retrieval_query") or refined.get("normalized_claim") or "")
        has_text_candidate = any((item.get("payload") or {}).get("modality") == "text" for item in items)
        ce = self._get_cross_encoder() if use_reranker and has_text_candidate else None
        for item in items:
            payload = item["payload"]
            branch_bonus = min(len({b["branch"] for b in item["branches"]}) * 0.025, 0.10)
            reranker_boost = 0.0
            if ce is not None and payload.get("modality") == "text":
                passage = " ".join(str(payload.get(k, "")) for k in ["title", "text", "source", "date"])
                if passage.strip():
                    raw = float(ce.predict([(query, passage[:3000])])[0])
                    reranker_boost = (1.0 / (1.0 + math.exp(-raw))) * 0.25
            item["reranker_boost"] = reranker_boost
            item["final_score"] = item["rrf_score"] + branch_bonus + reranker_boost
        return sorted(items, key=lambda x: x["final_score"], reverse=True)

    def _embed_text_bkai(self, texts: list[str]) -> list[list[float]]:
        from sentence_transformers import SentenceTransformer

        if self.bkai_model is None:
            self.bkai_model = SentenceTransformer(BKVEC_MODEL, device=self.device)
        return self.bkai_model.encode(texts, batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).tolist()

    def _embed_text_clip(self, texts: list[str]) -> list[list[float]]:
        from sentence_transformers import SentenceTransformer

        if self.clip_finetuned_model is None:
            self.clip_finetuned_model = SentenceTransformer(str(IMG_MODEL_FINETUNED), device=self.device)
        return self.clip_finetuned_model.encode(texts, batch_size=16, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False).tolist()

    def _get_cross_encoder(self):
        if self.cross_encoder is None:
            from sentence_transformers import CrossEncoder

            self.cross_encoder = CrossEncoder(
                CROSS_ENCODER_MODEL,
                device=self.device,
                max_length=CROSS_ENCODER_MAX_LENGTH,
            )
        return self.cross_encoder

    def _serialize_item(self, item: dict[str, Any], rank: int, lane: str) -> dict[str, Any]:
        payload = item["payload"]
        return {
            "lane": lane,
            "rank": rank,
            "point_id": item["point_id"],
            "modality": payload.get("modality", ""),
            "final_score": item.get("final_score", 0.0),
            "rrf_score": item.get("rrf_score", 0.0),
            "reranker_boost": item.get("reranker_boost", 0.0),
            "best_qdrant_score": item.get("best_qdrant_score", 0.0),
            "branches": item.get("branches", []),
            "title": payload.get("title", ""),
            "url": payload.get("url", ""),
            "image_path": payload.get("image_path", ""),
            "text": str(payload.get("text", ""))[:1000],
        }

    def _dedupe_urls(self, urls: list[str]) -> list[str]:
        seen = set()
        out = []
        for url in urls:
            value = str(url or "").strip()
            if value and value not in seen:
                seen.add(value)
                out.append(value)
        return out

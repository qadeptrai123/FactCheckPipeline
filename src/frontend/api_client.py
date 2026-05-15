"""API client — supports both mock data and real FastAPI backend calls.

Toggle between modes via the `use_backend` parameter or the sidebar switch
in the Streamlit UI.
"""

import json
import time
import requests
from pathlib import Path
from src.frontend.utils import load_mock_data


# ── Base URL ─────────────────────────────────────────────────────────────────
API_BASE_URL = "http://127.0.0.1:8000"


# ═════════════════════════════════════════════════════════════════════════════
#  HEALTH CHECK
# ═════════════════════════════════════════════════════════════════════════════

def check_backend_health() -> bool:
    """Check if the FastAPI backend is reachable."""
    try:
        resp = requests.get(f"{API_BASE_URL}/", timeout=3)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


# ═════════════════════════════════════════════════════════════════════════════
#  BACKEND → FRONTEND RESPONSE ADAPTER
# ═════════════════════════════════════════════════════════════════════════════

def _adapt_backend_response(backend_resp: dict, claim: str,
                            config: dict, elapsed: float) -> dict:
    """Convert PipelineResponse from backend into the rich frontend format.

    The backend returns a flat structure:
        {original_query, refined_query, retrieved_context[], final_answer, status}

    The frontend expects a nested structure with preprocessing, retrieval,
    reranking, and verdict steps.  This adapter bridges the two.
    """
    retrieved = backend_resp.get("retrieved_context", [])

    # ── Build retrieval results ──────────────────────────────────────────────
    retrieval_results = []
    for i, doc in enumerate(retrieved, 1):
        if isinstance(doc, dict):
            retrieval_results.append({
                "rank": i,
                "score": doc.get("score", 0.0),
                "chunk_id": doc.get("id", f"doc_{i}"),
                "title": doc.get("content", "")[:60],
                "text": doc.get("content", ""),
                "source": doc.get("source", "backend"),
                "date": doc.get("date", "—"),
                "modality": doc.get("modality", "text"),
            })
        else:
            # retrieved_context can be list of strings
            retrieval_results.append({
                "rank": i,
                "score": 0.0,
                "chunk_id": f"doc_{i}",
                "title": str(doc)[:60],
                "text": str(doc),
                "source": "backend",
                "date": "—",
                "modality": "text",
            })

    # ── Parse final_answer for verdict info ──────────────────────────────────
    final_answer = backend_resp.get("final_answer", "")
    label = "UNVERIFIABLE"
    confidence = 0.0

    # Try to detect label from the final answer text
    answer_upper = final_answer.upper()
    if "TRUE" in answer_upper and "HALF" not in answer_upper:
        label = "TRUE"
        confidence = 0.85
    elif "FALSE" in answer_upper:
        label = "FALSE"
        confidence = 0.85
    elif "HALF" in answer_upper:
        label = "HALF-TRUE"
        confidence = 0.60

    # ── Assemble the rich frontend-compatible response ───────────────────────
    return {
        "preprocessing": {
            "status": "success",
            "duration_seconds": round(elapsed * 0.3, 2),
            "image_analysis": {
                "has_image": False,
                "description": [],
            },
            "fact_extraction": {
                "atomic_facts": [
                    f"Query gốc: {backend_resp.get('original_query', claim)}",
                    f"Query tinh chỉnh: {backend_resp.get('refined_query', '')}",
                ],
            },
            "json_output": {
                "claim": claim,
                "image_provided": False,
                "image_facts": [],
                "text_facts": [
                    f"Query gốc: {backend_resp.get('original_query', claim)}",
                    f"Query tinh chỉnh: {backend_resp.get('refined_query', '')}",
                ],
                "normalized_facts": [
                    f"Query gốc: {backend_resp.get('original_query', claim)}",
                    f"Query tinh chỉnh: {backend_resp.get('refined_query', '')}",
                ],
                "alignment": [
                    "Đã xử lý qua Backend Pipeline",
                    f"Refined: {backend_resp.get('refined_query', '')}",
                ],
                "rag_queries": [
                    backend_resp.get("refined_query", claim),
                ],
                "hyde_doc": [],
            },
        },
        "retrieval": {
            "status": "success",
            "duration_seconds": round(elapsed * 0.4, 2),
            "strategy": config.get("retrieval_method", "vector"),
            "top_k": config.get("top_k", 5),
            "results": retrieval_results,
        },
        "reranking": {
            "status": "success",
            "duration_seconds": round(elapsed * 0.1, 2),
            "model": config.get("reranker_model", "—"),
            "reranked_results": [
                {
                    "rank": r["rank"],
                    "original_rank": r["rank"],
                    "score": r["score"],
                    "chunk_id": r["chunk_id"],
                    "title": r["title"],
                }
                for r in retrieval_results
            ],
        },
        "verdict": {
            "status": backend_resp.get("status", "success"),
            "duration_seconds": round(elapsed * 0.2, 2),
            "model": config.get("inference_model", "—"),
            "label": label,
            "confidence": confidence,
            "explanation": final_answer,
            "evidence_sources": [],
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — SINGLE INFERENCE
# ═════════════════════════════════════════════════════════════════════════════

def call_single_inference(claim: str, image_path: str | None,
                          preprocessing_model: str,
                          chunking_strategy: str,
                          retrieval_model: str,
                          reranker_model: str,
                          verdict_model: str,
                          use_backend: bool = False) -> dict:
    """Call the single-inference pipeline.

    Args:
        use_backend: If True, call the real FastAPI backend.
                     If False, return mock data.
    """
    if not use_backend:
        data = load_mock_data()
        return data["single_inference"]

    # ── REAL BACKEND CALL ────────────────────────────────────────────────────
    config = {
        "refine_model": preprocessing_model,
        "retrieval_method": retrieval_model,
        "inference_model": verdict_model,
        "top_k": 5,
        "reranker_model": reranker_model,
    }

    payload = {
        "query": claim,
        "refine_model": preprocessing_model,
        "retrieval_method": retrieval_model,
        "inference_model": verdict_model,
        "top_k": 5,
    }

    t0 = time.time()
    resp = requests.post(
        f"{API_BASE_URL}/api/v1/pipeline/execute",
        json=payload,
        timeout=120,
    )
    elapsed = time.time() - t0

    resp.raise_for_status()
    backend_resp = resp.json()

    return _adapt_backend_response(backend_resp, claim, config, elapsed)


# ═════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — BENCHMARK
# ═════════════════════════════════════════════════════════════════════════════

def call_benchmark(preprocessing_model: str,
                   chunking_strategy: str,
                   retrieval_model: str,
                   reranker_model: str,
                   verdict_model: str,
                   use_backend: bool = False) -> dict:
    """Call the benchmark endpoint.

    Benchmark always returns mock data for now — the backend does not
    have a batch-evaluation endpoint yet.
    """
    data = load_mock_data()
    return data["benchmark"]


# ═════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — MODEL OPTIONS
# ═════════════════════════════════════════════════════════════════════════════

def get_model_options() -> dict:
    """Return available model options for the sidebar dropdowns."""
    data = load_mock_data()
    return data["model_options"]

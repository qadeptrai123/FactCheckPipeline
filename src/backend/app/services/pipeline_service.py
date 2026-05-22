from __future__ import annotations

import os
import tempfile
import time
import uuid
from pathlib import Path
from threading import Lock, Thread
from typing import Any

from dotenv import load_dotenv

from app.components.inferencer import LLMInferencer
from app.components.refiner import QueryRefiner
from app.components.retriever import DocumentRetriever
from app.schemas.request import PipelineRequest
from app.schemas.response import PipelinePhase, PipelineResponse


PROJECT_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(PROJECT_ROOT / ".env")

PHASE_ORDER = ["refined", "retrieval", "judge"]
PHASE_LABELS = {
    "refined": "Hiệu chỉnh đầu vào",
    "retrieval": "Truy xuất dữ liệu",
    "judge": "Đưa ra quyết định",
}


def pipeline_max_phase() -> str:
    load_dotenv(PROJECT_ROOT / ".env", override=True)
    value = os.getenv("PIPELINE_MAX_PHASE", "judge").strip().lower()
    return value if value in PHASE_ORDER else "judge"


def is_phase_enabled(phase: str) -> bool:
    return PHASE_ORDER.index(phase) <= PHASE_ORDER.index(pipeline_max_phase())


class PipelineService:
    def __init__(self) -> None:
        self._jobs: dict[str, PipelineResponse] = {}
        self._lock = Lock()
        self._component_lock = Lock()
        self._refiner: QueryRefiner | None = None
        self._retriever: DocumentRetriever | None = None
        self._inferencer: LLMInferencer | None = None
        self._preload_error: str | None = None

    def preload_components(self) -> None:
        try:
            self._get_refiner()
            if is_phase_enabled("retrieval"):
                retriever = self._get_retriever()
                retriever.preload_models()
            if is_phase_enabled("judge"):
                self._get_inferencer()
            self._preload_error = None
        except Exception as exc:
            self._preload_error = str(exc)
            raise

    def start_pipeline(self, request: PipelineRequest) -> PipelineResponse:
        job_id = str(uuid.uuid4())
        response = PipelineResponse(
            job_id=job_id,
            status="queued",
            claim_text=request.claim_text,
            phases={
                "refined": PipelinePhase(),
                "retrieval": PipelinePhase(),
                "judge": PipelinePhase(),
            },
        )
        self._set_job(response)
        Thread(target=self._run_pipeline, args=(job_id, request), daemon=True).start()
        return response

    def get_job(self, job_id: str) -> PipelineResponse | None:
        with self._lock:
            return self._jobs.get(job_id)

    def diagnostics(self) -> dict[str, Any]:
        gpu = self._gpu_diagnostics()
        with self._component_lock:
            retriever_loaded = self._retriever is not None
            retriever_diag = self._retriever.diagnostics() if self._retriever is not None else None
            return {
                "gpu": gpu,
                "openrouter": {
                    "configured": bool(os.getenv("OPENROUTER_API_KEY", "").strip()),
                    "base_url": "https://openrouter.ai/api/v1",
                    "model": "google/gemini-2.5-flash",
                },
                "pipeline": {
                    "max_phase": pipeline_max_phase(),
                    "enabled_phases": [phase for phase in PHASE_ORDER if is_phase_enabled(phase)],
                    "phase_labels": PHASE_LABELS,
                },
                "refiner_loaded": self._refiner is not None,
                "retriever_loaded": retriever_loaded,
                "inferencer_loaded": self._inferencer is not None,
                "preload_error": self._preload_error,
                "retriever": retriever_diag,
                "active_jobs": len([job for job in self._jobs.values() if job.status in {"queued", "running"}]),
                "total_jobs": len(self._jobs),
            }

    def _run_pipeline(self, job_id: str, request: PipelineRequest) -> None:
        temp_image: Path | None = None
        try:
            self._update_job(job_id, status="running")
            image_paths = []
            if request.claim_image:
                temp_image = self._write_data_url_image(request.claim_image)
                image_paths = [temp_image]

            self._set_phase(job_id, "refined", "running")
            refiner = self._get_refiner()
            refined, duration = self._timed(lambda: refiner.refine(request.claim_text, image_paths))
            self._set_phase(job_id, "refined", "done", refined, duration)
            if pipeline_max_phase() == "refined":
                self._finish_at_phase(job_id, "refined", refined=refined)
                return

            self._set_phase(job_id, "retrieval", "running")
            retriever = self._get_retriever()
            retrieval, duration = self._timed(lambda: retriever.retrieve(refined))
            self._set_phase(job_id, "retrieval", "done", retrieval, duration)
            if pipeline_max_phase() == "retrieval":
                self._finish_at_phase(job_id, "retrieval", refined=refined, retrieval=retrieval)
                return

            self._set_phase(job_id, "judge", "running")
            inferencer = self._get_inferencer()
            judge, duration = self._timed(lambda: inferencer.generate(refined, retrieval))
            self._set_phase(job_id, "judge", "done", judge, duration)

            self._update_job(job_id, status="done", result=self._build_result(refined, retrieval, judge))
        except Exception as exc:
            self._mark_error(job_id, str(exc))
        finally:
            if temp_image and temp_image.exists():
                temp_image.unlink(missing_ok=True)

    def _build_result(self, refined: dict[str, Any], retrieval: dict[str, Any], judge: dict[str, Any]) -> dict[str, Any]:
        return {
            "normalized_claim": judge.get("normalized_claim") or refined.get("normalized_claim"),
            "verdict": judge.get("verdict"),
            "explanation": judge.get("explanation"),
            "thought_process": judge.get("thought_process", {}),
            "top3_urls_used": judge.get("top3_urls_used", []),
            "map_results": judge.get("map_results", []),
            "final_error": judge.get("final_error", ""),
            "retrieval_setup": {
                "experiment_id": retrieval.get("experiment_id"),
                "collection": retrieval.get("collection"),
                "image_variant": retrieval.get("image_variant"),
                "use_reranker": retrieval.get("use_reranker"),
            },
        }

    def _finish_at_phase(
        self,
        job_id: str,
        phase: str,
        refined: dict[str, Any],
        retrieval: dict[str, Any] | None = None,
    ) -> None:
        result: dict[str, Any] = {
            "stopped_at_phase": phase,
            "normalized_claim": refined.get("normalized_claim"),
            "primary_retrieval_query": refined.get("primary_retrieval_query"),
            "query_pack": refined.get("search_queries", {}),
        }
        if retrieval:
            result["retrieval_setup"] = {
                "experiment_id": retrieval.get("experiment_id"),
                "collection": retrieval.get("collection"),
                "image_variant": retrieval.get("image_variant"),
                "use_reranker": retrieval.get("use_reranker"),
            }
            result["top_urls"] = retrieval.get("top_urls", [])
        self._update_job(job_id, status="done", result=result)

    def _set_job(self, response: PipelineResponse) -> None:
        with self._lock:
            self._jobs[response.job_id] = response

    def _update_job(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs[job_id]
            updated = job.model_copy(update=changes)
            self._jobs[job_id] = updated

    def _set_phase(self, job_id: str, phase: str, status: str, data: dict[str, Any] | None = None, duration: float | None = None, error: str | None = None) -> None:
        with self._lock:
            job = self._jobs[job_id]
            phases = dict(job.phases)
            phases[phase] = PipelinePhase(status=status, data=data or phases[phase].data, duration_seconds=duration, error=error)
            self._jobs[job_id] = job.model_copy(update={"phases": phases})

    def _mark_error(self, job_id: str, error: str) -> None:
        with self._lock:
            job = self._jobs[job_id]
            phases = dict(job.phases)
            for key, phase in phases.items():
                if phase.status == "running":
                    phases[key] = phase.model_copy(update={"status": "error", "error": error})
            self._jobs[job_id] = job.model_copy(update={"status": "error", "error": error, "phases": phases})

    def _timed(self, fn):
        start = time.perf_counter()
        result = fn()
        return result, round(time.perf_counter() - start, 3)

    def _write_data_url_image(self, value: str) -> Path:
        import base64

        if "," in value and value.strip().startswith("data:"):
            header, payload = value.split(",", 1)
            suffix = ".png" if "png" in header else ".jpg"
            data = base64.b64decode(payload)
        else:
            suffix = ".jpg"
            data = base64.b64decode(value)

        handle = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            handle.write(data)
            return Path(handle.name)
        finally:
            handle.close()

    def _gpu_diagnostics(self) -> dict[str, Any]:
        try:
            import torch

            is_available = torch.cuda.is_available()
            return {
                "torch_version": torch.__version__,
                "cuda_available": is_available,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0) if is_available else None,
            }
        except Exception as exc:
            return {"error": str(exc)}

    def _get_refiner(self) -> QueryRefiner:
        with self._component_lock:
            if self._refiner is None:
                self._refiner = QueryRefiner()
            return self._refiner

    def _get_retriever(self) -> DocumentRetriever:
        with self._component_lock:
            if self._retriever is None:
                self._retriever = DocumentRetriever()
            return self._retriever

    def _get_inferencer(self) -> LLMInferencer:
        with self._component_lock:
            if self._inferencer is None:
                self._inferencer = LLMInferencer()
            return self._inferencer

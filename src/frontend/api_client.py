"""FastAPI client for the single-sample fact-check app."""

from __future__ import annotations

import base64
import time
from typing import Any

import requests


API_BASE_URL = "http://127.0.0.1:8000"


def check_backend_health() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=3)
        return response.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def get_runtime_status() -> dict[str, Any] | None:
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/pipeline/debug/runtime", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


def encode_uploaded_image(uploaded_file) -> str | None:
    if uploaded_file is None:
        return None
    mime = uploaded_file.type or "image/jpeg"
    encoded = base64.b64encode(uploaded_file.getvalue()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def start_pipeline(claim_text: str, uploaded_file) -> str:
    payload = {
        "claim_text": claim_text,
        "claim_image": encode_uploaded_image(uploaded_file),
    }
    response = requests.post(f"{API_BASE_URL}/api/v1/pipeline/execute", json=payload, timeout=20)
    response.raise_for_status()
    return response.json()["job_id"]


def get_pipeline_status(job_id: str) -> dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}/api/v1/pipeline/status/{job_id}", timeout=10)
    response.raise_for_status()
    return response.json()


def wait_for_pipeline(job_id: str, poll_seconds: float = 2.0, max_seconds: int = 900):
    started = time.time()
    while True:
        status = get_pipeline_status(job_id)
        yield status
        if status.get("status") in {"done", "error"}:
            return
        if time.time() - started > max_seconds:
            raise TimeoutError("Backend pipeline did not finish before timeout")
        time.sleep(poll_seconds)

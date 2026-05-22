"""
FactCheck Pipeline Streamlit frontend.

Run from the project root:
    streamlit run src/frontend/app.py
"""

import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.frontend.api_client import check_backend_health, get_runtime_status
from src.frontend.config import CUSTOM_CSS, PAGE_CONFIG, PHASE_LABELS, get_enabled_phases, get_pipeline_max_phase
from src.frontend.tab_inference import render
from src.frontend.utils import render_header


st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
render_header()

with st.sidebar:
    st.markdown("### Trạng thái hệ thống")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    backend_ok = check_backend_health()
    if backend_ok:
        st.success("Máy chủ đang chạy")
    else:
        st.error("Máy chủ chưa kết nối")
        st.caption("Chạy trong `src/backend`: `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`")

    runtime = get_runtime_status() if backend_ok else None
    if runtime:
        openrouter = runtime.get("openrouter", {})
        gpu = runtime.get("gpu", {})
        retriever = runtime.get("retriever") or {}
        st.markdown("### Dịch vụ")
        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.caption(f"OpenRouter: {'đã cấu hình' if openrouter.get('configured') else 'chưa cấu hình'}")
        st.caption(f"GPU: {'sẵn sàng' if gpu.get('cuda_available') else 'chưa sẵn sàng'}")
        if gpu.get("device_name"):
            st.caption(str(gpu["device_name"]))
        st.caption(f"Bi-encoder: {'đã tải' if retriever.get('bkai_model_loaded') else 'chưa tải'}")
        st.caption(f"CLIP fine-tuned: {'đã tải' if retriever.get('clip_finetuned_model_loaded') else 'chưa tải'}")
        st.caption(f"Reranker: {'đã tải' if retriever.get('cross_encoder_loaded') else 'chưa tải'}")

    st.markdown("### Cấu hình pipeline")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    max_phase = get_pipeline_max_phase()
    enabled = get_enabled_phases()
    runtime_pipeline = (runtime or {}).get("pipeline", {}) if backend_ok else {}
    backend_max_phase = runtime_pipeline.get("max_phase")
    st.caption(f"Chạy tới phase: {PHASE_LABELS[max_phase]}")
    st.caption("Các phase bật: " + " -> ".join(PHASE_LABELS[phase] for phase in enabled))
    if backend_max_phase and backend_max_phase != max_phase:
        st.warning(f"Backend đang chạy tới phase: {PHASE_LABELS.get(backend_max_phase, backend_max_phase)}")
    st.caption("Hiệu chỉnh: google/gemini-2.5-flash")
    if "retrieval" in enabled:
        st.caption("Truy xuất: semantic + clip_finetuned + reranker")
    if "judge" in enabled:
        st.caption("Quyết định: google/gemini-2.5-flash")


render()

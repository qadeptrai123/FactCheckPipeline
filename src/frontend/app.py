"""
FactCheck Pipeline — Streamlit Frontend
────────────────────────────────────────
Run from the project root:
    streamlit run src/frontend/app.py
"""

import sys
from pathlib import Path

# ── Ensure project root is on sys.path so `src.*` imports work ───────────────
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
from src.frontend.config import PAGE_CONFIG, CUSTOM_CSS
from src.frontend.utils import render_header
from src.frontend.api_client import get_model_options, check_backend_health
from src.frontend import tab_inference, tab_benchmark, tab_dashboard

# ── Page setup ───────────────────────────────────────────────────────────────
st.set_page_config(**PAGE_CONFIG)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
render_header()

# ── Sidebar: Backend connection toggle ───────────────────────────────────────
with st.sidebar:
    st.markdown("### Kết nối hệ thống")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    use_backend = st.toggle(
        "Sử dụng Backend API",
        value=False,
        key="toggle_backend",
        help="Bật để gửi request đến FastAPI backend tại http://127.0.0.1:8000",
    )

    if use_backend:
        is_healthy = check_backend_health()
        if is_healthy:
            st.markdown(
                '<div style="display:flex;align-items:center;gap:.4rem;margin:.3rem 0">'
                '<span style="width:8px;height:8px;border-radius:50%;'
                'background:#10b981;display:inline-block"></span>'
                '<span style="color:#10b981;font-size:.85rem;font-weight:500">'
                'Backend đang hoạt động</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="display:flex;align-items:center;gap:.4rem;margin:.3rem 0">'
                '<span style="width:8px;height:8px;border-radius:50%;'
                'background:#f43f5e;display:inline-block"></span>'
                '<span style="color:#f43f5e;font-size:.85rem;font-weight:500">'
                'Backend không phản hồi</span></div>',
                unsafe_allow_html=True,
            )
            st.caption("Kiểm tra: `uvicorn app.main:app` trong `src/backend/`")
    else:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:.4rem;margin:.3rem 0">'
            '<span style="width:8px;height:8px;border-radius:50%;'
            'background:#f59e0b;display:inline-block"></span>'
            '<span style="color:#f59e0b;font-size:.85rem;font-weight:500">'
            'Chế độ Mock Data</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")

# ── Load model options (used by tabs) ────────────────────────────────────────
model_opts = get_model_options()

# ── Tabs ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Kiểm chứng",
    "Đánh giá",
    "Tổng quan",
])

with tab1:
    tab_inference.render(model_opts, use_backend=use_backend)

with tab2:
    tab_benchmark.render(model_opts)

with tab3:
    tab_dashboard.render()

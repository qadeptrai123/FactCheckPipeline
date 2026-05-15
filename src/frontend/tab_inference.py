"""Tab 1 — Single Inference: nhập claim → chạy pipeline → hiển thị kết quả."""

import streamlit as st
from src.frontend.utils import (
    render_step_card,
    render_metric_box,
    render_score_bar,
    simulate_progress,
    get_verdict_class,
    get_verdict_color,
    get_verdict_icon,
)
from src.frontend.api_client import call_single_inference


def _render_preprocessing(data: dict):
    """Render the preprocessing step results."""
    pre = data["preprocessing"]

    # ── Image Analysis ───────────────────────────────────────────────────────
    img_analysis = pre["image_analysis"]
    if img_analysis["has_image"]:
        bullets = "".join(
            f'<div class="fact-chip">{d}</div>' for d in img_analysis["description"]
        )
        render_step_card(
            1,
            "Phân tích hình ảnh",
            f'<div style="margin-top:.3rem">{bullets}</div>',
            "badge-blue",
        )
    else:
        render_step_card(
            1,
            "Phân tích hình ảnh",
            '<p style="color:#64748b;font-size:.88rem">Không có hình ảnh đính kèm.</p>',
            "badge-blue",
        )

    # ── Fact Extraction ──────────────────────────────────────────────────────
    facts = pre["fact_extraction"]["atomic_facts"]
    facts_html = "".join(f'<div class="fact-chip">{f}</div>' for f in facts)
    render_step_card(
        2,
        "Trích xuất sự kiện",
        f'<div style="margin-top:.3rem">{facts_html}</div>',
        "badge-green",
    )

    # ── JSON / RAG Queries ───────────────────────────────────────────────────
    json_out = pre["json_output"]
    alignment = json_out["alignment"]
    alignment_text = " — ".join(alignment)
    rag_chips = "".join(f'<div class="fact-chip">{q}</div>' for q in json_out["rag_queries"])

    # Build HyDE collapsible sections (pure HTML details/summary)
    hyde_sections = ""
    for i, doc in enumerate(json_out.get("hyde_doc", []), 1):
        hyde_sections += (
            f'<details style="margin:.4rem 0;cursor:pointer">'
            f'<summary style="color:#c7d2fe;font-size:.88rem;font-weight:500">'
            f'HyDE Document {i}</summary>'
            f'<p style="color:#94a3b8;font-size:.85rem;padding:.5rem .8rem;'
            f'line-height:1.6;margin:0">{doc}</p></details>'
        )

    # Render entire step 3 as a single flat HTML string
    st.markdown(
        '<div class="step-card">'
        '<div class="step-title">'
        '<span class="step-badge badge-amber">3</span>'
        'Tổng hợp truy vấn tìm kiếm'
        '</div>'
        f'<div style="margin-bottom:.6rem">'
        f'<span style="color:#94a3b8;font-size:.85rem">Đối sánh văn bản – hình ảnh: </span>'
        f'<span style="color:#e2e8f0;font-size:.9rem;font-weight:500">{alignment_text}</span>'
        f'</div>'
        f'<div style="margin-bottom:.5rem">'
        f'<span style="color:#94a3b8;font-size:.85rem">Truy vấn RAG:</span>'
        f'<div style="margin-top:.3rem">{rag_chips}</div>'
        f'</div>'
        f'{hyde_sections}'
        '</div>',
        unsafe_allow_html=True,
    )


def _render_retrieval(data: dict):
    """Render retrieval results."""
    ret = data["retrieval"]
    results = ret["results"]

    rows_html = ""
    for r in results:
        mod_badge = (
            '<span style="background:rgba(99,102,241,.15);color:#818cf8;'
            'padding:2px 8px;border-radius:6px;font-size:.75rem;font-weight:600">'
            f'{r["modality"].upper()}</span>'
        )
        rows_html += f"""
        <div class="retrieval-row">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.35rem">
                <div style="display:flex;align-items:center;gap:.5rem">
                    <span style="color:#818cf8;font-weight:700;font-size:.85rem">#{r["rank"]}</span>
                    {mod_badge}
                    <span style="color:#e2e8f0;font-weight:500;font-size:.9rem">{r["title"][:55]}</span>
                </div>
                <span style="color:#10b981;font-weight:600;font-size:.9rem">{r["score"]:.4f}</span>
            </div>
            <p style="color:#94a3b8;font-size:.83rem;margin:0">{r["text"][:150]}...</p>
            <div style="display:flex;gap:.8rem;margin-top:.3rem">
                <span style="color:#64748b;font-size:.78rem">{r["source"]}</span>
                <span style="color:#64748b;font-size:.78rem">{r["date"]}</span>
            </div>
        </div>
        """
    info = (
        f'<span style="color:#94a3b8;font-size:.85rem">'
        f'Chiến lược: <b style="color:#e2e8f0">{ret["strategy"]}</b> · '
        f'Top-K: <b style="color:#e2e8f0">{ret["top_k"]}</b> · '
        f'Thời gian: <b style="color:#e2e8f0">{ret["duration_seconds"]:.2f}s</b></span>'
    )
    render_step_card(
        4,
        "Truy xuất tài liệu liên quan",
        f"{info}<div style='margin-top:.7rem'>{rows_html}</div>",
        "badge-purple",
    )


def _render_reranking(data: dict):
    """Render reranking comparison."""
    rerank = data["reranking"]
    results = rerank["reranked_results"]

    rows_html = ""
    for r in results:
        delta = r["original_rank"] - r["rank"]
        if delta > 0:
            arrow = f'<span style="color:#10b981">▲{delta}</span>'
        elif delta < 0:
            arrow = f'<span style="color:#f43f5e">▼{abs(delta)}</span>'
        else:
            arrow = '<span style="color:#64748b">—</span>'

        rows_html += f"""
        <div class="retrieval-row">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <div style="display:flex;align-items:center;gap:.6rem">
                    <span style="color:#818cf8;font-weight:700;font-size:.85rem">#{r["rank"]}</span>
                    <span style="color:#e2e8f0;font-size:.9rem">{r["title"][:55]}</span>
                    {arrow}
                </div>
                <span style="color:#10b981;font-weight:600;font-size:.9rem">{r["score"]:.4f}</span>
            </div>
        </div>
        """
    info = (
        f'<span style="color:#94a3b8;font-size:.85rem">'
        f'Reranker: <b style="color:#e2e8f0">{rerank["model"]}</b> · '
        f'Thời gian: <b style="color:#e2e8f0">{rerank["duration_seconds"]:.2f}s</b></span>'
    )
    render_step_card(
        5,
        "Sắp xếp lại độ liên quan",
        f"{info}<div style='margin-top:.7rem'>{rows_html}</div>",
        "badge-rose",
    )


def _render_verdict(data: dict):
    """Render the final verdict."""
    v = data["verdict"]
    label = v["label"]
    css_class = get_verdict_class(label)
    color = get_verdict_color(label)
    icon = get_verdict_icon(label)

    sources_html = ""
    for s in v.get("evidence_sources", []):
        rel_color = "#10b981" if s["relevance"] == "high" else "#f59e0b"
        sources_html += (
            f'<div style="display:flex;align-items:center;gap:.5rem;margin:.25rem 0">'
            f'<span style="color:{rel_color};font-size:.78rem;font-weight:600">'
            f'{s["relevance"].upper()}</span>'
            f'<span style="color:#e2e8f0;font-size:.85rem">{s["title"]}</span>'
            f'<span style="color:#64748b;font-size:.78rem">({s["source"]})</span>'
            f'</div>'
        )

    st.markdown(
        f"""
        <div class="{css_class}">
            <div class="verdict-label" style="color:{color}">{icon} {label}</div>
            <p style="color:#94a3b8;font-size:.88rem;margin:.2rem 0 .7rem">
                Độ tin cậy: <b style="color:{color}">{v["confidence"]*100:.0f}%</b> ·
                Model: <b style="color:#e2e8f0">{v["model"]}</b> ·
                Thời gian: <b style="color:#e2e8f0">{v["duration_seconds"]:.2f}s</b>
            </p>
            <p style="color:#cbd5e1;font-size:.9rem;text-align:left;line-height:1.6;
                       max-width:800px;margin:0 auto .8rem">
                {v["explanation"]}
            </p>
            <div style="text-align:left;max-width:800px;margin:0 auto">
                <p style="color:#94a3b8;font-size:.82rem;margin-bottom:.3rem;font-weight:600">
                    Nguồn bằng chứng:</p>
                {sources_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN RENDER
# ═════════════════════════════════════════════════════════════════════════════

def render(model_opts: dict, use_backend: bool = False):
    """Render Tab 1 — Single Inference."""

    # ── Input area ───────────────────────────────────────────────────────────
    st.markdown("#### Thông tin đầu vào")

    claim = st.text_area(
        "Tuyên bố cần kiểm chứng",
        height=90,
        placeholder="Ví dụ: Công an tỉnh Thái Bình đã khởi tố 10 đối tượng trong đường dây lừa đảo...",
        key="claim_input",
    )

    uploaded_file = st.file_uploader(
        "Hình ảnh đính kèm (tuỳ chọn)",
        type=["jpg", "jpeg", "png", "webp"],
        key="image_upload",
    )

    # ── Sidebar model selectors ──────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Tuỳ chỉnh mô hình")
        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        preprocess_opts = model_opts["preprocessing"]
        preprocess_model = st.selectbox(
            "Mô hình tiền xử lý",
            options=[o["id"] for o in preprocess_opts],
            format_func=lambda x: next(o["name"] for o in preprocess_opts if o["id"] == x),
            key="sb_preprocess",
        )

        chunking_opts = model_opts["chunking_strategy"]
        chunking_strategy = st.selectbox(
            "Phân đoạn văn bản",
            options=[o["id"] for o in chunking_opts],
            format_func=lambda x: next(o["name"] for o in chunking_opts if o["id"] == x),
            key="sb_chunking",
        )

        retrieval_opts = model_opts["retrieval"]
        retrieval_model = st.selectbox(
            "Mô hình truy xuất",
            options=[o["id"] for o in retrieval_opts],
            format_func=lambda x: next(o["name"] for o in retrieval_opts if o["id"] == x),
            key="sb_retrieval",
        )

        reranker_opts = model_opts["reranker"]
        reranker_model = st.selectbox(
            "Sắp xếp lại kết quả",
            options=[o["id"] for o in reranker_opts],
            format_func=lambda x: next(o["name"] for o in reranker_opts if o["id"] == x),
            key="sb_reranker",
        )

        verdict_opts = model_opts["verdict"]
        verdict_model = st.selectbox(
            "Mô hình kết luận",
            options=[o["id"] for o in verdict_opts],
            format_func=lambda x: next(o["name"] for o in verdict_opts if o["id"] == x),
            key="sb_verdict",
        )

    # ── Execute button ───────────────────────────────────────────────────────
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        execute = st.button("Thực thi", type="primary", use_container_width=True)

    if not execute:
        st.markdown(
            '<p style="color:#64748b;text-align:center;margin-top:2rem">'
            'Nhập tuyên bố cần kiểm chứng và nhấn <b>"Thực thi"</b> để bắt đầu.</p>',
            unsafe_allow_html=True,
        )
        return

    if not claim.strip():
        st.warning("Vui lòng nhập tuyên bố trước khi thực thi.")
        return

    # ── Run pipeline ──────────────────────────────────────────────────────────
    image_path = None
    if uploaded_file is not None:
        image_path = uploaded_file.name

    if use_backend:
        # Real backend — show spinner while waiting for API response
        with st.spinner("Đang gửi yêu cầu đến Backend API..."):
            try:
                result = call_single_inference(
                    claim=claim,
                    image_path=image_path,
                    preprocessing_model=preprocess_model,
                    chunking_strategy=chunking_strategy,
                    retrieval_model=retrieval_model,
                    reranker_model=reranker_model,
                    verdict_model=verdict_model,
                    use_backend=True,
                )
            except Exception as e:
                st.error(f"Lỗi kết nối Backend: {e}")
                st.info("Vui lòng kiểm tra Backend API đang chạy tại http://127.0.0.1:8000")
                return
    else:
        # Mock mode — simulate progress bars
        with st.spinner(""):
            simulate_progress("Bước 1/5 — Phân tích hình ảnh...", seconds=1.0)
            simulate_progress("Bước 2/5 — Trích xuất sự kiện...", seconds=0.8)
            simulate_progress("Bước 3/5 — Tổng hợp truy vấn tìm kiếm...", seconds=0.7)
            simulate_progress("Bước 4/5 — Truy xuất tài liệu liên quan...", seconds=1.0)
            simulate_progress("Bước 5/5 — Đưa ra kết luận..", seconds=0.8)

        result = call_single_inference(
            claim=claim,
            image_path=image_path,
            preprocessing_model=preprocess_model,
            chunking_strategy=chunking_strategy,
            retrieval_model=retrieval_model,
            reranker_model=reranker_model,
            verdict_model=verdict_model,
            use_backend=False,
        )

    # ── Display results ──────────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("#### Kết quả phân tích")

    # Timing summary
    total_time = (
        result["preprocessing"]["duration_seconds"]
        + result["retrieval"]["duration_seconds"]
        + result["reranking"]["duration_seconds"]
        + result["verdict"]["duration_seconds"]
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_box(f'{result["preprocessing"]["duration_seconds"]:.1f}s', "Tiền xử lý")
    with c2:
        render_metric_box(f'{result["retrieval"]["duration_seconds"]:.1f}s', "Truy xuất")
    with c3:
        render_metric_box(f'{result["reranking"]["duration_seconds"]:.1f}s', "Sắp xếp lại")
    with c4:
        render_metric_box(f"{total_time:.1f}s", "Tổng thời gian")

    st.markdown("")

    # Pipeline steps
    _render_preprocessing(result)
    _render_retrieval(result)
    _render_reranking(result)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("#### Kết luận")
    _render_verdict(result)

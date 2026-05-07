"""Tab 2 — Benchmark: chạy đánh giá trên toàn bộ dataset."""

import streamlit as st
import pandas as pd
import numpy as np
from src.frontend.utils import render_metric_box, render_score_bar, simulate_progress
from src.frontend.api_client import call_benchmark


def _render_confusion_matrix(matrix: list, labels: list):
    """Render confusion matrix as a styled DataFrame."""
    df = pd.DataFrame(matrix, index=labels, columns=labels)
    df.index.name = "Thực tế"
    df.columns.name = "Dự đoán"

    def _highlight(val):
        return ""

    st.dataframe(
        df.style.background_gradient(cmap="Blues", axis=None).format("{:.0f}"),
        use_container_width=True,
        height=220,
    )


def _render_per_label_chart(per_label: dict):
    """Render per-label metrics as a grouped bar chart via st.bar_chart."""
    rows = []
    for label, metrics in per_label.items():
        rows.append({
            "Nhãn": label,
            "Độ chuẩn xác": metrics["precision"],
            "Độ bao phủ": metrics["recall"],
            "F1": metrics["f1"],
        })
    df = pd.DataFrame(rows).set_index("Nhãn")
    st.bar_chart(df, height=320)


def render(model_opts: dict):
    """Render Tab 2 — Benchmark."""

    st.markdown("#### Đánh giá hiệu suất trên tập dữ liệu")
    st.markdown(
        '<p style="color:#94a3b8;font-size:.9rem">'
        "Thực thi pipeline trên toàn bộ tập kiểm thử và đo lường các chỉ số đánh giá.</p>",
        unsafe_allow_html=True,
    )

    # ── Sidebar selectors (reuse same keys with prefix) ──────────────────────
    with st.sidebar:
        st.markdown("---")
        st.markdown("### Tuỳ chỉnh Benchmark")

        bm_preprocess = st.selectbox(
            "Mô hình tiền xử lý",
            options=[o["id"] for o in model_opts["preprocessing"]],
            format_func=lambda x: next(
                o["name"] for o in model_opts["preprocessing"] if o["id"] == x
            ),
            key="bm_preprocess",
        )
        bm_chunking = st.selectbox(
            "Phân đoạn văn bản",
            options=[o["id"] for o in model_opts["chunking_strategy"]],
            format_func=lambda x: next(
                o["name"] for o in model_opts["chunking_strategy"] if o["id"] == x
            ),
            key="bm_chunking",
        )
        bm_retrieval = st.selectbox(
            "Mô hình truy xuất",
            options=[o["id"] for o in model_opts["retrieval"]],
            format_func=lambda x: next(
                o["name"] for o in model_opts["retrieval"] if o["id"] == x
            ),
            key="bm_retrieval",
        )
        bm_reranker = st.selectbox(
            "Sắp xếp lại kết quả",
            options=[o["id"] for o in model_opts["reranker"]],
            format_func=lambda x: next(
                o["name"] for o in model_opts["reranker"] if o["id"] == x
            ),
            key="bm_reranker",
        )
        bm_verdict = st.selectbox(
            "Mô hình kết luận",
            options=[o["id"] for o in model_opts["verdict"]],
            format_func=lambda x: next(
                o["name"] for o in model_opts["verdict"] if o["id"] == x
            ),
            key="bm_verdict",
        )

    # ── Run button ───────────────────────────────────────────────────────────
    col_btn, _ = st.columns([1, 3])
    with col_btn:
        run_bm = st.button(
            "Bắt đầu đánh giá", type="primary", use_container_width=True, key="run_bm"
        )

    if not run_bm:
        st.markdown(
            '<p style="color:#64748b;text-align:center;margin-top:2rem">'
            'Nhấn <b>"Bắt đầu đánh giá"</b> để thực thi trên toàn bộ tập kiểm thử.</p>',
            unsafe_allow_html=True,
        )
        return

    # ── Simulate long benchmark ──────────────────────────────────────────────
    simulate_progress("Đang đánh giá trên tập kiểm thử...", seconds=3.0, steps=30)

    data = call_benchmark(
        preprocessing_model=bm_preprocess,
        chunking_strategy=bm_chunking,
        retrieval_model=bm_retrieval,
        reranker_model=bm_reranker,
        verdict_model=bm_verdict,
    )

    # ── Dataset info ─────────────────────────────────────────────────────────
    ds = data["dataset_info"]
    st.markdown(
        f"""
        <div class="step-card">
            <div class="step-title">
                <span class="step-badge badge-blue">D</span>
                Tập dữ liệu: {ds["name"]}
            </div>
            <p style="color:#94a3b8;font-size:.88rem;margin:0">
                Tổng số mẫu: <b style="color:#e2e8f0">{ds["total_samples"]}</b> ·
                Thời gian thực thi: <b style="color:#e2e8f0">{data["duration_minutes"]:.1f} phút</b>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Overall metrics ──────────────────────────────────────────────────────
    st.markdown("##### Các chỉ số tổng quan")
    overall = data["results"]["overall"]
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric_box(f'{overall["accuracy"]*100:.1f}%', "Độ chính xác")
    with c2:
        render_metric_box(f'{overall["macro_f1"]*100:.1f}%', "Macro F1")
    with c3:
        render_metric_box(f'{overall["weighted_f1"]*100:.1f}%', "Weighted F1")
    with c4:
        render_metric_box(f'{overall["precision"]*100:.1f}%', "Độ chuẩn xác")
    with c5:
        render_metric_box(f'{overall["recall"]*100:.1f}%', "Độ bao phủ")

    st.markdown("")

    # ── Per-label metrics ────────────────────────────────────────────────────
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("##### Phân bổ hiệu suất theo nhãn")
        _render_per_label_chart(data["results"]["per_label"])

    with right:
        st.markdown("##### Đối chiếu thực tế — dự đoán")
        _render_confusion_matrix(
            data["results"]["confusion_matrix"],
            data["results"]["confusion_labels"],
        )

    # ── Detailed table ───────────────────────────────────────────────────────
    st.markdown("##### Bảng chi tiết theo nhãn")
    rows = []
    for label, m in data["results"]["per_label"].items():
        rows.append({
            "Nhãn": label,
            "Precision": f'{m["precision"]*100:.2f}%',
            "Recall": f'{m["recall"]*100:.2f}%',
            "F1-Score": f'{m["f1"]*100:.2f}%',
            "Số mẫu": m["support"],
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Model config summary ─────────────────────────────────────────────────
    cfg = data["model_config"]
    st.markdown(
        f"""
        <div class="step-card" style="margin-top:1rem">
            <div class="step-title">
                <span class="step-badge badge-green">C</span>
                Cấu hình thí nghiệm
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:.4rem .8rem;color:#94a3b8;font-size:.88rem">
                <span>Mô hình tiền xử lý:</span><b style="color:#e2e8f0">{cfg["preprocessing_model"]}</b>
                <span>Phân đoạn văn bản:</span><b style="color:#e2e8f0">{cfg["retrieval_strategy"]}</b>
                <span>Số tài liệu truy xuất:</span><b style="color:#e2e8f0">{cfg["retrieval_top_k"]}</b>
                <span>Sắp xếp lại:</span><b style="color:#e2e8f0">{cfg["reranker"]}</b>
                <span>Mô hình kết luận:</span><b style="color:#e2e8f0">{cfg["verdict_model"]}</b>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

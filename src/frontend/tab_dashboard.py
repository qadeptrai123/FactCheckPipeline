"""Tab 3 — Dashboard: biểu đồ thống kê tổng quan (layout sẵn)."""

import streamlit as st
import pandas as pd
import numpy as np
from src.frontend.utils import render_metric_box


def _placeholder_chart(title: str, chart_type: str = "bar"):
    """Render a placeholder chart with random demo data."""
    st.markdown(
        f'<p style="color:#94a3b8;font-size:.85rem;margin-bottom:.3rem">{title}</p>',
        unsafe_allow_html=True,
    )
    np.random.seed(hash(title) % 2**31)

    if chart_type == "bar":
        df = pd.DataFrame(
            np.random.rand(7, 3) * 100,
            columns=["Độ chuẩn xác", "Độ bao phủ", "F1"],
            index=[f"Lần {i+1}" for i in range(7)],
        )
        st.bar_chart(df, height=260)
    elif chart_type == "line":
        df = pd.DataFrame(
            np.cumsum(np.random.randn(30, 2), axis=0),
            columns=["Độ chính xác", "Macro-F1"],
        )
        st.line_chart(df, height=260)
    elif chart_type == "area":
        df = pd.DataFrame(
            np.abs(np.cumsum(np.random.randn(20, 3), axis=0)),
            columns=["TRUE", "FALSE", "HALF-TRUE"],
        )
        st.area_chart(df, height=260)


def render():
    """Render Tab 3 — Dashboard."""

    st.markdown("#### Tổng quan hệ thống")
    st.markdown(
        '<p style="color:#94a3b8;font-size:.9rem">'
        "Các biểu đồ thống kê sẽ được cập nhật khi kết nối với hệ thống backend. "
        "Hiện tại hiển thị dữ liệu mẫu để minh hoạ bố cục.</p>",
        unsafe_allow_html=True,
    )

    # ── Summary KPI row ──────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        render_metric_box("1,247", "Tổng claims đã xử lý")
    with c2:
        render_metric_box("78.4%", "Độ chính xác trung bình")
    with c3:
        render_metric_box("2.3s", "Thời gian TB / mẫu")
    with c4:
        render_metric_box("3", "Mô hình đang hoạt động")
    with c5:
        render_metric_box("92%", "Độ ổn định hệ thống")

    st.markdown("")

    # ── Row 1: two charts ────────────────────────────────────────────────────
    left, right = st.columns(2)
    with left:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-title">
                    <span class="step-badge badge-blue">A</span>
                    Diễn biến độ chính xác
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _placeholder_chart("Accuracy và Macro-F1 qua các lần thí nghiệm", "line")

    with right:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-title">
                    <span class="step-badge badge-green">B</span>
                    Phân bố kết quả phán định
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _placeholder_chart("Tỉ lệ các nhãn theo thời gian", "area")

    # ── Row 2: two more charts ───────────────────────────────────────────────
    left2, right2 = st.columns(2)
    with left2:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-title">
                    <span class="step-badge badge-amber">C</span>
                    Đối sánh hiệu suất giữa các mô hình
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _placeholder_chart("Precision / Recall / F1 theo thí nghiệm", "bar")

    with right2:
        st.markdown(
            """
            <div class="step-card">
                <div class="step-title">
                    <span class="step-badge badge-purple">D</span>
                    Độ trễ xử lý trung bình
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        _placeholder_chart("Thời gian trung bình theo giai đoạn", "bar")

    # ── Row 3: Retrieval stats ───────────────────────────────────────────────
    st.markdown(
        """
        <div class="step-card" style="margin-top:.5rem">
            <div class="step-title">
                <span class="step-badge badge-rose">R</span>
                Chỉ số truy xuất tài liệu
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_box("5,432", "Phân đoạn trong CSDL")
    with c2:
        render_metric_box("0.87", "Tỉ lệ tìm đúng (Top-5)")
    with c3:
        render_metric_box("0.92", "Tỉ lệ bao phủ (Top-10)")

    st.markdown("")
    _placeholder_chart("Recall@K theo chiến lược phân đoạn văn bản", "line")

    # ── Footer note ──────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center;margin-top:2rem;padding:1rem;
                    border-top:1px solid rgba(99,102,241,.1)">
            <p style="color:#475569;font-size:.82rem">
                Dữ liệu sẽ được cập nhật tự động khi kết nối hệ thống backend.
                <br>Nội dung hiện tại là dữ liệu mẫu dùng để minh hoạ bố cục.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

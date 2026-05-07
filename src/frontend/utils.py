"""Shared UI helper functions for FactCheck frontend."""

import streamlit as st
import time
import json
from pathlib import Path

# mock_data.json lives at FactCheckPipeline/src/frontend/mock_data.json
MOCK_DATA_PATH = Path(__file__).resolve().parent / "mock_data.json"


def load_mock_data() -> dict:
    """Load mock data from JSON file."""
    with open(MOCK_DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def render_header():
    """Render the main page header."""
    st.markdown(
        """
        <div class="main-header">
            <h1>FactCheck Pipeline</h1>
            <p>Hệ thống kiểm chứng thông tin đa phương thức</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_step_card(step_num: int, title: str, content_html: str, badge_class: str = "badge-blue"):
    """Render a pipeline step card."""
    st.markdown(
        f"""
        <div class="step-card">
            <div class="step-title">
                <span class="step-badge {badge_class}">{step_num}</span>
                {title}
            </div>
            {content_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_box(value: str, label: str):
    """Render a single metric box."""
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_score_bar(score: float, max_score: float = 1.0):
    """Render a horizontal score bar."""
    pct = min(score / max_score * 100, 100)
    st.markdown(
        f"""
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width:{pct:.1f}%"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def simulate_progress(message: str, seconds: float = 2.0, steps: int = 20):
    """Simulate a loading progress bar."""
    bar = st.progress(0, text=message)
    for i in range(steps):
        time.sleep(seconds / steps)
        bar.progress((i + 1) / steps, text=message)
    bar.empty()


def get_verdict_class(label: str) -> str:
    """Map verdict label to CSS class."""
    mapping = {
        "TRUE": "verdict-true",
        "FALSE": "verdict-false",
        "HALF-TRUE": "verdict-half",
        "UNVERIFIABLE": "verdict-unverifiable",
    }
    return mapping.get(label.upper(), "verdict-unverifiable")


def get_verdict_color(label: str) -> str:
    """Map verdict label to a hex colour."""
    mapping = {
        "TRUE": "#10b981",
        "FALSE": "#f43f5e",
        "HALF-TRUE": "#f59e0b",
        "UNVERIFIABLE": "#94a3b8",
    }
    return mapping.get(label.upper(), "#94a3b8")


def get_verdict_icon(label: str) -> str:
    mapping = {
        "TRUE": "✅",
        "FALSE": "❌",
        "HALF-TRUE": "⚠️",
        "UNVERIFIABLE": "❓",
    }
    return mapping.get(label.upper(), "❓")

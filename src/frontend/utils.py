"""Shared UI helper functions for FactCheck frontend."""

import streamlit as st
import time
import json
from pathlib import Path
from textwrap import dedent

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
        dedent(
            f"""
            <div class="step-card">
                <div class="step-title">
                    <span class="step-badge {badge_class}">{step_num}</span>
                    {title}
                </div>
                {dedent(content_html).strip()}
            </div>
            """
        ).strip(),
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
        "SUPPORTED": "verdict-true",
        "REFUTED": "verdict-false",
        "NEI": "verdict-unverifiable",
    }
    return mapping.get(label.upper(), "verdict-unverifiable")


def get_verdict_color(label: str) -> str:
    """Map verdict label to a hex colour."""
    mapping = {
        "SUPPORTED": "#10b981",
        "REFUTED": "#f43f5e",
        "NEI": "#94a3b8",
    }
    return mapping.get(label.upper(), "#94a3b8")


def get_verdict_icon(label: str) -> str:
    mapping = {
        "SUPPORTED": "fa-solid fa-circle-check",
        "REFUTED": "fa-solid fa-circle-xmark",
        "NEI": "fa-solid fa-circle-question",
    }
    return mapping.get(label.upper(), "fa-solid fa-circle-question")

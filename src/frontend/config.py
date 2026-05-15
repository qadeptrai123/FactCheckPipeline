"""Streamlit page config and CSS styling for FactCheck UI."""

import streamlit as st

PAGE_CONFIG = {
    "page_title": "FactCheck Pipeline",
    "page_icon": "FC",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Global ─────────────────────────────────────────── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.block-container { padding-top: 1.5rem; max-width: 1200px; }

/* ── Header ─────────────────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
    border: 1px solid rgba(99,102,241,.25);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(circle at 30% 50%, rgba(99,102,241,.12) 0%, transparent 60%),
                radial-gradient(circle at 70% 50%, rgba(16,185,129,.08) 0%, transparent 60%);
    pointer-events: none;
}
.main-header h1 {
    color: #f1f5f9; font-size: 1.75rem; font-weight: 700;
    margin: 0 0 .35rem; position: relative;
}
.main-header p {
    color: #94a3b8; font-size: .92rem; margin: 0; position: relative;
}

/* ── Cards / Steps ──────────────────────────────────── */
.step-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid rgba(99,102,241,.2);
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1rem;
    transition: border-color .25s, box-shadow .25s;
}
.step-card:hover {
    border-color: rgba(99,102,241,.45);
    box-shadow: 0 0 20px rgba(99,102,241,.08);
}
.step-title {
    display: flex; align-items: center; gap: .6rem;
    font-size: 1.05rem; font-weight: 600; color: #e2e8f0;
    margin-bottom: .7rem;
}
.step-badge {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 8px;
    font-size: .8rem; font-weight: 700; color: #fff; flex-shrink: 0;
}
.badge-blue   { background: linear-gradient(135deg,#6366f1,#4f46e5); }
.badge-green  { background: linear-gradient(135deg,#10b981,#059669); }
.badge-amber  { background: linear-gradient(135deg,#f59e0b,#d97706); }
.badge-rose   { background: linear-gradient(135deg,#f43f5e,#e11d48); }
.badge-purple { background: linear-gradient(135deg,#a78bfa,#7c3aed); }

/* ── Fact chips ─────────────────────────────────────── */
.fact-chip {
    display: inline-block;
    background: rgba(99,102,241,.1);
    border: 1px solid rgba(99,102,241,.25);
    border-radius: 8px;
    padding: .45rem .85rem;
    margin: .2rem .25rem;
    font-size: .85rem; color: #c7d2fe;
    transition: background .2s;
}
.fact-chip:hover { background: rgba(99,102,241,.18); }

/* ── Verdict banner ─────────────────────────────────── */
.verdict-true {
    background: linear-gradient(135deg, rgba(16,185,129,.12), rgba(16,185,129,.04));
    border: 1px solid rgba(16,185,129,.35);
    border-radius: 14px; padding: 1.4rem; text-align: center;
}
.verdict-false {
    background: linear-gradient(135deg, rgba(244,63,94,.12), rgba(244,63,94,.04));
    border: 1px solid rgba(244,63,94,.35);
    border-radius: 14px; padding: 1.4rem; text-align: center;
}
.verdict-half {
    background: linear-gradient(135deg, rgba(245,158,11,.12), rgba(245,158,11,.04));
    border: 1px solid rgba(245,158,11,.35);
    border-radius: 14px; padding: 1.4rem; text-align: center;
}
.verdict-unverifiable {
    background: linear-gradient(135deg, rgba(148,163,184,.12), rgba(148,163,184,.04));
    border: 1px solid rgba(148,163,184,.35);
    border-radius: 14px; padding: 1.4rem; text-align: center;
}
.verdict-label {
    font-size: 1.6rem; font-weight: 800; margin-bottom: .3rem;
}

/* ── Metric boxes ───────────────────────────────────── */
.metric-box {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid rgba(99,102,241,.2);
    border-radius: 12px; padding: 1.1rem; text-align: center;
}
.metric-value {
    font-size: 1.65rem; font-weight: 700; color: #818cf8;
}
.metric-label {
    font-size: .82rem; color: #94a3b8; margin-top: .2rem;
}

/* ── Retrieval result row ───────────────────────────── */
.retrieval-row {
    background: rgba(30,41,59,.6);
    border: 1px solid rgba(99,102,241,.15);
    border-radius: 10px;
    padding: .9rem 1.1rem;
    margin-bottom: .55rem;
    transition: border-color .2s;
}
.retrieval-row:hover { border-color: rgba(99,102,241,.35); }

/* ── Sidebar ────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e293b) !important;
    border-right: 1px solid rgba(99,102,241,.15);
}

/* ── Tabs ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: .5rem; border-bottom: 1px solid rgba(99,102,241,.15);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: .6rem 1.4rem;
    font-weight: 600;
    color: #94a3b8;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,.12) !important;
    color: #818cf8 !important;
    border-bottom: 2px solid #6366f1;
}

/* ── Score bar ───────────────────────────────────────── */
.score-bar-bg {
    background: rgba(51,65,85,.6);
    border-radius: 6px; height: 10px; width: 100%;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%; border-radius: 6px;
    background: linear-gradient(90deg, #6366f1, #818cf8);
    transition: width .6s ease;
}

/* ── Misc ────────────────────────────────────────────── */
.divider {
    border: none;
    border-top: 1px solid rgba(99,102,241,.12);
    margin: 1rem 0;
}
.caption { color: #64748b; font-size: .82rem; }
</style>
"""

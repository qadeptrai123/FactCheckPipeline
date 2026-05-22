"""Single-sample fact-check view."""

from __future__ import annotations

import html
import json
from textwrap import dedent
from typing import Any

import streamlit as st

from src.frontend.api_client import start_pipeline, wait_for_pipeline
from src.frontend.config import get_enabled_phases
from src.frontend.utils import get_verdict_class, get_verdict_color, get_verdict_icon, render_step_card


def _phase(status: dict[str, Any], name: str) -> dict[str, Any]:
    return status.get("phases", {}).get(name, {})


PHASE_LABELS = {
    "refined": "Hiệu chỉnh đầu vào",
    "retrieval": "Truy xuất dữ liệu",
    "judge": "Đưa ra quyết định",
}


def _items(values: list[Any], empty_text: str) -> None:
    if values:
        for value in values:
            st.markdown(f"- {value}")
        return
    st.caption(empty_text)


def _status_text(phase: dict[str, Any], label: str) -> str:
    status = phase.get("status", "pending")
    if status == "running":
        return f"Đang xử lý: {label}"
    if status == "error":
        return f"Lỗi tại bước: {label}"
    if status == "pending":
        return f"Chờ xử lý: {label}"
    duration = phase.get("duration_seconds")
    return f"{label}: {duration:.1f}s" if duration is not None else label


def _verdict_label(verdict: str) -> str:
    return verdict.upper() if verdict else "NEI"


def _relation_vi_label(relation: str) -> str:
    mapping = {
        "SUPPORT": "Ủng hộ",
        "REFUTE": "Bác bỏ",
        "PARTIAL_SUPPORT": "Ủng hộ một phần",
        "UNRELATED": "Không liên quan",
    }
    return mapping.get(str(relation).upper(), "Không xác định")


def _render_refined(data: dict[str, Any]) -> None:
    atoms = data.get("claim_atoms") or []
    visual_items = data.get("visual_observations") or []
    queries = data.get("search_queries") or {}
    focus = data.get("retrieval_focus") or {}
    image_count = data.get("input_image_count", 0)
    atom_queries = []
    for atom in atoms:
        if isinstance(atom, dict):
            atom_queries.extend(atom.get("retrieval_queries") or [])

    visual_terms = []
    for item in visual_items:
        if isinstance(item, dict):
            visual_terms.append(item.get("text", ""))
            visual_terms.extend(item.get("visible_evidence") or [])

    def html_list(values: list[Any], limit: int | None = None) -> str:
        selected = values[:limit] if limit else values
        items = [str(value).strip() for value in selected if str(value).strip()]
        if not items:
            return '<p style="color:#64748b;margin:.25rem 0">Không có</p>'
        return "<ul style='margin:.25rem 0 .7rem 1.1rem;padding:0;color:#cbd5e1'>" + "".join(
            f"<li style='margin:.22rem 0'>{html.escape(item)}</li>" for item in items
        ) + "</ul>"

    focus_chips = "".join(
        f'<span class="fact-chip">{html.escape(key)}: {str(bool(focus.get(key, False))).lower()}</span>'
        for key in ["text", "image", "cross_modal"]
    )

    content = f"""
    <div class="fact-chip">Model: {html.escape(data.get("model_name", ""))}</div>
    <div class="fact-chip">Ảnh: {int(image_count)}</div>
    <div class="fact-chip">Lần validate: {html.escape(str(data.get("validation_attempts", "")))}</div>
    <p style="color:#94a3b8;font-size:.85rem;margin:.8rem 0 .2rem">Câu truy vấn đã refine</p>
    <div class="fact-chip">{html.escape(data.get("primary_retrieval_query", ""))}</div>
    <p style="color:#94a3b8;font-size:.85rem;margin:.8rem 0 .2rem">Tuyên bố chuẩn hoá</p>
    <p style="color:#cbd5e1;margin:.2rem 0 .7rem">{html.escape(data.get("normalized_claim") or "N/A")}</p>
    <div style="display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:1rem;margin-top:.8rem">
        <div>
            <p style="color:#94a3b8;font-size:.85rem;font-weight:700;margin:.1rem 0 .3rem">semantic</p>
            {html_list(queries.get("semantic", []))}
        </div>
        <div>
            <p style="color:#94a3b8;font-size:.85rem;font-weight:700;margin:.1rem 0 .3rem">keywords</p>
            {html_list(queries.get("keywords", []))}
        </div>
        <div>
            <p style="color:#94a3b8;font-size:.85rem;font-weight:700;margin:.1rem 0 .3rem">visual</p>
            {html_list(queries.get("visual", []))}
        </div>
    </div>
    <p style="color:#94a3b8;font-size:.85rem;font-weight:700;margin:.4rem 0 .2rem">Atomic retrieval queries</p>
    {html_list(atom_queries, limit=8)}
    <p style="color:#94a3b8;font-size:.85rem;font-weight:700;margin:.4rem 0 .2rem">Visual terms đưa vào truy xuất ảnh</p>
    {html_list(visual_terms, limit=8)}
    <p style="color:#94a3b8;font-size:.85rem;font-weight:700;margin:.4rem 0 .2rem">Verification targets</p>
    {html_list(data.get("verification_targets") or [], limit=5)}
    <p style="color:#94a3b8;font-size:.85rem;font-weight:700;margin:.4rem 0 .2rem">Retrieval focus</p>
    <div>{focus_chips}</div>
    """
    render_step_card(1, "Hiệu chỉnh đầu vào", content, "badge-blue")
    if data.get("refine_error"):
        st.warning(data["refine_error"])


def _result_row(item: dict[str, Any]) -> str:
    title = html.escape(item.get("title") or item.get("url") or item.get("point_id", ""))
    url = html.escape(item.get("url", ""))
    text = html.escape((item.get("text") or "")[:240])
    score = float(item.get("final_score") or 0.0)
    lane = html.escape(str(item.get("lane", "")))
    rank = html.escape(str(item.get("rank", "")))
    image_path = html.escape(str(item.get("image_path", "")))
    source_line = url if url else image_path
    return f"""
    <div class="retrieval-row">
        <div style="display:flex;justify-content:space-between;gap:1rem;align-items:center">
            <div style="color:#e2e8f0;font-weight:600">{lane} #{rank} · {title}</div>
            <div style="color:#10b981;font-weight:700">{score:.4f}</div>
        </div>
        <div style="color:#94a3b8;font-size:.83rem;margin-top:.35rem">{text}</div>
        <div style="color:#64748b;font-size:.78rem;margin-top:.35rem">{source_line}</div>
    </div>
    """


def _render_retrieval(data: dict[str, Any]) -> None:
    setup = (
        f'Thử nghiệm: <b style="color:#e2e8f0">{html.escape(data.get("experiment_id", ""))}</b> · '
        f'Bộ dữ liệu: <b style="color:#e2e8f0">{html.escape(data.get("collection", ""))}</b> · '
        f'Mô hình ảnh: <b style="color:#e2e8f0">{html.escape(data.get("image_variant", ""))}</b> · '
        f'Sắp xếp lại: <b style="color:#e2e8f0">{data.get("use_reranker")}</b>'
    )
    evidence_items = data.get("evidence_for_judge") or [
        *data.get("text_results", [])[:3],
        *data.get("image_results", [])[:1],
    ]
    rows = "".join(_result_row(item) for item in evidence_items)
    urls = "".join(f'<span class="fact-chip">{html.escape(url)}</span>' for url in data.get("top_urls", []))
    content = f"""
    <p style="color:#94a3b8;font-size:.85rem">{setup}</p>
    <p style="color:#94a3b8;font-size:.85rem;font-weight:700;margin:.7rem 0 .2rem">Top URLs đưa vào phase quyết định</p>
    <div>{urls}</div>
    <p style="color:#94a3b8;font-size:.85rem;font-weight:700;margin:.9rem 0 .2rem">4 mẫu bằng chứng chuẩn bị cho phase sau</p>
    <div style="margin-top:.8rem">{rows}</div>
    """
    render_step_card(2, "Truy xuất dữ liệu", content, "badge-purple")


def _render_judge(data: dict[str, Any]) -> None:
    verdict = data.get("verdict") or "NEI"
    color = get_verdict_color(verdict)
    icon_class = get_verdict_icon(verdict)
    explanation = html.escape(data.get("explanation") or "")
    urls = "".join(f'<div class="fact-chip">{html.escape(url)}</div>' for url in data.get("top3_urls_used", []))
    map_items = data.get("map_results") or []
    map_html = ""
    for idx, item in enumerate(map_items, start=1):
        if not item:
            continue
        relation = str(item.get("relation", ""))
        relation_label = _relation_vi_label(relation)
        extracted_facts = html.escape(item.get("extracted_facts", ""))
        thought_process = html.escape(item.get("thought_process", ""))
        map_html += dedent(
            f"""
            <div class="retrieval-row">
                <div style="display:flex;justify-content:space-between;gap:1rem;align-items:center">
                    <div style="color:#e2e8f0;font-weight:700">Bằng chứng {idx}</div>
                    <div>
                        <span class="fact-chip">{html.escape(relation)}</span>
                        <span class="fact-chip">{html.escape(relation_label)}</span>
                    </div>
                </div>
                <p style="color:#94a3b8;font-size:.84rem;font-weight:700;margin:.55rem 0 .2rem">LLM đánh giá</p>
                <div style="color:#cbd5e1;font-size:.86rem;line-height:1.55">{thought_process or "Không có"}</div>
                <p style="color:#94a3b8;font-size:.84rem;font-weight:700;margin:.55rem 0 .2rem">Sự kiện trích xuất</p>
                <div style="color:#cbd5e1;font-size:.86rem;line-height:1.55">{extracted_facts or "Không có"}</div>
            </div>
            """
        ).strip()
    content = dedent(
        f"""
        <div style="text-align:center;margin:.3rem 0 1rem">
            <div class="verdict-label" style="color:{color}">
                <i class="{icon_class}" aria-hidden="true"></i> {html.escape(_verdict_label(verdict))}
            </div>
            <p style="color:#94a3b8;font-size:.85rem;font-weight:700;margin:.7rem 0 .25rem">Final explanation</p>
            <p style="color:#cbd5e1;font-size:.95rem;line-height:1.6;max-width:860px;margin:0 auto">{explanation}</p>
        </div>
        <p style="color:#94a3b8;font-size:.85rem;font-weight:700;margin:.8rem 0 .2rem">Nguồn đã dùng</p>
        <div>{urls}</div>
        <p style="color:#94a3b8;font-size:.85rem;font-weight:700;margin:1rem 0 .2rem">Đánh giá từng bằng chứng</p>
        <div>{map_html}</div>
        """
    ).strip()
    render_step_card(3, "Đưa ra quyết định", content, "badge-green")

    with st.expander("Kết quả thô của bước quyết định"):
        st.json(data)


def _render_completed(status: dict[str, Any]) -> None:
    phases = status.get("phases", {})
    refined = phases.get("refined", {}).get("data", {})
    retrieval = phases.get("retrieval", {}).get("data", {})
    judge = phases.get("judge", {}).get("data", {})

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("#### Kết quả")
    _render_judge(judge)
    _render_refined(refined)
    _render_retrieval(retrieval)

    with st.expander("Phản hồi thô của pipeline"):
        st.json(status)


def render() -> None:
    enabled_phases = get_enabled_phases()
    st.markdown("#### Kiểm chứng một tuyên bố")

    claim = st.text_area(
        "Tuyên bố cần kiểm chứng",
        height=110,
        placeholder="Nhập tuyên bố tiếng Việt cần kiểm chứng...",
        key="claim_input",
    )
    uploaded_file = st.file_uploader("Hình ảnh đính kèm tùy chọn", type=["jpg", "jpeg", "png", "webp"], key="image_upload")
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Hình ảnh đã chọn", use_container_width=True)

    execute = st.button("Kiểm chứng", type="primary", use_container_width=False)
    if not execute:
        st.caption("Quy trình sẽ chạy theo thứ tự: " + " -> ".join(PHASE_LABELS[phase] for phase in enabled_phases) + ".")
        return

    if not claim.strip():
        st.warning("Vui lòng nhập tuyên bố cần kiểm chứng.")
        return

    try:
        job_id = start_pipeline(claim.strip(), uploaded_file)
    except Exception as exc:
        st.error(f"Không thể khởi chạy pipeline: {exc}")
        return

    detail_placeholder = st.empty()
    last_status = None
    try:
        for status in wait_for_pipeline(job_id):
            last_status = status
            with detail_placeholder.container():
                phases = status.get("phases", {})
                for key in enabled_phases:
                    phase = phases.get(key, {})
                    label = PHASE_LABELS[key]
                    if phase.get("status") == "done" and phase.get("data"):
                        if key == "refined":
                            _render_refined(phase["data"])
                        elif key == "retrieval":
                            _render_retrieval(phase["data"])
                        elif key == "judge":
                            _render_judge(phase["data"])
                    elif phase.get("status") in {"running", "pending"}:
                        st.info(_status_text(phase, label))
                        break
                    elif phase.get("status") == "error":
                        st.error(f"{_status_text(phase, label)}: {phase.get('error')}")
                        break
            if status.get("status") == "done":
                break
    except Exception as exc:
        st.error(f"Lỗi khi chờ kết quả pipeline: {exc}")
        return

    if not last_status:
        st.error("Máy chủ chưa trả trạng thái.")
        return
    if last_status.get("status") == "error":
        st.error(last_status.get("error") or "Quy trình bị lỗi.")
        with st.expander("Phản hồi lỗi thô"):
            st.code(json.dumps(last_status, ensure_ascii=False, indent=2), language="json")
        return

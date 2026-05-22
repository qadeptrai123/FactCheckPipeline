from __future__ import annotations

import asyncio
import ast
import json
import os
import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(PROJECT_ROOT / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MODEL_ID = "google/gemini-2.5-flash"
CORPUS_FILE = PROJECT_ROOT / "chunking_scripts" / "final_corpus.csv"
MAX_EVIDENCE_CHARS = 4500
RETRY_LIMIT = 3


Relation = Literal["SUPPORT", "REFUTE", "PARTIAL_SUPPORT", "UNRELATED"]
Verdict = Literal["SUPPORTED", "REFUTED", "NEI"]


class EvidenceJudge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thought_process: str
    relation: Relation
    extracted_facts: str


class FinalThoughtProcess(BaseModel):
    model_config = ConfigDict(extra="forbid")

    synthesis: str
    target_check: str
    logical_deduction: str


class FinalVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    thought_process: FinalThoughtProcess
    verdict: Verdict
    explanation: str


EVIDENCE_SCHEMA = EvidenceJudge.model_json_schema()
FINAL_SCHEMA = FinalVerdict.model_json_schema()

MAP_SYSTEM_PROMPT = """You are a strict fact-checking evidence judge.
Task: Compare a structurally analyzed Claim with ONE piece of Evidence.

Rules:
- Use only information from this Evidence. Do not use external knowledge.
- Write every generated natural-language value strictly in accented Vietnamese.
- Do not generate English explanations, thoughts, or extracted facts.
- Treat title, URL, and retrieval rank as metadata, not facts by themselves.
- SUPPORT means the evidence fully supports the relevant claim atom.
- REFUTE means the evidence directly contradicts the relevant claim atom.
- PARTIAL_SUPPORT means the evidence is relevant but incomplete.
- UNRELATED means the evidence does not address the claim.
- Return JSON only, following the schema exactly.
"""

FINAL_SYSTEM_PROMPT = """You are a strict final fact-checking judge.
Use only the extracted evidence facts from the previous step.

Verdict rules:
- Write every generated natural-language value strictly in accented Vietnamese.
- Do not generate English explanations, thoughts, or extracted facts.
- SUPPORTED: enough evidence supports all central verification targets.
- REFUTED: at least one central target is directly contradicted by evidence.
- NEI: evidence is missing, unrelated, indirect, or insufficient.
- A direct contradiction has priority over partial support.
- The explanation must be in Vietnamese, under 50 words, natural for an end user.
- Do not mention internal process terms such as source number, map, reduce, retrieval, JSON, or model.
- Return JSON only, following the schema exactly.
"""


def safe_json_loads(value: Any, fallback: Any) -> Any:
    if value is None:
        return fallback
    if isinstance(value, (dict, list)):
        return value
    text = str(value).strip()
    if not text:
        return fallback
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return fallback


def extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def validate_model(model_cls: type[BaseModel], raw_text: str) -> dict[str, Any]:
    return model_cls.model_validate(extract_json(raw_text)).model_dump()


def format_atoms(refined_claim: dict[str, Any]) -> str:
    atoms = []
    for atom in refined_claim.get("claim_atoms", []) or []:
        if isinstance(atom, dict):
            text = atom.get("text", "")
            priority = atom.get("priority", "")
            atoms.append(f"+ {text} (Ưu tiên: {priority})" if priority else f"+ {text}")
        elif atom:
            atoms.append(f"+ {atom}")
    return "\n".join(atoms) if atoms else "Không có mệnh đề cụ thể."


def format_visuals(refined_claim: dict[str, Any]) -> str:
    visuals = []
    for item in refined_claim.get("visual_observations", []) or []:
        if isinstance(item, dict):
            visuals.append(f"+ {item.get('text', '')}")
        elif item:
            visuals.append(f"+ {item}")
    return "\n".join(visuals) if visuals else "Không có thông tin thị giác."


def format_targets(refined_claim: dict[str, Any]) -> str:
    targets = [f"- {target}" for target in refined_claim.get("verification_targets", []) or []]
    return "\n".join(targets) if targets else "Không có mục tiêu cụ thể."


class LLMInferencer:
    def __init__(self) -> None:
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        if not CORPUS_FILE.exists():
            raise FileNotFoundError(f"Corpus file not found: {CORPUS_FILE}")
        self.client = AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=OPENROUTER_API_KEY)
        frame = pd.read_csv(CORPUS_FILE).drop_duplicates(subset=["url"])
        self.corpus = frame.set_index("url")["content"].to_dict()

    def generate(self, refined_claim: dict[str, Any], retrieval: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(self.generate_async(refined_claim, retrieval))

    async def generate_async(self, refined_claim: dict[str, Any], retrieval: dict[str, Any]) -> dict[str, Any]:
        evidences_text = []
        urls_used = []
        missing_urls = []
        for url in retrieval.get("top_urls", []):
            if url in self.corpus:
                text = str(self.corpus[url])
                if len(text) > MAX_EVIDENCE_CHARS:
                    text = text[:MAX_EVIDENCE_CHARS].rsplit(" ", 1)[0] + "..."
                evidences_text.append(text)
                urls_used.append(url)
            else:
                missing_urls.append(url)

        map_results = []
        for idx, evidence_text in enumerate(evidences_text):
            result = await self._evaluate_single_evidence(refined_claim, evidence_text, idx + 1)
            map_results.append(result)

        final_verdict, final_error = await self._get_final_verdict(refined_claim, map_results)
        if final_verdict is None:
            final_verdict = {"verdict": "NEI", "explanation": "", "thought_process": {}}

        return {
            "model_alias": "gemini-2.5-flash",
            "model_name": MODEL_ID,
            "normalized_claim": refined_claim.get("normalized_claim"),
            "verdict": final_verdict.get("verdict"),
            "explanation": final_verdict.get("explanation"),
            "thought_process": final_verdict.get("thought_process", {}),
            "top3_urls_used": urls_used,
            "missing_urls": missing_urls,
            "map_results": map_results,
            "final_error": final_error,
        }

    async def _call_openrouter_schema(self, messages: list[dict[str, Any]], schema_name: str, schema: dict[str, Any], max_tokens: int, retry_note: str = "") -> str:
        if retry_note:
            messages = [
                *messages,
                {"role": "user", "content": f"Output trước đó lỗi validation: {retry_note}. Hãy trả lại JSON hợp lệ, không markdown. Mọi giá trị văn bản tự nhiên phải viết nghiêm ngặt bằng tiếng Việt có dấu."},
            ]
        response = await self.client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=0.0,
            max_tokens=max_tokens,
            response_format={"type": "json_schema", "json_schema": {"name": schema_name, "strict": True, "schema": schema}},
        )
        return response.choices[0].message.content or ""

    async def _evaluate_single_evidence(self, refined_claim: dict[str, Any], evidence_text: str, evidence_index: int) -> dict[str, Any] | None:
        user_prompt = f"""[CLAIM STRUCTURE TO VERIFY]
- Claim: {refined_claim.get("normalized_claim", "")}
- Claim Atoms to consider:
{format_atoms(refined_claim)}
- Visual observations from the attached image:
{format_visuals(refined_claim)}

[EVIDENCE {evidence_index}]
{evidence_text}
"""
        messages = [
            {"role": "system", "content": MAP_SYSTEM_PROMPT},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
        retry_note = ""
        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                raw = await self._call_openrouter_schema(messages, "single_evidence_judge", EVIDENCE_SCHEMA, 1200, retry_note)
                return validate_model(EvidenceJudge, raw)
            except Exception as exc:
                retry_note = str(exc)[:1000]
                await asyncio.sleep(min(2 * attempt, 8))
        return None

    async def _get_final_verdict(self, refined_claim: dict[str, Any], map_results: list[dict[str, Any] | None]) -> tuple[dict[str, Any] | None, str]:
        compiled_facts = ""
        for idx, result in enumerate(map_results):
            if result and result.get("relation") != "UNRELATED":
                compiled_facts += f"\n- Nguồn {idx + 1} ({result.get('relation')}): {result.get('extracted_facts')}"
        if not compiled_facts.strip():
            compiled_facts = "\nKhông có thông tin liên quan đến tuyên bố trong các bằng chứng được cung cấp."

        user_prompt = f"""[CLAIM]
{refined_claim.get("normalized_claim", "")}

[VERIFICATION TARGETS]
{format_targets(refined_claim)}

[EXTRACTED INFORMATION PIECES FROM EVIDENCE]
{compiled_facts}
"""
        messages = [
            {"role": "system", "content": FINAL_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        retry_note = ""
        last_error = ""
        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                raw = await self._call_openrouter_schema(messages, "final_fact_check_verdict", FINAL_SCHEMA, 1400, retry_note)
                return validate_model(FinalVerdict, raw), ""
            except Exception as exc:
                last_error = str(exc)
                retry_note = last_error[:1000]
                await asyncio.sleep(min(2 * attempt, 8))
        return None, last_error

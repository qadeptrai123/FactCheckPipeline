from __future__ import annotations

import asyncio
import base64
import io
import json
import mimetypes
import os
import re
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from openai import AsyncOpenAI
from PIL import Image
from pydantic import BaseModel, ConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[4]
load_dotenv(PROJECT_ROOT / ".env")

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
MODEL_ID = "google/gemini-2.5-flash"
TEMPERATURE = 0.1
MAX_TOKENS = 3200
MAX_IMAGE_SIDE = 768
JPEG_QUALITY = 70
RETRY_LIMIT = 3


class ClaimAtom(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    check_type: Literal["entity", "event", "time", "location", "number", "quote", "relation", "other"]
    priority: Literal["high", "medium", "low"]
    retrieval_queries: list[str]


class VisualObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    visible_evidence: list[str]
    confidence: Literal["high", "medium", "low"]


class Alignment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: Literal["match", "partial_match", "mismatch", "not_enough_visual_info"]
    text: str


class KeyEntities(BaseModel):
    model_config = ConfigDict(extra="forbid")

    people: list[str]
    organizations: list[str]
    locations: list[str]
    dates: list[str]
    numbers: list[str]
    other: list[str]


class SearchQueries(BaseModel):
    model_config = ConfigDict(extra="forbid")

    semantic: list[str]
    keywords: list[str]
    visual: list[str]


class RetrievalFocus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: bool
    image: bool
    cross_modal: bool


class Constraints(BaseModel):
    model_config = ConfigDict(extra="forbid")

    time: list[str]
    location: list[str]
    source_type: list[str]


class RefineOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    original_claim: str
    normalized_claim: str
    primary_retrieval_query: str
    image_provided: bool
    language: Literal["vi"]
    claim_atoms: list[ClaimAtom]
    visual_observations: list[VisualObservation]
    alignment: Alignment
    key_entities: KeyEntities
    search_queries: SearchQueries
    retrieval_focus: RetrievalFocus
    constraints: Constraints
    context_summary: str
    ambiguity_notes: list[str]
    verification_targets: list[str]


REFINE_JSON_SCHEMA = RefineOutput.model_json_schema()

SYSTEM_INSTRUCTION = """You are a fact-checking input refiner for a Vietnamese multimodal RAG pipeline.
Return only valid JSON that follows the requested schema.
Do not decide the final truth label of the claim.
Do not use external knowledge. Use only the provided claim text and visible image content.
Write all generated field values strictly in accented Vietnamese.
Every natural-language answer, explanation, note, query, observation, and target must be Vietnamese only."""


def build_refine_prompt(claim: str, image_count: int) -> str:
    image_provided = image_count > 0
    return f"""
# Task

Refine a noisy Vietnamese fact-checking input into a clean, structured representation for downstream RAG testing.

# Input

- Claim: {claim}
- Image provided: {str(image_provided).lower()}
- Number of provided images: {image_count}

# Rules

- Write all generated field values strictly in accented Vietnamese.
- Do not generate English natural-language text anywhere in the JSON values.
- The `language` field must be exactly "vi".
- Keep the JSON compact: no markdown, no commentary, no repeated whitespace, no long paragraphs.
- Preserve named entities, numbers, dates, quoted text, locations, and distinctive visual details.
- Normalize the claim without changing its meaning.
- Split the claim into atomic, verifiable facts.
- Create one best primary retrieval query for the whole input.
- Create at most 5 claim_atoms.
- Create at most 2 retrieval query variants for each atomic claim.
- If one or more images are provided, describe only visible evidence across all provided images.
- If multiple images are provided, include observations from all relevant images and keep each observation grounded in visible details.
- Create at most 3 visual_observations and at most 4 visible_evidence items per observation.
- Do not infer identity, intent, unseen events, or off-image context from images.
- Explain whether visible image evidence appears to support, partially support, contradict, or not sufficiently address the claim.
- If no image is provided, use an empty visual_observations array and set alignment.label to "not_enough_visual_info".
- Do not decide the final truth of the claim.
- Create general search query variants for the next phase. Do not mention any vector database, embedding model, or retrieval backend.
- Create at most 3 semantic queries, 6 keyword queries, and 3 visual queries.
- Set retrieval_focus booleans based on which modalities are useful for retrieval.
- Extract explicit time, location, and source-type constraints only when they appear in the input or are clearly required by the claim.
- Use empty arrays when information is absent.
- Keep context_summary to 1-2 short Vietnamese sentences.
- Create at most 5 ambiguity_notes and at most 5 verification_targets.

# Output

Return exactly one JSON object that follows the provided JSON schema.
""".strip()


def image_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    if mime not in {"image/jpeg", "image/png", "image/webp"}:
        mime = "image/jpeg"

    with Image.open(path) as img:
        img = img.convert("RGB")
        img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in model output: {text[:500]}")
        return json.loads(match.group(0))


def validate_refine_json(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("language") in {"Vietnamese", "Tiếng Việt", "vietnamese", "VI"}:
        data["language"] = "vi"
    return RefineOutput.model_validate(data).model_dump()


class QueryRefiner:
    def __init__(self) -> None:
        if not OPENROUTER_API_KEY:
            raise RuntimeError("OPENROUTER_API_KEY is not configured")
        self.client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

    def refine(self, claim: str, image_paths: list[Path]) -> dict[str, Any]:
        return asyncio.run(self.refine_async(claim, image_paths))

    async def refine_async(self, claim: str, image_paths: list[Path]) -> dict[str, Any]:
        retry_note = ""
        last_raw = ""
        last_error = ""
        for attempt in range(1, RETRY_LIMIT + 1):
            try:
                raw = await self._call_openrouter(claim, image_paths, retry_note)
                last_raw = raw
                refined = validate_refine_json(extract_json(raw))
                return {
                    "model_alias": "gemini-2.5-flash",
                    "model_name": MODEL_ID,
                    "input_claim": claim,
                    "input_image_count": len(image_paths),
                    "resolved_image_paths": [str(path) for path in image_paths],
                    "raw_output": raw,
                    "validation_attempts": attempt,
                    "refine_error": "",
                    **refined,
                }
            except Exception as exc:
                last_error = str(exc)
                retry_note = last_error[:1200]
                await asyncio.sleep(min(2 * attempt, 8))

        return self._fallback(claim, image_paths, last_raw, last_error)

    async def _call_openrouter(self, claim: str, image_paths: list[Path], retry_note: str) -> str:
        prompt = build_refine_prompt(claim=claim, image_count=len(image_paths))
        if retry_note:
            prompt += f"\n\n# Retry instruction\nThe previous output failed validation: {retry_note}. Return corrected JSON only. All generated natural-language values must remain strictly in accented Vietnamese."

        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})

        response = await self.client.chat.completions.create(
            model=MODEL_ID,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": content},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "fact_check_refine",
                    "strict": True,
                    "schema": REFINE_JSON_SCHEMA,
                },
            },
        )
        return response.choices[0].message.content or ""

    def _fallback(self, claim: str, image_paths: list[Path], raw: str, error: str) -> dict[str, Any]:
        return {
            "model_alias": "gemini-2.5-flash",
            "model_name": MODEL_ID,
            "input_claim": claim,
            "input_image_count": len(image_paths),
            "resolved_image_paths": [str(path) for path in image_paths],
            "raw_output": raw,
            "validation_attempts": RETRY_LIMIT,
            "refine_error": f"failed validation after {RETRY_LIMIT} attempts: {error}",
            "original_claim": claim,
            "normalized_claim": claim,
            "primary_retrieval_query": claim,
            "image_provided": bool(image_paths),
            "language": "vi",
            "claim_atoms": [],
            "visual_observations": [],
            "alignment": {"label": "not_enough_visual_info", "text": "Không tạo được JSON hợp lệ sau khi retry."},
            "key_entities": {"people": [], "organizations": [], "locations": [], "dates": [], "numbers": [], "other": []},
            "search_queries": {"semantic": [claim], "keywords": [], "visual": []},
            "retrieval_focus": {"text": True, "image": bool(image_paths), "cross_modal": bool(image_paths)},
            "constraints": {"time": [], "location": [], "source_type": []},
            "context_summary": "Không tạo được refine output hợp lệ.",
            "ambiguity_notes": [error],
            "verification_targets": [claim],
        }

import argparse
import base64
import json
import mimetypes
import os
import re
import sys
from pathlib import Path
from typing import Literal

import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

DEFAULT_MODEL = "google/gemini-2.5-flash"
DEFAULT_FALLBACK_MODELS = "qwen/qwen3-vl-8b-instruct,google/gemma-3-4b-it"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2200


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


def load_env() -> None:
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def data_url(image_path: Path) -> str:
    content_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{content_type};base64,{encoded}"


def extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in model output: {text[:500]}")
        return json.loads(match.group(0))


def response_text(response: object) -> str:
    choices = getattr(response, "choices", None)
    if choices:
        return getattr(choices[0].message, "content", "") or ""
    if isinstance(response, dict):
        return response.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
    raise ValueError(f"Unexpected OpenRouter response shape: {type(response).__name__}")


def build_messages(claim: str, image_path: Path | None) -> list[dict]:
    prompt = (PROMPT_DIR / "refine_json.md").read_text(encoding="utf-8").format(
        claim=claim,
        image_provided=str(bool(image_path)).lower(),
    )
    content = [{"type": "text", "text": prompt}]
    if image_path:
        content.append({"type": "image_url", "image_url": {"url": data_url(image_path)}})
    return [
        {"role": "system", "content": (PROMPT_DIR / "system.md").read_text(encoding="utf-8")},
        {"role": "user", "content": content},
    ]


def refine_row(client, claim: str, image_path: Path | None, args: argparse.Namespace) -> dict:
    request = {
        "model": args.model,
        "messages": build_messages(claim, image_path),
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "fact_check_refine",
                "strict": True,
                "schema": RefineOutput.model_json_schema(),
            },
        },
    }
    if args.fallback_models:
        request["extra_body"] = {"models": [args.model, *args.fallback_models], "route": "fallback"}

    raw = response_text(client.chat.completions.create(**request))
    data = extract_json(raw)
    data.setdefault("original_claim", claim)
    data.setdefault("image_provided", bool(image_path))
    return RefineOutput.model_validate(data).model_dump()


def resolve_image_path(value: object, input_dir: Path) -> Path | None:
    if pd.isna(value) or not str(value).strip():
        return None
    path = Path(str(value).strip())
    if not path.is_absolute():
        path = input_dir / path
    return path if path.exists() else None


def flatten_result(result: dict) -> dict:
    flat = {}
    for key, value in result.items():
        if isinstance(value, (dict, list)):
            flat[f"refined_{key}"] = json.dumps(value, ensure_ascii=False)
        else:
            flat[f"refined_{key}"] = value
    return flat


def parse_args() -> argparse.Namespace:
    load_env()
    parser = argparse.ArgumentParser(description="Refine a text+image CSV with OpenRouter VLM and export CSV.")
    parser.add_argument("--input", required=True, help="Input CSV file.")
    parser.add_argument("--output", required=True, help="Output CSV file.")
    parser.add_argument("--text-column", default="claim", help="Column containing claim/text input.")
    parser.add_argument("--image-column", default="image_path", help="Optional column containing image paths.")
    parser.add_argument("--limit", type=int, help="Optional max rows to process.")
    parser.add_argument("--model", default=os.getenv("REFINE_OPENROUTER_MODEL", DEFAULT_MODEL))
    parser.add_argument("--fallback-models", default=os.getenv("REFINE_FALLBACK_MODELS", DEFAULT_FALLBACK_MODELS))
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY"))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("REFINE_TEMPERATURE", DEFAULT_TEMPERATURE)))
    parser.add_argument("--max-tokens", type=int, default=int(os.getenv("REFINE_MAX_TOKENS", DEFAULT_MAX_TOKENS)))
    args = parser.parse_args()
    args.fallback_models = [item.strip() for item in args.fallback_models.split(",") if item.strip()]
    return args


def main() -> int:
    args = parse_args()
    if not args.api_key:
        print("Error: set OPENROUTER_API_KEY in .env or pass --api-key.", file=sys.stderr)
        return 1

    input_path = Path(args.input)
    df = pd.read_csv(input_path)
    if args.text_column not in df.columns:
        print(f"Error: missing text column '{args.text_column}'.", file=sys.stderr)
        return 1
    if args.image_column not in df.columns:
        df[args.image_column] = ""
    if args.limit:
        df = df.head(args.limit).copy()

    rows = []
    client = OpenAI(api_key=args.api_key, base_url=OPENROUTER_BASE_URL)
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Refining"):
        claim = str(row[args.text_column]).strip()
        image_path = resolve_image_path(row[args.image_column], input_path.parent)
        try:
            result = refine_row(client, claim, image_path, args)
            rows.append({**row.to_dict(), **flatten_result(result), "refine_error": ""})
        except Exception as exc:
            rows.append({**row.to_dict(), "refine_error": str(exc)})

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

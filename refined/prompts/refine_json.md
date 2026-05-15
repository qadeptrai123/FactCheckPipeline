# Task

Refine a noisy Vietnamese fact-checking input into a clean, structured representation.

# Input

- Claim: {claim}
- Image provided: {image_provided}

# Rules

- Write all generated text in Vietnamese.
- Preserve named entities, numbers, dates, quoted text, locations, and distinctive visual details.
- Normalize the claim without changing its meaning.
- Split the claim into atomic, verifiable facts.
- If an image is provided, describe only visible evidence. Do not infer identity, intent, unseen events, or off-image context.
- Explain whether visible image evidence appears to support, partially support, contradict, or not sufficiently address the claim.
- Do not decide the final truth of the claim.
- Create general search query variants for the next phase. Do not mention any vector database, embedding model, or retrieval backend.
- Use empty arrays when information is absent.

# Output JSON Shape

Return exactly this JSON structure:

```json
{{
  "original_claim": "...",
  "normalized_claim": "...",
  "image_provided": true,
  "language": "vi",
  "claim_atoms": [
    {{
      "id": "c1",
      "text": "Một mệnh đề kiểm chứng được.",
      "check_type": "entity",
      "priority": "high"
    }}
  ],
  "visual_observations": [
    {{
      "id": "v1",
      "text": "Một quan sát trực tiếp từ ảnh.",
      "visible_evidence": ["chi tiết nhìn thấy"],
      "confidence": "high"
    }}
  ],
  "alignment": {{
    "label": "match",
    "text": "Mô tả ngắn quan hệ giữa claim và ảnh."
  }},
  "key_entities": {{
    "people": ["..."],
    "organizations": ["..."],
    "locations": ["..."],
    "dates": ["..."],
    "numbers": ["..."],
    "other": ["..."]
  }},
  "search_queries": {{
    "semantic": ["Câu truy vấn đầy đủ ngữ nghĩa."],
    "keywords": ["từ khóa ngắn"],
    "visual": ["mô tả hình ảnh cần kiểm chứng"]
  }},
  "context_summary": "Tóm tắt ngắn về điều cần kiểm chứng.",
  "ambiguity_notes": ["Điểm mơ hồ hoặc thiếu thông tin."],
  "verification_targets": ["Điều cần tìm bằng chứng ở bước sau."]
}}
```

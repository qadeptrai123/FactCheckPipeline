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
- Create one best primary retrieval query for the whole input.
- Create retrieval query variants for each atomic claim.
- If one or more images are provided, describe only visible evidence across those images. Do not infer identity, intent, unseen events, or off-image context.
- If multiple images are provided, include observations from all relevant images and keep each observation grounded in visible details.
- Explain whether visible image evidence appears to support, partially support, contradict, or not sufficiently address the claim.
- If no image is provided, use an empty visual_observations array and set alignment.label to "not_enough_visual_info".
- Do not decide the final truth of the claim.
- Create general search query variants for the next phase. Do not mention any vector database, embedding model, or retrieval backend.
- Set retrieval_focus booleans based on which modalities are useful for retrieval.
- Extract explicit time, location, and source-type constraints only when they appear in the input or are clearly required by the claim.
- Use empty arrays when information is absent.

# Output JSON Shape

Return exactly this JSON structure:

```json
{{
  "original_claim": "...",
  "normalized_claim": "...",
  "primary_retrieval_query": "Best main retrieval query for finding evidence.",
  "image_provided": true,
  "language": "vi",
  "claim_atoms": [
    {{
      "id": "c1",
      "text": "A verifiable atomic claim.",
      "check_type": "entity",
      "priority": "high",
      "retrieval_queries": ["Evidence retrieval query for this atomic claim."]
    }}
  ],
  "visual_observations": [
    {{
      "id": "v1",
      "text": "A direct observation from the image.",
      "visible_evidence": ["visible detail"],
      "confidence": "high"
    }}
  ],
  "alignment": {{
    "label": "match",
    "text": "Short explanation of the relationship between the claim and the image."
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
    "semantic": ["Full semantic retrieval query."],
    "keywords": ["short keyword"],
    "visual": ["visual description to verify"]
  }},
  "retrieval_focus": {{
    "text": true,
    "image": true,
    "cross_modal": true
  }},
  "constraints": {{
    "time": ["..."],
    "location": ["..."],
    "source_type": ["..."]
  }},
  "context_summary": "Short summary of what needs verification.",
  "ambiguity_notes": ["Ambiguous or missing information."],
  "verification_targets": ["Evidence target for the next verification step."]
}}
```

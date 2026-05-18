# High Quality Evidence Retrieval Pipeline

## Objective

Retrieve the best top-3 evidence items for phase 3 LLM fact-check judging.

The important design choice is to retrieve broadly first, then rerank down to three evidence items. Directly asking Qdrant for `limit=3` is risky because a relevant item can be outside the first few results of any single retriever.

## Inputs

Refined outputs from `refined/refined_outputs_openrouter/*.csv`:

- `refined_primary_retrieval_query`
- `refined_normalized_claim`
- `refined_search_queries.semantic`
- `refined_search_queries.keywords`
- `refined_search_queries.visual`
- `refined_claim_atoms[].retrieval_queries`
- `refined_visual_observations[].text`
- `refined_visual_observations[].visible_evidence`
- `refined_verification_targets`
- `refined_retrieval_focus`

Qdrant collection vectors:

- `text_vector`: Vietnamese text embedding for text chunks
- `sparse`: lexical/keyword sparse vector
- `image_vector`: original CLIP image embedding
- `image_vector_finetuned`: fine-tuned CLIP image embedding

## Query Pack

For each claim, build a query pack:

- Text queries:
  - primary retrieval query
  - normalized claim
  - semantic queries
  - claim atom retrieval queries
  - verification targets
- Keyword queries:
  - keyword queries
  - verification targets
- Visual queries:
  - visual queries
  - visual observation text
  - visible evidence phrases
  - primary retrieval query when `retrieval_focus.cross_modal=true`

## Candidate Generation

Run several retrievers independently:

1. Dense text search
   - Query: text queries
   - Qdrant vector: `text_vector`
   - Filter: `modality="text"`

2. Sparse keyword search
   - Query: keyword query string
   - Qdrant vector: `sparse`
   - Filter: `modality="text"`

3. Original CLIP text-to-image search
   - Query: visual queries embedded by original CLIP text encoder
   - Qdrant vector: `image_vector`
   - Filter: `modality="image"`

4. Fine-tuned CLIP text-to-image search
   - Query: visual queries embedded by fine-tuned CLIP text encoder
   - Qdrant vector: `image_vector_finetuned`
   - Filter: `modality="image"`

Recommended starting point:

- Retrieve 30 candidates per branch.
- Keep `with_vectors=False`.
- Keep `with_payload=True` because phase 3 needs evidence metadata.

## Query Strategy

The retrieval pipeline is intentionally multi-branch. Each branch answers a different retrieval need:

- `text_dense`: semantic text retrieval.
  - Best for paraphrases and general semantic matching.
  - Uses BKAI Vietnamese text embeddings.

- `text_sparse`: lexical/keyword retrieval.
  - Best for exact entities, dates, numbers, source-specific terms, and named organizations.
  - Uses the `sparse` vector.

- `image_clip`: original CLIP text-to-image retrieval.
  - Best for checking whether original CLIP can retrieve visually relevant images from visual queries.
  - Uses `image_vector`.

- `image_clip_finetuned`: fine-tuned CLIP text-to-image retrieval.
  - Best for checking whether the fine-tuned CLIP improves image evidence retrieval.
  - Uses `image_vector_finetuned`.

For fair A/B evaluation between original CLIP and fine-tuned CLIP, do **not** search both image vectors in the same experiment. Run separate experiments:

- Experiment A: use only `image_vector`
- Experiment B: use only `image_vector_finetuned`

The text branches stay the same across both experiments. This isolates whether the image vector changed retrieval quality.

Do not use `limit=3` directly during candidate generation. Retrieve wider first:

- top 10 per query per branch as a lightweight baseline
- top 30-80 per branch for stronger final experiments

Then fuse and rerank down to top 3 evidence.

## Fusion

Use weighted Reciprocal Rank Fusion because dense text scores, sparse scores, and CLIP scores are not directly comparable.

Initial weights:

- `text_dense`: 1.20
- `text_sparse`: 1.00
- `image_clip`: 0.85
- `image_clip_finetuned`: 1.25

These are starting values only. Tune them against a golden set.

For fair first-pass evaluation, use equal branch weights:

```python
BRANCH_WEIGHTS = {
    "text_dense": 1.0,
    "text_sparse": 1.0,
    "image_clip": 1.0,
    "image_clip_finetuned": 1.0,
}
```

Weighted RRF should be introduced only after the equal-weight baseline is measured. The specific formula is:

```python
rrf_score += branch_weight / (RRF_K + rank_in_branch)
```

This is used because raw scores from dense vectors, sparse vectors, and CLIP vectors are not directly comparable.

## Reranking

Rerank fused candidates before taking top 3.

Current notebook implementation:

- weighted RRF score
- branch agreement bonus
- query/payload token overlap
- optional CrossEncoder boost for text candidates

Recommended next additions:

- Text reranker:
  - CrossEncoder `(query, text_chunk)` relevance scoring
  - Rerank top 30-80 text candidates

- Image reranker:
  - lightweight VLM relevance scoring
  - Input: claim/refined query + candidate image + title/date/source
  - Output: relevance score only, not truth label

## Top-3 Evidence Package

The final evidence objects should include:

- rank
- point id
- modality
- final score
- RRF score
- source branch details
- title
- source
- URL
- date
- text snippet
- image path
- corpus id

This package is passed to the LLM judge in phase 3.

## Evaluation

Before trusting the top-3 output, build a small golden set:

- `claim_id`
- query/refined query
- positive text chunk ids
- positive image paths
- positive URLs
- expected modality: `text`, `image`, or `both`

Measure:

- Recall@3
- Recall@10
- Recall@20
- MRR@10
- text evidence hit rate
- image evidence hit rate

For fact-checking, Recall@10/20 should be optimized first. Top-3 quality improves after reranking.

### Evaluation Matrix

Run metrics separately for each condition:

| Dimension | Values |
|---|---|
| Refiner | `gemini-2.5-flash`, `gpt4o_mini` |
| Collection | `fixed_size`, `semantic` |
| Image vector | `image_vector`, `image_vector_finetuned` |
| Reranker | `off`, `on` |

Recommended sequence:

1. Baseline without reranker:
   - both refiners
   - both collections
   - both image vectors

2. Reranker experiment:
   - same settings
   - enable `namdp-ptit/ViRanker`

3. Compare:
   - original CLIP vs fine-tuned CLIP
   - fixed-size chunking vs semantic chunking
   - Gemini refine vs GPT-4o mini refine
   - no-reranker vs reranker

### Ground Truth Matching

Gold schema:

```text
id
claim
image
text_evidences
text_evidences_url
image_evidences
image_evidence_path
reason
label
```

Text evidence is counted as a hit when one of these is true:

1. Retrieved payload URL exactly matches `text_evidences_url`.
2. Retrieved text/title/source has high token coverage against a gold evidence fragment.
3. Retrieved text/title/source has high token F1 against a gold evidence fragment.

Coverage is necessary because retrieved chunks can be much longer than the gold evidence sentence. If a long retrieved chunk contains the full gold sentence, pure F1 may be unfairly low due to many extra tokens.

Current suggested thresholds:

```python
text_hit = token_coverage >= 0.60 or token_f1 >= 0.45
```

Image evidence is counted as a hit when:

```python
normalize_path(retrieved_payload["image_path"]) == normalize_path(gold["image_evidence_path"])
```

### Metric Definitions

For one claim, let `top_k` be the first `k` retrieved evidence items after fusion/reranking.

#### Evidence Recall@k

Whether top-k contains at least one correct evidence item, text or image.

```python
evidence_recall_at_k = int(any(item.text_hit or item.image_hit for item in top_k))
```

Dataset score:

```python
mean(evidence_recall_at_k over all claims)
```

Use this as the main phase-3 readiness metric.

#### Text Recall@k

Whether top-k contains a correct text evidence item.

```python
text_recall_at_k = int(any(item.text_hit for item in top_k))
```

This isolates text retrieval quality.

#### Image Recall@k

Whether top-k contains a correct image evidence item.

```python
image_recall_at_k = int(any(item.image_hit for item in top_k))
```

This isolates image retrieval quality and is the key metric for comparing `image_vector` vs `image_vector_finetuned`.

#### Full Recall@k

Whether top-k contains both a correct text evidence item and a correct image evidence item.

```python
full_recall_at_k = int(
    any(item.text_hit for item in top_k)
    and any(item.image_hit for item in top_k)
)
```

This is stricter than Evidence Recall@k. It is useful when the final LLM judge needs both modalities.

#### MRR@3

Mean Reciprocal Rank over the first correct evidence item within top 3.

For one claim:

```python
if first_hit_rank in {1, 2, 3}:
    mrr_at_3 = 1 / first_hit_rank
else:
    mrr_at_3 = 0
```

Dataset score:

```python
mean(mrr_at_3 over all claims)
```

This measures whether correct evidence appears near the top, not just somewhere in the candidate list.

#### Recall@10 and Recall@20

Use Recall@10/20 to diagnose whether candidate generation is good enough:

- High Recall@20 but low Recall@3 means retrieval finds the evidence but fusion/reranking is weak.
- Low Recall@20 means query generation, embeddings, collection choice, or modality strategy is missing evidence.

### CSV Outputs

The evaluation notebook writes:

```text
database/retrieval_eval_outputs/
  metrics_summary.csv
  retrieval_results_long.csv
  claim_metrics_long.csv
  retrieval_details_<experiment_id>.csv
  claim_metrics_<experiment_id>.csv
```

`metrics_summary.csv` is the main comparison table.

`retrieval_results_long.csv` stores ranked evidence rows with hit flags and matching diagnostics.

`claim_metrics_long.csv` stores per-claim metrics for every experiment.

## Qdrant Performance Notes

For low-latency retrieval:

- Use payload indexes for filtered fields such as `modality`.
- Keep `with_vectors=False` during search.
- Batch queries where possible when evaluating many claims.
- If exact search is good but approximate search misses evidence, increase query-time `hnsw_ef`.
- If filtered search is slow, confirm payload indexes exist before blaming vector search.
- Watch optimizer status after bulk upload; unindexed segments can distort latency.

For memory:

- Multiple named vectors increase storage and index memory but do not increase point count.
- If memory pressure appears, consider scalar quantization or on-disk payload/sparse indexes.
- Do not put latency-critical HNSW indexes on disk unless storage is fast and the workload tolerates slower queries.

## Notebook

Implementation notebook:

- `database/high_quality_retrieval.ipynb`

## Reranker Recommendation

### Selected Vietnamese Reranker

Use `namdp-ptit/ViRanker` as the primary text reranker.

Selected setup for the notebooks:

```python
CROSS_ENCODER_MODEL = "namdp-ptit/ViRanker"
CROSS_ENCODER_MAX_LENGTH = 512
```

Use it as a text evidence reranker only:

- Rerank the top 30-80 text candidates after Qdrant candidate generation.
- Do not use it for image evidence directly; image evidence needs CLIP/VLM relevance scoring or OCR-to-text reranking.

### Previous Vietnamese Reranker Candidate

`AITeamVN/Vietnamese_Reranker` was the initial candidate, but it is no longer the default because it caused runtime issues in the local setup.

Its original reasons:

- Vietnamese-specific.
- Apache 2.0 license.
- Fine-tuned from `BAAI/bge-reranker-v2-m3`.
- Trained on about 1,100,000 Vietnamese query-positive-negative triplets.
- Supports long passages with max sequence length 2304.
- Reported evaluation on Legal Zalo 2021:
  - Accuracy@1: 0.7944
  - Accuracy@3: 0.9324
  - Accuracy@5: 0.9537
  - Accuracy@10: 0.9740
  - MRR@10: 0.8672
- Current observed Hugging Face downloads: about 7.8k/month.
- Developer names listed on the model card are Vietnamese.

### Strong Global Baseline

Also test `BAAI/bge-reranker-v2-m3`.

Reasons:

- Very popular multilingual reranker.
- About 0.6B parameters.
- Apache 2.0 license.
- Hugging Face shows about 11M+ monthly downloads.
- BAAI recommends it for multilingual reranking and efficiency.

Use this as a baseline against `namdp-ptit/ViRanker`. If ViRanker wins on the project golden set, keep ViRanker. If not, keep BGE reranker.

### Lightweight Vietnamese Alternative

`itdainb/PhoRanker` is worth testing if latency is more important than maximum quality.

Reasons:

- Vietnamese cross-encoder for text ranking.
- Based on Vietnamese/PhoBERT-style modeling.
- About 0.1B parameters.
- Apache 2.0 license.
- Hugging Face shows about 3.3k monthly downloads.
- Created by Dai Nguyen Ba according to the model card citation.

Use this when GPU memory or latency is tight.

### Practical Choice

Use this comparison order if you later benchmark alternatives:

1. `namdp-ptit/ViRanker`
2. `BAAI/bge-reranker-v2-m3`
3. `itdainb/PhoRanker`
4. `AITeamVN/Vietnamese_Reranker`

Evaluate all three on the same golden set using Recall@3, MRR@10, and NDCG@10.

For phase 3, the best reranker is the one that puts valid evidence in the final top 3 most consistently, not necessarily the one with the largest global benchmark score.

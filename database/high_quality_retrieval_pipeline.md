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

## Fusion

Use weighted Reciprocal Rank Fusion because dense text scores, sparse scores, and CLIP scores are not directly comparable.

Initial weights:

- `text_dense`: 1.20
- `text_sparse`: 1.00
- `image_clip`: 0.85
- `image_clip_finetuned`: 1.25

These are starting values only. Tune them against a golden set.

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

### Recommended Vietnamese Reranker

Use `AITeamVN/Vietnamese_Reranker` as the primary text reranker.

Reasons:

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

Recommended usage in this project:

- Use it as a text evidence reranker only.
- Rerank the top 30-80 text candidates after Qdrant candidate generation.
- Do not use it for image evidence directly; image evidence needs CLIP/VLM relevance scoring or OCR-to-text reranking.

### Strong Global Baseline

Also test `BAAI/bge-reranker-v2-m3`.

Reasons:

- Very popular multilingual reranker.
- About 0.6B parameters.
- Apache 2.0 license.
- Hugging Face shows about 11M+ monthly downloads.
- BAAI recommends it for multilingual reranking and efficiency.

Use this as a baseline against `AITeamVN/Vietnamese_Reranker`. If the Vietnamese model wins on the project golden set, keep the Vietnamese model. If not, keep BGE reranker.

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

Selected setup for the notebook:

```python
CROSS_ENCODER_MODEL = "AITeamVN/Vietnamese_Reranker"
CROSS_ENCODER_MAX_LENGTH = 2304
```

Use this comparison order if you later benchmark alternatives:

1. `AITeamVN/Vietnamese_Reranker`
2. `BAAI/bge-reranker-v2-m3`
3. `itdainb/PhoRanker`

Evaluate all three on the same golden set using Recall@3, MRR@10, and NDCG@10.

For phase 3, the best reranker is the one that puts valid evidence in the final top 3 most consistently, not necessarily the one with the largest global benchmark score.

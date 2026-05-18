# Pipeline Evaluation Retrieval

## Mục Tiêu

Đánh giá retrieval để lấy evidence cho phase judging. Với mỗi claim, output cần có:

- `top3_text_urls`: top 3 URL từ text retrieval.
- `top3_image_urls`: top 3 URL từ image retrieval.

Hai danh sách này được xử lý riêng, không bắt buộc khác nhau đôi một.

## File Chính

Notebook chạy evaluation:

```text
database/retrieval_evaluation.ipynb
```

Output chính:

```text
database/retrieval_eval_outputs/metrics_summary.csv
database/retrieval_eval_outputs/claim_metrics_long.csv
database/retrieval_eval_outputs/retrieval_results_long.csv
database/retrieval_eval_outputs/top3_urls_long.csv
database/retrieval_eval_outputs/by_experiment/<experiment_id>/
```

Ý nghĩa:

- `metrics_summary.csv`: mỗi dòng là một experiment, dùng để chọn cấu hình tốt nhất.
- `claim_metrics_long.csv`: mỗi dòng là metric của một claim trong một experiment.
- `retrieval_results_long.csv`: mỗi dòng là một retrieved item, dùng để debug ranking.
- `top3_urls_long.csv`: mỗi dòng là top 3 URL text và top 3 URL image cho một claim.
- `by_experiment/<experiment_id>/`: file CSV riêng cho từng combination.

Mỗi folder experiment có:

```text
retrieval_details.csv
claim_metrics.csv
top3_urls.csv
```

## Dữ Liệu Đầu Vào

Ground truth:

```text
FinalDataset/claims_merged.csv
```

Refined queries:

```text
refined/refined_outputs_openrouter/refined_gemini-2.5-flash.csv
refined/refined_outputs_openrouter/refined_gpt4o_mini.csv
```

Gold source URL:

```text
text_evidences_url
```

Vì mỗi claim hiện chỉ có một gold URL, `Recall@K` về bản chất tương đương `Hit@K`. Do đó metric mới dùng tên `Hit@K` cho đúng ngữ cảnh hơn.

## Vector Trong Qdrant

Collection có các named vectors:

- `text_vector`: dense text embedding tiếng Việt.
- `sparse`: lexical/sparse vector.
- `image_vector`: CLIP image vector gốc.
- `image_vector_finetuned`: CLIP image vector fine-tuned.

Một image point có thể có cả `image_vector` và `image_vector_finetuned`, nhưng số lượng point trong collection không tăng.

## Query Pack

Mỗi claim tạo các nhóm query:

- `text_queries`: dùng cho dense text retrieval.
- `keyword_query`: dùng cho sparse retrieval.
- `visual_queries`: dùng cho text-to-image retrieval bằng CLIP text encoder.

## Pipeline Eval

Mỗi experiment chạy theo hai lane riêng:

### 1. Text Lane

```text
text_queries -> text_vector
keyword_query -> sparse
RRF fusion -> optional ViRanker rerank -> top K text items
```

Reranker:

```text
namdp-ptit/ViRanker
```

Reranker chỉ áp dụng cho text candidate.

### 2. Image Lane

```text
visual_queries -> CLIP text encoder -> image_vector hoặc image_vector_finetuned
RRF ranking -> top K image items
```

Image lane không dùng ViRanker, vì ViRanker là text cross-encoder. Như vậy image retrieval được đánh giá riêng, không bị text reranker đẩy xuống trong cùng một ranked list.

## Ma Trận Experiment

Chạy đủ sẽ có:

```text
2 refiners × 2 collections × 2 image variants × 2 reranker settings = 16 experiments
```

Các chiều:

- Refiner: `gemini-2.5-flash`, `gpt4o_mini`
- Collection: `fixed_size`, `semantic`
- Image vector: `clip`, `clip_finetuned`
- Reranker: `False`, `True`

## Output Top 3 URL

Với mỗi claim, notebook lưu:

```text
top3_text_urls
top3_image_urls
top3_text_items
top3_image_items
```

Trong đó:

- `top3_text_urls`: danh sách 3 URL đầu từ text lane.
- `top3_image_urls`: danh sách 3 URL đầu từ image lane.
- `top3_text_items`, `top3_image_items`: JSON chi tiết gồm rank, point_id, url, title, image_path, score.

## Metrics Chính

### `text_source_hit_at_k`

Top-K text items có ít nhất một URL trùng gold URL.

```python
text_source_hit_at_k = int(
    any(item.url == gold.text_evidences_url for item in top_k_text)
)
```

### `image_source_hit_at_k`

Top-K image items có ít nhất một URL trùng gold URL.

```python
image_source_hit_at_k = int(
    any(item.url == gold.text_evidences_url for item in top_k_image)
)
```

### `source_hit_at_k`

Text lane hoặc image lane tìm được đúng gold URL.

```python
source_hit_at_k = int(
    text_source_hit_at_k or image_source_hit_at_k
)
```

Đây là metric chính nếu mục tiêu là tìm đúng source URL cho phase judging.

### `text_source_mrr_at_3`

Đo URL đúng trong text lane xuất hiện ở rank mấy trong top 3.

```python
text_source_mrr_at_3 = 1 / first_text_source_hit_rank
```

Nếu không hit trong top 3 thì bằng `0`.

### `image_source_mrr_at_3`

Đo URL đúng trong image lane xuất hiện ở rank mấy trong top 3.

```python
image_source_mrr_at_3 = 1 / first_image_source_hit_rank
```

Nếu không hit trong top 3 thì bằng `0`.

### `source_mrr_at_3`

Lấy lane tốt hơn giữa text và image:

```python
source_mrr_at_3 = max(text_source_mrr_at_3, image_source_mrr_at_3)
```

## Metrics Phụ

### `text_evidence_hit_at_k`

Text lane có text evidence đúng. Hit nếu:

- URL trùng gold URL, hoặc
- token coverage >= `0.60`, hoặc
- token F1 >= `0.45`.

### `image_exact_hit_at_k`

Image lane có image path trùng exact `image_evidence_path`.

```python
image_exact_hit_at_k = int(
    any(item.image_path == gold.image_evidence_path for item in top_k_image)
)
```

Metric này rất strict, chỉ dùng để diagnostic.

### `evidence_hit_at_k`

Text evidence hit hoặc exact image hit.

```python
evidence_hit_at_k = int(
    text_evidence_hit_at_k or image_exact_hit_at_k
)
```

## Cách Chọn Cấu Hình

Ưu tiên theo thứ tự:

1. `source_hit_at_3`
2. `source_mrr_at_3`
3. `text_source_hit_at_3`
4. `image_source_hit_at_3`

Nếu `image_source_hit_at_3` thấp nhưng `text_source_hit_at_3` cao, phase judging vẫn khả thi vì mục tiêu chính là đưa đúng source URL và text evidence cho LLM judge.

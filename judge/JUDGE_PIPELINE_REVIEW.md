# Luồng Notebook Judge Evaluation

Notebook:

```text
judge/judge_pipeline_evaluation.ipynb
```

## Input CSV

Retrieval top URLs:

```text
database/retrieval_eval_outputs/by_experiment/gemini-2.5-flash__semantic__clip_finetuned__reranker_1/top3_urls.csv
```

Refined claim:

```text
refined/refined_outputs_openrouter/refined_gemini-2.5-flash.csv
```

Corpus để lấy nội dung evidence theo URL:

```text
chunking_scripts/final_corpus.csv
```

Nhãn thật để đánh giá:

```text
FinalDataset/claims_merged.csv
```

## Luồng Xử Lý

1. Load `top3_urls.csv`, refined CSV và corpus CSV.
2. Merge retrieval output với refined claim theo `id`.
3. Với mỗi claim, tạo danh sách evidence URL:
   - lấy toàn bộ `top3_text_urls`
   - lấy thêm top 1 từ `top3_image_urls`
   - khử trùng URL theo thứ tự xuất hiện
4. Với từng URL, lấy `content` từ `final_corpus.csv`.
5. Gọi Gemini 2.5 Flash một lần cho từng evidence để lấy:
   - `thought_process`
   - `relation`
   - `extracted_facts`
6. Gọi Gemini 2.5 Flash thêm một lần để chốt verdict cuối:
   - `verdict`
   - `explanation`
   - `thought_process`

## Output CSV

Kết quả judge:

```text
judge/judge_outputs_openrouter/factchecking_final_results_gemini-2.5-flash__gemini-2.5-flash__semantic__clip_finetuned__reranker_1.csv
```

Các cột chính:

```text
claim_id
normalized_claim
verdict
explanation
thought_process
top3_urls_used
map_results
gold_label
final_error
```

Metric report:

```text
judge/judge_outputs_openrouter/factchecking_metrics_gemini-2.5-flash__gemini-2.5-flash__semantic__clip_finetuned__reranker_1.csv
```

Confusion matrix:

```text
judge/judge_outputs_openrouter/factchecking_confusion_gemini-2.5-flash__gemini-2.5-flash__semantic__clip_finetuned__reranker_1.csv
```

Bảng lỗi:

```text
judge/judge_outputs_openrouter/factchecking_errors_gemini-2.5-flash__gemini-2.5-flash__semantic__clip_finetuned__reranker_1.csv
```

## Metrics Đánh Giá

Verdict được map về nhãn dataset:

```text
SUPPORTED -> supported
REFUTED   -> refuted
NEI       -> nei
```

Metrics sử dụng:

```text
accuracy
precision theo từng nhãn
recall theo từng nhãn
F1 theo từng nhãn
macro F1
weighted F1
confusion matrix
```

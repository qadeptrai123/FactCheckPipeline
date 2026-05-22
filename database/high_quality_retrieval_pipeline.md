# Pipeline Import Và Evaluation Retrieval

Tài liệu này mô tả đầy đủ pipeline đang dùng để import dữ liệu đa phương thức vào Qdrant và chạy retrieval evaluation để lấy evidence cho bước judging.

Notebook import chính:

```text
database/multimodal_qdrant.ipynb
```

Notebook evaluation chính:

```text
database/retrieval_evaluation.ipynb
```

Notebook raw-claim baseline:

```text
database/retrieval_eval_raw.ipynb
```

## Mục Tiêu

Pipeline có hai phần:

1. Import corpus vào Qdrant dưới dạng text points và image points.
2. Retrieval top evidence cho mỗi claim, sau đó xuất CSV để đánh giá và đưa sang phase judging.

Output retrieval cần có riêng hai lane:

```text
top3_text_urls
top3_image_urls
top3_text_items
top3_image_items
```

Text retrieval và image retrieval được xử lý riêng. Hai danh sách URL không bắt buộc khác nhau đôi một.

## Nguồn Dữ Liệu

### Corpus chunk cho text import

```text
chunking_scripts/chunks_fixed_size.csv
chunking_scripts/chunks_semantic.csv
```

Hai file này được map vào hai collection:

```text
fixed_size  -> chunking_scripts/chunks_fixed_size.csv
semantic    -> chunking_scripts/chunks_semantic.csv
```

Các dòng có `modality == "text"` được dùng để tạo text points.

### Corpus gốc cho image import

```text
FinalDataset/final_corpus.csv
```

Cột `media` chứa danh sách đường dẫn ảnh, phân tách bằng:

```text
 |
```

Sau khi explode, mỗi ảnh hợp lệ trở thành một image point.

### Ground truth cho retrieval evaluation

```text
FinalDataset/claims_merged.csv
```

Các cột chính:

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

Gold source URL:

```text
text_evidences_url
```

Gold image path:

```text
image_evidence_path
```

### Refined query files

```text
refined/refined_outputs_openrouter/refined_gemini-2.5-flash.csv
refined/refined_outputs_openrouter/refined_gpt4o_mini.csv
```

Các file refined được merge với `claims_merged.csv` theo:

```text
id
claim
image
```

## Qdrant Config

Qdrant URL:

```text
http://localhost:6333
```

Collections đang dùng:

```text
fixed_size
semantic
```

Notebook cũng khai báo tên v2:

```text
v2_fixed_size_hybrid
v2_semantic_hybrid
```

Nhưng pipeline import hiện tại upsert vào:

```text
fixed_size
semantic
```

## Collection Schema

Mỗi collection có ba named dense vectors và một sparse vector:

```text
text_vector
image_vector
image_vector_finetuned
sparse
```

Chi tiết:

```text
text_vector             768d, cosine, từ bkai-foundation-models/vietnamese-bi-encoder
image_vector            512d, cosine, từ sentence-transformers/clip-ViT-B-32
image_vector_finetuned  512d, cosine, từ models/clip-vit-b32-finetuned-final-final/best
sparse                  sparse vector, modifier IDF
```

Collection được tạo với:

```text
vectors_config:
  text_vector: 768, cosine
  image_vector: 512, cosine
  image_vector_finetuned: 512, cosine

sparse_vectors_config:
  sparse: IDF
```

Nếu `RECREATE_COLLECTIONS = True`, notebook sẽ xóa collection cũ rồi tạo lại.

## Model Config

Text embedding:

```text
bkai-foundation-models/vietnamese-bi-encoder
```

Original CLIP:

```text
sentence-transformers/clip-ViT-B-32
```

Fine-tuned CLIP:

```text
models/clip-vit-b32-finetuned-final-final/best
```

Reranker:

```text
namdp-ptit/ViRanker
```

Thiết bị:

```text
cuda nếu có GPU
cpu nếu không có GPU
```

Batch size trong notebook:

```text
TEXT_BATCH = 16
IMG_BATCH = 8
Text upload batch = 256
Image upsert batch = 512
```

## Import Pipeline Tổng Quát

```text
chunks_fixed_size.csv / chunks_semantic.csv
        |
        |-- lọc modality == "text"
        |-- embed bằng Vietnamese bi-encoder
        |-- tạo sparse vector
        |-- upsert text points vào Qdrant

final_corpus.csv
        |
        |-- đọc cột media
        |-- split đường dẫn ảnh bằng " | "
        |-- explode thành từng image_path
        |-- kiểm tra file ảnh hợp lệ
        |-- embed ảnh bằng original CLIP
        |-- embed ảnh bằng fine-tuned CLIP
        |-- xóa image points cũ trong collection
        |-- upsert image points mới vào Qdrant
```

## Bước 1: Tạo Collection

Notebook kết nối Qdrant và tạo hai collection:

```text
fixed_size
semantic
```

Với mỗi collection:

1. Kiểm tra collection đã tồn tại hay chưa.
2. Nếu tồn tại và `RECREATE_COLLECTIONS = True`, xóa collection cũ.
3. Tạo collection mới với `text_vector`, `image_vector`, `image_vector_finetuned`, và `sparse`.
4. In trạng thái tạo collection.

Collection sau khi tạo có thể chứa cả text points và image points trong cùng một collection. Hai loại point được phân biệt bằng payload:

```text
modality = "text"
modality = "image"
```

## Bước 2: Chuẩn Bị Text Rows

Input:

```text
chunking_scripts/chunks_fixed_size.csv
chunking_scripts/chunks_semantic.csv
```

Luồng xử lý:

1. Đọc CSV bằng `utf-8-sig`.
2. Lọc các dòng có:

```text
modality == "text"
```

3. Chuyển DataFrame thành list record.
4. Lưu theo strategy:

```text
chunks_by_coll["fixed_size"]["text"]
chunks_by_coll["semantic"]["text"]
```

Text rows được upsert vào collection tương ứng:

```text
fixed_size rows -> fixed_size collection
semantic rows   -> semantic collection
```

## Bước 3: Embed Và Upsert Text Points

Với mỗi strategy:

```text
fixed_size
semantic
```

Pipeline text:

```text
text row
  -> text
  -> Vietnamese bi-encoder
  -> text_vector
  -> hashed sparse vector
  -> sparse
  -> PointStruct
  -> upload_points
```

Text point ID:

```text
uuid5(NAMESPACE_DNS, f"{row_id}_{chunk_index}")
```

Text point vector:

```text
text_vector: dense embedding 768d
sparse: sparse lexical vector
```

Text point payload:

```text
chunk_id
modality
row_id
chunk_index
text
image_path
text_chunk_id
source
url
title
author
date
corpus_id
```

Text upload:

```text
client.upload_points(collection_name=coll, points=text_points, batch_size=256)
```

Sau bước này:

```text
fixed_size có text points
semantic có text points
```

## Bước 4: Chuẩn Bị Image Rows

Input:

```text
FinalDataset/final_corpus.csv
```

Pipeline image metadata:

```text
final_corpus.csv
  -> đọc cột media
  -> split bằng " | "
  -> tạo image_path dạng set
  -> explode image_path
  -> dropna image_path
  -> lấy unique_paths
```

Sau khi explode:

```text
một corpus row có nhiều ảnh -> nhiều image rows
```

## Bước 5: Kiểm Tra Ảnh Hợp Lệ

Mỗi `image_path` được resolve theo root:

```text
./FinalDataset/<image_path>
```

Điều kiện hợp lệ:

```text
file tồn tại
file là file thật
PIL mở và verify được
convert RGB được
width > 1
height > 1
```

Kiểm tra ảnh chạy song song bằng:

```text
ThreadPoolExecutor(max_workers=10)
```

Kết quả:

```text
valid_paths_set
df_valid
```

`df_valid` chỉ giữ các dòng có `image_path` hợp lệ.

## Bước 6: Load Image Models

Notebook load hai CLIP model:

```text
sentence-transformers/clip-ViT-B-32
models/clip-vit-b32-finetuned-final-final/best
```

Kiểm tra dimension:

```text
original CLIP dimension == 512
fine-tuned CLIP dimension == 512
```

Nếu dimension không đúng, pipeline dừng bằng lỗi.

## Bước 7: Embed Image Points

Với mỗi batch image:

```text
image_path list
  -> load ảnh hợp lệ
  -> original CLIP image embedding
  -> fine-tuned CLIP image embedding
```

Mỗi image point có cả hai vector:

```text
image_vector: original CLIP 512d
image_vector_finetuned: fine-tuned CLIP 512d
```

Nếu ảnh lỗi trong lúc batch embedding:

```text
vector giữ dạng zero-vector đúng vị trí index
```

Cách này giữ số lượng vector khớp với số lượng row trong batch.

## Bước 8: Upsert Image Points

Trước khi upsert image mới, notebook xóa toàn bộ image points cũ trong từng collection:

```text
delete where modality == "image"
```

Sau đó upsert lại image points vào từng collection:

```text
fixed_size
semantic
```

Image point ID:

```text
uuid5(NAMESPACE_DNS, f"{id}_{image_path}")
```

Image point vector:

```text
image_vector: original CLIP image embedding 512d
image_vector_finetuned: fine-tuned CLIP image embedding 512d
```

Image point payload:

```text
modality
corpus_id
image_path
source
url
title
author
date
```

Image upload:

```text
client.upload_points(collection_name=collection, points=batch_points)
```

Batch size:

```text
512 image rows per batch
```

Sau mỗi batch:

```text
del batch objects
gc.collect()
```

Sau bước này, mỗi collection có:

```text
text points với text_vector + sparse
image points với image_vector + image_vector_finetuned
```

## Bước 9: Kiểm Tra Collection

Notebook in thống kê từng collection:

```text
collection name
points_count
indexed on text_vector + image_vector + image_vector_finetuned + sparse
```

Collections được kiểm tra:

```text
fixed_size
semantic
```

## Bước 10: Smoke Test Text-To-Image

Notebook có bước test text-to-image trên image lane.

Query test mặc định:

```text
thủ tướng
```

Luồng test:

```text
query text
  -> original CLIP text embedding
  -> search image_vector
  -> top image results

query text
  -> fine-tuned CLIP text embedding
  -> search image_vector_finetuned
  -> top image results
```

Filter khi search:

```text
modality == "image"
```

Collection test mặc định:

```text
semantic
```

Top-K test:

```text
8
```

Output hiển thị ảnh, title, score, và `image_path` để kiểm tra thủ công.

## Retrieval Query Pipeline Với Refined Output

Notebook:

```text
database/retrieval_evaluation.ipynb
```

Mỗi claim refined tạo ba nhóm query:

```text
text_queries
keyword_query
visual_queries
```

Text dense queries lấy từ:

```text
refined_primary_retrieval_query
refined_normalized_claim
refined_search_queries.semantic
refined_claim_atoms[*].retrieval_queries
refined_verification_targets
```

Sparse keyword query lấy từ:

```text
refined_search_queries.keywords
refined_verification_targets
```

Image lane CLIP text queries lấy từ:

```text
refined_search_queries.visual
refined_visual_observations[*].text
refined_visual_observations[*].visible_evidence
refined_primary_retrieval_query nếu refined_retrieval_focus.cross_modal == true
```

Các query được dedupe trước khi retrieval.

Giới hạn query:

```text
MAX_TEXT_QUERIES = 6
MAX_VISUAL_QUERIES = 6
CANDIDATES_PER_QUERY = 10
TOP_K_VALUES = [3, 10, 20]
FINAL_TOP_K = 3
```

## Retrieval Query Pipeline Với Raw Claim

Notebook:

```text
database/retrieval_eval_raw.ipynb
```

Input:

```text
FinalDataset/claims_merged.csv
```

Raw query pack:

```text
text_queries   = [claim]
keyword_query  = claim
visual_queries = [claim]
```

Raw notebook không dùng refined fields.

Raw notebook vẫn đi theo cùng cơ chế CLIP của notebook refined:

```text
raw claim text -> CLIP text encoder -> search image_vector_finetuned
```

Nó không encode file ảnh claim làm query image.

Raw config hiện tại:

```text
collection = semantic
image_variant = clip_finetuned
use_reranker = True
```

Output:

```text
database/retrieval_eval_raw_outputs/
```

## Retrieval Lane

Mỗi experiment chạy hai lane riêng.

### Text Lane

```text
text_queries
  -> Vietnamese bi-encoder
  -> query Qdrant text_vector
  -> text dense candidates

keyword_query
  -> sparse vector
  -> query Qdrant sparse
  -> text sparse candidates

text dense candidates + text sparse candidates
  -> weighted RRF
  -> optional ViRanker rerank
  -> top K text items
```

Filter:

```text
modality == "text"
```

Reranker chỉ áp dụng cho text candidates.

### Image Lane

```text
visual_queries
  -> CLIP text encoder
  -> query image_vector hoặc image_vector_finetuned
  -> image candidates
  -> weighted RRF
  -> top K image items
```

Filter:

```text
modality == "image"
```

Image lane không dùng ViRanker.

## Fusion Và Ranking

Mỗi branch có weight:

```text
text_dense: 1.00
text_sparse: 1.00
image_clip: 1.00
image_clip_finetuned: 1.00
```

RRF constant:

```text
RRF_K = 60
```

Final score trong text lane:

```text
rrf_score + branch_bonus + reranker_boost
```

Final score trong image lane:

```text
rrf_score + branch_bonus
```

## Refined Evaluation Experiment Matrix

Chạy đầy đủ:

```text
2 refiners × 2 collections × 2 image variants × 2 reranker settings = 16 experiments
```

Refiners:

```text
gemini-2.5-flash
gpt4o_mini
```

Collections:

```text
fixed_size
semantic
```

Image variants:

```text
clip
clip_finetuned
```

Reranker:

```text
False
True
```

Experiment ID format:

```text
<refiner>__<collection>__<image_variant>__reranker_<0_or_1>
```

Ví dụ:

```text
gemini-2.5-flash__semantic__clip_finetuned__reranker_1
```

## Raw Evaluation Experiment Matrix

Raw notebook hiện chạy một experiment:

```text
raw_claim__semantic__clip_finetuned__reranker_1
```

## Evaluation Outputs

Refined output dir:

```text
database/retrieval_eval_outputs/
```

Raw output dir:

```text
database/retrieval_eval_raw_outputs/
```

Các file chính:

```text
metrics_summary.csv
claim_metrics_long.csv
retrieval_results_long.csv
top3_urls_long.csv
best_combinations_by_metric.csv
```

Per-experiment output:

```text
by_experiment/<experiment_id>/retrieval_details.csv
by_experiment/<experiment_id>/claim_metrics.csv
by_experiment/<experiment_id>/top3_urls.csv
```

Ý nghĩa:

```text
metrics_summary.csv          mỗi dòng là một experiment
claim_metrics_long.csv       mỗi dòng là metric của một claim trong một experiment
retrieval_results_long.csv   mỗi dòng là một retrieved item
top3_urls_long.csv           mỗi dòng là top 3 text URL và top 3 image URL cho một claim
retrieval_details.csv        retrieved items của một experiment
claim_metrics.csv            per-claim metrics của một experiment
top3_urls.csv                top 3 URL/items của một experiment
```

## Top-3 Output Schema

Với mỗi claim:

```text
top3_text_urls
top3_image_urls
top3_text_items
top3_image_items
```

`top3_text_urls`:

```text
danh sách 3 URL đầu từ text lane
```

`top3_image_urls`:

```text
danh sách 3 URL đầu từ image lane
```

`top3_text_items` và `top3_image_items`:

```text
rank
point_id
url
title
image_path
score
```

## Main Metrics

Vì mỗi claim hiện có một gold source URL, `Recall@K` tương đương `Hit@K`.

### Source Hit

```text
text_source_hit_at_k
```

Top-K text items có ít nhất một item có URL trùng `text_evidences_url`.

```text
image_source_hit_at_k
```

Top-K image items có ít nhất một item có URL trùng `text_evidences_url`.

```text
source_hit_at_k
```

Text lane hoặc image lane tìm được đúng gold source URL.

### Source MRR

```text
text_source_mrr_at_3
```

Reciprocal rank của URL đúng trong top 3 text items.

```text
image_source_mrr_at_3
```

Reciprocal rank của URL đúng trong top 3 image items.

```text
source_mrr_at_3
```

Giá trị tốt hơn giữa text lane và image lane.

## Diagnostic Metrics

```text
text_evidence_hit_at_k
```

Hit nếu text item:

```text
URL trùng gold URL
hoặc token coverage >= 0.60
hoặc token F1 >= 0.45
```

```text
image_exact_hit_at_k
```

Hit nếu image item có:

```text
image_path == image_evidence_path
```

```text
evidence_hit_at_k
```

Hit nếu:

```text
text_evidence_hit_at_k == 1
hoặc image_exact_hit_at_k == 1
```

## Thứ Tự Ưu Tiên Chọn Cấu Hình

Ưu tiên chính:

```text
source_hit_at_3
source_mrr_at_3
text_source_hit_at_3
image_source_hit_at_3
evidence_hit_at_3
```

Nếu mục tiêu là đưa đúng source cho LLM judge, metric quan trọng nhất là:

```text
source_hit_at_3
```

Nếu cần ưu tiên source đúng xuất hiện ở rank cao hơn, dùng:

```text
source_mrr_at_3
```

Nếu `image_source_hit_at_3` thấp nhưng `text_source_hit_at_3` cao, phase judging vẫn có thể dùng tốt vì judge chủ yếu cần đúng URL và text evidence.

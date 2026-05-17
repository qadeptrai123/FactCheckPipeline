# Fine-tune CLIP ViT-B/32 for Vietnamese Text-to-Image Retrieval

This folder contains a simple SentenceTransformers fine-tuning pipeline for improving Vietnamese text -> image retrieval.

Main files:

- `clip_vit_b32_vietnamese_retrieval.ipynb`: training notebook
- `clip_vit_b32_vietnamese_retrieval_config.json`: editable training config

## Goal

The v1 multimodal database uses `sentence-transformers/clip-ViT-B-32` for 512-dimensional image embeddings. This pipeline continues fine-tuning that same checkpoint so Vietnamese text queries are better aligned with image embeddings.

The notebook does not include Qdrant indexing. It only trains and exports a checkpoint. You can integrate the exported model into your own Qdrant upsert/search flow.

## Dataset

Default dataset:

```text
ai-enthusiasm-community/UIT-ViIC
```

UIT-ViIC is a Vietnamese image captioning dataset. Each row contains an image and multiple Vietnamese captions. The notebook uses all available captions and flattens them into positive pairs:

```text
(Vietnamese caption, image)
```

Example training meaning:

```text
"Một cầu thủ đang đá bóng trên sân" -> matching image
```

## Model

Base model:

```text
sentence-transformers/clip-ViT-B-32
```

This is intentionally the same model family used in the existing v1 pipeline. The exported checkpoint remains a SentenceTransformers model and can be loaded with:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("finetune/clip-vit-b32-st-vi-retrieval/best")
```

## Training Objective

The notebook uses:

```python
losses.MultipleNegativesRankingLoss(model)
```

For each batch:

- the caption and its image are treated as a positive pair
- other images in the batch are treated as negatives
- the model is trained to give the paired image a higher similarity score than non-paired images

This is the standard bi-encoder retrieval objective for improving embedding search.

The trainer also uses:

```python
BatchSamplers.NO_DUPLICATES
```

This reduces the risk that duplicated captions/images in the same batch become false negatives.

## Evaluation

The notebook evaluates text -> image retrieval using recall:

- `recall@1`
- `recall@5`
- `recall@10`

Evaluation flow:

1. Encode Vietnamese captions as query vectors.
2. Encode unique images as image vectors.
3. Compute cosine similarity with normalized embeddings.
4. Check whether the correct image appears in the top K results.

The main checkpoint selection metric is:

```text
recall@10
```

## Training Flow

1. Load config from `clip_vit_b32_vietnamese_retrieval_config.json`.
2. Load UIT-ViIC from Hugging Face.
3. Flatten image rows into `(text, image)` pairs.
4. Load `sentence-transformers/clip-ViT-B-32`.
5. Compute baseline text-to-image recall before training.
6. Wrap the custom text-to-image recall logic in a `SentenceEvaluator`.
7. Train with `eval_strategy="epoch"` and `save_strategy="epoch"`.
8. Let `SentenceTransformerTrainer` select the best checkpoint with `load_best_model_at_end=True`.
9. Export the loaded best model to `best/`.
10. Write metrics to `finetune/logs/experiments.md`.

## Output

After running the notebook:

```text
finetune/clip-vit-b32-st-vi-retrieval/
  best/          # exported best checkpoint by recall@10
  checkpoints/   # trainer-managed checkpoints
```

Use `best/` for integration.

## Config

Important fields:

```json
{
  "base_model_name": "sentence-transformers/clip-ViT-B-32",
  "num_train_epochs": 3,
  "learning_rate": 0.00002,
  "per_device_train_batch_size": 32,
  "per_device_eval_batch_size": 64,
  "smoke_test": true
}
```

Set this before a real run:

```json
"smoke_test": false
```

`smoke_test=true` is only for checking that loading, training, evaluation, and export work. It uses a tiny slice and one training step.

## Using the Best Checkpoint

Text query embedding:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("finetune/clip-vit-b32-st-vi-retrieval/best")

query_vec = model.encode(
    ["một cầu thủ đang đá bóng trên sân"],
    normalize_embeddings=True,
    convert_to_numpy=True,
)[0]
```

Image embedding:

```python
from PIL import Image

image = Image.open("path/to/image.jpg").convert("RGB")

image_vec = model.encode(
    [image],
    normalize_embeddings=True,
    convert_to_numpy=True,
)[0]
```

Both vectors should be 512-dimensional. Use cosine distance in your vector database.

## Practical Notes

- Fine-tuning changes the embedding space. Re-embed your image corpus with the exported checkpoint before comparing against text query vectors from the same checkpoint.
- Do not mix vectors from the old base model with vectors from the fine-tuned model in the same collection.
- If training quality drops, reduce `learning_rate` to `0.000005` or train fewer epochs.
- Larger batch sizes usually help `MultipleNegativesRankingLoss` because each batch contains more in-batch negatives.
- GPU is strongly recommended. CPU is only practical for smoke tests.

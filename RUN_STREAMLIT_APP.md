# Run The FactCheck Streamlit App

## 1. Start Qdrant

Make sure Qdrant is running on:

```text
http://127.0.0.1:6333
```

Check that these collections exist:

```text
fixed_size
semantic
```

You can verify with:

```powershell
Invoke-RestMethod http://127.0.0.1:6333/collections
```

## 2. Check Environment

Run from the project root:

```powershell
cd D:\FactCheckPipeline
```

Make sure `.env` contains:

```text
OPENROUTER_API_KEY=your_key_here
```

The backend expects CUDA for local retrieval models. You can check CUDA with:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"
```

## 3. Start FastAPI Backend

Open a terminal and run:

```powershell
cd D:\FactCheckPipeline\src\backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Check backend health:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/
```

Check runtime diagnostics:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/pipeline/debug/runtime
```

Confirm that:

```text
cuda_available: true
device_name: NVIDIA GeForce RTX 4060 Laptop GPU
```

## 4. Start Streamlit Frontend

Open a second terminal and run:

```powershell
cd D:\FactCheckPipeline
streamlit run src/frontend/app.py --server.address 127.0.0.1 --server.port 8501
```

Open the app in your browser:

```text
http://127.0.0.1:8501
```

## 5. Test One Claim

Enter a claim, for example:

```text
Công an tỉnh Thái Bình đã khởi tố 10 đối tượng trong đường dây lừa đảo hỗ trợ vay vốn online.
```

Image upload is optional.

Click:

```text
Run fact check
```

The app will run:

```text
refined -> database RAG -> judge
```

The frontend polls the backend until the job status is `done`.

## 6. Expected First Run Behavior

The first run can be slow because the backend loads local models onto GPU:

```text
bkai-foundation-models/vietnamese-bi-encoder
clip-vit-b32-finetuned-final-final
namdp-ptit/ViRanker
```

Later runs should be faster because the backend keeps the model components loaded.

## 7. If The App Fails

Check FastAPI terminal logs first.

Then check:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/v1/pipeline/debug/runtime
```

Common problems:

```text
Qdrant is not running
OPENROUTER_API_KEY is missing
CUDA is not available to Python
Local model path is missing
Corpus file is missing at chunking_scripts/final_corpus.csv
```


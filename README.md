# FastAPI Clean Architecture ASR

Minimal FastAPI base with clean architecture and a Whisper speech-to-text endpoint, plus RAG-backed answers with citations.

## Requirements

- Python 3.10+
- `ffmpeg` installed and available on PATH
- Supabase (vector store) configured for RAG

## Install

```bash
pip install -r requirement.txt
```

## Run API

```bash
uvicorn app.main:app --reload
```

Open Swagger UI: `http://127.0.0.1:8000/docs`

## Environment

```bash
# LLM provider: gemini (RAG currently uses Gemini)
setx LLM_PROVIDER gemini

# Gemini
setx GEMINI_API_KEY your_key_here
setx GEMINI_MODEL_NAME gemini-1.5-flash

# ASR runtime (Whisper)
setx ASR_MODEL_NAME base
setx ASR_DEVICE cuda
setx ASR_USE_FP16 true
setx ASR_LANGUAGE vi
setx ASR_INITIAL_PROMPT "Day la hoi thoai tieng Viet ve tai chinh"

# Local Mistral (non-RAG)
setx LOCAL_MODEL_NAME mistralai/Mistral-7B-Instruct-v0.2
setx LOCAL_MAX_NEW_TOKENS 256

# Emotion detection (HuggingFace pipeline)
setx EMOTION_ENABLED true
setx EMOTION_MODEL_NAME j-hartmann/emotion-english-distilroberta-base
setx EMOTION_DEVICE auto
```

## Example request

```bash
curl -X POST "http://127.0.0.1:8000/asr/transcribe" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@data/docs/sample.wav"
```

## LLM (RAG) example

`/llm/answer` now uses RetrievalQA + Supabase and returns citations.

```bash
curl -X POST "http://127.0.0.1:8000/llm/answer" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"Toi nen lam gi voi khoan tiet kiem ngan han?\"}"
```

Example response:

```json
{
  "answer": "...",
  "citations": [{ "source": "financebench", "doc_id": "...", "chunk_index": 2 }]
}
```

## CLI (tiny runner)

```bash
python scripts/asr_cli.py --audio data/docs/sample.wav --model base
```

Force GPU + FP16 from CLI:

```bash
python scripts/asr_cli.py --audio data/docs/sample.wav --model base --device cuda --fp16
```

Prioritize Vietnamese from CLI:

```bash
python scripts/asr_cli.py --audio data/docs/sample.wav --model base --language vi --initial-prompt "Day la hoi thoai tieng Viet ve tai chinh"
```

## Chat voice streaming (Mic -> WebSocket -> Streaming ASR)

WebSocket endpoint: `ws://127.0.0.1:8000/chat/voice/ws`

Protocol:

- Client sends JSON `{"event":"start","filename":"mic.wav"}`
- Client streams microphone audio as **binary WebSocket frames**
- Client sends JSON `{"event":"end"}` to finalize
- Server emits:
  - `ready`, `started`
  - `partial` (incremental transcript)
  - `final` (transcript + RAG answer + citations)
  - `error`

Minimal browser sample:

```javascript
const ws = new WebSocket("ws://127.0.0.1:8000/chat/voice/ws");

ws.onopen = () => {
  ws.send(JSON.stringify({ event: "start", filename: "mic.wav" }));
};

ws.onmessage = (evt) => {
  const msg = JSON.parse(evt.data);
  console.log(msg.event, msg);
};

// Send microphone chunks as ArrayBuffer / Uint8Array:
// ws.send(audioChunk);

// When user stops recording:
// ws.send(JSON.stringify({ event: "end" }));
```

Final event payload:

```json
{
  "text": "...",
  "answer": "...",
  "emotion": {
    "label": "fear",
    "score": 0.92,
    "predictions": [
      { "label": "fear", "score": 0.92 },
      { "label": "sadness", "score": 0.05 }
    ]
  },
  "audio_base64": "<base64-mp3-or-null>",
  "audio_mime_type": "audio/mpeg",
  "audio_filename": "voice-response.mp3",
  "citations": [
    { "source": "vietnam-stock", "doc_id": "...", "chunk_index": 5 }
  ]
}
```

`/chat/voice` HTTP response also includes the same optional `emotion` and audio fields.

## Emotion Detection

Emotion detection is available via HuggingFace `pipeline("text-classification")`.

Endpoint: `POST /emotion/detect`

```bash
curl -X POST "http://127.0.0.1:8000/emotion/detect" \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"Toi rat lo lang ve bien dong thi truong hom nay\",\"top_k\":3}"
```

Example response:

```json
{
  "text": "Toi rat lo lang ve bien dong thi truong hom nay",
  "label": "fear",
  "score": 0.924,
  "predictions": [
    { "label": "fear", "score": 0.924 },
    { "label": "sadness", "score": 0.052 },
    { "label": "neutral", "score": 0.024 }
  ]
}
```

Tiny runner:

```bash
python scripts/emotion_cli.py --text "I am excited about this quarter" --top-k 3
```

### Emotion local training workflow

1. Baseline current model and save report:

```powershell
python scripts/evaluate_emotion_datasets.py --limit 500 --report-path reports/emotion/baseline.json
```

2. Prepare train/val/test splits in local workspace:

```powershell
python scripts/prepare_emotion_dataset.py --limit 8000
```

Full Vietnamese UIT-VSMEC:

```powershell
python scripts/prepare_emotion_dataset.py --source uit-vsmec --limit 0 --raw-output data/emotion/raw/uit-vsmec.jsonl --processed-output data/emotion/processed/uit-vsmec.cleaned.jsonl --splits-dir data/emotion/splits/uit-vsmec --profile-output reports/emotion/data_profile_uit_vsmec.json
```

3. Fine-tune local model checkpoint:

```powershell
python scripts/train_emotion_model.py --model distilbert-base-multilingual-cased --epochs 3 --batch-size 16
```

4. Evaluate fine-tuned model:

```powershell
python scripts/evaluate_emotion_model.py --model models/emotion/distilbert-v1/best --output-json reports/emotion/finetuned.json
```

5. Point API to your local checkpoint:

```powershell
setx EMOTION_ENABLED true
setx EMOTION_MODEL_NAME "models/emotion/distilbert-v1/best"
setx EMOTION_DEVICE auto
```

After changing env vars with `setx`, restart terminal before running API.

## Datasets (vector database sources)

### Install deps

```powershell
pip install -r requirement.txt
```

### Hugging Face: PatronusAI/financebench

```powershell
# Optional: set HF token if the dataset requires auth
setx HF_TOKEN your_hf_token_here
```

```python
from datasets import load_dataset

ds = load_dataset("PatronusAI/financebench")
print(ds)
```

### KaggleHub: vietnamese-stock-market-data

```powershell
# Kaggle auth (use either env vars or kaggle.json)
setx KAGGLE_USERNAME your_kaggle_username
setx KAGGLE_KEY your_kaggle_api_key
```

```python
import kagglehub

path = kagglehub.dataset_download("ngwkhai/vietnamese-stock-market-data")
print("Path to dataset files:", path)
```

### Notes

- Dataset cache locations are managed by Hugging Face and KaggleHub; keep `data/` for your processed artifacts.
- If you use `setx`, restart the terminal so env vars are applied.

### Emotion datasets (Kaggle baseline)

Evaluate your selected model on Kaggle Emotion dataset:

```powershell
python scripts/evaluate_emotion_datasets.py --limit 500
```

Optional custom dataset handles (if your Kaggle source differs):

```powershell
python scripts/evaluate_emotion_datasets.py --kaggle-emotion-handle "<owner>/<dataset>"
```

## Supabase vector store

Create a table and RPC for vector similarity before ingesting chunks.

```sql
create extension if not exists vector;

create table if not exists public.document_chunks (
  id uuid primary key default gen_random_uuid(),
  content text not null,
  embedding vector(384) not null,
  metadata jsonb not null default '{}'::jsonb
);

create or replace function match_documents(
  query_embedding vector(384),
  match_count int
)
returns table (
  id uuid,
  content text,
  embedding vector(384),
  metadata jsonb,
  similarity float
)
language plpgsql
as $$
begin
  return query
  select
    document_chunks.id,
    document_chunks.content,
    document_chunks.embedding,
    document_chunks.metadata,
    1 - (document_chunks.embedding <=> query_embedding) as similarity
  from document_chunks
  order by document_chunks.embedding <=> query_embedding
  limit match_count;
end;
$$;
```

Set env vars in `.env` (or with `setx`).

```powershell
setx SUPABASE_URL "https://your-project.supabase.co"
setx SUPABASE_SERVICE_KEY "your_service_role_key"
setx SUPABASE_TABLE "document_chunks"
setx SUPABASE_MATCH_RPC "match_documents"
setx EMBEDDING_MODEL_NAME "sentence-transformers/all-MiniLM-L6-v2"
setx CHUNK_SIZE 300
setx CHUNK_OVERLAP 50
setx RAG_TOP_K 4
```

## Ingest datasets into Supabase

```powershell
python scripts/ingest_datasets.py --source financebench --limit 200
python scripts/ingest_datasets.py --source vietnam-stock --limit 200
python scripts/ingest_datasets.py --source vifid-segraw-200k --limit 200
```

Notes:

- The SQL above uses vector size 384 (MiniLM); adjust if you change `EMBEDDING_MODEL_NAME`.
- Use the service role key for ingestion so inserts are allowed.
- For `vifid-segraw-200k`, install Kaggle adapter extras if needed: `pip install kagglehub[pandas-datasets]`.

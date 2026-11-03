# ContextLab — Real RAG vs No RAG

A two-assistant university demo. Both assistants use the **same OpenAI model**. The first receives only the question; the second receives retrieved PDF passages. Responses are real API outputs, not hard-coded.

## Run locally
1. Install Python 3.10+.
2. `pip install -r requirements.txt`
3. Set `OPENAI_API_KEY` as an environment variable (never commit it).
4. `uvicorn app:app --reload`
5. Open http://127.0.0.1:8000 and upload a **text-based PDF** before asking a question.

## Architecture
- FastAPI serves the existing HTML UI and `/api/*` endpoints.
- PDF extraction via PyMuPDF, chunking with overlap.
- Local TF-IDF retrieval ranks PDF passages; this is a **lexical retriever**, not a vector embedding database.
- Both responses use the same model and temperature. RAG includes top-3 passages and source page labels.
- Uploaded content stays in server memory until restarted or replaced; no permanent file storage.
- This is a teaching prototype, not a production system. It has no authentication, rate limiting or per-user isolation. **Do not deploy it publicly as-is** and do not upload confidential documents.

## Deployment
GitHub Pages can host only the static simulation frontend, **not** the Python API. For real AI, run FastAPI locally or deploy the whole application to a Python-capable host with `OPENAI_API_KEY` configured server-side. Never expose the API key in JavaScript.

## Fairness and limitations
The baseline model may correctly answer or abstain. RAG may also fail if retrieval misses evidence. The comparison demonstrates grounding, not a guarantee of superiority. Only use source documents you are authorized to upload.

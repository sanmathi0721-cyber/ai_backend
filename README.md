# AI PDF Summarizer (FastAPI + OpenAI) — Simple version for final-year demo

## What this is
A minimal FastAPI backend that accepts a PDF, extracts text (PyMuPDF), summarizes each chunk via OpenAI Responses API, and returns a final cohesive summary.

## Files
- `main.py` — the FastAPI app (single file)
- `requirements.txt`
- `.env.example`

## Run locally
1. Clone repo.
2. Create virtualenv and install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt

# main.py
import os
import tempfile
import logging
from typing import List
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from dotenv import load_dotenv
from openai import OpenAI
import time

# Load .env locally (optional)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf-summarizer")

# Initialize OpenAI client (reads OPENAI_API_KEY from env)
# Make sure you set OPENAI_API_KEY in Render or locally before running.
client = OpenAI()

app = FastAPI(title="Simple PDF Summarizer (FastAPI + OpenAI)")

# Allow all origins for demo; tighten this for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------- utils --------
def extract_text_from_pdf(path: str) -> str:
    """Extract text from all pages of a PDF (returns a single string)."""
    try:
        doc = fitz.open(path)
    except Exception as e:
        logger.exception("Failed to open PDF")
        raise

    parts = []
    for page in doc:
        try:
            txt = page.get_text("text")
            if txt:
                parts.append(txt)
        except Exception:
            # don't fail entire extraction on one page
            continue
    doc.close()
    return "\n\n".join(parts).strip()

def chunk_text(text: str, max_chars: int = 3000, overlap: int = 200) -> List[str]:
    """
    Split `text` into overlapping chunks of up to max_chars characters.
    Overlap helps keep context between chunks.
    """
    if not text:
        return []
    n = len(text)
    chunks = []
    start = 0
    while start < n:
        end = start + max_chars
        if end >= n:
            chunk = text[start:n].strip()
            chunks.append(chunk)
            break
        # Try to cut at last newline/space to avoid splitting in middle of words
        cut = text.rfind("\n", start, end)
        if cut <= start:
            cut = text.rfind(" ", start, end)
        if cut <= start:
            cut = end  # fallback
        chunk = text[start:cut].strip()
        chunks.append(chunk)
        start = max(cut - overlap, cut - overlap if cut - overlap > 0 else cut)
    return chunks

def parse_response_text(resp) -> str:
    """
    Robust parser for Responses API outputs. Returns plain text summary.
    """
    try:
        # Newer wrapper may expose 'output_text'
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text

        # Else try 'output' contents
        if hasattr(resp, "output") and resp.output:
            parts = []
            for item in resp.output:
                # item may be dict-like or object-like; handle both
                content = item.get("content", []) if isinstance(item, dict) else getattr(item, "content", [])
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        parts.append(c.get("text", ""))
                    elif isinstance(c, str):
                        parts.append(c)
            if parts:
                return "\n\n".join(parts)

        # fallback
        return str(resp)
    except Exception as e:
        logger.exception("Failed parsing response")
        return f"(parse error) {e}"

def summarize_chunk(chunk: str, model: str = "gpt-4o-mini", max_tokens: int = 400) -> str:
    """
    Summarize a single chunk using OpenAI Responses API.
    """
    prompt = (
        "You are an expert summarizer. Summarize the following text in 4-6 concise sentences, focusing on "
        "the main points, and avoid hallucination. If the text doesn't contain enough info, say so.\n\n"
        f"Text:\n{chunk}\n\nSummary:"
    )
    try:
        resp = client.responses.create(model=model, input=prompt, max_tokens=max_tokens)
        return parse_response_text(resp).strip()
    except Exception as e:
        logger.exception("OpenAI chunk summarization failed")
        return f"(error summarizing chunk: {e})"

def combine_summaries(summaries: List[str], model: str = "gpt-4o-mini", max_tokens: int = 512) -> str:
    """
    Combine multiple chunk summaries into a single cohesive summary & TL;DR.
    """
    joined = "\n\n---\n\n".join(summaries)
    prompt = (
        "You are an assistant that produces a clean final summary. Given the chunk summaries below, "
        "write a single cohesive summary of the whole document in 6-8 sentences and then provide a 3-bullet TL;DR.\n\n"
        f"Chunk summaries:\n{joined}\n\nFinal Summary and TL;DR:"
    )
    try:
        resp = client.responses.create(model=model, input=prompt, max_tokens=max_tokens)
        return parse_response_text(resp).strip()
    except Exception as e:
        logger.exception("OpenAI combine summaries failed")
        return f"(error combining summaries: {e})"

# -------- endpoints --------
@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <html><body>
    <h2>Simple PDF Summarizer</h2>
    <p>Use the /docs UI or POST /summarize-pdf (multipart form) to upload a PDF and get a summary.</p>
    </body></html>
    """

@app.post("/summarize-pdf")
async def summarize_pdf(file: UploadFile = File(...)):
    """
    Upload a PDF file. The endpoint will:
      - extract text,
      - split into chunks,
      - summarize each chunk,
      - combine chunk summaries into the final summary.
    Returns JSON: { "summary": "...", "chunk_count": N, "chunk_summaries": [...] }
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()

        text = extract_text_from_pdf(tmp.name)
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    if not text or text.strip() == "":
        raise HTTPException(status_code=400, detail="No extractable text found in the PDF. The file may be scanned images only (OCR required).")

    # Chunk text to avoid sending too large prompts
    chunks = chunk_text(text, max_chars=3000, overlap=300)
    logger.info("PDF text length=%d chars, chunk_count=%d", len(text), len(chunks))

    # Summarize each chunk sequentially (keeps memory low)
    chunk_summaries = []
    start_t = time.time()
    for i, chunk in enumerate(chunks):
        logger.info("Summarizing chunk %d/%d", i + 1, len(chunks))
        s = summarize_chunk(chunk)
        chunk_summaries.append(s)

    # Combine chunk summaries
    final_summary = combine_summaries(chunk_summaries)

    elapsed = time.time() - start_t
    return JSONResponse({
        "summary": final_summary,
        "chunk_count": len(chunks),
        "chunk_summaries": chunk_summaries,
        "elapsed_seconds": round(elapsed, 2)
    })

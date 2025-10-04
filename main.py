import os
import tempfile
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize OpenAI client with API key from environment
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="Simple AI PDF Summarizer")

# Enable CORS so frontend can call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_text_from_pdf(path: str) -> str:
    """Extract text from all pages of a PDF file."""
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    doc.close()
    return text.strip()


@app.post("/summarize-pdf")
async def summarize_pdf(file: UploadFile = File(...)):
    """Upload a PDF and return a summary using OpenAI."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Save file temporarily
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(await file.read())
    tmp.close()

    # Extract text
    text = extract_text_from_pdf(tmp.name)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    # Build prompt (truncate if too large)
    prompt = f"Summarize the following PDF content in 5-6 sentences:\n\n{text[:5000]}"

    try:
        # ✅ Fixed: use max_output_tokens (not max_tokens)
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=300,
        )

        # Extract output text
        summary = ""
        if hasattr(response, "output") and response.output:
            for item in response.output:
                for c in item.get("content", []):
                    if isinstance(c, dict) and c.get("type") == "output_text":
                        summary += c.get("text", "")
        if not summary:
            summary = getattr(response, "output_text", "No summary generated.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

    return {"summary": summary}

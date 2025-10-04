import os
import tempfile
import fitz  # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(title="AI PDF Summarizer")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def extract_text_from_pdf(path: str) -> str:
    """Extract plain text from all PDF pages."""
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    doc.close()
    return text.strip()


@app.post("/summarize-pdf")
async def summarize_pdf(file: UploadFile = File(...)):
    """Upload a PDF, extract text, and summarize with OpenAI."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    # Save PDF temporarily
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp.write(await file.read())
    tmp.close()

    # Extract text from PDF
    text = extract_text_from_pdf(tmp.name)
    if not text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF")

    # Truncate large PDFs
    content = text[:5000]

    # Prompt for summarization
    prompt = f"Summarize the following PDF content in 5-6 clear sentences:\n\n{content}"

    try:
        # ✅ Correct usage of Responses API
        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt,
            max_output_tokens=300,
        )

        # Easiest way: directly get the summary
        summary = response.output_text

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI error: {e}")

    return {"summary": summary}

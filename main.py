# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from openai import OpenAI
import io
import os

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Allow all origins (for Render + GitHub Pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can replace * with your GitHub Pages URL for extra security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Setup OpenAI client with key from environment
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_KEY:
    raise ValueError("⚠️ OPENAI_API_KEY not found in environment variables.")
client = OpenAI(api_key=OPENAI_KEY)

@app.get("/")
async def home():
    return {"message": "✅ AI Summarizer backend running successfully"}

# ✅ Extract text from uploaded PDF
def extract_text_from_pdf(file_bytes: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text
    except Exception as e:
        return f"Error reading PDF: {e}"

# ✅ Summarize route
@app.post("/summarize")
async def summarize_pdf(file: UploadFile = File(...)):
    try:
        # Read and extract text
        pdf_bytes = await file.read()
        content = extract_text_from_pdf(pdf_bytes)

        if not content.strip():
            return {"summary": "⚠️ The uploaded PDF seems empty or could not be read."}

        # ✅ AI summarization using OpenAI API
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"Summarize this PDF in 200-250 words clearly and concisely:\n\n{content[:6000]}"
        )

        # ✅ Extract summary safely
        summary = (
            response.output[0].content[0].text.strip()
            if response.output and response.output[0].content
            else "No summary generated."
        )

        return {"summary": summary}

    except Exception as e:
        return {"error": f"Internal Server Error: {str(e)}"}

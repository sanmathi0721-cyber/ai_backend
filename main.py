# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PyPDF2 import PdfReader
from openai import OpenAI
import io
import os

app = FastAPI()

# ✅ CORS setup — allows frontend from GitHub Pages
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or specify your frontend URL for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
async def home():
    return {"message": "AI Summarizer backend is running 🚀"}

# ✅ Extract text from PDF safely
def extract_text_from_pdf(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# ✅ Summarization endpoint
@app.post("/summarize")
async def summarize_pdf(file: UploadFile = File(...)):
    try:
        pdf_bytes = await file.read()
        content = extract_text_from_pdf(pdf_bytes)

        if not content.strip():
            return {"summary": "The PDF appears to be empty or unreadable."}

        # ✨ Use GPT-4 Mini (lightweight and accurate)
        response = client.responses.create(
            model="gpt-4.1-mini",
            input=f"Summarize this PDF content in about 200 words:\n\n{content[:6000]}"
        )

        summary = response.output[0].content[0].text.strip()
        return {"summary": summary}

    except Exception as e:
        return {"error": str(e)}

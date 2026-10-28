
import os
import base64
import mimetypes

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import types

app = FastAPI(title="TaniCam AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "online", "app": "TaniCam AI"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/analyze")
async def analyze(image: UploadFile = File(...)):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="API key Gemini belum dipasang di Render."
        )

    mime_type = image.content_type or ""
    if mime_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(
            status_code=400,
            detail="Gunakan foto JPG, PNG, atau WEBP."
        )

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Foto kosong.")

    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Ukuran foto maksimal 10 MB."
        )

    prompt = """
Kamu adalah asisten analisis kesehatan tanaman untuk petani Indonesia.
Analisis foto dan jawab dalam bahasa Indonesia sederhana dengan format:
1. Jenis tanaman yang terlihat
2. Gejala yang benar-benar terlihat
3. Dugaan penyakit atau hama, jika ada
4. Tingkat keyakinan: rendah, sedang, atau tinggi
5. Langkah penanganan awal yang aman dan praktis
6. Cara mencegah masalah berulang

Jangan memastikan diagnosis jika bukti foto tidak cukup.
Jangan mengarang gejala. Jika foto tidak jelas atau bukan tanaman,
jelaskan hal itu. Untuk masalah serius, sarankan konsultasi dengan
penyuluh pertanian setempat.
"""

    try:
        client = genai.Client(api_key=api_key)

        interaction = client.interactions.create(
            model="gemini-3.8-flash",
            input=[
                {"type": "text", "text": prompt},
                {
                    "type": "image",
                    "data": base64.b64encode(data).decode("utf-8"),
                    "mime_type": mime_type,
                },
            ],
            store=False,
        )

        analysis = interaction.output_text

        if not analysis:
            raise HTTPException(
                status_code=502,
                detail="Gemini belum memberikan hasil. Coba lagi."
            )

        return {
            "success": True,
            "filename": image.filename,
            "analysis": analysis,
        }

    except HTTPException:
        raise
    except Exception as error:
        print("Gemini analysis error:", type(error).__name__, str(error))
        raise HTTPException(
            status_code=502,
            detail="Analisis gagal. Periksa Runtime Logs di Render."
        )

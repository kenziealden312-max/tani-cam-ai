import os
import json
import urllib.request
import urllib.error

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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
            detail="API key Gemini belum dipasang di server."
        )

    mime_type = image.content_type or ""

    if mime_type not in [
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/heic",
        "image/heif",
    ]:
        raise HTTPException(
            status_code=400,
            detail="Gunakan foto JPG, PNG, atau WEBP."
        )

    data = await image.read()

    if not data:
        raise HTTPException(
            status_code=400,
            detail="Foto kosong. Silakan pilih foto lagi."
        )

    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Ukuran foto maksimal 10 MB."
        )

    import base64

    prompt = """
    Kamu adalah asisten analisis kesehatan tanaman untuk petani Indonesia.
    Periksa foto tanaman yang diberikan dan jawab dalam bahasa Indonesia
    yang sederhana.

    Berikan hasil dengan format:
    1. Tanaman yang terlihat
    2. Kondisi yang terlihat pada foto
    3. Dugaan penyakit atau hama, jika ada
    4. Tingkat keyakinan: rendah, sedang, atau tinggi
    5. Saran penanganan awal yang aman dan praktis
    6. Cara mencegah masalah berulang

    Jangan memastikan diagnosis jika bukti visual tidak cukup.
    Jika foto tidak menunjukkan tanaman dengan jelas, katakan demikian.
    Jangan mengarang gejala. Utamakan langkah yang aman dan sesuai
    untuk petani. Jika perlu, sarankan pemeriksaan oleh penyuluh pertanian.
    """

    payload = {
        "model": "gemini-3.5-flash-lite",
        "input": [
            {"type": "text", "text": prompt},
            {
                "type": "image",
                "data": base64.b64encode(data).decode("utf-8"),
                "mime_type": mime_type,
            },
        ],
    }

    request = urllib.request.Request(
        "https://generativelanguage.googleapis.com/v1beta/interactions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            result = json.loads(response.read().decode("utf-8"))

        analysis = result.get("output_text")

        if not analysis:
            for item in result.get("outputs", []):
                if item.get("type") == "text":
                    analysis = item.get("text")
                    break

        if not analysis:
            raise HTTPException(
                status_code=502,
                detail="AI belum memberikan hasil. Coba lagi."
            )

        return {
            "success": True,
            "filename": image.filename,
            "analysis": analysis,
        }

    except urllib.error.HTTPError as error:
        print("Gemini API HTTP error:", error.code)
        raise HTTPException(
            status_code=502,
            detail="Gemini gagal menganalisis foto. Periksa konfigurasi API."
        )
    except urllib.error.URLError:
        raise HTTPException(
            status_code=502,
            detail="Server tidak dapat terhubung ke layanan Gemini."
)

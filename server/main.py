import os
import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai

app = FastAPI(title="TaniCam AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
# Render's filesystem may reset on redeploy. Set DATA_DIR to a mounted persistent
# disk path in Render if you want scan history to survive redeploys.
DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data"))
HISTORY_FILE = DATA_DIR / "tanicam_history.json"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}


@app.get("/")
def home():
    return {"status": "online", "app": "TaniCam AI"}


@app.get("/health")
def health():
    return {"status": "ok"}


def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=500,
            detail="API key Gemini belum dipasang di Render.",
        )
    return genai.Client(api_key=api_key)


def load_history():
    try:
        if HISTORY_FILE.exists():
            raw = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            return raw if isinstance(raw, list) else []
    except Exception as error:
        print("History read error:", type(error).__name__, str(error))
    return []


def save_history(items):
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(
            json.dumps(items, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception as error:
        print("History save error:", type(error).__name__, str(error))
        raise HTTPException(
            status_code=500,
            detail="Riwayat gagal disimpan. Periksa penyimpanan Render.",
        )


def output_text(interaction):
    return getattr(interaction, "output_text", None)


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    plant: str = Form(default="Tanaman tidak diketahui"),
    planted_date: str = Form(default=""),
    notes: str = Form(default=""),
):
    mime_type = image.content_type or ""
    if mime_type not in ALLOWED_MIME:
        raise HTTPException(status_code=400, detail="Gunakan foto JPG, PNG, atau WEBP.")

    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Foto kosong.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Ukuran foto maksimal 10 MB.")

    prompt = f"""
Kamu adalah asisten budidaya tanaman untuk petani Indonesia.
Tanaman yang dipilih petani: {plant}
Tanggal tanam (jika diketahui): {planted_date or "tidak diketahui"}
Catatan petani: {notes or "tidak ada"}

Analisis foto secara hati-hati. Jawab bahasa Indonesia sederhana dan ringkas
(maksimal sekitar 180 kata), dengan bagian:
1. Kondisi yang terlihat
2. Dugaan masalah (jelaskan tingkat kepastian; jangan menyatakan diagnosis pasti dari foto)
3. Air: tanda kemungkinan kekurangan/kelebihan air dan cara memeriksa tanah
4. Nutrisi/pupuk: hanya rekomendasi umum yang sesuai; jangan mengarang dosis
5. Langkah berikut dan kapan perlu foto ulang

Foto tidak dapat memastikan kelembapan, pH, atau unsur hara tanah. Jika tidak terlihat,
katakan perlu pemeriksaan langsung. Jangan menyarankan pestisida spesifik tanpa bukti.
"""

    try:
        client = get_client()
        interaction = client.interactions.create(
            model=MODEL,
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
        analysis = output_text(interaction)
        if not analysis:
            raise HTTPException(status_code=502, detail="Gemini belum memberikan hasil. Coba lagi.")

        record = {
            "id": str(uuid4()),
            "date": datetime.now(timezone.utc).isoformat(),
            "filename": image.filename or "foto",
            "plant": plant,
            "planted_date": planted_date,
            "notes": notes,
            "analysis": analysis,
        }
        history = load_history()
        history.insert(0, record)
        save_history(history[:200])

        return {"success": True, **record}
    except HTTPException:
        raise
    except Exception as error:
        print("Gemini analysis error:", type(error).__name__, str(error))
        raise HTTPException(status_code=502, detail="Analisis gagal. Periksa Runtime Logs di Render.")


@app.get("/history")
def history(limit: int = 30):
    limit = max(1, min(limit, 100))
    return {"success": True, "items": load_history()[:limit]}


@app.post("/chat")
async def chat(
    question: str = Form(...),
    plant: str = Form(default="Kangkung atau cabai"),
    planted_date: str = Form(default=""),
    watering: str = Form(default=""),
    fertilizing: str = Form(default=""),
    soil: str = Form(default=""),
):
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Tulis pertanyaan terlebih dahulu.")
    if len(question) > 2000:
        raise HTTPException(status_code=400, detail="Pertanyaan maksimal 2.000 karakter.")

    recent = load_history()[:5]
    history_context = "\n\n".join(
        f"Scan {item.get('date','')}: tanaman={item.get('plant','')}; "
        f"hasil={item.get('analysis','')}"
        for item in recent
    ) or "Belum ada riwayat scan."

    prompt = f"""
Kamu adalah TaniCam AI, asisten budidaya kangkung dan cabai untuk petani Indonesia.
Jawab dengan bahasa Indonesia sederhana, praktis, dan tidak berlebihan.
Tanaman: {plant}
Tanggal tanam: {planted_date or "belum diketahui"}
Jadwal/riwayat siram: {watering or "belum dicatat"}
Jadwal/riwayat pupuk: {fertilizing or "belum dicatat"}
Keterangan tanah dari petani: {soil or "belum dicatat"}

Riwayat scan terakhir:
{history_context}

Pertanyaan petani:
{question}

Berikan langkah yang bisa dilakukan dan jelaskan apa yang perlu diperiksa langsung.
Jangan mengklaim foto bisa mengukur kadar air, pH, atau kandungan pupuk tanah.
Jangan memberi dosis pupuk atau pestisida spesifik jika informasi tanaman, umur,
produk, dan kondisi belum cukup. Bila perlu, tanyakan data tambahan.
"""

    try:
        client = get_client()
        interaction = client.interactions.create(
            model=MODEL,
            input=prompt,
            store=False,
        )
        answer = output_text(interaction)
        if not answer:
            raise HTTPException(status_code=502, detail="AI belum memberikan jawaban. Coba lagi.")
        return {"success": True, "answer": answer}
    except HTTPException:
        raise
    except Exception as error:
        print("Gemini chat error:", type(error).__name__, str(error))
        raise HTTPException(status_code=502, detail="AI gagal menjawab. Periksa Runtime Logs di Render.")

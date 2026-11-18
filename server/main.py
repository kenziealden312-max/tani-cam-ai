import os, base64, json, asyncio, random
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai

app = FastAPI(title="TaniCam AI")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
DATA_DIR = Path(os.getenv("DATA_DIR", "/var/data"))
HISTORY_FILE = DATA_DIR / "tanicam_history.json"
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}


@app.get("/")
def home():
    return {"status": "online", "app": "TaniCam AI"}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}


def get_client():
    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY belum dipasang di Render.")
    return genai.Client(api_key=key)


def load_history():
    try:
        if HISTORY_FILE.exists():
            data = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))
            return data if isinstance(data, list) else []
    except Exception as e:
        print("History read error:", type(e).__name__, str(e))
    return []


def save_history(items):
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        tmp = HISTORY_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(items[:200], ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(HISTORY_FILE)
    except Exception as e:
        print("History save error:", type(e).__name__, str(e))
        raise HTTPException(status_code=500, detail="Riwayat gagal disimpan. Cek Persistent Disk Render.")


def response_text(result):
    text = getattr(result, "output_text", None)
    return text.strip() if text else ""


def temporary_error(error):
    msg = str(error).lower()
    return any(x in msg for x in ("429", "500", "502", "503", "504", "resource_exhausted",
                                  "rate limit", "temporarily unavailable", "service unavailable",
                                  "timeout", "deadline"))


async def ask_gemini(contents):
    client = get_client()
    last_error = None
    for attempt in range(4):
        try:
            result = await asyncio.to_thread(
                client.interactions.create, model=MODEL, input=contents, store=False
            )
            answer = response_text(result)
            if not answer:
                raise RuntimeError("Gemini mengembalikan jawaban kosong.")
            return answer
        except Exception as e:
            last_error = e
            print(f"Gemini attempt {attempt + 1}/4:", type(e).__name__, str(e))
            if attempt == 3 or not temporary_error(e):
                break
            await asyncio.sleep(min(2 ** attempt, 8) + random.uniform(0, 0.5))

    msg = str(last_error or "").lower()
    if "429" in msg or "resource_exhausted" in msg or "rate limit" in msg:
        detail = "Batas permintaan Gemini sedang tercapai. Tunggu sebentar lalu coba lagi."
    elif any(x in msg for x in ("503", "502", "504", "timeout")):
        detail = "Server Gemini sedang sibuk. Tunggu sebentar lalu coba lagi."
    elif "api key" in msg or "permission" in msg or "unauthorized" in msg:
        detail = "API key Gemini bermasalah. Cek GEMINI_API_KEY di Render."
    else:
        detail = "Gemini gagal memproses. Cek Runtime Logs Render."
    raise HTTPException(status_code=502, detail=detail)


@app.post("/analyze")
async def analyze(
    image: UploadFile = File(...),
    plant: str = Form(default="Identifikasi otomatis dari foto"),
    planted_date: str = Form(default=""),
    notes: str = Form(default=""),
):
    mime = image.content_type or ""
    if mime not in ALLOWED_MIME:
        raise HTTPException(status_code=400, detail="Pakai foto JPG, PNG, atau WEBP.")
    data = await image.read()
    if not data:
        raise HTTPException(status_code=400, detail="Foto kosong.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Ukuran foto maksimal 10 MB.")

    prompt = f"""
Kamu TaniCam AI, asisten pertanian untuk berbagai jenis tanaman di Indonesia,
bukan hanya cabai dan kangkung.
Tanaman yang dipilih pengguna: {plant}
Tanggal tanam: {planted_date or "belum diketahui"}
Catatan petani: {notes or "tidak ada"}

Jika nama tanaman tidak diketahui atau pengguna meminta identifikasi otomatis,
perkirakan jenis tanaman dari foto. Sebutkan ciri visual yang mendukung identifikasi
dan tingkat keyakinan. Jika foto tidak cukup jelas, jangan menebak secara pasti;
minta foto seluruh tanaman, batang, daun, bunga, atau buah.

Analisis tanaman pangan, sayuran, buah, tanaman hias, tanaman obat, dan perkebunan.
Sesuaikan saran dengan tanaman yang teridentifikasi.
Jawab bahasa Indonesia sederhana dan maksimal sekitar 220 kata:
1. Identifikasi tanaman atau beberapa kemungkinan
2. Kondisi yang terlihat
3. Dugaan hama, penyakit, atau gangguan dan tingkat kepastian
4. Saran perawatan: air, cahaya, nutrisi, dan media/tanah bila relevan
5. Langkah berikut dan kapan perlu foto ulang

Foto tidak bisa memastikan kelembapan, pH, atau unsur hara tanah.
Jangan menyebut dugaan sebagai diagnosis pasti. Jangan memberi dosis pupuk
atau pestisida spesifik jika jenis tanaman, umur, dan kondisi belum cukup jelas.
"""
    contents = [
        {"type": "text", "text": prompt},
        {"type": "image", "data": base64.b64encode(data).decode("utf-8"), "mime_type": mime},
    ]
    analysis = await ask_gemini(contents)
    record = {
        "id": str(uuid4()), "date": datetime.now(timezone.utc).isoformat(),
        "filename": image.filename or "foto", "plant": plant,
        "planted_date": planted_date, "notes": notes, "analysis": analysis,
    }
    history = load_history()
    history.insert(0, record)
    save_history(history)
    return {"success": True, **record}


@app.get("/history")
def history(limit: int = 30):
    return {"success": True, "items": load_history()[:max(1, min(limit, 100))]}


@app.post("/chat")
async def chat(
    question: str = Form(...),
    plant: str = Form(default="Berbagai jenis tanaman"),
    planted_date: str = Form(default=""),
    watering: str = Form(default=""),
    fertilizing: str = Form(default=""),
    soil: str = Form(default=""),
):
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Tulis pertanyaan dulu.")
    if len(question) > 2000:
        raise HTTPException(status_code=400, detail="Pertanyaan maksimal 2.000 karakter.")

    recent = load_history()[:5]
    context = "\n\n".join(
        f"Scan {x.get('date', '')}: tanaman={x.get('plant', '')}; hasil={x.get('analysis', '')}"
        for x in recent
    ) or "Belum ada riwayat scan."

    prompt = f"""
Kamu TaniCam AI, asisten pertanian untuk berbagai jenis tanaman di Indonesia.
Kamu bisa membantu tanaman pangan, sayuran, buah, tanaman hias, tanaman obat,
dan perkebunan. Jawab sederhana, praktis, dan sesuaikan saran dengan jenis
serta umur tanaman.
Tanaman: {plant}
Tanggal tanam: {planted_date or "belum diketahui"}
Siram: {watering or "belum dicatat"}
Pupuk: {fertilizing or "belum dicatat"}
Tanah/media: {soil or "belum dicatat"}
Riwayat scan terakhir:
{context}
Pertanyaan petani: {question}

Jika jenis tanaman belum jelas, tanyakan nama atau cirinya sebelum memberi
rekomendasi sangat spesifik. Berikan langkah yang bisa dilakukan dan apa yang
perlu dicek langsung. Jangan mengklaim foto bisa mengukur kadar air, pH, atau
unsur hara. Jangan memberi dosis pupuk/pestisida spesifik jika data belum cukup.
"""
    return {"success": True, "answer": await ask_gemini(prompt)}

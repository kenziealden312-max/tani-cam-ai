
from fastapi import FastAPI, UploadFile, File
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
    data = await image.read()

    if not data:
        return {"success": False, "message": "Foto kosong"}

    return {
        "success": True,
        "message": "Foto berhasil diterima, tetapi model AI tanaman belum dipasang.",
        "filename": image.filename,
        "next_step": "Integrasikan model pengenal penyakit tanaman."
    }

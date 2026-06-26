"""
FastAPI service for Hebrew handwriting generation.

  GET  /                -> the web UI
  GET  /api/info        -> {style_count}
  POST /api/generate    -> {image (base64 png), style, mean_cer}

GPU work is serialized with a lock (single device, models not thread-safe).
Run:  uvicorn app:app --host 0.0.0.0 --port 8000
"""
import base64
import io
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import Generator

app = FastAPI(title="Synth DiffusionPen — Hebrew Handwriting")
_gpu_lock = threading.Lock()
gen: Generator | None = None


@app.on_event("startup")
def _load():
    global gen
    gen = Generator()


class GenReq(BaseModel):
    text: str
    style: int | None = None          # None => random writer
    candidates: int = 5               # best-of-N per word
    aberration: bool = False          # TTA robustness scoring
    normalize: bool = True            # equalize per-character width


@app.get("/api/info")
def info():
    return {"style_count": gen.style_classes if gen else 0}


@app.post("/api/generate")
def generate(req: GenReq):
    if gen is None:
        raise HTTPException(503, "model still loading")
    if not req.text.strip():
        raise HTTPException(400, "text is empty")
    with _gpu_lock:
        try:
            out = gen.generate(text=req.text, style=req.style,
                               candidates=req.candidates, aberration=req.aberration,
                               normalize=req.normalize)
        except Exception as e:
            raise HTTPException(500, f"generation failed: {e}")
    buf = io.BytesIO()
    out["image"].save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"image": "data:image/png;base64," + b64,
            "style": out["style"], "words": out["words"], "mean_cer": out["mean_cer"]}


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")

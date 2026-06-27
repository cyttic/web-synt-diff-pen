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
import re
import threading

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline import Generator

MAX_WORDS = 14
MAX_CANDIDATES = 50
HEBREW_ONLY = re.compile(r"^[֐-׿\s]+$")     # Hebrew block + whitespace
HEBREW_LETTER = re.compile(r"[א-ת]")        # at least one real letter


def validate_text(text: str) -> str | None:
    """Return an error message if the input isn't acceptable, else None."""
    t = (text or "").strip()
    if not t:
        return "Please enter some Hebrew text."
    words = t.split()
    if len(words) > MAX_WORDS:
        return f"Too long: {len(words)} words (max {MAX_WORDS})."
    if not HEBREW_ONLY.match(t):
        return "Hebrew letters only — please remove non-Hebrew characters, digits or punctuation."
    if not HEBREW_LETTER.search(t):
        return "Please enter Hebrew text."
    return None

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
    sampler: str = "dpm"              # "dpm" (fast) or "ddim"


@app.get("/api/info")
def info():
    return {"style_count": gen.style_classes if gen else 0}


@app.post("/api/generate")
def generate(req: GenReq):
    if gen is None:
        raise HTTPException(503, "model still loading")
    msg = validate_text(req.text)
    if msg:
        raise HTTPException(400, msg)
    if not 1 <= req.candidates <= MAX_CANDIDATES:
        raise HTTPException(400, f"Candidates per word must be 1–{MAX_CANDIDATES}.")
    if req.style is not None and not 0 <= req.style < gen.style_classes:
        raise HTTPException(400, f"Writer style must be 0–{gen.style_classes - 1}.")
    if req.sampler not in ("dpm", "ddim", "ddim100"):
        raise HTTPException(400, "Sampler must be 'dpm', 'ddim', or 'ddim100'.")
    with _gpu_lock:
        try:
            out = gen.generate(text=req.text, style=req.style,
                               candidates=req.candidates, aberration=req.aberration,
                               normalize=req.normalize, sampler=req.sampler)
        except Exception as e:
            raise HTTPException(500, f"generation failed: {e}")
    buf = io.BytesIO()
    out["image"].save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"image": "data:image/png;base64," + b64,
            "style": out["style"], "words": out["words"], "mean_cer": out["mean_cer"],
            "sampler": out["sampler"], "steps": out["steps"]}


@app.get("/")
def index():
    return FileResponse("static/index.html")


app.mount("/static", StaticFiles(directory="static"), name="static")

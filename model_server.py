"""
Model inference server — runs on the GPU notebook (where there is a GPU + RAM).

Exposes a tiny API the Azure frontend calls (through the reverse SSH tunnel):
    GET  /health       -> {"status": "ok", "style_count": N}
    GET  /info         -> {"style_count": N}
    POST /generate     -> JSON {text, style?, candidates, aberration, normalize, sampler, vendi}
                          -> {image (base64 png), style, words, mean_nll, sampler, steps, vendi}
    POST /font_baseline -> JSON {text, style, normalize, font_jitter, candidates, vendi}
                          -> {image (base64 png), style, font, jitter, vendi}

GPU work is serialized with a lock (single device, models not thread-safe).

Run (from this directory, so pipeline/generate_styled_sheet resolve):
    uvicorn model_server:app --host 127.0.0.1 --port 8001
"""
import base64
import io
import re
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import font_baseline
from pipeline import Generator, vendi as vendi_score

MAX_WORDS = 14
MAX_CANDIDATES = 50
# Kept in sync with app.py / static/index.html -- see the comment there for why this
# whitelist and the quote fold look the way they do.
PUNCT_ASCII = r"!(),\-./:;?"
ALLOWED_CHARS = re.compile(r"^[\u0590-\u05FF0-9" + PUNCT_ASCII + r"\s]+$")
HEBREW_LETTER = re.compile(r"[א-ת]")        # at least one real letter
QUOTE_FOLD = {
    '"': "\u05f4", "\u201c": "\u05f4", "\u201d": "\u05f4", "\u201e": "\u05f4",
    "'": "\u05f3", "\u2018": "\u05f3", "\u2019": "\u05f3",
    "\u2013": "-", "\u2014": "-",
}


def normalize_text(text: str) -> str:
    for a, b in QUOTE_FOLD.items():
        text = text.replace(a, b)
    return text


def validate_text(text: str) -> str | None:
    t = (text or "").strip()
    if not t:
        return "Please enter some Hebrew text."
    words = t.split()
    if len(words) > MAX_WORDS:
        return f"Too long: {len(words)} words (max {MAX_WORDS})."
    if not ALLOWED_CHARS.match(t):
        return "Hebrew letters, digits and punctuation (! ( ) , - . / : ; ?) only — please remove English letters or other symbols."
    if not HEBREW_LETTER.search(t):
        return "Please enter Hebrew text."
    return None


app = FastAPI(title="Synth DiffusionPen — model server")
_gpu_lock = threading.Lock()
gen: Generator | None = None


@app.on_event("startup")
def _load():
    global gen
    gen = Generator()


class GenReq(BaseModel):
    text: str
    style: int | None = None
    candidates: int = 5
    aberration: bool = False
    normalize: bool = True
    sampler: str = "dpm"
    font_jitter: bool = True   # font_baseline only: per-glyph rotation/scale/baseline jitter on/off
    vendi: bool = False        # score diversity across the `candidates`-size pool (extra compute)


@app.get("/health")
def health():
    return {"status": "ok" if gen else "loading", "style_count": gen.style_classes if gen else 0}


@app.get("/info")
def info():
    return {"style_count": gen.style_classes if gen else 0}


@app.post("/generate")
def generate(req: GenReq):
    if gen is None:
        raise HTTPException(503, "model still loading")
    req.text = normalize_text(req.text)
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
                               normalize=req.normalize, sampler=req.sampler,
                               diversity=req.vendi)
        except Exception as e:
            raise HTTPException(500, f"generation failed: {e}")
    buf = io.BytesIO()
    out["image"].save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"image": "data:image/png;base64," + b64,
            "style": out["style"], "words": out["words"], "mean_nll": out["mean_nll"],
            "sampler": out["sampler"], "steps": out["steps"], "vendi": out["vendi"]}


@app.post("/font_baseline")
def font_baseline_endpoint(req: GenReq):
    """Same deterministic-font comparison render as app.py's /api/font_baseline -- pure
    CPU/PIL, runs fine on the GPU notebook alongside /generate. req.vendi additionally
    embeds through the resident HTR encoder, so that step takes _gpu_lock."""
    req.text = normalize_text(req.text)
    msg = validate_text(req.text)
    if msg:
        raise HTTPException(400, msg)
    if req.style is None:
        raise HTTPException(400, "style is required for the font baseline.")
    if gen is not None and not 0 <= req.style < gen.style_classes:
        raise HTTPException(400, f"Writer style must be 0–{gen.style_classes - 1}.")
    jitter = font_baseline.JITTER if req.font_jitter else 0.0
    try:
        img, font_name = font_baseline.render_line(req.text, req.style, normalize_width=req.normalize,
                                                    jitter=jitter)
        vs = None
        if req.vendi:
            n = max(1, min(int(req.candidates), MAX_CANDIDATES))
            pool = [im for w in req.text.split()
                    for im in font_baseline.render_word_variants(w, req.style, n, jitter=jitter)]
            if len(pool) >= 2:
                if gen is None:
                    raise HTTPException(503, "model still loading")
                with _gpu_lock:
                    vs = vendi_score(gen.embed(pool))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"font baseline render failed: {e}")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return {"image": "data:image/png;base64," + b64, "style": req.style, "font": font_name,
            "jitter": req.font_jitter, "vendi": vs}

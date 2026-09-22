"""
FastAPI service for Hebrew handwriting generation.

  GET  /                -> the web UI
  GET  /api/info        -> {style_count}
  POST /api/generate    -> {image (base64 png), style, mean_nll}

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

import font_baseline
from pipeline import Generator, vendi as vendi_score

MAX_WORDS = 14
MAX_CANDIDATES = 50
# Allowed input: the full Hebrew Unicode block (letters + niqqud + Hebrew punctuation --
# geresh/gershayim already live at U+05F3/U+05F4, inside this range), ASCII digits,
# whitespace, and a fixed ASCII-punctuation whitelist. This is the same 12-mark set
# letter-scan-extractor captured from the participant collection form (! ( ) , - . / : ; ?
# plus the Hebrew geresh/gershayim already covered by the block above) -- so a mark being
# accepted here means a font in this project can actually draw it. Latin letters and any
# other symbol stay blocked.
PUNCT_ASCII = r"!(),\-./:;?"
ALLOWED_CHARS = re.compile(r"^[\u0590-\u05FF0-9" + PUNCT_ASCII + r"\s]+$")
HEBREW_LETTER = re.compile(r"[א-ת]")        # at least one real letter

# A keyboard types straight/curly quotes and an em/en-dash, not the Hebrew-native
# geresh/gershayim (׳/״) the fonts and models were actually built with --
# same fold as letter-scan-extractor/render_text.py, applied here so typing "'" or '"'
# doesn't get rejected by ALLOWED_CHARS as "not Hebrew punctuation".
QUOTE_FOLD = {
    '"': "״", "“": "״", "”": "״", "„": "״",
    "'": "׳", "‘": "׳", "’": "׳",
    "–": "-", "—": "-",
}


def normalize_text(text: str) -> str:
    for a, b in QUOTE_FOLD.items():
        text = text.replace(a, b)
    return text


def validate_text(text: str) -> str | None:
    """Return an error message if the input isn't acceptable, else None."""
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
    font_jitter: bool = True          # font_baseline only: per-glyph rotation/scale/baseline jitter on/off
    vendi: bool = False                # score diversity across the `candidates`-size pool (extra compute)


@app.get("/api/info")
def info():
    return {"style_count": gen.style_classes if gen else 0}


@app.post("/api/generate")
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


@app.post("/api/font_baseline")
def font_baseline_endpoint(req: GenReq):
    """Deterministic-font comparison render: same text, same resolved writer (`req.style`,
    required here -- no "random", the caller must pass the writer /api/generate actually
    used), rendered with the participant TTF fonts + the measured-spacing/jitter pipeline
    that built the current DiffusionPen checkpoint's training data. The render itself is
    pure CPU/PIL, no GPU lock needed -- but req.vendi additionally embeds `req.candidates`
    independent draws per word through the resident HTR model's encoder (the same one
    /api/generate uses), so that step alone takes the GPU lock and will queue behind a
    concurrent /api/generate call; the render is unaffected and still returns immediately.
    """
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


@app.get("/")
def index():
    # This is a locally-served dev UI, not a CDN'd production page -- force the browser
    # to always revalidate instead of serving a stale cached index.html after an edit
    # (this exact class of bug already cost a debugging round-trip once).
    return FileResponse("static/index.html", headers={"Cache-Control": "no-store"})


app.mount("/static", StaticFiles(directory="static"), name="static")

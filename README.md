# ✍️ Synth DiffusionPen — Hebrew Handwriting Web Service

A small **FastAPI** app that renders user-typed Hebrew text as synthetic handwriting,
using the matan-finetuned DiffusionPen checkpoint and the exp10 TrOCR Hebrew
selector. Type text → pick a writer style → get a rendered image.

The UI has two areas — an **input field** (Hebrew text) and an **output image** —
plus settings for style, candidates-per-word, aberration robustness, and width
normalization.

---

## How it works

Per word in the sentence:
1. **Generate** `N` candidate images with the diffusion model (best-of-N).
2. **Select** the best candidate with TrOCR — lowest CER vs. the intended word
   (ties broken by reader confidence). Optionally score each candidate under
   **aberrations** (blur / rotation / noise) and keep the most robust.
3. **Stitch** the chosen words right-to-left into a line, optionally normalizing
   per-character width so short words aren't stretched to long-word size.

Models are loaded **once at startup** and kept resident, so each request is
inference-only (the first request after boot is the slowest).

### Settings
| Setting | Meaning |
|---|---|
| **Writer style** | writer id `0…N-1`; each is a different handwriting. Tick *random* for a random writer. |
| **Candidates per word** | best-of-N drafts generated per word; higher = cleaner but slower. |
| **Aberration robustness** | test-time augmentation — score each draft under blur/rotation/noise and keep the most robust. |
| **Normalize letter width** | equalize per-character width so a 4-letter word isn't as wide as an 8-letter one. |

---

## Requirements

- An NVIDIA GPU (the service targets `cuda:0`).
- The **source repo** with the models + helper code, default:
  `/mnt/ssd2/cyttic/projects/test-diff-pen`
  (must contain `matan_model/`, `matan_clean/`, `style_models/`,
  `stable-diffusion-v1-5/`, `unet.py`, `generate_styled_sheet.py`).
- The **TrOCR Hebrew repo**, default: `/mnt/ssd2/cyttic/projects/TrOCR_Hebrew`
  (provides `block_processor.py`).
- A Python env with `torch`, `diffusers`, `transformers`, `pillow`, `numpy`,
  `torchvision` — e.g. the existing `/mnt/ssd2/cyttic/ml_env`.

Override the paths via env vars if they differ:
`DIFFPEN_SRC`, `HTR_REPO`, `HTR_MODEL`.

---

## Install

Install the web deps into the ML env that already has the heavy libraries:

```bash
/mnt/ssd2/cyttic/ml_env/bin/pip install -r requirements.txt
```

## Run

```bash
cd /mnt/ssd2/cyttic/projects/web-synt-diff-pen
/mnt/ssd2/cyttic/ml_env/bin/uvicorn app:app --host 0.0.0.0 --port 8000
```

Then open **http://localhost:8000** in a browser.

> First launch loads all models (~10–20 s) before the page can generate;
> watch for `[pipeline] ready: ... styles` in the console.

### API (without the UI)

```bash
curl -s -X POST http://localhost:8000/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"text":"השרים התלוננו בקבינט","style":73,"candidates":5,"aberration":false,"normalize":true}' \
  | python -c "import sys,json,base64; d=json.load(sys.stdin); open('out.png','wb').write(base64.b64decode(d['image'].split(',')[1])); print('style',d['style'],'cer',d['mean_cer'])"
```

| Endpoint | Method | Body / Returns |
|---|---|---|
| `/` | GET | the web UI |
| `/api/info` | GET | `{style_count}` |
| `/api/generate` | POST | `{text, style?, candidates, aberration, normalize}` → `{image (base64 png), style, words, mean_cer}` |

---

## Notes / limits

- Single GPU, ~8 GB: requests are **serialized** (one generation at a time).
- `candidates=1` skips TrOCR selection entirely (raw single draft, fastest).
- `style` omitted / `null` → a random writer is used and returned in the response.

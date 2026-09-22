"""
Deterministic-font baseline renderer for the web UI's comparison panel.

Renders the SAME text as /api/generate, but with the participant TTF fonts and the
exact per-pair measured spacing + per-glyph jitter pipeline (word_render_spaced.py,
diffpen-hebrew2) used to build the training data for the currently-loaded DiffusionPen
checkpoint. No neural network here -- this IS the pre-diffusion synthetic pipeline
itself, shown so the diffusion model's output can be judged against what it was
actually trained to improve on, rather than against nothing.

Stitching reuses generate_styled_sheet.normalize_widths / stitch_rtl (test-diff-pen) so
the final line is pixel-comparable (same 64px height, same 34px inter-word spacing) to
the diffusion panel -- but those two helpers expect INK-TIGHT, height=64, proportional-
width crops (that is what tight_clean() produces on the diffusion side), never the
256px-padded canvas word_render_spaced.normalize() produces for TRAINING data. Mixing
the two conventions up front silently wrecked the word-length-aware stitching (see
_tight_resize below), which is why this module has its own resize step instead of
reusing W.normalize().
"""
import glob
import os
import re
import random
import sys

import numpy as np
from PIL import Image, ImageFont

DIFFPEN_HEBREW2 = os.environ.get("DIFFPEN_HEBREW2", "/mnt/ssd2/cyttic/projects/diffpen-hebrew2")
TEST_DIFF_PEN = os.environ.get("DIFFPEN_SRC", "/mnt/ssd2/cyttic/projects/test-diff-pen")
for _p in (DIFFPEN_HEBREW2, TEST_DIFF_PEN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import word_render_spaced as W       # PT, load_stats, render_word
import generate_styled_sheet as G    # normalize_widths, stitch_rtl

FONTS_DIR = os.environ.get("FONT_BASELINE_FONTS_DIR", W.FONTS_DIR)
JITTER = float(os.environ.get("FONT_BASELINE_JITTER", "4.0"))    # matches the words9-spaced training config
TIGHTEN = float(os.environ.get("FONT_BASELINE_TIGHTEN", "1.0"))
LINE_HEIGHT = 64

_stats_cache = None
_font_cache = {}


def _stats():
    global _stats_cache
    if _stats_cache is None:
        _stats_cache = W.load_stats()
    return _stats_cache


def _writer_id(path):
    """participant_04_r2.ttf -> p04r2 ; participant_04.ttf -> p04 (repetition 1).
    Same transform as build_word_ds27.writer_name() -- must match exactly, since
    that transform is what produced the p01..p09 ids the backend's `style` is in."""
    b = os.path.basename(path)[:-4].replace("participant_", "p")
    return b.replace("_r", "r")


def _writer_fonts(style_idx):
    """style_idx: 0-8 (matches the backend's 0-indexed writer id, p01..p09) -> that
    writer's 1-3 repetition TTF paths, e.g. participant_04.ttf / _r2.ttf / _r3.ttf."""
    wr = f"p{style_idx + 1:02d}"
    paths = sorted(glob.glob(os.path.join(FONTS_DIR, "participant_*.ttf")))
    exact = [p for p in paths if re.sub(r"r\d$", "", _writer_id(p)) == wr]
    if not exact:
        raise ValueError(f"no font files for writer {wr} in {FONTS_DIR}")
    return exact


def _tight_resize(raw_crop, height=LINE_HEIGHT):
    """render_word()'s raw output is already cropped tight to ink, just not yet at a
    common height. Scale height to `height`, width proportionally -- NO padding, so
    normalize_widths sees each word's true relative width, exactly like tight_clean()
    on the diffusion side."""
    img = raw_crop.convert("RGB")
    w, h = img.size
    nw = max(1, round(w * height / h))
    return img.resize((nw, height), Image.LANCZOS)


def render_word_variants(word, style_idx, n, jitter=None, seed0=0):
    """N independent stochastic draws of ONE word for the same writer -- each draw
    re-picks a font repetition (participant_NN.ttf / _r2 / _r3, real pen-stroke
    variation) and re-samples the spacing/jitter rng, mirroring how build_word_ds27
    built k*3 variants per (word, writer). This is the font-baseline analogue of the
    diffusion model's own N candidates for a word, for a diversity (Vendi) comparison
    between the two pipelines on equal footing: same word, same writer, N draws each."""
    if jitter is None:
        jitter = JITTER
    fonts = _writer_fonts(style_idx)
    touch, over = _stats()
    blank = Image.new("RGB", (40, LINE_HEIGHT), "white")
    out = []
    for i in range(n):
        seed = seed0 + i
        fp = random.Random(seed).choice(fonts)
        if fp not in _font_cache:
            _font_cache[fp] = ImageFont.truetype(fp, W.PT)
        font = _font_cache[fp]
        ascent, _ = font.getmetrics()
        nprng = np.random.default_rng(seed)
        raw = W.render_word(word, font, ascent, touch, over, nprng, TIGHTEN, jitter)
        out.append(_tight_resize(raw) if raw is not None else blank)
    return out


def render_glyphs(text, style_idx, font_seed=None, jitter=None):
    """One tight RGB crop of an arbitrary glyph run -- letters, digits, or punctuation --
    for the given writer. Same per-glyph pipeline as render_line()'s per-word loop, just
    exposed for a single run so pipeline.py can attach a punctuation mark DiffusionPen was
    never trained to draw directly onto a diffusion-generated word.

    `font_seed` fixes WHICH of the writer's physical repetitions is used -- pass the same
    value for every call across one line so every glued-in mark looks like the same hand.
    The per-glyph spacing/jitter draw is deliberately never seeded, so repeated marks in
    one line (e.g. two commas) don't render pixel-identical."""
    if jitter is None:
        jitter = JITTER
    fonts = _writer_fonts(style_idx)
    fp = random.Random(font_seed).choice(fonts) if font_seed is not None else random.choice(fonts)
    if fp not in _font_cache:
        _font_cache[fp] = ImageFont.truetype(fp, W.PT)
    font = _font_cache[fp]
    ascent, _ = font.getmetrics()
    touch, over = _stats()
    nprng = np.random.default_rng()
    raw = W.render_word(text, font, ascent, touch, over, nprng, TIGHTEN, jitter)
    return _tight_resize(raw) if raw is not None else None


def render_line(text, style_idx, seed=None, normalize_width=True, jitter=None):
    """Returns (PIL.Image line, font_basename actually used).

    `jitter=None` uses the training-matched default (JITTER, module const); pass 0.0 to
    render with the measured spacing but no per-glyph rotation/scale/baseline jitter --
    e.g. to show what the same pipeline looks like with augmentation switched off."""
    if jitter is None:
        jitter = JITTER
    fonts = _writer_fonts(style_idx)
    fp = random.Random(seed).choice(fonts)
    if fp not in _font_cache:
        _font_cache[fp] = ImageFont.truetype(fp, W.PT)
    font = _font_cache[fp]
    ascent, _ = font.getmetrics()
    touch, over = _stats()
    nprng = np.random.default_rng(seed)

    words = text.split()
    blank = Image.new("RGB", (40, LINE_HEIGHT), "white")   # only hit if a word has zero renderable ink
    imgs = []
    for w in words:
        raw = W.render_word(w, font, ascent, touch, over, nprng, TIGHTEN, jitter)
        imgs.append(_tight_resize(raw) if raw is not None else blank)

    if normalize_width:
        imgs = G.normalize_widths(imgs, words, LINE_HEIGHT)
    line = G.stitch_rtl(imgs, LINE_HEIGHT, space=34, pad=10)
    return line, os.path.basename(fp)

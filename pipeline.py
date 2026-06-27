"""
Handwriting generation pipeline for the web service.

Wraps the matan-finetuned DiffusionPen checkpoint + exp10 TrOCR selector that
live in the test-diff-pen repo. Models are loaded ONCE (Generator() in __init__)
and kept resident, so each web request only runs inference, not loading.

Reuses the proven helpers (load_ref_images, augment_views, stitch_rtl,
normalize_widths, cer) straight from generate_styled_sheet.py to stay in sync
with the CLI experiments instead of duplicating them.
"""
import os
# must be set before torch initializes the CUDA caching allocator
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import sys, json, random
from types import SimpleNamespace
import torch
import torchvision
from torch.nn import DataParallel
from diffusers import AutoencoderKL, DPMSolverMultistepScheduler, DDIMScheduler
from transformers import CanineModel, CanineTokenizer

# --- locate the source repo that holds the models + helper code ---
SRC = os.environ.get("DIFFPEN_SRC", "/mnt/ssd2/cyttic/projects/test-diff-pen")
HTR_REPO = os.environ.get("HTR_REPO", "/mnt/ssd2/cyttic/projects/TrOCR_Hebrew")
sys.path.insert(0, SRC)
sys.path.insert(0, HTR_REPO)

from unet import UNetModel
from feature_extractor import ImageEncoder
import generate_styled_sheet as G   # load_ref_images, augment_views, stitch_rtl, normalize_widths, cer

MATAN = os.path.join(SRC, "matan_clean")
SAVE_PATH = os.path.join(SRC, "matan_model")
STYLE_PATH = os.path.join(SRC, "style_models", "iam_style_diffusionpen.pth")
SD = os.path.join(SRC, "stable-diffusion-v1-5")
HTR_MODEL = os.environ.get("HTR_MODEL", "cyttic/exp10-trocr-hebrew-matan-full")
IMG = (64, 256)
DEV = G.DEVICE  # "cuda:0"

# UNet checkpoint source. Precedence: explicit local path (MATAN_CKPT) ->
# download ema from an HF repo (MATAN_HF_REPO, defaults to the recently
# fine-tuned model) -> local matan_model. Same architecture (491 styles / 29
# vocab), so any of these load into the UNet below.
MATAN_CKPT = os.environ.get("MATAN_CKPT")
MATAN_HF_REPO = os.environ.get("MATAN_HF_REPO", "cyttic/diffusionpen-matan-ft")


def resolve_ckpt():
    if MATAN_CKPT and os.path.isfile(MATAN_CKPT):
        return MATAN_CKPT, MATAN_CKPT
    if MATAN_HF_REPO:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(MATAN_HF_REPO, "ema_ckpt.pt"), MATAN_HF_REPO
    local = os.path.join(SAVE_PATH, "models", "ema_ckpt.pt")
    return local, local


class Generator:
    """Resident model bundle + single-sentence generation."""

    def __init__(self):
        print("[pipeline] loading models ...")
        # vocab + writer/style maps + reference word list
        char_classes = json.load(open(os.path.join(MATAN, "character_classes.json"), encoding="utf-8"))
        self.vocab_size = len(char_classes)
        wr_dict = json.load(open(os.path.join(MATAN, "writers_dict_train.json")))
        self.reverse_wr = {v: k for k, v in wr_dict.items()}
        self.style_classes = len(wr_dict)
        self.train_data = [l.strip().split(",") for l in
                           open(os.path.join(MATAN, "splits_words", "matan_train.txt"), encoding="utf-8")]
        self.words_root = os.path.join(MATAN, "words")

        # --- diffusion stack ---
        self.tok = CanineTokenizer.from_pretrained("google/canine-c")
        te = DataParallel(CanineModel.from_pretrained("google/canine-c"), device_ids=[0]).to(DEV)
        self.unet = UNetModel(image_size=IMG, in_channels=4, model_channels=320, out_channels=4,
                              num_res_blocks=1, attention_resolutions=(1, 1), channel_mult=(1, 1),
                              num_heads=4, num_classes=self.style_classes, context_dim=320,
                              vocab_size=self.vocab_size, text_encoder=te,
                              args=SimpleNamespace(interpolation=False, mix_rate=None))
        self.unet = DataParallel(self.unet, device_ids=[0]).to(DEV)
        ckpt_path, ckpt_src = resolve_ckpt()
        self.unet.load_state_dict(torch.load(ckpt_path, map_location=DEV, weights_only=False))
        self.unet.eval()
        print(f"[pipeline] checkpoint: {ckpt_src}")
        self.vae = DataParallel(AutoencoderKL.from_pretrained(SD, subfolder="vae"), device_ids=[0]).to(DEV).eval()
        # samplers the UI can pick; per-sampler default steps for good quality
        ddim = DDIMScheduler.from_pretrained(SD, subfolder="scheduler")
        self.schedulers = {
            "dpm": DPMSolverMultistepScheduler.from_pretrained(SD, subfolder="scheduler"),
            "ddim": ddim,
            "ddim100": ddim,   # same solver, more steps (best CER in the sweep)
        }
        self.default_steps = {"dpm": 15, "ddim": 50, "ddim100": 100}
        fe = ImageEncoder(model_name="mobilenetv2_100", num_classes=0, pretrained=True, trainable=True)
        st = torch.load(STYLE_PATH, map_location=DEV, weights_only=False); md = fe.state_dict()
        fe.load_state_dict({**md, **{k: v for k, v in st.items() if k in md and md[k].shape == v.shape}})
        self.fe = DataParallel(fe, device_ids=[0]).to(DEV).eval()
        self._feat_cache = {}

        # --- TrOCR selector (resident) ---
        from transformers import VisionEncoderDecoderModel, AutoTokenizer
        from block_processor import HebrewBlockProcessor
        self.htr = VisionEncoderDecoderModel.from_pretrained(HTR_MODEL).to(DEV).half().eval()
        self.htok = AutoTokenizer.from_pretrained(HTR_MODEL)
        self.proc = HebrewBlockProcessor()
        self.htr.generation_config.decoder_start_token_id = self.htok.cls_token_id
        self.htr.generation_config.pad_token_id = self.htok.pad_token_id
        self.htr.generation_config.eos_token_id = self.htok.sep_token_id
        print(f"[pipeline] ready: {self.style_classes} styles / {self.vocab_size} vocab")

    # ---- helpers ----
    def _style_feat(self, s):
        if s not in self._feat_cache:
            with torch.no_grad():
                refs = G.load_ref_images(s, self.train_data, self.reverse_wr, self.words_root)
                self._feat_cache[s] = self.fe(refs).detach()
        return self._feat_cache[s]

    def _gen_candidates(self, word, sfeat, s, k, steps, sched, batch=8):
        """Generate k candidates in sub-batches so peak VRAM is bounded by `batch`."""
        out = []
        for start in range(0, k, batch):
            kk = min(batch, k - start)
            labels = torch.tensor([s] * kk).long().to(DEV)
            sfeat_b = sfeat.repeat(kk, 1)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                tf = self.tok([word] * kk, padding="max_length", truncation=True,
                              return_tensors="pt", max_length=40).to(DEV)
                x = torch.randn((kk, 4, IMG[0] // 8, IMG[1] // 8)).to(DEV)
                sched.set_timesteps(steps)
                for t in sched.timesteps:
                    tt = (torch.ones(kk) * t.item()).long().to(DEV)
                    resid = self.unet(x, tt, tf, labels, original_images=None,
                                      mix_rate=None, style_extractor=sfeat_b)
                    x = sched.step(resid, t, x).prev_sample
                latents = 1 / 0.18215 * x
                imgs = self.vae.module.decode(latents).sample
                imgs = (imgs / 2 + 0.5).clamp(0, 1).cpu()
            out.extend(G.tight_clean(torchvision.transforms.ToPILImage()(im).convert("RGB")) for im in imgs)
            del labels, sfeat_b, x, latents, imgs, resid
        return out

    def _ocr(self, imgs, batch=16):
        """OCR in sub-batches so a big candidate*views set can't spike VRAM."""
        preds, confs = [], []
        for start in range(0, len(imgs), batch):
            chunk = imgs[start:start + batch]
            pv = self.proc(chunk)["pixel_values"].to(DEV, dtype=self.htr.dtype)
            with torch.no_grad():
                gen = self.htr.generate(pv, num_beams=2, max_length=48,
                                        output_scores=True, return_dict_in_generate=True)
            preds += self.htok.batch_decode(gen.sequences, skip_special_tokens=True)
            ss = getattr(gen, "sequences_scores", None)
            confs += torch.exp(ss).tolist() if ss is not None else [0.0] * len(chunk)
            del pv, gen
        return preds, confs

    def _select(self, imgs, w, aberration):
        """Return (index, clean_cer) of the chosen candidate."""
        if aberration:
            av = lambda im: G.augment_views(im)
            V = len(av(imgs[0]))
            flat = [v for im in imgs for _, v in av(im)]
            vp, vc = self._ocr(flat)
            mean_cer, mean_conf, clean = [], [], []
            for i in range(len(imgs)):
                cers = [G.cer(vp[i * V + j], w) for j in range(V)]
                mean_cer.append(sum(cers) / V)
                mean_conf.append(sum(vc[i * V + j] for j in range(V)) / V)
                clean.append(cers[0])
            bi = min(range(len(imgs)), key=lambda i: (mean_cer[i], -mean_conf[i]))
            return bi, clean[bi]
        preds, confs = self._ocr(imgs)
        cers = [G.cer(preds[i], w) for i in range(len(imgs))]
        bi = min(range(len(imgs)), key=lambda i: (cers[i], -confs[i]))
        return bi, cers[bi]

    # ---- public API ----
    def generate(self, text, style=None, candidates=5, aberration=False,
                 normalize=True, sampler="dpm", steps=None):
        words = text.split()
        if not words:
            raise ValueError("empty text")
        candidates = max(1, min(int(candidates), 50))
        sampler = sampler if sampler in self.schedulers else "dpm"
        sched = self.schedulers[sampler]
        steps = int(steps) if steps else self.default_steps[sampler]
        s = random.randint(0, self.style_classes - 1) if style is None else \
            max(0, min(int(style), self.style_classes - 1))
        try:
            sfeat = self._style_feat(s)
            chosen, cers = [], []
            for w in words:
                imgs = self._gen_candidates(w, sfeat, s, candidates, steps, sched)
                if candidates == 1:
                    chosen.append(imgs[0]); cers.append(None)
                else:
                    bi, c = self._select(imgs, w, aberration)
                    chosen.append(imgs[bi]); cers.append(c)

            if normalize:
                chosen = G.normalize_widths(chosen, words, 64)
            line = G.stitch_rtl(chosen, 64, space=34, pad=10)
            mean_cer = None
            if any(c is not None for c in cers):
                vals = [c for c in cers if c is not None]
                mean_cer = sum(vals) / len(vals)
            return {"image": line, "style": s, "words": len(words), "mean_cer": mean_cer,
                    "sampler": sampler, "steps": steps}
        finally:
            torch.cuda.empty_cache()

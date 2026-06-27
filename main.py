# -*- coding: utf-8 -*-
"""
Thin web frontend (deployed to Azure). Serves the UI and forwards requests to the
model server running on the GPU notebook, reached via the reverse SSH tunnel.

No torch / model here — just static files + an HTTP forward, so the container
stays tiny and needs almost no RAM.

    MODEL_SERVER_URL  env var -> where the model server is (default the tunnel).

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000
"""
import os

import httpx
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

MODEL_SERVER = os.environ.get("MODEL_SERVER_URL", "http://127.0.0.1:8001").rstrip("/")

app = FastAPI(title="Synth DiffusionPen — frontend")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/api/info")
async def info():
    """Proxy the style count from the model server (0 if it's unreachable)."""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(f"{MODEL_SERVER}/info")
            return r.json()
    except Exception:
        return {"style_count": 0}


@app.get("/api/health")
async def health():
    """Is the GPU model server reachable through the tunnel? Drives the UI indicator."""
    try:
        async with httpx.AsyncClient(timeout=5) as c:
            d = (await c.get(f"{MODEL_SERVER}/health")).json()
        return {"connected": True, "status": d.get("status", "ok"),
                "style_count": d.get("style_count", 0)}
    except Exception:
        return {"connected": False, "status": "offline", "style_count": 0}


@app.post("/api/generate")
async def generate(req: Request):
    """Forward the generation request to the model server and relay its response."""
    body = await req.json()
    try:
        async with httpx.AsyncClient(timeout=180) as c:
            r = await c.post(f"{MODEL_SERVER}/generate", json=body)
    except Exception as e:
        raise HTTPException(502, f"Model server unreachable (is the SSH tunnel up?): {e}")
    # relay status + JSON so the UI sees the same shape (image or {detail})
    return JSONResponse(status_code=r.status_code, content=r.json())


app.mount("/static", StaticFiles(directory="static"), name="static")

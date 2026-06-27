# Remote-backend deployment (same pattern as webHebrewOCR)

The GPU work stays on the notebook; only a **thin frontend** runs on the Azure VM.

```
 browser ──HTTPS──> Azure VM (Docker: main.py, port 80)
                          │  HTTP to 127.0.0.1:8001
                          ▼
                 reverse SSH tunnel
                          ▲
                          │
       GPU notebook: model_server.py (pipeline.Generator), 127.0.0.1:8001
```

- **Frontend** `main.py` — serves `static/index.html`, proxies `/api/info` and
  `/api/generate` to `MODEL_SERVER_URL` (default `http://127.0.0.1:8001`). No torch.
- **Backend** `model_server.py` — loads `pipeline.Generator` (diffusion + TrOCR),
  exposes `/health`, `/info`, `/generate`. Runs on the notebook.
- **Deploy** `.github/workflows/deploy.yml` — builds the frontend image, pushes to
  Docker Hub, SSHes into the Azure VM and restarts the container with
  `--network host` so it can reach the tunnel.

## 1. Run the model server on the notebook
```bash
cd /mnt/ssd2/cyttic/projects/web-synt-diff-pen
/mnt/ssd2/cyttic/ml_env/bin/uvicorn model_server:app --host 127.0.0.1 --port 8001
# (uses the fine-tuned model by default; MATAN_HF_REPO / MATAN_CKPT to change)
```

## 2. Open the reverse SSH tunnel (notebook -> Azure VM)
Forwards the VM's `127.0.0.1:8001` back to the notebook's model server:
```bash
ssh -N -R 8001:127.0.0.1:8001 <azure_user>@<azure_vm_host>
# keep it alive: add  -o ServerAliveInterval=30  (or run under autossh / systemd)
```
The VM's `sshd` needs `GatewayPorts clientspecified` only if you bind a non-loopback
address; binding to `127.0.0.1` (default above) works out of the box.

## 3. Deploy the frontend (automatic)
Push to `main` (or `remote-backend`) — the workflow builds + deploys. Required
GitHub **secrets** (same names as webHebrewOCR):

| secret | value |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub user |
| `DOCKERHUB_TOKEN` | Docker Hub access token |
| `AZURE_VM_HOST` | Azure VM public IP / hostname |
| `AZURE_VM_USER` | SSH user on the VM |
| `AZURE_VM_SSH_KEY` | private SSH key for that user |

After deploy, the site is on the VM's port 80, and every request flows through the
tunnel to the notebook's GPU. If the tunnel is down, the UI loads but generation
returns `502 Model server unreachable`.

## Local all-in-one (no split)
For local dev you can still run the monolithic `app.py` (frontend + backend in one
process): `uvicorn app:app --host 0.0.0.0 --port 8001`.

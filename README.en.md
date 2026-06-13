# AI-PhotoViewer

**English** | [繁體中文](README.md)

<p align="center">
  <img src="repo-cover.png" alt="AI-PhotoViewer — local, private semantic photo search (YOLOE + SigLIP 2)" width="820">
</p>

A local, privacy-first photo browser that combines **YOLOE open-vocabulary object detection** with **SigLIP 2 natural-language semantic search** (Chinese & English). Scan a folder once, then find photos by *meaning* — `two girls`, `海邊日落`, `a laptop on a desk` — and inspect every detected object with interactive masks. Everything runs locally on your GPU; nothing leaves the machine.

<p align="center">
  <img src="demo.png" alt="Web UI: results grid · photo viewer · detection inspector" width="900">
</p>

## Features

- 🔍 **Semantic search (zh/en)** — type natural language, get ranked photos (SigLIP 2 embeddings + sqlite-vec)
- 🏷️ **Open-vocabulary detection** — YOLOE labels + segmentation masks per photo
- 🖼️ **Results grid** — thumbnails with similarity scores; click to inspect
- 🎯 **Detection inspector** — hover / lock object masks on a canvas
- 🎲 **Shuffle + pagination** — browse in random order, page through, adjustable page size
- 🎚️ **Top-N + threshold** — control how many results and how relevant
- 🗂️ **Welcome page (all in the Web UI)** — add / reindex / cancel / remove folders, live progress, and backend status (GPU/VRAM/model/coverage) — no command line needed
- 100% local · single SQLite file · runs on one GPU

## Pipeline

```
your image folder
   ──► scan.py    YOLOE-11s-seg-pf: detect + segment      ──► photos.db (SQLite)
   ──► embed.py   SigLIP 2: image → vector                ──► vec_photos (sqlite-vec)
   ──► web_demo   FastAPI: /api/search /api/photos ...     ──► browser UI
```

## Requirements

- NVIDIA GPU (developed on RTX 5070 Ti, Blackwell)
- Python 3.12 (Windows or Linux)
- ~1.5–4.5 GB disk for a SigLIP model

## Setup

```bash
# 1) Create a virtual env (uv recommended; plain `python -m venv` works too)
uv venv

# 2) Install PyTorch FIRST, matching your GPU.
#    RTX 50-series (Blackwell) needs cu128:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3) Install the rest
uv pip install -r requirements.txt
```

### Download a SigLIP 2 model

Put a model directory somewhere (e.g. `../models/`) and point `embed.py` / the server at it:

| Model | Dim | Size | Notes |
|-------|-----|------|-------|
| `google/siglip2-base-patch16-224` | 768 | ~1.4 GB | fast |
| `google/siglip2-so400m-patch14-384` | 1152 | ~4.2 GB | better, esp. **Chinese** |

> If HuggingFace downloads hang (Xet protocol), set `HF_HUB_DISABLE_XET=1`, or download the
> files directly with `curl -L -C - --retry 40 <resolve-url>` into a local folder.

## Usage

### Quick start (scripts)

```powershell
# Windows / PowerShell
.venv\Scripts\python.exe scripts\check-env.py   # env check (packages / CUDA / sqlite-vec / model)
scripts\run-server.ps1                  # start (defaults to 127.0.0.1:8000, local only)
scripts\run-server.ps1 -Port 8080       # different port
scripts\run-server.ps1 -BindHost 0.0.0.0    # expose on LAN
scripts\run-server.ps1 -Stop            # stop
```
```bash
# Linux / macOS / Git Bash
.venv/bin/python scripts/check-env.py
scripts/run-server.sh                   # start
scripts/run-server.sh --port 8080
scripts/run-server.sh --stop            # stop
```

An empty `photos.db` is **created automatically on first launch** — then just index from the welcome page.
The model path defaults to `../models/siglip2-so400m`; override with `-Model` (ps1) / `--model` (sh) or the `SIGLIP_MODEL` env var.

### Manual CLI (advanced)

```bash
# 1) Scan a photo folder (incremental & resumable)
python scan.py "C:\path\to\your\photos"

# 2) Build the semantic index (incremental — only embeds new photos)
python embed.py --model ..\models\siglip2-so400m

# 3) Launch the web UI
python web_demo\main.py --db photos.db --model ..\models\siglip2-so400m
# open http://127.0.0.1:8000
```

Switching to a model with a different dimension: `python embed.py --model <dir> --rebuild`.
Quick CLI search without the web UI: `python embed.py --search "海邊" --model <dir>`.

### All in the Web UI (no command line)

Once the server is up, you can index entirely from the welcome page — **no need to run `scan.py` / `embed.py` first**:

- Drop photos into the project's **`default-image/`** folder (it contains a `PUT_YOUR_IMAGES_HERE.txt` marker) → back in the UI, click **⟳ All** on it to index; or
- Use **📁 Browse…** to pick any folder (or paste a path) → **Index new photos**

Indexing, reindexing, cancel, remove-source, and backend status all happen on the welcome page. `default-image/` is auto-registered as the default source when the server starts.

## Web UI guide

- **Welcome page (default home)**: backend status card (GPU/VRAM, model, photo count, coverage) · ➕ add a folder (paste a path or 📁 browse) · each indexed source has **🔁 New / ⟳ All / 🗑 Remove** · job progress bar + cancel · **📂 Enter gallery**, and **🏠** in the gallery returns home
- **Search box** (zh/en) + `top` (how many) + `threshold` (absolute similarity cutoff — the number
  matches the green badge on each thumbnail; far-left = show all)
- **Results grid** (left): thumbnails + similarity; `🎲` reshuffle · `‹ ›` prev/next set · page-size 10–40
- **Filter**: narrow by YOLOE-detected class
- **Inspector** (right): hover an object to highlight its mask, click to lock; full detection list
- **❓** usage modal (zh / EN toggle) · **🗺** pipeline diagram
- Shortcuts: `←` `→` change photo · `R` random · `Esc` unlock / close window

> SigLIP similarity scores are small in absolute terms (~0.04–0.11) and clustered — what matters is the
> *ranking*, not the raw number.

## Project structure

```
scan.py            YOLOE detection + masks  → photos.db
embed.py           SigLIP 2 embeddings      → vec_photos   (also: --search CLI, --rebuild)
inspect_db.py      DB stats / co-occurrence
add_masks.py       backfill masks for existing rows
web_demo/
  main.py          FastAPI server + REST API (/api/search, /api/photos, /api/thumb, /api/health, /api/index, ...)
  jobs.py          background index jobs (scan→embed, progress/cancel, prune, sources)
  static/          index.html · app.js · style.css
default-image/     default source folder (drop photos here; auto-registered at startup)
requirements.txt
```

## Notes

- **Blackwell (RTX 50xx)** GPUs require PyTorch **cu128** wheels — the default PyPI build won't use the GPU.
- The vector index lives **inside `photos.db`** (sqlite-vec) — one file, easy to back up.
- `photos.db`, SigLIP weights (`models/`, `*.pt`), the venv and generated thumbnails are git-ignored.
- **Server log**: rotating, at `web_demo/server.log` — 1 MB per file, 2 backups kept (3 MB cap).
- The SigLIP 2 weights are subject to Google's model license; YOLOE/Ultralytics under AGPL-3.0.

## Performance (RTX 5070 Ti · so400m)

| Metric | Value |
|--------|-------|
| VRAM (search, SigLIP loaded) | ~6.1 GB |
| VRAM (indexing, YOLOE + SigLIP) | ~6.6 GB |
| YOLOE detect + segment | ~41 img/s |
| SigLIP embedding | ~22 img/s |
| End-to-end indexing | ~15 img/s (GPU inference; a bit lower with disk / EXIF / DB) |

> Measured on a single GPU. The `base` model (768-dim) is lighter and faster, but weaker on Chinese.

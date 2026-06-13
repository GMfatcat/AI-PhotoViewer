#!/usr/bin/env python3
"""
Photo Archaeology web demo — FastAPI backend.

Loads photos from a SQLite DB (created by scan.py), serves them with their
detections and mask polygons. Frontend (in ./static/) provides hover/click
interaction.
"""

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import sqlite_vec
from fastapi import Body, FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Allow importing embed.py (lives one dir up) for the SigLIP embedder
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import jobs  # noqa: E402  (background indexing manager, in this dir)


# Global state set by CLI
DB_PATH: Optional[Path] = None
MODEL_PATH: Optional[str] = None
YOLOE_PATH: Path = Path(__file__).resolve().parent.parent / "yoloe-11s-seg-pf.pt"
_EMBEDDER = None  # SigLIP embedder (eager-loaded at startup; reused by search + indexing)
LOG_FILE: Path = Path(__file__).parent / "server.log"  # rotating; 3 MB total cap
DEFAULT_DIR: Path = Path(__file__).resolve().parent.parent / "default-image"  # project-local default source

# Image/thumb URLs are keyed by photo_id, which is reassigned on every DB rebuild.
# Force the browser to revalidate (via ETag) so it never displays a stale image
# cached under an id that now points to a different file.
NO_CACHE = {"Cache-Control": "no-cache"}


def get_conn() -> sqlite3.Connection:
    if DB_PATH is None or not DB_PATH.exists():
        raise HTTPException(status_code=500, detail="DB not configured")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")  # tolerate writes from an indexing job
    return conn


app = FastAPI(title="Photo Archaeology", docs_url="/docs")


def _ver(file_hash) -> str:
    """Content-version token for cache-busting image/thumb URLs. photo_id is
    reused across DB rebuilds, so URLs keyed only by id can serve a stale
    browser-cached image for a file that now lives under a different id.
    Tying the URL to file_hash (content) makes the URL change when the file does."""
    return (file_hash or "0")[:10]


@app.middleware("http")
async def no_cache_media(request, call_next):
    """Force `Cache-Control: no-cache` on every image/thumb response, INCLUDING
    error responses (404/410). Error responses go through the exception handler,
    which doesn't carry the per-FileResponse headers — so a transient "file
    missing" 410 would otherwise get browser-cached and keep showing a broken
    thumbnail even after the file is restored (same id+hash → same URL)."""
    response = await call_next(request)
    path = request.url.path
    if path.startswith("/api/image") or path.startswith("/api/thumb"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def containment_ratio(b1: dict, b2: dict) -> float:
    """
    Intersection over smaller-area (containment ratio).
    1.0 = one bbox fully contained inside the other.
    Inputs are dicts with 'x' (center), 'y' (center), 'w', 'h' (all normalized).
    """
    x1a, y1a = b1['x'] - b1['w'] / 2, b1['y'] - b1['h'] / 2
    x2a, y2a = b1['x'] + b1['w'] / 2, b1['y'] + b1['h'] / 2
    x1b, y1b = b2['x'] - b2['w'] / 2, b2['y'] - b2['h'] / 2
    x2b, y2b = b2['x'] + b2['w'] / 2, b2['y'] + b2['h'] / 2
    iw = max(0.0, min(x2a, x2b) - max(x1a, x1b))
    ih = max(0.0, min(y2a, y2b) - max(y1a, y1b))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = (x2a - x1a) * (y2a - y1a)
    a2 = (x2b - x1b) * (y2b - y1b)
    smaller = min(a1, a2)
    return inter / smaller if smaller > 0 else 0.0


def dedup_overlapping(detections: list, threshold: float) -> list:
    """
    Drop smaller of same-class pairs whose containment ratio is >= threshold.
    Expects detections sorted by bbox area DESC (so the FIRST occurrence is larger).
    threshold <= 0 disables dedup.
    """
    if threshold <= 0 or len(detections) < 2:
        return detections
    suppressed = set()
    n = len(detections)
    for i in range(n):
        if i in suppressed:
            continue
        for j in range(i + 1, n):
            if j in suppressed:
                continue
            if detections[i]['class'] != detections[j]['class']:
                continue
            if containment_ratio(detections[i]['bbox'], detections[j]['bbox']) >= threshold:
                # j is the smaller one (sort order) — suppress it
                suppressed.add(j)
    return [d for i, d in enumerate(detections) if i not in suppressed]


@app.get("/api/stats")
def stats():
    """Overall stats for the navigation bar."""
    conn = get_conn()
    try:
        n_photos = conn.execute(
            "SELECT COUNT(*) FROM photos WHERE processed_at IS NOT NULL AND error IS NULL"
        ).fetchone()[0]
        n_classes = conn.execute(
            "SELECT COUNT(DISTINCT class) FROM detections"
        ).fetchone()[0]
        return {"photos": n_photos, "classes": n_classes}
    finally:
        conn.close()


@app.get("/api/classes")
def classes():
    """List all classes with photo counts, for the filter dropdown."""
    conn = get_conn()
    try:
        rows = conn.execute("""
            SELECT class, COUNT(DISTINCT photo_id) AS n
            FROM detections
            GROUP BY class
            ORDER BY n DESC, class ASC
        """).fetchall()
        return [{"class": r["class"], "photos": r["n"]} for r in rows]
    finally:
        conn.close()


@app.get("/api/photos")
def list_photos(
    cls: Optional[str] = Query(None, description="Filter to photos with this class"),
    min_conf: float = Query(0.4),
    limit: int = Query(2000, le=10000),
):
    """List photo IDs (newest first). For navigation."""
    conn = get_conn()
    try:
        if cls:
            rows = conn.execute("""
                SELECT DISTINCT p.id, p.path, p.taken_at, p.file_hash
                FROM photos p
                JOIN detections d ON d.photo_id = p.id
                WHERE p.processed_at IS NOT NULL AND p.error IS NULL
                  AND d.class = ? AND d.confidence >= ?
                ORDER BY p.id DESC
                LIMIT ?
            """, (cls, min_conf, limit)).fetchall()
        else:
            rows = conn.execute("""
                SELECT id, path, taken_at, file_hash FROM photos
                WHERE processed_at IS NOT NULL AND error IS NULL
                ORDER BY id DESC
                LIMIT ?
            """, (limit,)).fetchall()
        return [{"id": r["id"], "name": Path(r["path"]).name, "taken_at": r["taken_at"],
                 "v": _ver(r["file_hash"])}
                for r in rows]
    finally:
        conn.close()


@app.get("/api/photo/{photo_id}")
def get_photo(
    photo_id: int,
    min_conf: float = Query(0.4),
    dedup_overlap: float = Query(
        0.8,
        description="Suppress smaller of same-class detections when containment "
                    "ratio (intersection / smaller-area) >= this value. "
                    "Set to 0 to disable.",
    ),
):
    """Return everything needed to render one photo + interactive masks."""
    conn = get_conn()
    try:
        photo_row = conn.execute("""
            SELECT id, path, width, height, taken_at, gps_lat, gps_lon,
                   camera_make, camera_model, file_hash
            FROM photos WHERE id = ?
        """, (photo_id,)).fetchone()
        if not photo_row:
            raise HTTPException(status_code=404, detail="Photo not found")

        photo_path = Path(photo_row["path"])
        if not photo_path.exists():
            raise HTTPException(status_code=410,
                                detail=f"Photo file missing: {photo_path}")

        det_rows = conn.execute("""
            SELECT d.id, d.class, d.confidence,
                   d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h, d.bbox_area_ratio,
                   m.polygon_json
            FROM detections d
            LEFT JOIN masks m ON m.detection_id = d.id
            WHERE d.photo_id = ? AND d.confidence >= ?
            ORDER BY d.bbox_area_ratio DESC
        """, (photo_id, min_conf)).fetchall()

        detections = []
        for r in det_rows:
            poly = json.loads(r["polygon_json"]) if r["polygon_json"] else None
            detections.append({
                "id": r["id"], "class": r["class"], "confidence": round(r["confidence"], 3),
                "bbox": {"x": r["bbox_x"], "y": r["bbox_y"],
                         "w": r["bbox_w"], "h": r["bbox_h"]},
                "area_ratio": round(r["bbox_area_ratio"] or 0, 4),
                "polygon": poly,
            })

        # Drop nested same-class duplicates (rows already sorted by area DESC)
        before = len(detections)
        detections = dedup_overlapping(detections, dedup_overlap)
        suppressed = before - len(detections)

        return {
            "id": photo_row["id"],
            "name": photo_path.name,
            "image_url": f"/api/image/{photo_id}?v={_ver(photo_row['file_hash'])}",
            "width": photo_row["width"], "height": photo_row["height"],
            "taken_at": photo_row["taken_at"],
            "gps": ({"lat": photo_row["gps_lat"], "lon": photo_row["gps_lon"]}
                    if photo_row["gps_lat"] is not None else None),
            "camera": (f"{photo_row['camera_make'] or ''} "
                       f"{photo_row['camera_model'] or ''}").strip() or None,
            "detections": detections,
            "suppressed_count": suppressed,
        }
    finally:
        conn.close()


@app.get("/api/image/{photo_id}")
def get_image(photo_id: int):
    """Serve the actual image file from its DB-stored absolute path."""
    conn = get_conn()
    try:
        row = conn.execute("SELECT path FROM photos WHERE id = ?",
                           (photo_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Photo not found")
        photo_path = Path(row["path"])
        if not photo_path.exists():
            raise HTTPException(status_code=410, detail="Photo file missing")

        media_type = "image/jpeg"
        ext = photo_path.suffix.lower()
        if ext == ".png":
            media_type = "image/png"
        elif ext == ".webp":
            media_type = "image/webp"
        elif ext in (".heic", ".heif"):
            # Browsers can't display HEIC; convert on the fly
            from io import BytesIO
            from PIL import Image, ImageOps
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except ImportError:
                pass
            with Image.open(photo_path) as img:
                try:
                    img = ImageOps.exif_transpose(img)
                except Exception:
                    pass
                if img.mode != "RGB":
                    img = img.convert("RGB")
                buf = BytesIO()
                img.save(buf, format="JPEG", quality=88)
                buf.seek(0)
                from fastapi.responses import Response
                return Response(content=buf.read(), media_type="image/jpeg",
                                headers=NO_CACHE)

        return FileResponse(photo_path, media_type=media_type, headers=NO_CACHE)
    finally:
        conn.close()


# ── Semantic search (SigLIP 2 + sqlite-vec) ───────────────
def vec_conn() -> sqlite3.Connection:
    """A DB connection with the sqlite-vec extension loaded."""
    if DB_PATH is None or not DB_PATH.exists():
        raise HTTPException(status_code=500, detail="DB not configured")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def get_embedder():
    """Return the shared SigLIP embedder (loaded once; reused by search + indexing)."""
    global _EMBEDDER
    if _EMBEDDER is None:
        import embed  # reuse the same Embedder as embed.py
        _EMBEDDER = embed.Embedder(MODEL_PATH)
        _EMBEDDER.dim = len(_EMBEDDER.embed_text("dimension probe"))
    return _EMBEDDER


@app.get("/api/search")
def search_photos(
    q: str = Query(..., min_length=1, description="Natural-language query (zh/en)"),
    top: int = Query(12, le=200, description="Max results to return"),
    min_ratio: float = Query(0.0, ge=0, le=1,
                             description="Relative cutoff: keep results with sim >= top_sim * ratio"),
):
    """Semantic search over photo embeddings. Returns same shape as /api/photos plus `sim`."""
    conn = vec_conn()
    try:
        has_index = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vec_photos'"
        ).fetchone()
        if not has_index:
            raise HTTPException(status_code=503,
                                detail="Semantic index not built (run embed.py first)")
        qvec = get_embedder().embed_text(q)
        rows = conn.execute("""
            SELECT v.photo_id AS id, v.distance AS dist, p.path AS path,
                   p.taken_at AS taken_at, p.file_hash AS file_hash
            FROM vec_photos v
            JOIN photos p ON p.id = v.photo_id
            WHERE v.embedding MATCH ? AND k = ?
            ORDER BY v.distance
        """, (sqlite_vec.serialize_float32(qvec), top)).fetchall()
        results = [{"id": r["id"], "name": Path(r["path"]).name,
                    "taken_at": r["taken_at"], "sim": round(1.0 - r["dist"], 3),
                    "v": _ver(r["file_hash"])}
                   for r in rows]
        # Relative threshold: SigLIP sims are narrow, so cut the tail by a fraction
        # of the best score rather than an unreliable absolute value.
        if results and min_ratio > 0:
            cutoff = results[0]["sim"] * min_ratio
            results = [r for r in results if r["sim"] >= cutoff]
        return results
    finally:
        conn.close()


# ── On-the-fly thumbnails (cached) for the results grid ───
THUMB_DIR = Path(__file__).parent / ".thumbs"


@app.get("/api/thumb/{photo_id}")
def get_thumb(photo_id: int, size: int = Query(240, le=512)):
    conn = get_conn()
    try:
        row = conn.execute("SELECT path FROM photos WHERE id = ?", (photo_id,)).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Photo not found")
        src = Path(row["path"])
    finally:
        conn.close()
    if not src.exists():
        raise HTTPException(status_code=410, detail="Photo file missing")

    THUMB_DIR.mkdir(exist_ok=True)
    # Key the cache by file PATH, not photo_id: ids are reassigned when the DB is
    # rebuilt, so a photo_id-keyed cache would serve a stale thumbnail for a file
    # that now lives under a different id. (mtime check below still catches edits.)
    import hashlib
    key = hashlib.md5(str(src).encode("utf-8")).hexdigest()[:16]
    cache = THUMB_DIR / f"{key}_{size}.jpg"
    if cache.exists() and cache.stat().st_mtime >= src.stat().st_mtime:
        return FileResponse(cache, media_type="image/jpeg", headers=NO_CACHE)

    from PIL import Image, ImageOps
    try:
        import pillow_heif
        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    with Image.open(src) as img:
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail((size, size), Image.LANCZOS)
        img.save(cache, format="JPEG", quality=82)
    return FileResponse(cache, media_type="image/jpeg", headers=NO_CACHE)


# ── Welcome page: health, sources, indexing jobs ─────────
@app.get("/api/health")
def health():
    gpu = {"available": False}
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            gpu = {"available": True, "name": torch.cuda.get_device_name(0),
                   "vram_total_mb": round(total / 1048576),
                   "vram_used_mb": round((total - free) / 1048576)}
    except Exception as e:                       # noqa: BLE001
        gpu = {"available": False, "error": str(e)}

    conn = get_conn()
    try:
        jobs.ensure_sources_table(conn)
        photos = conn.execute("SELECT COUNT(*) FROM photos "
                              "WHERE processed_at IS NOT NULL AND error IS NULL").fetchone()[0]
        detections = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
        classes = conn.execute("SELECT COUNT(DISTINCT class) FROM detections").fetchone()[0]
        n_sources = conn.execute("SELECT COUNT(*) FROM sources").fetchone()[0]
        has_vec = conn.execute("SELECT 1 FROM sqlite_master "
                               "WHERE type='table' AND name='vec_photos'").fetchone()
    finally:
        conn.close()

    # vec_photos is a vec0 virtual table → needs a connection with the
    # sqlite-vec extension loaded (a plain get_conn() raises "no such module: vec0").
    embedded = 0
    if has_vec:
        try:
            vconn = vec_conn()
            try:
                embedded = vconn.execute("SELECT COUNT(*) FROM vec_photos").fetchone()[0]
            finally:
                vconn.close()
        except Exception:                            # noqa: BLE001
            embedded = 0

    model = {"path": MODEL_PATH, "loaded": _EMBEDDER is not None,
             "dim": getattr(_EMBEDDER, "dim", None) if _EMBEDDER is not None else None}
    return {
        "gpu": gpu,
        "model": model,
        "db": {"photos": photos, "embedded": embedded, "detections": detections,
               "classes": classes, "sources": n_sources},
        "coverage": round(embedded / photos, 3) if photos else 0.0,
        "job": jobs.current_job(),
    }


@app.get("/api/sources")
def sources():
    conn = get_conn()
    try:
        return jobs.list_sources(conn)
    finally:
        conn.close()


@app.post("/api/index")
def start_index(payload: dict = Body(...)):
    path = (payload or {}).get("path", "").strip().strip('"')
    mode = (payload or {}).get("mode", "new")
    if not path:
        raise HTTPException(status_code=400, detail="path required")
    p = Path(path)
    if not p.exists() or not p.is_dir():
        raise HTTPException(status_code=400, detail=f"not a directory: {path}")
    job = jobs.start_index(p, DB_PATH, YOLOE_PATH, get_embedder,
                           mode=("reindex-full" if mode == "full" else "new"))
    if job is None:
        raise HTTPException(status_code=409, detail="an index job is already running")
    return job


@app.get("/api/job")
def job():
    return jobs.current_job() or {"state": "idle"}


@app.post("/api/job/cancel")
def cancel_job():
    return {"cancelled": jobs.request_cancel()}


@app.get("/api/browse-folder")
async def browse_folder():
    """Open a native folder picker on the SERVER machine and return the chosen
    absolute path. Convenience for the local-desktop case (server == client).
    Runs on the event-loop thread so tkinter stays on the process main thread;
    the modal briefly blocks other requests, which is fine for a single user.
    """
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        path = filedialog.askdirectory(title="選擇要索引的資料夾")
        root.destroy()
        return {"path": path or None}
    except Exception as e:                       # noqa: BLE001
        raise HTTPException(status_code=500,
                            detail=f"無法開啟資料夾對話框:{e}")


def ensure_default_source():
    """Register the project-local default-image folder as a source, so it always
    shows up on the welcome page list — even before it has been indexed."""
    try:
        DEFAULT_DIR.mkdir(exist_ok=True)
    except Exception as e:                       # noqa: BLE001
        print(f"WARNING: could not create {DEFAULT_DIR}: {e}")
        return
    conn = get_conn()
    try:
        jobs.ensure_sources_table(conn)
        p = str(DEFAULT_DIR.resolve())
        if not conn.execute("SELECT 1 FROM sources WHERE path = ?", (p,)).fetchone():
            cnt = conn.execute("SELECT COUNT(*) FROM photos WHERE path LIKE ?",
                               (p + "%",)).fetchone()[0]
            conn.execute(
                "INSERT INTO sources(path, added_at, last_indexed_at, photo_count) "
                "VALUES (?, ?, ?, ?)",
                (p, datetime.now().isoformat(timespec="seconds"), None, cnt))
            conn.commit()
    finally:
        conn.close()


# Static files served at root (/, /app.js, /style.css)
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("photos.db"))
    ap.add_argument("--model", default=str(Path(__file__).resolve().parent.parent.parent
                                           / "models" / "siglip2-base"),
                    help="SigLIP model path/name for semantic search")
    ap.add_argument("--host", default="127.0.0.1",
                    help="Bind address (use 0.0.0.0 for LAN)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    if not args.db.exists():
        print(f"DB not found: {args.db}")
        return

    global DB_PATH, MODEL_PATH
    DB_PATH = args.db.resolve()
    MODEL_PATH = args.model
    print(f"DB: {DB_PATH}")
    print(f"Search model: {MODEL_PATH}")
    try:
        print("Loading SigLIP model ...")
        get_embedder()
        print(f"Model ready (dim={_EMBEDDER.dim}).")
    except Exception as e:                       # noqa: BLE001
        print(f"WARNING: model not loaded — search/indexing will fail: {e}")

    ensure_default_source()
    print(f"Default source folder: {DEFAULT_DIR}")
    print(f"Open http://{args.host}:{args.port}/ in your browser")
    print(f"Logs (rotating, 3 MB cap): {LOG_FILE}")

    import copy
    import uvicorn
    from uvicorn.config import LOGGING_CONFIG

    # Keep uvicorn's console output, but also tee logs to a rotating file.
    # 1 MB per file + 2 backups = 3 MB total on disk, never more.
    log_config = copy.deepcopy(LOGGING_CONFIG)
    log_config["formatters"]["file"] = {
        "()": "logging.Formatter",
        "fmt": "%(asctime)s %(levelname)s %(name)s %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }
    log_config["handlers"]["file"] = {
        "class": "logging.handlers.RotatingFileHandler",
        "filename": str(LOG_FILE),
        "maxBytes": 1024 * 1024,   # 1 MB per file
        "backupCount": 2,          # + 2 rotated backups → 3 MB total cap
        "encoding": "utf-8",
        "formatter": "file",
    }
    # uvicorn.error propagates up to "uvicorn", so the file handler on these two
    # captures startup, errors, and access logs.
    for lname in ("uvicorn", "uvicorn.access"):
        log_config["loggers"][lname]["handlers"].append("file")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info",
                log_config=log_config)


if __name__ == "__main__":
    main()

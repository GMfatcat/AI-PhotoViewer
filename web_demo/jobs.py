"""Background indexing jobs for the web UI.

One job at a time = index a folder = scan (YOLOE detection) then embed (SigLIP).
Runs in a daemon thread so the FastAPI event loop stays responsive; the SigLIP
embedder is shared with search (the model is thread-safe via its own lock).
"""

import sys
import threading
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import scan    # noqa: E402  (YOLOE detection helpers + schema)
import embed   # noqa: E402  (SigLIP embedder + sqlite-vec helpers)

_job = None                       # the current/last job (a plain dict)
_start_lock = threading.Lock()
_yoloe = None                     # cached YOLOE model (loaded on first index)


class _Cancelled(Exception):
    pass


def current_job():
    """Public snapshot of the current job (or None)."""
    if _job is None:
        return None
    return {k: _job[k] for k in
            ("path", "mode", "state", "phase", "done", "total", "message",
             "error", "started_at")}


def request_cancel():
    j = _job
    if j and j["state"] == "running":
        j["cancel"] = True
        return True
    return False


def ensure_sources_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            path            TEXT PRIMARY KEY,
            added_at        TIMESTAMP,
            last_indexed_at TIMESTAMP,
            photo_count     INTEGER
        )
    """)
    conn.commit()


def list_sources(conn):
    ensure_sources_table(conn)
    rows = conn.execute("""
        SELECT path, added_at, last_indexed_at, photo_count
        FROM sources ORDER BY last_indexed_at IS NULL DESC, last_indexed_at DESC
    """).fetchall()
    return [{"path": r[0], "added_at": r[1],
             "last_indexed_at": r[2], "photo_count": r[3]} for r in rows]


def delete_photos(conn, ids):
    """Delete photo rows + their detections/masks. FK cascade isn't enabled, so
    delete manually. Caller commits. vec_photos must be cleaned separately with a
    sqlite-vec connection. Returns the number of rows targeted."""
    for pid in ids:
        conn.execute("DELETE FROM masks WHERE detection_id IN "
                     "(SELECT id FROM detections WHERE photo_id = ?)", (pid,))
        conn.execute("DELETE FROM detections WHERE photo_id = ?", (pid,))
        conn.execute("DELETE FROM photos WHERE id = ?", (pid,))
    return len(ids)


def prune_missing(conn):
    """Delete DB photos whose file no longer exists (+ their detections/masks).

    Keeps photo_count and the gallery honest after files are deleted or moved.
    vec_photos orphans are cleaned separately in the embed phase, which has the
    sqlite-vec extension loaded. Returns the number of photo rows removed.
    """
    gone = [pid for (pid, path) in conn.execute("SELECT id, path FROM photos")
            if not Path(path).exists()]
    delete_photos(conn, gone)
    conn.commit()
    return len(gone)


def start_index(path, db_path, yoloe_path, get_embedder, mode="new", device="cuda"):
    """Enqueue an index job. Returns the job dict, or None if one is already running."""
    global _job
    with _start_lock:
        if _job and _job["state"] == "running":
            return None
        _job = {
            "path": str(Path(path)), "mode": mode,
            "state": "running", "phase": "scan",
            "done": 0, "total": 0,
            "message": "啟動中…", "error": None, "cancel": False,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
    threading.Thread(
        target=_run,
        args=(str(Path(path).resolve()), str(db_path), str(yoloe_path), get_embedder, mode, device),
        daemon=True,
    ).start()
    return current_job()


def _set(**kw):
    if _job is not None:
        _job.update(kw)


def _check_cancel():
    if _job and _job["cancel"]:
        raise _Cancelled()


def _run(path, db_path, yoloe_path, get_embedder, mode, device):
    global _yoloe
    try:
        # ---------- PHASE 1: scan (YOLOE detection) ----------
        _set(phase="scan", message="尋找照片…")
        conn = scan.init_db(Path(db_path))
        conn.execute("PRAGMA busy_timeout=5000")
        ensure_sources_table(conn)

        # Drop rows for files deleted/moved since last index, so counts stay honest.
        removed = prune_missing(conn)
        if removed:
            _set(message=f"清理已移除的 {removed} 張…")

        all_photos = scan.find_photos([Path(path)], recursive=True)
        full = (mode == "reindex-full")
        if full:
            todo = [(p, scan.file_fingerprint(p)) for p in all_photos]
        else:
            _ptf, done_fps = scan.get_processed_info(conn)
            todo = [(p, scan.file_fingerprint(p)) for p in all_photos
                    if not (scan.file_fingerprint(p) in done_fps)]
        _set(total=len(todo), done=0)

        if todo:
            if _yoloe is None:
                _set(message="載入 YOLOE…")
                _yoloe = scan.YOLOE(str(yoloe_path))
                _yoloe.to(device)
            for i, (p, fp) in enumerate(todo):
                _check_cancel()
                scan.process_photo(p, fp, _yoloe, conn, 0.25, extract_masks=True)
                _set(done=i + 1, message=f"偵測中… ({i + 1}/{len(todo)})")

        # record / refresh the source folder
        cnt = conn.execute("SELECT COUNT(*) FROM photos WHERE path LIKE ?",
                            (path + "%",)).fetchone()[0]
        now = datetime.now()
        conn.execute("""
            INSERT INTO sources(path, added_at, last_indexed_at, photo_count)
            VALUES(?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                last_indexed_at = excluded.last_indexed_at,
                photo_count     = excluded.photo_count
        """, (path, now, now, cnt))
        conn.commit()
        conn.close()

        # ---------- PHASE 2: embed (SigLIP) ----------
        _check_cancel()
        _set(phase="embed", message="建立語意向量…", done=0, total=0)
        vconn = embed.open_db(Path(db_path))
        vconn.execute("PRAGMA busy_timeout=5000")
        embedder = get_embedder()
        dim = len(embedder.embed_text("dimension probe"))
        embed.ensure_vec_table(vconn, dim)

        # Drop embeddings whose photo row was pruned above (vec0 deletes by PK).
        valid = {r[0] for r in vconn.execute("SELECT id FROM photos")}
        orphans = [pid for (pid,) in vconn.execute("SELECT photo_id FROM vec_photos")
                   if pid not in valid]
        for pid in orphans:
            vconn.execute("DELETE FROM vec_photos WHERE photo_id = ?", (pid,))
        if orphans:
            vconn.commit()

        embed.build(
            vconn, embedder, 8,
            progress_cb=lambda d, t: _set(done=d, total=t, message=f"建立語意向量… ({d}/{t})"),
            cancel_cb=lambda: bool(_job and _job["cancel"]),
        )
        vconn.close()

        _check_cancel()
        _set(state="done", phase="done", message="完成 ✓")
    except _Cancelled:
        _set(state="cancelled", message="已取消")
    except Exception as e:               # noqa: BLE001
        _set(state="error", error=str(e), message=f"錯誤:{e}")

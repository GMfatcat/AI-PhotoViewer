#!/usr/bin/env python3
"""
scan.py -- Photo Archaeology scanner

Walks one or more photo directories, runs YOLOE open-vocabulary detection
on each image, and stores results in SQLite for later analysis.

Resumable: re-running skips files whose (basename + size + mtime) fingerprint
matches a previously processed row. Modified files are re-scanned automatically.
"""

import argparse
import hashlib
import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, ImageOps
from PIL.ExifTags import GPSTAGS, TAGS
from tqdm import tqdm

# HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_OK = True
except ImportError:
    HEIC_OK = False

from ultralytics import YOLOE


# Configuration
SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.heic', '.heif', '.webp', '.bmp'}
RESIZE_LONGEST = 640
DEFAULT_CONF = 0.25

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger("scan")


# Database
SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    path              TEXT UNIQUE NOT NULL,
    file_hash         TEXT,
    scan_fingerprint  TEXT,
    taken_at          TIMESTAMP,
    width             INTEGER,
    height            INTEGER,
    gps_lat           REAL,
    gps_lon           REAL,
    camera_make       TEXT,
    camera_model      TEXT,
    file_size         INTEGER,
    processed_at      TIMESTAMP,
    error             TEXT
);

CREATE TABLE IF NOT EXISTS detections (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id        INTEGER NOT NULL REFERENCES photos(id),
    class           TEXT NOT NULL,
    confidence      REAL NOT NULL,
    bbox_x          REAL,
    bbox_y          REAL,
    bbox_w          REAL,
    bbox_h          REAL,
    bbox_area_ratio REAL
);

CREATE TABLE IF NOT EXISTS masks (
    detection_id  INTEGER PRIMARY KEY REFERENCES detections(id) ON DELETE CASCADE,
    polygon_json  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_class       ON detections(class);
CREATE INDEX IF NOT EXISTS idx_photo       ON detections(photo_id);
CREATE INDEX IF NOT EXISTS idx_confidence  ON detections(confidence);
CREATE INDEX IF NOT EXISTS idx_processed   ON photos(processed_at);
CREATE INDEX IF NOT EXISTS idx_fingerprint ON photos(scan_fingerprint);
CREATE INDEX IF NOT EXISTS idx_masks_det   ON masks(detection_id);
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(SCHEMA)
    # Migration: backfill scan_fingerprint column if upgrading from old schema
    cols = {row[1] for row in conn.execute("PRAGMA table_info(photos)")}
    if 'scan_fingerprint' not in cols:
        log.info("Migrating DB: adding scan_fingerprint column")
        conn.execute("ALTER TABLE photos ADD COLUMN scan_fingerprint TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_fingerprint ON photos(scan_fingerprint)")
    conn.commit()
    return conn


# Fingerprinting (skip logic)
def file_fingerprint(path: Path) -> str:
    """
    Quick fingerprint: hash of (basename, size, mtime).
    Same file moved to a new directory still matches.
    Modified file (different mtime or size) gets a new fingerprint -> rescan.
    """
    try:
        stat = path.stat()
        s = f"{path.name}|{stat.st_size}|{int(stat.st_mtime)}"
        return hashlib.md5(s.encode()).hexdigest()
    except OSError:
        return ""


def quick_hash(path: Path) -> str:
    """Content-aware hash: size + first 64KB. Stored, not used for skip."""
    h = hashlib.md5()
    try:
        size = path.stat().st_size
        h.update(str(size).encode())
        with open(path, 'rb') as f:
            h.update(f.read(65536))
        return h.hexdigest()
    except Exception:
        return ""


# EXIF
def _to_decimal(coord, ref) -> Optional[float]:
    try:
        d, m, s = [float(x) for x in coord]
        dec = d + m / 60 + s / 3600
        if ref in ('S', 'W'):
            dec = -dec
        return dec
    except (ValueError, TypeError):
        return None


def parse_gps(gps_info) -> tuple:
    if not gps_info:
        return None, None
    try:
        gps = {GPSTAGS.get(k, k): v for k, v in gps_info.items()}
        lat = _to_decimal(gps.get('GPSLatitude'), gps.get('GPSLatitudeRef'))
        lon = _to_decimal(gps.get('GPSLongitude'), gps.get('GPSLongitudeRef'))
        return lat, lon
    except Exception:
        return None, None


def extract_exif(img: Image.Image) -> dict:
    out = {'taken_at': None, 'gps_lat': None, 'gps_lon': None,
           'camera_make': None, 'camera_model': None}
    try:
        exif_raw = img._getexif()
        if not exif_raw:
            return out
        exif = {TAGS.get(k, k): v for k, v in exif_raw.items()}
    except (AttributeError, OSError):
        return out

    out['camera_make'] = exif.get('Make')
    out['camera_model'] = exif.get('Model')
    dt = exif.get('DateTimeOriginal') or exif.get('DateTime')
    if dt:
        try:
            out['taken_at'] = datetime.strptime(dt, '%Y:%m:%d %H:%M:%S')
        except (ValueError, TypeError):
            pass
    out['gps_lat'], out['gps_lon'] = parse_gps(exif.get('GPSInfo'))
    return out


# Photo discovery
def find_photos(roots: Iterable[Path], recursive: bool = True) -> list[Path]:
    """Walk one or more roots, return sorted unique image paths."""
    seen = set()
    for root in roots:
        if not root.exists():
            log.warning(f"Directory not found: {root}")
            continue
        if not root.is_dir():
            log.warning(f"Not a directory: {root}")
            continue
        iterator = root.rglob('*') if recursive else root.iterdir()
        for p in iterator:
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
                try:
                    seen.add(p.resolve())
                except OSError:
                    seen.add(p)
    return sorted(seen)


def get_processed_info(conn: sqlite3.Connection) -> tuple[dict, set]:
    """
    Return (path -> stored_fingerprint, set of all stored fingerprints).
    Stored fingerprint may be None for legacy rows (will be backfilled).
    """
    path_to_fp = {}
    fps = set()
    for path, fp in conn.execute(
        "SELECT path, scan_fingerprint FROM photos WHERE processed_at IS NOT NULL"
    ):
        path_to_fp[path] = fp
        if fp:
            fps.add(fp)
    return path_to_fp, fps


def serialize_polygon(xyn_array) -> str:
    """Convert YOLO normalized polygon (Nx2 array) to compact JSON string."""
    try:
        pts = [[round(float(x), 4), round(float(y), 4)] for x, y in xyn_array]
        if len(pts) < 3:
            return ""
        return json.dumps(pts, separators=(',', ':'))
    except Exception:
        return ""


# Detection
def process_photo(photo_path: Path, fingerprint: str, model,
                  conn: sqlite3.Connection, conf_threshold: float,
                  extract_masks: bool = True) -> bool:
    try:
        with Image.open(photo_path) as img:
            try:
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
            width, height = img.size
            meta = extract_exif(img)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if RESIZE_LONGEST and max(width, height) > RESIZE_LONGEST:
                scale = RESIZE_LONGEST / max(width, height)
                img_small = img.resize((int(width * scale), int(height * scale)), Image.LANCZOS)
            else:
                img_small = img.copy()
            img_small.load()

        results = model.predict(img_small, conf=conf_threshold, verbose=False)

        conn.execute("""
            INSERT INTO photos
            (path, file_hash, scan_fingerprint, taken_at, width, height,
             gps_lat, gps_lon, camera_make, camera_model, file_size, processed_at, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
            ON CONFLICT(path) DO UPDATE SET
                file_hash        = excluded.file_hash,
                scan_fingerprint = excluded.scan_fingerprint,
                taken_at         = excluded.taken_at,
                width            = excluded.width,
                height           = excluded.height,
                gps_lat          = excluded.gps_lat,
                gps_lon          = excluded.gps_lon,
                camera_make      = excluded.camera_make,
                camera_model     = excluded.camera_model,
                file_size        = excluded.file_size,
                processed_at     = excluded.processed_at,
                error            = NULL
        """, (
            str(photo_path), quick_hash(photo_path), fingerprint, meta['taken_at'],
            width, height, meta['gps_lat'], meta['gps_lon'],
            meta['camera_make'], meta['camera_model'],
            photo_path.stat().st_size, datetime.now(),
        ))
        photo_id = conn.execute("SELECT id FROM photos WHERE path = ?",
                                (str(photo_path),)).fetchone()[0]

        # Clear previous detections + their masks
        conn.execute("""
            DELETE FROM masks WHERE detection_id IN
            (SELECT id FROM detections WHERE photo_id = ?)
        """, (photo_id,))
        conn.execute("DELETE FROM detections WHERE photo_id = ?", (photo_id,))

        # Insert detections one at a time so we can capture IDs for mask linking
        if results:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                names = r.names
                boxes = r.boxes

                # Mask polygons (only if seg model and extract_masks is True)
                polygons = [None] * len(boxes)
                if extract_masks and getattr(r, 'masks', None) is not None:
                    masks_obj = r.masks
                    if hasattr(masks_obj, 'xyn') and masks_obj.xyn is not None:
                        for i, poly in enumerate(masks_obj.xyn):
                            if i < len(polygons):
                                polygons[i] = serialize_polygon(poly)

                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i])
                    cls_name = (names.get(cls_id, str(cls_id)) if isinstance(names, dict)
                                else names[cls_id])
                    conf = float(boxes.conf[i])
                    x, y, w, h = boxes.xywhn[i].tolist()

                    cur = conn.execute("""
                        INSERT INTO detections
                        (photo_id, class, confidence, bbox_x, bbox_y, bbox_w, bbox_h, bbox_area_ratio)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (photo_id, cls_name, conf, x, y, w, h, w * h))
                    det_id = cur.lastrowid

                    if polygons[i]:
                        conn.execute("""
                            INSERT INTO masks (detection_id, polygon_json) VALUES (?, ?)
                        """, (det_id, polygons[i]))

        conn.commit()
        return True

    except Exception as e:
        log.warning(f"Failed {photo_path.name}: {e}")
        try:
            conn.execute("""
                INSERT INTO photos (path, scan_fingerprint, processed_at, error)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    scan_fingerprint = excluded.scan_fingerprint,
                    processed_at     = excluded.processed_at,
                    error            = excluded.error
            """, (str(photo_path), fingerprint, datetime.now(), str(e)[:500]))
            conn.commit()
        except Exception:
            pass
        return False


# Main
def main():
    ap = argparse.ArgumentParser(
        description="Photo archaeology scanner -- YOLOE open-vocabulary detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("photo_dirs", type=Path, nargs='+',
                    help="One or more root directories of photos")
    ap.add_argument("--no-recursive", action="store_true",
                    help="Don't scan subdirectories (default: recursive)")
    ap.add_argument("--db", type=Path, default=Path("photos.db"))
    ap.add_argument("--model", default="yoloe-11s-seg-pf.pt",
                    help="YOLOE weights (use '-pf' suffix for prompt-free mode)")
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF,
                    help="Confidence threshold for stored detections")
    ap.add_argument("--device", default="cuda", help="cuda or cpu")
    ap.add_argument("--rescan", action="store_true",
                    help="Reprocess all photos (ignore fingerprint cache)")
    ap.add_argument("--no-masks", action="store_true",
                    help="Skip extracting segmentation masks (faster, smaller DB)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N photos (for testing)")
    args = ap.parse_args()

    if not HEIC_OK:
        log.warning("pillow-heif not installed -- HEIC files will be skipped")

    log.info(f"DB: {args.db}")
    conn = init_db(args.db)

    recursive = not args.no_recursive
    log.info(f"Scanning {len(args.photo_dirs)} root(s) "
             f"({'recursive' if recursive else 'top-level only'}) ...")
    for d in args.photo_dirs:
        log.info(f"  - {d}")

    all_photos = find_photos(args.photo_dirs, recursive=recursive)
    log.info(f"Found {len(all_photos)} photos total")

    # Build to-do list using fingerprint-based skip
    if args.rescan:
        log.info("--rescan: will reprocess all photos")
        to_do = [(p, file_fingerprint(p)) for p in all_photos]
    else:
        log.info("Checking fingerprints against DB...")
        path_to_fp, done_fps = get_processed_info(conn)
        backfill_count = 0
        skip_by_fp = 0
        skip_by_legacy_path = 0
        to_do = []
        for p in all_photos:
            fp = file_fingerprint(p)
            path_str = str(p)
            if fp and fp in done_fps:
                skip_by_fp += 1
                continue
            # Legacy row: same path in DB but no fingerprint stored yet
            if path_str in path_to_fp and path_to_fp[path_str] is None and fp:
                conn.execute("UPDATE photos SET scan_fingerprint = ? WHERE path = ?",
                             (fp, path_str))
                done_fps.add(fp)
                backfill_count += 1
                skip_by_legacy_path += 1
                continue
            to_do.append((p, fp))
        if backfill_count:
            conn.commit()
            log.info(f"Backfilled {backfill_count} legacy fingerprints")
        log.info(f"Skipping {skip_by_fp + skip_by_legacy_path} already-scanned "
                 f"({skip_by_fp} by fingerprint, {skip_by_legacy_path} legacy)")
        log.info(f"Remaining to process: {len(to_do)}")

    if args.limit:
        to_do = to_do[: args.limit]
        log.info(f"--limit: processing first {len(to_do)}")

    if not to_do:
        log.info("Nothing to do.")
        conn.close()
        return

    log.info(f"Loading model: {args.model}")
    model = YOLOE(args.model)
    if args.device:
        try:
            model.to(args.device)
        except Exception as e:
            log.warning(f"Could not move model to {args.device}: {e}")

    extract_masks = not args.no_masks
    if not extract_masks:
        log.info("--no-masks: segmentation masks will NOT be stored")

    ok, fail = 0, 0
    with tqdm(total=len(to_do), desc="Scanning", unit="img") as pbar:
        for p, fp in to_do:
            if process_photo(p, fp, model, conn, args.conf, extract_masks=extract_masks):
                ok += 1
            else:
                fail += 1
            pbar.update(1)
            pbar.set_postfix(ok=ok, fail=fail)

    log.info(f"Done. Success: {ok} | Failed: {fail}")

    n_det = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    n_cls = conn.execute("SELECT COUNT(DISTINCT class) FROM detections").fetchone()[0]
    n_mask = conn.execute("SELECT COUNT(*) FROM masks").fetchone()[0]
    log.info(f"Total detections in DB: {n_det} | With masks: {n_mask} | Unique classes: {n_cls}")
    log.info("Next: python inspect_db.py")

    conn.close()


if __name__ == "__main__":
    main()

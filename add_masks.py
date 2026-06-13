#!/usr/bin/env python3
"""
add_masks.py -- Backfill segmentation masks for existing detections.

For DBs that were scanned without masks (--no-masks), or partially.
Re-runs YOLOE-seg on each photo and matches new mask polygons to existing
detections by class + IoU. Original bbox detections are NOT modified.

If matching fails for some detections, they are reported but left alone.
"""

import argparse
import json
import logging
import sqlite3
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

from ultralytics import YOLOE

RESIZE_LONGEST = 640

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger("add_masks")


def serialize_polygon(xyn_array) -> str:
    try:
        pts = [[round(float(x), 4), round(float(y), 4)] for x, y in xyn_array]
        if len(pts) < 3:
            return ""
        return json.dumps(pts, separators=(',', ':'))
    except Exception:
        return ""


def bbox_iou(b1, b2) -> float:
    """IoU of two bboxes in (cx, cy, w, h) normalized form."""
    x1a, y1a = b1[0] - b1[2] / 2, b1[1] - b1[3] / 2
    x2a, y2a = b1[0] + b1[2] / 2, b1[1] + b1[3] / 2
    x1b, y1b = b2[0] - b2[2] / 2, b2[1] - b2[3] / 2
    x2b, y2b = b2[0] + b2[2] / 2, b2[1] + b2[3] / 2

    ix1, iy1 = max(x1a, x1b), max(y1a, y1b)
    ix2, iy2 = min(x2a, x2b), min(y2a, y2b)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    a1 = (x2a - x1a) * (y2a - y1a)
    a2 = (x2b - x1b) * (y2b - y1b)
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def match_and_insert(conn, photo_id: int, old_dets: list, new_results,
                     min_iou: float = 0.5) -> tuple[int, int]:
    """
    Match new mask polygons to old detections by class + IoU.
    old_dets: list of (det_id, class, bbox_x, bbox_y, bbox_w, bbox_h) with no mask yet
    Returns (matched, unmatched_old_count).
    """
    if new_results is None or len(new_results) == 0:
        return 0, len(old_dets)

    r = new_results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return 0, len(old_dets)
    if getattr(r, 'masks', None) is None or r.masks.xyn is None:
        return 0, len(old_dets)

    names = r.names
    new_dets = []
    for i in range(len(r.boxes)):
        cls_id = int(r.boxes.cls[i])
        cls = (names.get(cls_id, str(cls_id)) if isinstance(names, dict)
               else names[cls_id])
        x, y, w, h = r.boxes.xywhn[i].tolist()
        poly = serialize_polygon(r.masks.xyn[i])
        new_dets.append((cls, (x, y, w, h), poly))

    # Greedy matching by (class, IoU)
    matched_old = set()
    matched_new = set()
    candidates = []
    for old_idx, (det_id, cls_old, *bbox_old) in enumerate(old_dets):
        for new_idx, (cls_new, bbox_new, poly) in enumerate(new_dets):
            if cls_old != cls_new or not poly:
                continue
            iou = bbox_iou(tuple(bbox_old), bbox_new)
            if iou >= min_iou:
                candidates.append((iou, old_idx, new_idx))
    candidates.sort(reverse=True)

    matched_count = 0
    for iou, old_idx, new_idx in candidates:
        if old_idx in matched_old or new_idx in matched_new:
            continue
        det_id = old_dets[old_idx][0]
        poly = new_dets[new_idx][2]
        conn.execute("""
            INSERT OR REPLACE INTO masks (detection_id, polygon_json) VALUES (?, ?)
        """, (det_id, poly))
        matched_old.add(old_idx)
        matched_new.add(new_idx)
        matched_count += 1

    return matched_count, len(old_dets) - matched_count


def main():
    ap = argparse.ArgumentParser(
        description="Backfill segmentation masks for detections that don't have them",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--db", type=Path, default=Path("photos.db"))
    ap.add_argument("--model", default="yoloe-11s-seg-pf.pt",
                    help="Must be a -seg variant")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Confidence threshold (use same as original scan)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--min-iou", type=float, default=0.5,
                    help="Min IoU for matching new mask to existing detection")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process at most N photos")
    args = ap.parse_args()

    if not args.db.exists():
        log.error(f"DB not found: {args.db}")
        return

    conn = sqlite3.connect(args.db)

    # Find photos that have at least one detection without a mask
    rows = conn.execute("""
        SELECT DISTINCT p.id, p.path
        FROM photos p
        JOIN detections d ON d.photo_id = p.id
        LEFT JOIN masks m ON m.detection_id = d.id
        WHERE m.detection_id IS NULL
          AND p.processed_at IS NOT NULL
          AND p.error IS NULL
        ORDER BY p.id
    """).fetchall()

    if not rows:
        log.info("No photos with missing masks. Nothing to do.")
        conn.close()
        return

    if args.limit:
        rows = rows[: args.limit]

    log.info(f"Photos needing mask backfill: {len(rows)}")
    log.info(f"Loading model: {args.model}")
    model = YOLOE(args.model)
    try:
        model.to(args.device)
    except Exception as e:
        log.warning(f"Could not move model to {args.device}: {e}")

    total_matched, total_unmatched = 0, 0
    skipped_missing_file = 0

    with tqdm(total=len(rows), desc="Backfilling", unit="img") as pbar:
        for photo_id, path in rows:
            photo_path = Path(path)
            if not photo_path.exists():
                skipped_missing_file += 1
                pbar.update(1)
                continue

            old_dets = conn.execute("""
                SELECT d.id, d.class, d.bbox_x, d.bbox_y, d.bbox_w, d.bbox_h
                FROM detections d
                LEFT JOIN masks m ON m.detection_id = d.id
                WHERE d.photo_id = ? AND m.detection_id IS NULL
                  AND d.confidence >= ?
            """, (photo_id, args.conf)).fetchall()
            if not old_dets:
                pbar.update(1)
                continue

            try:
                with Image.open(photo_path) as img:
                    try:
                        img = ImageOps.exif_transpose(img)
                    except Exception:
                        pass
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    W, H = img.size
                    if RESIZE_LONGEST and max(W, H) > RESIZE_LONGEST:
                        scale = RESIZE_LONGEST / max(W, H)
                        img = img.resize((int(W * scale), int(H * scale)), Image.LANCZOS)
                    img.load()

                results = model.predict(img, conf=args.conf, verbose=False)
                m, u = match_and_insert(conn, photo_id, old_dets, results,
                                        min_iou=args.min_iou)
                total_matched += m
                total_unmatched += u
                conn.commit()

            except Exception as e:
                log.warning(f"Failed {photo_path.name}: {e}")

            pbar.update(1)
            pbar.set_postfix(matched=total_matched, unmatched=total_unmatched)

    log.info(f"Done. Masks added: {total_matched} | Unmatched detections: {total_unmatched}")
    if skipped_missing_file:
        log.info(f"Photos with missing files: {skipped_missing_file}")
    if total_unmatched > 0:
        log.info("Unmatched detections probably need --rescan in scan.py to get masks "
                 "(model output drifted from original scan).")
    conn.close()


if __name__ == "__main__":
    main()

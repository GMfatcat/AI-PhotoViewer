#!/usr/bin/env python3
"""
inspect_db.py -- Quick sanity check after scanning.

Shows overview stats, top N classes, busiest photos (with bbox visualizations),
rare class finds (with original paths), and N-way co-occurrence.
"""

import argparse
import hashlib
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass


def class_color(cls: str) -> tuple:
    h = int(hashlib.md5(cls.encode()).hexdigest(), 16)
    r = 60 + (h & 0xFF) * 3 // 4
    g = 60 + ((h >> 8) & 0xFF) * 3 // 4
    b = 60 + ((h >> 16) & 0xFF) * 3 // 4
    return (min(r, 255), min(g, 255), min(b, 255))


def load_font(size: int):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf",
                 "Arial.ttf", "arial.ttf",
                 "/System/Library/Fonts/Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size=size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def draw_detections(photo_path: Path, detections: list, output_path: Path) -> bool:
    try:
        with Image.open(photo_path) as img:
            try:
                img = ImageOps.exif_transpose(img)
            except Exception:
                pass
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img = img.copy()
    except Exception as e:
        print(f"    [skip] cannot open {photo_path.name}: {e}")
        return False

    W, H = img.size
    draw = ImageDraw.Draw(img)
    font_size = max(14, H // 60)
    font = load_font(font_size)

    for cls, conf, bx, by, bw, bh in detections:
        x1 = max(0, int((bx - bw / 2) * W))
        y1 = max(0, int((by - bh / 2) * H))
        x2 = min(W, int((bx + bw / 2) * W))
        y2 = min(H, int((by + bh / 2) * H))

        color = class_color(cls)
        line_w = max(2, H // 400)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

        label = f"{cls} {conf:.2f}"
        try:
            tb = draw.textbbox((0, 0), label, font=font)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
        except AttributeError:
            tw, th = font.getsize(label)

        ty = y1 - th - 4
        if ty < 0:
            ty = y1 + 2
        draw.rectangle([x1, ty, x1 + tw + 6, ty + th + 4], fill=color)
        draw.text((x1 + 3, ty + 2), label, fill="white", font=font)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, quality=85)
    return True


def compute_cooccurrence(conn, min_conf: float, n: int) -> Counter:
    """
    Build N-way co-occurrence counts.
    For each photo, take its set of distinct classes (above conf threshold).
    For each photo with >= N classes, count every size-N combination once.
    """
    photo_classes = defaultdict(set)
    for photo_id, cls in conn.execute("""
        SELECT photo_id, class FROM detections WHERE confidence >= ?
    """, (min_conf,)):
        photo_classes[photo_id].add(cls)

    counts = Counter()
    for classes in photo_classes.values():
        if len(classes) >= n:
            for combo in combinations(sorted(classes), n):
                counts[combo] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=Path("photos.db"))
    ap.add_argument("--top", type=int, default=50, help="Show top N classes")
    ap.add_argument("--min-conf", type=float, default=0.4,
                    help="Min confidence for all stats")
    ap.add_argument("--busy-top", type=int, default=5,
                    help="How many busiest photos to visualize")
    ap.add_argument("--no-busy-images", action="store_true",
                    help="Skip generating bbox-overlay images")
    ap.add_argument("--busy-dir", type=Path, default=Path("busy"),
                    help="Output dir for busy photo visualizations")
    ap.add_argument("--rare-top", type=int, default=10,
                    help="Show top N rare classes")
    ap.add_argument("--cooccur-n", type=int, default=2,
                    help="N-way co-occurrence (2=pairs, 3=triples...)")
    ap.add_argument("--cooccur-top", type=int, default=10,
                    help="Show top N co-occurring groups")
    ap.add_argument("--cooccur-min", type=int, default=1,
                    help="Min photos a combo must appear in to be shown")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"DB not found: {args.db}")
        return

    conn = sqlite3.connect(args.db)

    # Overview
    n_photos = conn.execute(
        "SELECT COUNT(*) FROM photos WHERE processed_at IS NOT NULL"
    ).fetchone()[0]
    n_failed = conn.execute(
        "SELECT COUNT(*) FROM photos WHERE error IS NOT NULL"
    ).fetchone()[0]
    n_det = conn.execute(
        "SELECT COUNT(*) FROM detections WHERE confidence >= ?", (args.min_conf,)
    ).fetchone()[0]
    n_cls = conn.execute(
        "SELECT COUNT(DISTINCT class) FROM detections WHERE confidence >= ?",
        (args.min_conf,)
    ).fetchone()[0]
    n_with_gps = conn.execute(
        "SELECT COUNT(*) FROM photos WHERE gps_lat IS NOT NULL"
    ).fetchone()[0]
    n_with_time = conn.execute(
        "SELECT COUNT(*) FROM photos WHERE taken_at IS NOT NULL"
    ).fetchone()[0]

    print()
    print("-- Overview " + "-" * 50)
    print(f"  Photos processed:           {n_photos:>8,}")
    print(f"  Photos failed:              {n_failed:>8,}")
    print(f"  Photos with timestamp:      {n_with_time:>8,}")
    print(f"  Photos with GPS:            {n_with_gps:>8,}")
    print(f"  Detections (conf >= {args.min_conf}):   {n_det:>8,}")
    print(f"  Unique classes:             {n_cls:>8,}")
    if n_photos > 0:
        print(f"  Avg detections per photo:   {n_det / n_photos:>8.2f}")

    date_range = conn.execute(
        "SELECT MIN(taken_at), MAX(taken_at) FROM photos WHERE taken_at IS NOT NULL"
    ).fetchone()
    if date_range[0]:
        print(f"  Date range:                 {date_range[0]} -> {date_range[1]}")

    # Top classes
    print()
    print(f"-- Top {args.top} classes " + "-" * max(1, 62 - len(str(args.top))))
    rows = conn.execute("""
        SELECT class, COUNT(*) AS instances, COUNT(DISTINCT photo_id) AS photos,
               AVG(confidence) AS avg_conf
        FROM detections
        WHERE confidence >= ?
        GROUP BY class
        ORDER BY instances DESC
        LIMIT ?
    """, (args.min_conf, args.top)).fetchall()

    for i, (cls, instances, photos, avg_conf) in enumerate(rows, 1):
        print(f"  {i:>3}. {cls:<32} {instances:>6} instances / {photos:>5} photos "
              f"(avg conf {avg_conf:.2f})")

    # Busiest photos
    print()
    print(f"-- Top {args.busy_top} 'busiest' photos (most objects) "
          + "-" * max(1, 35 - len(str(args.busy_top))))
    busy = conn.execute("""
        SELECT p.id, p.path, COUNT(d.id) AS n
        FROM photos p
        JOIN detections d ON d.photo_id = p.id
        WHERE d.confidence >= ?
        GROUP BY p.id
        ORDER BY n DESC
        LIMIT ?
    """, (args.min_conf, args.busy_top)).fetchall()

    for photo_id, path, n in busy:
        print(f"  {n:>3} objects: {Path(path).name}")

    if busy and not args.no_busy_images:
        run_stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        out_dir = args.busy_dir / run_stamp
        print(f"\n  Writing annotated images -> {out_dir}/")
        for rank, (photo_id, path, n) in enumerate(busy, 1):
            photo_path = Path(path)
            if not photo_path.exists():
                print(f"    [skip] not found: {photo_path}")
                continue
            dets = conn.execute("""
                SELECT class, confidence, bbox_x, bbox_y, bbox_w, bbox_h
                FROM detections
                WHERE photo_id = ? AND confidence >= ?
                ORDER BY confidence DESC
            """, (photo_id, args.min_conf)).fetchall()
            out_name = f"{rank:02d}_{n:03d}obj_{photo_path.stem}.jpg"
            if draw_detections(photo_path, dets, out_dir / out_name):
                print(f"    {out_name}  ({n} objects)")

    # N-way co-occurrence
    print()
    print(f"-- {args.cooccur_n}-way co-occurrence "
          f"(top {args.cooccur_top}, min {args.cooccur_min} photo{'s' if args.cooccur_min > 1 else ''}) "
          + "-" * 20)

    if args.cooccur_n < 2:
        print("  (skipped -- need N >= 2)")
    else:
        cooccur = compute_cooccurrence(conn, args.min_conf, args.cooccur_n)
        filtered = [(combo, c) for combo, c in cooccur.most_common()
                    if c >= args.cooccur_min]
        if not filtered:
            print(f"  (no combinations of {args.cooccur_n} classes meet threshold)")
        else:
            for combo, count in filtered[: args.cooccur_top]:
                combo_str = " + ".join(combo)
                if len(combo_str) > 56:
                    combo_str = combo_str[:53] + "..."
                print(f"  {count:>4} photos  |  {combo_str}")
            total_unique = len(cooccur)
            shown = min(len(filtered), args.cooccur_top)
            print(f"\n  ({total_unique} unique {args.cooccur_n}-way combos found, "
                  f"showing top {shown} above threshold)")

    # Rare finds
    print()
    print(f"-- Rare finds (classes in only 1-2 photos, top {args.rare_top}) "
          + "-" * 10)
    rare = conn.execute("""
        SELECT d.class, COUNT(DISTINCT d.photo_id) AS photos,
               GROUP_CONCAT(DISTINCT p.path) AS paths
        FROM detections d
        JOIN photos p ON p.id = d.photo_id
        WHERE d.confidence >= ?
        GROUP BY d.class
        HAVING photos <= 2
        ORDER BY photos, d.class
        LIMIT ?
    """, (args.min_conf, args.rare_top)).fetchall()

    total_rare = conn.execute("""
        SELECT COUNT(*) FROM (
            SELECT class FROM detections WHERE confidence >= ?
            GROUP BY class HAVING COUNT(DISTINCT photo_id) <= 2
        )
    """, (args.min_conf,)).fetchone()[0]

    if rare:
        for cls, photos, paths_str in rare:
            print(f"  {cls}  ({photos} photo{'s' if photos > 1 else ''})")
            for p in (paths_str or "").split(","):
                if p:
                    print(f"      {p}")
        if total_rare > args.rare_top:
            print(f"\n  (... and {total_rare - args.rare_top} more rare classes; "
                  f"use --rare-top to show more)")
    else:
        print("  (none -- every class appears in 3+ photos)")

    print()
    conn.close()


if __name__ == "__main__":
    main()

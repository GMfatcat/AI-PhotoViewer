#!/usr/bin/env python3
"""
embed.py -- Semantic search layer for Photo Archaeology.

Computes SigLIP 2 (multilingual) image embeddings for photos already in
photos.db (created by scan.py) and stores them in a sqlite-vec virtual table
*inside the same photos.db*. Then natural-language queries (Chinese or English)
retrieve photos by meaning, complementing YOLOE's class-label filter.

Usage:
    # Build / update the semantic index (only embeds new photos)
    python embed.py

    # Search by natural language (zh or en)
    python embed.py --search "海邊日落的合照"
    python embed.py --search "two girls" --top 5
"""

import argparse
import logging
import sqlite3
from pathlib import Path

import sqlite_vec
import torch
from PIL import Image, ImageOps
from transformers import AutoModel, AutoProcessor

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("embed")

DEFAULT_MODEL = "google/siglip2-so400m-patch14-384"


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def ensure_vec_table(conn: sqlite3.Connection, dim: int) -> None:
    conn.execute(f"""
        CREATE VIRTUAL TABLE IF NOT EXISTS vec_photos USING vec0(
            photo_id INTEGER PRIMARY KEY,
            embedding FLOAT[{dim}] distance_metric=cosine
        )
    """)
    conn.commit()


class Embedder:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        log.info(f"Loading {model_name} on {self.device} ...")
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name)

    @staticmethod
    def _pool(out):
        # transformers SigLIP2 returns BaseModelOutputWithPooling; the embedding
        # used for image-text similarity is pooler_output. Fall back to the raw
        # tensor for model versions that already return one.
        return out.pooler_output if hasattr(out, "pooler_output") else out

    @torch.inference_mode()
    def embed_images(self, images: list) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        feats = self._pool(self.model.get_image_features(**inputs))
        return torch.nn.functional.normalize(feats, p=2, dim=-1).cpu()

    @torch.inference_mode()
    def embed_text(self, text: str) -> list:
        # SigLIP text encoder requires max_length padding
        inputs = self.processor(text=[text], padding="max_length",
                                return_tensors="pt").to(self.device)
        feats = self._pool(self.model.get_text_features(**inputs))
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats[0].cpu().tolist()


def load_image(path: Path):
    with Image.open(path) as img:
        try:
            img = ImageOps.exif_transpose(img)
        except Exception:
            pass
        return img.convert("RGB")


def build(conn, embedder, batch_size: int):
    # Photos that are scanned OK, file still exists, and not yet embedded
    rows = conn.execute("""
        SELECT p.id, p.path FROM photos p
        WHERE p.processed_at IS NOT NULL AND p.error IS NULL
          AND p.id NOT IN (SELECT photo_id FROM vec_photos)
        ORDER BY p.id
    """).fetchall()

    todo = [(pid, Path(path)) for pid, path in rows if Path(path).exists()]
    missing = len(rows) - len(todo)
    if missing:
        log.warning(f"{missing} photo(s) skipped (file not found on disk)")
    if not todo:
        log.info("Nothing to embed -- index is up to date.")
        return

    log.info(f"Embedding {len(todo)} photo(s) (batch={batch_size}) ...")
    done = 0
    for i in range(0, len(todo), batch_size):
        chunk = todo[i:i + batch_size]
        imgs, ids = [], []
        for pid, path in chunk:
            try:
                imgs.append(load_image(path))
                ids.append(pid)
            except Exception as e:
                log.warning(f"skip {path.name}: {e}")
        if not imgs:
            continue
        vecs = embedder.embed_images(imgs)
        for pid, vec in zip(ids, vecs):
            conn.execute("INSERT OR REPLACE INTO vec_photos(photo_id, embedding) VALUES (?, ?)",
                         (pid, sqlite_vec.serialize_float32(vec.tolist())))
        conn.commit()
        done += len(ids)
        log.info(f"  {done}/{len(todo)}")
    log.info(f"Done. Embedded {done} photo(s).")


def search(conn, embedder, query: str, top: int):
    qvec = embedder.embed_text(query)
    rows = conn.execute("""
        SELECT v.photo_id, v.distance, p.path
        FROM vec_photos v
        JOIN photos p ON p.id = v.photo_id
        WHERE v.embedding MATCH ? AND k = ?
        ORDER BY v.distance
    """, (sqlite_vec.serialize_float32(qvec), top)).fetchall()

    print(f'\n  Query: "{query}"  (top {top})')
    print("  " + "-" * 60)
    for pid, dist, path in rows:
        sim = 1.0 - dist  # cosine distance -> similarity
        classes = [r[0] for r in conn.execute("""
            SELECT DISTINCT class FROM detections
            WHERE photo_id = ? AND confidence >= 0.4
            ORDER BY confidence DESC LIMIT 5
        """, (pid,)).fetchall()]
        cls_str = ", ".join(classes) if classes else "-"
        print(f"  sim {sim:.3f} | {Path(path).name}")
        print(f"            yolo: {cls_str}")
    print()


def main():
    ap = argparse.ArgumentParser(description="Semantic embedding + search for photos.db")
    ap.add_argument("--db", type=Path,
                    default=Path(__file__).parent / "photos.db")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--search", type=str, default=None,
                    help="Run a natural-language search instead of building")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--rebuild", action="store_true",
                    help="Drop & recreate the vector index (needed when switching models / dims)")
    args = ap.parse_args()

    if not args.db.exists():
        log.error(f"DB not found: {args.db} (run scan.py first)")
        return

    conn = open_db(args.db)
    embedder = Embedder(args.model, device=args.device)

    # Determine embedding dim from the model and ensure the vec table exists
    dim = embedder.embed_text("dimension probe").__len__()
    if args.rebuild and args.search is None:
        log.info("--rebuild: dropping existing vec_photos table")
        conn.execute("DROP TABLE IF EXISTS vec_photos")
        conn.commit()
    ensure_vec_table(conn, dim)

    if args.search is not None:
        search(conn, embedder, args.search, args.top)
    else:
        build(conn, embedder, args.batch_size)

    conn.close()


if __name__ == "__main__":
    main()

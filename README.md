# Photo Archaeology — Scanner

A tool to scan your photo library with **YOLOE open-vocabulary detection**
and store results in SQLite for later analysis.

Built for the "what's actually in my last few years of photos" question —
scan once, query forever.

## Setup

```bash
# Recommended: virtual env
python -m venv venv
source venv/bin/activate              # macOS / Linux
# venv\Scripts\activate               # Windows

pip install -r requirements.txt
```

GPU is automatic if PyTorch sees CUDA. On RTX 3060 laptop expect
~100-200 photos/minute with default settings.

## Usage

```bash
# Test pipeline first with 100 photos
python scan.py /path/to/photos --limit 100

# Then run full library
python scan.py /path/to/photos

# Quick sanity check
python inspect_db.py
```

### Options

```
positional:
  photo_dir              Root directory of photos (recursive)

optional:
  --db PATH              SQLite DB path (default: photos.db)
  --model NAME           YOLOE weights (default: yoloe-11s-seg-pf.pt)
  --conf FLOAT           Confidence threshold (default: 0.25)
  --device cuda|cpu      (default: cuda)
  --rescan               Reprocess all photos
  --limit N              Process at most N (for testing)
```

### Resumable

The scanner records `processed_at` per photo. If interrupted (Ctrl+C,
crash, laptop sleep), just rerun the same command — it skips done photos.

## Model selection

Default `yoloe-11s-seg-pf.pt`:
- **YOLOE** = open-vocabulary YOLO
- **11s** = v11 backbone, small variant
- **seg** = supports segmentation (only detection used here)
- **-pf** = **prompt-free** mode, uses YOLOE's built-in vocabulary
  of thousands of categories

For your first scan, prompt-free is what you want — you don't have to
guess class names in advance.

Other options:

```bash
# Better quality, slower (~2x)
python scan.py /path/to/photos --model yoloe-11m-seg-pf.pt

# Highest quality, slow (~4x)
python scan.py /path/to/photos --model yoloe-11l-seg-pf.pt

# Try YOLO26 backbone (requires recent Ultralytics)
python scan.py /path/to/photos --model yoloe-26s-seg-pf.pt
```

First run downloads model from Ultralytics automatically.

## After scanning

`inspect_db.py` shows:
- Overview stats (photos, detections, classes, date range, GPS coverage)
- Top N classes with instance + photo counts
- 5 "busiest" photos (most objects detected)
- Rare classes (only in 1-2 photos) — usually the most interesting finds

```bash
python inspect_db.py --top 100 --min-conf 0.3
```

## Schema

`photos.db` SQLite file with two tables:

**`photos`** — one row per image
```
id, path, file_hash, taken_at, width, height,
gps_lat, gps_lon, camera_make, camera_model,
file_size, processed_at, error
```

**`detections`** — one row per detected object (many per photo)
```
id, photo_id, class, confidence,
bbox_x, bbox_y, bbox_w, bbox_h,    -- normalized 0-1
bbox_area_ratio                     -- bbox area / image area
```

All bbox coords are **normalized** (0-1), so they work regardless of
original image dimensions.

## Performance notes

- Images are auto-resized to `RESIZE_LONGEST` (640 by default) before
  detection. Edit this constant in `scan.py` for higher quality.
- HEIC files need `pillow-heif`. Already in requirements.
- WAL journal mode is enabled — DB stays consistent even on crash.

## Troubleshooting

**CUDA out of memory**:
Lower `RESIZE_LONGEST` to 512 in `scan.py`, or use `--device cpu`.

**Model not found**:
Ultralytics auto-downloads on first use. Check internet, or manually
download the `.pt` file from Ultralytics docs.

**HEIC files skipped**:
```bash
pip install pillow-heif
```

**Lots of photos fail**:
Check what's in `photos.error` column:
```bash
sqlite3 photos.db "SELECT path, error FROM photos WHERE error IS NOT NULL LIMIT 20"
```

## Next steps

After scan completes:
1. `python inspect_db.py` — sanity check
2. **(coming next)** `analyze.py` — co-occurrence analysis, PMI scoring,
   time-series breakdown
3. **(coming next)** `report.py` — generate HTML report with charts
   and thumbnail galleries

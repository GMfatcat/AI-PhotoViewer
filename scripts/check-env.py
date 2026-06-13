#!/usr/bin/env python3
"""Environment check for AI-PhotoViewer — verifies packages are installed and the
GPU stack works, independently of starting the server.

Run with the project venv, e.g.:
    .venv\\Scripts\\python.exe scripts\\check-env.py        (Windows)
    .venv/bin/python scripts/check-env.py                   (Linux/macOS)

Exit code 0 = ready, 1 = something required is missing.
(ASCII-only output so it prints fine in any console code page.)
"""
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# (display name, import module, required?)
PACKAGES = [
    ("ultralytics", "ultralytics", True),
    ("Pillow", "PIL", True),
    ("pillow-heif", "pillow_heif", False),
    ("tqdm", "tqdm", True),
    ("fastapi", "fastapi", True),
    ("uvicorn", "uvicorn", True),
    ("transformers", "transformers", True),
    ("sentence-transformers", "sentence_transformers", False),
    ("sqlite-vec", "sqlite_vec", True),
    ("torch", "torch", True),
    ("torchvision", "torchvision", True),
]

problems = 0
warnings = 0


def line(status, name, detail=""):
    print(f"  [{status:>4}] {name}{('  ' + detail) if detail else ''}")


print(f"Python {sys.version.split()[0]}  ({sys.executable})")
if sys.version_info < (3, 12):
    line("WARN", "Python", "3.12 recommended")
    warnings += 1
print("\nPackages:")
for name, mod, required in PACKAGES:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        line("OK", name, str(ver))
    except Exception as e:                        # noqa: BLE001
        if required:
            line("MISS", name, f"REQUIRED - {e}")
            problems += 1
        else:
            line("warn", name, f"optional - not installed")
            warnings += 1

print("\nGPU / CUDA:")
try:
    import torch
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        line("OK", torch.cuda.get_device_name(0),
             f"torch {torch.__version__} | VRAM {round((total-free)/1048576)}/"
             f"{round(total/1048576)} MB used")
    else:
        line("WARN", "CUDA not available", f"torch {torch.__version__} -> CPU only (slow)")
        warnings += 1
except Exception as e:                            # noqa: BLE001
    line("MISS", "torch CUDA check failed", str(e))
    problems += 1

print("\nsqlite-vec load:")
try:
    import sqlite3
    import sqlite_vec
    c = sqlite3.connect(":memory:")
    c.enable_load_extension(True)
    sqlite_vec.load(c)
    v = c.execute("SELECT vec_version()").fetchone()[0]
    c.close()
    line("OK", "vec extension loads", f"vec_version {v}")
except Exception as e:                            # noqa: BLE001
    line("MISS", "sqlite-vec failed to load", str(e))
    problems += 1

print("\nSigLIP model dir:")
candidates = []
if os.environ.get("SIGLIP_MODEL"):
    candidates.append(Path(os.environ["SIGLIP_MODEL"]))
candidates += [ROOT.parent / "models" / "siglip2-so400m",
               ROOT.parent / "models" / "siglip2-base"]
found = next((p for p in candidates if p.is_dir()), None)
if found:
    line("OK", "found", str(found))
else:
    line("warn", "no model dir found", "set SIGLIP_MODEL or put one under ../models/")
    warnings += 1

print(f"\nSummary: {problems} problem(s), {warnings} warning(s).")
print("READY" if problems == 0 else "NOT READY - install the REQUIRED items above")
sys.exit(1 if problems else 0)

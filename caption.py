"""Optional VLM image captioning for the photo album.

Two interchangeable backends:
  - local: a tiny in-process VLM (Florence-2) via transformers
  - api:   an OpenAI-compatible vision endpoint (Ollama / vLLM / SGLang)

Captions are stored in an FTS5 (trigram) table for keyword/description search.
The whole feature is OPTIONAL: with no backend configured, none of this runs and
the rest of the app (YOLOE + SigLIP) is unaffected.
"""
import base64
import io
import json
import sqlite3
import threading
import urllib.request

DEFAULT_LOCAL_MODEL = "microsoft/Florence-2-base"   # 0.23B, MIT, caption-purpose-built
DEFAULT_PROMPT = "Describe this image in one detailed sentence."


# ── Backends ───────────────────────────────────────────
class LocalCaptioner:
    """Tiny in-process VLM (Florence-2). Lazy/thread-safe; shares the GPU."""
    name = "local"

    def __init__(self, model_name=DEFAULT_LOCAL_MODEL, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
        self.model_name = model_name
        self._torch = torch
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, torch_dtype=self.dtype).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.task = "<DETAILED_CAPTION>"
        self._lock = threading.Lock()

    def caption(self, pil):
        torch = self._torch
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        with self._lock:
            inputs = self.processor(text=self.task, images=pil, return_tensors="pt")
            inputs = {k: (v.to(self.device, self.dtype) if v.is_floating_point()
                          else v.to(self.device)) for k, v in inputs.items()}
            with torch.inference_mode():
                ids = self.model.generate(
                    input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"],
                    max_new_tokens=256, num_beams=1, do_sample=False)
            text = self.processor.batch_decode(ids, skip_special_tokens=False)[0]
            parsed = self.processor.post_process_generation(
                text, task=self.task, image_size=(pil.width, pil.height))
            return (parsed.get(self.task) or "").strip()


class ApiCaptioner:
    """OpenAI-compatible vision chat endpoint (Ollama / vLLM / SGLang).

    base_url is the OpenAI base, e.g. http://127.0.0.1:11434/v1 (Ollama) or
    http://127.0.0.1:8001/v1 (vLLM/SGLang). We POST one image per request.
    """
    name = "api"

    def __init__(self, base_url, model, api_key=None, prompt=DEFAULT_PROMPT, timeout=120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.model_name = model
        self.api_key = api_key
        self.prompt = prompt
        self.timeout = timeout

    def caption(self, pil):
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
        b64 = base64.b64encode(buf.getvalue()).decode()
        payload = {
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                ],
            }],
            "max_tokens": 256,
            "temperature": 0,
            "stream": False,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return (data["choices"][0]["message"]["content"] or "").strip()


def make_captioner(backend=None, model=None, api_url=None, api_model=None,
                   api_key=None, device="cuda"):
    """Build a captioner from config, or return None when disabled."""
    backend = (backend or "none").lower()
    if backend == "none":
        return None
    if backend == "local":
        return LocalCaptioner(model or DEFAULT_LOCAL_MODEL, device=device)
    if backend == "api":
        if not api_url or not api_model:
            raise ValueError("api backend needs caption-api-url and caption-api-model")
        return ApiCaptioner(api_url, api_model, api_key=api_key)
    raise ValueError(f"unknown caption backend: {backend!r}")


# ── Storage + search (FTS5 trigram; LIKE fallback for short queries) ──
def ensure_caption_table(conn):
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS captions_fts USING fts5(
            caption, photo_id UNINDEXED, model UNINDEXED, tokenize='trigram'
        )
    """)
    conn.commit()


def has_captions(conn):
    return bool(conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='captions_fts'"
    ).fetchone())


def caption_count(conn):
    return conn.execute("SELECT COUNT(*) FROM captions_fts").fetchone()[0] if has_captions(conn) else 0


def get_caption(conn, photo_id):
    if not has_captions(conn):
        return None
    row = conn.execute("SELECT caption FROM captions_fts WHERE photo_id = ?", (photo_id,)).fetchone()
    return row[0] if row else None


def set_caption(conn, photo_id, caption_text, model):
    ensure_caption_table(conn)
    conn.execute("DELETE FROM captions_fts WHERE photo_id = ?", (photo_id,))
    conn.execute("INSERT INTO captions_fts(caption, photo_id, model) VALUES (?, ?, ?)",
                 (caption_text, photo_id, model))


def uncaptioned_ids(conn, photo_ids):
    """Subset of photo_ids that have no caption yet (preserves order)."""
    if not has_captions(conn):
        return list(photo_ids)
    done = {r[0] for r in conn.execute("SELECT photo_id FROM captions_fts")}
    return [pid for pid in photo_ids if pid not in done]


def delete_captions(conn, ids):
    if not ids or not has_captions(conn):
        return
    for pid in ids:
        conn.execute("DELETE FROM captions_fts WHERE photo_id = ?", (pid,))


def search_captions(conn, query, top=50):
    """Return [(photo_id, caption)] matching the query. FTS5 trigram (rank-
    ordered) for >=3 chars; LIKE fallback for shorter queries (e.g. 2-char zh)."""
    if not has_captions(conn):
        return []
    q = (query or "").strip()
    if not q:
        return []
    rows = []
    if len(q) >= 3:
        fts = '"' + q.replace('"', '""') + '"'
        try:
            rows = conn.execute(
                "SELECT photo_id, caption FROM captions_fts WHERE captions_fts MATCH ? "
                "ORDER BY rank LIMIT ?", (fts, top)).fetchall()
        except sqlite3.OperationalError:
            rows = []
    if not rows:
        like = "%" + q.replace("%", "").replace("_", "") + "%"
        rows = conn.execute(
            "SELECT photo_id, caption FROM captions_fts WHERE caption LIKE ? LIMIT ?",
            (like, top)).fetchall()
    return [(r[0], r[1]) for r in rows]

// Photo Archaeology — frontend interaction

let pageSize = 20;   // photos per page (adjustable via the page-size dropdown)

const state = {
  fullList: [],        // the whole current set (browse: shuffled all photos; search: filtered results)
  page: 0,             // current page index into fullList
  photoList: [],       // the current page slice — what the grid renders & ←/→ navigates
  searchResults: [],   // full unfiltered search results (threshold filters this client-side)
  lastQuery: "",       // active search query text
  currentIndex: -1,
  currentPhoto: null,  // full photo response from /api/photo/{id}
  hoveredDet: null,    // detection object under cursor (or null)
  lockedDet: null,     // clicked detection (sticky highlight)
  image: null,         // HTMLImageElement
  imgScale: 1,         // canvas-pixel-per-image-pixel
  filterClass: "",     // active class filter, "" = all
};

const els = {
  canvas: document.getElementById("canvas"),
  hoverLabel: document.getElementById("hover-label"),
  filename: document.getElementById("filename"),
  photoMeta: document.getElementById("photo-meta"),
  hoverInfo: document.getElementById("hover-info"),
  detList: document.getElementById("det-list"),
  detCount: document.getElementById("det-count"),
  classFilter: document.getElementById("class-filter"),
  posCurrent: document.getElementById("pos-current"),
  posTotal: document.getElementById("pos-total"),
  stats: document.getElementById("stats"),
  btnPrev: document.getElementById("btn-prev"),
  btnNext: document.getElementById("btn-next"),
  btnRandom: document.getElementById("btn-random"),
  searchInput: document.getElementById("search-input"),
  searchBtn: document.getElementById("search-btn"),
  searchClear: document.getElementById("search-clear"),
  searchTop: document.getElementById("search-top"),
  searchThr: document.getElementById("search-thr"),
  thrVal: document.getElementById("thr-val"),
  searchMode: document.getElementById("search-mode"),
  searchModeWrap: document.getElementById("search-mode-wrap"),
  photoCaption: document.getElementById("photo-caption"),
  resultsGrid: document.getElementById("results-grid"),
  btnShuffle: document.getElementById("btn-shuffle"),
  btnPagePrev: document.getElementById("btn-page-prev"),
  btnPageNext: document.getElementById("btn-page-next"),
  pageLabel: document.getElementById("page-label"),
  pageSizeSel: document.getElementById("page-size"),
  btnHelp: document.getElementById("btn-help"),
  btnFlow: document.getElementById("btn-flow"),
  modalOverlay: document.getElementById("modal-overlay"),
  modalTitle: document.getElementById("modal-title"),
  modalTools: document.getElementById("modal-tools"),
  modalBody: document.getElementById("modal-body"),
  modalClose: document.getElementById("modal-close"),
};

let statsBase = "";  // remembers the "N photos · M classes" line

const ctx = els.canvas.getContext("2d");

// ── Deterministic per-class colors ─────────────────────
function classColor(cls, alpha = 1.0) {
  let h = 0;
  for (let i = 0; i < cls.length; i++) h = (h * 31 + cls.charCodeAt(i)) | 0;
  const hue = ((h % 360) + 360) % 360;
  return `hsla(${hue}, 70%, 55%, ${alpha})`;
}

// ── API ───────────────────────────────────────────────
async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`API ${path}: ${r.status}`);
  return r.json();
}

// ── Init ──────────────────────────────────────────────
// (Re)load all gallery data: stats line, class filter, photo list.
// Called on first entry and again whenever the user returns after indexing.
async function reloadGalleryData() {
  try {
    const stats = await api("/api/stats");
    statsBase = `${stats.photos} photos · ${stats.classes} classes`;
    els.stats.textContent = statsBase;
  } catch (e) {
    els.stats.textContent = `error: ${e.message}`;
    return;
  }

  // Populate class filter (rebuild from scratch — counts may have changed)
  try {
    const classes = await api("/api/classes");
    els.classFilter.innerHTML = '<option value="">— all photos —</option>';
    for (const c of classes) {
      const opt = document.createElement("option");
      opt.value = c.class;
      opt.textContent = `${c.class}  (${c.photos})`;
      els.classFilter.appendChild(opt);
    }
  } catch (e) { console.warn("class list:", e); }

  // Show the 語意/描述 mode toggle only when captions exist (optional feature).
  try {
    const h = await api("/api/health");
    const capOn = !!(h.caption && h.caption.enabled && h.caption.count > 0);
    els.searchModeWrap.style.display = capOn ? "" : "none";
    if (!capOn) els.searchMode.value = "semantic";
  } catch (e) { els.searchModeWrap.style.display = "none"; }

  await loadPhotoList("");
}

let galleryInited = false;

async function init() {
  await reloadGalleryData();

  // Events
  els.btnPrev.addEventListener("click", () => navigate(-1));
  els.btnNext.addEventListener("click", () => navigate(+1));
  els.btnRandom.addEventListener("click", randomPhoto);
  els.classFilter.addEventListener("change", (e) => {
    els.searchInput.value = "";       // class filter and search are mutually exclusive
    loadPhotoList(e.target.value);
  });
  els.searchBtn.addEventListener("click", runSearch);
  els.searchClear.addEventListener("click", clearSearch);
  els.searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); runSearch(); }
  });
  // top-N changes the number fetched → re-query. Threshold filters client-side (live).
  els.searchTop.addEventListener("change", () => { if (els.searchInput.value.trim()) runSearch(); });
  els.searchMode.addEventListener("change", () => { if (els.searchInput.value.trim()) runSearch(); });
  els.searchThr.addEventListener("input", () => applyThreshold(false));   // live while dragging
  els.searchThr.addEventListener("change", () => applyThreshold(true));   // load detail on release
  els.btnShuffle.addEventListener("click", shuffleBrowse);
  els.btnPagePrev.addEventListener("click", () => gotoPage(state.page - 1));
  els.btnPageNext.addEventListener("click", () => gotoPage(state.page + 1));
  els.pageSizeSel.addEventListener("change", (e) => {
    pageSize = parseInt(e.target.value) || 20;
    state.page = 0;
    renderPage(true);
  });
  els.btnHelp.addEventListener("click", openHelp);
  els.btnFlow.addEventListener("click", openFlow);
  els.modalClose.addEventListener("click", closeModal);
  els.modalOverlay.addEventListener("click", (e) => {
    if (e.target === els.modalOverlay) closeModal();   // click backdrop to close
  });
  els.canvas.addEventListener("mousemove", onMouseMove);
  els.canvas.addEventListener("mouseleave", () => {
    state.hoveredDet = null;
    els.hoverLabel.classList.add("hidden");
    updateHoverInfo();
    redraw();
  });
  els.canvas.addEventListener("click", onCanvasClick);
  window.addEventListener("keydown", onKey);
  window.addEventListener("resize", () => { if (state.image) drawPhoto(); });
}

// ── Results grid ──────────────────────────────────────
function renderGrid() {
  const list = state.photoList;
  els.resultsGrid.innerHTML = list.map((p, i) => `
    <div class="grid-item" data-idx="${i}" title="${p.name}">
      <img loading="lazy" src="/api/thumb/${p.id}?v=${p.v || 0}" alt=""
           onerror="this.closest('.grid-item').classList.add('missing')">
      ${p.sim != null ? `<span class="sim-badge">${p.sim.toFixed(2)}</span>` : ""}
    </div>`).join("");
  els.resultsGrid.querySelectorAll(".grid-item").forEach(el => {
    el.addEventListener("click", () => selectIndex(parseInt(el.dataset.idx)));
  });
}

function selectIndex(i) {
  if (i < 0 || i >= state.photoList.length) return;
  state.currentIndex = i;
  els.resultsGrid.querySelectorAll(".grid-item").forEach((el, k) =>
    el.classList.toggle("selected", k === i));
  const sel = els.resultsGrid.querySelector(`.grid-item[data-idx="${i}"]`);
  if (sel) sel.scrollIntoView({ block: "nearest" });
  loadPhoto(state.photoList[i].id);
}

// ── Paging over fullList ──────────────────────────────
function shuffleInPlace(a) {
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function pageCount() {
  return Math.max(1, Math.ceil(state.fullList.length / pageSize));
}

function setFullList(list) {
  state.fullList = list;
  state.page = 0;
  renderPage(true);
}

function renderPage(loadDetail) {
  const start = state.page * pageSize;
  const slice = state.fullList.slice(start, start + pageSize);
  state.photoList = slice;
  els.posTotal.textContent = slice.length;
  updatePageLabel();
  renderGrid();
  if (slice.length === 0) {
    els.posCurrent.textContent = "0";
    clearCanvas("No photos");
  } else if (loadDetail) {
    selectIndex(0);
  }
}

function updatePageLabel() {
  const n = state.fullList.length;
  if (n === 0) { els.pageLabel.textContent = "0 / 0"; return; }
  const start = state.page * pageSize + 1;
  const end = Math.min(start + pageSize - 1, n);
  els.pageLabel.textContent = `${start}–${end} / ${n}`;
}

function gotoPage(p) {
  const n = pageCount();
  if (n <= 1) return;
  state.page = ((p % n) + n) % n;   // wrap around
  renderPage(true);
}

function shuffleBrowse() {
  if (state.searchResults.length) return;   // shuffle only in browse mode
  shuffleInPlace(state.fullList);
  state.page = 0;
  renderPage(true);
}

// ── Photo list management ─────────────────────────────
async function loadPhotoList(cls) {
  state.filterClass = cls;
  state.searchResults = [];   // entering browse mode
  state.lastQuery = "";
  const params = new URLSearchParams();
  if (cls) params.set("cls", cls);
  const list = await api("/api/photos?" + params.toString());
  shuffleInPlace(list);       // random order on every open (shuffle-once)
  setFullList(list);
}

function navigate(delta) {
  if (state.photoList.length === 0) return;
  selectIndex((state.currentIndex + delta + state.photoList.length)
              % state.photoList.length);
}

function randomPhoto() {
  if (state.photoList.length === 0) return;
  selectIndex(Math.floor(Math.random() * state.photoList.length));
}

// ── Semantic search ───────────────────────────────────
async function runSearch() {
  const q = els.searchInput.value.trim();
  if (!q) { clearSearch(); return; }
  if (els.searchMode && els.searchMode.value === "text") { runTextSearch(q); return; }
  els.classFilter.value = "";        // search overrides class filter
  els.stats.textContent = `🔍 searching "${q}"…`;
  const params = new URLSearchParams({ q });
  params.set("top", els.searchTop.value || "12");
  // threshold is applied client-side (live), so fetch the full top-N here
  params.set("min_ratio", "0");
  let list;
  try {
    list = await api("/api/search?" + params.toString());
  } catch (e) {
    els.stats.textContent = `search error: ${e.message}`;
    return;
  }
  state.searchResults = list;
  state.lastQuery = q;
  configureThresholdSlider(list);
  applyThreshold(true);
}

// ── Description (caption) search — optional feature ───────
async function runTextSearch(q) {
  els.classFilter.value = "";
  els.stats.textContent = `🔤 描述搜尋 "${q}"…`;
  let list;
  try {
    list = await api("/api/search_text?" + new URLSearchParams({ q, top: "100" }));
  } catch (e) {
    els.stats.textContent = `search error: ${e.message}`;
    return;
  }
  state.searchResults = list;     // reuse so clearSearch / threshold-guard work
  state.lastQuery = q;
  els.thrVal.textContent = "—";   // threshold doesn't apply to text results
  els.stats.textContent = `🔤 "${q}" · ${list.length} results`;
  state.fullList = list;
  state.page = 0;
  renderPage(true);
}

// Set the threshold slider's range to the actual sim span of these results, so the
// slider value is an ABSOLUTE sim cutoff that matches the green badges exactly.
function configureThresholdSlider(list) {
  if (!list.length) return;
  const sims = list.map(r => r.sim);
  const lo = Math.min(...sims), hi = Math.max(...sims);
  els.searchThr.min = lo.toFixed(3);
  els.searchThr.max = hi.toFixed(3);
  els.searchThr.step = Math.max(0.001, (hi - lo) / 50).toFixed(3);
  els.searchThr.value = lo.toFixed(3);   // leftmost = show all
}

// Keep results whose absolute sim >= the slider value (matches the badge numbers).
function applyThreshold(loadDetail) {
  if (!state.searchResults.length) return;
  if (state.searchResults[0].sim == null) return;   // text-search results have no sim
  const cutoff = parseFloat(els.searchThr.value) || 0;
  const lo = parseFloat(els.searchThr.min) || 0;
  els.thrVal.textContent = cutoff <= lo ? "off" : "≥" + cutoff.toFixed(2);
  const list = state.searchResults.filter(r => r.sim >= cutoff);
  els.stats.textContent = `🔍 "${state.lastQuery}" · ${list.length} results`;
  state.fullList = list;
  state.page = 0;
  renderPage(loadDetail);
}

function clearSearch() {
  els.searchInput.value = "";
  state.searchResults = [];
  state.lastQuery = "";
  els.stats.textContent = statsBase;
  loadPhotoList(state.filterClass);
}

// ── Load one photo ────────────────────────────────────
async function loadPhoto(id) {
  state.hoveredDet = null;
  state.lockedDet = null;
  els.posCurrent.textContent = state.currentIndex + 1;

  try {
    state.currentPhoto = await api(`/api/photo/${id}`);
  } catch (e) {
    console.error("loadPhoto:", e);
    clearCanvas("⚠ 無法載入此照片");
    return;
  }
  renderPhotoMeta();
  renderDetectionList();

  const img = new Image();
  img.onload = () => {
    state.image = img;
    drawPhoto();
  };
  img.onerror = () => clearCanvas("⚠ 此照片檔案無法顯示或已移除");
  img.src = state.currentPhoto.image_url;
  const li = state.photoList[state.currentIndex];
  els.filename.textContent = state.currentPhoto.name +
    (li && li.sim != null ? `   ·   sim ${li.sim}` : "");
}

function clearCanvas(msg) {
  els.canvas.width = 800;
  els.canvas.height = 200;
  ctx.fillStyle = "#1a1a1c";
  ctx.fillRect(0, 0, els.canvas.width, els.canvas.height);
  ctx.fillStyle = "#707078";
  ctx.font = "16px ui-monospace, monospace";
  ctx.textAlign = "center";
  ctx.fillText(msg, els.canvas.width / 2, els.canvas.height / 2);
}

// ── Drawing ───────────────────────────────────────────
function drawPhoto() {
  if (!state.image) return;
  const img = state.image;

  const container = els.canvas.parentElement;
  const maxW = container.clientWidth;
  const maxH = container.clientHeight;
  const scale = Math.min(maxW / img.width, maxH / img.height, 1);
  state.imgScale = scale;
  const W = Math.round(img.width * scale);
  const H = Math.round(img.height * scale);
  els.canvas.width = W;
  els.canvas.height = H;
  els.canvas.style.width = W + "px";
  els.canvas.style.height = H + "px";
  redraw();
}

function redraw() {
  if (!state.image) return;
  const W = els.canvas.width;
  const H = els.canvas.height;
  ctx.drawImage(state.image, 0, 0, W, H);

  // Show at most one polygon: locked takes priority over hover
  if (state.lockedDet) {
    drawPolygon(state.lockedDet, false, true);
  } else if (state.hoveredDet) {
    drawPolygon(state.hoveredDet, true, false);
  }
}

function drawPolygon(det, isHover, isLock) {
  if (!det.polygon || det.polygon.length < 3) return;
  const W = els.canvas.width;
  const H = els.canvas.height;

  ctx.beginPath();
  for (let i = 0; i < det.polygon.length; i++) {
    const [x, y] = det.polygon[i];
    const px = x * W;
    const py = y * H;
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.closePath();

  if (isLock) {
    ctx.fillStyle = classColor(det.class, 0.40);
    ctx.strokeStyle = classColor(det.class, 1.0);
    ctx.lineWidth = 3;
  } else {  // hover
    ctx.fillStyle = classColor(det.class, 0.25);
    ctx.strokeStyle = classColor(det.class, 0.95);
    ctx.lineWidth = 2;
  }
  ctx.fill();
  ctx.stroke();
}

// ── Hit-test ──────────────────────────────────────────
function pointInPolygon(x, y, polygon) {
  // Ray-casting in normalized coords
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if (((yi > y) !== (yj > y)) &&
        (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)) {
      inside = !inside;
    }
  }
  return inside;
}

function findDetAt(nx, ny) {
  // nx, ny normalized 0-1. Return smallest-area polygon containing point.
  let best = null;
  let bestArea = Infinity;
  for (const d of state.currentPhoto.detections) {
    if (!d.polygon || d.polygon.length < 3) continue;
    if (pointInPolygon(nx, ny, d.polygon)) {
      const area = d.area_ratio || (d.bbox.w * d.bbox.h);
      if (area < bestArea) {
        bestArea = area;
        best = d;
      }
    }
  }
  return best;
}

// ── Mouse interaction ─────────────────────────────────
function onMouseMove(e) {
  if (!state.currentPhoto) return;
  const rect = els.canvas.getBoundingClientRect();
  const nx = (e.clientX - rect.left) / rect.width;
  const ny = (e.clientY - rect.top) / rect.height;

  const found = findDetAt(nx, ny);
  if (found !== state.hoveredDet) {
    state.hoveredDet = found;
    updateHoverInfo();
    redraw();
  }

  if (found) {
    els.hoverLabel.classList.remove("hidden");
    els.hoverLabel.style.left = (e.clientX - rect.left) + "px";
    els.hoverLabel.style.top = (e.clientY - rect.top) + "px";
    els.hoverLabel.textContent =
      `${found.class}  ${found.confidence.toFixed(2)}`;
    els.hoverLabel.style.borderColor = classColor(found.class);
  } else {
    els.hoverLabel.classList.add("hidden");
  }
}

function onCanvasClick(e) {
  if (!state.currentPhoto) return;
  const rect = els.canvas.getBoundingClientRect();
  const nx = (e.clientX - rect.left) / rect.width;
  const ny = (e.clientY - rect.top) / rect.height;
  const found = findDetAt(nx, ny);

  if (!found) {
    state.lockedDet = null;
  } else if (found === state.lockedDet) {
    state.lockedDet = null;
  } else {
    state.lockedDet = found;
  }
  redraw();
  highlightInDetList();
}

function onKey(e) {
  if (e.key === "Escape" && !els.modalOverlay.classList.contains("hidden")) {
    closeModal(); return;
  }
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowLeft")  { navigate(-1); e.preventDefault(); }
  else if (e.key === "ArrowRight") { navigate(+1); e.preventDefault(); }
  else if (e.key === "r" || e.key === "R") { randomPhoto(); }
  else if (e.key === "Escape") { state.lockedDet = null; redraw(); }
}

// ── Panel rendering ───────────────────────────────────
function renderPhotoMeta() {
  const p = state.currentPhoto;
  const rows = [];
  rows.push(["id", p.id]);
  rows.push(["size", `${p.width} × ${p.height}`]);
  if (p.taken_at) rows.push(["taken", p.taken_at]);
  if (p.camera) rows.push(["camera", p.camera]);
  if (p.gps) rows.push(["gps", `${p.gps.lat.toFixed(4)}, ${p.gps.lon.toFixed(4)}`]);
  rows.push(["objects", p.detections.length]);
  const masked = p.detections.filter(d => d.polygon).length;
  rows.push(["with masks", `${masked} / ${p.detections.length}`]);
  if (p.suppressed_count > 0) {
    rows.push(["dedup'd", `${p.suppressed_count} hidden`]);
  }
  els.photoMeta.innerHTML = rows
    .map(([k, v]) => `<div class="row"><span class="label">${k}</span>` +
                     `<span class="value">${v}</span></div>`)
    .join("");
  // optional VLM caption (only shown when present)
  if (els.photoCaption) {
    els.photoCaption.innerHTML = p.caption
      ? `<h3>描述</h3><div class="cap-text">${escHtml(p.caption)}</div>` : "";
  }
}

function updateHoverInfo() {
  const target = state.lockedDet || state.hoveredDet;
  if (!target) {
    els.hoverInfo.innerHTML = state.lockedDet
      ? `<em class="muted">click empty area to unlock</em>`
      : `<em class="muted">hover an object on the photo</em>`;
    return;
  }
  const color = classColor(target.class);
  const lockNote = (target === state.lockedDet)
    ? `<div style="color:${color};font-size:11px;margin-top:6px;">🔒 locked</div>`
    : "";
  els.hoverInfo.innerHTML = `
    <div class="cls" style="color:${color}">${target.class}</div>
    <div class="row"><span class="label">confidence</span>
      <span>${target.confidence.toFixed(3)}</span></div>
    <div class="row"><span class="label">bbox area</span>
      <span>${(target.area_ratio * 100).toFixed(1)}%</span></div>
    <div class="row"><span class="label">center</span>
      <span>(${target.bbox.x.toFixed(2)}, ${target.bbox.y.toFixed(2)})</span></div>
    <div class="row"><span class="label">size</span>
      <span>${(target.bbox.w * 100).toFixed(0)}% × ${(target.bbox.h * 100).toFixed(0)}%</span></div>
    <div class="row"><span class="label">has mask</span>
      <span>${target.polygon ? "yes" : "no (bbox only)"}</span></div>
    ${lockNote}`;
  highlightInDetList();
}

function renderDetectionList() {
  const dets = state.currentPhoto.detections;
  els.detCount.textContent = dets.length;
  els.detList.innerHTML = dets.map((d, i) => `
    <div class="det-row" data-idx="${i}">
      <span class="color-chip" style="background:${classColor(d.class)}"></span>
      <span class="cls">${d.class}</span>
      <span class="conf">${d.confidence.toFixed(2)}</span>
    </div>`).join("");

  els.detList.querySelectorAll(".det-row").forEach(row => {
    const idx = parseInt(row.dataset.idx);
    row.addEventListener("mouseenter", () => {
      state.hoveredDet = dets[idx];
      updateHoverInfo();
      redraw();
    });
    row.addEventListener("mouseleave", () => {
      state.hoveredDet = null;
      updateHoverInfo();
      redraw();
    });
    row.addEventListener("click", () => {
      const d = dets[idx];
      state.lockedDet = (state.lockedDet === d) ? null : d;
      updateHoverInfo();
      redraw();
    });
  });
}

function highlightInDetList() {
  const target = state.lockedDet || state.hoveredDet;
  const idx = target
    ? state.currentPhoto.detections.indexOf(target)
    : -1;
  els.detList.querySelectorAll(".det-row").forEach((row, i) => {
    row.classList.toggle("active", i === idx);
  });
}

// ── Floating modals (usage / flowchart) ───────────────
function openModal(title, bodyHTML, toolsHTML = "") {
  els.modalTitle.textContent = title;
  els.modalTools.innerHTML = toolsHTML;
  els.modalBody.innerHTML = bodyHTML;
  els.modalOverlay.classList.remove("hidden");
}
function closeModal() {
  els.modalOverlay.classList.add("hidden");
}

const HELP = {
  zh: `
    <h4>🔍 語意搜尋</h4>
    <ul>
      <li>搜尋框輸入<b>中文或英文</b>(例:海邊、two girls),按 Enter 或 🔍。</li>
      <li><b>top</b>:回傳幾張結果(預設 12)。</li>
      <li><b>門檻</b>:絕對相似度下限,只留 sim ≥ 該值的;數字對應縮圖右下角綠色徽章;最左 = 全部顯示。</li>
      <li><b>✕</b>:清除搜尋、回到瀏覽。</li>
    </ul>
    <h4>🖼️ 結果網格(左欄)</h4>
    <ul>
      <li>縮圖右下的綠色數字 = 相似度(sim)。</li>
      <li><b>點縮圖</b> → 右側顯示大圖與偵測。</li>
      <li><b>🎲</b> 重新洗牌(瀏覽模式)· <b>‹ ›</b> 上/下一組 · 下拉選每頁張數(10–40)。</li>
    </ul>
    <h4>🏷️ 類別過濾</h4>
    <ul><li>Filter 下拉:依 YOLOE 偵測到的物件類別篩選照片。</li></ul>
    <h4>🎯 偵測檢視(右欄)</h4>
    <ul>
      <li>滑鼠移到照片中的物件 → 反白其遮罩;<b>點一下</b>鎖定。</li>
      <li>右欄列出所有偵測,可互相對應。</li>
    </ul>
    <h4>⌨️ 快捷鍵</h4>
    <ul><li><kbd>←</kbd> <kbd>→</kbd> 換照片 · <kbd>R</kbd> 隨機 · <kbd>Esc</kbd> 解除鎖定 / 關閉視窗</li></ul>`,
  en: `
    <h4>🔍 Semantic search</h4>
    <ul>
      <li>Type a query in <b>Chinese or English</b> (e.g. beach, two girls), press Enter or 🔍.</li>
      <li><b>top</b>: how many results to return (default 12).</li>
      <li><b>threshold</b>: absolute similarity floor — keeps photos with sim ≥ the value; the number matches the green badge on each thumbnail; far-left = show all.</li>
      <li><b>✕</b>: clear search, back to browsing.</li>
    </ul>
    <h4>🖼️ Results grid (left)</h4>
    <ul>
      <li>The green number on a thumbnail = similarity (sim).</li>
      <li><b>Click a thumbnail</b> → inspect it (image + detections) on the right.</li>
      <li><b>🎲</b> reshuffle (browse mode) · <b>‹ ›</b> prev/next set · dropdown picks page size (10–40).</li>
    </ul>
    <h4>🏷️ Class filter</h4>
    <ul><li>The Filter dropdown narrows photos by YOLOE-detected object class.</li></ul>
    <h4>🎯 Detection inspector (right)</h4>
    <ul>
      <li>Hover an object on the photo → its mask highlights; <b>click</b> to lock it.</li>
      <li>The right panel lists every detection.</li>
    </ul>
    <h4>⌨️ Shortcuts</h4>
    <ul><li><kbd>←</kbd> <kbd>→</kbd> change photo · <kbd>R</kbd> random · <kbd>Esc</kbd> unlock / close window</li></ul>`,
};

function renderHelp(lang) {
  const tools = `<span class="lang-toggle">` +
    `<button data-lang="zh" class="${lang === "zh" ? "active" : ""}">中文</button>` +
    `<button data-lang="en" class="${lang === "en" ? "active" : ""}">EN</button></span>`;
  openModal(lang === "zh" ? "使用說明" : "How to use", HELP[lang], tools);
  els.modalTools.querySelectorAll("button").forEach(b =>
    b.addEventListener("click", () => renderHelp(b.dataset.lang)));
}
function openHelp() { renderHelp("zh"); }

const FLOW = `<pre>
  📁 your image folder   (照片來源)
        │
        ▼
  ┌──────────── scan.py ─────────────┐
  │  YOLOE-11s-seg-pf                 │
  │  開放詞彙偵測 + 分割  (GPU)         │
  └───────────────┬───────────────────┘
                  ▼
         偵測框 · 遮罩 · EXIF
                  │
                  ▼
  🗄  photos.db   (SQLite)
      ├ photos / detections / masks
                  │
                  ▼
  ┌──────────── embed.py ────────────┐
  │  SigLIP 2  so400m                 │
  │  影像 → 1152 維向量  (GPU)         │
  └───────────────┬───────────────────┘
                  ▼
  🔢  vec_photos  (sqlite-vec · cosine)  ← 同一個 photos.db
                  │
                  ▼
  🌐  web_demo   (FastAPI)
      ├ /api/search    文字 → SigLIP → 向量最近鄰
      ├ /api/photos · /api/photo · /api/thumb
                  │
                  ▼
  🖥  瀏覽器 UI
      ├ 語意搜尋 (中/英) + top-N + 門檻
      ├ 結果網格 (縮圖 + sim) → 點選詳情
      └ 偵測檢視 (hover / 鎖定 遮罩)
</pre>`;

function openFlow() { openModal("架構流程圖 · Pipeline", FLOW); }

// ── Welcome page ──────────────────────────────────────
const wc = {
  welcome:   document.getElementById("welcome"),
  gallery:   document.getElementById("gallery"),
  enter:     document.getElementById("wc-enter"),
  home:      document.getElementById("btn-home"),
  path:      document.getElementById("wc-path"),
  browse:    document.getElementById("wc-browse"),
  indexNew:  document.getElementById("wc-index-new"),
  indexFull: document.getElementById("wc-index-full"),
  formMsg:   document.getElementById("wc-form-msg"),
  job:       document.getElementById("wc-job"),
  bar:       document.getElementById("wc-bar"),
  jobMsg:    document.getElementById("wc-job-msg"),
  cancel:    document.getElementById("wc-cancel"),
  health:    document.getElementById("wc-health"),
  sources:   document.getElementById("wc-sources"),
};

let healthTimer = null;
let pollActive = false;
let lastJobState = null;

function escHtml(s) {
  return String(s).replace(/[&<>"']/g, c =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function fmtTime(iso) {
  if (!iso) return "—";
  // sqlite stores e.g. "2026-06-13 14:05:09.123456"; trim to minutes
  return String(iso).replace("T", " ").slice(0, 16);
}

async function postJSON(path, body) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  let data = {};
  try { data = await r.json(); } catch (e) { /* empty body */ }
  if (!r.ok) throw new Error(data.detail || `${path}: ${r.status}`);
  return data;
}

// ── Page switching ────────────────────────────────────
function showWelcome() {
  wc.gallery.classList.add("hidden");
  wc.welcome.classList.remove("hidden");
  refreshSources();
  startHealthPolling();   // does an immediate poll, then self-schedules
}

async function enterGallery() {
  stopHealthPolling();
  wc.welcome.classList.add("hidden");
  wc.gallery.classList.remove("hidden");
  if (!galleryInited) {
    galleryInited = true;
    await init();
  } else {
    await reloadGalleryData();   // pick up anything indexed since last visit
  }
}

// ── Health / status polling (adaptive: fast while a job runs, slow when idle) ──
const POLL_BUSY_MS = 1000;   // an index job is running → near-real-time progress
const POLL_IDLE_MS = 5000;   // nothing happening → just keep the status card fresh

function startHealthPolling() {
  if (pollActive) return;
  pollActive = true;
  pollLoop();                 // immediate first poll, then self-schedules
}

function stopHealthPolling() {
  pollActive = false;
  if (healthTimer) { clearTimeout(healthTimer); healthTimer = null; }
}

async function pollLoop() {
  if (!pollActive) return;
  const running = await refreshHealth();
  if (!pollActive) return;    // may have been stopped while awaiting
  healthTimer = setTimeout(pollLoop, running ? POLL_BUSY_MS : POLL_IDLE_MS);
}

// Re-arm the loop right now (used when a job starts, so we switch to fast cadence
// immediately instead of waiting out the current idle interval).
function bumpPolling() {
  if (!pollActive) return;
  if (healthTimer) { clearTimeout(healthTimer); healthTimer = null; }
  pollLoop();
}

async function refreshHealth() {
  let h;
  try { h = await api("/api/health"); }
  catch (e) { wc.health.textContent = `讀取失敗:${e.message}`; return false; }
  renderHealth(h);
  renderJob(h.job);

  const js = h.job ? h.job.state : "idle";
  if (lastJobState === "running" && js !== "running") {
    refreshSources();          // a job just finished — refresh the folder list
  }
  lastJobState = js;
  return js === "running";
}

function renderHealth(h) {
  const g = h.gpu || {};
  const gpuLine = g.available
    ? `${g.name} · VRAM ${g.vram_used_mb}/${g.vram_total_mb} MB`
    : `無 GPU${g.error ? " (" + g.error + ")" : ""}`;

  const m = h.model || {};
  const modelName = m.path ? m.path.replace(/\\/g, "/").replace(/\/$/, "").split("/").pop() : "—";
  const modelLine = `${modelName}${m.dim ? ` · ${m.dim} 維` : ""}${m.loaded ? "" : " (未載入)"}`;

  const db = h.db || {};
  const cov = Math.round((h.coverage || 0) * 100);
  const rows = [
    ["GPU", gpuLine],
    ["模型", modelLine],
    ["照片", `${db.photos}`],
    ["語意向量", `${db.embedded} / ${db.photos} · 覆蓋 ${cov}%`],
    ["偵測 / 類別", `${db.detections} / ${db.classes}`],
    ["資料夾", `${db.sources}`],
  ];
  wc.health.innerHTML = rows.map(([k, v]) =>
    `<div class="row"><span class="label">${k}</span>` +
    `<span class="value">${escHtml(v)}</span></div>`).join("");
}

const PHASE_LABEL = { scan: "偵測", embed: "語意向量", done: "" };
const STATE_LABEL = { running: "進行中", done: "完成", cancelled: "已取消", error: "錯誤" };

function renderJob(job) {
  const running = job && job.state === "running";

  // idle (or never run) → hide panel, re-enable form
  if (!job || job.state === "idle") {
    wc.job.classList.add("hidden");
    setFormDisabled(false);
    return;
  }

  wc.job.classList.remove("hidden");
  setFormDisabled(running);
  wc.cancel.classList.toggle("hidden", !running);
  if (running) wc.cancel.textContent = "取消";

  const total = job.total || 0;
  const indet = running && total === 0;
  wc.bar.classList.toggle("indeterminate", indet);
  if (indet) {
    wc.bar.style.width = "";
  } else {
    const pct = total > 0 ? Math.round((job.done / total) * 100) : (running ? 0 : 100);
    wc.bar.style.width = pct + "%";
  }
  wc.bar.style.background = job.state === "error" ? "var(--warn)" : "";

  const phase = PHASE_LABEL[job.phase] != null ? PHASE_LABEL[job.phase] : (job.phase || "");
  const st = STATE_LABEL[job.state] || job.state;
  const parts = [st];
  if (phase) parts.push(phase);
  if (job.message) parts.push(job.message);
  wc.jobMsg.textContent = parts.join(" · ");
}

// ── Indexing actions ──────────────────────────────────
function setFormDisabled(d) {
  wc.indexNew.disabled = d;
  wc.indexFull.disabled = d;
  wc.path.disabled = d;
  wc.sources.querySelectorAll(".src-reindex, .src-remove").forEach(b => b.disabled = d);
}

async function submitIndex(mode) {
  const path = wc.path.value.trim();
  if (!path) { wc.formMsg.textContent = "請先輸入資料夾路徑"; return; }
  wc.formMsg.textContent = "啟動中…";
  try {
    await postJSON("/api/index", { path, mode: mode === "full" ? "full" : "new" });
    wc.formMsg.textContent = `已開始索引(${mode === "full" ? "重掃全部" : "新照片"})✓`;
    lastJobState = "running";
    bumpPolling();   // switch to fast cadence immediately
  } catch (e) {
    wc.formMsg.textContent = `錯誤:${e.message}`;
  }
}

async function cancelJob() {
  wc.cancel.textContent = "取消中…";
  try { await postJSON("/api/job/cancel", {}); } catch (e) { /* ignore */ }
  refreshHealth();
}

// Native server-side folder picker (local-desktop convenience).
async function browseFolder() {
  wc.formMsg.textContent = "開啟資料夾選取視窗(在執行 server 的這台機器)…";
  try {
    const r = await api("/api/browse-folder");
    if (r.path) {
      wc.path.value = r.path;
      wc.formMsg.textContent = `已選擇:${r.path}`;
    } else {
      wc.formMsg.textContent = "未選擇資料夾";
    }
  } catch (e) {
    wc.formMsg.textContent = `錯誤:${e.message}`;
  }
}

// ── Sources list ──────────────────────────────────────
async function refreshSources() {
  let list;
  try { list = await api("/api/sources"); }
  catch (e) { wc.sources.innerHTML = `<em class="muted">讀取失敗:${escHtml(e.message)}</em>`; return; }

  if (!list.length) {
    wc.sources.innerHTML = `<em class="muted">尚無已索引的資料夾,先在上方新增一個。</em>`;
    return;
  }
  wc.sources.innerHTML = list.map(s => {
    const p = escHtml(s.path);
    return `
    <div class="src-row">
      <div class="src-info">
        <div class="src-path" title="${p}">${p}</div>
        <div class="src-meta">${s.photo_count || 0} 張 · 最後索引 ${fmtTime(s.last_indexed_at)}</div>
      </div>
      <div class="src-actions">
        <button class="src-reindex" data-path="${p}" data-mode="new" title="只索引新照片">🔁 新增</button>
        <button class="src-reindex" data-path="${p}" data-mode="full" title="整個資料夾重掃">⟳ 全部</button>
        <button class="src-remove" data-path="${p}" title="從索引移除此資料夾">🗑 移除</button>
      </div>
    </div>`;
  }).join("");

  wc.sources.querySelectorAll(".src-reindex").forEach(b =>
    b.addEventListener("click", () => {
      wc.path.value = b.dataset.path;
      submitIndex(b.dataset.mode);
    }));
  wc.sources.querySelectorAll(".src-remove").forEach(b =>
    b.addEventListener("click", () => removeSource(b.dataset.path)));
}

async function removeSource(path) {
  if (!confirm(`確定要解除索引「${path}」?\n會從資料庫刪除此來源的索引(照片檔案不會被刪)。`)) return;
  wc.formMsg.textContent = "移除中…";
  try {
    const r = await postJSON("/api/sources/remove", { path });
    wc.formMsg.textContent = `已移除「${path}」(刪除 ${r.removed} 張索引)`;
    refreshHealth();
    refreshSources();
  } catch (e) {
    wc.formMsg.textContent = `移除失敗:${e.message}`;
  }
}

// ── Boot ──────────────────────────────────────────────
function bootWelcome() {
  wc.enter.addEventListener("click", enterGallery);
  wc.home.addEventListener("click", showWelcome);
  wc.browse.addEventListener("click", browseFolder);
  wc.indexNew.addEventListener("click", () => submitIndex("new"));
  wc.indexFull.addEventListener("click", () => submitIndex("full"));
  wc.path.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); submitIndex("new"); }
  });
  wc.cancel.addEventListener("click", cancelJob);
  showWelcome();
}

// Go!
bootWelcome();

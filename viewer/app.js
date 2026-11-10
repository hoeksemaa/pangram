"use strict";
// Data viewer for ASVspoof 2019 LA: filter, sort, histogram, listen, flag.
// Runs two ways: viewer/serve.py (all clips, flags saved to notes/flags.jsonl)
// or as a static site (a sample, flags saved in the browser).

const $ = (id) => document.getElementById(id);
const BINS = 40;
const MQ = matchMedia("(max-width: 899px)");
const mobile = () => MQ.matches;
const rowH = () => (mobile() ? 36 : 22);
const FLAG_KEY = "flags:asvspoof2019-la";

// Numeric statistics from scripts/build_index.py. Each gets a histogram.
const NUM = [
  { k: "dur", short: "dur", name: "duration", unit: "s", dec: 2, tip: "clip length" },
  { k: "lead_ms", short: "lead", name: "lead silence", unit: "ms", dec: 0, tip: "silence before the first frame within 40 dB of the loudest 20 ms frame" },
  { k: "trail_ms", short: "trail", name: "trail silence", unit: "ms", dec: 0, tip: "silence after the last such frame" },
  { k: "sil", short: "sil", name: "silent share", unit: "", dec: 2, tip: "share of 20 ms frames more than 40 dB below the loudest frame" },
  { k: "rms_db", short: "rms", name: "loudness", unit: "dBFS", dec: 1, tip: "overall RMS level" },
  { k: "peak_db", short: "peak", name: "peak", unit: "dBFS", dec: 1, tip: "highest sample level" },
  { k: "noise_db", short: "noise", name: "noise floor", unit: "dBFS", dec: 1, tip: "10th percentile of 20 ms frame levels" },
  { k: "zero", short: "zero", name: "exact zeros", unit: "", dec: 3, tip: "share of samples that are exactly 0 (digital silence)" },
  { k: "bw_khz", short: "bw", name: "bandwidth", unit: "kHz", dec: 2, tip: "frequency below which 99% of the energy lies" },
  { k: "clip", short: "clip", name: "clipped", unit: "", dec: 4, tip: "share of samples at full scale" },
];
const COLS = [
  { k: "flag", name: "⚑", w: 22, tip: "flagged" },
  { k: "id", name: "id", w: 108 },
  { k: "split", name: "split", w: 42, cat: true },
  { k: "label", name: "label", w: 54, cat: true },
  { k: "system", name: "sys", w: 38, cat: true, tip: "attack system (– = human)" },
  { k: "speaker", name: "spk", w: 68, cat: true },
  { k: "gender", name: "g", w: 22, cat: true, tip: "speaker gender (dev and eval only)" },
  ...NUM.map((f) => ({ k: f.k, name: f.short, w: 48, num: true, dec: f.dec, tip: `${f.name}${f.unit ? " (" + f.unit + ")" : ""}: ${f.tip}` })),
  { k: "sr", name: "sr", w: 46, num: true, dec: 0, tip: "sample rate (Hz)" },
  { k: "dup", name: "dup", w: 32, num: true, dec: 0, tip: "number of clips with identical audio" },
];
const MOBILE_KEYS = ["flag", "id", "label", "system", "speaker"];
const CAT_KEYS = ["split", "label", "system", "speaker", "gender"];

let N = 0, doc, cat, ids, hashes, errs, HUMAN = -1, AI = -1, serverFlags = false;
const C = {};
const S = {
  rows: new Uint32Array(0), sortKey: "id", sortDir: 1, sel: -1,
  ranges: {}, dom: {}, hist: [], flags: new Map(), counts: null, nhna: [0, 0],
};

// ---------- helpers ----------
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const fmt = (v, dec) => (v !== v ? "" : v.toFixed(dec));
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const dpr = () => window.devicePixelRatio || 1;
function fit(cv) {
  cv.width = Math.max(1, Math.round(cv.clientWidth * dpr()));
  cv.height = Math.max(1, Math.round(cv.clientHeight * dpr()));
  return cv.getContext("2d");
}
function domain(arr) {
  const v = Array.from(arr).filter((x) => x === x).sort((a, b) => a - b);
  if (!v.length) return [0, 1];
  let lo = v[Math.floor(v.length * 0.005)], hi = v[Math.min(v.length - 1, Math.ceil(v.length * 0.995))];
  if (hi <= lo) { lo -= 0.5; hi += 0.5; }
  return [lo, hi];
}
function visCols() {
  if (!mobile()) return COLS;
  const cols = COLS.filter((c) => MOBILE_KEYS.includes(c.k));
  const s = COLS.find((c) => c.k === S.sortKey);
  if (s && !MOBILE_KEYS.includes(s.k)) cols.push(s);
  return cols;
}
const audioUrl = (i) => `${doc.audio.base}/${cat.split[C.split[i]]}/${ids[i]}.${doc.audio.ext}`;

// ---------- load ----------
async function loadFlags(tryServer) {
  if (tryServer) try {
    const r = await fetch("flags");
    if (r.ok && (r.headers.get("content-type") || "").includes("json")) { serverFlags = true; return await r.json(); }
  } catch { /* static site */ }
  try { return JSON.parse(localStorage.getItem(FLAG_KEY) || "{}"); } catch { return {}; }
}
async function load() {
  $("stats").textContent = "loading…";
  const d = await fetch("index.json").then((r) => { if (!r.ok) throw new Error(`index.json ${r.status}`); return r.json(); });
  const flags = await loadFlags(!d.sample); // the static site (a sample) has no flag server
  doc = d; N = d.n; cat = d.cat; ids = d.cols.id; hashes = d.cols.hash; errs = d.cols.err;
  doc.audio = doc.audio || { base: "audio", ext: "flac" };
  for (const k of CAT_KEYS) C[k] = Uint16Array.from(d.cols[k]);
  for (const k of [...NUM.map((f) => f.k), "sr", "ch", "dup"]) {
    const src = d.cols[k], a = new Float64Array(N);
    for (let i = 0; i < N; i++) a[i] = src[i] == null ? NaN : src[i];
    C[k] = a;
  }
  HUMAN = cat.label.indexOf("human"); AI = cat.label.indexOf("ai");
  for (const f of NUM) S.dom[f.k] = domain(C[f.k]);
  S.flags = new Map(Object.entries(flags));
  $("flag-export").hidden = serverFlags;
  const src = '<a href="https://datashare.ed.ac.uk/handle/10283/3336" target="_blank" rel="noopener">ASVspoof 2019</a> (ODC-By 1.0)';
  $("note").innerHTML = doc.sample
    ? `Sample: ${N.toLocaleString()} of ${doc.sample.n_total.toLocaleString()} clips, ${doc.sample.per_group} per split × system. Data: ${src}.`
    : `Data: ${src}.`;
  initFilters(); initHists(); bind(); layout();
  apply();
  if (S.rows.length) select(0, false);
}

// ---------- filters ----------
function initFilters() {
  const opts = (id, name, vocab, show = (v) => v) => {
    $(id).innerHTML = `<option value="">${name}</option>` + vocab.map((v, i) => `<option value="${i}">${esc(show(v))}</option>`).join("");
  };
  opts("f-split", "split", cat.split);
  opts("f-label", "label", cat.label);
  opts("f-system", "system", cat.system, (v) => (v === "-" ? "– human" : v));
  opts("f-gender", "gender", cat.gender);
  $("f-sort").innerHTML = COLS.filter((c) => c.k !== "flag").map((c) => `<option value="${c.k}">sort: ${c.k === "id" ? "id" : c.name}</option>`).join("") + '<option value="flag">sort: flagged</option>';
}
function readF() {
  const n = (id) => ($(id).value === "" ? -1 : +$(id).value);
  return {
    split: n("f-split"), label: n("f-label"), system: n("f-system"), gender: n("f-gender"),
    q: $("f-q").value.trim().toUpperCase(), flag: $("f-flag").checked, dup: $("f-dup").checked,
  };
}

// One pass over all clips: table rows, class counts, histograms (each ignores its own
// range filter), and the system x split counts (which ignore the split and system filters).
function apply() {
  const F = readF();
  const prev = S.sel >= 0 && S.sel < S.rows.length ? S.rows[S.sel] : -1;
  const ranges = Object.entries(S.ranges).map(([k, r]) => ({ fi: NUM.findIndex((f) => f.k === k), v: C[k], lo: r[0], hi: r[1] }));
  const hist = NUM.map(() => ({ h: new Float64Array(BINS), a: new Float64Array(BINS) }));
  const bins = NUM.map((f) => { const [lo, hi] = S.dom[f.k]; return { v: C[f.k], lo, scale: BINS / (hi - lo) }; });
  const nSplit = cat.split.length;
  const counts = new Uint32Array(cat.system.length * nSplit);
  const out = [];
  let nh = 0, na = 0;
  for (let i = 0; i < N; i++) {
    const lab = C.label[i];
    if (F.label >= 0 && lab !== F.label) continue;
    if (F.gender >= 0 && C.gender[i] !== F.gender) continue;
    if (F.flag && !S.flags.has(ids[i])) continue;
    if (F.dup && !(C.dup[i] > 1)) continue;
    if (F.q && !(ids[i].includes(F.q) || cat.speaker[C.speaker[i]].includes(F.q) || hashes[i].toUpperCase().startsWith(F.q))) continue;
    let fails = 0, failed = -1;
    for (const r of ranges) {
      const v = r.v[i];
      if (!(v >= r.lo && v <= r.hi)) { fails++; failed = r.fi; if (fails > 1) break; }
    }
    if (fails === 0) counts[C.system[i] * nSplit + C.split[i]]++;
    if (F.split >= 0 && C.split[i] !== F.split) continue;
    if (F.system >= 0 && C.system[i] !== F.system) continue;
    if (fails === 0) { out.push(i); if (lab === HUMAN) nh++; else if (lab === AI) na++; }
    if (fails > 1 || (lab !== HUMAN && lab !== AI)) continue;
    for (let fi = 0; fi < NUM.length; fi++) {
      if (fails === 1 && failed !== fi) continue;
      const b = bins[fi], v = b.v[i];
      if (v !== v) continue;
      const k = Math.min(BINS - 1, Math.max(0, Math.floor((v - b.lo) * b.scale)));
      (lab === HUMAN ? hist[fi].h : hist[fi].a)[k]++;
    }
  }
  S.rows = Uint32Array.from(out);
  S.hist = hist; S.counts = counts; S.nhna = [nh, na];
  sortRows();
  S.sel = prev >= 0 ? S.rows.indexOf(prev) : -1;
  updateStats(); layoutRows(); drawHists(); drawCounts();
}
function updateStats() {
  const [h, a] = S.nhna;
  $("stats").innerHTML = `${S.rows.length.toLocaleString()} / ${N.toLocaleString()} · <span class="h">human ${h.toLocaleString()}</span> · <span class="a">AI ${a.toLocaleString()}</span> · ⚑ ${S.flags.size}`;
}

// ---------- table ----------
function sortRows() {
  const k = S.sortKey, d = S.sortDir;
  if (k === "id") S.rows.sort((x, y) => d * (x - y)); // the index is sorted by id
  else if (k === "flag") S.rows.sort((x, y) => (S.flags.has(ids[y]) - S.flags.has(ids[x])) * d || x - y);
  else {
    const v = C[k];
    S.rows.sort((x, y) => {
      const a = v[x], b = v[y], na = a !== a, nb = b !== b;
      if (na || nb) return na - nb;
      return d * (a - b) || x - y;
    });
  }
}
function setSort(k, dir) {
  const col = COLS.find((c) => c.k === k);
  S.sortKey = k; S.sortDir = dir ?? (col && col.num ? -1 : 1);
  const id = S.sel >= 0 ? S.rows[S.sel] : -1;
  sortRows(); S.sel = id >= 0 ? S.rows.indexOf(id) : -1;
  $("f-sort").value = k; $("f-dir").textContent = S.sortDir > 0 ? "▲" : "▼";
  layoutRows(); if (S.sel >= 0) scrollToRow(S.sel);
}
function layoutRows() {
  const cols = visCols(), w = Math.max(cols.reduce((s, c) => s + c.w, 0), $("tbody").clientWidth);
  $("rows").style.width = $("spacer").style.width = `${w}px`;
  $("spacer").style.height = `${S.rows.length * rowH()}px`;
  $("thead").innerHTML = `<div style="display:flex;min-width:${w}px">` + cols.map((c) => {
    const on = c.k === S.sortKey, arrow = on ? (S.sortDir > 0 ? " ▲" : " ▼") : "";
    return `<span data-k="${c.k}" class="${on ? "on" : ""}" title="${esc(c.tip || c.k)}" style="flex:${c.k === "id" ? 1 : 0} 0 ${c.w}px;${c.num ? "text-align:right" : ""}">${c.name}${arrow}</span>`;
  }).join("") + "</div>";
  renderRows();
}
function cell(c, i) {
  const st = `style="flex:0 0 ${c.w}px"`;
  if (c.k === "flag") return `<span ${st} class="fl">${S.flags.has(ids[i]) ? "⚑" : ""}</span>`;
  if (c.k === "id") return `<span style="flex:1 0 ${c.w}px">${ids[i]}</span>`; // takes the spare width
  if (c.cat) {
    const v = cat[c.k][C[c.k][i]];
    const cls = c.k === "label" ? v : "";
    const show = c.k === "system" && v === "-" ? "–" : c.k === "gender" ? v[0] : v;
    return `<span ${st} class="${cls}" data-k="${c.k}">${esc(show)}</span>`;
  }
  return `<span ${st} class="n">${fmt(C[c.k][i], c.dec)}</span>`;
}
function renderRows() {
  const tb = $("tbody"), h = rowH(), top = tb.scrollTop, H = tb.clientHeight || 800, cols = visCols();
  const a = Math.max(0, Math.floor(top / h) - 15);
  const b = Math.min(S.rows.length, Math.ceil((top + H) / h) + 15);
  let html = "";
  for (let p = a; p < b; p++) {
    const i = S.rows[p];
    html += `<div class="r${p === S.sel ? " sel" : ""}${errs[i] ? " err" : ""}" data-p="${p}" style="top:${p * h}px;height:${h}px">`;
    for (const c of cols) html += cell(c, i);
    html += "</div>";
  }
  $("rows").innerHTML = html;
}
function scrollToRow(p) {
  const tb = $("tbody"), h = rowH(), y = p * h;
  if (y < tb.scrollTop) tb.scrollTop = y;
  else if (y + h > tb.scrollTop + tb.clientHeight) tb.scrollTop = y + h - tb.clientHeight;
}

// ---------- histograms ----------
function initHists() {
  $("hists").innerHTML = NUM.map((f, fi) => `
    <div class="hist" data-fi="${fi}">
      <div class="cap"><span title="${esc(f.tip)}"><b>${f.name}</b> ${f.unit}</span><span class="rng"></span>
        <button class="clr" hidden title="clear this filter">×</button>
        <span class="auc" title="AUC of this statistic alone, human vs AI (0.5 = no information). Red = a possible shortcut."></span></div>
      <canvas title="Drag to filter a range. Tap a bar for one bin. Double-click or × to clear."></canvas>
    </div>`).join("");
  const xOf = (e, cv) => Math.max(0, Math.min(cv.clientWidth, e.clientX - cv.getBoundingClientRect().left));
  $("hists").querySelectorAll(".hist").forEach((el) => {
    const fi = +el.dataset.fi, cv = el.querySelector("canvas");
    let drag = null;
    cv.addEventListener("pointerdown", (e) => { try { cv.setPointerCapture(e.pointerId); } catch { /* keep dragging */ } drag = { x0: xOf(e, cv), x1: xOf(e, cv) }; });
    cv.addEventListener("pointermove", (e) => { if (drag) { drag.x1 = xOf(e, cv); drawHist(fi, drag); } });
    cv.addEventListener("pointercancel", () => { drag = null; drawHist(fi); });
    cv.addEventListener("pointerup", () => {
      if (!drag) return;
      const { x0, x1 } = drag, f = NUM[fi], [lo, hi] = S.dom[f.k], W = cv.clientWidth;
      drag = null;
      let a, b;
      if (Math.abs(x1 - x0) < 4) { const k = Math.min(BINS - 1, Math.floor((x0 / W) * BINS)); a = k / BINS; b = (k + 1) / BINS; }
      else { a = Math.min(x0, x1) / W; b = Math.max(x0, x1) / W; }
      S.ranges[f.k] = [a <= 0.001 ? -Infinity : lo + a * (hi - lo), b >= 0.999 ? Infinity : lo + b * (hi - lo)];
      apply();
    });
    const clear = () => { delete S.ranges[NUM[fi].k]; apply(); };
    cv.addEventListener("dblclick", clear);
    el.querySelector(".clr").addEventListener("click", clear);
  });
}
function drawHists() { NUM.forEach((_, fi) => drawHist(fi)); }
function drawHist(fi, drag) {
  const f = NUM[fi], el = $("hists").children[fi], cv = el.querySelector("canvas");
  if (!cv.clientWidth || !S.hist.length) return; // hidden tab, or before the first apply()
  const g = fit(cv), W = cv.width, H = cv.height, { h, a } = S.hist[fi];
  const th = h.reduce((s, x) => s + x, 0), ta = a.reduce((s, x) => s + x, 0);
  let ymax = 0;
  for (let k = 0; k < BINS; k++) ymax = Math.max(ymax, th ? h[k] / th : 0, ta ? a[k] / ta : 0);
  g.clearRect(0, 0, W, H);
  const bw = W / BINS;
  for (const [arr, tot, col] of [[h, th, css("--human")], [a, ta, css("--ai")]]) {
    if (!tot) continue;
    g.fillStyle = col; g.globalAlpha = 0.55;
    for (let k = 0; k < BINS; k++) {
      const y = ymax ? ((arr[k] / tot) / ymax) * (H - 2) : 0;
      if (y > 0) g.fillRect(k * bw + 0.5, H - y, Math.max(1, bw - 1), y);
    }
  }
  g.globalAlpha = 1;
  const [lo, hi] = S.dom[f.k], r = S.ranges[f.k];
  let sel = null;
  if (drag) sel = [Math.min(drag.x0, drag.x1) * dpr(), Math.max(drag.x0, drag.x1) * dpr()];
  else if (r) sel = [r[0] === -Infinity ? 0 : ((r[0] - lo) / (hi - lo)) * W, r[1] === Infinity ? W : ((r[1] - lo) / (hi - lo)) * W];
  if (sel) {
    g.fillStyle = "rgba(128,128,128,0.18)"; g.fillRect(sel[0], 0, sel[1] - sel[0], H);
    g.strokeStyle = css("--fg"); g.lineWidth = dpr(); g.strokeRect(sel[0], 0.5, sel[1] - sel[0], H - 1);
  }
  const show = (v) => (Math.abs(v) === Infinity ? "…" : v.toFixed(f.dec));
  el.querySelector(".rng").textContent = r ? `[${show(r[0])}, ${show(r[1])}]` : `${lo.toFixed(f.dec)} – ${hi.toFixed(f.dec)}`;
  el.querySelector(".clr").hidden = !r;
  let auc = NaN;
  if (th && ta) {
    let below = 0, s = 0;
    for (let k = 0; k < BINS; k++) { s += a[k] * (below + 0.5 * h[k]); below += h[k]; }
    auc = s / (th * ta);
    auc = Math.max(auc, 1 - auc);
  }
  const ae = el.querySelector(".auc");
  ae.textContent = auc === auc ? `AUC ${auc.toFixed(2)}` : "";
  ae.className = `auc${auc >= 0.75 ? " hi" : ""}`;
}

// ---------- counts: system x split ----------
function drawCounts() {
  const order = ["train", "dev", "eval"].map((s) => cat.split.indexOf(s)).filter((x) => x >= 0);
  const nSplit = cat.split.length, F = readF();
  let html = `<tr><th>system</th>${order.map((c) => `<th>${cat.split[c]}</th>`).join("")}<th>all</th></tr>`;
  cat.system.forEach((s, si) => {
    let tot = 0;
    const tds = order.map((c) => {
      const v = S.counts[si * nSplit + c]; tot += v;
      const on = F.system === si && F.split === c ? ' style="font-weight:600"' : "";
      return `<td class="c${v ? "" : " z"}" data-sys="${si}" data-split="${c}"${on}>${v.toLocaleString()}</td>`;
    }).join("");
    html += `<tr><td class="c" data-sys="${si}" data-split="">${s === "-" ? "– human" : esc(s)}</td>${tds}<td>${tot.toLocaleString()}</td></tr>`;
  });
  $("counts").innerHTML = html;
}

// ---------- player ----------
const A = $("audio");
let token = 0, cur = null, ctx = null;
// Synchronous up to A.play(), so a tap counts as the user gesture that iOS needs.
function select(p, autoplay = $("b-auto").checked) {
  if (!S.rows.length) return;
  p = Math.max(0, Math.min(S.rows.length - 1, p));
  S.sel = p; scrollToRow(p); renderRows();
  const i = S.rows[p], t = ++token, src = audioUrl(i);
  showMeta(i);
  cur = { i, ab: null };
  A.src = src;
  if (autoplay) A.play().catch(() => {});
  loadViz(i, src, t);
}
async function loadViz(i, src, t) {
  try {
    const r = await fetch(src);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const buf = await r.arrayBuffer();
    if (t !== token) return;
    const ab = await decode(buf, C.sr[i]);
    if (t !== token) return;
    cur.ab = ab; drawViz();
  } catch (e) { if (t === token) clearViz(`no audio: ${e.message}`); }
}
async function decode(buf, sr) {
  try { return await new OfflineAudioContext(1, 1, sr > 0 ? sr : 16000).decodeAudioData(buf.slice(0)); }
  catch { ctx = ctx || new AudioContext(); return await ctx.decodeAudioData(buf.slice(0)); }
}
function showMeta(i) {
  const L = cat.label[C.label[i]], sys = cat.system[C.system[i]];
  $("meta-sum").innerHTML = `${ids[i]} · <span class="${esc(L)}">${esc(L)}</span> · ${sys === "-" ? "–" : esc(sys)} · ${cat.speaker[C.speaker[i]]} · ${fmt(C.dur[i], 2)} s`;
  const kv = [
    ["split", cat.split[C.split[i]]], ["gender", cat.gender[C.gender[i]]],
    ...NUM.filter((f) => f.k !== "dur").map((f) => [f.short, `${fmt(C[f.k][i], f.dec)} ${f.unit}`]),
    ["sr", C.sr[i]], ["dup", C.dup[i]], ["hash", hashes[i]],
  ];
  if (errs[i]) kv.push(["error", esc(errs[i])]);
  $("meta").innerHTML = kv.map(([k, v]) => `<span class="k">${k}</span><span>${v}</span>`).join("");
  $("flag-note").value = S.flags.get(ids[i]) ?? "";
}
function clearViz(msg) {
  for (const id of ["wave", "spec"]) fit($(id)).clearRect(0, 0, $(id).width, $(id).height);
  const g = $("spec").getContext("2d");
  g.fillStyle = "#ccc"; g.font = `${12 * dpr()}px system-ui`; g.fillText(msg, 10, 24);
}
function drawViz() { if (cur && cur.ab && $("spec").clientWidth) { drawWave(cur.ab, cur.i); drawSpec(cur.ab, C.sr[cur.i]); } }
function drawWave(ab, i) {
  const cv = $("wave"), g = fit(cv), W = cv.width, H = cv.height, x = ab.getChannelData(0);
  g.clearRect(0, 0, W, H);
  const dur = x.length / ab.sampleRate, lead = C.lead_ms[i] / 1000, trail = C.trail_ms[i] / 1000;
  g.fillStyle = "rgba(128,128,128,0.28)"; // shade the leading and trailing silence
  if (lead > 0) g.fillRect(0, 0, (W * lead) / dur, H);
  if (trail > 0) g.fillRect(W * (1 - trail / dur), 0, W, H);
  g.strokeStyle = C.label[i] === HUMAN ? css("--human") : css("--ai");
  g.lineWidth = 1; g.beginPath();
  const spp = x.length / W;
  for (let px = 0; px < W; px++) {
    const a = Math.floor(px * spp), b = Math.min(x.length, Math.floor((px + 1) * spp) + 1);
    let mn = 1, mx = -1;
    for (let j = a; j < b; j++) { const v = x[j]; if (v < mn) mn = v; if (v > mx) mx = v; }
    if (mn > mx) continue;
    g.moveTo(px + 0.5, (H / 2) * (1 - mx)); g.lineTo(px + 0.5, (H / 2) * (1 - mn) + 1);
  }
  g.stroke();
}
function fft(re, im) {
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len, wr = Math.cos(ang), wi = Math.sin(ang), half = len >> 1;
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let j = 0; j < half; j++) {
        const p = i + j, q = p + half;
        const vr = re[q] * cr - im[q] * ci, vi = re[q] * ci + im[q] * cr;
        re[q] = re[p] - vr; im[q] = im[p] - vi; re[p] += vr; im[p] += vi;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}
const LUT = (() => { // dark-to-bright colormap
  const stops = [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]];
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (stops.length - 1), k = Math.min(stops.length - 2, Math.floor(t)), u = t - k;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = stops[k][c] + (stops[k + 1][c] - stops[k][c]) * u;
  }
  return lut;
})();
function drawSpec(ab, sr) {
  const x = ab.getChannelData(0), rate = ab.sampleRate;
  const nfft = 1 << Math.round(Math.log2(0.032 * rate)), hop = Math.max(1, Math.round(0.008 * rate));
  const fmax = (sr > 0 ? sr : rate) / 2;
  const nb = Math.max(1, Math.min(nfft / 2, Math.floor((fmax / rate) * nfft)));
  const nfr = Math.max(1, Math.floor((x.length - nfft) / hop) + 1);
  const win = new Float32Array(nfft).map((_, j) => 0.5 - 0.5 * Math.cos((2 * Math.PI * j) / (nfft - 1)));
  const re = new Float32Array(nfft), im = new Float32Array(nfft), db = new Float32Array(nfr * nb);
  let mx = -Infinity;
  for (let f = 0; f < nfr; f++) {
    const o = f * hop;
    for (let j = 0; j < nfft; j++) { re[j] = (x[o + j] || 0) * win[j]; im[j] = 0; }
    fft(re, im);
    for (let b = 0; b < nb; b++) {
      const v = 10 * Math.log10(re[b] * re[b] + im[b] * im[b] + 1e-12);
      db[f * nb + b] = v; if (v > mx) mx = v;
    }
  }
  const off = document.createElement("canvas"); off.width = nfr; off.height = nb;
  const og = off.getContext("2d"), img = og.createImageData(nfr, nb), floor = mx - 80;
  for (let f = 0; f < nfr; f++) for (let b = 0; b < nb; b++) {
    const t = Math.max(0, Math.min(255, Math.round(((db[f * nb + b] - floor) / 80) * 255)));
    const p = ((nb - 1 - b) * nfr + f) * 4;
    img.data[p] = LUT[t * 3]; img.data[p + 1] = LUT[t * 3 + 1]; img.data[p + 2] = LUT[t * 3 + 2]; img.data[p + 3] = 255;
  }
  og.putImageData(img, 0, 0);
  const cv = $("spec"), g = fit(cv), W = cv.width, H = cv.height;
  g.imageSmoothingEnabled = true; g.drawImage(off, 0, 0, W, H);
  g.fillStyle = "rgba(255,255,255,0.75)"; g.font = `${10 * dpr()}px system-ui`;
  const fTop = (nb / nfft) * rate;
  for (let k = 2; k * 1000 < fTop; k += 2) g.fillText(`${k}k`, 3 * dpr(), H - ((k * 1000) / fTop) * H + 4 * dpr());
}
function tick() {
  if (!A.duration || !isFinite(A.duration)) return;
  const head = $("head");
  head.style.display = "block"; head.style.left = `${(A.currentTime / A.duration) * 100}%`;
  $("time").textContent = `${A.currentTime.toFixed(2)} / ${A.duration.toFixed(2)} s`;
  if (!A.paused) requestAnimationFrame(tick);
}
const togglePlay = () => { if (A.src) A.paused ? A.play().catch(() => {}) : A.pause(); };
const randomPick = () => { if (S.rows.length) select(Math.floor(Math.random() * S.rows.length)); };
async function setFlag(note) {
  if (!cur) return;
  const id = ids[cur.i];
  if (serverFlags) {
    const r = await fetch("flags", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id, note }) });
    if (!r.ok) return;
    S.flags = new Map(Object.entries(await r.json()));
  } else {
    if (note == null) S.flags.delete(id); else S.flags.set(id, note);
    try { localStorage.setItem(FLAG_KEY, JSON.stringify(Object.fromEntries(S.flags))); } catch { /* private mode */ }
  }
  $("flag-note").value = S.flags.get(id) ?? "";
  renderRows(); updateStats();
}
function exportFlags() {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([JSON.stringify(Object.fromEntries(S.flags), null, 1)], { type: "application/json" }));
  a.download = "flags.json"; a.click();
}

// ---------- layout + events ----------
function layout() {
  $("meta-box").open = !mobile();
  layoutRows(); drawHists(); drawViz();
}
function setTab(t) {
  document.body.dataset.tab = t;
  document.querySelectorAll("#tabs button").forEach((b) => b.classList.toggle("on", b.dataset.tab === t));
  if (t === "stats") drawHists(); else { layoutRows(); drawViz(); }
}
function bind() {
  let qt, rt;
  for (const id of ["f-split", "f-label", "f-system", "f-gender", "f-flag", "f-dup"]) $(id).addEventListener("change", apply);
  $("f-q").addEventListener("input", () => { clearTimeout(qt); qt = setTimeout(apply, 150); });
  $("f-sort").addEventListener("change", () => setSort($("f-sort").value));
  $("f-dir").addEventListener("click", () => setSort(S.sortKey, -S.sortDir));
  $("reset").addEventListener("click", () => {
    for (const id of ["f-split", "f-label", "f-system", "f-gender", "f-q"]) $(id).value = "";
    $("f-flag").checked = $("f-dup").checked = false; S.ranges = {}; apply();
  });
  $("b-filters").addEventListener("click", () => {
    const on = document.body.classList.toggle("show-filters");
    $("b-filters").setAttribute("aria-expanded", on);
    renderRows();
  });
  document.querySelectorAll("#tabs button").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));
  $("tbody").addEventListener("scroll", () => { $("thead").scrollLeft = $("tbody").scrollLeft; renderRows(); }, { passive: true });
  $("thead").addEventListener("click", (e) => {
    const k = e.target.closest("span")?.dataset.k; if (!k) return;
    setSort(k, S.sortKey === k ? -S.sortDir : undefined);
  });
  $("rows").addEventListener("click", (e) => {
    const sp = e.target.closest("span"), row = e.target.closest(".r"); if (!row) return;
    const i = S.rows[+row.dataset.p];
    if (!mobile() && sp?.dataset.k === "speaker") { $("f-q").value = cat.speaker[C.speaker[i]]; apply(); return; }
    if (!mobile() && sp?.dataset.k === "system") { $("f-system").value = C.system[i]; apply(); return; }
    select(+row.dataset.p);
  });
  $("counts").addEventListener("click", (e) => {
    const td = e.target.closest("td.c"); if (!td) return;
    $("f-system").value = td.dataset.sys; $("f-split").value = td.dataset.split; apply();
  });
  $("b-prev").addEventListener("click", () => select(S.sel - 1));
  $("b-next").addEventListener("click", () => select(S.sel + 1));
  $("b-play").addEventListener("click", togglePlay);
  $("b-rand").addEventListener("click", randomPick);
  $("flag-save").addEventListener("click", () => setFlag($("flag-note").value.trim() || "flagged"));
  $("flag-clear").addEventListener("click", () => setFlag(null));
  $("flag-export").addEventListener("click", exportFlags);
  $("viz").addEventListener("click", (e) => {
    const r = $("viz").getBoundingClientRect();
    if (A.duration && isFinite(A.duration)) { A.currentTime = ((e.clientX - r.left) / r.width) * A.duration; tick(); }
  });
  A.addEventListener("play", () => { $("b-play").textContent = "❚❚"; requestAnimationFrame(tick); });
  A.addEventListener("pause", () => { $("b-play").textContent = "▶"; tick(); });
  A.addEventListener("loadedmetadata", tick);
  A.addEventListener("ended", () => { if ($("b-chain").checked && S.sel < S.rows.length - 1) select(S.sel + 1, true); });
  MQ.addEventListener("change", layout);
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(layout, 100); });
  document.addEventListener("keydown", (e) => {
    const el = document.activeElement, tag = el?.tagName;
    if ((tag === "INPUT" && el.type !== "checkbox") || tag === "SELECT") {
      if (e.key === "Escape") el.blur();
      if (e.key === "Enter" && el === $("flag-note")) { $("flag-save").click(); el.blur(); }
      return;
    }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const keys = {
      j: () => select(S.sel + 1), ArrowDown: () => select(S.sel + 1),
      k: () => select(S.sel - 1), ArrowUp: () => select(S.sel - 1),
      " ": togglePlay, r: randomPick, f: () => $("flag-note").focus(), "/": () => $("f-q").focus(),
      a: () => $("b-auto").click(), c: () => $("b-chain").click(),
    };
    if (keys[e.key]) { e.preventDefault(); keys[e.key](); }
  });
}

load().catch((e) => { $("stats").textContent = `error: ${e.message}`; });

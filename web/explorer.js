// The Data tab: explore ASVspoof 2019 LA. Filter, listen, and compare features.
// Runs two ways: web/serve.py (all clips) or as the static site (a sample).

const $ = (id) => document.getElementById(id);
const BINS = 40;
const MQ = matchMedia("(max-width: 899px)");
const mobile = () => MQ.matches;
const rowH = () => (mobile() ? 48 : 36);

// Auditory features from scripts/build_index.py, in groups. Each gets a histogram on the Stats tab.
const NUM = [
  { g: "time", k: "dur", name: "duration", unit: "s", dec: 2, tip: "clip length" },
  { g: "time", k: "syl_rate", name: "syllable rate", unit: "/s", dec: 2, tip: "loudness peaks per second, close to syllables per second (eGeMAPS)" },
  { g: "time", k: "voiced_s", name: "voiced run", unit: "s", dec: 2, tip: "mean length of one stretch of voiced sound (eGeMAPS)" },
  { g: "pitch", k: "f0_hz", name: "pitch", unit: "Hz", dec: 0, tip: "mean pitch of the voice (eGeMAPS F0)" },
  { g: "pitch", k: "f0_range_st", name: "pitch range", unit: "st", dec: 1, tip: "20th to 80th percentile of pitch, in semitones: monotone vs. expressive (eGeMAPS)" },
  { g: "pitch", k: "f0_rise", name: "pitch rise speed", unit: "st/s", dec: 0, tip: "mean slope of rising pitch, in semitones per second (eGeMAPS)" },
  { g: "voice quality", k: "jitter_pct", name: "jitter", unit: "%", dec: 2, tip: "cycle-to-cycle change in pitch period: rough vs. smooth (eGeMAPS)" },
  { g: "voice quality", k: "shimmer_db", name: "shimmer", unit: "dB", dec: 2, tip: "cycle-to-cycle change in loudness: rough vs. smooth (eGeMAPS)" },
  { g: "voice quality", k: "hnr_db", name: "harmonics-to-noise", unit: "dB", dec: 1, tip: "energy in the harmonics vs. noise: clear vs. breathy or hoarse (eGeMAPS)" },
  { g: "voice quality", k: "h1h2_db", name: "breathiness (H1–H2)", unit: "dB", dec: 1, tip: "first harmonic minus second harmonic: higher = breathier (eGeMAPS)" },
  { g: "vowels", k: "f1_hz", name: "F1", unit: "Hz", dec: 0, tip: "mean first formant: how open the mouth is (eGeMAPS)" },
  { g: "vowels", k: "f2_hz", name: "F2", unit: "Hz", dec: 0, tip: "mean second formant: tongue front vs. back (eGeMAPS)" },
  { g: "timbre", k: "bright_hz", name: "brightness", unit: "Hz", dec: 0, tip: "median spectral centroid of speech frames: dull vs. bright" },
  { g: "timbre", k: "flux", name: "spectral change", unit: "", dec: 2, tip: "how much the spectrum changes from one frame to the next (eGeMAPS spectral flux)" },
  { g: "timbre", k: "spec_detail_db", name: "spectral detail", unit: "dB", dec: 1, tip: "mean variation over time of 40 mel bands in speech frames: low = over-smoothed" },
  { g: "loudness", k: "rms_db", name: "loudness", unit: "dBFS", dec: 1, tip: "overall RMS level" },
  { g: "loudness", k: "loud_range_db", name: "dynamics", unit: "dB", dec: 1, tip: "10th to 90th percentile of 20 ms frame levels in speech" },
];
// The AI systems in plain words (dataset README). TTS = text-to-speech; VC = voice conversion.
const SYSTEMS = {
  A01: "text-to-speech, neural waveform", A02: "text-to-speech, vocoder", A03: "text-to-speech, vocoder",
  A04: "text-to-speech, waveform concatenation", A05: "voice conversion, vocoder", A06: "voice conversion, spectral filtering",
  A07: "text-to-speech, vocoder + GAN", A08: "text-to-speech, neural waveform", A09: "text-to-speech, vocoder",
  A10: "text-to-speech, neural waveform", A11: "text-to-speech, Griffin-Lim", A12: "text-to-speech, neural waveform",
  A13: "TTS + VC, concatenation + filtering", A14: "TTS + VC, vocoder", A15: "TTS + VC, neural waveform",
  A16: "text-to-speech, waveform concatenation", A17: "voice conversion, waveform filtering", A18: "voice conversion, vocoder",
  A19: "voice conversion, spectral filtering", enroll: "speaker enrollment, not labeled",
};
// How each kind of AI system makes speech, in one line.
const KIND = {
  "text-to-speech": "Text-to-speech: reads typed text aloud in the target voice.",
  "voice conversion": "Voice conversion: turns a real recording of another person into the target voice.",
  "TTS + VC": "Text-to-speech, then voice conversion into the target voice.",
};
const SPLIT_NAME = { train: "train", dev: "validation", eval: "test" };
const labelName = (v) => (v === "ai" ? "AI" : v === "?" ? "unlabeled" : v);
const CAT_KEYS = ["split", "label", "system", "gender"];

let N = 0, doc, cat, ids, errs, texts = null, HUMAN = -1, AI = -1;
const C = {};
const S = { rows: new Uint32Array(0), sel: -1, ranges: {}, dom: {}, hist: [], nhna: [0, 0] };

// ---------- helpers ----------
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
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
const audioUrl = (i) => `${doc.audio.base}/${cat.split[C.split[i]]}/${ids[i]}.${doc.audio.ext}`;
// Display only: clip number 001, 002, ... in index order (the dataset id stays in the hover title).
const num = (i) => String(i + 1).padStart(String(N).length, "0");

// ---------- load ----------
let loading = null;
export function initExplorer() {
  loading = loading || load().catch((e) => { $("stats").textContent = `error: ${e.message}`; throw e; });
  return loading;
}
export function showExplorer() { layout(); }
const visible = () => document.body.dataset.screen === "data";
const onClips = () => visible() && $("data").dataset.tab === "clips"; // the player is on screen

async function load() {
  $("stats").textContent = "loading…";
  // transcripts.json (optional, display only): Whisper's guess of what each clip says.
  const [d, tr] = await Promise.all([
    fetch("index.json").then((r) => { if (!r.ok) throw new Error(`index.json ${r.status}`); return r.json(); }),
    fetch("transcripts.json").then((r) => (r.ok ? r.json() : null)).catch(() => null),
  ]);
  doc = d; N = d.n; cat = d.cat; ids = d.cols.id; errs = d.cols.err; texts = tr && tr.text ? tr.text : null;
  doc.audio = doc.audio || { base: "audio", ext: "flac" };
  for (const k of CAT_KEYS) C[k] = Uint16Array.from(d.cols[k]);
  for (const k of [...NUM.map((f) => f.k), "sr"]) {
    const src = d.cols[k], a = new Float64Array(N);
    for (let i = 0; i < N; i++) a[i] = src[i] == null ? NaN : src[i];
    C[k] = a;
  }
  HUMAN = cat.label.indexOf("human"); AI = cat.label.indexOf("ai");
  for (const f of NUM) S.dom[f.k] = domain(C[f.k]);
  const src = '<a href="https://datashare.ed.ac.uk/handle/10283/3336" target="_blank" rel="noopener">ASVspoof 2019</a> (ODC-By 1.0)';
  $("note").innerHTML = doc.sample
    ? `A sample: ${N.toLocaleString()} of ${doc.sample.n_total.toLocaleString()} clips, ${doc.sample.per_group} from each split × AI system. Data: ${src}.`
    : `Data: ${src}.`;
  initHists(); bind();
  apply();
  if (S.rows.length) select(0, false);
}

// One pass over all clips: list rows, class counts, and histograms (each ignores its own range filter).
// The only filters are the histogram ranges (Stats tab).
function apply() {
  const prev = S.sel >= 0 && S.sel < S.rows.length ? S.rows[S.sel] : -1;
  const ranges = Object.entries(S.ranges).map(([k, r]) => ({ fi: NUM.findIndex((f) => f.k === k), v: C[k], lo: r[0], hi: r[1] }));
  const hist = NUM.map(() => ({ h: new Float64Array(BINS), a: new Float64Array(BINS) }));
  const bins = NUM.map((f) => { const [lo, hi] = S.dom[f.k]; return { v: C[f.k], lo, scale: BINS / (hi - lo) }; });
  const out = [];
  let nh = 0, na = 0;
  for (let i = 0; i < N; i++) {
    const lab = C.label[i];
    let fails = 0, failed = -1;
    for (const r of ranges) {
      const v = r.v[i];
      if (!(v >= r.lo && v <= r.hi)) { fails++; failed = r.fi; if (fails > 1) break; }
    }
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
  S.rows = Uint32Array.from(out); // the index is sorted by id
  S.hist = hist; S.nhna = [nh, na];
  S.sel = prev >= 0 ? S.rows.indexOf(prev) : -1;
  updateStats(); layoutRows(); drawHists();
}
function updateStats() {
  const [h, a] = S.nhna;
  const n = (x) => x.toLocaleString();
  const k = (x) => (x >= 10000 ? `${Math.round(x / 1000)}k` : n(x));
  const filtered = S.rows.length < N, total = doc.sample ? doc.sample.n_total : 0;
  // "560 of 122k clips" (sample of the full set) or "42 of 560 clips" (histogram ranges on)
  const of = filtered ? N : total;
  const tip = filtered ? "clips inside the histogram ranges (Stats tab)" : `a sample of the ${n(total)} clips in ASVspoof 2019 LA`;
  $("stats").innerHTML = `<span class="chip" title="${tip}">${n(S.rows.length)}${of ? ` of ${k(of)}` : ""} clips</span>`
    + `<span class="chip h" title="human clips shown">${n(h)} human</span><span class="chip a" title="AI clips shown">${n(a)} AI</span>`;
}

// ---------- clip list: number and label ----------
function layoutRows() {
  $("spacer").style.height = `${S.rows.length * rowH()}px`;
  renderRows();
}
function renderRows() {
  const tb = $("tbody"), h = rowH(), top = tb.scrollTop, H = tb.clientHeight || 800;
  const a = Math.max(0, Math.floor(top / h) - 15);
  const b = Math.min(S.rows.length, Math.ceil((top + H) / h) + 15);
  let html = "";
  for (let p = a; p < b; p++) {
    const i = S.rows[p], L = cat.label[C.label[i]];
    html += `<div class="r${p === S.sel ? " sel" : ""}${errs[i] ? " err" : ""}" data-p="${p}" title="${ids[i]}" style="top:${p * h}px;height:${h}px">`
      + `<span class="id">${num(i)}</span><span class="lab ${esc(L)}">${esc(labelName(L))}</span></div>`;
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
  $("hists").innerHTML = NUM.map((f, fi) => `${fi === 0 || NUM[fi - 1].g !== f.g ? `<div class="grp">${f.g}</div>` : ""}
    <div class="hist" data-fi="${fi}">
      <div class="cap"><span class="nm" title="${esc(f.tip)}"><b>${f.name}</b> ${f.unit}</span><span class="rng"></span>
        <button class="clr" hidden title="clear this filter">×</button>
        <span class="auc" title="Pick one human and one AI clip at random: AUC is the chance this feature alone tells which is which."></span></div>
      <div class="tip">${esc(f.tip)}<br><span class="dim">AUC: pick one human and one AI clip at random; AUC is the chance this feature alone tells which is which. 0.5 = no clue, 1 = always.</span></div>
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
    for (const sel of [".nm", ".auc"]) el.querySelector(sel).addEventListener("click", () => el.classList.toggle("info"));
  });
}
function drawHists() { NUM.forEach((_, fi) => drawHist(fi)); }
function drawHist(fi, drag) {
  const f = NUM[fi], el = $("hists").querySelector(`.hist[data-fi="${fi}"]`), cv = el.querySelector("canvas");
  if (!cv.clientWidth || !S.hist.length) return; // hidden tab, or before the first apply()
  const g = fit(cv), W = cv.width, H = cv.height, { h, a } = S.hist[fi];
  const th = h.reduce((s, x) => s + x, 0), ta = a.reduce((s, x) => s + x, 0);
  let ymax = 0;
  for (let k = 0; k < BINS; k++) ymax = Math.max(ymax, th ? h[k] / th : 0, ta ? a[k] / ta : 0);
  g.clearRect(0, 0, W, H);
  const bw = W / BINS, gap = Math.max(1, Math.round(dpr())), half = Math.max(1, Math.floor((bw - gap) / 2));
  [[h, th, css("--human"), 0], [a, ta, css("--ai"), half]].forEach(([arr, tot, col, dx]) => {
    if (!tot) return;
    g.fillStyle = col;
    for (let k = 0; k < BINS; k++) {
      const y = ymax ? Math.round(((arr[k] / tot) / ymax) * (H - 3 * gap)) : 0;
      if (y > 0) g.fillRect(Math.round(k * bw) + dx, H - y, half, y);
    }
  });
  const [lo, hi] = S.dom[f.k], r = S.ranges[f.k];
  let sel = null;
  if (drag) sel = [Math.min(drag.x0, drag.x1) * dpr(), Math.max(drag.x0, drag.x1) * dpr()];
  else if (r) sel = [r[0] === -Infinity ? 0 : ((r[0] - lo) / (hi - lo)) * W, r[1] === Infinity ? W : ((r[1] - lo) / (hi - lo)) * W];
  if (sel) {
    g.fillStyle = "rgba(255,176,0,0.16)"; g.fillRect(sel[0], 0, sel[1] - sel[0], H);
    g.strokeStyle = css("--gold"); g.lineWidth = 2 * dpr(); g.strokeRect(sel[0] + dpr(), dpr(), sel[1] - sel[0] - 2 * dpr(), H - 2 * dpr());
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

// ---------- player ----------
const A = $("audio");
let token = 0, cur = null, ctx = null;
// Synchronous up to A.play(), so a tap counts as the user gesture that iOS needs.
function select(p, autoplay = true) {
  if (!S.rows.length) return;
  p = Math.max(0, Math.min(S.rows.length - 1, p));
  S.sel = p; scrollToRow(p); renderRows();
  const i = S.rows[p], t = ++token, src = audioUrl(i);
  showClip(i);
  cur = { i, ab: null };
  A.src = src;
  if (autoplay) A.play().catch(() => {});
  loadViz(i, src, t);
}
function infoHTML(i) {
  const L = cat.label[C.label[i]], sys = cat.system[C.system[i]];
  const row = (k, v) => `<span class="k">${esc(k)}</span><span>${v}</span>`;
  let html;
  if (L === "human") {
    html = `<div class="method"><span class="k">AI method</span><b>none: a real person</b><span class="dim">A studio recording from the VCTK corpus.</span></div>`;
  } else {
    const desc = SYSTEMS[sys] || "";
    const kind = Object.keys(KIND).find((k) => desc.startsWith(k));
    html = `<div class="method"><span class="k">AI method</span><b>${esc(sys)} · ${esc(desc)}</b>`
      + `${kind ? `<span class="dim">${esc(KIND[kind])}</span>` : ""}</div>`;
  }
  const spk = doc.cols.speaker ? cat.speaker[doc.cols.speaker[i]] : "?";
  html += `<div class="grid">`
    + row("voice of", `speaker ${esc(spk)}${cat.gender[C.gender[i]] !== "?" ? ` (${esc(cat.gender[C.gender[i]])})` : ""}`)
    + row("split", esc(SPLIT_NAME[cat.split[C.split[i]]] || cat.split[C.split[i]]))
    + row("dataset id", esc(ids[i]));
  let g = "";
  for (const f of NUM) {
    if (f.g !== g) { g = f.g; html += `<span class="g">${esc(g)}</span>`; }
    const v = C[f.k][i];
    html += row(f.name, v === v ? `${v.toFixed(f.dec)}${f.unit ? ` ${esc(f.unit)}` : ""}` : "–");
  }
  return html + `</div>`;
}
function setInfo(open) {
  $("b-info").setAttribute("aria-expanded", open);
  $("clip-info").hidden = !open;
}
function showClip(i) {
  $("clip-info").innerHTML = infoHTML(i);
  const L = cat.label[C.label[i]];
  $("clip-head").innerHTML = `<span class="badge ${esc(L)}">${esc(labelName(L))}</span><b title="${ids[i]}">${num(i)}</b>`;
  const text = texts && texts[ids[i]];
  $("text").hidden = !text;
  $("text").innerHTML = text ? `<span class="dim" title="Whisper's guess of the words">guess:</span> ${esc(text)}` : "";
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
function clearViz(msg) {
  for (const id of ["wave", "spec"]) fit($(id)).clearRect(0, 0, $(id).width, $(id).height);
  const g = $("spec").getContext("2d");
  g.fillStyle = "#ccc"; g.font = `${16 * dpr()}px "Share Tech Mono", monospace`; g.fillText(msg, 10 * dpr(), 26 * dpr());
}
function drawViz() { if (cur && cur.ab && $("spec").clientWidth) { drawWave(cur.ab, cur.i); drawSpec(cur.ab, C.sr[cur.i]); } }
function drawWave(ab, i) {
  const cv = $("wave"), g = fit(cv), W = cv.width, H = cv.height, x = ab.getChannelData(0);
  g.clearRect(0, 0, W, H);
  g.strokeStyle = C.label[i] === HUMAN ? css("--human") : C.label[i] === AI ? css("--ai") : css("--ink-2");
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
const LUT = (() => { // neutral dark-to-bright colormap (blue and red stay for human and AI)
  const stops = [[2, 4, 10], [22, 30, 50], [52, 64, 94], [98, 110, 140], [156, 164, 184], [210, 210, 214], [246, 242, 234]];
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
  g.fillStyle = "rgba(255,255,255,0.8)"; g.font = `${15 * dpr()}px "Share Tech Mono", monospace`;
  const fTop = (nb / nfft) * rate;
  for (let k = 2; k * 1000 < fTop; k += 2) g.fillText(`${k}k`, 4 * dpr(), H - ((k * 1000) / fTop) * H + 5 * dpr());
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

// ---------- layout + events ----------
function layout() { layoutRows(); drawHists(); drawViz(); }
function bind() {
  let rt;
  $("tbody").addEventListener("scroll", renderRows, { passive: true });
  $("rows").addEventListener("click", (e) => { const row = e.target.closest(".r"); if (row) select(+row.dataset.p); });
  $("b-prev").addEventListener("click", () => select(S.sel - 1));
  $("b-next").addEventListener("click", () => select(S.sel + 1));
  $("b-play").addEventListener("click", togglePlay);
  $("b-rand").addEventListener("click", randomPick);
  $("b-info").addEventListener("click", () => setInfo($("clip-info").hidden));
  $("viz").addEventListener("click", (e) => {
    const r = $("viz").getBoundingClientRect();
    if (A.duration && isFinite(A.duration)) { A.currentTime = ((e.clientX - r.left) / r.width) * A.duration; tick(); }
  });
  A.addEventListener("play", () => { $("b-play").textContent = "❚❚"; requestAnimationFrame(tick); });
  A.addEventListener("pause", () => { $("b-play").textContent = "▶"; tick(); });
  A.addEventListener("loadedmetadata", tick);
  A.addEventListener("ended", () => { if ($("b-chain").checked && S.sel < S.rows.length - 1) select(S.sel + 1, true); });
  MQ.addEventListener("change", () => visible() && layout());
  window.addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => visible() && layout(), 100); });
  window.addEventListener("hashchange", () => { if (!onClips()) A.pause(); });
  document.addEventListener("keydown", (e) => {
    if (!onClips()) return;
    const el = document.activeElement, tag = el?.tagName;
    if ((tag === "INPUT" && el.type !== "checkbox") || tag === "SELECT") { if (e.key === "Escape") el.blur(); return; }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    const keys = {
      j: () => select(S.sel + 1), ArrowDown: () => select(S.sel + 1),
      k: () => select(S.sel - 1), ArrowUp: () => select(S.sel - 1),
      " ": togglePlay, r: randomPick,
      c: () => $("b-chain").click(), i: () => $("b-info").click(),
    };
    if (keys[e.key]) { e.preventDefault(); keys[e.key](); }
  });
}

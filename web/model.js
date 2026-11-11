// Model and Try tabs. Data: models.json (scripts/build_models.py).
// Try runs the exported model (models/<run>.onnx) in the browser with ONNX Runtime Web:
// audio at 16 kHz -> 3 s windows (a short clip is repeated) -> AI score per window -> mean.
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const pct = (x, d = 1) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`);
const ORT = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
const SR = 16000, N = 48000; // 3 s windows, as in training
const COL = { train: "#1fa898", validation: "#c98500" }; // checked with the dataviz palette validator (dark)

let doc = null;
export function initModels() {
  doc = doc || fetch("models.json").then((r) => (r.ok ? r.json() : null)).catch(() => null);
  return doc;
}

// ---------- Model tab ----------
let shown = null, resizeBound = false;
export async function showModel() {
  const d = await initModels();
  if (!d || !d.models.length) { $("m-body").innerHTML = `<p class="t-note">No trained model yet.</p>`; return; }
  shown = shown && d.models.find((m) => m.id === shown) ? shown : d.models[0].id;
  $("m-pick").innerHTML = d.models.map((m) => `<button data-id="${m.id}" class="${m.id === shown ? "on" : ""}">${esc(m.name)}</button>`).join("");
  $("m-pick").hidden = d.models.length < 2;
  $("m-pick").onclick = (e) => { const b = e.target.closest("button"); if (b) { shown = b.dataset.id; showModel(); } };
  renderModel(d, d.models.find((m) => m.id === shown));
  if (!resizeBound) {
    resizeBound = true;
    let t;
    window.addEventListener("resize", () => { clearTimeout(t); t = setTimeout(() => document.body.dataset.screen === "model" && showModel(), 150); });
  }
}

function renderModel(d, m) {
  const r = m.results, g = m.fpr_per_gender;
  const seen = ["A16", "A19"]; // A16 = A04 and A19 = A06: the same systems as in training
  const va = m.curves.validation, minV = Math.min(...va.map((p) => p[1]));
  const sys = Object.entries(m.per_system);
  const maxE = Math.max(...sys.map(([, v]) => v), 0.01);
  $("m-body").innerHTML = `
    <article class="panel m-card">
      <h3>${esc(m.name)}</h3>
      <p class="m-meta">${(m.parameters / 1000).toFixed(0)}k parameters · ${m.mb} MB · trained ${Math.round(m.train_minutes)} min on a laptop</p>
      <p>${esc(m.about)}</p>
      <p class="m-dim">${esc(d.common)}</p>
    </article>

    <section class="panel m-card">
      <h3>Loss over training</h3>
      <div class="legend"><span><i style="background:${COL.train}"></i>train</span><span><i style="background:${COL.validation}"></i>validation</span></div>
      <div class="chart" id="m-chart"></div>
      <p class="m-dim">Lower is better. The kept checkpoint (epoch ${m.best_epoch}) has the lowest validation loss.${va[va.length - 1][1] > minV * 1.2 ? " After it, validation loss rose while train loss kept falling: the model began to memorize the training speakers." : ""}</p>
    </section>

    <section class="panel m-card">
      <h3>Results</h3>
      <table class="m-res">
        <tr><th></th><th>error (EER)</th><th>humans flagged</th><th>AI caught</th></tr>
        ${[["Validation", r.validation], ["Test", r.test], ["In-the-Wild", r.in_the_wild]].map(([k, v]) =>
          `<tr><td>${k}</td><td>${pct(v.eer)}</td><td>${pct(v.fpr)}</td><td>${pct(v.catch, 0)}</td></tr>`).join("")}
      </table>
      <p class="m-dim">EER: the error rate where misses equal false alarms. "Humans flagged" and "AI caught" use one fixed line, set so it flags 1% of validation humans.
      Test: new speakers and 11 AI systems the model never heard. In-the-Wild: real and fake public figures from the internet.</p>
      ${g && g.female != null ? `<p>Humans flagged on test: female ${pct(g.female)}, male ${pct(g.male)}.</p>` : ""}
    </section>

    <section class="panel m-card">
      <h3>Test error by AI system</h3>
      <div class="bars">${sys.map(([s, v]) => `
        <div class="bar-row" title="${s}: EER ${pct(v)}"><span class="bl">${s}</span>
          <span class="bt"><span style="width:${(v / maxE) * 100}%"></span></span><span class="bv">${pct(v)}</span></div>`).join("")}
      </div>
      <p class="m-dim">${seen.join(" and ")} reuse systems from training; the rest are new.</p>
    </section>

    ${d.baselines.length ? `<section class="panel m-card">
      <h3>All models, test split</h3>
      <table class="m-res">
        <tr><th></th><th>error (EER)</th><th>humans flagged</th></tr>
        ${[...d.baselines.map((b) => [b.name, b.test]), ...d.models.map((x) => [x.name, x.results.test])].map(([k, v]) =>
          `<tr><td>${esc(k)}</td><td>${pct(v.eer)}</td><td>${pct(v.fpr)}</td></tr>`).join("")}
      </table>
    </section>` : ""}`;
  lossChart($("m-chart"), m);
}

// One axis (loss). Train: bucket means as a line. Validation: line + markers at each check.
function lossChart(el, m) {
  const W = el.clientWidth || 340, H = 230, P = { l: 40, r: 12, t: 10, b: 50 };
  const tr = m.curves.train, va = m.curves.validation;
  const xMax = Math.max(tr[tr.length - 1][0], va[va.length - 1][0]);
  const top = Math.max(...tr.map((p) => p[1]), ...va.map((p) => p[1]));
  const step = top > 1.2 ? 0.5 : 0.2, yMax = Math.ceil(top / step) * step;
  const x = (v) => P.l + (v / xMax) * (W - P.l - P.r), y = (v) => P.t + (1 - v / yMax) * (H - P.t - P.b);
  const path = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${x(p[0]).toFixed(1)},${y(p[1]).toFixed(1)}`).join("");
  const yt = Array.from({ length: Math.round(yMax / step) + 1 }, (_, i) => i * step), xt = Array.from({ length: Math.floor(xMax) + 1 }, (_, i) => i);
  el.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" role="img" aria-label="train and validation loss by epoch">
    ${yt.map((v) => `<line x1="${P.l}" x2="${W - P.r}" y1="${y(v)}" y2="${y(v)}" class="grid"/><text x="${P.l - 6}" y="${y(v) + 5}" text-anchor="end">${v.toFixed(1)}</text>`).join("")}
    ${xt.map((v) => `<text x="${x(v)}" y="${H - P.b + 20}" text-anchor="middle">${v}</text>`).join("")}
    <text x="${(P.l + W - P.r) / 2}" y="${H - 6}" text-anchor="middle">epoch</text>
    <line x1="${x(m.best_epoch)}" x2="${x(m.best_epoch)}" y1="${P.t}" y2="${H - P.b}" class="kept"/>
    <text x="${x(m.best_epoch) + 5}" y="${P.t + 14}" class="kept-t">kept</text>
    <path d="${path(tr)}" fill="none" stroke="${COL.train}" stroke-width="2" stroke-linejoin="round"/>
    <path d="${path(va)}" fill="none" stroke="${COL.validation}" stroke-width="2" stroke-linejoin="round"/>
    ${va.map((p) => `<circle cx="${x(p[0])}" cy="${y(p[1])}" r="4" fill="${COL.validation}" stroke="var(--panel)" stroke-width="2"/>`).join("")}
    <line class="xh" y1="${P.t}" y2="${H - P.b}" visibility="hidden"/>
  </svg><div class="tip-box" hidden></div>`;
  const svg = el.querySelector("svg"), xh = svg.querySelector(".xh"), tip = el.querySelector(".tip-box");
  const at = (pts, e) => pts.reduce((b, p) => (Math.abs(p[0] - e) < Math.abs(b[0] - e) ? p : b));
  const show = (ev) => {
    const r = svg.getBoundingClientRect(), e = Math.max(0, Math.min(xMax, ((ev.clientX - r.left - P.l) / (W - P.l - P.r)) * xMax));
    const v = at(va, e), t = at(tr, v[0]);
    xh.setAttribute("x1", x(v[0])); xh.setAttribute("x2", x(v[0])); xh.setAttribute("visibility", "visible");
    tip.hidden = false;
    tip.innerHTML = `epoch ${v[0]}<br><i style="background:${COL.train}"></i>train ${t[1].toFixed(3)}<br><i style="background:${COL.validation}"></i>validation ${v[1].toFixed(3)}<br>validation EER ${pct(v[2])}`;
    tip.style.left = `${Math.min(x(v[0]) + 8, W - tip.offsetWidth - 4)}px`;
  };
  svg.addEventListener("pointermove", show);
  svg.addEventListener("pointerdown", show);
  svg.addEventListener("pointerleave", () => { xh.setAttribute("visibility", "hidden"); tip.hidden = true; });
}

// ---------- Try tab ----------
let cur = null; // { name, label?, audio: Float32Array, url }
let ortP = null, recState = null, starting = false;
const sessions = {};

function loadOrt() {
  ortP = ortP || new Promise((res, rej) => {
    const s = document.createElement("script");
    s.src = `${ORT}ort.min.js`;
    s.onload = () => { window.ort.env.wasm.wasmPaths = ORT; window.ort.env.wasm.numThreads = 1; res(window.ort); };
    s.onerror = () => { ortP = null; rej(new Error("could not load the model runtime")); };
    document.head.appendChild(s);
  });
  return ortP;
}
async function session(id) {
  const ort = await loadOrt();
  sessions[id] = sessions[id] || ort.InferenceSession.create(`models/${id}.onnx`, { executionProviders: ["wasm"] });
  return sessions[id];
}
function windows(x) {
  if (x.length <= N) { const w = new Float32Array(N); for (let i = 0; i < N; i++) w[i] = x[i % x.length]; return [w]; }
  const starts = [];
  for (let s = 0; s <= x.length - N; s += N) starts.push(s);
  if (starts[starts.length - 1] !== x.length - N) starts.push(x.length - N);
  return starts.map((s) => x.subarray(s, s + N));
}
export async function scoreAudio(id, audio) {
  const [ort, s] = [await loadOrt(), await session(id)];
  const w = windows(audio), data = new Float32Array(w.length * N);
  w.forEach((v, i) => data.set(v, i * N));
  const out = await s.run({ wav: new ort.Tensor("float32", data, [w.length, N]) });
  const v = out.score.data;
  return v.reduce((a, b) => a + b, 0) / v.length;
}

export async function decode16k(buf) {
  const ab = await new OfflineAudioContext(1, 1, SR).decodeAudioData(buf);
  const x = new Float32Array(ab.length);
  for (let c = 0; c < ab.numberOfChannels; c++) { const d = ab.getChannelData(c); for (let i = 0; i < x.length; i++) x[i] += d[i] / ab.numberOfChannels; }
  return x;
}
// Recordings only: cut silence at the ends (20 ms frames more than 35 dB under the loudest, 30 ms margin),
// then scale to the training clips' median loudness. Test clips were trimmed like training already.
export function prepRecording(x, targetDb) {
  const F = 320, n = Math.floor(x.length / F), db = new Float32Array(n);
  for (let f = 0; f < n; f++) { let e = 0; for (let i = f * F; i < (f + 1) * F; i++) e += x[i] * x[i]; db[f] = 10 * Math.log10(e / F + 1e-12); }
  const top = Math.max(...db);
  let a = 0, b = n - 1;
  while (a < n && db[a] < top - 35) a++;
  while (b > a && db[b] < top - 35) b--;
  const y = x.slice(Math.max(0, a * F - 480), Math.min(x.length, (b + 1) * F + 480));
  if (top < -60 || y.length < 0.3 * SR) return null;
  let e = 0, peak = 0;
  for (const v of y) { e += v * v; peak = Math.max(peak, Math.abs(v)); }
  const g = Math.min(10 ** ((targetDb - 10 * Math.log10(e / y.length + 1e-12)) / 20), 0.99 / peak);
  for (let i = 0; i < y.length; i++) y[i] *= g;
  return y;
}

let tryReady = false;
export async function showTry() {
  const d = await initModels();
  if (!d || !d.models.length) { $("t-slot").innerHTML = `<p class="t-empty">No trained model yet.</p>`; return; }
  if (tryReady) return;
  tryReady = true;
  $("t-model").innerHTML = d.models.map((m) => `<option value="${m.id}">${esc(m.name)}</option>`).join("");
  $("t-model").addEventListener("change", () => cur && run());
  const groups = [["train", "Training data (seen)"], ["wild", "In the wild (never heard)"]];
  $("t-clips").innerHTML = groups.map(([g, title]) => `<p class="t-grp">${title}</p><div class="t-grid">${d.clips.filter((c) => c.group === g).map((c) => `
    <button class="t-tile" data-id="${c.id}"><span class="lab ${c.label === "AI" ? "ai" : "human"}">${c.label}</span><span>${esc(c.name)}</span></button>`).join("")}</div>`).join("");
  bindDrag(d);
  $("t-rec").addEventListener("click", () => { if (recState) stopRec(); else if (!starting) startRec(d); });
  loadOrt().catch(() => {}); // start the download early
}

function bindDrag(d) {
  const slot = $("t-slot");
  let drag = null;
  const over = (e) => { const r = slot.getBoundingClientRect(); return e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom; };
  $("t-clips").addEventListener("pointerdown", (e) => {
    const t = e.target.closest(".t-tile"); if (!t) return;
    stopRec(true); // picking a clip ends a recording at once
    drag = { t, x0: e.clientX, y0: e.clientY, ghost: null };
    try { t.setPointerCapture(e.pointerId); } catch { /* fine */ }
  });
  $("t-clips").addEventListener("pointermove", (e) => {
    if (!drag) return;
    if (!drag.ghost && Math.hypot(e.clientX - drag.x0, e.clientY - drag.y0) > 8) {
      drag.ghost = drag.t.cloneNode(true); drag.ghost.classList.add("ghost"); document.body.appendChild(drag.ghost);
    }
    if (drag.ghost) {
      drag.ghost.style.left = `${e.clientX}px`; drag.ghost.style.top = `${e.clientY}px`;
      slot.classList.toggle("over", over(e));
    }
  });
  const end = (e, cancel) => {
    if (!drag) return;
    const { t, ghost } = drag; drag = null;
    if (ghost) ghost.remove();
    slot.classList.remove("over");
    if (!cancel && (!ghost || over(e))) loadClip(d.clips.find((c) => c.id === t.dataset.id)); // a tap also loads
  };
  $("t-clips").addEventListener("pointerup", (e) => end(e, false));
  $("t-clips").addEventListener("pointercancel", (e) => end(e, true));
  $("t-clips").addEventListener("click", (e) => { // keyboard (Enter / Space) only; pointers use pointerup
    const t = e.target.closest(".t-tile");
    if (t && e.detail === 0) loadClip(d.clips.find((c) => c.id === t.dataset.id));
  });
}

async function loadClip(c) {
  stopRec(true);
  setSlot(`<p class="t-empty">Loading ${esc(c.name)}…</p>`);
  try {
    const buf = await (await fetch(`try/${c.id}.wav`)).arrayBuffer();
    cur = { name: c.name, label: c.label, sub: c.group === "train" ? `training data · ${c.system === "-" ? "human" : c.system}` : "In-the-Wild",
      audio: await decode16k(buf), url: `try/${c.id}.wav`, python: c.scores };
    document.querySelectorAll(".t-tile").forEach((t) => t.classList.toggle("on", t.dataset.id === c.id));
    run();
  } catch (err) { setSlot(`<p class="t-empty">Could not load the clip: ${esc(err.message)}</p>`); }
}

// Recording: raw samples straight from the mic, not a compressed MediaRecorder file (the models never heard
// codec audio). Tested: test clips played into a fake mic over a quiet noise floor keep their verdicts.
// An AnalyserNode drives the live waveform.
const MAX_S = 8, COLS_PER_S = 25; // waveform: one column per 40 ms, 8 s across
async function startRec(d) {
  const ac = new AudioContext(); // made inside the tap, so Safari lets it run
  starting = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    const st = { d, stream, ac, pcm: [], raf: 0, timer: 0, t0: 0, last: -1, cols: new Float32Array(MAX_S * COLS_PER_S) };
    const src = ac.createMediaStreamSource(stream), an = ac.createAnalyser(), proc = ac.createScriptProcessor(4096, 1, 1);
    an.fftSize = 2048;
    proc.onaudioprocess = (e) => { if (recState === st) st.pcm.push(new Float32Array(e.inputBuffer.getChannelData(0))); };
    src.connect(an);
    src.connect(proc);
    proc.connect(ac.destination); // a processor only runs when connected; its output stays silent
    st.proc = proc;
    await ac.resume();
    setSlot(`<div class="t-head"><p class="t-rec-note">Recording… speak for 2–8 seconds.</p>${CLEAR}</div><canvas id="t-live" aria-label="live waveform of the recording"></canvas>`);
    $("t-slot").querySelector(".t-clear").onclick = () => { stopRec(true); clearSlot(); };
    document.querySelectorAll(".t-tile").forEach((t) => t.classList.remove("on"));
    st.t0 = performance.now();
    st.timer = setTimeout(() => recState === st && stopRec(), MAX_S * 1000); // also when the tab is hidden
    recState = st;
    starting = false;
    $("t-rec").classList.add("on");
    const buf = new Float32Array(an.fftSize);
    const frame = () => {
      if (recState !== st) return;
      const sec = (performance.now() - st.t0) / 1000;
      an.getFloatTimeDomainData(buf);
      let e = 0;
      for (const v of buf) e += v * v;
      const rms = Math.sqrt(e / buf.length);
      const k = Math.min(st.cols.length - 1, Math.floor(sec * COLS_PER_S));
      for (let i = st.last + 1; i < k; i++) st.cols[i] = rms; // frames can skip columns
      st.cols[k] = Math.max(st.cols[k], rms);
      st.last = k;
      drawLive($("t-live"), st.cols, Math.min(sec, MAX_S));
      $("t-rec-txt").textContent = `Stop (${Math.min(sec, MAX_S).toFixed(1)} s)`;
      st.raf = requestAnimationFrame(frame);
    };
    frame();
  } catch (err) {
    starting = false;
    ac.close().catch(() => {});
    setSlot(`<p class="t-empty">The microphone is off (${esc(err.message)}). Allow it for this site, then try again.</p>`);
  }
}
// discard = true: drop the audio (a clip was picked, or the tab closed).
function stopRec(discard = false) {
  const st = recState;
  if (!st) return;
  recState = null;
  cancelAnimationFrame(st.raf);
  clearTimeout(st.timer);
  st.proc.disconnect();
  st.stream.getTracks().forEach((t) => t.stop());
  $("t-rec").classList.remove("on");
  $("t-rec-txt").textContent = "Record";
  if (!discard) finishRec(st);
  else st.ac.close().catch(() => {});
}
async function finishRec(st) {
  try {
    const n = st.pcm.reduce((a, b) => a + b.length, 0), raw = new Float32Array(n);
    let o = 0;
    for (const b of st.pcm) { raw.set(b, o); o += b.length; }
    const sr = st.ac.sampleRate;
    st.ac.close().catch(() => {});
    const x = n ? prepRecording(await resample16k(raw, sr), st.d.target_rms_db) : null;
    if (!x) { setSlot(`<p class="t-empty">No speech heard. Try again, closer to the mic.</p>`); return; }
    cur = { name: "Your recording", label: null, sub: `${(x.length / SR).toFixed(1)} s after trimming silence`, audio: x, url: wavURL(x) };
    run();
  } catch (err) { setSlot(`<p class="t-empty">Could not read the recording: ${esc(err.message)}</p>`); }
}
async function resample16k(x, sr) {
  if (sr === SR) return x;
  const oc = new OfflineAudioContext(1, Math.ceil((x.length * SR) / sr), SR);
  const b = oc.createBuffer(1, x.length, sr);
  b.copyToChannel(x, 0);
  const s = oc.createBufferSource();
  s.buffer = b;
  s.connect(oc.destination);
  s.start();
  return (await oc.startRendering()).getChannelData(0);
}
// 16 kHz mono 16-bit WAV, so "play" gives exactly what the model heard.
function wavURL(x) {
  const v = new DataView(new ArrayBuffer(44 + x.length * 2));
  const w = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
  w(0, "RIFF"); v.setUint32(4, 36 + x.length * 2, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true); v.setUint32(24, SR, true);
  v.setUint32(28, SR * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true); w(36, "data"); v.setUint32(40, x.length * 2, true);
  for (let i = 0; i < x.length; i++) v.setInt16(44 + i * 2, Math.round(Math.max(-1, Math.min(1, x[i])) * 32767), true);
  return URL.createObjectURL(new Blob([v.buffer], { type: "audio/wav" }));
}
// Live waveform: loudness (RMS) per 40 ms column on a 0-8 s axis, one mark per second, playhead at "now".
function drawLive(cv, cols, sec) {
  if (!cv) return;
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const dpr = window.devicePixelRatio || 1, W = cv.clientWidth, H = cv.clientHeight, AX = 22;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  g.clearRect(0, 0, W, H);
  const mid = (H - AX) / 2, cw = W / cols.length;
  g.font = `15px ${css("--mono")}`;
  g.textBaseline = "alphabetic";
  for (let t = 0; t <= MAX_S; t++) {
    const x = Math.min(W - 1, Math.round((t / MAX_S) * W));
    g.fillStyle = css("--edge");
    g.fillRect(x, 0, 1, H - AX);
    g.fillStyle = css("--ink-2");
    g.textAlign = t === 0 ? "left" : t === MAX_S ? "right" : "center";
    g.fillText(t === MAX_S ? `${t} s` : String(t), x, H - 5);
  }
  g.fillStyle = css("--edge");
  g.fillRect(0, Math.round(mid), W, 1);
  const level = (r) => Math.max(0, Math.min(1, (20 * Math.log10(r + 1e-6) + 60) / 54)); // -60..-6 dBFS RMS
  g.fillStyle = css("--gold");
  const n = Math.min(cols.length, Math.floor(sec * COLS_PER_S) + 1);
  for (let i = 0; i < n; i++) {
    const a = Math.max(1, level(cols[i]) * (mid - 3));
    g.fillRect(i * cw, mid - a, Math.max(1, cw - 1), 2 * a);
  }
  g.fillStyle = css("--ink");
  g.fillRect(Math.min(W - 2, (sec / MAX_S) * W), 0, 2, H - AX);
}

function setSlot(html) { $("t-slot").innerHTML = html; }
const EMPTY = `<p class="t-empty">Drag a clip here, tap one, or record your voice.</p>`;
const CLEAR = `<button class="btn btn-sm t-clear" aria-label="clear the recording" title="clear">×</button>`;
// Empty the slot: stop a pending run and playback, free the recording.
function clearSlot() {
  runId++;
  const A = $("t-audio");
  A.pause();
  A.removeAttribute("src");
  A.dataset.src = "";
  if (cur && cur.url && cur.url.startsWith("blob:")) URL.revokeObjectURL(cur.url);
  cur = null;
  setSlot(EMPTY);
}
let runId = 0;
async function run() {
  const d = await initModels(), m = d.models.find((x) => x.id === $("t-model").value), c = cur, id = ++runId;
  const head = `<div class="t-head">${c.label ? `<span class="lab ${c.label === "AI" ? "ai" : "human"}">${c.label}</span>` : ""}<b>${esc(c.name)}</b>
    <button class="btn btn-sm t-play" aria-label="play">▶</button>${c.label ? "" : CLEAR}</div><p class="t-sub">${esc(c.sub)}</p>`;
  setSlot(`${head}<p class="t-wait">${ortP ? "Running" : "Loading"} ${esc(m.name)}…</p>`);
  bindPlay(c);
  try {
    const s = await scoreAudio(m.id, c.audio);
    if (id !== runId) return;
    const p = 1 / (1 + Math.exp(-(s - m.threshold))), ai = s > m.threshold, right = c.label ? ai === (c.label === "AI") : null;
    setSlot(`${head}
      <div class="t-score ${ai ? "a" : "h"}"><b>${p > 0.99 ? "99%+" : p < 0.01 ? "<1%" : `${Math.round(p * 100)}%`}</b> AI</div>
      <div class="meter"><span class="${ai ? "a" : "h"}" style="width:${p * 100}%"></span><i></i></div>
      <p class="t-verdict">${esc(m.name)} says <b class="${ai ? "a" : "h"}">${ai ? "AI" : "human"}</b>${right === null ? "" : right ? ` · <b class="ok">✓ right</b>` : ` · <b class="no">✗ wrong</b>`}</p>`);
    bindPlay(c);
    if (c.python && c.python[m.id] != null) $("t-slot").dataset.diff = Math.abs(s - c.python[m.id]).toFixed(5); // parity check (dev)
  } catch (err) {
    if (id === runId) setSlot(`${head}<p class="t-empty">The model did not run: ${esc(err.message)}</p>`);
  }
}
function bindPlay(c) {
  const x = $("t-slot").querySelector(".t-clear");
  if (x) x.onclick = clearSlot;
  const b = $("t-slot").querySelector(".t-play"), A = $("t-audio");
  if (!b) return;
  b.onclick = () => {
    if (A.dataset.src !== c.url) { A.src = c.url; A.dataset.src = c.url; }
    A.paused ? A.play().catch(() => {}) : A.pause();
  };
  A.onplay = () => { const x = $("t-slot").querySelector(".t-play"); if (x) x.textContent = "❚❚"; };
  A.onpause = A.onended = () => { const x = $("t-slot").querySelector(".t-play"); if (x) x.textContent = "▶"; };
}
export function leaveTry() { $("t-audio").pause(); stopRec(true); }

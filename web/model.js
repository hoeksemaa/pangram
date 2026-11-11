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
let ortP = null, recState = null;
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
  $("t-rec").addEventListener("click", () => (recState ? stopRec() : startRec(d)));
  loadOrt().catch(() => {}); // start the download early
}

function bindDrag(d) {
  const slot = $("t-slot");
  let drag = null;
  const over = (e) => { const r = slot.getBoundingClientRect(); return e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom; };
  $("t-clips").addEventListener("pointerdown", (e) => {
    const t = e.target.closest(".t-tile"); if (!t) return;
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
}

async function loadClip(c) {
  setSlot(`<p class="t-empty">Loading ${esc(c.name)}…</p>`);
  try {
    const buf = await (await fetch(`try/${c.id}.wav`)).arrayBuffer();
    cur = { name: c.name, label: c.label, sub: c.group === "train" ? `training data · ${c.system === "-" ? "human" : c.system}` : "In-the-Wild",
      audio: await decode16k(buf), url: `try/${c.id}.wav`, python: c.scores };
    document.querySelectorAll(".t-tile").forEach((t) => t.classList.toggle("on", t.dataset.id === c.id));
    run();
  } catch (err) { setSlot(`<p class="t-empty">Could not load the clip: ${esc(err.message)}</p>`); }
}

async function startRec(d) {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    const rec = new MediaRecorder(stream), chunks = [];
    rec.ondataavailable = (e) => chunks.push(e.data);
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunks, { type: rec.mimeType });
      try {
        const x = prepRecording(await decode16k(await blob.arrayBuffer()), d.target_rms_db);
        if (!x) { setSlot(`<p class="t-empty">No speech heard. Try again, closer to the mic.</p>`); return; }
        document.querySelectorAll(".t-tile").forEach((t) => t.classList.remove("on"));
        cur = { name: "Your recording", label: null, sub: `${(x.length / SR).toFixed(1)} s after trimming silence`, audio: x, url: URL.createObjectURL(blob) };
        run();
      } catch (err) { setSlot(`<p class="t-empty">Could not read the recording: ${esc(err.message)}</p>`); }
    };
    const t0 = Date.now();
    recState = { rec, timer: setInterval(() => {
      const s = (Date.now() - t0) / 1000;
      $("t-rec-txt").textContent = `Stop (${s.toFixed(0)} s)`;
      if (s >= 8) stopRec();
    }, 200) };
    rec.start();
    $("t-rec").classList.add("on");
    $("t-rec-txt").textContent = "Stop (0 s)";
    setSlot(`<p class="t-empty">Recording… speak for 2–8 seconds.</p>`);
  } catch (err) {
    setSlot(`<p class="t-empty">The microphone is off (${esc(err.message)}). Allow it for this site, then try again.</p>`);
  }
}
function stopRec() {
  if (!recState) return;
  clearInterval(recState.timer);
  if (recState.rec.state !== "inactive") recState.rec.stop();
  recState = null;
  $("t-rec").classList.remove("on");
  $("t-rec-txt").textContent = "Record";
}

function setSlot(html) { $("t-slot").innerHTML = html; }
let runId = 0;
async function run() {
  const d = await initModels(), m = d.models.find((x) => x.id === $("t-model").value), c = cur, id = ++runId;
  const head = `<div class="t-head">${c.label ? `<span class="lab ${c.label === "AI" ? "ai" : "human"}">${c.label}</span>` : ""}<b>${esc(c.name)}</b>
    <button class="btn btn-sm t-play" aria-label="play">▶</button></div><p class="t-sub">${esc(c.sub)}</p>`;
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
  const b = $("t-slot").querySelector(".t-play"), A = $("t-audio");
  if (!b) return;
  b.onclick = () => {
    if (A.dataset.src !== c.url) { A.src = c.url; A.dataset.src = c.url; }
    A.paused ? A.play().catch(() => {}) : A.pause();
  };
  A.onplay = () => { const x = $("t-slot").querySelector(".t-play"); if (x) x.textContent = "❚❚"; };
  A.onpause = A.onended = () => { const x = $("t-slot").querySelector(".t-play"); if (x) x.textContent = "▶"; };
}
export function leaveTry() { $("t-audio").pause(); stopRec(); }

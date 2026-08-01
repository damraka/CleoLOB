/* CLEO — LOB Execution Lab · frontend
   Sections: dom/state · controls · run flow · data prep · three (3D canyon)
             canvas charts · tape/cards/table · playback · export · boot     */
"use strict";
console.log("CLEO ui v3");

/* ---------------------------------------------------------------- dom/state */
const $ = (id) => document.getElementById(id);
const els = {
  led: $("led"), modelChip: $("modelChip"), engineChip: $("engineChip"),
  runBtn: $("runBtn"), heroRun: $("heroRun"), preset: $("preset"),
  qty: $("qty"), horizon: $("horizon"), horizonV: $("horizonV"), dt: $("dt"),
  seed: $("seed"), dice: $("dice"), latency: $("latency"), latencyV: $("latencyV"),
  resil: $("resil"), resilV: $("resilV"), mrate: $("mrate"), mrateV: $("mrateV"),
  lambda: $("lambda"), modelPath: $("modelPath"),
  viewBase: $("viewBase"), viewRL: $("viewRL"),
  play: $("play"), speed: $("speed"), scrub: $("scrub"), clock: $("clock"),
  exportPng: $("exportPng"), exportJson: $("exportJson"),
  emptyHero: $("emptyHero"), loading: $("loading"),
  loadLabel: $("loadLabel"), loadBar: $("loadBar"), axisLegend: $("axisLegend"),
  cards: $("cards"), row1: $("row1"), row2: $("row2"), tablePanel: $("tablePanel"),
  cArrival: $("cArrival"), cArrivalS: $("cArrivalS"), cAC: $("cAC"), cACS: $("cACS"),
  cRL: $("cRL"), cRLk: $("cRLk"), cDelta: $("cDelta"), cFill: $("cFill"), cFillS: $("cFillS"),
  trajChart: $("trajChart"), trajRead: $("trajRead"),
  midChart: $("midChart"), midRead: $("midRead"),
  depthChart: $("depthChart"), depthT: $("depthT"), depthMeta: $("depthMeta"),
  tape: $("tape"), tapeMeta: $("tapeMeta"), cmp: $("cmp"), toast: $("toast"),
};

const COLORS = {
  bid: "#2DD4A7", ask: "#F4586E", ac: "#5B8DEF", rl: "#F5B84B",
  ink: "#E8ECF4", mut: "#8C96AB", dim: "#5C6478", line: "#232B3D",
};

const state = {
  result: null,        // raw payload from /api/result
  prep: null,          // {baseline, rl} prepared grids
  active: "baseline",  // which run drives 3D + depth + tape
  cursor: 0,           // playback position in steps (float)
  playing: false,
  three: null,         // three.js context or null
};

/* ----------------------------------------------------------------- controls */
const fmt$ = (v) => "$" + v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const fmt = (v, d = 1) => v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });

function bindLive(input, label, suffix) {
  const upd = () => (label.textContent = input.value + suffix);
  input.addEventListener("input", upd); upd();
}
bindLive(els.horizon, els.horizonV, " s");
bindLive(els.latency, els.latencyV, " ms");
bindLive(els.resil, els.resilV, "");
bindLive(els.mrate, els.mrateV, " /s");

els.dice.addEventListener("click", () => {
  els.seed.value = Math.floor(Math.random() * 100000);
});

const PRESETS = {
  calm:      { latency: 10, resil: 0.8, mrate: 6,  lambda: "1e-6", dt: "0.5" },
  stressed:  { latency: 20, resil: 0.3, mrate: 14, lambda: "1e-6", dt: "0.5" },
  hft:       { latency: 2,  resil: 0.8, mrate: 6,  lambda: "1e-6", dt: "0.25" },
  impatient: { latency: 10, resil: 0.8, mrate: 6,  lambda: "1e-4", dt: "0.5" },
};
els.preset.addEventListener("change", () => {
  const p = PRESETS[els.preset.value]; if (!p) return;
  els.latency.value = p.latency; els.resil.value = p.resil; els.mrate.value = p.mrate;
  els.lambda.value = p.lambda; els.dt.value = p.dt;
  for (const inp of [els.latency, els.resil, els.mrate]) inp.dispatchEvent(new Event("input"));
});

function readParams() {
  return {
    qty: +els.qty.value, horizon: +els.horizon.value, dt: +els.dt.value,
    seed: +els.seed.value, latency_ms: +els.latency.value,
    resilience: +els.resil.value, market_rate: +els.mrate.value,
    risk_aversion: +els.lambda.value, model_path: els.modelPath.value.trim(),
  };
}

function toast(msg) {
  els.toast.textContent = msg;
  els.toast.classList.remove("hidden");
  clearTimeout(toast._t);
  toast._t = setTimeout(() => els.toast.classList.add("hidden"), 6000);
}

/* ----------------------------------------------------------------- run flow */
let running = false;

async function runPair() {
  if (running) return;
  running = true;
  els.runBtn.disabled = true; els.runBtn.textContent = "Simulating…";
  els.led.className = "led busy";
  els.engineChip.textContent = "engine · simulating";
  els.emptyHero.classList.add("hidden");
  els.loading.classList.remove("hidden");
  els.loadBar.style.width = "0%"; els.loadLabel.textContent = "warming the book…";
  stopPlayback();
  $("scrollArea").scrollTop = 0;
  window.scrollTo(0, 0);

  try {
    const r = await fetch("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readParams()),
    });
    if (!r.ok) throw new Error((await r.json()).detail?.[0]?.msg || "bad parameters");
    const { job_id } = await r.json();

    let job;
    do {
      await new Promise((res) => setTimeout(res, 220));
      job = await (await fetch(`/api/job/${job_id}`)).json();
      const phase = job.phase === "rl" ? "RL agent" : "Almgren-Chriss baseline";
      const pct = (job.phase === "rl" ? 0.5 : 0) + (job.pct || 0) * 0.5;
      els.loadLabel.textContent = `simulating ${phase} · ${Math.round((job.pct || 0) * 100)}%`;
      els.loadBar.style.width = `${Math.round(pct * 100)}%`;
    } while (job.status === "running");
    if (job.status === "error") throw new Error(job.error);

    state.result = await (await fetch(`/api/result/${job_id}`)).json();
    onResult();
  } catch (err) {
    toast("Run failed — " + err.message);
    els.engineChip.textContent = "engine · error";
    els.led.className = "led";
    if (!state.result) els.emptyHero.classList.remove("hidden");
  } finally {
    els.loading.classList.add("hidden");
    els.runBtn.disabled = false; els.runBtn.textContent = "Run head-to-head";
    running = false;
  }
}
els.runBtn.addEventListener("click", runPair);
els.heroRun.addEventListener("click", runPair);

function onResult() {
  const res = state.result;
  state.prep = { baseline: prepRun(res.runs.baseline, res), rl: prepRun(res.runs.rl, res) };
  state.cursor = 0;

  els.led.className = "led ok";
  els.engineChip.textContent = "engine · done";
  const rlLabel = res.runs.rl.label;
  const isPPO = rlLabel.includes("PPO");
  els.modelChip.textContent = isPPO ? "model · PPO loaded" : "model · heuristic fallback";
  els.modelChip.className = "chip " + (isPPO ? "good" : "warn");
  if (!isPPO) toast("No trained PPO model found — RL side is a heuristic. Train with: python train_rl.py");

  for (const el of [els.cards, els.row1, els.row2, els.tablePanel]) el.classList.remove("hidden");
  els.axisLegend.classList.remove("hidden");
  for (const el of [els.play, els.speed, els.scrub, els.exportPng]) el.disabled = false;

  const steps = maxSteps();
  els.scrub.max = String(steps - 1);
  els.scrub.step = "0.01";

  renderCards(); renderTable();
  buildThree();
  resizeAll();
  setCursor(0);
  state.playing = true; els.play.textContent = "❚❚";   // autoplay one pass
  requestAnimationFrame(() => requestAnimationFrame(() => { $("scrollArea").scrollTop = 0; }));
}

/* ---------------------------------------------------------------- data prep */
function nearestIdx(arr, t) {
  let lo = 0, hi = arr.length - 1;
  while (lo < hi) { const m = (lo + hi) >> 1; if (arr[m] < t) lo = m + 1; else hi = m; }
  if (lo > 0 && Math.abs(arr[lo - 1] - t) <= Math.abs(arr[lo] - t)) lo--;
  return lo;
}

function prepRun(run, res) {
  const s = run.series, N = s.t.length;
  const mid0 = Math.round(run.arrival_ticks);

  let pmin = Infinity, pmax = -Infinity;
  for (let i = 0; i < N; i++) {
    for (const [p] of s.bids[i]) { if (p < pmin) pmin = p; }
    for (const [p] of s.asks[i]) { if (p > pmax) pmax = p; }
  }
  pmin = Math.max(pmin, mid0 - 45); pmax = Math.min(pmax, mid0 + 45);
  if (!isFinite(pmin) || !isFinite(pmax) || pmax <= pmin) { pmin = mid0 - 10; pmax = mid0 + 10; }
  const cols = pmax - pmin + 1;

  const bidCum = new Float32Array(N * cols);
  const askCum = new Float32Array(N * cols);
  let maxCum = 1;
  for (let i = 0; i < N; i++) {
    let acc = 0;
    for (const [p, v] of s.bids[i]) {            // best-first, price descending
      if (p > pmax) continue;
      acc += v;
      if (p >= pmin) bidCum[i * cols + (p - pmin)] = acc;
    }
    if (acc > maxCum) maxCum = acc;
    // fill plateau between listed levels (step function)
    let last = 0;
    for (let c = cols - 1; c >= 0; c--) {
      const v = bidCum[i * cols + c];
      if (v > 0) last = v; else if (last > 0 && pmin + c > (s.bids[i].length ? s.bids[i][s.bids[i].length - 1][0] : Infinity)) bidCum[i * cols + c] = last;
    }
    acc = 0;
    for (const [p, v] of s.asks[i]) {            // best-first, price ascending
      if (p < pmin) continue;
      acc += v;
      if (p <= pmax) askCum[i * cols + (p - pmin)] = acc;
    }
    if (acc > maxCum) maxCum = acc;
    last = 0;
    for (let c = 0; c < cols; c++) {
      const v = askCum[i * cols + c];
      if (v > 0) last = v; else if (last > 0 && pmin + c < (s.asks[i].length ? s.asks[i][s.asks[i].length - 1][0] : -Infinity)) askCum[i * cols + c] = last;
    }
  }

  let maxFill = 1;
  for (const f of run.fills) if (f.qty > maxFill) maxFill = f.qty;

  return { run, N, cols, pmin, pmax, mid0, bidCum, askCum, maxCum, maxFill };
}

const maxSteps = () => Math.max(state.prep.baseline.N, state.prep.rl.N);
const horizonT = () => +state.result.params.horizon;
const activePrep = () => state.prep[state.active];

/* --------------------------------------------------------------- three (3D) */
function buildThree(attempt = 0) {
  if (!window.THREE) {                    // CDN may still be loading (or fallback CDN kicking in)
    if (attempt < 12) { setTimeout(() => buildThree(attempt + 1), 300); return; }
    if (!buildThree._warned) {
      buildThree._warned = true;
      const msg = document.createElement("div");
      msg.className = "three-missing";
      msg.textContent = "3D view unavailable — three.js could not be loaded from any CDN "
        + "(check network / ad-blocker). All 2D panels remain live.";
      $("three-wrap").appendChild(msg);
    }
    return;
  }
  if (!state.three) initThree();
  rebuildScene();
  syncCursor3D();
}

function initThree() {
  const wrap = $("three-wrap"), canvas = $("gl");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(46, 1, 0.1, 2000);
  scene.add(new THREE.AmbientLight(0xffffff, 0.62));
  const sun = new THREE.DirectionalLight(0xffffff, 0.72);
  sun.position.set(60, 140, 50); scene.add(sun);

  const root = new THREE.Group(); scene.add(root);
  const orbit = { theta: 0.9, phi: 1.02, r: 235, tx: 0, ty: 14, tz: 0 };

  const t = { renderer, scene, camera, root, orbit, wrap, canvas, cursorGroup: null, spanX: 100 };
  state.three = t;

  // -- minimal orbit controls (drag rotate · wheel zoom · dblclick reset)
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => { drag = { x: e.clientX, y: e.clientY }; canvas.setPointerCapture(e.pointerId); });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    orbit.theta -= (e.clientX - drag.x) * 0.0055;
    orbit.phi = Math.min(1.45, Math.max(0.15, orbit.phi - (e.clientY - drag.y) * 0.0045));
    drag = { x: e.clientX, y: e.clientY };
  });
  canvas.addEventListener("pointerup", () => (drag = null));
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    orbit.r = Math.min(600, Math.max(70, orbit.r * (1 + e.deltaY * 0.0012)));
  }, { passive: false });
  canvas.addEventListener("dblclick", () => Object.assign(orbit, { theta: 0.9, phi: 1.02, r: 235 }));

  new ResizeObserver(() => {
    const w = wrap.clientWidth, h = wrap.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h; camera.updateProjectionMatrix();
  }).observe(wrap);
}

function disposeGroup(g) {
  g.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
  });
  g.clear();
}

function rebuildScene() {
  const t = state.three; if (!t) return;
  disposeGroup(t.root);

  const P = activePrep();
  const { N, cols, pmin, mid0, bidCum, askCum, maxCum, maxFill } = P;
  const H = 52;                                  // max terrace height (world units)
  const DX = 172 / Math.max(N - 1, 1);           // time spacing
  const DZ = Math.min(2.4, 132 / Math.max(cols - 1, 1)); // price spacing
  const YS = H / maxCum;
  t.spanX = DX * (N - 1);
  t.DX = DX;

  const mkSurface = (cum, hexLow, hexHigh) => {
    const pos = new Float32Array(N * cols * 3);
    const col = new Float32Array(N * cols * 3);
    const cLow = new THREE.Color(hexLow), cHigh = new THREE.Color(hexHigh), cc = new THREE.Color();
    for (let i = 0; i < N; i++) for (let c = 0; c < cols; c++) {
      const k = i * cols + c, y = cum[k] * YS;
      pos[3 * k] = i * DX; pos[3 * k + 1] = y; pos[3 * k + 2] = (pmin + c - mid0) * DZ;
      cc.copy(cLow).lerp(cHigh, Math.sqrt(Math.min(1, y / H)));
      col[3 * k] = cc.r; col[3 * k + 1] = cc.g; col[3 * k + 2] = cc.b;
    }
    const idx = [];
    for (let i = 0; i < N - 1; i++) for (let c = 0; c < cols - 1; c++) {
      const a = i * cols + c, b = a + 1, d = a + cols, e = d + 1;
      if (cum[a] || cum[b] || cum[d] || cum[e]) idx.push(a, d, b, b, d, e);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
    geo.setIndex(idx); geo.computeVertexNormals();
    return new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide }));
  };
  t.root.add(mkSurface(bidCum, "#0E5C49", COLORS.bid));
  t.root.add(mkSurface(askCum, "#6E1F2C", COLORS.ask));

  // mid-price path on the valley floor
  const s = P.run.series;
  const mids = [];
  for (let i = 0; i < N; i++) mids.push(new THREE.Vector3(i * DX, 1.1, (s.mid[i] - mid0) * DZ));
  t.root.add(new THREE.Line(
    new THREE.BufferGeometry().setFromPoints(mids),
    new THREE.LineBasicMaterial({ color: 0xdce4ff })));

  // fill spikes (amber)
  const fills = P.run.fills;
  if (fills.length) {
    const inst = new THREE.InstancedMesh(
      new THREE.BoxGeometry(Math.max(0.5, DX * 0.35), 1, 0.8),
      new THREE.MeshBasicMaterial({ color: COLORS.rl, transparent: true, opacity: 0.85 }),
      fills.length);
    const dummy = new THREE.Object3D();
    const dtDec = state.result.decision_dt;
    for (let j = 0; j < fills.length; j++) {
      const f = fills[j];
      const i = Math.min(N - 1, Math.max(0, Math.round(f.t / dtDec)));
      const h = 2 + 15 * (f.qty / maxFill);
      dummy.position.set(i * DX, h / 2, (f.px - mid0) * DZ);
      dummy.scale.set(1, h, 1);
      dummy.updateMatrix();
      inst.setMatrixAt(j, dummy.matrix);
    }
    t.root.add(inst);
  }

  // time cursor plane
  const spanZ = (cols - 1) * DZ;
  const cg = new THREE.Group();
  const plane = new THREE.Mesh(
    new THREE.PlaneGeometry(spanZ * 1.04, H * 1.05),
    new THREE.MeshBasicMaterial({ color: COLORS.rl, transparent: true, opacity: 0.10, side: THREE.DoubleSide, depthWrite: false }));
  plane.rotation.y = Math.PI / 2; plane.position.y = H * 0.5;
  cg.add(plane);
  const eg = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0.4, -spanZ * 0.52), new THREE.Vector3(0, 0.4, spanZ * 0.52)]);
  cg.add(new THREE.Line(eg, new THREE.LineBasicMaterial({ color: COLORS.rl })));
  t.root.add(cg); t.cursorGroup = cg;

  const grid = new THREE.GridHelper(Math.max(t.spanX, spanZ) * 1.25, 24, 0x2a3450, 0x1a2233);
  grid.position.set(t.spanX / 2, -0.25, 0);
  t.root.add(grid);

  t.root.position.set(-t.spanX / 2, 0, 0);
  syncCursor3D();
}

function syncCursor3D() {
  const t = state.three; if (!t || !t.cursorGroup) return;
  t.cursorGroup.position.x = Math.min(state.cursor, activePrep().N - 1) * t.DX;
}

function renderThree() {
  const t = state.three; if (!t) return;
  const o = t.orbit;
  t.camera.position.set(
    o.tx + o.r * Math.sin(o.phi) * Math.cos(o.theta),
    o.ty + o.r * Math.cos(o.phi),
    o.tz + o.r * Math.sin(o.phi) * Math.sin(o.theta));
  t.camera.lookAt(o.tx, o.ty, o.tz);
  t.renderer.render(t.scene, t.camera);
}

/* ------------------------------------------------------------ canvas charts */
function ctx2d(cv) {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { ctx, w, h };
}

const PAD = { l: 54, r: 12, t: 10, b: 24 };

function frame(ctx, w, h, yTicks, xTicks, fmtY, fmtX) {
  ctx.clearRect(0, 0, w, h);
  ctx.font = "10.5px 'IBM Plex Mono', monospace";
  ctx.strokeStyle = COLORS.line; ctx.fillStyle = COLORS.dim; ctx.lineWidth = 1;
  for (const [v, y] of yTicks) {
    ctx.globalAlpha = 0.7; ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(w - PAD.r, y); ctx.stroke();
    ctx.globalAlpha = 1; ctx.textAlign = "right"; ctx.fillText(fmtY(v), PAD.l - 7, y + 3.5);
  }
  ctx.textAlign = "center";
  for (const [v, x] of xTicks) ctx.fillText(fmtX(v), x, h - 7);
}

function linTicks(min, max, n) {
  const out = [], span = max - min || 1;
  for (let i = 0; i <= n; i++) out.push(min + (span * i) / n);
  return out;
}

function drawCursorLine(ctx, x, h) {
  ctx.save();
  ctx.strokeStyle = COLORS.rl; ctx.globalAlpha = 0.65; ctx.setLineDash([4, 4]);
  ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, h - PAD.b); ctx.stroke();
  ctx.restore();
}

function drawTrajectory() {
  const { ctx, w, h } = ctx2d(els.trajChart);
  const T = horizonT(), qty = +state.result.params.qty;
  const X = (t) => PAD.l + ((w - PAD.l - PAD.r) * t) / T;
  const Y = (v) => PAD.t + (h - PAD.t - PAD.b) * (1 - v / qty);
  frame(ctx, w, h,
    linTicks(0, qty, 4).map((v) => [v, Y(v)]),
    linTicks(0, T, 6).map((v) => [v, X(v)]),
    (v) => (v >= 1000 ? (v / 1000) + "k" : v), (v) => v + "s");

  for (const key of ["baseline", "rl"]) {
    const s = state.prep[key].run.series;
    ctx.strokeStyle = key === "baseline" ? COLORS.ac : COLORS.rl;
    ctx.lineWidth = 2; ctx.beginPath();
    for (let i = 0; i < s.t.length; i++) {
      const x = X(s.t[i]), y = Y(s.remaining[i]);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
  }
  drawCursorLine(ctx, X(cursorT()), h);
}

function drawMid() {
  const { ctx, w, h } = ctx2d(els.midChart);
  const T = horizonT(), tick = state.result.tick;
  let lo = Infinity, hi = -Infinity;
  for (const key of ["baseline", "rl"])
    for (const m of state.prep[key].run.series.mid) { if (m < lo) lo = m; if (m > hi) hi = m; }
  const pad = Math.max(2, (hi - lo) * 0.15); lo -= pad; hi += pad;
  const X = (t) => PAD.l + ((w - PAD.l - PAD.r) * t) / T;
  const Y = (m) => PAD.t + (h - PAD.t - PAD.b) * (1 - (m - lo) / (hi - lo));
  frame(ctx, w, h,
    linTicks(lo, hi, 4).map((v) => [v, Y(v)]),
    linTicks(0, T, 6).map((v) => [v, X(v)]),
    (v) => (v * tick).toFixed(2), (v) => v + "s");

  for (const key of ["baseline", "rl"]) {
    const P = state.prep[key], s = P.run.series;
    const color = key === "baseline" ? COLORS.ac : COLORS.rl;
    ctx.strokeStyle = color; ctx.lineWidth = 1.6; ctx.beginPath();
    for (let i = 0; i < s.t.length; i++) {
      const x = X(s.t[i]), y = Y(s.mid[i]);
      i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = color; ctx.globalAlpha = 0.8;
    for (const f of P.run.fills) {
      ctx.beginPath();
      ctx.arc(X(f.t), Y(f.px), 1.4 + 2.4 * (f.qty / P.maxFill), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }
  drawCursorLine(ctx, X(cursorT()), h);
}

function drawDepth2D() {
  const { ctx, w, h } = ctx2d(els.depthChart);
  const P = activePrep(), s = P.run.series, tick = state.result.tick;
  const i = Math.min(P.N - 1, Math.round(state.cursor));
  const bids = s.bids[i], asks = s.asks[i];

  let cum = 0; const bpts = bids.map(([p, v]) => [p, (cum += v)]); const bmax = cum;
  cum = 0;     const apts = asks.map(([p, v]) => [p, (cum += v)]); const amax = cum;
  const ymax = Math.max(bmax, amax, 1) * 1.08;
  const plo = Math.min(bpts.length ? bpts[bpts.length - 1][0] : P.mid0 - 5,
                       apts.length ? apts[0][0] : P.mid0) - 1;
  const phi = Math.max(apts.length ? apts[apts.length - 1][0] : P.mid0 + 5,
                       bpts.length ? bpts[0][0] : P.mid0) + 1;
  const X = (p) => PAD.l + ((w - PAD.l - PAD.r) * (p - plo)) / (phi - plo || 1);
  const Y = (v) => PAD.t + (h - PAD.t - PAD.b) * (1 - v / ymax);

  frame(ctx, w, h,
    linTicks(0, ymax, 4).map((v) => [v, Y(v)]),
    linTicks(plo, phi, 5).map((v) => [v, X(v)]),
    (v) => (v >= 1000 ? (v / 1000).toFixed(1) + "k" : Math.round(v)),
    (v) => (v * tick).toFixed(2));

  const stepArea = (pts, ext, colFill, colLine) => {
    if (!pts.length) return;
    ctx.beginPath();
    ctx.moveTo(X(pts[0][0]), Y(0));
    let prevY = Y(pts[0][1]);
    ctx.lineTo(X(pts[0][0]), prevY);
    for (let k = 1; k < pts.length; k++) {
      ctx.lineTo(X(pts[k][0]), prevY);
      prevY = Y(pts[k][1]);
      ctx.lineTo(X(pts[k][0]), prevY);
    }
    const lastP = pts[pts.length - 1][0];
    ctx.lineTo(X(lastP + ext), prevY);
    ctx.lineTo(X(lastP + ext), Y(0));
    ctx.closePath();
    ctx.fillStyle = colFill; ctx.globalAlpha = 0.22; ctx.fill();
    ctx.globalAlpha = 1; ctx.strokeStyle = colLine; ctx.lineWidth = 1.6; ctx.stroke();
  };
  stepArea(bpts, -0.6, COLORS.bid, COLORS.bid);
  stepArea(apts, +0.6, COLORS.ask, COLORS.ask);

  ctx.save();
  ctx.strokeStyle = COLORS.mut; ctx.setLineDash([3, 4]);
  ctx.beginPath(); ctx.moveTo(X(s.mid[i]), PAD.t); ctx.lineTo(X(s.mid[i]), h - PAD.b); ctx.stroke();
  ctx.restore();

  els.depthT.textContent = `t = ${s.t[i].toFixed(1)} s`;
  const bb = s.best_bid[i], ba = s.best_ask[i];
  els.depthMeta.textContent = bb != null && ba != null
    ? `bb ${(bb * tick).toFixed(2)}×${bids[0][1]} · ba ${(ba * tick).toFixed(2)}×${asks[0][1]} · spread ${s.spread[i]}t`
    : "one-sided book";
}

/* chart hover + click-to-seek */
function bindTimeChart(cv, readEl) {
  const toT = (e) => {
    const r = cv.getBoundingClientRect();
    const x = e.clientX - r.left;
    return Math.min(horizonT(), Math.max(0, ((x - PAD.l) / (r.width - PAD.l - PAD.r)) * horizonT()));
  };
  cv.addEventListener("mousemove", (e) => {
    if (!state.result) return;
    const t = toT(e);
    const b = state.prep.baseline.run.series, r = state.prep.rl.run.series;
    const bi = nearestIdx(b.t, t), ri = nearestIdx(r.t, t);
    readEl.textContent =
      `t=${t.toFixed(1)}s · AC ${b.remaining[bi].toLocaleString()} · RL ${r.remaining[ri].toLocaleString()} left`;
  });
  cv.addEventListener("mouseleave", () => (readEl.textContent = ""));
  cv.addEventListener("click", (e) => {
    if (!state.result) return;
    stopPlayback();
    setCursor(toT(e) / state.result.decision_dt);
  });
}

/* --------------------------------------------------- tape / cards / table */
function renderTape() {
  const P = activePrep(), tick = state.result.tick, tNow = cursorT();
  const shown = P.run.fills.filter((f) => f.t <= tNow + 1e-9);
  els.tapeMeta.textContent = `${shown.length}/${P.run.fills.length} fills`;
  const last = shown.slice(-14).reverse();
  els.tape.innerHTML = last.length
    ? last.map((f) => {
        const cls = f.px >= P.run.arrival_ticks ? "up" : "dn";
        return `<li><span class="t">${f.t.toFixed(1)}s</span><span class="side">SELL</span>` +
               `<span class="px ${cls}">${(f.px * tick).toFixed(2)}</span>` +
               `<span class="qty">${f.qty.toLocaleString()}</span></li>`;
      }).join("")
    : `<li class="empty-tape">No fills yet at this point of the timeline.</li>`;
}

function renderCards() {
  const b = state.result.runs.baseline.raw, r = state.result.runs.rl.raw;
  els.cArrival.textContent = fmt$(b.arrival_price);
  els.cArrivalS.textContent = `seed ${state.result.params.seed} · T ${state.result.params.horizon}s`;
  els.cAC.textContent = `${fmt(b.shortfall_bps)} bps`;
  els.cACS.textContent = `vwap ${fmt$(b.vwap)} · ${b.children} children`;
  els.cRLk.textContent = `${state.result.runs.rl.label.includes("PPO") ? "RL (PPO)" : "RL (heuristic)"} · shortfall`;
  els.cRL.textContent = `${fmt(r.shortfall_bps)} bps`;
  const d = b.shortfall_bps - r.shortfall_bps;
  els.cDelta.textContent = `${d >= 0 ? "−" : "+"}${fmt(Math.abs(d))} bps vs AC`;
  els.cDelta.className = "delta " + (d >= 0 ? "good" : "bad");
  els.cFill.textContent =
    `${Math.round((100 * b.filled) / b.target)}% · ${Math.round((100 * r.filled) / r.target)}%`;
  els.cFillS.textContent = `AC ${fmt(b.duration)}s · RL ${fmt(r.duration)}s`;
}

const TABLE_KEYS = ["Strategy", "Filled / Target", "Arrival Price", "Exec VWAP",
  "Impl. Shortfall (bps)", "Total Cost ($)", "Child Orders", "Duration (s)"];

function renderTable() {
  els.cmp.tHead.innerHTML =
    "<tr>" + TABLE_KEYS.map((k) => `<th>${k}</th>`).join("") + "</tr>";
  const row = (rep, cls) =>
    `<tr class="${cls}">` + TABLE_KEYS.map((k) => `<td>${rep[k]}</td>`).join("") + "</tr>";
  els.cmp.tBodies[0].innerHTML =
    row(state.result.runs.baseline.report, "base") + row(state.result.runs.rl.report, "rl");
}

/* ---------------------------------------------------------------- playback */
const cursorT = () => Math.min(state.cursor, activePrep().N - 1) * state.result.decision_dt;

function setCursor(v) {
  const last = maxSteps() - 1;
  state.cursor = Math.min(last, Math.max(0, v));
  els.scrub.value = String(state.cursor);
  els.clock.textContent = `t = ${cursorT().toFixed(1)} s`;
  syncCursor3D();
  drawDepth2D(); renderTape();
  drawTrajectory(); drawMid();
}

function stopPlayback() {
  state.playing = false;
  els.play.textContent = "▶";
}

els.play.addEventListener("click", () => {
  if (!state.result) return;
  if (!state.playing && state.cursor >= maxSteps() - 1) state.cursor = 0;
  state.playing = !state.playing;
  els.play.textContent = state.playing ? "❚❚" : "▶";
});
els.scrub.addEventListener("input", () => { stopPlayback(); setCursor(+els.scrub.value); });

function setView(which) {
  state.active = which;
  els.viewBase.classList.toggle("on", which === "baseline");
  els.viewRL.classList.toggle("on", which === "rl");
  if (state.result) { rebuildScene(); setCursor(state.cursor); }
}
els.viewBase.addEventListener("click", () => setView("baseline"));
els.viewRL.addEventListener("click", () => setView("rl"));

document.addEventListener("keydown", (e) => {
  if (!state.result || e.target.matches("input,select,textarea")) return;
  if (e.code === "Space") { e.preventDefault(); els.play.click(); }
  else if (e.key === "ArrowRight") { stopPlayback(); setCursor(state.cursor + (e.shiftKey ? 10 : 1)); }
  else if (e.key === "ArrowLeft") { stopPlayback(); setCursor(state.cursor - (e.shiftKey ? 10 : 1)); }
  else if (e.key.toLowerCase() === "b") setView(state.active === "baseline" ? "rl" : "baseline");
});

let lastTs = performance.now();
function loop(ts) {
  const delta = Math.min(0.1, (ts - lastTs) / 1000); lastTs = ts;
  if (state.result && state.playing) {
    const next = state.cursor + (delta * +els.speed.value) / state.result.decision_dt;
    if (next >= maxSteps() - 1) { setCursor(maxSteps() - 1); stopPlayback(); }
    else setCursor(next);
  }
  renderThree();
  requestAnimationFrame(loop);
}
requestAnimationFrame(loop);

/* ------------------------------------------------------------------ export */
els.exportJson.addEventListener("click", () => {
  if (!state.result) return;
  const blob = new Blob([JSON.stringify(state.result, null, 1)], { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `cleo-run-seed${state.result.params.seed}.json`;
  a.click(); URL.revokeObjectURL(a.href);
});
els.exportPng.addEventListener("click", () => {
  const t = state.three; if (!t) return;
  renderThree();
  const a = document.createElement("a");
  a.href = t.canvas.toDataURL("image/png");
  a.download = "cleo-liquidity-canyon.png";
  a.click();
});

/* -------------------------------------------------------------------- boot */
if ("scrollRestoration" in history) history.scrollRestoration = "manual";
window.addEventListener("load", () => { $("scrollArea").scrollTop = 0; });

function resizeAll() {
  if (!state.result) return;
  drawTrajectory(); drawMid(); drawDepth2D();
}
new ResizeObserver(resizeAll).observe($("scrollArea"));
bindTimeChart(els.trajChart, els.trajRead);
bindTimeChart(els.midChart, els.midRead);
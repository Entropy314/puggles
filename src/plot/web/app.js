/* puggles dashboard.
 *
 * Two data modes:
 *   server   — fetch /api/meta + /api/history (+ /api/compare), then stream new generations over
 *              /api/events (SSE).
 *   embedded — window.PUGGLES_DATA holds everything: {kind: "run", meta, frames, comparisons}
 *              replays a run, {kind: "plot", spec} renders one ggplot-style figure (see plot.js).
 *
 * Rendering is coalesced into one requestAnimationFrame per burst of updates, and every view is
 * drawn with Plotly.react + a stable uirevision, so live updates never reset zoom or 3-D camera.
 *
 * Color roles: the main run is series slot 1, each compared run the next fixed slot; selection,
 * pins and the knee are neutral ink shapes with text labels, so they never compete with runs.
 */
(function () {
  "use strict";

  const P = window.PugglesPlot;
  const DATA = window.PUGGLES_DATA || null;
  const $ = (id) => document.getElementById(id);
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  if (!window.Plotly) {
    document.body.innerHTML = '<p style="padding:24px;font:14px system-ui">Could not load Plotly.js from cdn.jsdelivr.net — the dashboard needs an internet connection to load its charting library.</p>';
    return;
  }

  restoreTheme();
  $("theme").addEventListener("click", toggleTheme);

  if (DATA && DATA.kind === "plot") {
    initFigure(DATA.spec);
    return;
  }

  /* -- state ----------------------------------------------------------------------------- */

  const MAX_PINS = 4;
  const S = {
    mode: DATA ? "embedded" : "server",
    meta: {},
    frames: [],
    idx: -1, // frame shown
    follow: true, // track the newest frame
    state: "connecting",
    paused: false,
    objView: null,
    axes: [0, 1, 2],
    colorBy: "set",
    showPop: true,
    showTrails: false,
    showRef: true,
    cmpAt: "same",
    varView: null,
    metric: null,
    sideTab: "front",
    selected: null, // {x, f, ok} — kept by value so it survives generation changes
    pins: [], // [{x, f, ok, label}]
    lasso: null, // Set of solution keys from a lasso / box selection
    brush: { obj: {}, var: {} }, // parallel-coordinates ranges: dimension index -> constraintrange
    sort: { col: 1, dir: 1 },
    playing: null,
    comparisons: [], // [{name, meta, frames, local, hidden}]
    goals: [], // aspiration level per objective (null = none)
    goalsOnly: false, // list only solutions meeting every goal
    steer: false,
    spread: 0.25,
    convMode: "each", // "each" run, or median + IQR "group"
    shapes: false, // tell runs apart by marker shape / line dash too
    notify: false,
    notified: {},
    notice: null, // a short message in the runs strip (e.g. a rejected comparison file)
    hash: null, // view state restored from the URL
  };
  const dirty = { obj: true, var: true, conv: true, side: true, header: true };
  let rafPending = false;

  /* -- helpers ---------------------------------------------------------------------------- */

  const cur = () => S.frames[S.idx] || null;
  const nObj = () => S.meta.n_obj || firstLen("F") || firstLen("PF") || 0;
  const nVar = () => S.meta.n_var || firstLen("X") || 0;
  function firstLen(key) {
    for (const f of S.frames) if (f[key] && f[key].length) return f[key][0].length;
    return 0;
  }
  const dir = (i) => (S.meta.direction && S.meta.direction[i]) || -1; // -1 minimize, 1 maximize
  const objName = (i) => (S.meta.objective_names && S.meta.objective_names[i]) || `f${i + 1}`;
  const objLabel = (i) => `${objName(i)} · ${dir(i) === 1 ? "max" : "min"}`;
  const varName = (j) => (S.meta.variable_names && S.meta.variable_names[j]) || `x${j + 1}`;
  const keyOf = (x) => x.join(",");
  const better = (i, a, b) => (dir(i) === 1 ? a > b : a < b);
  const runName = () => S.meta.run_label || S.meta.title || "this run";
  const reference = () => (S.meta.reference_front && S.meta.reference_front.length ? S.meta.reference_front : null);
  const hasConstraints = () => S.frames.some((f) => f.Pcv && f.Pcv.length);
  const visibleComparisons = () => S.comparisons.filter((c) => !c.hidden);
  const runColor = (t, i) => (i < 0 ? t.series[0] : t.series[(i + 1) % t.series.length]); // i = comparison index
  // Shapes mode: a second, color-independent identity channel per run (index 0 = main run).
  const SYMBOLS = ["circle", "square", "diamond", "triangle-up", "cross", "x", "star", "hexagon"];
  const DASHES = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot", "6px,2px,1px,2px", "2px,3px"];
  const runSymbol = (i, fallback) => (S.shapes ? SYMBOLS[(i + 1) % SYMBOLS.length] : fallback); // i = comparison index, -1 main
  const runDash = (i) => (S.shapes ? DASHES[(i + 1) % DASHES.length] : "solid");

  // Your own columns (frame.E), aligned with the archive.
  const extraNames = () => { for (const f of S.frames) if (f.E) return Object.keys(f.E); return []; };
  const extraOf = (f, name, k) => (f.E && f.E[name] ? f.E[name][k] : null);
  const extrasFor = (f, k) => Object.fromEntries(extraNames().map((n) => [n, extraOf(f, n, k)]));

  // Goals: aspiration levels per objective; the gap is how far a solution's worst objective
  // misses its goal, as a fraction of the front's range (≤ 0 = meets every goal).
  const goalsActive = () => S.goals.some((g) => g !== null && g !== undefined && Number.isFinite(g));
  const gapCache = new WeakMap();
  function goalGaps(f) {
    if (!f || !f.F || !goalsActive()) return null;
    const key = JSON.stringify(S.goals);
    const hit = gapCache.get(f);
    if (hit && hit.key === key) return hit.gaps;
    const m = Math.min(nObj(), f.F[0] ? f.F[0].length : 0);
    const lo = Array(m).fill(Infinity), hi = Array(m).fill(-Infinity);
    for (const r of f.F) for (let i = 0; i < m; i++) if (Number.isFinite(r[i])) { lo[i] = Math.min(lo[i], r[i]); hi[i] = Math.max(hi[i], r[i]); }
    const gaps = f.F.map((r) => {
      let worst = -Infinity;
      for (let i = 0; i < m; i++) {
        const z = S.goals[i];
        if (z === null || z === undefined || !Number.isFinite(z)) continue;
        const miss = dir(i) === 1 ? z - r[i] : r[i] - z;
        worst = Math.max(worst, miss / (hi[i] - lo[i] || Math.abs(z) || 1));
      }
      return worst;
    });
    gapCache.set(f, { key, gaps });
    return gaps;
  }
  const fmtGap = (g) => (g === null || g === undefined ? "–" : g <= 0 ? "meets" : `+${(g * 100).toPrecision(2)}%`);

  const keyCache = new WeakMap();
  function keysOf(frame) {
    let m = keyCache.get(frame);
    if (!m) {
      m = new Map();
      (frame.X || []).forEach((x, i) => m.set(keyOf(x), i));
      keyCache.set(frame, m);
    }
    return m;
  }
  function indexOf(frame, sol) {
    if (!sol || !frame) return -1;
    const i = keysOf(frame).get(keyOf(sol.x));
    return i === undefined ? -1 : i;
  }
  const selectedIndex = (frame) => indexOf(frame, S.selected);
  const pinOf = (frame, k) => S.pins.find((p) => indexOf(frame, p) === k);

  // Filters: lasso (2-D / matrix) and parallel-coordinates brushes, combined with AND.
  const anyBrush = (b) => Object.values(b).some(Boolean);
  const filtering = () => Boolean(S.lasso) || anyBrush(S.brush.obj) || anyBrush(S.brush.var) || (S.goalsOnly && goalsActive());
  function inRange(v, cr) {
    const ranges = Array.isArray(cr[0]) ? cr : [cr];
    return ranges.some(([a, b]) => v >= a && v <= b);
  }
  function passes(f, k) {
    if (S.lasso && !S.lasso.has(keyOf(f.X[k]))) return false;
    for (const [i, cr] of Object.entries(S.brush.obj)) if (cr && !inRange(f.F[k][i], cr)) return false;
    for (const [j, cr] of Object.entries(S.brush.var)) if (cr && !inRange(f.X[k][j], cr)) return false;
    if (S.goalsOnly && goalsActive()) { const g = goalGaps(f); if (g && g[k] > 0) return false; }
    return true;
  }
  /** Archive indices currently listed: all of them, or those passing every filter. */
  function listed(frame) {
    const n = (frame && frame.F) ? frame.F.length : 0;
    const all = [...Array(n).keys()];
    return filtering() ? all.filter((k) => passes(frame, k)) : all;
  }

  function bounds(j, frame) {
    const t = S.meta.variable_types && S.meta.variable_types[j];
    if (t && Number.isFinite(t[1]) && Number.isFinite(t[2]) && t[2] > t[1]) return [t[1], t[2]];
    let lo = Infinity, hi = -Infinity;
    for (const x of (frame && frame.X) || []) { lo = Math.min(lo, x[j]); hi = Math.max(hi, x[j]); }
    if (!Number.isFinite(lo)) return [0, 1];
    return hi > lo ? [lo, hi] : [lo - 0.5, hi + 0.5];
  }

  /**
   * The most balanced trade-off: the feasible front solution whose worst objective, normalized to
   * the front's range, is smallest (ties broken by the sum). Unlike "smallest sum", this lands
   * mid-front on concave fronts too.
   */
  const kneeCache = new WeakMap();
  function kneeIndex(f) {
    if (!f || !f.F || f.F.length < 3) return -1;
    let k = kneeCache.get(f);
    if (k !== undefined) return k;
    const m = Math.min(nObj(), f.F[0].length);
    const pts = f.F.map((r) => r.slice(0, m).map((v, i) => (dir(i) === 1 ? -v : v)));
    const lo = Array(m).fill(Infinity), hi = Array(m).fill(-Infinity);
    for (const p of pts) for (let i = 0; i < m; i++) if (Number.isFinite(p[i])) { lo[i] = Math.min(lo[i], p[i]); hi[i] = Math.max(hi[i], p[i]); }
    let best = [Infinity, Infinity];
    k = -1;
    pts.forEach((p, idx) => {
      if (f.ok && f.ok[idx] === false) return;
      let worst = -Infinity, sum = 0;
      for (let i = 0; i < m; i++) { const v = (p[i] - lo[i]) / (hi[i] - lo[i] || 1); worst = Math.max(worst, v); sum += v; }
      if (Number.isFinite(worst) && (worst < best[0] - 1e-12 || (Math.abs(worst - best[0]) <= 1e-12 && sum < best[1]))) {
        best = [worst, sum]; k = idx;
      }
    });
    kneeCache.set(f, k);
    return k;
  }

  /** The last frame of `frames` at or before `nfe` evaluations (binary search). */
  function frameAt(frames, nfe) {
    let lo = 0, hi = frames.length - 1, ans = null;
    while (lo <= hi) {
      const mid = (lo + hi) >> 1;
      if (frames[mid].nfe <= nfe) { ans = frames[mid]; lo = mid + 1; } else hi = mid - 1;
    }
    return ans;
  }
  const comparisonFrame = (c) => (S.cmpAt === "final" || !cur() ? c.frames[c.frames.length - 1] : frameAt(c.frames, cur().nfe));

  function markDirty(...keys) {
    for (const k of keys.length ? keys : Object.keys(dirty)) dirty[k] = true;
    if (!rafPending) {
      rafPending = true;
      requestAnimationFrame(render);
    }
  }

  /* -- URL state ---------------------------------------------------------------------------- */

  function readHash() {
    const h = {};
    for (const part of location.hash.replace(/^#/, "").split("&")) {
      const [k, v] = part.split("=");
      if (k) h[decodeURIComponent(k)] = decodeURIComponent(v || "");
    }
    return h;
  }
  let hashTimer = null;
  function writeHash() {
    clearTimeout(hashTimer);
    hashTimer = setTimeout(() => {
      const f = cur();
      const h = { v: S.objView, ax: S.axes.join("."), c: S.colorBy, vv: S.varView, m: S.metric, tr: S.showTrails ? 1 : 0 };
      if (!S.follow && f) h.g = f.gen;
      const text = Object.entries(h).filter(([, v]) => v !== null && v !== undefined)
        .map(([k, v]) => `${k}=${encodeURIComponent(v)}`).join("&");
      try { history.replaceState(null, "", `#${text}`); } catch (_) { /* sandboxed frames */ }
    }, 300);
  }

  /* -- data feed -------------------------------------------------------------------------- */

  function start() {
    const hash = readHash();
    S.hash = hash; // setupControls applies (and clears) it
    setupControls();
    S.idx = S.frames.length - 1;
    if (hash.g !== undefined && S.frames.length) {
      const g = Number(hash.g);
      const i = S.frames.findIndex((f) => f.gen >= g);
      if (i >= 0) { S.idx = i; S.follow = false; }
    }
    for (const k in dirty) dirty[k] = true;
    render(); // first paint now, not a frame later
  }

  function boot() {
    if (S.mode === "embedded") {
      S.meta = DATA.meta || {};
      S.frames = DATA.frames || [];
      S.comparisons = (DATA.comparisons || []).map((c) => ({ name: c.name, meta: c.meta || {}, frames: c.frames || [], local: false, hidden: false }));
      S.goals = (S.meta.goals || []).slice();
      S.state = "replay";
      S.follow = false;
      start();
    } else {
      Promise.all([getJSON("api/meta"), getJSON("api/history")])
        .then(([meta, hist]) => {
          S.meta = meta || {};
          S.frames = hist.frames || [];
          S.state = hist.state;
          S.goals = (S.meta.goals || []).slice();
          start();
          connect();
        })
        .catch(() => setState("disconnected"));
    }
  }

  function getJSON(url) {
    return fetch(url, { cache: "no-store" }).then((r) => {
      if (!r.ok) throw new Error(r.statusText);
      return r.json();
    });
  }

  function post(url) {
    return fetch(url, { method: "POST" }).then((r) => r.json()).then((d) => {
      if (d.state) setState(d.state);
      if (d.paused !== undefined) { S.paused = d.paused; markDirty("header"); }
    }).catch(() => {});
  }

  function connect() {
    const es = new EventSource("api/events");
    es.addEventListener("frame", (e) => addFrame(JSON.parse(e.data)));
    es.addEventListener("meta", (e) => {
      const meta = JSON.parse(e.data);
      const reshape = meta.n_obj !== S.meta.n_obj || meta.n_var !== S.meta.n_var
        || Boolean(meta.reference_front) !== Boolean(S.meta.reference_front);
      S.meta = meta;
      const typing = document.activeElement && document.activeElement.dataset && document.activeElement.dataset.goal !== undefined;
      if (!typing && JSON.stringify(meta.goals || []) !== JSON.stringify(S.goals.map((g) => (Number.isFinite(g) ? g : null)))) {
        S.goals = (meta.goals || []).map((g) => (g === null ? null : g));
      }
      if (reshape && S.controlsReady) setupControls();
      else setupSideTabs();
      markDirty();
    });
    es.addEventListener("state", (e) => {
      const { state } = JSON.parse(e.data);
      const ended = (state === "done" || state === "stopped") && S.state !== state;
      setState(state);
      // The stream only carries the newest generation; pull the full recorded history at the end.
      if (ended) getJSON("api/history").then((h) => replaceFrames(h.frames || []));
    });
    es.addEventListener("paused", (e) => { S.paused = JSON.parse(e.data).paused; markDirty("header"); });
    es.addEventListener("compare", () => {
      getJSON("api/compare").then((runs) => {
        const local = S.comparisons.filter((c) => c.local);
        const hidden = new Set(S.comparisons.filter((c) => c.hidden).map((c) => c.name));
        S.comparisons = runs.map((c) => ({ name: c.name, meta: c.meta || {}, frames: c.frames || [], local: false, hidden: hidden.has(c.name) }))
          .concat(local);
        if (S.controlsReady) setupControls();
        markDirty();
      }).catch(() => {});
    });
    es.addEventListener("reset", () => {
      S.frames = []; S.idx = -1; S.selected = null; S.lasso = null; S.follow = true;
      S.brush = { obj: {}, var: {} };
      markDirty();
    });
    es.onerror = () => { if (es.readyState === EventSource.CLOSED || S.state !== "connecting") setState("disconnected"); };
    es.onopen = () => { if (S.state === "disconnected") getJSON("api/history").then((h) => { setState(h.state); replaceFrames(h.frames || []); }); };
  }

  function addFrame(frame) {
    const last = S.frames[S.frames.length - 1];
    if (last && frame.gen <= last.gen) {
      if (frame.gen === last.gen) return;
      S.frames = []; // generation counter went backwards: a new run
    }
    const firstFrame = S.frames.length === 0;
    S.frames.push(frame);
    if (S.follow) S.idx = S.frames.length - 1;
    if (firstFrame) setupControls();
    markDirty();
  }

  function replaceFrames(frames) {
    if (!frames.length) return;
    const gen = cur() ? cur().gen : null;
    S.frames = frames;
    S.idx = S.follow || gen === null ? frames.length - 1 : Math.max(0, frames.findIndex((f) => f.gen >= gen));
    markDirty();
  }

  function setState(state) {
    if (state !== S.state) say(`Run ${({ done: "finished", stopped: "stopped", paused: "paused", running: "running" })[state] || state}`);
    S.state = state;
    if (state === "done" || state === "stopped") setTimeout(maybeNotify, 0);
    if (state === "done" || state === "stopped" || state === "replay") S.playing && togglePlay(false);
    markDirty("header");
  }

  /* -- controls --------------------------------------------------------------------------- */

  const OBJ_VIEWS = {
    hist: "Distribution", scatter: "2-D", "3d": "3-D", parallel: "Parallel", matrix: "Matrix",
  };

  function objViewsFor(m) {
    if (m <= 1) return ["hist"];
    if (m === 2) return ["scatter", "parallel"];
    if (m === 3) return ["3d", "scatter", "parallel", "matrix"];
    return ["parallel", "matrix", "scatter", "3d"];
  }

  function metricsFor(m) {
    const out = [];
    if (reference()) out.push(["igd", "IGD to the true front"]);
    if (m === 2 || m === 3) out.push(["hv", "Hypervolume (normalized)"]);
    if (m >= 4) out.push(["hvmc", "Hypervolume (Monte Carlo estimate)"]);
    for (let i = 0; i < m; i++) out.push([`best:${i}`, `Best ${objName(i)}`]);
    out.push(["size", "Front size"]);
    if (hasConstraints() || S.frames.some((f) => (f.Pok || []).includes(false))) out.push(["feasible", "Feasible share of population"]);
    out.push(["rate", "Evaluations per second"]);
    if (S.frames.some((f) => f.timing)) out.push(["time", "Time per generation (breakdown)"]);
    return out;
  }

  function setupControls() {
    const m = nObj(), n = nVar();
    if (!m) return; // wait for data
    const h = S.hash || {};
    S.hash = null; // apply the URL state once

    const views = objViewsFor(m);
    if (h.v && views.includes(h.v)) S.objView = h.v;
    if (!views.includes(S.objView)) S.objView = views[0];
    seg($("obj-view"), views.map((v) => [v, OBJ_VIEWS[v]]), S.objView, (v) => {
      S.objView = v; purge($("obj-plot")); refreshObjControls(); markDirty("obj");
    });
    if (!S.controlsReady) {
      S.axes = [0, Math.min(1, m - 1), Math.min(2, m - 1)];
      if (h.ax) {
        const ax = h.ax.split(".").map(Number);
        if (ax.length === 3 && ax.every((a) => Number.isInteger(a) && a >= 0 && a < m)) S.axes = ax;
      }
      if (h.tr) S.showTrails = h.tr === "1";
    }

    const colorSel = $("color-by");
    const opts = [["set", "Front / population"]];
    if (hasConstraints()) opts.push(["cv", "Constraint violations (population)"]);
    if (goalsActive()) opts.push(["gap", "Goal gap"]);
    for (let i = 0; i < m; i++) opts.push([`o:${i}`, objName(i)]);
    for (const e of extraNames()) opts.push([`e:${e}`, e]);
    for (let j = 0; j < Math.min(n, 200); j++) opts.push([`v:${j}`, varName(j)]);
    colorSel.innerHTML = opts.map(([v, l]) => `<option value="${v}">${esc(l)}</option>`).join("");
    if (h.c && opts.some(([v]) => v === h.c)) S.colorBy = h.c;
    if (!opts.some(([v]) => v === S.colorBy)) S.colorBy = "set";
    colorSel.value = S.colorBy;
    colorSel.onchange = () => { S.colorBy = colorSel.value; markDirty("obj"); };
    $("show-pop").onchange = (e) => { S.showPop = e.target.checked; markDirty("obj"); };
    $("show-trails").checked = S.showTrails;
    $("show-trails").onchange = (e) => { S.showTrails = e.target.checked; markDirty("obj"); };
    $("show-ref").onchange = (e) => { S.showRef = e.target.checked; markDirty("obj"); };
    $("cmp-at").value = S.cmpAt;
    $("cmp-at").onchange = (e) => { S.cmpAt = e.target.value; markDirty("obj"); };

    const varViews = [["parallel", "Parallel"], ["heatmap", "Heatmap"], ["drivers", "Drivers"]];
    if (["parallel", "heatmap", "drivers"].includes(h.vv)) S.varView = h.vv;
    if (!S.varView) S.varView = n <= 24 ? "parallel" : "heatmap";
    seg($("var-view"), varViews, S.varView, (v) => { S.varView = v; purge($("var-plot")); markDirty("var"); });

    const metrics = metricsFor(m);
    if (h.m && metrics.some(([v]) => v === h.m)) S.metric = h.m;
    if (!metrics.some(([v]) => v === S.metric)) S.metric = metrics[0][0];
    const metricSel = $("metric");
    metricSel.innerHTML = metrics.map(([v, l]) => `<option value="${v}">${esc(l)}</option>`).join("");
    metricSel.value = S.metric;
    metricSel.onchange = () => { S.metric = metricSel.value; markDirty("conv"); };

    const convMode = $("conv-mode");
    convMode.classList.toggle("hidden", S.comparisons.length === 0);
    convMode.value = S.convMode;
    convMode.onchange = () => { S.convMode = convMode.value; markDirty("conv"); };

    S.controlsReady = true;
    setupSideTabs();
    refreshObjControls();
  }

  function refreshObjControls() {
    const m = nObj();
    const pickers = $("axis-pickers");
    const count = S.objView === "scatter" ? 2 : S.objView === "3d" ? 3 : 0;
    pickers.innerHTML = "";
    if (count && m > count) {
      ["x", "y", "z"].slice(0, count).forEach((name, k) => {
        const lab = document.createElement("label");
        lab.className = "inline";
        lab.textContent = name;
        const sel = document.createElement("select");
        sel.setAttribute("aria-label", `${name} axis`);
        for (let i = 0; i < m; i++) sel.add(new Option(objName(i), String(i)));
        sel.value = String(S.axes[k]);
        sel.onchange = () => { S.axes[k] = Number(sel.value); markDirty("obj"); };
        lab.appendChild(sel);
        pickers.appendChild(lab);
      });
    }
    const v = S.objView;
    $("pop-wrap").classList.toggle("hidden", v === "parallel" || v === "hist");
    $("color-wrap").classList.toggle("hidden", v === "hist" || v === "matrix");
    $("trails-wrap").classList.toggle("hidden", v !== "scatter" && v !== "3d");
    $("ref-wrap").classList.toggle("hidden", !reference() || v === "parallel" || v === "hist");
    $("cmp-at-wrap").classList.toggle("hidden", !S.comparisons.length || v === "parallel" || v === "hist");
  }

  function seg(el, items, value, onPick) {
    el.innerHTML = items.length < 2 ? "" : items.map(([v, l]) =>
      `<button type="button" data-v="${v}" aria-pressed="${v === value}">${esc(l)}</button>`).join("");
    el.onclick = (e) => {
      const b = e.target.closest("button");
      if (!b) return;
      for (const x of el.querySelectorAll("button")) x.setAttribute("aria-pressed", String(x === b));
      onPick(b.dataset.v);
    };
  }

  // Header actions
  $("stop").addEventListener("click", () => { $("stop").disabled = true; post("api/stop"); });
  $("pause").addEventListener("click", () => post(S.paused ? "api/resume" : "api/pause"));
  $("download").addEventListener("click", () => {
    const run = { format: "puggles-run", version: 1, meta: S.meta, frames: S.frames };
    const name = (S.meta.title && S.meta.title !== "puggles" ? S.meta.title : "run").replace(/[^\w.-]+/g, "_");
    download(`${name}.pgrun.json`, JSON.stringify(run), "application/json");
  });
  $("export-csv").addEventListener("click", exportCsv);
  $("knee-btn").addEventListener("click", () => {
    const f = cur(), k = kneeIndex(f);
    if (k >= 0) { S.selected = null; select(f, k); S.sideTab = "front"; setupSideTabs(); }
  });
  $("compare-btn").addEventListener("click", () => $("compare-file").click());
  $("compare-file").addEventListener("change", async (e) => {
    for (const file of e.target.files) {
      try {
        const run = await readRunFile(file);
        const name = (run.meta && run.meta.title) || file.name.replace(/\.gz$/, "").replace(/\.json$/, "").replace(/\.pgrun$/, "");
        const theirs = (run.meta && run.meta.n_obj) || ((run.frames || []).find((f) => f.F && f.F.length) || { F: [[]] }).F[0].length;
        if (nObj() && theirs && theirs !== nObj()) {
          showNotice(`Skipped ${file.name}: it has ${theirs} objectives, this run has ${nObj()}.`);
          continue;
        }
        S.comparisons.push({ name, meta: run.meta || {}, frames: run.frames || [], local: true, hidden: false });
      } catch (err) {
        showNotice(`Couldn't read ${file.name}: ${err.message}`);
      }
    }
    e.target.value = "";
    setupControls();
    markDirty();
  });

  function showNotice(text) {
    S.notice = text;
    markDirty("header");
    clearTimeout(showNotice.timer);
    showNotice.timer = setTimeout(() => { S.notice = null; markDirty("header"); }, 8000);
  }

  async function readRunFile(file) {
    let buf = new Uint8Array(await file.arrayBuffer());
    if (buf[0] === 0x1f && buf[1] === 0x8b) {
      const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"));
      buf = new Uint8Array(await new Response(stream).arrayBuffer());
    }
    const run = JSON.parse(new TextDecoder().decode(buf));
    if (run.format !== "puggles-run") throw new Error("not a puggles run file");
    return run;
  }

  // Scrubber
  const scrub = $("scrub");
  scrub.addEventListener("input", () => { S.follow = false; S.idx = Number(scrub.value); markDirty(); });
  $("live").addEventListener("click", () => { S.follow = true; S.idx = S.frames.length - 1; markDirty(); });
  $("play").addEventListener("click", () => togglePlay());
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !$("help-overlay").classList.contains("hidden")) { toggleHelp(false); return; }
    if (e.target.closest("input, select, textarea, button") && e.target !== scrub) return;
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const key = e.key.toLowerCase();
    if (key === "?" || (key === "/" && e.shiftKey)) { e.preventDefault(); toggleHelp(true); return; }
    if (key === "p" && S.selected) { togglePin(S.selected); say(`Pins: ${S.pins.map((x) => x.label).join(", ") || "none"}`); return; }
    if (key === "b") { $("knee-btn").click(); return; }
    if (key === "g") { S.sideTab = "goals"; setupSideTabs(); return; }
    if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      const order = S.tableOrder || [];
      if (!order.length) return;
      e.preventDefault();
      const si = selectedIndex(cur());
      const pos = order.indexOf(si);
      const next = order[Math.max(0, Math.min(order.length - 1, pos < 0 ? 0 : pos + (e.key === "ArrowDown" ? 1 : -1)))];
      S.selected = null;
      select(cur(), next);
      requestAnimationFrame(() => { const tr = document.querySelector(`#sol-table tr[data-k="${next}"]`); if (tr) tr.scrollIntoView({ block: "nearest" }); });
      return;
    }
    if (e.key === " ") { e.preventDefault(); togglePlay(); }
    else if (e.key === "ArrowLeft" || e.key === "ArrowRight") {
      if (e.target === scrub) return; // the range input handles its own arrows
      e.preventDefault();
      step(e.key === "ArrowLeft" ? -1 : 1);
    } else if (e.key === "Escape") { S.selected = null; clearFilters(); }
  });

  function step(d) {
    if (!S.frames.length) return;
    S.follow = false;
    S.idx = Math.max(0, Math.min(S.frames.length - 1, S.idx + d));
    markDirty();
  }

  function togglePlay(force) {
    const on = force === undefined ? !S.playing : force;
    if (S.playing) { clearInterval(S.playing); S.playing = null; }
    if (on && S.frames.length > 1) {
      S.follow = false;
      if (S.idx >= S.frames.length - 1) S.idx = 0;
      const interval = Math.max(33, Math.min(250, 6000 / S.frames.length)); // a replay takes ~6 s
      S.playing = setInterval(() => {
        if (S.idx >= S.frames.length - 1) { togglePlay(false); return; }
        S.idx++;
        markDirty();
      }, interval);
    }
    $("play-icon").setAttribute("d", S.playing ? "M3 1.5h3v11H3zM8 1.5h3v11H8z" : "M3 1.5v11l9-5.5z");
    $("play").setAttribute("aria-label", S.playing ? "Pause replay" : "Play");
  }

  function clearFilters() {
    S.lasso = null;
    S.brush = { obj: {}, var: {} };
    // Parallel coordinates hold their brushes in the plot; redraw them without.
    purge($("obj-plot")); purge($("var-plot"));
    markDirty();
  }

  /* -- render ----------------------------------------------------------------------------- */

  function render() {
    rafPending = false;
    if (!S.controlsReady && nObj()) setupControls();
    const t = P.tokens();
    const f = cur();
    try {
      if (dirty.header) renderHeader(f, t);
      if (dirty.obj) renderObjectives(t, f);
      if (dirty.var) renderVariables(t, f);
      if (dirty.conv) renderConvergence(t);
      if (dirty.side) renderSide(f, t);
    } finally {
      for (const k in dirty) dirty[k] = false;
    }
    if (S.controlsReady) writeHash();
  }

  function renderHeader(f, t) {
    document.title = `${S.meta.title && S.meta.title !== "puggles" ? S.meta.title + " · " : ""}puggles dashboard`;
    $("title").textContent = S.meta.title || "puggles";
    $("title").title = S.meta.title || "";
    $("algo").textContent = S.meta.algorithm || "";
    $("algo").classList.toggle("hidden", !S.meta.algorithm);
    renderExecution(f);
    renderRuns(t);
    const rate = evalRate(S.frames, S.idx);
    $("stat-rate").textContent = rate === null ? "–" : compact(rate);
    const pill = $("state");
    const labels = { connecting: "connecting", waiting: "waiting for run", running: "running", stopping: "stopping…",
      stopped: "stopped", done: "finished", replay: "replay", disconnected: "disconnected", paused: "paused" };
    // Pause takes effect at the end of the current generation; say so until it does.
    const pausing = S.paused && S.state === "running";
    pill.dataset.state = pausing ? "stopping" : S.state;
    pill.textContent = pausing ? "pausing…" : labels[S.state] || S.state;
    $("stat-gen").textContent = f ? P.fmt(f.gen) : "–";
    $("stat-nfe").textContent = f ? P.fmt(f.nfe) : "–";
    $("stat-front").textContent = f ? P.fmt((f.F || []).length) : "–";
    $("stat-time").textContent = f && f.t ? `${f.t < 10 ? f.t.toFixed(2) : f.t.toFixed(1)} s` : "–";
    const server = S.mode === "server";
    const active = ["running", "waiting", "paused"].includes(S.state);
    $("stop").classList.toggle("hidden", !server);
    $("stop").disabled = !(server && active);
    $("pause").classList.toggle("hidden", !(server && active));
    $("pause").textContent = S.paused ? "Resume" : "Pause";
    $("pause").title = S.paused ? "Continue the run" : "Hold the run after the current generation";
    $("download").disabled = !S.frames.length;
    $("report").disabled = !S.frames.length;
    $("export-csv").disabled = !(f && f.F && f.F.length);
    $("knee-btn").disabled = kneeIndex(f) < 0;

    scrub.max = String(Math.max(0, S.frames.length - 1));
    scrub.value = String(Math.max(0, S.idx));
    scrub.disabled = S.frames.length < 2;
    $("play").disabled = S.frames.length < 2;
    $("scrub-label").textContent = f
      ? `gen ${P.fmt(f.gen)} · ${P.fmt(f.nfe)} evals${S.frames.length > 1 ? ` · frame ${S.idx + 1}/${S.frames.length}` : ""}`
      : "–";
    const liveBtn = $("live");
    liveBtn.classList.toggle("hidden", !server);
    liveBtn.disabled = S.follow;
    liveBtn.textContent = S.follow ? "Following live" : "Jump to live";
  }

  /** The strip of runs being compared: color swatch, name, show/hide and remove. */
  function renderRuns(t) {
    const el = $("runs");
    const notice = S.notice ? `<span class="note">${esc(S.notice)}</span>` : "";
    if (!S.comparisons.length) { el.classList.toggle("hidden", !notice); el.innerHTML = notice; return; }
    el.classList.remove("hidden");
    const chips = [notice, `<span>Comparing</span><span class="run-chip main"><span class="swatch" style="background:${runColor(t, -1)}"></span>${esc(runName())}</span>`];
    S.comparisons.forEach((c, i) => {
      chips.push(`<span class="run-chip${c.hidden ? " off" : ""}" data-i="${i}">
        <span class="swatch" style="background:${runColor(t, i)}"></span>
        <span class="name" role="button" tabindex="0" title="Show or hide">${esc(c.name)}</span>
        ${c.local ? `<button type="button" data-remove="${i}" aria-label="Remove ${esc(c.name)}">×</button>` : ""}</span>`);
    });
    el.innerHTML = chips.join("");
    el.onclick = (e) => {
      const rm = e.target.closest("[data-remove]");
      if (rm) { S.comparisons.splice(Number(rm.dataset.remove), 1); setupControls(); markDirty(); return; }
      const chip = e.target.closest(".run-chip[data-i]");
      if (chip) { const c = S.comparisons[Number(chip.dataset.i)]; c.hidden = !c.hidden; markDirty(); }
    };
  }

  const ICON_GPU = '<svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><rect x="3" y="3" width="10" height="10" rx="1.5" fill="none" stroke="currentColor" stroke-width="1.4"/><rect x="6" y="6" width="4" height="4" fill="currentColor"/><path d="M5.5 1v2M8 1v2M10.5 1v2M5.5 13v2M8 13v2M10.5 13v2M1 5.5h2M1 8h2M1 10.5h2M13 5.5h2M13 8h2M13 10.5h2" stroke="currentColor" stroke-width="1.2"/></svg>';
  const ICON_CPU = '<svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><rect x="2.5" y="3.5" width="11" height="8" rx="1.2" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M5 14h6M8 11.5V14" stroke="currentColor" stroke-width="1.4"/></svg>';
  const ICON_WARN = '<svg width="14" height="14" viewBox="0 0 16 16" aria-hidden="true"><path d="M8 1.8 15 14H1z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><path d="M8 6v4" stroke="currentColor" stroke-width="1.5"/><circle cx="8" cy="12" r=".9" fill="currentColor"/></svg>';

  /** Header badge: where evaluations actually run — GPU + device, CPU threads, or a fallback. */
  function renderExecution(f) {
    const el = $("exec");
    const ex = S.meta.execution;
    const gpuEvals = f && f.gpu_evals != null ? f.gpu_evals : null;
    $("stat-gpu-wrap").classList.toggle("hidden", gpuEvals === null);
    if (gpuEvals !== null) $("stat-gpu").textContent = P.fmt(gpuEvals);
    if (!ex) { el.classList.add("hidden"); return; }
    const threads = ex.cpu_threads ? `${ex.cpu_threads} thread${ex.cpu_threads === 1 ? "" : "s"}` : "";
    const device = ex.device || "";
    const deviceName = device.replace(/\s*\([^)]*\)\s*$/, "");
    let kind, html, tip;
    if (ex.effective === "GPU") {
      kind = "gpu";
      html = `${ICON_GPU}<span class="kind">GPU</span><span>${esc(deviceName)}</span>`;
      tip = `Evaluating on the GPU: ${device}` + (gpuEvals !== null ? `\n${P.fmt(gpuEvals)} solutions evaluated on the GPU so far` : "");
    } else if (ex.requested !== ex.effective && ex.requested === "GPU") {
      kind = "fallback";
      html = `${ICON_WARN}<span class="kind">GPU not used</span><span>CPU · ${esc(threads)}</span>`;
      tip = ex.note || "GPU requested, running on CPU";
    } else {
      kind = "cpu";
      html = `${ICON_CPU}<span class="kind">CPU</span><span>${esc(threads)}</span>`;
      tip = `Evaluating on the CPU: ${ex.effective === "Sequential" ? "one thread" : threads}` + (ex.note ? `\n${ex.note}` : "");
    }
    el.dataset.kind = kind;
    el.innerHTML = html;
    el.title = tip;
    el.setAttribute("aria-label", tip);
    el.classList.remove("hidden");
  }

  /** Evaluations per second around frame i (over the preceding ~2 s of recorded frames). */
  function evalRate(fr, i) {
    if (i < 1 || !fr[i]) return null;
    let j = i - 1;
    while (j > 0 && fr[i].t - fr[j].t < 2) j--;
    const dt = fr[i].t - fr[j].t, dn = fr[i].nfe - fr[j].nfe;
    return dt > 0 && dn >= 0 ? dn / dt : null;
  }

  function compact(v) {
    if (v >= 1e6) return (v / 1e6).toFixed(v >= 1e7 ? 0 : 1) + "M";
    if (v >= 1e3) return (v / 1e3).toFixed(v >= 1e4 ? 0 : 1) + "k";
    return v.toFixed(0);
  }

  // Objective space ----------------------------------------------------------------------

  function colorValues(f, idx) {
    if (S.colorBy.startsWith("o:")) { const i = +S.colorBy.slice(2); return { title: objName(i), values: idx.map((k) => f.F[k][i]) }; }
    if (S.colorBy.startsWith("v:")) { const j = +S.colorBy.slice(2); return { title: varName(j), values: idx.map((k) => f.X[k][j]) }; }
    if (S.colorBy.startsWith("e:")) { const n = S.colorBy.slice(2); return { title: n, values: idx.map((k) => extraOf(f, n, k)) }; }
    if (S.colorBy === "gap") { const g = goalGaps(f); if (g) return { title: "goal gap", values: idx.map((k) => Math.max(0, g[k])) }; }
    return null;
  }

  function renderObjectives(t, f) {
    const el = $("obj-plot");
    const empty = !f || (!(f.F && f.F.length) && !(f.PF && f.PF.length));
    $("obj-empty").classList.toggle("hidden", !empty);
    if (empty) { purge(el); return; }
    const view = S.objView;
    if (view === "hist") return drawHistogram(t, f, el);
    if (view === "parallel") return drawObjParallel(t, f, el);
    if (view === "matrix") return drawMatrix(t, f, el);
    return drawScatter(t, f, el, view === "3d");
  }

  /** Earlier recorded fronts to fade in behind the current one (up to 6, evenly spaced). */
  function trailFrames() {
    if (S.idx < 1) return [];
    const stride = Math.max(1, Math.round(S.idx / 24));
    const out = [];
    for (let j = 1; j <= 6; j++) {
      const i = S.idx - j * stride;
      if (i < 0) break;
      out.push(S.frames[i]);
    }
    return out;
  }

  function drawScatter(t, f, el, three) {
    const m = nObj();
    const [ix, iy, iz] = S.axes.map((a) => Math.min(a, m - 1));
    const F = f.F || [];
    const all = [...F.keys()];
    const traces = [];
    const coords = (rows, idx) => {
      const o = { x: idx.map((k) => rows[k][ix]), y: idx.map((k) => rows[k][iy]) };
      if (three) o.z = idx.map((k) => rows[k][iz]);
      return o;
    };
    const pointsXY = (pts) => {
      const o = { x: pts.map((p) => p[ix]), y: pts.map((p) => p[iy]) };
      if (three) o.z = pts.map((p) => p[iz]);
      return o;
    };
    const type = three ? "scatter3d" : "scattergl";
    const hover = (extra) => [
      `${esc(objName(ix))}: %{x:.5g}`, `${esc(objName(iy))}: %{y:.5g}`, three ? `${esc(objName(iz))}: %{z:.5g}` : null, extra,
    ].filter(Boolean).join("<br>");
    const sortedLine = (pts) => [...pts].sort((a, b) => a[ix] - b[ix]);

    // The known true front, recessive.
    const ref = reference();
    if (ref && S.showRef) {
      const twoD = !three && m === 2;
      traces.push(Object.assign({
        type, mode: twoD ? "lines" : "markers", name: "True front", hoverinfo: "skip",
        line: { color: t.muted, width: 1.5, dash: "dot" },
        marker: { size: three ? 2 : 4, color: t.muted, opacity: 0.45 },
      }, pointsXY(twoD ? sortedLine(ref) : ref)));
    }

    if (S.showPop && f.PF && f.PF.length) {
      const PF = f.PF, ok = f.Pok || [];
      const byViolation = S.colorBy === "cv" && f.Pcv && f.Pcv.length;
      if (byViolation) {
        const idx = [...PF.keys()];
        // Feasible (0) stays recessive gray; 1..max violations climb the sequential ramp.
        const cmax = Math.max(1, ...f.Pcv);
        const ramp = P.colorscale("sequential", t).slice(2);
        const edge = 0.5 / cmax;
        const scale = [[0, t.muted], [edge, t.muted]].concat(ramp.map(([s, c]) => [edge + s * (1 - edge), c]));
        scale[2][0] += 1e-9;
        traces.push(Object.assign({
          type, mode: "markers", name: "Population", meta: "pop", customdata: idx.map((k) => f.Pcv[k]),
          hovertemplate: hover("%{customdata} constraint(s) violated") + "<extra>population</extra>",
          marker: {
            size: three ? 3 : 7, color: idx.map((k) => f.Pcv[k]), cmin: 0, cmax, colorscale: scale, showscale: true,
            colorbar: Object.assign(P.colorbar(t, "violated constraints"), { dtick: 1, tickformat: "d" }),
            line: { width: 0 },
          },
        }, coords(PF, idx)));
      } else {
        const feas = [...PF.keys()].filter((k) => ok[k] !== false);
        const infeas = [...PF.keys()].filter((k) => ok[k] === false);
        traces.push(Object.assign({
          type, mode: "markers", name: f.Pn ? `Population (sample of ${P.fmt(f.Pn)})` : "Population", meta: "pop", hovertemplate: hover() + "<extra>population</extra>",
          marker: { size: three ? 3 : 6, color: t.muted, opacity: 0.45, line: { width: 0 } },
        }, coords(PF, feas)));
        if (infeas.length) traces.push(Object.assign({
          type, mode: "markers", name: "Infeasible", meta: "pop", hovertemplate: hover() + "<extra>infeasible</extra>",
          marker: { size: three ? 3 : 6, color: t.muted, opacity: 0.6, symbol: "circle-open", line: { width: 1, color: t.muted } },
        }, coords(PF, infeas)));
      }
    }

    // Earlier fronts, fading with age.
    if (S.showTrails) {
      trailFrames().forEach((tf, j) => {
        const pts = tf.F || [];
        if (!pts.length) return;
        traces.push(Object.assign({
          type, mode: !three && m === 2 ? "lines+markers" : "markers", name: "Earlier fronts", legendgroup: "trails",
          showlegend: j === 0, hoverinfo: "skip", opacity: 0.32 - j * 0.045,
          marker: { size: three ? 3 : 5, color: t.series[0], line: { width: 0 } }, line: { color: t.series[0], width: 1 },
        }, pointsXY(!three && m === 2 ? sortedLine(pts) : pts)));
      });
    }

    // Compared runs at the same budget (or their final generation).
    visibleComparisons().forEach((c) => {
      const ci = S.comparisons.indexOf(c);
      const cf = comparisonFrame(c);
      if (!cf || !cf.F || !cf.F.length) return;
      const color = runColor(t, ci);
      if (!three && m === 2) {
        traces.push(Object.assign({ type, mode: "lines", hoverinfo: "skip", showlegend: false, legendgroup: c.name,
          line: { color, width: 1.5, dash: runDash(ci) }, opacity: 0.5 }, pointsXY(sortedLine(cf.F))));
      }
      traces.push(Object.assign({
        type, mode: "markers", name: c.name, legendgroup: c.name, meta: "cmp",
        hovertemplate: hover(`${esc(c.name)} · ${P.fmt(cf.nfe)} evals`) + "<extra></extra>",
        marker: { size: three ? 4 : 7, color, symbol: runSymbol(ci, three ? "circle" : "square"), line: three ? { width: 0 } : { color: t.surface, width: 1 } },
      }, pointsXY(cf.F)));
    });

    if (!three && m === 2 && ix !== iy && F.length > 1) {
      const order = [...all].sort((a, b) => F[a][ix] - F[b][ix]);
      traces.push(Object.assign({
        type, mode: "lines", name: "front", showlegend: false, hoverinfo: "skip",
        line: { color: t.series[0], width: 1.5 }, opacity: 0.35,
      }, coords(F, order)));
    }
    const cv = colorValues(f, all);
    const marker = {
      size: three ? 5 : 9, color: cv ? cv.values : t.series[0],
      line: three ? { width: 0 } : { color: t.surface, width: 1.5 },
    };
    if (cv) Object.assign(marker, { colorscale: P.colorscale("sequential", t), showscale: true, colorbar: P.colorbar(t, cv.title) });
    const archive = Object.assign({
      type, mode: "markers", name: S.comparisons.length ? runName() : "Pareto front", meta: "archive", customdata: all, marker,
      hovertemplate: hover(cv ? `${esc(cv.title)}: %{marker.color:.5g}` : null) + "<extra>#%{customdata}</extra>",
    }, coords(F, all));
    if (!three && filtering()) {
      archive.selectedpoints = all.filter((k) => passes(f, k));
      archive.unselected = { marker: { opacity: 0.2 } };
    }
    traces.push(archive);

    traces.push(...markerTraces(t, f, type, three, coords, pointsXY));

    const lay = P.layout(t, {
      uirevision: `scatter-${three}-${ix}-${iy}-${iz}`,
      margin: three ? { l: 0, r: 0, t: 0, b: 0 } : { l: 60, r: 16, t: 28, b: 48 },
      dragmode: three ? "orbit" : "zoom",
      selectionrevision: S.lasso ? "f" : "none",
    });
    if (three) {
      lay.scene = {
        xaxis: P.sceneAxis(t, objLabel(ix)), yaxis: P.sceneAxis(t, objLabel(iy)), zaxis: P.sceneAxis(t, objLabel(iz)),
        bgcolor: "rgba(0,0,0,0)", aspectmode: "cube",
      };
      lay.legend = Object.assign(lay.legend, { y: 1, yanchor: "top" });
    } else {
      lay.xaxis = P.axis(t, { title: { text: objLabel(ix) } });
      lay.yaxis = P.axis(t, { title: { text: objLabel(iy) } });
    }
    plot(el, traces, lay, bindScatterEvents);
  }

  /** Knee, pinned solutions and the selection: neutral ink shapes, labelled, on top. */
  function markerTraces(t, f, type, three, coords, pointsXY) {
    const out = [];
    const F = f.F || [];
    if (goalsActive() && F.length) {
      // The goal point; objectives without a goal sit at the front's best value.
      const m = nObj();
      const g = [...Array(m).keys()].map((i) => {
        const z = S.goals[i];
        if (z !== null && z !== undefined && Number.isFinite(z)) return z;
        let b = null;
        for (const r of F) if (b === null || better(i, r[i], b)) b = r[i];
        return b;
      });
      out.push(Object.assign({
        type, mode: "markers+text", name: "Goal", text: ["goal"], textposition: "middle right", textfont: { color: t.ink, size: 11 },
        hoverinfo: "skip", showlegend: false, marker: { size: three ? 6 : 13, symbol: three ? "x" : "x-thin", color: t.ink, line: { color: t.ink, width: 2 } },
      }, pointsXY([g])));
    }
    const k = kneeIndex(f);
    if (k >= 0) {
      out.push(Object.assign({
        type, mode: "markers+text", name: "Balanced", meta: "knee", text: ["balanced"], textposition: "top center",
        textfont: { color: t.ink, size: 11 }, hoverinfo: "skip", showlegend: false,
        marker: { size: three ? 7 : 15, symbol: three ? "diamond-open" : "star", color: three ? t.ink : "rgba(0,0,0,0)", line: { color: t.ink, width: 1.5 } },
      }, coords(F, [k])));
    }
    if (S.pins.length) {
      out.push(Object.assign({
        type, mode: "markers+text", name: "Pinned", meta: "pins", text: S.pins.map((p) => p.label), textposition: "bottom center",
        textfont: { color: t.ink, size: 11 }, hovertemplate: "%{text}<extra></extra>", showlegend: false,
        marker: { size: three ? 6 : 12, symbol: three ? "diamond" : "diamond-open", color: t.ink, line: { color: t.ink, width: 2 } },
      }, pointsXY(S.pins.map((p) => p.f))));
    }
    const si = selectedIndex(f);
    if (si >= 0) {
      out.push(Object.assign({
        type, mode: "markers", name: "Selected", meta: "sel", hoverinfo: "skip", showlegend: false,
        marker: { size: three ? 9 : 18, color: "rgba(0,0,0,0)", line: { color: t.ink, width: three ? 4 : 2 }, symbol: three ? "circle-open" : "circle" },
      }, coords(F, [si])));
    }
    return out;
  }

  function drawHistogram(t, f, el) {
    const vals = (f.PF || []).filter((_, k) => !f.Pok || f.Pok[k] !== false).map((r) => r[0]);
    const F = f.F || [];
    let best = null;
    for (const r of F) if (best === null || better(0, r[0], best)) best = r[0];
    const lay = P.layout(t, {
      uirevision: "hist", bargap: 0.04, showlegend: false, margin: { l: 56, r: 16, t: 28, b: 48 },
      xaxis: P.axis(t, { title: { text: objLabel(0) } }), yaxis: P.axis(t, { title: { text: "solutions" } }),
      shapes: best === null ? [] : [{ type: "line", xref: "x", yref: "paper", x0: best, x1: best, y0: 0, y1: 1, line: { color: t.ink, width: 2 } }],
      annotations: best === null ? [] : [{ x: best, xref: "x", y: 1, yref: "paper", yanchor: "bottom", showarrow: false,
        text: `best ${P.fmt(best)}`, font: { color: t.ink, size: 12 } }],
    });
    plot(el, [{
      type: "histogram", x: vals, name: "Population", marker: { color: t.series[0], line: { color: t.surface, width: 1 } },
      hovertemplate: `${esc(objName(0))}: %{x}<br>%{y} solutions<extra></extra>`,
    }], lay);
  }

  /** Parallel coordinates whose brushes filter the table (and survive live redraws). */
  function drawParallel(t, f, el, dims, brushKey, uirevision, margin) {
    const brush = S.brush[brushKey];
    dims.forEach((d, i) => { if (brush[i]) d.constraintrange = brush[i]; });
    const idx = el._pugglesOrder;
    // Parcoords draws a halo behind tick labels in the paper color; match it to the card.
    plot(el, [P.parcoords(t, dims, highlightLine(t, f, idx))], P.layout(t, { uirevision, margin, paper_bgcolor: t.surface }), (gd) => {
      gd.on("plotly_restyle", () => {
        const next = {};
        ((gd.data[0] && gd.data[0].dimensions) || []).forEach((d, i) => {
          if (d.constraintrange && d.constraintrange.length) next[i] = d.constraintrange;
        });
        S.brush[brushKey] = next;
        markDirty("side", "header");
      });
    });
  }

  function drawObjParallel(t, f, el) {
    const m = nObj();
    const idx = orderedForHighlight(f);
    el._pugglesOrder = idx;
    const F = f.F || [];
    const dims = [];
    for (let i = 0; i < m; i++) dims.push({ label: objLabel(i), values: idx.map((k) => F[k][i]) });
    drawParallel(t, f, el, dims, "obj", "obj-parallel", { l: 56, r: 48, t: 56, b: 24 });
  }

  function drawMatrix(t, f, el) {
    const m = nObj();
    const F = f.F || [];
    const all = [...F.keys()];
    const traces = [];
    const dimsOf = (rows) => [...Array(m).keys()].map((i) => ({ label: objName(i), values: rows.map((r) => r[i]) }));
    const ref = reference();
    if (ref && S.showRef) {
      traces.push(P.splom(t, dimsOf(ref), { name: "True front", hoverinfo: "skip", marker: { size: 2.5, color: t.muted, opacity: 0.4, line: { width: 0 } } }));
    }
    if (S.showPop && f.PF && f.PF.length) {
      traces.push(P.splom(t, dimsOf(f.PF), {
        name: "Population", meta: "pop", hoverinfo: "skip", marker: { size: 4, color: t.muted, opacity: 0.4, line: { width: 0 } },
      }));
    }
    visibleComparisons().forEach((c) => {
      const cf = comparisonFrame(c);
      if (!cf || !cf.F || !cf.F.length) return;
      traces.push(P.splom(t, dimsOf(cf.F), {
        name: c.name, meta: "cmp", hoverinfo: "skip",
        marker: { size: 5, symbol: runSymbol(S.comparisons.indexOf(c), "square"), color: runColor(t, S.comparisons.indexOf(c)), line: { width: 0 } },
      }));
    });
    const si = selectedIndex(f), knee = kneeIndex(f);
    traces.push(P.splom(t, dimsOf(F), {
      name: S.comparisons.length ? runName() : "Pareto front", meta: "archive", customdata: all,
      hovertemplate: "#%{customdata}<extra></extra>",
      selectedpoints: filtering() ? all.filter((k) => passes(f, k)) : undefined,
      unselected: { marker: { opacity: 0.2 } },
      marker: {
        size: all.map((k) => (k === si ? 12 : k === knee || pinOf(f, k) ? 10 : 6)),
        symbol: all.map((k) => (k === knee ? "star" : pinOf(f, k) ? "diamond" : "circle")),
        color: all.map((k) => (k === si || pinOf(f, k) || k === knee ? t.ink : t.series[0])),
        line: { color: t.surface, width: 0.5 },
      },
    }));
    const lay = P.layout(t, { uirevision: "matrix", margin: { l: 56, r: 16, t: 28, b: 44 }, dragmode: "select" });
    P.styleSplomAxes(t, lay, m);
    plot(el, traces, lay, bindScatterEvents);
  }

  /** Normal lines first, then pins, then the selection, so highlights draw on top. */
  function orderedForHighlight(f) {
    const all = [...((f.F || []).keys())];
    const si = selectedIndex(f);
    const pinned = all.filter((k) => k !== si && pinOf(f, k));
    const rest = all.filter((k) => k !== si && !pinOf(f, k));
    return rest.concat(pinned, si >= 0 ? [si] : []);
  }

  function highlightLine(t, f, idx) {
    const si = selectedIndex(f);
    const anyPin = idx.some((k) => pinOf(f, k));
    if (si >= 0 || anyPin) {
      // 0 = other, 1 = pinned, 2 = selected; a stepped scale keeps the three distinct.
      return {
        color: idx.map((k) => (k === si ? 2 : pinOf(f, k) ? 1 : 0)), cmin: 0, cmax: 2, showscale: false,
        colorscale: [[0, t.series[0]], [0.33, t.series[0]], [0.34, t.ink2], [0.66, t.ink2], [0.67, t.ink], [1, t.ink]],
      };
    }
    const cv = colorValues(f, idx);
    if (cv) return { color: cv.values, colorscale: P.colorscale("sequential", t), showscale: true, colorbar: P.colorbar(t, cv.title) };
    return { color: t.series[0] };
  }

  // Decision variables -------------------------------------------------------------------

  function renderVariables(t, f) {
    const el = $("var-plot");
    const n = nVar();
    const empty = !f || !(f.X && f.X.length) || !n;
    $("var-empty").classList.toggle("hidden", !empty);
    if (empty) { purge(el); return; }
    $("var-empty").textContent = "No solutions on the front yet.";
    if (S.varView === "heatmap") return drawHeatmap(t, f, el, n);
    if (S.varView === "drivers") return drawDrivers(t, f, el, n);
    const cap = Math.min(n, 60);
    const idx = orderedForHighlight(f);
    el._pugglesOrder = idx;
    const dims = [];
    for (let j = 0; j < cap; j++) {
      const [lo, hi] = bounds(j, f);
      const type = S.meta.variable_types && S.meta.variable_types[j] && S.meta.variable_types[j][0];
      const d = { label: varName(j), range: [lo, hi], values: idx.map((k) => f.X[k][j]) };
      if (type === "binary") { d.tickvals = [0, 1]; d.range = [0, 1]; }
      dims.push(d);
    }
    drawParallel(t, f, el, dims, "var", "var-parallel", { l: 40, r: 40, t: n > 8 ? 64 : 48, b: 16 });
  }

  function drawHeatmap(t, f, el, n) {
    const X = f.X, F = f.F;
    // Rows ordered best-first on the first objective, so structure along the front shows up.
    const rows = [...X.keys()].sort((a, b) => (dir(0) === 1 ? F[b][0] - F[a][0] : F[a][0] - F[b][0]));
    const b = [...Array(n).keys()].map((j) => bounds(j, f));
    const z = rows.map((k) => X[k].map((v, j) => (v - b[j][0]) / (b[j][1] - b[j][0])));
    const raw = rows.map((k) => X[k]);
    const si = selectedIndex(f);
    const rowBox = (r, color, width) => ({ type: "rect", xref: "paper", x0: 0, x1: 1, yref: "y", y0: r - 0.5, y1: r + 0.5, line: { color, width } });
    const shapes = [];
    rows.forEach((k, r) => { if (pinOf(f, k)) shapes.push(rowBox(r, t.ink2, 1.5)); });
    const sr = rows.indexOf(si);
    if (sr >= 0) shapes.push(rowBox(sr, t.ink, 2));
    const lay = P.layout(t, {
      uirevision: "var-heatmap", margin: { l: 48, r: 16, t: 8, b: 56 },
      xaxis: P.axis(t, { showline: false, showgrid: false, tickangle: n > 12 ? -45 : 0, tickformat: "" }),
      yaxis: P.axis(t, { showline: false, showgrid: false, autorange: "reversed", title: { text: `rank by ${objName(0)}` } }),
      shapes,
    });
    el._pugglesRows = rows;
    plot(el, [{
      type: "heatmap", z, customdata: raw.map((r, ri) => r.map((v) => [v, rows[ri]])),
      x: [...Array(n).keys()].map(varName), y: rows.map((_, r) => r),
      zmin: 0, zmax: 1, colorscale: P.colorscale("sequential", t), xgap: n < 40 ? 1 : 0, ygap: rows.length < 60 ? 1 : 0,
      colorbar: Object.assign(P.colorbar(t, "within bounds"), { tickvals: [0, 1], ticktext: ["lower", "upper"] }),
      hovertemplate: "%{x}: %{customdata[0]:.5g}<br>solution #%{customdata[1]}<extra></extra>",
    }], lay, (gd) => {
      gd.on("plotly_click", (ev) => {
        const pt = ev.points && ev.points[0];
        if (pt && gd._pugglesRows) select(cur(), gd._pugglesRows[pt.y]);
      });
    });
  }

  // Convergence --------------------------------------------------------------------------

  /** Every run on screen: the main run, then visible comparisons, as {name, frames, color}. */
  function runsShown(t) {
    const runs = [{ name: runName(), frames: S.frames, color: runColor(t, -1), main: true, ci: -1, meta: S.meta }];
    visibleComparisons().forEach((c) => {
      const ci = S.comparisons.indexOf(c);
      runs.push({ name: c.name, frames: c.frames, color: runColor(t, ci), ci, meta: c.meta || {} });
    });
    return runs;
  }

  /** Runs summarized together: an explicit group, else the algorithm, else the run itself. */
  const groupOf = (run) => (run.meta && (run.meta.group || run.meta.algorithm)) || run.name;

  /** Whether a higher value of the current metric is better. */
  function higherBetter(metric) {
    if (metric.startsWith("best:")) return dir(+metric.slice(5)) === 1;
    return ["hv", "hvmc", "feasible", "rate", "size"].includes(metric);
  }

  /** Value of a stepwise series at x: the last point at or before it. */
  function valueAt(series, x) {
    let v = null;
    for (let i = 0; i < series.x.length && series.x[i] <= x; i++) if (series.y[i] !== null && Number.isFinite(series.y[i])) v = series.y[i];
    return v;
  }

  function quantile(sorted, q) {
    if (!sorted.length) return null;
    const pos = (sorted.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }

  /** Where the main run stopped improving: the first point within 1% of its total improvement. */
  function plateauStart(series, up) {
    const pts = series.x.map((x, i) => [x, series.y[i]]).filter(([, y]) => y !== null && Number.isFinite(y));
    if (pts.length < 8) return null;
    const first = pts[0][1], last = pts[pts.length - 1][1];
    const total = up ? last - first : first - last;
    if (!(total > 0)) return null;
    const i = pts.findIndex(([, y]) => (up ? last - y : y - last) <= 0.01 * total);
    if (i < 0 || i >= pts.length - 2) return null;
    return { x: pts[i][0], end: pts[pts.length - 1][0] };
  }

  /** Points of a frame's front in minimization space, finite only. */
  function minSpace(f, m) {
    return (f.F || []).map((r) => r.slice(0, m).map((v, k) => (dir(k) === 1 ? -v : v)))
      .filter((p) => p.length === m && p.every((v) => v !== null && Number.isFinite(v)));
  }

  /**
   * One reference box (minimization space) shared by every run and frame, so hypervolumes are
   * comparable. Its corners come from the true front when it's known, otherwise from each run's
   * latest front — not from every point ever seen, which would let poor early generations
   * stretch the box until every later front scores ~1. The reference point sits 10% beyond the
   * nadir, and the box is snapped outward to a coarse grid so it rarely moves during a live run.
   */
  function hvBox(runs, m) {
    const flip = (r) => r.slice(0, m).map((v, k) => (dir(k) === 1 ? -v : v));
    const ref0 = reference();
    let pts = ref0 ? ref0.map(flip) : [];
    if (!pts.length) {
      for (const r of runs) {
        const last = r.frames[r.frames.length - 1];
        if (last) pts = pts.concat(minSpace(last, m));
      }
    }
    const lo = Array(m).fill(Infinity), hi = Array(m).fill(-Infinity);
    for (const p of pts) for (let k = 0; k < m; k++) if (Number.isFinite(p[k])) { lo[k] = Math.min(lo[k], p[k]); hi[k] = Math.max(hi[k], p[k]); }
    if (!Number.isFinite(lo[0])) return null;
    const base = [], ref = [];
    for (let k = 0; k < m; k++) {
      const range = hi[k] - lo[k] || Math.abs(hi[k]) || 1;
      const step = Math.pow(10, Math.floor(Math.log10(range))) / 4;
      base.push(Math.floor(lo[k] / step) * step);
      ref.push(Math.ceil((hi[k] + 0.1 * range) / step) * step);
    }
    let volume = 1;
    for (let k = 0; k < m; k++) volume *= ref[k] - base[k];
    return { base, ref, volume, key: base.concat(ref).join(",") };
  }

  function hv2(pts, rx, ry) {
    const sorted = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    let hv = 0, prevY = ry;
    for (const p of sorted) if (p[1] < prevY) { hv += (rx - p[0]) * (prevY - p[1]); prevY = p[1]; }
    return hv;
  }

  /** Exact 3-D hypervolume by slicing along the third objective: O(n² log n), fine for a front. */
  function hv3(pts, ref) {
    const sorted = [...pts].sort((a, b) => a[2] - b[2]);
    let hv = 0;
    for (let i = 0; i < sorted.length; i++) {
      const zNext = i + 1 < sorted.length ? sorted[i + 1][2] : ref[2];
      if (zNext > sorted[i][2]) hv += hv2(sorted.slice(0, i + 1), ref[0], ref[1]) * (zNext - sorted[i][2]);
    }
    return hv;
  }

  /** Fixed pseudo-random sample points in the box, so Monte Carlo estimates are stable across frames. */
  const MC_SAMPLES = 4096;
  let mcCache = { key: null, pts: null };
  function mcSamples(box, m) {
    if (mcCache.key === box.key) return mcCache.pts;
    let s = 0x9e3779b9;
    const rand = () => { s = (s + 0x6d2b79f5) | 0; let x = Math.imul(s ^ (s >>> 15), 1 | s); x ^= x + Math.imul(x ^ (x >>> 7), 61 | x); return ((x ^ (x >>> 14)) >>> 0) / 4294967296; };
    const pts = [];
    for (let i = 0; i < MC_SAMPLES; i++) pts.push(Array.from({ length: m }, (_, k) => box.base[k] + rand() * (box.ref[k] - box.base[k])));
    mcCache = { key: box.key, pts };
    return pts;
  }

  function hvMonteCarlo(pts, box, m) {
    const samples = mcSamples(box, m);
    const front = [...pts].sort((a, b) => a[0] - b[0]);
    let hit = 0;
    for (const s of samples) {
      for (const p of front) {
        if (p[0] > s[0]) break; // sorted: no later point dominates s
        let dom = true;
        for (let k = 1; k < m; k++) if (p[k] > s[k]) { dom = false; break; }
        if (dom) { hit++; break; }
      }
    }
    return hit / samples.length;
  }

  /** IGD: mean distance from each true-front point to the nearest found point, per-objective scaled. */
  function igd(f, ref, m) {
    const F = (f.F || []).filter((r) => r.slice(0, m).every((v) => Number.isFinite(v)));
    if (!F.length) return null;
    const lo = Array(m).fill(Infinity), hi = Array(m).fill(-Infinity);
    for (const r of ref) for (let k = 0; k < m; k++) { lo[k] = Math.min(lo[k], r[k]); hi[k] = Math.max(hi[k], r[k]); }
    const scale = lo.map((l, k) => hi[k] - l || 1);
    let total = 0;
    for (const r of ref) {
      let best = Infinity;
      for (const p of F) {
        let d = 0;
        for (let k = 0; k < m; k++) { const x = (p[k] - r[k]) / scale[k]; d += x * x; if (d >= best) break; }
        if (d < best) best = d;
      }
      total += Math.sqrt(best);
    }
    return total / ref.length;
  }

  // Per-frame metric values, cached; keyed so they invalidate when their inputs change.
  const metricCache = new Map(); // metric+key -> WeakMap(frame -> value)
  function cached(name, f, compute) {
    let wm = metricCache.get(name);
    if (!wm) { wm = new WeakMap(); metricCache.set(name, wm); }
    let v = wm.get(f);
    if (v === undefined) { v = compute(); wm.set(f, v); }
    return v;
  }

  /** Indices to evaluate an expensive metric at: ≤ 150 evenly spaced, plus the last and current. */
  function sampleIndices(n, current) {
    if (n <= 150) return [...Array(n).keys()];
    const out = new Set([n - 1]);
    for (let i = 0; i < 150; i++) out.add(Math.round((i * (n - 1)) / 149));
    if (current >= 0 && current < n) out.add(current);
    return [...out].sort((a, b) => a - b);
  }

  /** The metric for one run, as {x: nfe[], y: value[]}. */
  function metricSeries(run, ctx) {
    const fr = run.frames;
    const m = ctx.m;
    const metric = S.metric;
    const all = () => [...fr.keys()];
    const at = (idxs, fn) => ({ x: idxs.map((i) => fr[i].nfe), y: idxs.map((i) => fn(fr[i], i)) });
    if (metric === "size") return at(all(), (f) => (f.F || []).length);
    if (metric === "rate") return at(all(), (_, i) => evalRate(fr, i));
    if (metric === "feasible") return at(all(), (f) => (f.Pok && f.Pok.length ? f.Pok.filter((v) => v !== false).length / f.Pok.length : null));
    if (metric.startsWith("best:")) {
      const i = +metric.slice(5);
      return at(all(), (f) => {
        const rows = (f.F && f.F.length) ? f.F : (f.PF || []);
        let b = null;
        for (const r of rows) if (r[i] !== null && Number.isFinite(r[i]) && (b === null || better(i, r[i], b))) b = r[i];
        return b;
      });
    }
    const current = run.main ? S.idx : -1;
    // Points inside the reference point, clamped to the box's ideal corner so the fraction stays ≤ 1.
    const inBox = (f) => minSpace(f, m).filter((p) => p.every((x, k) => x < ctx.box.ref[k]))
      .map((p) => p.map((x, k) => Math.max(x, ctx.box.base[k])));
    if (metric === "hv" && ctx.box) {
      return at(all(), (f) => cached(`hv:${ctx.box.key}`, f, () => {
        const pts = inBox(f);
        return (m === 2 ? hv2(pts, ctx.box.ref[0], ctx.box.ref[1]) : hv3(pts, ctx.box.ref)) / ctx.box.volume;
      }));
    }
    if (metric === "hvmc" && ctx.box) {
      return at(sampleIndices(fr.length, current), (f) => cached(`hvmc:${ctx.box.key}`, f, () => hvMonteCarlo(inBox(f), ctx.box, m)));
    }
    if (metric === "igd" && reference()) {
      return at(sampleIndices(fr.length, current), (f) => cached("igd", f, () => igd(f, reference(), m)));
    }
    return { x: [], y: [] };
  }

  const PHASES = [
    ["evaluation", "Evaluation"], ["selection", "Sorting & selection"], ["variation", "Variation"],
    ["archive", "Archive"], ["record", "Dashboard"], ["other", "Other (callbacks, pacing)"],
  ];

  /** Milliseconds per generation in each phase; "other" is wall time the phases don't explain. */
  function timingRows(fr) {
    return fr.map((f, i) => {
      if (!f.timing) return null;
      const row = {};
      let sum = 0;
      for (const [k] of PHASES.slice(0, 5)) { row[k] = (f.timing[k] || 0) * 1000; sum += row[k]; }
      row.other = null;
      if (i > 0) {
        const dg = f.gen - fr[i - 1].gen, dt = f.t - fr[i - 1].t;
        if (dg > 0 && dt >= 0) row.other = Math.max(0, (dt / dg) * 1000 - sum);
      }
      return row;
    });
  }

  function renderConvergence(t) {
    const el = $("conv-plot");
    const fr = S.frames;
    const empty = fr.length < 2;
    $("conv-empty").classList.toggle("hidden", !empty);
    if (empty) { purge(el); return; }
    const label = $("metric").selectedOptions[0] ? $("metric").selectedOptions[0].textContent : "";
    const xs = fr.map((f) => f.nfe);
    const traces = [];
    const shapes = [], annotations = [];
    let ylabel = label, showlegend = false, hovermode = "closest";

    if (S.metric === "time") {
      const rows = timingRows(fr);
      const colors = [t.series[0], t.series[1], t.series[2], t.series[3], t.muted, t.ink2 + "55"];
      PHASES.forEach(([k, name], pi) => {
        traces.push({
          type: "scatter", mode: "lines", stackgroup: "time", name, x: xs, y: rows.map((r) => (r ? r[k] : null)),
          line: { width: 0.5, color: colors[pi] }, fillcolor: colors[pi],
          hovertemplate: `${name}: %{y:.3g} ms<extra></extra>`,
        });
      });
      ylabel = "ms per generation";
      showlegend = true;
      hovermode = "x unified";
    } else {
      const m = nObj();
      const runs = runsShown(t);
      const ctx = { m, box: (S.metric === "hv" || S.metric === "hvmc") ? hvBox(runs, m) : null };
      const series = runs.map((run) => metricSeries(run, ctx));
      if (S.convMode === "group" && runs.length > 1) {
        // Median and interquartile band per group, on a shared evaluation grid.
        const groups = new Map();
        runs.forEach((run, i) => {
          const key = groupOf(run);
          if (!groups.has(key)) groups.set(key, { key, members: [], color: t.series[groups.size % t.series.length], gi: groups.size });
          groups.get(key).members.push(series[i]);
        });
        for (const g of groups.values()) {
          const maxX = Math.max(...g.members.map((sr) => (sr.x.length ? sr.x[sr.x.length - 1] : 0)));
          const grid = [...Array(80).keys()].map((i) => (maxX * (i + 1)) / 80);
          const med = [], q1 = [], q3 = [];
          for (const x of grid) {
            const vals = g.members.map((sr) => valueAt(sr, x)).filter((v) => v !== null).sort((a, b) => a - b);
            med.push(quantile(vals, 0.5)); q1.push(quantile(vals, 0.25)); q3.push(quantile(vals, 0.75));
          }
          const name = `${g.key} (${g.members.length} run${g.members.length === 1 ? "" : "s"})`;
          traces.push({ type: "scatter", mode: "lines", x: grid, y: q3, line: { width: 0, color: g.color }, hoverinfo: "skip", showlegend: false, legendgroup: g.key });
          traces.push({ type: "scatter", mode: "lines", x: grid, y: q1, line: { width: 0, color: g.color }, fill: "tonexty", fillcolor: g.color + "33", hoverinfo: "skip", showlegend: false, legendgroup: g.key });
          traces.push({
            type: "scatter", mode: "lines", x: grid, y: med, name, legendgroup: g.key,
            line: { color: g.color, width: 2, dash: S.shapes ? DASHES[g.gi % DASHES.length] : "solid" },
            hovertemplate: `${esc(name)}<br>%{x:,.0f} evals<br>median ${esc(label)}: %{y:.5g}<extra></extra>`,
          });
        }
        showlegend = true;
      } else {
        runs.forEach((run, i) => {
          const sr = series[i];
          traces.push({
            type: "scatter", mode: "lines", x: sr.x, y: sr.y, name: run.name, connectgaps: true,
            line: { color: run.color, width: run.main ? 2 : 1.5, dash: runDash(run.ci), shape: S.metric === "size" ? "hv" : "linear" },
            hovertemplate: `${esc(run.name)}<br>%{x:,} evals<br>${esc(label)}: %{y:.5g}<extra></extra>`,
          });
          if (run.main) {
            const f = cur();
            const j = f ? sr.x.lastIndexOf(f.nfe) : -1;
            if (j >= 0) {
              traces.push({
                type: "scatter", mode: "markers", x: [sr.x[j]], y: [sr.y[j]], hoverinfo: "skip", showlegend: false,
                marker: { size: 10, color: t.ink, line: { color: t.surface, width: 2 } },
              });
            }
          }
        });
        showlegend = runs.length > 1;
        // Run events (e.g. steering) as labelled markers: they explain jumps in the curve.
        (S.meta.events || []).forEach((ev, i) => {
          shapes.push({ type: "line", xref: "x", yref: "paper", x0: ev.nfe, x1: ev.nfe, y0: 0, y1: 1, line: { color: t.ink2, width: 1 } });
          annotations.push({
            x: ev.nfe, xref: "x", y: 1 - 0.09 * (i % 3), yref: "paper", xanchor: "left", yanchor: "top", showarrow: false,
            text: esc(ev.label), font: { color: t.ink, size: 11 }, bgcolor: t.surface,
          });
        });
        // Plateau: from here on, the run gained less than 1% of its total improvement.
        if (!["rate", "size", "feasible"].includes(S.metric)) {
          const lastEvent = Math.max(-1, ...(S.meta.events || []).map((ev) => ev.nfe));
          const after = { x: [], y: [] };
          series[0].x.forEach((x, i) => { if (x > lastEvent) { after.x.push(x); after.y.push(series[0].y[i]); } });
          const p = plateauStart(after, higherBetter(S.metric));
          if (p) {
            const share = 1 - p.x / p.end;
            shapes.push({ type: "line", xref: "x", yref: "paper", x0: p.x, x1: p.x, y0: 0, y1: 1, line: { color: t.muted, width: 1, dash: "dot" } });
            annotations.push({
              x: p.x, xref: "x", y: 0.04, yref: "paper", xanchor: "left", yanchor: "bottom", showarrow: false,
              text: share >= 0.2 ? `plateau: last ${Math.round(share * 100)}% of evals added <1%` : "plateau", font: { color: t.ink2, size: 11 },
              bgcolor: t.surface,
            });
            S.plateau = { x: p.x, share };
          } else {
            S.plateau = null;
          }
        }
      }
    }
    const legendRows = S.metric === "time" ? 2 : 1;
    plot(el, traces, P.layout(t, {
      uirevision: `conv-${S.metric}-${S.convMode}`, showlegend, hovermode, shapes, annotations,
      legend: { orientation: "h", x: 0, y: 1, yanchor: "bottom", font: { color: t.ink2, size: 11 }, bgcolor: "rgba(0,0,0,0)", traceorder: "normal" },
      margin: { l: 56, r: 16, t: showlegend ? 12 + 18 * legendRows : 8, b: 44 },
      xaxis: P.axis(t, { title: { text: "evaluations" }, rangemode: "tozero", tickformat: "~s" }),
      yaxis: P.axis(t, Object.assign({ title: { text: ylabel }, rangemode: S.metric === "time" ? "tozero" : "normal" },
        S.metric === "feasible" ? { range: [0, 1.04], tickformat: ".0%" } : {})),
    }), (gd) => {
      gd.on("plotly_click", (ev) => {
        const pt = ev.points && ev.points[0];
        if (!pt || pt.curveNumber !== 0) return;
        const i = S.frames.findIndex((f) => f.nfe === pt.x);
        if (i >= 0) { S.follow = false; S.idx = i; togglePlay(false); markDirty(); }
      });
    });
    maybeNotify();
  }

  // Side panel ---------------------------------------------------------------------------

  function setupSideTabs() {
    const tabs = [["front", "Front"], ["pins", `Pinned${S.pins.length ? ` (${S.pins.length})` : ""}`], ["goals", goalsActive() ? "Goals ●" : "Goals"]];
    if (S.comparisons.length) tabs.push(["results", "Results"]);
    if (!tabs.some(([v]) => v === S.sideTab)) S.sideTab = "front";
    seg($("side-tabs"), tabs, S.sideTab, (v) => { S.sideTab = v; markDirty("side"); });
    markDirty("side");
  }

  function renderSide(f, t) {
    for (const tab of ["front", "pins", "goals", "results"]) $(`${tab}-pane`).classList.toggle("hidden", S.sideTab !== tab);
    if (S.sideTab === "pins") renderPins(t);
    else if (S.sideTab === "goals") renderGoals(f);
    else if (S.sideTab === "results") renderResults(t);
    else renderFront(f);
  }

  function renderFront(f) {
    const m = nObj();
    const F = (f && f.F) || [];
    const idx = listed(f);
    $("sol-count").textContent = F.length ? (filtering() ? `${idx.length} of ${F.length}` : `${F.length}`) : "";

    const note = $("filter-note");
    note.classList.toggle("hidden", !filtering());
    if (filtering()) {
      const parts = [];
      if (S.lasso) parts.push("lasso");
      const bo = Object.keys(S.brush.obj).filter((i) => S.brush.obj[i]).map((i) => objName(+i));
      const bv = Object.keys(S.brush.var).filter((j) => S.brush.var[j]).map((j) => varName(+j));
      if (bo.length) parts.push(`ranges on ${bo.join(", ")}`);
      if (bv.length) parts.push(`ranges on ${bv.join(", ")}`);
      note.innerHTML = `<span>${idx.length} solutions match: ${esc(parts.join(" + "))}</span><button type="button" id="clear-filter">Clear filters</button>`;
      $("clear-filter").onclick = clearFilters;
    }

    // Detail card
    const si = selectedIndex(f);
    const detail = $("detail");
    if (!S.selected) {
      detail.innerHTML = '<p class="hint">Click a point or row to inspect a solution; pin up to four to compare them. Lasso points (2-D) or drag along parallel-coordinate axes to filter. ★ Balanced picks the most even trade-off.</p>';
    } else {
      const sol = si >= 0 ? { f: F[si], x: f.X[si], ok: f.ok ? f.ok[si] : true } : S.selected;
      const pin = S.pins.find((p) => keyOf(p.x) === keyOf(sol.x));
      const knee = si >= 0 && si === kneeIndex(f);
      const objs = sol.f.map((v, i) => `<span title="${esc(objLabel(i))}">${esc(objLabel(i))}</span><span>${P.fmt(v)}</span>`).join("");
      const vars = sol.x.map((v, j) => `<span title="${esc(varName(j))}">${esc(varName(j))}</span><span>${P.fmt(v)}</span>`).join("");
      detail.innerHTML = `
        <div class="detail-head"><strong>${si >= 0 ? `Solution #${si}` : "Selected solution"}${knee ? '<span class="badge knee">★ balanced</span>' : ""}${pin ? `<span class="badge">${pin.label}</span>` : ""}</strong>
          <button class="btn${pin ? " on" : ""}" type="button" id="pin-sol" title="${pin ? "Unpin" : "Pin to compare (up to four)"}" ${!pin && S.pins.length >= MAX_PINS ? "disabled" : ""}>${pin ? "Unpin" : "Pin"}</button>
          <button class="btn" type="button" id="copy-sol" title="Copy as JSON">Copy</button>
          <button class="btn" type="button" id="clear-sol" title="Clear selection (Esc)">Clear</button></div>
        ${si < 0 ? '<p class="note">Not on the front at this generation.</p>' : ""}
        ${sol.ok === false ? '<p class="note">Infeasible: violates a constraint.</p>' : ""}
        <h3>Objectives</h3><div class="kv">${objs}</div>
        ${extraNames().length && si >= 0 ? `<h3>Your columns</h3><div class="kv">${extraNames().map((n) => `<span>${esc(n)}</span><span>${P.fmt(extraOf(f, n, si))}</span>`).join("")}</div>` : ""}
        ${goalsActive() && si >= 0 ? `<h3>Goals</h3><div class="kv"><span>gap to goals</span><span>${fmtGap(goalGaps(f)[si])}</span></div>` : ""}
        <h3>Variables</h3><div class="kv">${vars}</div>`;
      $("clear-sol").onclick = () => { S.selected = null; markDirty("obj", "var", "side"); };
      $("pin-sol").onclick = () => togglePin(sol);
      $("copy-sol").onclick = () => {
        navigator.clipboard && navigator.clipboard.writeText(JSON.stringify(solutionJson(sol), null, 2)).then(() => {
          $("copy-sol").textContent = "Copied"; setTimeout(() => { const b = $("copy-sol"); if (b) b.textContent = "Copy"; }, 1200);
        });
      };
    }

    // Table
    const kn = kneeIndex(f);
    const table = $("sol-table");
    const cols = [{ label: "#", get: (k) => k }];
    for (let i = 0; i < m; i++) cols.push({ label: objLabel(i), get: (k) => F[k][i] });
    for (const e of extraNames()) cols.push({ label: e, get: (k) => extraOf(f, e, k) });
    const gaps = goalGaps(f);
    if (gaps) cols.push({ label: "goal gap", get: (k) => gaps[k], show: fmtGap });
    const sc = Math.min(S.sort.col, cols.length - 1);
    const rows = [...idx].sort((a, b) => {
      const va = cols[sc].get(a), vb = cols[sc].get(b);
      return (va < vb ? -1 : va > vb ? 1 : 0) * S.sort.dir;
    }).slice(0, 1000);
    S.tableOrder = rows;
    table.tHead.innerHTML = "<tr>" + cols.map((c, ci) =>
      `<th scope="col" data-col="${ci}" aria-sort="${ci === sc ? (S.sort.dir > 0 ? "ascending" : "descending") : "none"}">${esc(c.label)}</th>`).join("") + "</tr>";
    table.tBodies[0].innerHTML = rows.map((k) => {
      const pin = pinOf(f, k);
      const tags = (k === kn ? '<span class="badge knee" title="most balanced trade-off">★</span>' : "") + (pin ? `<span class="badge">${pin.label}</span>` : "");
      return `<tr data-k="${k}" tabindex="0" aria-selected="${k === si}" class="${f.ok && f.ok[k] === false ? "infeasible" : ""}">` +
        cols.map((c, ci) => `<td>${(c.show || P.fmt)(c.get(k))}${ci === 0 ? tags : ""}</td>`).join("") + "</tr>";
    }).join("");
    table.tHead.onclick = (e) => {
      const th = e.target.closest("th");
      if (!th) return;
      const c = Number(th.dataset.col);
      S.sort = { col: c, dir: S.sort.col === c ? -S.sort.dir : 1 };
      markDirty("side");
    };
    table.tBodies[0].onclick = (e) => {
      const tr = e.target.closest("tr");
      if (tr) select(cur(), Number(tr.dataset.k));
    };
    table.tBodies[0].onkeydown = (e) => {
      if (e.key === "Enter") { const tr = e.target.closest("tr"); if (tr) select(cur(), Number(tr.dataset.k)); }
    };
  }

  /** Side-by-side comparison of pinned solutions; best value per objective in bold. */
  function renderPins() {
    const pane = $("pins-pane");
    if (!S.pins.length) {
      pane.innerHTML = '<div class="pins-wrap"><p class="hint">Select a solution and press <b>Pin</b> to compare up to four side by side. Pins stay put while the run moves on.</p></div>';
      return;
    }
    const m = Math.min(nObj(), S.pins[0].f.length), n = S.pins[0].x.length;
    const head = `<tr><th scope="col"></th>${S.pins.map((p) => `<th scope="col">${p.label}</th>`).join("")}</tr>`;
    const rows = [`<tr class="section"><td colspan="${S.pins.length + 1}">Objectives</td></tr>`];
    for (let i = 0; i < m; i++) {
      let best = null;
      for (const p of S.pins) if (best === null || better(i, p.f[i], best)) best = p.f[i];
      rows.push(`<tr><td>${esc(objLabel(i))}</td>${S.pins.map((p) => `<td class="${p.f[i] === best && S.pins.length > 1 ? "best" : ""}">${P.fmt(p.f[i])}</td>`).join("")}</tr>`);
    }
    const extras = extraNames();
    if (extras.length) {
      rows.push(`<tr class="section"><td colspan="${S.pins.length + 1}">Your columns</td></tr>`);
      for (const e of extras) rows.push(`<tr><td>${esc(e)}</td>${S.pins.map((p) => `<td>${P.fmt(p.e ? p.e[e] : null)}</td>`).join("")}</tr>`);
    }
    rows.push(`<tr class="section"><td colspan="${S.pins.length + 1}">Variables <span class="muted">(shaded where they differ from ${S.pins[0].label})</span></td></tr>`);
    for (let j = 0; j < n; j++) {
      const base = S.pins[0].x[j];
      rows.push(`<tr><td>${esc(varName(j))}</td>${S.pins.map((p, pi) => `<td class="${pi > 0 && p.x[j] !== base ? "diff" : ""}">${P.fmt(p.x[j])}</td>`).join("")}</tr>`);
    }
    pane.innerHTML = `
      <div class="pins-actions">
        <button class="btn" type="button" id="pins-csv">Export CSV</button>
        <button class="btn" type="button" id="pins-json">Export JSON</button>
        ${S.pins.map((p, i) => `<button class="btn" type="button" data-unpin="${i}" title="Unpin ${p.label}">× ${p.label}</button>`).join("")}
        <button class="btn" type="button" id="pins-clear">Clear all</button>
      </div>
      <div class="pins-wrap"><table class="compare"><thead>${head}</thead><tbody>${rows.join("")}</tbody></table></div>`;
    $("pins-csv").onclick = () => exportRows(S.pins, "pinned-solutions.csv");
    $("pins-json").onclick = () => download("pinned-solutions.json", JSON.stringify(S.pins.map((p) => ({ label: p.label, ...solutionJson(p) })), null, 2), "application/json");
    $("pins-clear").onclick = () => { S.pins = []; afterPinChange(); };
    pane.querySelectorAll("[data-unpin]").forEach((b) => { b.onclick = () => { S.pins.splice(Number(b.dataset.unpin), 1); afterPinChange(); }; });
  }

  function togglePin(sol) {
    const key = keyOf(sol.x);
    const i = S.pins.findIndex((p) => keyOf(p.x) === key);
    if (i >= 0) S.pins.splice(i, 1);
    else if (S.pins.length < MAX_PINS) {
      const f = cur(), k = indexOf(f, sol);
      S.pins.push({ x: sol.x, f: sol.f, ok: sol.ok, e: k >= 0 ? extrasFor(f, k) : {}, label: "" });
    }
    afterPinChange();
  }

  function afterPinChange() {
    S.pins.forEach((p, i) => { p.label = `P${i + 1}`; });
    setupSideTabs();
    markDirty("obj", "var", "side");
  }

  const solutionJson = (sol) => Object.assign({
    objectives: Object.fromEntries(sol.f.map((v, i) => [objName(i), v])),
  }, sol.e && Object.keys(sol.e).length ? { columns: sol.e } : {}, {
    variables: Object.fromEntries(sol.x.map((v, j) => [varName(j), v])),
  });

  // Goals ------------------------------------------------------------------------------------

  let goalsTimer = null;
  function renderGoals(f) {
    const pane = $("goals-pane");
    const m = nObj();
    const server = S.mode === "server";
    const active = ["running", "waiting", "paused"].includes(S.state);
    const canSteer = server && active && S.meta.steerable;
    const shape = `${m}|${canSteer}|${server}|${S.meta.algorithm}`;
    if (pane.dataset.shape !== shape) {
      pane.dataset.shape = shape;
      const rows = [...Array(m).keys()].map((i) => `<label class="goal-row"><span title="${esc(objLabel(i))}">${esc(objLabel(i))}</span>
        <span class="op">${dir(i) === 1 ? "≥" : "≤"}</span><input type="number" step="any" data-goal="${i}" placeholder="any" aria-label="Goal for ${esc(objName(i))}"></label>`).join("");
      const steerUi = canSteer ? `
        <label class="row"><input type="checkbox" id="steer"> Steer the run toward the goals</label>
        <div class="row" id="spread-row">Focus <input type="range" id="spread" min="0.05" max="1" step="0.05" aria-label="Focus width"><span id="spread-val"></span></div>`
        : server && active && S.meta.algorithm === "NSGAII" ? '<p class="hint">Steering needs NSGA-III; with NSGA-II goals mark and rank solutions.</p>' : "";
      pane.innerHTML = `<div class="form">
        <p class="hint">Set a target for any objective. Solutions meeting every goal are marked; the rest are ranked by how far their worst objective misses, as a share of the front's range.</p>
        ${rows}
        <label class="row"><input type="checkbox" id="goals-only"> Only list solutions meeting the goals</label>
        ${steerUi}
        <div class="status" id="goals-status"></div>
        <div class="row"><button class="btn" type="button" id="goals-closest">Select the closest</button><button class="btn" type="button" id="goals-clear">Clear goals</button></div>
      </div>`;
      pane.querySelectorAll("[data-goal]").forEach((inp) => {
        inp.addEventListener("input", () => {
          const v = inp.value.trim() === "" ? null : Number(inp.value);
          S.goals[Number(inp.dataset.goal)] = Number.isFinite(v) ? v : null;
          goalsChanged();
        });
      });
      $("goals-only").onchange = (e) => { S.goalsOnly = e.target.checked; markDirty("obj", "side", "header"); };
      $("goals-clear").onclick = () => { S.goals = []; S.goalsOnly = false; S.steer = false; pane.dataset.shape = ""; goalsChanged(); };
      $("goals-closest").onclick = () => {
        const g = goalGaps(cur());
        if (!g) return;
        let best = -1;
        g.forEach((v, k) => { if (best < 0 || v < g[best]) best = k; });
        if (best >= 0) { S.selected = null; select(cur(), best); }
      };
      if (canSteer) {
        $("steer").onchange = (e) => { S.steer = e.target.checked; goalsChanged(); };
        $("spread").oninput = (e) => { S.spread = Number(e.target.value); $("spread-val").textContent = `${Math.round(S.spread * 100)}% of the front`; goalsChanged(); };
      }
    }
    // Values (never while the person is typing in a field) and status.
    pane.querySelectorAll("[data-goal]").forEach((inp) => {
      if (document.activeElement === inp) return;
      const v = S.goals[Number(inp.dataset.goal)];
      inp.value = v === null || v === undefined ? "" : String(v);
    });
    $("goals-only").checked = S.goalsOnly;
    if ($("steer")) {
      $("steer").checked = S.steer;
      $("spread").value = String(S.spread);
      $("spread-val").textContent = `${Math.round(S.spread * 100)}% of the front`;
      $("spread-row").classList.toggle("hidden", !S.steer);
    }
    const g = goalGaps(f);
    const status = $("goals-status");
    if (!g) status.textContent = "No goals set.";
    else {
      const meet = g.filter((v) => v <= 0).length;
      status.innerHTML = `<b>${meet}</b> of ${g.length} front solutions meet every goal.` + (S.steer ? " Steering: the run concentrates near the goals." : "");
    }
    $("goals-closest").disabled = !g;
  }

  function goalsChanged() {
    gapCacheBust();
    setupSideTabs();
    markDirty("obj", "side", "header");
    clearTimeout(goalsTimer);
    goalsTimer = setTimeout(() => {
      setupControls(); // the color-by list gains/loses "Goal gap"
      if (S.mode === "server") {
        fetch("api/goals", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ goals: [...Array(nObj()).keys()].map((i) => (Number.isFinite(S.goals[i]) ? S.goals[i] : null)), steer: S.steer, spread: S.spread }),
        }).catch(() => {});
      }
    }, 350);
  }
  function gapCacheBust() { /* gap cache entries are keyed by the goals themselves */ }

  // Results across runs ------------------------------------------------------------------------

  function erf(x) {
    // Abramowitz–Stegun 7.1.26, |error| < 1.5e-7.
    const s = Math.sign(x); x = Math.abs(x);
    const t = 1 / (1 + 0.3275911 * x);
    const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
    return s * y;
  }

  /** Two-sided Mann–Whitney U test: exact without ties (small samples), else normal approximation. */
  function mannWhitneyP(a, b) {
    const n1 = a.length, n2 = b.length;
    if (!n1 || !n2) return null;
    const all = a.map((v) => [v, 0]).concat(b.map((v) => [v, 1])).sort((x, y) => x[0] - y[0]);
    const ranks = new Array(all.length);
    let tieTerm = 0;
    for (let i = 0; i < all.length;) {
      let j = i;
      while (j + 1 < all.length && all[j + 1][0] === all[i][0]) j++;
      for (let k = i; k <= j; k++) ranks[k] = (i + j) / 2 + 1;
      const c = j - i + 1;
      if (c > 1) tieTerm += c * c * c - c;
      i = j + 1;
    }
    let r1 = 0;
    all.forEach((x, i) => { if (x[1] === 0) r1 += ranks[i]; });
    const u = r1 - (n1 * (n1 + 1)) / 2;
    if (tieTerm === 0 && n1 * n2 <= 400) {
      // Exact null distribution of U by recurrence.
      const table = [];
      for (let i = 0; i <= n1; i++) {
        table.push([]);
        for (let j = 0; j <= n2; j++) {
          if (i === 0 || j === 0) { table[i].push([1]); continue; }
          const row = new Array(i * j + 1).fill(0);
          for (let k = 0; k <= i * j; k++) {
            const a1 = k - j >= 0 && k - j < table[i - 1][j].length ? table[i - 1][j][k - j] : 0;
            const a2 = k < table[i][j - 1].length ? table[i][j - 1][k] : 0;
            row[k] = a1 + a2;
          }
          table[i].push(row);
        }
      }
      const dist = table[n1][n2];
      const total = dist.reduce((x, y) => x + y, 0);
      let lo = 0, hi = 0;
      dist.forEach((c, k) => { if (k <= u) lo += c; if (k >= u) hi += c; });
      return Math.min(1, (2 * Math.min(lo, hi)) / total);
    }
    const N = n1 + n2;
    const sigma = Math.sqrt(((n1 * n2) / 12) * (N + 1 - tieTerm / (N * (N - 1))));
    if (!(sigma > 0)) return 1;
    const z = (Math.abs(u - (n1 * n2) / 2) - 0.5) / sigma;
    return Math.min(1, 2 * (1 - 0.5 * (1 + erf(Math.max(0, z) / Math.SQRT2))));
  }

  /** Each run's final value of the current metric, grouped and ranked. */
  function computeResults(t) {
    const metric = S.metric === "time" ? "rate" : S.metric;
    const label = metric === S.metric && $("metric").selectedOptions[0] ? $("metric").selectedOptions[0].textContent : "Evaluations per second";
    const m = nObj();
    const runs = runsShown(t);
    const saved = S.metric;
    S.metric = metric;
    const ctx = { m, box: (metric === "hv" || metric === "hvmc") ? hvBox(runs, m) : null };
    const finals = runs.map((run) => {
      const sr = metricSeries(run, ctx);
      let v = null;
      for (let i = sr.y.length - 1; i >= 0; i--) if (sr.y[i] !== null && Number.isFinite(sr.y[i])) { v = sr.y[i]; break; }
      const last = run.frames[run.frames.length - 1];
      return { run, value: v, nfe: last ? last.nfe : 0, group: groupOf(run) };
    });
    S.metric = saved;
    const up = higherBetter(metric);
    const groups = new Map();
    finals.forEach((r) => {
      if (!groups.has(r.group)) groups.set(r.group, { key: r.group, rows: [], color: t.series[groups.size % t.series.length] });
      groups.get(r.group).rows.push(r);
    });
    const summary = [...groups.values()].map((g) => {
      const vals = g.rows.map((r) => r.value).filter((v) => v !== null).sort((a, b) => a - b);
      const mean = vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
      const sd = vals.length > 1 ? Math.sqrt(vals.reduce((a, v) => a + (v - mean) ** 2, 0) / (vals.length - 1)) : null;
      return { ...g, vals, n: vals.length, median: quantile(vals, 0.5), q1: quantile(vals, 0.25), q3: quantile(vals, 0.75), mean, sd };
    }).sort((a, b) => (up ? b.median - a.median : a.median - b.median));
    const best = summary[0];
    summary.forEach((g, i) => { g.p = i === 0 ? null : mannWhitneyP(best.vals, g.vals); });
    return { label, up, summary, finals };
  }

  /** The smallest two-sided p a Mann–Whitney test can give for these sample sizes (no ties). */
  function minimumP(n1, n2) {
    let c = 1;
    for (let i = 1; i <= n1; i++) c = (c * (n2 + i)) / i; // C(n1 + n2, n1)
    return Math.min(1, 2 / c);
  }

  function renderResults(t) {
    const pane = $("results-pane");
    const r = computeResults(t);
    const best = r.summary[0];
    const floor = r.summary.slice(1).map((g) => minimumP(best.n, g.n)).reduce((a, b) => Math.max(a, b), 0);
    const verdict = (g, i) => {
      if (i === 0) return '<span class="verdict-best">best</span>';
      if (g.p === null) return "–";
      const p = g.p < 0.001 ? "p < 0.001" : `p = ${g.p.toFixed(3)}`;
      return g.p < 0.05 ? `<span class="verdict-worse">worse</span><br><span class="muted">${p}</span>` : `no clear difference<br><span class="muted">${p}</span>`;
    };
    const spread = (g) => g.n > 1 ? `<br><span class="muted">IQR ${P.fmt(g.q1)}–${P.fmt(g.q3)} · mean ${P.fmt(g.mean)} ± ${P.fmt(g.sd)}</span>` : "";
    const rows = r.summary.map((g, i) => `<tr>
      <td><span class="swatch" style="background:${g.color}"></span>${esc(g.key)}</td><td>${g.n}</td>
      <td>${P.fmt(g.median)}${spread(g)}</td><td>${verdict(g, i)}</td></tr>`).join("");
    const runRows = r.finals.map((f) => `<tr><td>${esc(f.run.name)}</td><td>${esc(f.group)}</td><td>${P.fmt(f.nfe)}</td><td>${P.fmt(f.value)}</td></tr>`).join("");
    const caution = r.summary.length > 1 && floor >= 0.05
      ? `<p>With these group sizes the smallest p-value the test can produce is ${floor.toFixed(2)}, so it can't confirm a difference at 0.05 however consistent the runs look. Run at least 4 seeds per group (5–10 for a solid answer).</p>`
      : "";
    pane.innerHTML = `<div class="results-wrap">
      <h3>${esc(r.label)} at each run's last generation</h3>
      <p>Groups are ${r.summary.length > 1 ? "ranked by median" : "summarized"} (${r.up ? "higher" : "lower"} is better) and compared with the best by a two-sided Mann–Whitney U test.
        Name groups with <code>set_group()</code>; otherwise runs group by algorithm.</p>
      <table class="results"><thead><tr><th>Group</th><th>Runs</th><th>Median</th><th>vs best</th></tr></thead><tbody>${rows}</tbody></table>
      ${caution}
      <h3>Runs</h3>
      <table class="results"><thead><tr><th>Run</th><th>Group</th><th>Evals</th><th>Final</th></tr></thead><tbody>${runRows}</tbody></table>
    </div>`;
  }

  // Drivers: which variables move with which objectives ----------------------------------------

  function ranksOf(v) {
    const idx = v.map((x, i) => [x, i]).sort((a, b) => a[0] - b[0]);
    const r = new Array(v.length);
    for (let i = 0; i < idx.length;) {
      let j = i;
      while (j + 1 < idx.length && idx[j + 1][0] === idx[i][0]) j++;
      for (let k = i; k <= j; k++) r[idx[k][1]] = (i + j) / 2;
      i = j + 1;
    }
    return r;
  }

  /** Spearman rank correlation; null when either side is constant. */
  function spearman(a, b) {
    const ra = ranksOf(a), rb = ranksOf(b), n = a.length;
    const ma = ra.reduce((x, y) => x + y, 0) / n, mb = rb.reduce((x, y) => x + y, 0) / n;
    let sab = 0, saa = 0, sbb = 0;
    for (let i = 0; i < n; i++) { const da = ra[i] - ma, db = rb[i] - mb; sab += da * db; saa += da * da; sbb += db * db; }
    return saa > 0 && sbb > 0 ? sab / Math.sqrt(saa * sbb) : null;
  }

  function drawDrivers(t, f, el, n) {
    const F = f.F, X = f.X;
    if (F.length < 4) { purge(el); $("var-empty").textContent = "Needs four or more front solutions."; $("var-empty").classList.remove("hidden"); return; }
    const m = nObj();
    const cols = [...Array(m).keys()].map((i) => ({ name: objName(i), values: F.map((r) => r[i]) }))
      .concat(extraNames().map((e) => ({ name: e, values: F.map((_, k) => extraOf(f, e, k)) })));
    let rows = [...Array(n).keys()].map((j) => {
      const xs = X.map((x) => x[j]);
      const rho = cols.map((c) => spearman(xs, c.values));
      return { j, rho, strength: Math.max(0, ...rho.map((r) => Math.abs(r || 0))) };
    });
    rows.sort((a, b) => b.strength - a.strength);
    const total = rows.length;
    rows = rows.slice(0, 30);
    const z = rows.map((r) => r.rho.map((v) => (v === null ? null : v)));
    const ys = rows.map((r) => varName(r.j));
    plot(el, [{
      type: "heatmap", z, x: cols.map((c) => c.name), y: ys, zmin: -1, zmax: 1, zmid: 0, xgap: 1, ygap: 1,
      colorscale: P.colorscale("diverging", t), customdata: rows.map((r) => cols.map(() => r.j)),
      texttemplate: rows.length * cols.length <= 240 ? "%{z:.2f}" : "", textfont: { size: 10 },
      colorbar: Object.assign(P.colorbar(t, "rank correlation"), { tickvals: [-1, 0, 1] }),
      hovertemplate: "%{y} vs %{x}: ρ = %{z:.2f}<br>click to color the front by %{y}<extra></extra>",
    }], P.layout(t, {
      uirevision: "drivers", margin: { l: 56, r: 16, t: 8, b: 48 },
      xaxis: P.axis(t, { showline: false, showgrid: false, tickformat: "", side: "bottom" }),
      yaxis: P.axis(t, { showline: false, showgrid: false, autorange: "reversed", tickformat: "",
        title: { text: total > rows.length ? `top ${rows.length} of ${total} variables` : "" } }),
    }), (gd) => {
      gd.on("plotly_click", (ev) => {
        const pt = ev.points && ev.points[0];
        if (!pt) return;
        const j = pt.customdata;
        S.colorBy = `v:${j}`;
        $("color-by").value = S.colorBy;
        if (S.objView === "matrix" || S.objView === "hist") { const views = objViewsFor(nObj()); if (views.includes("scatter")) { S.objView = "scatter"; purge($("obj-plot")); setupControls(); } }
        say(`Coloring the front by ${varName(j)}`);
        markDirty("obj");
      });
    });
  }

  // Notifications, report, keyboard, announcements ---------------------------------------------

  function say(text) { $("announce").textContent = text; }

  function notifyOnce(key, title, body) {
    if (!S.notify || S.notified[key] || !("Notification" in window) || Notification.permission !== "granted") return;
    S.notified[key] = true;
    try { new Notification(title, { body }); } catch (_) { /* some browsers only allow service-worker notifications */ }
  }

  function maybeNotify() {
    const f = cur();
    const what = `${runName()}${f ? ` · ${P.fmt(f.nfe)} evaluations` : ""}`;
    if (S.state === "done" || S.state === "stopped") notifyOnce(`end-${S.frames.length}`, "puggles: run finished", what);
    if (S.state === "running" && S.plateau && S.plateau.share >= 0.3) {
      notifyOnce("plateau", "puggles: run has plateaued", `${what} — the last ${Math.round(S.plateau.share * 100)}% of evaluations added under 1%.`);
    }
  }

  $("notify").addEventListener("click", async () => {
    if (!("Notification" in window)) { showNotice("This browser can't show notifications."); return; }
    if (!S.notify) {
      const perm = Notification.permission === "granted" ? "granted" : await Notification.requestPermission();
      S.notify = perm === "granted";
      if (!S.notify) showNotice("Notifications are blocked for this page in the browser settings.");
    } else S.notify = false;
    $("notify").setAttribute("aria-pressed", String(S.notify));
    say(S.notify ? "Notifications on" : "Notifications off");
  });

  function restoreShapes() {
    let on = window.matchMedia("(forced-colors: active)").matches;
    try { const v = localStorage.getItem("puggles-shapes"); if (v !== null) on = v === "1"; } catch (_) { /* default */ }
    S.shapes = on;
    $("shapes").setAttribute("aria-pressed", String(on));
  }
  $("shapes").addEventListener("click", () => {
    S.shapes = !S.shapes;
    $("shapes").setAttribute("aria-pressed", String(S.shapes));
    try { localStorage.setItem("puggles-shapes", S.shapes ? "1" : "0"); } catch (_) { /* ignore */ }
    say(S.shapes ? "Shapes mode on: runs differ by marker shape and line style" : "Shapes mode off");
    markDirty();
  });

  function toggleHelp(open) {
    const ov = $("help-overlay");
    const show = open === undefined ? ov.classList.contains("hidden") : open;
    ov.classList.toggle("hidden", !show);
    if (show) $("help-close").focus();
  }
  $("help").addEventListener("click", () => toggleHelp(true));
  $("help-close").addEventListener("click", () => toggleHelp(false));
  $("help-overlay").addEventListener("click", (e) => { if (e.target.id === "help-overlay") toggleHelp(false); });

  /** A one-page HTML report: summary, results, goals, chosen solutions, and the charts. */
  $("report").addEventListener("click", async () => {
    const t = P.tokens(), f = cur();
    if (!f) return;
    $("report").disabled = true;
    try {
      const shot = async (id) => {
        const el = $(id);
        if (!el || !el.data || !el.clientWidth) return "";
        const url = await Plotly.toImage(el, { format: "png", width: el.clientWidth, height: el.clientHeight, scale: 2 });
        return `<img src="${url}" alt="" style="width:100%;max-width:${el.clientWidth}px">`;
      };
      // The report is a white page: draw the charts in the light theme for the capture only.
      const root = document.documentElement, prevTheme = root.dataset.theme;
      root.dataset.theme = "light";
      for (const k in dirty) dirty[k] = true;
      render();
      await new Promise((r) => setTimeout(r, 120));
      let obj, vars, conv;
      try {
        [obj, vars, conv] = [await shot("obj-plot"), await shot("var-plot"), await shot("conv-plot")];
      } finally {
        if (prevTheme) root.dataset.theme = prevTheme; else delete root.dataset.theme;
        markDirty();
      }
      const ex = S.meta.execution;
      const summary = [
        ["Algorithm", S.meta.algorithm || "–"],
        ["Evaluated on", ex ? (ex.effective === "GPU" ? `GPU · ${ex.device || ""}` : `CPU · ${ex.cpu_threads || 1} thread(s)`) : "–"],
        ["Generations", P.fmt(f.gen)], ["Evaluations", P.fmt(f.nfe)], ["Wall time", f.t ? `${f.t.toFixed(2)} s` : "–"],
        ["Solutions on the front", P.fmt((f.F || []).length)],
      ];
      const metricLabel = $("metric").selectedOptions[0] ? $("metric").selectedOptions[0].textContent : "";
      const res = S.comparisons.length ? computeResults(t) : null;
      const solTable = (title, sols) => !sols.length ? "" : `<h2>${esc(title)}</h2><table><thead><tr><th></th>${[...Array(nObj()).keys()].map((i) => `<th>${esc(objLabel(i))}</th>`).join("")}${extraNames().map((e) => `<th>${esc(e)}</th>`).join("")}</tr></thead><tbody>${
        sols.map((s) => `<tr><td>${esc(s.label)}</td>${s.f.map((v) => `<td>${P.fmt(v)}</td>`).join("")}${extraNames().map((e) => `<td>${P.fmt(s.e ? s.e[e] : null)}</td>`).join("")}</tr>`).join("")}</tbody></table>
        <details><summary>Variables</summary><table><tbody>${sols.map((s) => `<tr><td>${esc(s.label)}</td><td>${s.x.map((v, j) => `${esc(varName(j))}=${P.fmt(v)}`).join(", ")}</td></tr>`).join("")}</tbody></table></details>`;
      const chosen = [];
      const si = selectedIndex(f), kn = kneeIndex(f);
      if (si >= 0) chosen.push({ label: "Selected", f: f.F[si], x: f.X[si], e: extrasFor(f, si) });
      if (kn >= 0) chosen.push({ label: "Most balanced", f: f.F[kn], x: f.X[kn], e: extrasFor(f, kn) });
      const goals = goalsActive() ? `<h2>Goals</h2><p>${[...Array(nObj()).keys()].filter((i) => Number.isFinite(S.goals[i])).map((i) => `${esc(objName(i))} ${dir(i) === 1 ? "≥" : "≤"} ${P.fmt(S.goals[i])}`).join(" · ")} — ${(goalGaps(f) || []).filter((g) => g <= 0).length} of ${f.F.length} front solutions meet them.</p>` : "";
      const html = `<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(S.meta.title || "puggles")} — report</title><style>
body{font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;color:#0b0b0b;background:#fff;max-width:1000px;margin:24px auto;padding:0 16px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:15px;margin:28px 0 8px}p.muted{color:#52514e;margin:0}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums;font-size:13px}
th,td{border-bottom:1px solid #e1e0d9;padding:4px 8px;text-align:right}th:first-child,td:first-child{text-align:left}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:700px){.grid{grid-template-columns:1fr}}
@media print{body{margin:0}}</style></head><body>
<h1>${esc(S.meta.title || "puggles run")}</h1><p class="muted">Report generated ${new Date().toLocaleString()} from the puggles dashboard.</p>
<h2>Summary</h2><table><tbody>${summary.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("")}</tbody></table>
${res ? `<h2>Results: ${esc(res.label)} (${res.up ? "higher" : "lower"} is better)</h2><table><thead><tr><th>Group</th><th>Runs</th><th>Median</th><th>Mean ± sd</th><th>p vs best</th></tr></thead><tbody>${
  res.summary.map((g, i) => `<tr><td>${esc(g.key)}</td><td>${g.n}</td><td>${P.fmt(g.median)}</td><td>${P.fmt(g.mean)}${g.sd === null ? "" : ` ± ${P.fmt(g.sd)}`}</td><td>${i === 0 ? "best" : g.p === null ? "–" : g.p.toFixed(3)}</td></tr>`).join("")}</tbody></table>` : ""}
${goals}
${solTable("Chosen solutions", chosen)}
${solTable("Pinned solutions", S.pins)}
<h2>Objective space</h2>${obj}
<div class="grid"><div><h2>Decision variables</h2>${vars}</div><div><h2>Convergence: ${esc(metricLabel)}</h2>${conv}</div></div>
</body></html>`;
      const name = (S.meta.title && S.meta.title !== "puggles" ? S.meta.title : "puggles").replace(/[^\w.-]+/g, "_");
      download(`${name}-report.html`, html, "text/html");
      say("Report downloaded");
    } catch (err) {
      showNotice(`Couldn't build the report: ${err.message}`);
    } finally {
      $("report").disabled = false;
    }
  });

  /* -- interactions ----------------------------------------------------------------------- */

  function select(f, k) {
    if (!f || !f.X || k == null || k < 0 || k >= f.X.length) return;
    const same = S.selected && keyOf(S.selected.x) === keyOf(f.X[k]);
    S.selected = same ? null : { x: f.X[k], f: f.F[k], ok: f.ok ? f.ok[k] : true };
    say(S.selected ? `Selected solution ${k}: ${f.F[k].map((v, i) => `${objName(i)} ${P.fmt(v)}`).join(", ")}` : "Selection cleared");
    markDirty("obj", "var", "side", "header");
  }

  function bindScatterEvents(gd) {
    gd.on("plotly_click", (ev) => {
      const pt = ev.points && ev.points.find((p) => p.data.meta === "archive");
      if (pt) select(cur(), pt.customdata !== undefined ? pt.customdata : pt.pointIndex);
    });
    gd.on("plotly_selected", (ev) => {
      const f = cur();
      if (!ev || !f) return;
      const pts = (ev.points || []).filter((p) => p.data.meta === "archive");
      S.lasso = pts.length ? new Set(pts.map((p) => keyOf(f.X[p.customdata !== undefined ? p.customdata : p.pointIndex]))) : null;
      markDirty("side");
    });
    gd.on("plotly_deselect", () => { S.lasso = null; markDirty("obj", "side"); });
  }

  /** Clear a plot and forget its handlers, so whatever is drawn next binds its own. */
  function purge(el) {
    Plotly.purge(el);
    el._pugglesBound = false;
  }

  /** Plotly.react, binding event handlers once per element. */
  function plot(el, traces, lay, bind) {
    const first = !el._pugglesBound;
    Plotly.react(el, traces, lay, P.config);
    if (first && bind) bind(el);
    el._pugglesBound = true;
  }

  /* -- export ------------------------------------------------------------------------------ */

  function exportCsv() {
    const f = cur();
    if (!f || !f.F) return;
    exportRows(listed(f).map((k) => ({ f: f.F[k], x: f.X[k], ok: f.ok ? f.ok[k] : true, e: extrasFor(f, k) })), `front-gen${f.gen}.csv`);
  }

  function exportRows(sols, name) {
    const m = nObj(), n = nVar();
    const extras = extraNames();
    const header = (sols[0] && sols[0].label !== undefined ? ["label"] : [])
      .concat([...Array(m).keys()].map(objName), extras, [...Array(n).keys()].map(varName), ["feasible"]);
    const q = (s) => (/[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);
    const lines = [header.map(q).join(",")];
    for (const s of sols) {
      lines.push((s.label !== undefined ? [s.label] : []).concat(s.f, extras.map((e) => (s.e ? s.e[e] : "")), s.x, [s.ok !== false]).join(","));
    }
    download(name, lines.join("\n") + "\n", "text/csv");
  }

  function download(name, text, type) {
    const url = URL.createObjectURL(new Blob([text], { type }));
    const a = document.createElement("a");
    a.href = url; a.download = name;
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // Re-theme everything when the OS scheme flips (and no explicit theme is set).
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => markDirty());
  document.addEventListener("puggles:theme", () => markDirty());

  restoreShapes();
  boot(); // last, so every declaration above is initialized

  /* -- theme + single-figure mode (shared) ------------------------------------------------- */

  function restoreTheme() {
    try {
      const saved = localStorage.getItem("puggles-theme");
      if (saved === "light" || saved === "dark") document.documentElement.dataset.theme = saved;
    } catch (_) { /* storage unavailable: follow the OS */ }
    if (DATA && DATA.kind === "plot" && DATA.spec && (DATA.spec.theme === "light" || DATA.spec.theme === "dark")
        && !document.documentElement.dataset.theme) {
      document.documentElement.dataset.theme = DATA.spec.theme;
    }
  }

  function toggleTheme() {
    const next = P.tokens().mode === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    try { localStorage.setItem("puggles-theme", next); } catch (_) { /* ignore */ }
    document.dispatchEvent(new Event("puggles:theme"));
  }

  function initFigure(spec) {
    document.body.classList.add("plot-only");
    const title = (spec.labs && spec.labs.title) || "puggles plot";
    document.title = title;
    $("title").textContent = "puggles";
    const el = $("figure");
    const draw = () => {
      try { P.renderSpec(el, spec); }
      catch (err) { el.innerHTML = `<p style="padding:16px">Could not render this plot: ${esc(err.message)}</p>`; }
    };
    draw();
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
    document.addEventListener("puggles:theme", draw);
  }
})();

/* puggles plot layer: theme tokens, palettes, Plotly layout helpers, and the renderer for
 * ggplot-style specs written by the Rust crate (`puggles::plot`). The dashboard (app.js) builds
 * its views from the same helpers, so a Rust figure and a dashboard view look the same.
 *
 * Spec (version 1):
 *   { version: 1,
 *     data:   { columns: { name: [values...] } },
 *     layers: [{ geom: "point"|"line"|"parallel"|"histogram"|"splom",
 *                aes: { x, y, z, color },          // column names
 *                params: { size, alpha, width, bins, dims: [column names] } }],
 *     labs:   { title, x, y, z, color },
 *     scales: { x: "linear"|"log", y: "linear"|"log", color: "sequential"|"diverging"|"categorical" },
 *     theme:  "auto"|"light"|"dark" }
 */
(function () {
  "use strict";

  // Categorical order is fixed (never cycled): identity follows the series, not its rank.
  const PALETTE = {
    light: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    dark: ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"],
  };
  // Sequential: one hue (blue). The low end recedes toward the surface in each mode.
  const SEQUENTIAL = {
    light: ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"],
    dark: ["#104281", "#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4", "#cde2fb"],
  };
  // Diverging: blue <-> red poles around a neutral gray midpoint.
  const DIVERGING = {
    light: ["#1c5cab", "#3987e5", "#86b6ef", "#f0efec", "#f1a09f", "#e34948", "#a8302f"],
    dark: ["#6da7ec", "#3987e5", "#1c5cab", "#383835", "#a8302f", "#e34948", "#f1a09f"],
  };
  const MAX_CATEGORIES = PALETTE.light.length; // past this, the tail folds into "Other"

  function mode() {
    const forced = document.documentElement.dataset.theme;
    if (forced === "light" || forced === "dark") return forced;
    return window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  function tokens() {
    const css = getComputedStyle(document.documentElement);
    const v = (name) => css.getPropertyValue(name).trim();
    const m = mode();
    return {
      mode: m,
      surface: v("--surface"), ink: v("--ink"), ink2: v("--ink-2"), muted: v("--muted"),
      grid: v("--grid"), axis: v("--axis"), border: v("--border"),
      series: PALETTE[m], accent: PALETTE[m][1],
    };
  }

  function colorscale(name, t) {
    const stops = (name === "diverging" ? DIVERGING : SEQUENTIAL)[t.mode];
    return stops.map((c, i) => [i / (stops.length - 1), c]);
  }

  const FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif';

  function axis(t, extra) {
    return Object.assign({
      gridcolor: t.grid, linecolor: t.axis, zerolinecolor: t.axis, zeroline: false,
      tickfont: { color: t.muted, size: 11 }, title: { font: { color: t.ink2, size: 12 }, standoff: 8 },
      // "~g": 9300, 0.25, 5e-7 — plain where readable, exponent only where needed.
      automargin: true, showline: true, ticks: "", tickformat: "~g",
    }, extra || {});
  }

  function sceneAxis(t, title) {
    return {
      title: { text: title, font: { color: t.ink2, size: 12 } },
      gridcolor: t.grid, linecolor: t.axis, zerolinecolor: t.axis, showbackground: false,
      tickfont: { color: t.muted, size: 10 }, tickformat: "~g",
    };
  }

  function layout(t, extra) {
    return Object.assign({
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: FONT, color: t.ink2, size: 12 },
      margin: { l: 56, r: 20, t: 10, b: 44 },
      hoverlabel: { bgcolor: t.surface, bordercolor: t.border, font: { family: FONT, color: t.ink, size: 12 } },
      legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { color: t.ink2, size: 12 }, bgcolor: "rgba(0,0,0,0)" },
      hovermode: "closest", dragmode: "zoom",
      modebar: { bgcolor: "rgba(0,0,0,0)", color: t.muted, activecolor: t.ink },
      colorway: t.series,
    }, extra || {});
  }

  function colorbar(t, title) {
    return {
      title: { text: title, side: "right", font: { color: t.ink2, size: 11 } },
      thickness: 10, outlinewidth: 0, len: 0.85, tickfont: { color: t.muted, size: 10 }, tickformat: "~g",
    };
  }

  // The camera button downloads a 2x PNG; the extra button downloads SVG (for papers).
  const config = {
    displaylogo: false, responsive: true, displayModeBar: "hover",
    modeBarButtonsToRemove: ["sendDataToCloud", "toggleSpikelines", "hoverCompareCartesian"],
    toImageButtonOptions: { format: "png", scale: 2, filename: "puggles" },
    modeBarButtonsToAdd: window.Plotly && Plotly.Icons ? [{
      name: "Download as SVG",
      icon: Plotly.Icons.disk || Plotly.Icons.camera,
      click: (gd) => Plotly.downloadImage(gd, { format: "svg", filename: "puggles" }),
    }] : [],
  };

  function fmt(v) {
    if (v === null || v === undefined || Number.isNaN(v)) return "–";
    if (typeof v === "boolean") return v ? "yes" : "no";
    if (typeof v !== "number") return String(v);
    if (!Number.isFinite(v)) return v > 0 ? "∞" : "−∞";
    if (Number.isInteger(v) && Math.abs(v) < 1e15) return v.toLocaleString("en-US");
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(3);
    return String(Number(v.toPrecision(5)));
  }

  /* -- spec rendering ----------------------------------------------------------------------- */

  function isNumeric(values) {
    return values.every((v) => v === null || typeof v === "number");
  }

  /** Split row indices by a categorical column, in sorted (entity-stable) order. */
  function categoryGroups(values) {
    const keys = [...new Set(values.map((v) => String(v)))].sort((a, b) =>
      a.localeCompare(b, undefined, { numeric: true }));
    const kept = keys.slice(0, keys.length > MAX_CATEGORIES ? MAX_CATEGORIES - 1 : MAX_CATEGORIES);
    const groups = new Map(kept.map((k) => [k, []]));
    const other = [];
    values.forEach((v, i) => (groups.get(String(v)) || other).push(i));
    const out = [...groups].map(([name, idx], slot) => ({ name, idx, slot }));
    if (other.length) out.push({ name: "Other", idx: other, slot: -1 });
    return out;
  }

  function pick(values, idx) {
    return idx.map((i) => values[i]);
  }

  function renderSpec(el, spec) {
    const t = tokens();
    const cols = (spec.data && spec.data.columns) || {};
    const labs = spec.labs || {};
    const scales = spec.scales || {};
    const col = (name) => {
      if (name == null) return null;
      if (!(name in cols)) throw new Error(`plot spec references missing column "${name}"`);
      return cols[name];
    };
    const traces = [];
    let is3d = false, hasCartesian = false, splomDims = 0, histogram = false;
    let showLegend = false;

    for (const layer of spec.layers || []) {
      const a = layer.aes || {};
      const p = layer.params || {};
      const colorVals = col(a.color);
      const categorical = colorVals && (scales.color === "categorical" || !isNumeric(colorVals));
      const colorTitle = labs.color || a.color;

      if (layer.geom === "point" || layer.geom === "line") {
        const xs = col(a.x), ys = col(a.y), zs = col(a.z);
        const three = zs !== null;
        is3d = is3d || three;
        hasCartesian = hasCartesian || !three;
        const n = xs.length;
        const groups = categorical ? categoryGroups(colorVals) : [{ name: layer.name || a.y, idx: [...Array(n).keys()], slot: 0 }];
        if (groups.length > 1) showLegend = true;
        for (const g of groups) {
          let idx = g.idx;
          if (layer.geom === "line" && !three) idx = [...idx].sort((i, j) => xs[i] - xs[j]); // like geom_line
          const solid = g.slot < 0 ? t.muted : t.series[g.slot];
          const continuous = colorVals && !categorical;
          const color = continuous ? pick(colorVals, idx) : solid;
          const hover = [
            `${labs.x || a.x}: %{x}`, `${labs.y || a.y}: %{y}`,
            three ? `${labs.z || a.z}: %{z}` : null,
            continuous ? `${colorTitle}: %{marker.color}` : null,
          ].filter(Boolean).join("<br>");
          const tr = {
            type: three ? "scatter3d" : "scattergl",
            mode: layer.geom === "point" ? "markers" : "lines",
            name: groups.length > 1 ? g.name : (layer.name || (layer.geom === "line" ? "line" : "points")),
            x: pick(xs, idx), y: pick(ys, idx),
            hovertemplate: hover + "<extra></extra>",
            opacity: p.alpha == null ? 1 : p.alpha,
          };
          if (three) tr.z = pick(zs, idx);
          if (layer.geom === "point") {
            tr.marker = {
              size: p.size || (three ? 4 : 8), color,
              line: three ? { width: 0 } : { color: t.surface, width: 1 },
            };
            if (continuous) Object.assign(tr.marker, {
              colorscale: colorscale(scales.color, t), showscale: true, colorbar: colorbar(t, colorTitle),
            });
          } else {
            tr.line = { width: p.width || 2, color: continuous ? t.series[0] : solid };
          }
          traces.push(tr);
        }
      } else if (layer.geom === "histogram") {
        histogram = true; hasCartesian = true;
        const xs = col(a.x);
        const groups = categorical ? categoryGroups(colorVals) : [{ name: a.x, idx: [...xs.keys()], slot: 0 }];
        if (groups.length > 1) showLegend = true;
        for (const g of groups) {
          traces.push({
            type: "histogram", name: g.name, x: pick(xs, g.idx), nbinsx: p.bins || 0,
            opacity: groups.length > 1 ? 0.6 : (p.alpha == null ? 1 : p.alpha),
            marker: { color: g.slot < 0 ? t.muted : t.series[g.slot], line: { color: t.surface, width: 1 } },
            hovertemplate: `${labs.x || a.x}: %{x}<br>count: %{y}<extra></extra>`,
          });
        }
      } else if (layer.geom === "parallel") {
        const dims = (p.dims || []).map((name) => ({ label: name, values: col(name) }));
        const line = colorVals && !categorical
          ? { color: colorVals, colorscale: colorscale(scales.color, t), showscale: true, colorbar: colorbar(t, colorTitle) }
          : { color: t.series[0] };
        traces.push(parcoords(t, dims, line));
      } else if (layer.geom === "splom") {
        const dims = (p.dims || []).map((name) => ({ label: name, values: col(name) }));
        splomDims = Math.max(splomDims, dims.length);
        const groups = categorical ? categoryGroups(colorVals) : [{ name: "points", idx: [...Array(dims[0] ? dims[0].values.length : 0).keys()], slot: 0 }];
        if (groups.length > 1) showLegend = true;
        for (const g of groups) {
          traces.push(splom(t, dims.map((d) => ({ label: d.label, values: pick(d.values, g.idx) })), {
            name: g.name, marker: { size: p.size || 5, color: g.slot < 0 ? t.muted : t.series[g.slot], line: { color: t.surface, width: 0.5 } },
          }));
        }
      } else {
        throw new Error(`unknown geom "${layer.geom}"`);
      }
    }

    const lay = layout(t, {
      showlegend: showLegend,
      margin: { l: 64, r: 24, t: labs.title ? 48 : 16, b: 52 },
      title: labs.title ? { text: labs.title, x: 0.01, xanchor: "left", font: { color: t.ink, size: 15 } } : undefined,
      barmode: histogram ? "overlay" : undefined,
    });
    if (hasCartesian) {
      const first = (spec.layers || [])[0] || {};
      const fa = first.aes || {};
      lay.xaxis = axis(t, { title: { text: labs.x || fa.x || "", font: { color: t.ink2, size: 12 } }, type: scales.x === "log" ? "log" : "linear" });
      lay.yaxis = axis(t, { title: { text: labs.y || fa.y || (histogram ? "count" : ""), font: { color: t.ink2, size: 12 } }, type: scales.y === "log" ? "log" : "linear" });
    }
    if (is3d) {
      const fa = ((spec.layers || []).find((l) => l.aes && l.aes.z) || {}).aes || {};
      lay.scene = {
        xaxis: sceneAxis(t, labs.x || fa.x), yaxis: sceneAxis(t, labs.y || fa.y), zaxis: sceneAxis(t, labs.z || fa.z),
        bgcolor: "rgba(0,0,0,0)", aspectmode: "cube",
      };
    }
    if (splomDims) styleSplomAxes(t, lay, splomDims);
    return Plotly.react(el, traces, lay, config);
  }

  /* -- shared trace builders (used by the dashboard too) ------------------------------------ */

  function parcoords(t, dimensions, line) {
    return {
      type: "parcoords", dimensions, line,
      // Plotly's default text shadow smudges labels on light backgrounds.
      labelfont: { color: t.ink2, size: 12, shadow: "none" }, tickfont: { color: t.muted, size: 10, shadow: "none" },
      // Range labels repeat the tick labels, so hide them; tilt labels only when they'd collide.
      rangefont: { color: "rgba(0,0,0,0)", size: 1, shadow: "none" },
      labelangle: dimensions.length * Math.max(...dimensions.map((d) => String(d.label).length)) > 90 ? -30 : 0,
      labelside: "top",
    };
  }

  function splom(t, dimensions, extra) {
    return Object.assign({
      type: "splom", dimensions, showupperhalf: false, diagonal: { visible: false },
    }, extra || {});
  }

  function styleSplomAxes(t, lay, n) {
    for (let i = 1; i <= n; i++) {
      const s = i === 1 ? "" : String(i);
      lay["xaxis" + s] = axis(t, { showline: false, tickfont: { color: t.muted, size: 9 } });
      lay["yaxis" + s] = axis(t, { showline: false, tickfont: { color: t.muted, size: 9 } });
    }
  }

  window.PugglesPlot = {
    PALETTE, tokens, colorscale, axis, sceneAxis, layout, colorbar, config, fmt,
    parcoords, splom, styleSplomAxes, renderSpec,
  };
})();

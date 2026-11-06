import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def imports():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import altair as alt
    import anywidget
    import traitlets

    return alt, anywidget, mo, np, pd, traitlets


@app.cell(hide_code=True)
def title(mo):
    mo.vstack([
        mo.md("# Normalization vs Standardization: an explorable lesson"),
        mo.Html('<iframe width="560" height="315" src="https://www.youtube.com/embed/87C5hkTY8RI" title="Normalization vs Standardization - Explained" frameborder="0" allowfullscreen></iframe>'),
        mo.md("""Based on **DataMListic: *Normalization vs Standardization - Explained***. Feature scaling is just a shift plus a stretch. In this notebook you move points, run gradient descent and add outliers to see **why** it matters and **which** scaler to choose.

    **Color language used throughout:** <span style="color:#6c5b7b">**purple = raw data**</span> ·
    <span style="color:#2E86AB">**blue = min-max normalization**</span> ·
    <span style="color:#F18F01">**orange = z-score standardization**</span> ·
    <span style="color:#e4572e">**red = outliers / the minimum we are looking for**</span>"""),
    ])
    return


@app.cell
def helpers(np):
    COLORS = {"raw": "#6c5b7b", "minmax": "#2E86AB", "zscore": "#F18F01", "outlier": "#e4572e"}

    def minmax(x):
        """Min-max normalization: x' = (x - min) / (max - min)."""
        x = np.asarray(x, float)
        _rng = x.max() - x.min()
        return (x - x.min()) / _rng if _rng > 0 else np.zeros_like(x)

    def zscore(x):
        """Z-score standardization: x' = (x - mu) / sigma  (population sigma, like sklearn's StandardScaler)."""
        x = np.asarray(x, float)
        _sd = x.std()
        return (x - x.mean()) / _sd if _sd > 0 else np.zeros_like(x)

    return COLORS, minmax, zscore


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. The scale problem: income is *louder* than age

    The video's running example: predict something about a person from **age** (about 25 to 52) and **income** (45,000 to 200,000).

    | feature | range |
    |---|---|
    | age | ≈ 27 |
    | income | ≈ 155,000 (**≈ 6,000× larger**) |

    Many algorithms quietly assume that **bigger numbers mean more importance**. Without scaling, the model hears income shouting and barely notices age.
    That is not because age is irrelevant. Income is simply measured in bigger units.

    The cost shows up in the **loss landscape**. With mismatched scales the loss contours become **long, thin ellipses** instead of circles.
    Gradient descent follows the steepest slope, so it **zig-zags** across the narrow valley and creeps along the long one.
    For a quadratic loss $L(w) = \tfrac12\,(w_1^2 + \kappa\, w_2^2)$, the stretch factor $\kappa$ grows like (scale ratio)².
    """)
    return


@app.cell
def people(mo, np, pd):
    people = (lambda _rng: pd.DataFrame({
        "age": np.round(_rng.uniform(25, 52, 30)),
        "income": np.round(np.clip(45_000 + 4_000 * (_rng.uniform(25, 52, 30) - 25) + _rng.normal(0, 25_000, 30), 45_000, 200_000), -2),
    }))(np.random.default_rng(7))
    _ranges = people.max() - people.min()
    mo.hstack([
        mo.stat(f"{_ranges['age']:.0f}", label="age range (years)", bordered=True),
        mo.stat(f"{_ranges['income']:,.0f}", label="income range", bordered=True),
        mo.stat(f"{_ranges['income'] / _ranges['age']:,.0f}×", label="how much 'louder' income is", bordered=True),
    ], justify="start")
    return (people,)


@app.cell
def gd_class(anywidget, traitlets):
    class GDWidget(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 440, H = 440, D = 10;
      const ctl = document.createElement("div"); ctl.style.marginBottom = "6px"; el.appendChild(ctl);
      const canvas = document.createElement("canvas"); canvas.width = W; canvas.height = H;
      Object.assign(canvas.style, { maxWidth: "100%", border: "1px solid #8884", borderRadius: "8px", cursor: "crosshair" });
      el.appendChild(canvas);
      const info = document.createElement("div"); info.style.font = "13px sans-serif"; info.style.marginTop = "4px"; el.appendChild(info);
      const g = canvas.getContext("2d");
      const sx = v => (v + D) / (2 * D) * W, sy = v => H - (v + D) / (2 * D) * H;
      const ix = p => p / W * 2 * D - D, iy = p => (H - p) / H * 2 * D - D;
      let timer = null, path = [];
      const btn = (t, f) => { const b = document.createElement("button"); b.textContent = t; b.onclick = f; b.style.marginRight = "6px"; ctl.appendChild(b); return b; };
      function draw() {
        const k = model.get("ratio");
        g.clearRect(0, 0, W, H);
        g.strokeStyle = "#8888"; g.lineWidth = 1;
        [0.5, 2, 5, 10, 18, 30, 45, 62].forEach(c => {
          const rx = Math.sqrt(2 * c), ry = Math.sqrt(2 * c / k);
          g.beginPath(); g.ellipse(sx(0), sy(0), rx / (2 * D) * W, ry / (2 * D) * H, 0, 0, 2 * Math.PI); g.stroke();
        });
        g.fillStyle = "#888"; g.font = "12px sans-serif";
        g.fillText("w_age  →", W - 70, sy(0) - 6); g.fillText("↑ w_income", sx(0) + 6, 14);
        g.fillStyle = "#e4572e"; g.beginPath(); g.arc(sx(0), sy(0), 6, 0, 2 * Math.PI); g.fill();
        const s = model.get("start");
        g.fillStyle = "#6c5b7b"; g.beginPath(); g.arc(sx(s[0]), sy(s[1]), 6, 0, 2 * Math.PI); g.fill();
        if (path.length > 1) {
          g.strokeStyle = "#6c5b7b"; g.lineWidth = 2; g.beginPath();
          path.forEach((p, i) => i ? g.lineTo(sx(p[0]), sy(p[1])) : g.moveTo(sx(p[0]), sy(p[1]))); g.stroke();
          g.fillStyle = "#6c5b7b"; path.forEach(p => { g.beginPath(); g.arc(sx(p[0]), sy(p[1]), 2.5, 0, 2 * Math.PI); g.fill(); });
        }
      }
      function stop(status) {
        clearInterval(timer); timer = null; run.textContent = "▶ run gradient descent";
        model.set("steps", path.length - 1); model.set("status", status); model.save_changes();
        info.innerHTML = `<b>${status}</b> after <b>${path.length - 1}</b> steps`;
      }
      function step() {
        const k = model.get("ratio"), lr = model.get("alpha") / k;
        const [x, y] = path[path.length - 1];
        const nx = x - lr * x, ny = y - lr * k * y;
        path.push([nx, ny]); draw();
        info.innerHTML = `step ${path.length - 1}: loss = ${(0.5 * (nx * nx + k * ny * ny)).toFixed(4)}`;
        if (Math.hypot(nx, ny) < 0.05) stop("converged");
        else if (!isFinite(nx) || Math.hypot(nx, ny) > 1e4) stop("diverged");
        else if (path.length > 400) stop("gave up (400 steps)");
      }
      const run = btn("▶ run gradient descent", () => {
        if (timer) { clearInterval(timer); timer = null; run.textContent = "▶ run gradient descent"; return; }
        if (!path.length || model.get("status") !== "running") path = [model.get("start").slice()];
        model.set("status", "running");
        timer = setInterval(step, 40); run.textContent = "⏸ pause";
      });
      btn("↺ reset", () => { clearInterval(timer); timer = null; path = []; run.textContent = "▶ run gradient descent"; info.textContent = "click the canvas to move the start point"; draw(); });
      canvas.addEventListener("click", e => {
        const r = canvas.getBoundingClientRect();
        const px = (e.clientX - r.left) * W / r.width, py = (e.clientY - r.top) * H / r.height;
        clearInterval(timer); timer = null; path = []; run.textContent = "▶ run gradient descent";
        model.set("start", [ix(px), iy(py)]); model.set("status", "idle"); model.save_changes(); draw();
      });
      info.textContent = "click the canvas to move the start point, then press run";
      model.on("change:ratio", () => { path = []; draw(); });
      model.on("change:start", draw);
      draw();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        ratio = traitlets.Float(1.0).tag(sync=True)
        alpha = traitlets.Float(1.8).tag(sync=True)
        start = traitlets.List(traitlets.Float(), default_value=[-8.0, 6.0]).tag(sync=True)
        steps = traitlets.Int(0).tag(sync=True)
        status = traitlets.Unicode("idle").tag(sync=True)

    return (GDWidget,)


@app.cell
def gd_controls(mo):
    ratio_slider = mo.ui.slider(1, 100, value=40, step=1, label="stretch κ (1 = scaled features)", show_value=True)
    alpha_slider = mo.ui.slider(0.1, 1.95, value=1.8, step=0.05, label="step size (× 1/κ in age direction)", show_value=True)
    mo.hstack([ratio_slider, alpha_slider], justify="start")
    return alpha_slider, ratio_slider


@app.cell
def gd(GDWidget, alpha_slider, mo, ratio_slider):
    gd = mo.ui.anywidget(GDWidget(ratio=float(ratio_slider.value), alpha=float(alpha_slider.value)))
    gd
    return (gd,)


@app.cell
def gd_readout(COLORS, alpha_slider, alt, gd, mo, np, pd, ratio_slider):
    def gd_steps(kappa, alpha, start=(-8.0, 6.0), tol=0.05, max_steps=400):
        """Steps of plain gradient descent on L = 0.5 (w1^2 + kappa w2^2) with lr = alpha / kappa."""
        _w = np.array(start, float)
        _lr = alpha / kappa
        for _t in range(1, max_steps + 1):
            _w = _w - _lr * np.array([1.0, kappa]) * _w
            if np.hypot(*_w) < tol:
                return _t
        return max_steps

    _ks = np.arange(1, 101)
    _df = pd.DataFrame({"κ": _ks, "steps": [gd_steps(k, alpha_slider.value) for k in _ks]})
    _line = alt.Chart(_df).mark_line(color=COLORS["raw"]).encode(x=alt.X("κ:Q", title="stretch κ"), y=alt.Y("steps:Q", title="steps to converge"))
    _now = alt.Chart(pd.DataFrame({"κ": [ratio_slider.value], "steps": [gd_steps(ratio_slider.value, alpha_slider.value)]})).mark_point(color=COLORS["outlier"], size=150, filled=True).encode(x="κ:Q", y="steps:Q")
    mo.vstack([
        mo.hstack([
            mo.stat(f"{gd.value['steps']}" if gd.value["status"] != "idle" else "–", label=f"your run ({gd.value['status']})", bordered=True),
            mo.stat(f"{gd_steps(1, 1.0)}", label="perfectly scaled (κ=1, step 1)", bordered=True),
        ], justify="start"),
        (_line + _now).properties(width="container", height=220, title="Same algorithm, same data, only the units differ"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Try this**
    1. Keep κ = 40 and press **run**. Watch the zig-zag across the narrow (income) direction and the crawl along the age direction.
    2. Drag κ down to **1**. That is what scaling does. The ellipses become circles and the path goes straight to the red minimum.
    3. With κ = 1, set the step to 1.0. It converges in a **single step**. With a large κ, no step size helps both directions at once.
    """), kind="info")
    return


@app.cell
def scaleline_class(anywidget, traitlets):
    class ScaleLine(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const NS = "http://www.w3.org/2000/svg", W = 700, PAD = 40, Y0 = 60;
      const svg = document.createElementNS(NS, "svg");
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none" });
      el.appendChild(svg);
      const info = document.createElement("div"); info.style.font = "13px sans-serif"; el.appendChild(info);
      const mk = (tag, attrs, parent = svg) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
      const toSvg = evt => { const p = svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const COL = { raw: "#6c5b7b", minmax: "#2E86AB", zscore: "#F18F01" };
      const RANGE = { minmax: [-0.1, 1.1], zscore: [-3.5, 3.5] };
      const TICKS = { minmax: [0, 0.25, 0.5, 0.75, 1], zscore: [-3, -2, -1, 0, 1, 2, 3] };
      const LABEL = { minmax: "min-max:  x' = (x − min) / (max − min)", zscore: "z-score:  x' = (x − μ) / σ" };
      let drag = -1;
      const rows = () => model.get("mode") === "both" ? ["minmax", "zscore"] : [model.get("mode")];
      const sxr = v => { const a = model.get("xmin"), b = model.get("xmax"); return PAD + (v - a) / (b - a) * (W - 2 * PAD); };
      const invr = px => { const a = model.get("xmin"), b = model.get("xmax"); return a + (px - PAD) / (W - 2 * PAD) * (b - a); };
      const sxt = (m, v) => { const [a, b] = RANGE[m]; return PAD + (Math.max(a, Math.min(b, v)) - a) / (b - a) * (W - 2 * PAD); };
      const stats = xs => { const n = xs.length, mn = Math.min(...xs), mx = Math.max(...xs), mu = xs.reduce((a, b) => a + b, 0) / n;
        return { mn, mx, mu, sd: Math.sqrt(xs.reduce((a, b) => a + (b - mu) ** 2, 0) / n) }; };
      const tf = (m, x, s) => m === "minmax" ? (s.mx > s.mn ? (x - s.mn) / (s.mx - s.mn) : 0) : (s.sd > 0 ? (x - s.mu) / s.sd : 0);
      const fmt = v => Math.abs(v) >= 100 ? v.toFixed(0) : (Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(2));
      function axis(y, sx, color, ticks, label) {
        mk("line", { x1: PAD, x2: W - PAD, y1: y, y2: y, stroke: color, "stroke-width": 2 });
        ticks.forEach(t => { mk("line", { x1: sx(t), x2: sx(t), y1: y, y2: y + 5, stroke: color });
          const tx = mk("text", { x: sx(t), y: y + 18, "text-anchor": "middle", "font-size": 11, fill: "currentColor" }); tx.textContent = fmt(t); });
        const l = mk("text", { x: PAD, y: y + 34, "font-size": 12, fill: color, "font-weight": "bold" }); l.textContent = label;
      }
      function draw() {
        svg.innerHTML = "";
        const R = rows(), h = Y0 + 50 + R.length * 105;
        svg.setAttribute("viewBox", `0 0 ${W} ${h}`);
        const xs = model.get("values"), s = stats(xs), a = model.get("xmin"), b = model.get("xmax");
        mk("rect", { x: sxr(s.mu - s.sd), y: Y0 - 26, width: Math.max(0, sxr(s.mu + s.sd) - sxr(s.mu - s.sd)), height: 26, fill: COL.zscore, opacity: 0.15 });
        const rt = []; for (let i = 0; i <= 5; i++) rt.push(a + (b - a) * i / 5);
        axis(Y0, sxr, COL.raw, rt, "raw x (" + model.get("unit") + "): drag dots · double-click to add / remove");
        mk("polygon", { points: `${sxr(s.mu)},${Y0 + 2} ${sxr(s.mu) - 7},${Y0 + 12} ${sxr(s.mu) + 7},${Y0 + 12}`, fill: COL.zscore });
        [s.mn, s.mx].forEach(v => mk("line", { x1: sxr(v), x2: sxr(v), y1: Y0 - 32, y2: Y0 + 4, stroke: COL.minmax, "stroke-width": 2, "stroke-dasharray": "4 3" }));
        R.forEach((m, r) => {
          const y = Y0 + 105 * (r + 1);
          xs.forEach(x => mk("line", { x1: sxr(x), y1: Y0, x2: sxt(m, tf(m, x, s)), y2: y, stroke: COL[m], opacity: 0.25 }));
          axis(y, v => sxt(m, v), COL[m], TICKS[m], LABEL[m]);
          xs.forEach(x => mk("circle", { cx: sxt(m, tf(m, x, s)), cy: y - 9, r: 7, fill: COL[m], stroke: "#fff", "stroke-width": 1.5 }));
        });
        xs.forEach((v, i) => {
          const c = mk("circle", { cx: sxr(v), cy: Y0 - 11, r: 9, fill: COL.raw, stroke: "#fff", "stroke-width": 2 });
          c.style.cursor = "grab";
          c.addEventListener("pointerdown", e => { drag = i; svg.setPointerCapture(e.pointerId); });
          c.addEventListener("dblclick", e => { e.stopPropagation(); const arr = [...model.get("values")];
            if (arr.length > 2) { arr.splice(i, 1); model.set("values", arr); model.save_changes(); } });
        });
        info.innerHTML = `<span style="color:${COL.minmax}">min = ${fmt(s.mn)}, max = ${fmt(s.mx)}</span> · <span style="color:${COL.zscore}">μ = ${fmt(s.mu)}, σ = ${fmt(s.sd)}</span> · n = ${xs.length}`;
      }
      const snap = v => { const st = model.get("snap"); return Math.max(model.get("xmin"), Math.min(model.get("xmax"), Math.round(v / st) * st)); };
      svg.addEventListener("dblclick", e => { const p = toSvg(e); if (p.y > Y0 + 20) return;
        model.set("values", [...model.get("values"), snap(invr(p.x))]); model.save_changes(); });
      svg.addEventListener("pointermove", e => { if (drag < 0) return; const arr = [...model.get("values")];
        arr[drag] = snap(invr(toSvg(e).x)); model.set("values", arr); });
      const stop = () => { if (drag >= 0) { drag = -1; model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointerleave", stop);
      model.on("change:values", draw); model.on("change:mode", draw); draw();
    }
    export default { render };
    """
        values = traitlets.List(traitlets.Float()).tag(sync=True)
        mode = traitlets.Unicode("minmax").tag(sync=True)
        xmin = traitlets.Float(0.0).tag(sync=True)
        xmax = traitlets.Float(100.0).tag(sync=True)
        snap = traitlets.Float(1.0).tag(sync=True)
        unit = traitlets.Unicode("").tag(sync=True)

    return (ScaleLine,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Min-max normalization: squeeze everything into [0, 1]

    $$x' = \frac{x - x_{\min}}{x_{\max} - x_{\min}}$$

    * The **numerator** shifts everything so the smallest value lands at **0**.
    * The **denominator** rescales so the largest value lands at **1**.

    Every feature is **guaranteed** to live in $[0, 1]$. It is clean and intuitive, and you always know where your data ends up.
    Below, the purple dots are incomes in thousands. The blue dashed lines mark min and max, the two numbers this scaler uses.
    """)
    return


@app.cell
def mm_line(ScaleLine, mo):
    mm_line = mo.ui.anywidget(ScaleLine(values=[45.0, 55.0, 68.0, 85.0, 100.0, 120.0, 158.0, 200.0], mode="minmax", xmin=0.0, xmax=250.0, snap=1.0, unit="income, $k"))
    mm_line
    return (mm_line,)


@app.cell
def mm_readout(minmax, mm_line, mo, np, pd):
    _x = np.array(mm_line.value["values"])
    _df = pd.DataFrame({"income ($k)": _x, "x − min": _x - _x.min(), "x' = (x − min)/(max − min)": minmax(_x).round(3)}).sort_values("income ($k)")
    mo.vstack([
        mo.md(f"min = **{_x.min():.0f}**, max = **{_x.max():.0f}**, range = **{np.ptp(_x):.0f}**. Every row gets divided by the same {np.ptp(_x):.0f}."),
        mo.ui.table(_df, selection=None, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Try this**
    1. Drag a **middle** dot. Only its own blue dot moves. Now drag the **largest** dot. *Every* blue dot moves, because the max is part of every row's formula.
    2. Drag the max far to the right (≈ 250). The other blue dots all crowd toward 0. Keep this in mind for section 4.
    3. Spacing ratios are preserved. If two dots are twice as far apart as two others in purple, the same holds in blue.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Z-score standardization: center at 0, one unit = one σ

    $$x' = \frac{x - \mu}{\sigma}$$

    Subtract the **mean** μ and divide by the **standard deviation** σ. For the video's income data μ ≈ 103,800 and σ ≈ 53,800.
    After the transform the data is centered at **0** and has **σ = 1**, whatever the original units were.
    A value of $x' = 2$ simply means *"two standard deviations above average"*.

    Same dots as before. The orange triangle is μ and the shaded band is μ ± σ, which maps to [−1, 1] below.
    """)
    return


@app.cell
def z_line(ScaleLine, mo):
    z_line = mo.ui.anywidget(ScaleLine(values=[45.0, 55.0, 68.0, 85.0, 100.0, 120.0, 158.0, 200.0], mode="zscore", xmin=0.0, xmax=250.0, snap=1.0, unit="income, $k"))
    z_line
    return (z_line,)


@app.cell
def z_readout(mo, np, z_line, zscore):
    _z = zscore(z_line.value["values"])
    mo.hstack([
        mo.stat(f"{np.mean(z_line.value['values']):.1f}k", label="μ (raw)", bordered=True),
        mo.stat(f"{np.std(z_line.value['values']):.1f}k", label="σ (raw)", bordered=True),
        mo.stat(f"{_z.mean():+.2f}", label="mean after z-score", bordered=True),
        mo.stat(f"{_z.std():.2f}", label="σ after z-score", bordered=True),
        mo.stat(f"[{_z.min():.2f}, {_z.max():.2f}]", label="range after (not fixed!)", bordered=True),
    ], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Misconception: "standardizing makes my data normal"

    It **does not**. Standardization is a **linear** transformation: a shift and a rescale. The *shape* of the distribution stays exactly the same.
    A skewed feature stays skewed and a bimodal feature stays bimodal. Pick a shape and compare.
    """)
    return


@app.cell
def dist_dropdown(mo):
    dist_dropdown = mo.ui.dropdown(["right-skewed (lognormal)", "exponential", "bimodal", "uniform"], value="right-skewed (lognormal)", label="raw distribution")
    dist_dropdown
    return (dist_dropdown,)


@app.cell
def shape_demo(COLORS, alt, dist_dropdown, minmax, mo, np, pd, zscore):
    _rng = np.random.default_rng(3)
    _raw = {
        "right-skewed (lognormal)": lambda: _rng.lognormal(10.5, 0.6, 3000),
        "exponential": lambda: _rng.exponential(40, 3000),
        "bimodal": lambda: np.r_[_rng.normal(30, 5, 1500), _rng.normal(70, 8, 1500)],
        "uniform": lambda: _rng.uniform(25, 52, 3000),
    }[dist_dropdown.value]()

    def _hist(x, color, title):
        _c, _e = np.histogram(x, bins=40)
        _d = pd.DataFrame({"lo": _e[:-1], "hi": _e[1:], "count": _c})
        return alt.Chart(_d).mark_bar(color=color).encode(x=alt.X("lo:Q", title=title), x2="hi:Q", y=alt.Y("count:Q", title=None)).properties(width=210, height=170)

    def _skew(x):
        return float(np.mean(zscore(x) ** 3))

    mo.vstack([
        alt.hconcat(_hist(_raw, COLORS["raw"], "raw"), _hist(minmax(_raw), COLORS["minmax"], "min-max"), _hist(zscore(_raw), COLORS["zscore"], "z-score")),
        mo.md(f"Skewness: raw **{_skew(_raw):.3f}** · min-max **{_skew(minmax(_raw)):.3f}** · z-score **{_skew(zscore(_raw)):.3f}**. Identical, so the shape is untouched."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Try this**
    1. Drag one dot far right in the z-score widget. Unlike min-max, the result has **no fixed range**. The extreme dot just gets a big |z|.
    2. Check that "mean after" stays **0.00** and "σ after" stays **1.00** however you drag.
    3. In the dropdown, choose *exponential*. All three histograms have the same shape, and the skewness is unchanged.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. The outlier test

    So which one should you use? The video says: **it comes down to outliers.**
    Take a well-behaved feature with values roughly in 0–100 and add **one** extreme value, say **500**.

    * **Min-max:** the outlier becomes the new max and stretches the range enormously. All regular data gets **compressed near 0** and loses most of its resolution.
    * **Z-score:** the outlier drags μ and σ a bit, but the result is **not boxed into [0, 1]**. The outlier just shows up far from the center.

    **Predict first:** after adding the 500 and applying min-max, what fraction of the [0, 1] interval will the 40 regular points cover?
    """)
    return


@app.cell
def guess(mo, np):
    regular = np.clip(np.random.default_rng(11).normal(50, 15, 40), 0, 100).round(1)
    guess_slider = mo.ui.slider(0.0, 1.0, value=0.5, step=0.01, label="my guess: fraction of [0,1] covered by regular data", show_value=True)
    lock_btn = mo.ui.run_button(label="Lock in guess")
    mo.hstack([guess_slider, lock_btn], justify="start")
    return guess_slider, lock_btn, regular


@app.cell
def guess_reveal(guess_slider, lock_btn, mo, np, regular):
    _truth = np.ptp(regular) / (500 - regular.min())
    mo.stop(not lock_btn.value, mo.md("_Make your guess, then lock it in to reveal the answer._"))
    _err = abs(guess_slider.value - _truth)
    mo.callout(mo.md(f"Answer: **{_truth:.2f}**. Your guess was {guess_slider.value:.2f}, off by **{_err:.2f}** {'🎯' if _err < 0.05 else ''}. "
                     f"The regular data spans {regular.min():.0f}–{regular.max():.0f}, but the range is now 500 − {regular.min():.0f}."),
               kind="success" if _err < 0.05 else "warn")
    return


@app.cell
def outlier_controls(mo):
    outlier_switch = mo.ui.switch(value=True, label="add one outlier")
    outlier_slider = mo.ui.slider(100, 1000, value=500, step=10, label="outlier value", show_value=True)
    mo.hstack([outlier_switch, outlier_slider], justify="start")
    return outlier_slider, outlier_switch


@app.cell
def outlier_chart(
    COLORS,
    alt,
    minmax,
    mo,
    np,
    outlier_slider,
    outlier_switch,
    pd,
    regular,
    zscore,
):
    def robust(x):
        """Robust scaling (bonus): (x - median) / IQR."""
        x = np.asarray(x, float)
        _q1, _q3 = np.percentile(x, [25, 75])
        return (x - np.median(x)) / (_q3 - _q1)

    _x = np.r_[regular, [outlier_slider.value]] if outlier_switch.value else regular.copy()
    _kind = np.where(np.arange(len(_x)) == 40, "outlier", "regular")
    _rows = []
    for _name, _f in [("min-max", minmax), ("z-score", zscore), ("robust (bonus)", robust)]:
        _rows.append(pd.DataFrame({"scaler": _name, "value": _f(_x), "kind": _kind}))
    outlier_df = pd.concat(_rows)

    def _strip(name, color):
        _d = outlier_df[outlier_df.scaler == name]
        return alt.Chart(_d).mark_tick(thickness=2, size=28).encode(
            x=alt.X("value:Q", title=f"{name} value"),
            color=alt.Color("kind:N", scale=alt.Scale(domain=["regular", "outlier"], range=[color, COLORS["outlier"]]), legend=None),
            tooltip=["value:Q", "kind:N"],
        ).properties(width=620, height=45, title=name)

    _reg = outlier_df[outlier_df.kind == "regular"].groupby("scaler", sort=False)["value"].agg(lambda v: v.max() - v.min())
    mo.vstack([
        alt.vconcat(_strip("min-max", COLORS["minmax"]), _strip("z-score", COLORS["zscore"]), _strip("robust (bonus)", "#888")).resolve_scale(x="independent"),
        mo.hstack([
            mo.stat(f"{_reg['min-max']:.2f}", label="regular data width, min-max (bounded box is 1)", bordered=True),
            mo.stat(f"{_reg['z-score']:.2f}", label="regular data width, z-score (in σ units)", bordered=True),
            mo.stat(f"{_reg['robust (bonus)']:.2f}", label="regular data width, robust (in IQR units)", bordered=True),
        ], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Try this**
    1. Toggle the outlier off and on. In the blue strip the regular ticks collapse into a sliver near 0. In the orange strip they keep a usable spread.
    2. Push the outlier to 1000. Min-max gets worse and worse. Z-score also compresses, but more slowly, because σ grows only about linearly with one extreme value.
    3. **Honest footnote:** both scalers are linear, so neither is truly outlier-proof. The grey *robust* row uses median and IQR, which one outlier barely moves (sklearn's `RobustScaler`).
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Two sides of the same coin

    Both methods are the **same idea**. Each is a linear map $x' = a\,x + b$:

    | | scale $a$ | shift $b$ | output |
    |---|---|---|---|
    | <span style="color:#2E86AB">min-max</span> | $1/(x_{\max}-x_{\min})$ | $-x_{\min}/(x_{\max}-x_{\min})$ | always in [0, 1] |
    | <span style="color:#F18F01">z-score</span> | $1/\sigma$ | $-\mu/\sigma$ | mean 0, σ 1, unbounded |

    Order, relative spacing and distribution shape stay intact. What changes is **what the algorithm sees**: a fair representation in which no feature shouts because of its units.

    That matters most for algorithms that use **distances** (k-NN, k-means, SVM) or **gradients** (linear and logistic regression, neural nets).
    Tree-based models split one feature at a time on thresholds, so they barely care.
    Below, pick a person from the age/income table and see who their **nearest neighbor** is under each view.
    """)
    return


@app.cell
def person_dropdown(mo, people):
    person_dropdown = mo.ui.dropdown({f"person {i}: age {int(a)}, income {int(inc):,}": i for i, (a, inc) in people.iterrows()}, value=f"person 0: age {int(people.age[0])}, income {int(people.income[0]):,}", label="target person")
    person_dropdown
    return (person_dropdown,)


@app.cell
def knn_demo(COLORS, alt, minmax, mo, np, pd, people, person_dropdown, zscore):
    _t = person_dropdown.value
    _views = {"raw": people[["age", "income"]].to_numpy(float),
              "min-max": np.column_stack([minmax(people.age), minmax(people.income)]),
              "z-score": np.column_stack([zscore(people.age), zscore(people.income)])}
    _cols = {"raw": COLORS["raw"], "min-max": COLORS["minmax"], "z-score": COLORS["zscore"]}
    _charts, _summary = [], []
    for _name, _X in _views.items():
        _d2 = ((_X - _X[_t]) ** 2)
        _dist = np.sqrt(_d2.sum(1)); _dist[_t] = np.inf
        _nn = int(np.argmin(_dist))
        _share = _d2[_nn, 1] / _d2[_nn].sum()
        _summary.append({"view": _name, "nearest neighbor": _nn, "neighbor age": int(people.age[_nn]), "neighbor income": f"{int(people.income[_nn]):,}",
                         "income's share of squared distance": f"{_share:.1%}"})
        _d = pd.DataFrame({"x": _X[:, 0], "y": _X[:, 1], "role": "other"})
        _d.loc[_nn, "role"] = "nearest"; _d.loc[_t, "role"] = "target"
        _charts.append(alt.Chart(_d).mark_circle(size=70).encode(
            x=alt.X("x:Q", title=f"age ({_name})", scale=alt.Scale(zero=False)), y=alt.Y("y:Q", title=f"income ({_name})", scale=alt.Scale(zero=False)),
            color=alt.Color("role:N", scale=alt.Scale(domain=["other", "target", "nearest"], range=["#bbb", "#222", _cols[_name]]), legend=None),
            size=alt.condition("datum.role != 'other'", alt.value(180), alt.value(50)),
        ).properties(width=200, height=200, title=_name))
    mo.vstack([alt.hconcat(*_charts), mo.ui.table(pd.DataFrame(_summary), selection=None, pagination=False),
               mo.md("_Black = target, colored = its nearest neighbor. In the raw view the 'nearest' person is simply whoever has the closest income. Age is ignored._")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md("""
    **Try this**
    1. Look at *income's share of squared distance* in the raw row. It is close to 100%, so age has no say at all.
    2. Click through a few people. How often does the raw neighbor differ from the scaled one? Do min-max and z-score usually agree?
    3. **Practical rule from the video:** naturally bounded data (pixels 0–255, percentages) → **min-max**. Possible outliers, or distance/gradient-based models → **z-score**.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Sandbox: both scalers at once

    Pick a preset, then drag, add (double-click the top axis) or remove (double-click a dot) points. The **blue** and **orange** rows update live.
    Challenges:
    * Make the min-max and z-score rows look **identical** in spacing. (They always are! Only the axis labels differ.)
    * Create a data set where z-score values reach beyond ±3.
    * Find out how few points you need before one point pins min-max to 1 while its z is only ≈ 1.
    """)
    return


@app.cell
def preset_dropdown(mo):
    preset_dropdown = mo.ui.dropdown(["even spread", "tight cluster + 1 outlier", "two clusters", "pixel values 0–255"], value="tight cluster + 1 outlier", label="preset")
    preset_dropdown
    return (preset_dropdown,)


@app.cell
def sandbox(ScaleLine, mo, preset_dropdown):
    _presets = {
        "even spread": [10.0, 25.0, 40.0, 55.0, 70.0, 85.0],
        "tight cluster + 1 outlier": [40.0, 42.0, 44.0, 45.0, 47.0, 48.0, 50.0, 52.0, 95.0],
        "two clusters": [15.0, 18.0, 20.0, 22.0, 75.0, 78.0, 80.0, 83.0],
        "pixel values 0–255": [0.0, 30.0, 60.0, 90.0, 128.0, 170.0, 210.0, 255.0],
    }
    _hi = 255.0 if preset_dropdown.value.startswith("pixel") else 100.0
    sandbox = mo.ui.anywidget(ScaleLine(values=_presets[preset_dropdown.value], mode="both", xmin=0.0, xmax=_hi, snap=1.0, unit="any units"))
    sandbox
    return (sandbox,)


@app.cell
def sandbox_readout(minmax, mo, np, pd, sandbox, zscore):
    _x = np.array(sandbox.value["values"])
    mo.ui.table(pd.DataFrame({
        "scaler": ["raw", "min-max", "z-score"],
        "min": [_x.min(), minmax(_x).min(), zscore(_x).min()],
        "max": [_x.max(), minmax(_x).max(), zscore(_x).max()],
        "mean": [_x.mean(), minmax(_x).mean(), zscore(_x).mean()],
        "std": [_x.std(), minmax(_x).std(), zscore(_x).std()],
        "corr with raw": [1.0, np.corrcoef(_x, minmax(_x))[0, 1], np.corrcoef(_x, zscore(_x))[0, 1]],
    }).round(3), selection=None, pagination=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Quiz: check yourself
    """)
    return


@app.cell
def quiz(mo):
    quiz_questions = {
        "q1": ("Grayscale pixel intensities, always between 0 and 255. Best default?", ["min-max", "z-score"], "min-max"),
        "q2": ("After z-score standardization, a right-skewed feature becomes…", ["normally distributed", "still right-skewed", "symmetric but not normal"], "still right-skewed"),
        "q3": ("Data in 0–100 plus one value of 500. Which scaler crushes the regular data into a narrow box near 0?", ["min-max", "z-score"], "min-max"),
        "q4": ("A standardized value of x' = −2 means…", ["2 units below the minimum", "2 standard deviations below the mean", "the 2nd smallest value"], "2 standard deviations below the mean"),
        "q5": ("Which model is LEAST affected by feature scaling?", ["k-nearest neighbors", "decision tree", "logistic regression trained with gradient descent"], "decision tree"),
        "q6": ("Unscaled features turn the loss contours into…", ["circles", "elongated ellipses", "straight lines"], "elongated ellipses"),
    }
    quiz = mo.ui.dictionary({k: mo.ui.dropdown(opts, label=q) for k, (q, opts, _a) in quiz_questions.items()})
    reveal = mo.ui.switch(label="reveal answers")
    mo.vstack([quiz, reveal])
    return quiz, quiz_questions, reveal


@app.cell
def quiz_score(mo, quiz, quiz_questions, reveal):
    _score = sum(quiz.value[k] == a for k, (_q, _o, a) in quiz_questions.items())
    _rows = [f"| {q} | {quiz.value[k] or '–'} | {'✅' if quiz.value[k] == a else '❌'} {a if reveal.value else ''} |" for k, (q, _o, a) in quiz_questions.items()]
    mo.vstack([
        mo.stat(f"{_score} / {len(quiz_questions)}", label="score", bordered=True),
        mo.md("| question | your answer | |\n|---|---|---|\n" + "\n".join(_rows)) if reveal.value else mo.md("_Flip **reveal answers** to see the correct ones._"),
    ])
    return


@app.cell(hide_code=True)
def recap(mo):
    mo.md(r"""
    ## Recap

    | | <span style="color:#2E86AB">Min-max normalization</span> | <span style="color:#F18F01">Z-score standardization</span> |
    |---|---|---|
    | formula | $x' = \dfrac{x - x_{\min}}{x_{\max} - x_{\min}}$ | $x' = \dfrac{x - \mu}{\sigma}$ |
    | output | always in [0, 1] | mean 0, σ 1, unbounded |
    | uses | min & max (2 extreme points) | μ & σ (all points) |
    | outliers | one outlier squashes the rest near 0 | handles them more gracefully (use robust scaling for heavy outliers) |
    | shape of distribution | unchanged | unchanged (**not** made normal) |
    | good for | naturally bounded data (pixels, percentages) | data with outliers; distance- and gradient-based models |

    **Shared core:** both are linear maps $x' = a x + b$. They shift and rescale, keep order and shape, and give each feature a fair voice, so gradient descent gets round contours and k-NN looks at every feature.
    """)
    return


if __name__ == "__main__":
    app.run()

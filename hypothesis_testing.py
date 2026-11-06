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
    from scipy import stats

    return alt, anywidget, mo, np, pd, stats, traitlets


@app.cell(hide_code=True)
def title(mo):
    mo.vstack([
        mo.md(r"""
    # Hypothesis testing, explained interactively
    *An interactive companion to DataMListic's video **"Hypothesis Testing - Explained"**.*
    """),
        mo.Html('<iframe width="560" height="315" src="https://www.youtube.com/embed/xpbmlxFmw_c" title="Hypothesis Testing - Explained" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'),
        mo.md(r"""
    **One-line framing:** a hypothesis test is a trial. You assume the boring explanation is true, then ask whether the evidence is too surprising to keep believing it.

    We follow the video's chapters and its running example: a factory claims its **light bulbs last 1,000 hours on average**. We test **36 bulbs**, the known standard deviation is **σ = 50**, and we measure **x̄ = 985**.

    | Chapter | What you'll touch |
    |---|---|
    | 1 · The courtroom | Choose the verdict rules |
    | 2 · The null world | A **bulb-factory simulator** that builds the null distribution of x̄ |
    | 3 · The test statistic | A **draggable z-ruler**: move x̄ and read off z |
    | 4 · The p-value | Shaded tails that follow your drag |
    | 5 · The decision | Run 2,000 trials and count Type I and Type II errors |
    | 6 · The whole picture | **Drag the alternative world** and watch the power grow |

    (The companion notebook `p_values.py` covers how to interpret p-values and the problem of p-hacking. Here we focus on how a test is put together.)

    **Colour language (used everywhere):**
    <span style="color:#4c78a8">■ blue = the null world (H₀)</span> ·
    <span style="color:#f58518">■ orange = the alternative world (H₁)</span> ·
    <span style="color:#e4572e">■ red = the rejection region / α</span> ·
    <span style="color:#2ca02c">■ green = power</span> ·
    <span style="color:#6c5b7b">■ purple = your observed data</span>
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The courtroom

    > *"A defendant walks into a courtroom. The court doesn't start by asking 'Is this person guilty?' It starts with a presumption: innocent until proven guilty."*

    | Courtroom | Hypothesis test | Bulb example |
    |---|---|---|
    | Innocent (presumed) | **Null hypothesis H₀** | $\mu = 1000$ h (the factory's claim) |
    | Guilty | **Alternative hypothesis H₁** | $\mu \neq 1000$ h |
    | Evidence | Sample data → test statistic | 36 bulbs, $\bar x = 985$ |
    | "Beyond reasonable doubt" | Significance level **α** | usually 0.05 |
    | Convict | **Reject H₀** | |
    | Acquit | **Fail to reject H₀** (never "accept H₀") | |

    The burden of proof is on the evidence. We **start by assuming H₀ is true** and only abandon it if the data would be very surprising in that world.
    """)
    return


@app.cell
def court_q(mo):
    court_q = mo.ui.radio(
        options=[
            "Guilty or not guilty",
            "Guilty or innocent",
        ],
        label="**Quick check:** a jury acquits. Which pair of verdicts did the court actually choose between?",
    )
    court_q
    return (court_q,)


@app.cell(hide_code=True)
def _(court_q, mo):
    mo.md(
        "_Pick one above._" if court_q.value is None else
        ("✅ **Exactly.** \"Not guilty\" means *not enough evidence*. It does not prove innocence. "
         "In the same way, **fail to reject H₀** does not mean H₀ is true."
         if court_q.value == "Guilty or not guilty" else
         "🤔 **Not quite.** A court never declares anyone *innocent*, only *not guilty*. "
         "A test works the same way: we **fail to reject** H₀. We never **accept** it.")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · The null world

    If H₀ is true, every bulb comes from a factory with mean $\mu_0 = 1000$ and $\sigma = 50$.
    We never see one bulb at a time. We see the **average of 36**. The **central limit theorem** says that average is approximately normal:

    $$\bar X \;\overset{H_0}{\sim}\; \mathcal N\!\left(\mu_0,\ \frac{\sigma}{\sqrt n}\right),\qquad \text{SE} = \frac{50}{\sqrt{36}} = 8.33 \text{ h}$$

    This bell curve is *"what the world looks like if H₀ is true"*, the **null world**.
    Below, the factory makes batches of bulbs. Each batch of *n* bulbs (purple ticks) becomes **one** dot in the histogram of x̄.
    The blue curve is the CLT prediction. Even single bulbs from a *skewed* factory produce a bell-shaped x̄.
    """)
    return


@app.cell
def sim_controls(mo):
    sim_n = mo.ui.slider(1, 100, value=36, label="bulbs per test, n", show_value=True)
    sim_shape = mo.ui.dropdown(["normal", "skewed (exponential-ish)"], value="normal", label="lifetime shape of one bulb")
    mo.hstack([sim_n, sim_shape], justify="start", gap=2)
    return sim_n, sim_shape


@app.cell
def BulbFactory_cls(anywidget, traitlets):
    class BulbFactory(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 680, H = 300;
      let timer = null, means = [];
      const ctl = document.createElement("div");
      ctl.style.cssText = "display:flex;gap:6px;align-items:center;margin-bottom:6px;font-family:sans-serif;font-size:13px";
      el.appendChild(ctl);
      const btn = (t, f) => { const b = document.createElement("button"); b.textContent = t; b.onclick = f;
        b.style.cssText = "padding:3px 10px;border-radius:6px;border:1px solid #999;background:transparent;color:inherit;cursor:pointer"; ctl.appendChild(b); return b; };
      const info = document.createElement("span"); info.style.marginLeft = "10px";
      const canvas = document.createElement("canvas"); canvas.width = W; canvas.height = H;
      canvas.style.cssText = "width:100%;max-width:" + W + "px";
      el.appendChild(canvas);
      const ctx = canvas.getContext("2d");
      const gauss = () => { let u = 0, v = 0; while (!u) u = Math.random(); v = Math.random();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
      function bulb() {
        const mu = model.get("mu"), s = model.get("sigma");
        if (model.get("shape") === "normal") return mu + s * gauss();
        return mu - s + s * (-Math.log(1 - Math.random()));   // shifted exponential: mean mu, sd s
      }
      let lastBatch = [];
      const XMIN = 900, XMAX = 1100;
      const sx = v => 30 + (v - XMIN) / (XMAX - XMIN) * (W - 60);
      function step() {
        const n = model.get("n"); lastBatch = [];
        for (let i = 0; i < n; i++) lastBatch.push(bulb());
        means.push(lastBatch.reduce((a, b) => a + b, 0) / n);
        draw();
        if (means.length % 20 === 0) push();
      }
      function push() { model.set("means", means.slice(-5000)); model.save_changes(); }
      function draw() {
        ctx.clearRect(0, 0, W, H);
        const n = model.get("n"), s = model.get("sigma"), mu0 = model.get("mu0"), se = s / Math.sqrt(n);
        ctx.fillStyle = getComputedStyle(el).color || "#333"; ctx.font = "12px sans-serif";
        // strip of single bulbs (top)
        ctx.fillText("last batch of single bulbs", 30, 16);
        ctx.strokeStyle = "rgba(108,91,123,0.7)";
        lastBatch.forEach(v => { const x = sx(Math.max(XMIN, Math.min(XMAX, v))); ctx.beginPath(); ctx.moveTo(x, 22); ctx.lineTo(x, 40); ctx.stroke(); });
        // histogram of means
        const B = 50, bw = (XMAX - XMIN) / B, counts = new Array(B).fill(0);
        means.forEach(m => { const k = Math.floor((m - XMIN) / bw); if (k >= 0 && k < B) counts[k]++; });
        const base = H - 30, top = 60, ph = base - top;
        // density scale: expected peak height of normal curve
        const peak = Math.max(1 / (se * Math.sqrt(2 * Math.PI)), 1e-9);
        const scale = ph / (peak * 1.15);
        const tot = Math.max(means.length, 1);
        ctx.fillStyle = "rgba(108,91,123,0.55)";
        counts.forEach((c, k) => { const d = c / tot / bw; const h = Math.min(ph, d * scale);
          ctx.fillRect(sx(XMIN + k * bw) + 1, base - h, (W - 60) / B - 2, h); });
        // null curve
        ctx.strokeStyle = "#4c78a8"; ctx.lineWidth = 2.5; ctx.beginPath();
        for (let px = 0; px <= W - 60; px++) { const v = XMIN + px / (W - 60) * (XMAX - XMIN);
          const d = Math.exp(-0.5 * ((v - mu0) / se) ** 2) / (se * Math.sqrt(2 * Math.PI));
          const y = base - Math.min(ph + 20, d * scale); px ? ctx.lineTo(30 + px, y) : ctx.moveTo(30 + px, y); }
        ctx.stroke(); ctx.lineWidth = 1;
        // axis
        ctx.strokeStyle = "#888"; ctx.beginPath(); ctx.moveTo(30, base); ctx.lineTo(W - 30, base); ctx.stroke();
        ctx.fillStyle = getComputedStyle(el).color || "#333";
        for (let t = XMIN; t <= XMAX; t += 25) { ctx.fillText(t, sx(t) - 14, base + 16); }
        // observed 985
        const obs = model.get("observed");
        ctx.strokeStyle = "#6c5b7b"; ctx.setLineDash([5, 4]); ctx.beginPath(); ctx.moveTo(sx(obs), top - 5); ctx.lineTo(sx(obs), base); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillStyle = "#6c5b7b"; ctx.fillText("x̄ = " + obs, sx(obs) - 50, top + 4);
        info.textContent = `tests run: ${means.length}   ·   SE = σ/√n = ${se.toFixed(2)} h`;
      }
      const play = btn("▶ play", () => { if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; push(); }
        else { timer = setInterval(() => { for (let i = 0; i < model.get("speed"); i++) step(); }, 40); play.textContent = "⏸ pause"; } });
      btn("one test", step);
      btn("+500 tests", () => { for (let i = 0; i < 500; i++) step(); push(); });
      btn("↺ reset", () => { means = []; lastBatch = []; draw(); push(); });
      ctl.appendChild(info);
      const reset = () => { means = []; lastBatch = []; draw(); push(); };
      model.on("change:n", reset); model.on("change:shape", reset); draw();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        n = traitlets.Int(36).tag(sync=True)
        mu = traitlets.Float(1000.0).tag(sync=True)
        mu0 = traitlets.Float(1000.0).tag(sync=True)
        sigma = traitlets.Float(50.0).tag(sync=True)
        shape = traitlets.Unicode("normal").tag(sync=True)
        observed = traitlets.Float(985.0).tag(sync=True)
        speed = traitlets.Int(3).tag(sync=True)
        means = traitlets.List(traitlets.Float()).tag(sync=True)

    return (BulbFactory,)


@app.cell
def factory(BulbFactory, mo, sim_n, sim_shape):
    factory = mo.ui.anywidget(BulbFactory(n=sim_n.value, shape="normal" if sim_shape.value == "normal" else "skewed"))
    factory
    return (factory,)


@app.cell
def factory_readout(factory, mo, np, sim_n, stats):
    _m = np.array(factory.value["means"])
    _se = 50 / np.sqrt(sim_n.value)
    if len(_m) < 2:
        _out = mo.md("_Press **▶ play** or **+500 tests** above. Stats appear here every 20 tests._")
    else:
        _out = mo.hstack([
            mo.stat(f"{len(_m)}", label="simulated tests"),
            mo.stat(f"{_m.mean():.1f} h", label="mean of x̄", caption="CLT: 1000"),
            mo.stat(f"{_m.std(ddof=1):.2f} h", label="spread of x̄", caption=f"CLT: σ/√n = {_se:.2f}"),
            mo.stat(f"{np.mean(_m <= 985) * 100:.1f}%", label="tests with x̄ ≤ 985",
                    caption=f"theory: {stats.norm.cdf((985 - 1000) / _se) * 100:.1f}%"),
        ], justify="start", gap=2)
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Press **+500 tests** a few times. Does the purple histogram fit the blue curve? About how often does an *honest* factory give x̄ ≤ 985?
    2. Set **n = 4**. The null world gets wider, so 985 is no longer unusual. Set **n = 100**. Now 985 sits in the far tail.
    3. Switch to the **skewed** bulb shape with n = 1, then with n = 36. Single bulbs are lopsided, but averages of 36 still form a bell. That is the CLT at work.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · The test statistic: a ruler for surprise

    Is 985 *far* from 1000? A 15-hour gap could be huge or tiny. It depends on how much x̄ normally wobbles.
    So we measure the gap **in units of standard error**. That ruler is the **z-score**:

    $$z = \frac{\bar x - \mu_0}{\sigma / \sqrt n} = \frac{985 - 1000}{50/\sqrt{36}} = \frac{-15}{8.33} = -1.80$$

    *"Our sample mean sits 1.8 standard errors below the claimed average."*

    **Drag the purple marker** below. The top axis is in **hours**, and the bottom ruler is in **z (standard errors)**.
    Notice that z sees the hours only through the SE: change *n* and the same 985 h lands on a different z.
    """)
    return


@app.cell
def zr_controls(mo):
    zr_n = mo.ui.slider(4, 144, step=4, value=36, label="sample size n", show_value=True)
    alpha = mo.ui.slider(0.005, 0.20, step=0.005, value=0.05, label="significance level α", show_value=True)
    tails = mo.ui.dropdown(["two-sided (μ ≠ 1000)", "left (μ < 1000)", "right (μ > 1000)"],
                           value="two-sided (μ ≠ 1000)", label="alternative H₁")
    mo.hstack([zr_n, alpha, tails], justify="start", gap=2)
    return alpha, tails, zr_n


@app.cell
def ZRuler_cls(anywidget, traitlets):
    class ZRuler(anywidget.AnyWidget):
        _esm = r"""
    function erf(x) { const s = Math.sign(x); x = Math.abs(x);
      const t = 1 / (1 + 0.3275911 * x);
      const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
      return s * y; }
    const Phi = z => 0.5 * (1 + erf(z / Math.SQRT2));
    function render({ model, el }) {
      const W = 680, H = 300, PAD = 40, ZMIN = -4, ZMAX = 4, BASE = 200, TOP = 40;
      const NS = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none", fontFamily: "sans-serif" });
      el.appendChild(svg);
      const mk = (tag, a, p = svg) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
      const txt = (s, a) => { const t = mk("text", Object.assign({ fill: "currentColor", "font-size": 12 }, a)); t.textContent = s; return t; };
      const toSvg = e => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const sz = z => PAD + (z - ZMIN) / (ZMAX - ZMIN) * (W - 2 * PAD);
      const invz = px => ZMIN + (px - PAD) / (W - 2 * PAD) * (ZMAX - ZMIN);
      const pdf = z => Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
      const sy = d => BASE - d / 0.4 * (BASE - TOP);
      function zcrit(a, tl) { // invert Phi by bisection
        const target = tl === "two" ? 1 - a / 2 : 1 - a; let lo = 0, hi = 6;
        for (let i = 0; i < 50; i++) { const m = (lo + hi) / 2; Phi(m) < target ? lo = m : hi = m; } return lo; }
      function area(z1, z2, color, op) {
        if (z2 <= z1) return; let d = `M ${sz(z1)} ${BASE}`;
        for (let z = z1; z <= z2; z += 0.02) d += ` L ${sz(z)} ${sy(pdf(z))}`;
        d += ` L ${sz(z2)} ${sy(pdf(z2))} L ${sz(z2)} ${BASE} Z`;
        mk("path", { d, fill: color, "fill-opacity": op });
      }
      let dragging = false;
      function draw() {
        svg.innerHTML = "";
        const mu0 = model.get("mu0"), se = model.get("se"), xbar = model.get("xbar"), a = model.get("alpha"), tl = model.get("tails");
        const z = (xbar - mu0) / se, zc = zcrit(a, tl);
        // p-value shading (purple)
        const zc_ = Math.min(Math.max(z, ZMIN), ZMAX);
        let p;
        if (tl === "two") { const az = Math.abs(z); area(ZMIN, -Math.min(az, 4), "#6c5b7b", 0.45); area(Math.min(az, 4), ZMAX, "#6c5b7b", 0.45); p = 2 * (1 - Phi(az)); }
        else if (tl === "left") { area(ZMIN, zc_, "#6c5b7b", 0.45); p = Phi(z); }
        else { area(zc_, ZMAX, "#6c5b7b", 0.45); p = 1 - Phi(z); }
        // null curve
        let d = ""; for (let zz = ZMIN; zz <= ZMAX + 1e-9; zz += 0.02) d += (d ? " L " : "M ") + sz(zz) + " " + sy(pdf(zz));
        mk("path", { d, fill: "none", stroke: "#4c78a8", "stroke-width": 3 });
        // critical lines (red)
        const crits = tl === "two" ? [-zc, zc] : tl === "left" ? [-zc] : [zc];
        crits.forEach(c => { mk("line", { x1: sz(c), x2: sz(c), y1: TOP - 10, y2: BASE, stroke: "#e4572e", "stroke-width": 2, "stroke-dasharray": "6 4" });
          txt((c > 0 ? "+" : "") + c.toFixed(2), { x: sz(c) - 16, y: TOP - 14, fill: "#e4572e" }); });
        // axes: hours (top of ruler) and z ruler
        mk("line", { x1: PAD, x2: W - PAD, y1: BASE, y2: BASE, stroke: "#888" });
        for (let k = ZMIN; k <= ZMAX; k++) {
          mk("line", { x1: sz(k), x2: sz(k), y1: BASE, y2: BASE + 6, stroke: "#888" });
          txt((mu0 + k * se).toFixed(0) + " h", { x: sz(k) - 18, y: BASE + 20 });
        }
        // z ruler
        const RY = 250;
        mk("rect", { x: PAD, y: RY - 12, width: W - 2 * PAD, height: 24, rx: 4, fill: "#4c78a8", "fill-opacity": 0.12, stroke: "#4c78a8" });
        for (let k = ZMIN; k <= ZMAX; k += 0.5) {
          mk("line", { x1: sz(k), x2: sz(k), y1: RY - 12, y2: RY - (Number.isInteger(k) ? 0 : 6), stroke: "#4c78a8" });
          if (Number.isInteger(k)) txt("z=" + k, { x: sz(k) - 12, y: RY + 26, fill: "#4c78a8" });
        }
        // marker
        const mx = sz(Math.max(ZMIN, Math.min(ZMAX, z)));
        mk("line", { x1: mx, x2: mx, y1: TOP, y2: RY + 12, stroke: "#6c5b7b", "stroke-width": 2 });
        const h = mk("circle", { cx: mx, cy: BASE, r: 11, fill: "#6c5b7b", stroke: "white", "stroke-width": 2 });
        h.style.cursor = "grab";
        h.addEventListener("pointerdown", e => { dragging = true; svg.setPointerCapture(e.pointerId); });
        const rej = p < a;
        txt(`x̄ = ${xbar.toFixed(1)} h  →  z = ${z.toFixed(2)}  →  p = ${p.toFixed(4)}  ${rej ? "< α: reject H₀" : "≥ α: fail to reject"}`,
            { x: PAD, y: 20, "font-size": 14, "font-weight": "bold", fill: rej ? "#e4572e" : "currentColor" });
      }
      svg.addEventListener("pointermove", e => { if (!dragging) return;
        const z = Math.max(ZMIN, Math.min(ZMAX, invz(toSvg(e).x)));
        model.set("xbar", Math.round((model.get("mu0") + z * model.get("se")) * 2) / 2); draw(); });
      const stop = () => { if (dragging) { dragging = false; model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointerleave", stop);
      ["xbar", "se", "alpha", "tails"].forEach(t => model.on("change:" + t, draw)); draw();
    }
    export default { render };
    """
        xbar = traitlets.Float(985.0).tag(sync=True)
        mu0 = traitlets.Float(1000.0).tag(sync=True)
        se = traitlets.Float(50 / 6).tag(sync=True)
        alpha = traitlets.Float(0.05).tag(sync=True)
        tails = traitlets.Unicode("two").tag(sync=True)

    return (ZRuler,)


@app.cell
def zruler(ZRuler, alpha, mo, np, tails, zr_n):
    tail_key = {"two-sided (μ ≠ 1000)": "two", "left (μ < 1000)": "left", "right (μ > 1000)": "right"}[tails.value]
    zruler = mo.ui.anywidget(ZRuler(se=50 / np.sqrt(zr_n.value), alpha=alpha.value, tails=tail_key))
    zruler
    return tail_key, zruler


@app.cell
def z_test(np, stats):
    def z_test(xbar, mu0=1000.0, sigma=50.0, n=36, tail="two"):
        """One-sample z-test. Returns (z, p)."""
        z = (xbar - mu0) / (sigma / np.sqrt(n))
        if tail == "two":
            p = 2 * stats.norm.sf(abs(z))
        elif tail == "left":
            p = stats.norm.cdf(z)
        else:
            p = stats.norm.sf(z)
        return z, p

    return (z_test,)


@app.cell(hide_code=True)
def z_readout(mo, np, tail_key, z_test, zr_n, zruler):
    _xb = zruler.value["xbar"]
    _se = 50 / np.sqrt(zr_n.value)
    obs_z, obs_p = z_test(_xb, n=zr_n.value, tail=tail_key)
    mo.md(rf"""
    $$z = \frac{{\bar x - \mu_0}}{{\sigma/\sqrt n}} = \frac{{{_xb:.1f} - 1000}}{{50/\sqrt{{{zr_n.value}}}}} = \frac{{{_xb - 1000:+.1f}}}{{{_se:.2f}}} = \mathbf{{{obs_z:+.2f}}}$$

    Your x̄ sits **{abs(obs_z):.2f} standard errors** {"below" if obs_z < 0 else "above"} the claimed mean.
    """)
    return obs_p, obs_z


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Leave the marker at **985** and slide **n** from 4 to 144. The gap in hours stays the same, but *z* grows like √n.
    2. Find the x̄ that sits exactly on the red line for n = 36 and α = 0.05. You should get about 1000 − 1.96 × 8.33 ≈ **983.7 h**.
    3. Switch H₁ to **left (μ < 1000)**. Now only *low* lifetimes count as evidence, and the single red line moves to −1.645.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · The p-value: how likely is data this extreme, if H₀ is true?

    Shade every z that is *at least as extreme* as yours, in the null world. That shaded area is the **p-value**:

    $$p = P\big(|Z| \ge |z_{\text{obs}}| \;\big|\; H_0\big) = 2\,\Phi(-|z_{\text{obs}}|) \quad\text{(two-sided)}$$

    For the bulbs, $p = 2\Phi(-1.80) \approx 0.072$: *"a 7.2% chance of seeing data this extreme if the null hypothesis is actually true."*
    The purple area in the ruler above *is* this number. (For pitfalls in reading p-values, see the companion `p_values.py` notebook.)

    **Predict first:** before you look, guess the p-value for z = −2.5 (two-sided).
    """)
    return


@app.cell
def p_guess(mo):
    p_guess = mo.ui.slider(0.0, 0.30, step=0.005, value=0.10, label="your guess for p at z = −2.5", show_value=True)
    p_lock = mo.ui.run_button(label="Lock in guess")
    mo.hstack([p_guess, p_lock], justify="start")
    return p_guess, p_lock


@app.cell
def p_reveal(mo, p_guess, p_lock, z_test):
    mo.stop(not p_lock.value, mo.md("_Make your guess, then lock it in._"))
    _p = z_test(1000 - 2.5 * 50 / 6, n=36, tail="two")[1]
    _err = abs(p_guess.value - _p)
    mo.md(f"True p = 2Φ(−2.5) = **{_p:.4f}**. Your guess was off by {_err:.3f} "
          f"{'🎯' if _err < 0.01 else ''}. The tails of a bell curve get thin *fast*: going from z = 1.8 to 2.5 shrinks p by a factor of about {0.0719 / _p:.0f}.")
    return


@app.cell
def p_readout(alpha, mo, obs_p, obs_z, pd, stats, tail_key):
    _rows = []
    for _z in [-1.0, -1.645, -1.80, -1.96, -2.5, -3.0]:
        _rows.append({"z": _z, "two-sided p": round(2 * stats.norm.sf(abs(_z)), 4), "one-sided (left) p": round(stats.norm.cdf(_z), 4)})
    mo.vstack([
        mo.hstack([
            mo.stat(f"{obs_z:+.2f}", label="your z (from the ruler)"),
            mo.stat(f"{obs_p:.4f}", label=f"p-value ({tail_key})"),
            mo.stat(f"{alpha.value:.3f}", label="α"),
            mo.stat("reject H₀" if obs_p < alpha.value else "fail to reject", label="verdict"),
        ], justify="start", gap=2),
        mo.md("**Reference table.** A one-sided p is half the two-sided p, because only one tail counts."),
        mo.ui.table(pd.DataFrame(_rows), selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Drag x̄ back to 985 (n = 36, two-sided). Check that the readout says **p ≈ 0.072**, the number from the video.
    2. Switch to **left (μ < 1000)**. The same data now gives p ≈ 0.036 < 0.05. The choice of H₁ has to be made **before** you see the data, or this becomes a way to cheat.
    3. Drag the marker *to the right* while the test is left-sided. p climbs toward 1, because high lifetimes are no evidence that bulbs fail *early*.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · The decision, and the two ways to be wrong

    We need *"a line in the sand"*: the **significance level α** (usually 0.05).

    $$\text{if } p < \alpha \Rightarrow \text{reject } H_0 \qquad\qquad \text{if } p \ge \alpha \Rightarrow \text{fail to reject } H_0$$

    For the bulbs, $0.072 \ge 0.05$, so we **fail to reject**. *"The data is a bit suspicious, but not suspicious enough."*

    But you can always be wrong:

    | | **H₀ true** (innocent) | **H₁ true** (guilty) |
    |---|---|---|
    | **Reject H₀** (convict) | ❌ **Type I error**, false positive, rate = **α** | ✅ correct, rate = **power** |
    | **Fail to reject** (acquit) | ✅ correct, rate = 1 − α | ❌ **Type II error**, false negative, rate = **β** |

    Below we run **2,000 independent bulb tests** in a world *you* choose, and count the verdicts. (This uses α and the sidedness from section 3.)
    """)
    return


@app.cell
def world_controls(mo):
    world_mu = mo.ui.slider(970, 1030, step=1, value=1000, label="true mean μ of the factory (1000 = H₀ is true)", show_value=True)
    world_n = mo.ui.slider(4, 144, step=4, value=36, label="bulbs per test n", show_value=True)
    reroll = mo.ui.button(label="🎲 re-run 2,000 trials", value=0, on_click=lambda v: v + 1)
    mo.vstack([mo.hstack([world_mu, world_n], justify="start", gap=2), reroll])
    return reroll, world_mu, world_n


@app.cell
def trials(
    alpha,
    alt,
    mo,
    np,
    pd,
    reroll,
    stats,
    tail_key,
    world_mu,
    world_n,
    z_test,
):
    _rng = np.random.default_rng(100 + reroll.value)
    _T = 2000
    _xbars = _rng.normal(world_mu.value, 50, size=(_T, world_n.value)).mean(axis=1)
    _, trial_p = z_test(_xbars, n=world_n.value, tail=tail_key)
    trial_reject = trial_p < alpha.value
    _h0 = world_mu.value == 1000
    _label = "Type I error (false positive)" if _h0 else "correct: power"
    _label2 = "correct (1 − α)" if _h0 else "Type II error (false negative)"
    _df = pd.DataFrame({
        "verdict": ["reject H₀", "fail to reject"],
        "count": [int(trial_reject.sum()), int((~trial_reject).sum())],
        "meaning": [_label, _label2],
    })
    _color = alt.Scale(domain=["Type I error (false positive)", "correct: power", "correct (1 − α)", "Type II error (false negative)"],
                       range=["#e4572e", "#2ca02c", "#4c78a8", "#f58518"])
    _bars = alt.Chart(_df).mark_bar().encode(
        y=alt.Y("verdict:N", title=None), x=alt.X("count:Q", scale=alt.Scale(domain=[0, _T])),
        color=alt.Color("meaning:N", scale=_color, legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["verdict", "count", "meaning"],
    ).properties(width="container", height=120, title=f"2,000 trials in a world where μ = {world_mu.value}")
    _hist = alt.Chart(pd.DataFrame({"p": trial_p})).mark_bar(color="#6c5b7b", opacity=0.7).encode(
        x=alt.X("p:Q", bin=alt.Bin(step=0.025), title="p-value of each trial"), y=alt.Y("count()", title="trials"),
    ).properties(width="container", height=160, title="Distribution of the 2,000 p-values")
    _rule = alt.Chart(pd.DataFrame({"a": [alpha.value]})).mark_rule(color="#e4572e", strokeWidth=2, strokeDash=[5, 3]).encode(x="a:Q")
    _rate = trial_reject.mean()
    _theory_reject = stats.norm.sf(stats.norm.isf(alpha.value / 2) - (world_mu.value - 1000) / (50 / np.sqrt(world_n.value))) + \
        stats.norm.cdf(-stats.norm.isf(alpha.value / 2) - (world_mu.value - 1000) / (50 / np.sqrt(world_n.value))) if tail_key == "two" else \
        (stats.norm.cdf(stats.norm.ppf(alpha.value) - (world_mu.value - 1000) / (50 / np.sqrt(world_n.value))) if tail_key == "left" else
         stats.norm.sf(stats.norm.isf(alpha.value) - (world_mu.value - 1000) / (50 / np.sqrt(world_n.value))))
    mo.vstack([
        mo.hstack([
            mo.stat(f"{_rate:.3f}", label="share rejected (simulated)",
                    caption=("should be ≈ α = " if _h0 else "theoretical power = ") + f"{_theory_reject:.3f}"),
            mo.stat("Type I rate" if _h0 else "power", label="what that share means"),
            mo.stat(f"{1 - _rate:.3f}", label="share not rejected", caption="1 − α" if _h0 else "β (Type II rate)"),
        ], justify="start", gap=2),
        _bars, _hist + _rule,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Keep **μ = 1000** (H₀ true). About α × 2000 ≈ 100 tests still reject, purely by chance. *"Alpha controls the Type I error rate: the probability of convicting an innocent null."* The p-values are spread **uniformly** from 0 to 1.
    2. Set **μ = 985**, the video's sample mean, as if it were the truth. With n = 36, fewer than half of the tests catch it. Most guilty factories walk free. That is a high **Type II** rate.
    3. Keep μ = 985 and raise **n** to 144. Power climbs toward 1. Then lower α to 0.01 in section 3. Fewer false alarms, but more misses.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · The whole picture: two overlapping worlds

    Put the **null distribution** of z (blue) next to a world where **H₁ is true** (orange), which is shifted by

    $$\delta = \frac{\mu_1 - \mu_0}{\sigma/\sqrt n}.$$

    - **Red tails** under blue: the rejection region, with total area **α**, beyond **±1.96** for α = 0.05.
    - **Green** under orange: the part of the alternative that lands in the rejection region. That is the **power**, $1-\beta$.
    - **Faded orange** under orange: the alternative falls in the "fail to reject" zone. That is **β**, the Type II error rate.

    *"The overlap between these two curves is where mistakes happen."* **Drag the orange curve** sideways.
    """)
    return


@app.cell
def pw_alpha(mo):
    pw_alpha = mo.ui.slider(0.005, 0.20, step=0.005, value=0.05, label="α", show_value=True)
    pw_alpha
    return (pw_alpha,)


@app.cell
def TwoWorlds_cls(anywidget, traitlets):
    class TwoWorlds(anywidget.AnyWidget):
        _esm = r"""
    function erf(x) { const s = Math.sign(x); x = Math.abs(x);
      const t = 1 / (1 + 0.3275911 * x);
      const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
      return s * y; }
    const Phi = z => 0.5 * (1 + erf(z / Math.SQRT2));
    function render({ model, el }) {
      const W = 680, H = 270, PAD = 30, ZMIN = -5, ZMAX = 7, BASE = 220, TOP = 45;
      const NS = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none", fontFamily: "sans-serif" });
      el.appendChild(svg);
      const mk = (tag, a, p = svg) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
      const txt = (s, a) => { const t = mk("text", Object.assign({ fill: "currentColor", "font-size": 12 }, a)); t.textContent = s; return t; };
      const toSvg = e => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const sz = z => PAD + (z - ZMIN) / (ZMAX - ZMIN) * (W - 2 * PAD);
      const invz = px => ZMIN + (px - PAD) / (W - 2 * PAD) * (ZMAX - ZMIN);
      const pdf = (z, m) => Math.exp(-0.5 * (z - m) ** 2) / Math.sqrt(2 * Math.PI);
      const sy = d => BASE - d / 0.4 * (BASE - TOP);
      const zcrit = a => { let lo = 0, hi = 6; for (let i = 0; i < 50; i++) { const m = (lo + hi) / 2; Phi(m) < 1 - a / 2 ? lo = m : hi = m; } return lo; };
      function area(z1, z2, m, color, op) {
        z1 = Math.max(z1, ZMIN); z2 = Math.min(z2, ZMAX); if (z2 <= z1) return;
        let d = `M ${sz(z1)} ${BASE}`;
        for (let z = z1; z <= z2; z += 0.02) d += ` L ${sz(z)} ${sy(pdf(z, m))}`;
        d += ` L ${sz(z2)} ${sy(pdf(z2, m))} L ${sz(z2)} ${BASE} Z`;
        mk("path", { d, fill: color, "fill-opacity": op });
      }
      const curve = (m, color, w) => { let d = ""; for (let z = ZMIN; z <= ZMAX + 1e-9; z += 0.02) d += (d ? " L " : "M ") + sz(z) + " " + sy(pdf(z, m));
        return mk("path", { d, fill: "none", stroke: color, "stroke-width": w }); };
      let dragging = false, grabOff = 0;
      function draw() {
        svg.innerHTML = "";
        const d = model.get("delta"), a = model.get("alpha"), zc = zcrit(a);
        // beta (faded orange) & power (green) under alternative
        area(-zc, zc, d, "#f58518", 0.18);
        area(zc, ZMAX, d, "#2ca02c", 0.45); area(ZMIN, -zc, d, "#2ca02c", 0.45);
        // alpha tails under null (red)
        area(ZMIN, -zc, 0, "#e4572e", 0.75); area(zc, ZMAX, 0, "#e4572e", 0.75);
        curve(0, "#4c78a8", 3);
        const alt = curve(d, "#f58518", 3);
        // fat invisible hit area for dragging the alt curve
        const hit = mk("rect", { x: sz(d - 1.5), y: TOP - 10, width: sz(1.5) - sz(-1.5), height: BASE - TOP + 10, fill: "transparent" });
        hit.style.cursor = "grab";
        hit.addEventListener("pointerdown", e => { dragging = true; grabOff = invz(toSvg(e).x) - model.get("delta"); svg.setPointerCapture(e.pointerId); });
        mk("line", { x1: PAD, x2: W - PAD, y1: BASE, y2: BASE, stroke: "#888" });
        for (let k = ZMIN; k <= ZMAX; k++) { mk("line", { x1: sz(k), x2: sz(k), y1: BASE, y2: BASE + 5, stroke: "#888" }); txt(k, { x: sz(k) - 4, y: BASE + 18 }); }
        [-zc, zc].forEach(c => mk("line", { x1: sz(c), x2: sz(c), y1: TOP - 5, y2: BASE, stroke: "#e4572e", "stroke-dasharray": "5 4", "stroke-width": 1.5 }));
        txt("H₀", { x: sz(0) - 8, y: TOP - 8, fill: "#4c78a8", "font-weight": "bold" });
        txt("H₁ (drag me)", { x: sz(d) - 34, y: TOP - 8, fill: "#f58518", "font-weight": "bold" });
        const power = (1 - Phi(zc - d)) + Phi(-zc - d);
        txt(`δ = ${d.toFixed(2)}   α = ${a.toFixed(3)}   power = ${power.toFixed(3)}   β = ${(1 - power).toFixed(3)}`,
            { x: PAD, y: 18, "font-size": 14, "font-weight": "bold" });
        txt("z", { x: W - PAD + 6, y: BASE + 4 });
      }
      svg.addEventListener("pointermove", e => { if (!dragging) return;
        const v = Math.max(-4, Math.min(6, invz(toSvg(e).x) - grabOff)); model.set("delta", Math.round(v * 20) / 20); draw(); });
      const stop = () => { if (dragging) { dragging = false; model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointerleave", stop);
      model.on("change:delta", draw); model.on("change:alpha", draw); draw();
    }
    export default { render };
    """
        delta = traitlets.Float(-1.8).tag(sync=True)
        alpha = traitlets.Float(0.05).tag(sync=True)

    return (TwoWorlds,)


@app.cell
def worlds(TwoWorlds, mo, pw_alpha):
    worlds = mo.ui.anywidget(TwoWorlds(alpha=pw_alpha.value))
    worlds
    return (worlds,)


@app.cell
def z_power(stats):
    def z_power(delta, alpha=0.05):
        """Power of a two-sided z-test when the alternative is shifted by delta standard errors."""
        zc = stats.norm.isf(alpha / 2)
        return stats.norm.sf(zc - delta) + stats.norm.cdf(-zc - delta)

    return (z_power,)


@app.cell
def power_readout(alt, mo, np, pd, pw_alpha, worlds, z_power):
    _d = worlds.value["delta"]
    _pow = z_power(_d, pw_alpha.value)
    _mu1 = 1000 + _d * 50 / 6
    _ns = np.arange(4, 401, 4)
    _curve = pd.DataFrame({"n": _ns, "power": z_power((_mu1 - 1000) / (50 / np.sqrt(_ns)), pw_alpha.value)})
    _line = alt.Chart(_curve).mark_line(color="#2ca02c", strokeWidth=3).encode(
        x=alt.X("n:Q", title="bulbs per test, n"), y=alt.Y("power:Q", scale=alt.Scale(domain=[0, 1])))
    _pt = alt.Chart(pd.DataFrame({"n": [36], "power": [_pow]})).mark_point(color="#6c5b7b", size=120, filled=True).encode(x="n:Q", y="power:Q")
    _ref = alt.Chart(pd.DataFrame({"y": [0.8]})).mark_rule(strokeDash=[4, 4], color="#888").encode(y="y:Q")
    _need = next((int(n) for n, p in zip(_curve["n"], _curve["power"]) if p >= 0.8), None)
    mo.vstack([
        mo.hstack([
            mo.stat(f"{_d:+.2f}", label="shift δ (in SE)"),
            mo.stat(f"{_mu1:.1f} h", label="true μ₁ at n = 36", caption="μ₁ = 1000 + δ·σ/√n"),
            mo.stat(f"{_pow:.3f}", label="power 1 − β"),
            mo.stat(f"{1 - _pow:.3f}", label="β (Type II)"),
            mo.stat(f"n ≈ {_need}" if _need else "> 400", label="n needed for 80% power", caption=f"for this μ₁, α = {pw_alpha.value}"),
        ], justify="start", gap=2),
        (_line + _ref + _pt).properties(width="container", height=220,
            title=f"Power vs sample size if the true mean is {_mu1:.1f} h (dot = n = 36)"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Drag H₁ right on top of H₀ (δ = 0). Power drops to α: when the worlds coincide, "rejecting" is just a false alarm.
    2. Drag to **δ ≈ 2.8**. Power ≈ 0.80, the usual design target. *"The further apart these two curves sit, the easier it is to tell them apart."*
    3. Lower **α** to 0.01. The red tails shrink, and so does the green area. Fewer Type I errors cost you Type II errors. Look at the power-vs-n chart to see how many more bulbs it takes to win the power back.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Sandbox: run your own test, start to finish

    Make up any factory, then draw a fresh sample of bulbs and walk through the five steps from the video:
    **hypotheses → null world → test statistic → p-value → decision**. If you set the true μ yourself,
    you also know whether the verdict was *right*, which is something a real analyst never gets to know.
    """)
    return


@app.cell
def sandbox_controls(mo):
    sb = mo.ui.dictionary({
        "mu0": mo.ui.number(start=0, stop=5000, step=10, value=1000, label="claimed μ₀ (H₀)"),
        "sigma": mo.ui.number(start=1, stop=500, step=1, value=50, label="known σ"),
        "n": mo.ui.slider(2, 200, value=36, label="sample size n", show_value=True),
        "mu_true": mo.ui.number(start=0, stop=5000, step=1, value=985, label="TRUE μ (hidden from the analyst)"),
        "alpha": mo.ui.dropdown(["0.10", "0.05", "0.01", "0.001"], value="0.05", label="α"),
        "tail": mo.ui.dropdown(["two", "left", "right"], value="two", label="H₁ direction"),
    })
    sb_roll = mo.ui.button(label="🎲 test a new batch", value=0, on_click=lambda v: v + 1)
    mo.vstack([mo.hstack([sb["mu0"], sb["sigma"], sb["n"]], justify="start"),
               mo.hstack([sb["mu_true"], sb["alpha"], sb["tail"]], justify="start"), sb_roll])
    return sb, sb_roll


@app.cell
def sandbox_report(alt, mo, np, pd, sb, sb_roll, stats, z_test):
    _v = sb.value
    _mu0, _s, _n, _a = float(_v["mu0"]), float(_v["sigma"]), int(_v["n"]), float(_v["alpha"])
    _rng = np.random.default_rng(7 + sb_roll.value)
    _sample = _rng.normal(_v["mu_true"], _s, _n)
    _xb = _sample.mean()
    _se = _s / np.sqrt(_n)
    _z, _p = z_test(_xb, mu0=_mu0, sigma=_s, n=_n, tail=_v["tail"])
    _rej = _p < _a
    _truth_h0 = float(_v["mu_true"]) == _mu0
    _outcome = ("❌ Type I error (false positive)" if _rej else "✅ correct: H₀ is true and was not rejected") if _truth_h0 else \
               ("✅ correct rejection (power did its job)" if _rej else "❌ Type II error (false negative)")
    _sym = {"two": "\\neq", "left": "<", "right": ">"}[_v["tail"]]
    _xs = np.linspace(_mu0 - 4.5 * _se, _mu0 + 4.5 * _se, 300)
    _zc = stats.norm.isf(_a / 2) if _v["tail"] == "two" else stats.norm.isf(_a)
    _lo, _hi = (_mu0 - _zc * _se, _mu0 + _zc * _se) if _v["tail"] == "two" else ((_mu0 - _zc * _se, np.inf) if _v["tail"] == "left" else (-np.inf, _mu0 + _zc * _se))
    _dens = pd.DataFrame({"x̄": _xs, "density": stats.norm.pdf(_xs, _mu0, _se)})
    _dens["region"] = np.where((_dens["x̄"] < _lo) | (_dens["x̄"] > _hi), "rejection region", "fail to reject")
    _area = alt.Chart(_dens).mark_area(opacity=0.6).encode(
        x=alt.X("x̄:Q", scale=alt.Scale(domain=[_xs[0], _xs[-1]], clamp=True)), y="density:Q",
        color=alt.Color("region:N", scale=alt.Scale(domain=["rejection region", "fail to reject"], range=["#e4572e", "#4c78a8"]),
                        legend=alt.Legend(orient="bottom", title=None)))
    _obs = alt.Chart(pd.DataFrame({"x": [np.clip(_xb, _xs[0], _xs[-1])]})).mark_rule(color="#6c5b7b", strokeWidth=3).encode(x="x:Q")
    mo.vstack([
        mo.md(rf"""
    1. **Hypotheses:** $H_0: \mu = {_mu0:g}$ vs $H_1: \mu {_sym} {_mu0:g}$
    2. **Null world:** $\bar X \sim \mathcal N({_mu0:g},\ {_se:.2f})$, where SE $= {_s:g}/\sqrt{{{_n}}}$
    3. **Test statistic:** $\bar x = {_xb:.2f}$, so $z = ({_xb:.2f} - {_mu0:g})/{_se:.2f} = {_z:+.2f}$
    4. **p-value:** $p = {_p:.4f}$
    5. **Decision at α = {_a}:** **{"REJECT H₀" if _rej else "FAIL TO REJECT H₀"}**. Reality check: {_outcome}
    """),
        (_area + _obs).properties(width="container", height=200, title="Null world of x̄ (purple = your sample mean)"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Challenges**
    - Set true μ = μ₀ and click 🎲 twenty times. How many false alarms do you get at α = 0.05? (You should expect about 1.)
    - With true μ = 990 and σ = 50, find the smallest n where most of your rolls reject. Compare it with the power chart in section 6.
    - Set σ = 5. Now even a 1-hour difference is detected easily. A tiny SE makes *trivial* gaps "significant".
    """), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Quiz: put the framework to work

    Answer all six, then flip the reveal switch.
    """)
    return


@app.cell
def quiz(mo):
    quiz = mo.ui.dictionary({
        "q1": mo.ui.dropdown(["μ = 1000", "μ ≠ 1000", "x̄ = 985", "x̄ ≠ 1000"], label="1. In the bulb example, H₀ is"),
        "q2": mo.ui.dropdown(["-0.30", "-1.80", "-15.0", "-2.50"], label="2. x̄ = 985, μ₀ = 1000, σ = 50, n = 36. What is z?"),
        "q3": mo.ui.dropdown(["Reject H₀", "Fail to reject H₀", "Accept H₀"], label="3. p = 0.072 and α = 0.05. The decision is"),
        "q4": mo.ui.dropdown(["Type I error", "Type II error", "Correct decision"], label="4. A factory truly lies (μ = 980), but your test fails to reject. That is a"),
        "q5": mo.ui.dropdown(["Increase it", "Decrease it", "No change"], label="5. You quadruple n (μ₁ fixed, α fixed). What happens to power?"),
        "q6": mo.ui.dropdown(["≈ 0.05", "≈ 0.50", "≈ 0.95", "It depends on n"], label="6. H₀ is true and α = 0.05. What share of tests will (wrongly) reject?"),
    })
    quiz_reveal = mo.ui.switch(label="Reveal answers")
    mo.vstack([*quiz.elements.values(), quiz_reveal])
    return quiz, quiz_reveal


@app.cell
def quiz_check(mo, quiz, quiz_reveal, z_test):
    _z36 = z_test(985, n=36)[0]
    _key = {
        "q1": ("μ = 1000", "H₀ is the factory's claim about the *population* mean. It is never about the sample x̄."),
        "q2": (f"{_z36:.2f}", "(985 − 1000) / (50/√36) = −15 / 8.33 = −1.80"),
        "q3": ("Fail to reject H₀", "p ≥ α, so we fail to reject. We never 'accept' H₀."),
        "q4": ("Type II error", "H₁ was true, but the test missed it. That is a false negative, with rate β."),
        "q5": ("Increase it", "SE halves, so δ doubles and the orange curve moves away from the blue one."),
        "q6": ("≈ 0.05", "α *is* the Type I error rate, whatever n is (see section 5)."),
    }
    _ans = quiz.value
    _score = sum(_ans[k] == v[0] for k, v in _key.items())
    if not quiz_reveal.value:
        _out = mo.md(f"_Answered {sum(a is not None for a in _ans.values())}/6. Flip the switch when you're ready._")
    else:
        _rows = "\n".join(f"| {k[1:]} | {_ans[k] or '—'} | {v[0]} | {'✅' if _ans[k] == v[0] else '❌'} | {v[1]} |" for k, v in _key.items())
        _out = mo.vstack([
            mo.callout(mo.md(f"### Score: {_score} / 6"), kind="success" if _score >= 5 else "warn"),
            mo.md("| # | yours | correct | | why |\n|---|---|---|---|---|\n" + _rows),
        ])
    _out
    return


@app.cell(hide_code=True)
def recap(mo):
    mo.md(r"""
    ## Recap

    | Step (video chapter) | Idea | Formula / rule | Where you touched it |
    |---|---|---|---|
    | The courtroom | Presume H₀ ("innocent"), require evidence to reject | H₀: μ = μ₀, H₁: μ ≠ μ₀ | §1 verdict check |
    | The null world | The distribution of the statistic if H₀ is true | $\bar X \sim \mathcal N(\mu_0, \sigma/\sqrt n)$ (CLT) | §2 bulb-factory simulator |
    | The test statistic | A ruler for surprise, in standard errors | $z = (\bar x-\mu_0)/(\sigma/\sqrt n)$ | §3 draggable z-ruler |
    | The p-value | Chance of data at least this extreme under H₀ | $p = 2\Phi(-\lvert z\rvert)$ (two-sided) | §4 shaded tails, predict-then-reveal |
    | The decision | Compare with α; never "accept" H₀ | reject if $p<\alpha$ | §5 2,000 trials |
    | Errors | Type I = false positive (rate α), Type II = false negative (rate β) | | §5 verdict counts |
    | The whole picture | Power = rejection-region area under H₁ | $1-\beta = \Phi(\delta - z_{\alpha/2}) + \Phi(-\delta - z_{\alpha/2})$ | §6 draggable alternative |

    **Bulb example, final numbers:** SE = 8.33 h, z = −1.80, p ≈ 0.072 > 0.05, so we **fail to reject**. If the true mean really were 985 h, a 36-bulb test would catch it only about 44% of the time.

    *Next:* `p_values.py` covers what a p-value is **not**, and how p-hacking breaks the α guarantee you saw in section 5.
    """)
    return


if __name__ == "__main__":
    app.run()

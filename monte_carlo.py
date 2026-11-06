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
        mo.md(r"""
    # Monte Carlo simulation: a time machine made of random numbers
    *An interactive companion to IBM Technology's **"What is Monte Carlo Simulation?"** (Martin Keen).*
    """),
        mo.Html('<iframe width="560" height="315" src="https://www.youtube.com/embed/7TqhmX92P6U" title="What is Monte Carlo Simulation?" frameborder="0" allowfullscreen></iframe>'),
        mo.md(r"""
    > *"Monte Carlo simulation is a mathematical technique used to estimate the possible outcomes of an uncertain event.
    > It's a chance to see into the future."*

    The video answers three questions, and so does this notebook:

    1. **How do they work?** Random sampling, illustrated with **two dice**.
    2. **Who uses them?** Portfolio management, risk analysis, option pricing, capacity planning.
    3. **How do you run one?** Three steps: **model → distributions → simulate repeatedly**, then measure the **spread**.

    **Color language** used throughout:
    <span style="color:#6c5b7b">■ purple = simulated samples / estimates</span> ·
    <span style="color:#e4572e">■ red = the exact (true) answer</span> ·
    <span style="color:#f18f01">■ orange = risk, thresholds and "bad" outcomes</span> ·
    <span style="color:#2a9d8f">■ teal = "good" outcomes / targets met</span>.
    """)
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · How do they work? Random sampling with two dice

    The video's example: *what is the probability that two dice sum to **7**?*

    **Brute force by reasoning:** there are $6 \times 6 = 36$ equally likely combinations, and 6 of them sum to 7, so

    $$P(\text{sum}=7) = \frac{6}{36} = \frac{1}{6} \approx 0.1667.$$

    **Monte Carlo:** forget the counting. Just *roll* (randomly sample) many times and count:

    $$\hat P(\text{sum}=7) = \frac{\#\{\text{rolls that sum to 7}\}}{N}.$$

    The simulator below rolls the dice in your browser. The purple bars are the observed frequencies,
    the red ticks are the exact probabilities $\tfrac{6-|s-7|}{36}$.
    """)
    return


@app.cell
def dice_class(anywidget, traitlets):
    class DiceRoller(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 640, H = 300, PADL = 40, PADB = 30, TOP = 70;
      let timer = null;
      let counts = model.get("counts").slice();
      let last = [1, 1];
      const ctl = document.createElement("div");
      ctl.style.cssText = "display:flex;gap:6px;align-items:center;margin-bottom:6px;flex-wrap:wrap";
      el.appendChild(ctl);
      const btn = (label, fn) => { const b = document.createElement("button"); b.textContent = label; b.onclick = fn;
        b.style.cssText = "padding:4px 10px;border-radius:6px;border:1px solid #aaa;cursor:pointer"; ctl.appendChild(b); return b; };
      const NS = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none" });
      el.appendChild(svg);
      const mk = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); svg.appendChild(e); return e; };
      const pips = { 1: [[0,0]], 2: [[-1,-1],[1,1]], 3: [[-1,-1],[0,0],[1,1]], 4: [[-1,-1],[1,-1],[-1,1],[1,1]],
                     5: [[-1,-1],[1,-1],[0,0],[-1,1],[1,1]], 6: [[-1,-1],[1,-1],[-1,0],[1,0],[-1,1],[1,1]] };
      function die(cx, cy, v) {
        mk("rect", { x: cx - 24, y: cy - 24, width: 48, height: 48, rx: 8, fill: "#e9e4f0", stroke: "#6c5b7b", "stroke-width": 2 });
        pips[v].forEach(([a, b]) => mk("circle", { cx: cx + a * 13, cy: cy + b * 13, r: 4.5, fill: "#6c5b7b" }));
      }
      function draw() {
        svg.innerHTML = "";
        const n = counts.reduce((a, b) => a + b, 0);
        die(50, 34, last[0]); die(110, 34, last[1]);
        const t = mk("text", { x: 150, y: 30, "font-size": 15, fill: "currentColor" });
        t.textContent = n ? `last sum = ${last[0] + last[1]}   ·   rolls N = ${n}` : "press ▶ roll";
        const t2 = mk("text", { x: 150, y: 52, "font-size": 15, fill: "#6c5b7b", "font-weight": "bold" });
        t2.textContent = n ? `P̂(7) = ${(counts[5] / n).toFixed(4)}   (exact 0.1667)` : "";
        const plotH = H - TOP - PADB, bw = (W - PADL - 10) / 11, ymax = 0.25;
        const sy = p => TOP + plotH - Math.min(p, ymax) / ymax * plotH;
        [0, 0.05, 0.1, 0.15, 0.2, 0.25].forEach(p => {
          mk("line", { x1: PADL, x2: W - 10, y1: sy(p), y2: sy(p), stroke: "#8884", "stroke-width": 1 });
          const l = mk("text", { x: PADL - 4, y: sy(p) + 4, "font-size": 10, "text-anchor": "end", fill: "currentColor" }); l.textContent = p.toFixed(2);
        });
        for (let k = 0; k < 11; k++) {
          const s = k + 2, x = PADL + k * bw, f = n ? counts[k] / n : 0, exact = (6 - Math.abs(s - 7)) / 36;
          mk("rect", { x: x + 4, y: sy(f), width: bw - 8, height: TOP + plotH - sy(f), fill: s === 7 ? "#6c5b7b" : "#6c5b7baa", rx: 3 });
          mk("line", { x1: x + 1, x2: x + bw - 1, y1: sy(exact), y2: sy(exact), stroke: "#e4572e", "stroke-width": 3 });
          const l = mk("text", { x: x + bw / 2, y: H - 10, "font-size": 12, "text-anchor": "middle", fill: "currentColor" }); l.textContent = s;
        }
      }
      function roll(k) {
        for (let i = 0; i < k; i++) {
          const a = 1 + Math.floor(Math.random() * 6), b = 1 + Math.floor(Math.random() * 6);
          counts[a + b - 2]++; last = [a, b];
        }
        draw();
      }
      const sync = () => { model.set("counts", counts.slice()); model.save_changes(); };
      let speed = 1;
      const play = btn("▶ roll", () => {
        if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ roll"; sync(); }
        else { timer = setInterval(() => { roll(speed); }, 60); play.textContent = "⏸ pause"; }
      });
      btn("+1", () => { roll(1); sync(); });
      btn("+100", () => { roll(100); sync(); });
      btn("+10,000", () => { roll(10000); sync(); });
      const fast = btn("⏩ speed ×1", () => { speed = speed >= 100 ? 1 : speed * 10; fast.textContent = `⏩ speed ×${speed}`; });
      btn("↺ reset", () => { counts = Array(11).fill(0); last = [1, 1]; draw(); sync(); });
      const syncTimer = setInterval(() => { if (timer) sync(); }, 1500);
      model.on("change:counts", () => { if (!timer) { counts = model.get("counts").slice(); draw(); } });
      draw();
      return () => { clearInterval(timer); clearInterval(syncTimer); };
    }
    export default { render };
    """
        counts = traitlets.List(traitlets.Int(), default_value=[0] * 11).tag(sync=True)

    return (DiceRoller,)


@app.cell
def dice(DiceRoller, mo):
    dice = mo.ui.anywidget(DiceRoller())
    dice
    return (dice,)


@app.cell
def dice_readout(dice, mo, np):
    _counts = np.array(dice.value["counts"])
    _n = int(_counts.sum())
    _exact = 6 / 36
    _phat = _counts[5] / _n if _n else float("nan")
    _se = np.sqrt(_exact * (1 - _exact) / _n) if _n else float("nan")
    mo.hstack([
        mo.stat(f"{_n:,}", label="rolls N"),
        mo.stat(f"{_phat:.4f}" if _n else "–", label="Monte Carlo P̂(7)"),
        mo.stat(f"{_exact:.4f}", label="exact P(7) = 1/6"),
        mo.stat(f"{abs(_phat - _exact):.4f}" if _n else "–", label="absolute error"),
        mo.stat(f"±{1.96 * _se:.4f}" if _n else "–", label="95% sampling range"),
    ], justify="space-around")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Press **+1** ten times. How far is P̂(7) from 1/6? With 10 rolls it can easily be 0 or 0.4.
    2. Press **+100** a few times, then **+10,000**. Watch the purple bars snap onto the red ticks.
    3. Hit **↺ reset** and repeat. The *answer* changes every run, but its **error shrinks** as N grows.
       That shrinking "95 % sampling range" is the topic of Section 2.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · "The more you sample, the more accurate": the $1/\sqrt{N}$ law

    The video ends with a key rule: *"the more you sample, the more accurate your sampling range, and the better your estimation."*
    How much more accurate? An estimate built from $N$ independent samples has standard deviation (the **standard error**)

    $$\operatorname{SE}(\hat P) = \sqrt{\frac{P(1-P)}{N}} \;\propto\; \frac{1}{\sqrt N}.$$

    So **4× more samples only halves the error**, and 100× more samples buys one extra decimal digit.

    **Predict first:** how many rolls do you need so that the 95 % range of $\hat P(7)$ is within **±0.01** of 1/6?
    """)
    return


@app.cell
def n_guess(mo):
    n_guess = mo.ui.dropdown(["~100", "~500", "~1,000", "~5,000", "~50,000"], label="My guess for N")
    n_lock = mo.ui.run_button(label="Lock in guess")
    mo.hstack([n_guess, n_lock], justify="start")
    return n_guess, n_lock


@app.cell(hide_code=True)
def _(mo, n_guess, n_lock):
    mo.stop(not n_lock.value or n_guess.value is None, mo.md("_Pick a guess and lock it in to see the answer._"))
    _p = 1 / 6
    _need = 1.96**2 * _p * (1 - _p) / 0.01**2
    _ok = n_guess.value == "~5,000"
    mo.callout(mo.md(
        f"Solve $1.96\\sqrt{{P(1-P)/N}} = 0.01$: $N = 1.96^2 \\cdot \\tfrac16\\cdot\\tfrac56 / 0.01^2 \\approx$ **{_need:,.0f} rolls**. "
        + ("🎯 Spot on." if _ok else f"You guessed {n_guess.value}. Most people underestimate this: precision is expensive.")
    ), kind="success" if _ok else "warn")
    return


@app.cell
def conv_controls(mo):
    conv_reps = mo.ui.slider(10, 400, step=10, value=100, label="How many independent Monte Carlo runs (replications)", show_value=True)
    conv_target = mo.ui.dropdown({"P(sum = 7) = 1/6": 7, "P(sum = 2) = 1/36": 2, "P(sum ≥ 10) = 1/6": 10}, value="P(sum = 7) = 1/6", label="Quantity to estimate")
    conv_reroll = mo.ui.button(label="🎲 re-roll everything", value=0, on_click=lambda v: v + 1)
    mo.hstack([conv_reps, conv_target, conv_reroll], justify="start")
    return conv_reps, conv_reroll, conv_target


@app.cell
def conv_chart(alt, conv_reps, conv_reroll, conv_target, np, pd):
    def mc_dice_estimates(target, n_max, reps, seed):
        """Running Monte Carlo estimates for `reps` independent runs of `n_max` two-dice rolls each."""
        _rng = np.random.default_rng(seed)
        _s = _rng.integers(1, 7, (reps, n_max)) + _rng.integers(1, 7, (reps, n_max))
        _hit = (_s >= 10) if target == 10 else (_s == target)
        return np.cumsum(_hit, axis=1) / np.arange(1, n_max + 1)

    conv_truth = {7: 6 / 36, 2: 1 / 36, 10: 6 / 36}[conv_target.value]
    _N_MAX = 10_000
    _runs = mc_dice_estimates(conv_target.value, _N_MAX, conv_reps.value, seed=conv_reroll.value)
    _grid = np.unique(np.logspace(0, 4, 60).astype(int))
    _est = _runs[:, _grid - 1]
    _lines = pd.DataFrame({
        "N": np.tile(_grid, min(8, conv_reps.value)),
        "estimate": _est[:8].ravel(),
        "run": np.repeat(np.arange(min(8, conv_reps.value)), len(_grid)),
    })
    _band = pd.DataFrame({
        "N": _grid,
        "lo": np.percentile(_est, 2.5, axis=0),
        "hi": np.percentile(_est, 97.5, axis=0),
        "theory_lo": conv_truth - 1.96 * np.sqrt(conv_truth * (1 - conv_truth) / _grid),
        "theory_hi": conv_truth + 1.96 * np.sqrt(conv_truth * (1 - conv_truth) / _grid),
    })
    _x = alt.X("N:Q", scale=alt.Scale(type="log"), title="number of rolls N (log scale)")
    _area = alt.Chart(_band).mark_area(opacity=0.25, color="#6c5b7b").encode(_x, y="lo:Q", y2="hi:Q")
    _th = alt.Chart(_band).transform_fold(["theory_lo", "theory_hi"]).mark_line(strokeDash=[4, 3], color="#e4572e", opacity=0.7).encode(_x, y="value:Q", detail="key:N")
    _ln = alt.Chart(_lines).mark_line(color="#6c5b7b", opacity=0.55, strokeWidth=1).encode(_x, y=alt.Y("estimate:Q", title="estimate P̂", scale=alt.Scale(domain=[0, min(1, conv_truth * 3.5)], clamp=True)), detail="run:N")
    _truth = alt.Chart(pd.DataFrame({"t": [conv_truth]})).mark_rule(color="#e4572e", strokeWidth=2).encode(y="t:Q")
    conv_chart = (_area + _th + _ln + _truth).properties(width="container", height=320, title="8 example runs (lines), 95% of all runs (shaded), theory ±1.96·SE (dashed red)")
    conv_runs = _runs
    conv_chart
    return conv_runs, conv_truth


@app.cell
def conv_table(conv_reps, conv_runs, conv_truth, mo, np, pd):
    _rows = []
    for _N in [10, 100, 1_000, 10_000]:
        _e = conv_runs[:, _N - 1]
        _rows.append({"N": _N, "mean of estimates": round(_e.mean(), 4),
                      "observed std (spread)": round(_e.std(ddof=1), 4),
                      "theory √(P(1−P)/N)": round(np.sqrt(conv_truth * (1 - conv_truth) / _N), 4)})
    mo.vstack([mo.md(f"**Spread of {conv_reps.value} independent estimates at selected N** (truth = {conv_truth:.4f})"), mo.ui.table(pd.DataFrame(_rows), selection=None)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Read the table down the "spread" column: each 10× more rolls shrinks the std by about $\sqrt{10}\approx 3.2$.
    2. Switch to **P(sum = 2) = 1/36**, a *rare* event. At N = 10 most runs say 0! Rare events need many more samples.
    3. Press **🎲 re-roll** a few times: individual runs wander, but the shaded band always hugs the red dashed theory.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Who uses them? A portfolio you can't predict

    *"Perhaps the most well-known application is portfolio management and investment planning. By running thousands or
    even millions of simulations, investors get a better idea of how their portfolio might perform under different market conditions."*

    We'll build exactly that, following the video's **three steps to run a Monte Carlo simulation**:

    | Step (video) | In our portfolio |
    |---|---|
    | **1. Set up the predictive model**: dependent variable ← independent (input/risk) variables | final wealth $V_T$ ← yearly returns $r_1,\dots,r_T$: $\;V_t = V_{t-1}(1+r_t) + c$ |
    | **2. Specify the probability distribution** of the inputs (historical data or analyst judgment) | *you paint* the distribution of a yearly return $r$ |
    | **3. Run simulations repeatedly**, drawing random inputs each time | draw $T$ returns, compute $V_T$, repeat thousands of times |

    ### Step 2 · Paint the distribution of yearly returns
    Drag across the bars to paint "probability weights" (the video: *assign probability weights to a range of likely values*).
    Presets give you a starting point. The red line is the mean return, the orange band ±1 standard deviation.
    """)
    return


@app.cell
def painter_class(anywidget, np, traitlets):
    class ReturnPainter(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 640, H = 260, PADL = 36, PADB = 34, TOP = 14, R0 = -0.5, R1 = 0.6;
      const plotH = H - TOP - PADB, plotW = W - PADL - 10;
      const toolbar = document.createElement("div");
      toolbar.style.cssText = "display:flex;gap:6px;flex-wrap:wrap;margin-bottom:6px";
      el.appendChild(toolbar);
      const NS = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none", cursor: "crosshair" });
      el.appendChild(svg);
      const mk = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); svg.appendChild(e); return e; };
      const toSvg = (evt) => { const p = svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const n = () => model.get("weights").length;
      const center = k => R0 + (k + 0.5) * (R1 - R0) / n();
      const sx = r => PADL + (r - R0) / (R1 - R0) * plotW;
      const gauss = (m, s) => (k) => Math.exp(-0.5 * ((center(k) - m) / s) ** 2);
      const presets = {
        "Stocks (7% ± 16%)": gauss(0.07, 0.16),
        "Bonds (3% ± 5%)": gauss(0.03, 0.05),
        "Stocks + crash risk": k => gauss(0.09, 0.12)(k) + 0.12 * gauss(-0.35, 0.06)(k),
        "Savings account (2%)": gauss(0.02, 0.012),
        "Lottery-ish": k => gauss(-0.05, 0.05)(k) + 0.06 * gauss(0.5, 0.04)(k),
        "Flat": k => 1,
      };
      for (const name in presets) {
        const b = document.createElement("button"); b.textContent = name;
        b.style.cssText = "padding:3px 8px;border-radius:6px;border:1px solid #aaa;cursor:pointer";
        b.onclick = () => { const w = Array.from({ length: n() }, (_, k) => presets[name](k));
          const m = Math.max(...w, 1e-9); model.set("weights", w.map(v => v / m)); model.save_changes(); };
        toolbar.appendChild(b);
      }
      function draw() {
        svg.innerHTML = "";
        const w = model.get("weights"), N = w.length, bw = plotW / N;
        const tot = w.reduce((a, b) => a + b, 0) || 1;
        let mu = 0; w.forEach((v, k) => mu += v / tot * center(k));
        let v2 = 0; w.forEach((v, k) => v2 += v / tot * (center(k) - mu) ** 2);
        const sd = Math.sqrt(v2);
        mk("rect", { x: sx(mu - sd), y: TOP, width: Math.max(0, sx(mu + sd) - sx(mu - sd)), height: plotH, fill: "#f18f01", opacity: 0.13 });
        w.forEach((v, k) => mk("rect", { x: PADL + k * bw + 1, y: TOP + plotH - v * plotH, width: bw - 2, height: v * plotH,
          fill: center(k) < 0 ? "#6c5b7b99" : "#6c5b7b", rx: 2 }));
        mk("line", { x1: sx(0), x2: sx(0), y1: TOP, y2: TOP + plotH, stroke: "currentColor", "stroke-dasharray": "2 3", opacity: 0.5 });
        mk("line", { x1: sx(mu), x2: sx(mu), y1: TOP, y2: TOP + plotH, stroke: "#e4572e", "stroke-width": 3 });
        const t = mk("text", { x: sx(mu) + 5, y: TOP + 14, "font-size": 13, fill: "#e4572e", "font-weight": "bold" });
        t.textContent = `mean ${(mu * 100).toFixed(1)}%, sd ${(sd * 100).toFixed(1)}%`;
        for (let r = -0.5; r <= 0.61; r += 0.1) {
          const l = mk("text", { x: sx(r), y: H - 12, "font-size": 11, "text-anchor": "middle", fill: "currentColor" });
          l.textContent = `${Math.round(r * 100)}%`;
        }
        const xl = mk("text", { x: PADL + plotW / 2, y: H, "font-size": 11, "text-anchor": "middle", fill: "currentColor", opacity: 0.7 });
        xl.textContent = "yearly return r";
      }
      let painting = false;
      function paint(evt) {
        const p = toSvg(evt), N = n(), bw = plotW / N;
        const k = Math.floor((p.x - PADL) / bw); if (k < 0 || k >= N) return;
        const w = [...model.get("weights")];
        w[k] = Math.max(0, Math.min(1, (TOP + plotH - p.y) / plotH));
        model.set("weights", w); draw();
      }
      svg.addEventListener("pointerdown", e => { painting = true; svg.setPointerCapture(e.pointerId); paint(e); });
      svg.addEventListener("pointermove", e => painting && paint(e));
      const stop = () => { if (painting) { painting = false; model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointercancel", stop);
      model.on("change:weights", draw); draw();
    }
    export default { render };
    """
        weights = traitlets.List(traitlets.Float()).tag(sync=True)

    RET_LO, RET_HI, RET_BINS = -0.5, 0.6, 44
    ret_centers = RET_LO + (np.arange(RET_BINS) + 0.5) * (RET_HI - RET_LO) / RET_BINS
    return RET_HI, RET_LO, ReturnPainter, ret_centers


@app.cell
def painter(ReturnPainter, mo, np, ret_centers):
    _w0 = np.exp(-0.5 * ((ret_centers - 0.07) / 0.16) ** 2)
    painter = mo.ui.anywidget(ReturnPainter(weights=list(_w0 / _w0.max())))
    painter
    return (painter,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Steps 1 & 3 · The model, run thousands of times
    Set the rest of the **predictive model** (start value, yearly contribution $c$, horizon $T$) and a **goal**.
    Each simulated future draws $T$ independent returns from *your painted distribution*. Purple lines are a few individual futures;
    the shaded fan shows the 5–95 % and 25–75 % ranges across all of them.
    """)
    return


@app.cell
def pf_controls(mo):
    pf_start = mo.ui.number(start=0, stop=1_000_000, step=1000, value=10_000, label="Start value V₀ ($)")
    pf_contrib = mo.ui.number(start=0, stop=100_000, step=500, value=3_000, label="Yearly contribution c ($)")
    pf_years = mo.ui.slider(1, 40, value=20, label="Horizon T (years)", show_value=True)
    pf_goal = mo.ui.number(start=0, stop=10_000_000, step=5000, value=150_000, label="Goal ($)")
    pf_nsims = mo.ui.dropdown({"100": 100, "1,000": 1000, "10,000": 10000, "50,000": 50000}, value="10,000", label="Simulations")
    pf_seed = mo.ui.button(label="🎲 new random futures", value=0, on_click=lambda v: v + 1)
    mo.vstack([mo.hstack([pf_start, pf_contrib, pf_goal], justify="start"), mo.hstack([pf_years, pf_nsims, pf_seed], justify="start")])
    return pf_contrib, pf_goal, pf_nsims, pf_seed, pf_start, pf_years


@app.cell
def pf_sim(
    RET_HI,
    RET_LO,
    np,
    painter,
    pf_contrib,
    pf_nsims,
    pf_seed,
    pf_start,
    pf_years,
    ret_centers,
):
    def simulate_portfolio(weights, start, contrib, years, n_sims, seed=0):
        """Monte Carlo: draw yearly returns from the painted histogram, return wealth paths (n_sims x years+1)."""
        _rng = np.random.default_rng(seed)
        _p = np.asarray(weights, float)
        _p = _p / _p.sum() if _p.sum() > 0 else np.full(len(_p), 1 / len(_p))
        _width = (RET_HI - RET_LO) / len(_p)
        _bins = _rng.choice(len(_p), size=(n_sims, years), p=_p)
        _r = ret_centers[_bins] + _rng.uniform(-_width / 2, _width / 2, size=_bins.shape)  # uniform within bin
        _V = np.empty((n_sims, years + 1))
        _V[:, 0] = start
        for _t in range(years):
            _V[:, _t + 1] = _V[:, _t] * (1 + _r[:, _t]) + contrib
        return _V

    pf_paths = simulate_portfolio(painter.value["weights"], pf_start.value, pf_contrib.value, pf_years.value, pf_nsims.value, seed=pf_seed.value)
    pf_final = pf_paths[:, -1]
    return pf_final, pf_paths


@app.cell
def pf_fan(alt, np, pd, pf_goal, pf_paths):
    _T = pf_paths.shape[1]
    _q = np.percentile(pf_paths, [5, 25, 50, 75, 95], axis=0)
    _fan = pd.DataFrame({"year": np.arange(_T), "p5": _q[0], "p25": _q[1], "p50": _q[2], "p75": _q[3], "p95": _q[4]})
    _k = min(12, len(pf_paths))
    _few = pd.DataFrame({"year": np.tile(np.arange(_T), _k), "value": pf_paths[:_k].ravel(), "sim": np.repeat(np.arange(_k), _T)})
    _x = alt.X("year:Q", title="year")
    _y = alt.Y("p5:Q", title="portfolio value ($)", axis=alt.Axis(format="$,.0f"))
    _c = (
        alt.Chart(_fan).mark_area(color="#6c5b7b", opacity=0.15).encode(_x, _y, y2="p95:Q")
        + alt.Chart(_fan).mark_area(color="#6c5b7b", opacity=0.25).encode(_x, y="p25:Q", y2="p75:Q")
        + alt.Chart(_few).mark_line(color="#6c5b7b", opacity=0.45, strokeWidth=1).encode(_x, y="value:Q", detail="sim:N")
        + alt.Chart(_fan).mark_line(color="#6c5b7b", strokeWidth=3).encode(_x, y="p50:Q")
        + alt.Chart(pd.DataFrame({"g": [pf_goal.value]})).mark_rule(color="#2a9d8f", strokeDash=[6, 4], strokeWidth=2).encode(y="g:Q")
    ).properties(width="container", height=340, title="Simulated futures: median (thick), 25–75% and 5–95% fans, goal (teal dashed)")
    _c
    return


@app.cell
def pf_hist(
    alt,
    mo,
    np,
    pd,
    pf_contrib,
    pf_final,
    pf_goal,
    pf_paths,
    pf_start,
):
    _hist, _edges = np.histogram(np.clip(pf_final, None, np.percentile(pf_final, 99.5)), bins=50)
    _hdf = pd.DataFrame({"lo": _edges[:-1], "hi": _edges[1:], "count": _hist})
    _hdf["zone"] = np.where(_hdf["hi"] <= pf_goal.value, "misses goal", "reaches goal")
    _invested = pf_start.value + pf_contrib.value * (pf_paths.shape[1] - 1)
    _hchart = alt.Chart(_hdf).mark_bar().encode(
        x=alt.X("lo:Q", title="final value V_T ($)", axis=alt.Axis(format="$,.0f")), x2="hi:Q", y=alt.Y("count:Q", title="# simulations"),
        color=alt.Color("zone:N", scale=alt.Scale(domain=["misses goal", "reaches goal"], range=["#f18f01", "#2a9d8f"]), legend=alt.Legend(orient="top")),
    ) + alt.Chart(pd.DataFrame({"g": [pf_goal.value]})).mark_rule(color="#2a9d8f", strokeDash=[6, 4], strokeWidth=2).encode(x="g:Q")
    mo.vstack([
        mo.hstack([
            mo.stat(f"{(pf_final >= pf_goal.value).mean():.1%}", label="P(reach goal)"),
            mo.stat(f"${np.median(pf_final):,.0f}", label="median outcome"),
            mo.stat(f"${np.percentile(pf_final, 5):,.0f}", label="bad case (5th pct)"),
            mo.stat(f"{(pf_final < _invested).mean():.1%}", label=f"P(less than invested ${_invested:,.0f})"),
        ], justify="space-around"),
        _hchart.properties(width="container", height=240, title="Distribution of the dependent variable: final value V_T"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Click **Bonds** vs **Stocks**: bonds give a narrow fan (predictable but lower), stocks a wide one. Compare P(reach goal).
    2. Click **Stocks + crash risk**. The mean barely moves, but the 5th-percentile "bad case" drops. A single number
       (the average) hides this; the *distribution* of outcomes is the whole point of Monte Carlo.
    3. Paint a tiny bump at −40 % onto the Stocks preset (a rare crash). How much does it cost you in P(reach goal)?
    4. Set **Simulations = 100** and press 🎲 a few times: the headline numbers jitter. At 50,000 they barely move (Section 2 again!).
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Measure the spread: variance, standard deviation, and "enough" simulations

    The video: *"you'll also want to compute the range of variation within a sample by calculating the **variance** and the
    **standard deviation**, which are commonly used measures of spread."* Two different spreads matter:

    | | formula | what it tells you | shrinks with more sims? |
    |---|---|---|---|
    | **spread of outcomes** | $s = \sqrt{\tfrac{1}{N-1}\sum_i (V^{(i)} - \bar V)^2}$ | how uncertain *the future* is | **no**, it's a property of the world |
    | **spread of the estimate** | $\operatorname{SE} = s/\sqrt{N}$ | how uncertain *your Monte Carlo answer* is | **yes**, like $1/\sqrt N$ |

    Drag the slider to use only the first $N$ of your portfolio simulations from Section 3.
    """)
    return


@app.cell
def spread_n(mo, pf_final):
    spread_n = mo.ui.slider(steps=[s for s in [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000] if s <= len(pf_final)],
                             value=min(100, len(pf_final)), label="Use the first N simulations", show_value=True)
    spread_n
    return (spread_n,)


@app.cell
def spread_chart(alt, mo, np, pd, pf_final, spread_n):
    _N = spread_n.value
    _sub = pf_final[:_N]
    _mean, _sd = _sub.mean(), _sub.std(ddof=1)
    _se = _sd / np.sqrt(_N)
    _grid = np.unique(np.logspace(1, np.log10(len(pf_final)), 80).astype(int))
    _cm = np.cumsum(pf_final)[_grid - 1] / _grid
    _cs = np.array([pf_final[:g].std(ddof=1) for g in _grid])
    _run = pd.DataFrame({"N": _grid, "mean": _cm, "lo": _cm - 1.96 * _cs / np.sqrt(_grid), "hi": _cm + 1.96 * _cs / np.sqrt(_grid), "sd": _cs})
    _x = alt.X("N:Q", scale=alt.Scale(type="log"), title="number of simulations N (log)")
    _left = (
        alt.Chart(_run).mark_area(color="#6c5b7b", opacity=0.25).encode(_x, y=alt.Y("lo:Q", title="running mean of V_T ($)", axis=alt.Axis(format="$,.0f")), y2="hi:Q")
        + alt.Chart(_run).mark_line(color="#6c5b7b", strokeWidth=2).encode(_x, y="mean:Q")
        + alt.Chart(pd.DataFrame({"N": [_N]})).mark_rule(color="#e4572e").encode(x="N:Q")
    ).properties(width=330, height=260, title="Estimate of E[V_T] ± 1.96·SE: narrows")
    _right = (
        alt.Chart(_run).mark_line(color="#f18f01", strokeWidth=2).encode(_x, y=alt.Y("sd:Q", title="running std of V_T ($)", axis=alt.Axis(format="$,.0f")))
        + alt.Chart(pd.DataFrame({"N": [_N]})).mark_rule(color="#e4572e").encode(x="N:Q")
    ).properties(width=330, height=260, title="Std of outcomes s: settles, doesn't shrink")
    mo.vstack([
        mo.hstack([
            mo.stat(f"${_mean:,.0f}", label=f"mean of first {_N:,}"),
            mo.stat(f"${_sd:,.0f}", label="std of outcomes s"),
            mo.stat(f"{_sd**2:.3g} $²", label="variance s²"),
            mo.stat(f"±${1.96*_se:,.0f}", label="95% range of the mean estimate"),
        ], justify="space-around"),
        alt.hconcat(_left, _right),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Slide N from 10 to the max. The left band (uncertainty of *your answer*) collapses; the right curve (uncertainty of
       *the future*) just stabilises. More simulations can't make the future less risky, only your picture of it sharper.
    2. Go back to Section 3, pick **Savings account**: s becomes tiny, so even N = 100 is plenty.
       Pick **Lottery-ish**: rare huge wins make s large and the running mean jumpy, so you need far more simulations.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Sandbox: capacity planning ("planning for spare capacity")

    Another application from the video. You run a service and must decide **how many servers** to provision.
    Daily demand is uncertain, so build the model with the three steps:

    1. **Model:** load $= \text{users} \times \text{requests per user}$ (+ a rare traffic spike). Overloaded if load $>$ capacity.
    2. **Distributions:** users ~ Normal(μ, σ), requests/user ~ Uniform(a, b), spike ~ Bernoulli(p) × multiplier.
    3. **Simulate** many days and read off $P(\text{overload})$.

    Play with the knobs; the challenges below give you targets.
    """)
    return


@app.cell
def sandbox_controls(mo):
    sb = mo.ui.dictionary({
        "users_mu": mo.ui.slider(1000, 20000, step=500, value=8000, label="users: mean μ", show_value=True),
        "users_sd": mo.ui.slider(0, 5000, step=250, value=1500, label="users: std σ", show_value=True),
        "req_a": mo.ui.slider(1, 20, value=5, label="requests/user: min a", show_value=True),
        "req_b": mo.ui.slider(1, 40, value=15, label="requests/user: max b", show_value=True),
        "spike_p": mo.ui.slider(0.0, 0.2, step=0.01, value=0.03, label="P(traffic spike) per day", show_value=True),
        "spike_x": mo.ui.slider(1.0, 5.0, step=0.25, value=2.5, label="spike multiplier", show_value=True),
        "servers": mo.ui.slider(1, 60, value=14, label="servers provisioned", show_value=True),
        "per_server": mo.ui.slider(2000, 20000, step=1000, value=10000, label="requests one server handles/day", show_value=True),
        "days": mo.ui.dropdown({"1,000": 1000, "10,000": 10000, "100,000": 100000}, value="10,000", label="simulated days"),
    })
    mo.hstack([
        mo.vstack([mo.md("**Inputs (independent variables)**"), sb["users_mu"], sb["users_sd"], sb["req_a"], sb["req_b"], sb["spike_p"], sb["spike_x"]]),
        mo.vstack([mo.md("**Decision & simulation**"), sb["servers"], sb["per_server"], sb["days"]]),
    ], justify="start", gap=2)
    return (sb,)


@app.cell
def sandbox_sim(alt, mo, np, pd, sb):
    def simulate_load(users_mu, users_sd, req_a, req_b, spike_p, spike_x, days, seed=7):
        """Monte Carlo daily request load."""
        _rng = np.random.default_rng(seed)
        _users = np.maximum(0, _rng.normal(users_mu, users_sd, days))
        _lo, _hi = min(req_a, req_b), max(req_a, req_b)
        _req = _rng.uniform(_lo, _hi, days)
        _spike = np.where(_rng.random(days) < spike_p, spike_x, 1.0)
        return _users * _req * _spike

    _v = sb.value
    sb_load = simulate_load(_v["users_mu"], _v["users_sd"], _v["req_a"], _v["req_b"], _v["spike_p"], _v["spike_x"], _v["days"])
    sb_capacity = _v["servers"] * _v["per_server"]
    _p_over = (sb_load > sb_capacity).mean()
    _se = np.sqrt(_p_over * (1 - _p_over) / len(sb_load))
    _need99 = int(np.ceil(np.percentile(sb_load, 99) / _v["per_server"]))
    _hist, _edges = np.histogram(sb_load, bins=60)
    _df = pd.DataFrame({"lo": _edges[:-1], "hi": _edges[1:], "count": _hist})
    _df["zone"] = np.where(_df["lo"] >= sb_capacity, "overload", "ok")
    _chart = (
        alt.Chart(_df).mark_bar().encode(
            x=alt.X("lo:Q", title="requests per day", axis=alt.Axis(format=",.0f")), x2="hi:Q", y=alt.Y("count:Q", title="# simulated days"),
            color=alt.Color("zone:N", scale=alt.Scale(domain=["ok", "overload"], range=["#6c5b7b", "#f18f01"]), legend=alt.Legend(orient="top")))
        + alt.Chart(pd.DataFrame({"c": [sb_capacity]})).mark_rule(color="#e4572e", strokeWidth=2).encode(x="c:Q")
    ).properties(width="container", height=260, title="Simulated daily load vs capacity (red)")
    mo.vstack([
        mo.hstack([
            mo.stat(f"{_p_over:.2%}", label="P(overload) per day", caption=f"± {1.96*_se:.2%} (95%)"),
            mo.stat(f"{365*_p_over:.1f}", label="expected overload days / year"),
            mo.stat(f"{sb_load.mean():,.0f}", label="mean load"),
            mo.stat(f"{sb_load.std(ddof=1):,.0f}", label="std of load"),
            mo.stat(f"{_need99}", label="servers for 99% of days"),
        ], justify="space-around"),
        _chart,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Challenges**

    1. Find the **fewest servers** that keep P(overload) under **1 %**. Then double the spike probability: how many more do you need?
    2. Set users σ = 0 and requests/user a = b. The "simulation" becomes deterministic: one bar. Monte Carlo only matters when inputs are uncertain.
    3. Set P(overload) ≈ 0.1 % and compare 1,000 vs 100,000 simulated days. Why is the ± range so wide with 1,000? (Hint: rare events, Section 2.)
    """), kind="neutral")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Quiz: check your Monte Carlo intuition
    Answer all, then flip the switch. Question 3 is computed live by a Monte Carlo simulation in this notebook.
    """)
    return


@app.cell
def quiz(mo):
    quiz_items = {
        "q1": ("You estimate P(sum = 7) with 400 rolls. Roughly how big is the 95% error range?",
               ["±0.002", "±0.036", "±0.17"], "±0.036"),
        "q2": ("To make a Monte Carlo estimate 10× more precise you need about…",
               ["10× more samples", "100× more samples", "√10× more samples"], "100× more samples"),
        "q3": ("Two dice: which is closest to P(sum ≥ 10)? (the notebook checks with 200,000 simulated rolls)",
               ["0.08", "0.17", "0.28"], None),
        "q4": ("Running 10× more portfolio simulations will make the std of final wealth s…",
               ["shrink by √10", "stay about the same", "grow"], "stay about the same"),
        "q5": ("In the video's step 2 you…",
               ["pick the dependent variable", "specify probability distributions of the inputs", "average the results"],
               "specify probability distributions of the inputs"),
    }
    quiz = mo.ui.dictionary({_k: mo.ui.radio(_opts, label=f"**{_i + 1}.** {_q}") for _i, (_k, (_q, _opts, _a)) in enumerate(quiz_items.items())})
    quiz_reveal = mo.ui.switch(label="Reveal answers")
    mo.vstack([*quiz.values(), quiz_reveal])
    return quiz, quiz_items, quiz_reveal


@app.cell(hide_code=True)
def quiz_result(mo, np, quiz, quiz_items, quiz_reveal):
    _rng = np.random.default_rng(2026)
    _s = _rng.integers(1, 7, 200_000) + _rng.integers(1, 7, 200_000)
    _p10 = (_s >= 10).mean()
    _q3 = min(["0.08", "0.17", "0.28"], key=lambda o: abs(float(o) - _p10))
    mo.stop(not quiz_reveal.value, mo.md("_Answer the questions, then flip **Reveal answers**._"))
    _answers = {_k: (_a if _a is not None else _q3) for _k, (_q, _o, _a) in quiz_items.items()}
    _why = {
        "q1": "1.96·√(⅙·⅚/400) ≈ 0.036.",
        "q2": "Error ∝ 1/√N, so 10× precision needs 10² = 100× samples.",
        "q3": f"Simulated P̂(sum ≥ 10) = {_p10:.4f}; exact 6/36 = 0.1667.",
        "q4": "s measures how uncertain the future is; only the SE of the *estimate* shrinks.",
        "q5": "Step 1 model, step 2 input distributions, step 3 simulate repeatedly.",
    }
    _score = sum(quiz.value[_k] == _answers[_k] for _k in quiz_items)
    _rows = "\n".join(
        f"| {_i + 1} | {quiz.value[_k] or '–'} | {_answers[_k]} | {'✅' if quiz.value[_k] == _answers[_k] else '❌'} | {_why[_k]} |"
        for _i, _k in enumerate(quiz_items)
    )
    mo.vstack([
        mo.stat(f"{_score} / {len(quiz_items)}", label="your score"),
        mo.md("| # | your answer | correct | | why |\n|---|---|---|---|---|\n" + _rows),
    ])
    return


@app.cell(hide_code=True)
def recap(mo):
    mo.md(r"""
    ## Recap

    | Video idea | Formula / rule | Where you played with it |
    |---|---|---|
    | Monte Carlo = estimate outcomes of an uncertain event by **random sampling** | $\hat P = \#\text{hits}/N$ | §1 dice roller (anywidget) |
    | More samples → more accurate | $\operatorname{SE} = \sqrt{P(1-P)/N} \propto 1/\sqrt N$ | §2 convergence fan + predict-N |
    | Applications: portfolios, risk, option pricing, spare capacity | outcome **distribution**, not one number | §3 portfolio, §5 capacity sandbox |
    | Step 1: predictive model (dependent ← independent variables) | $V_t = V_{t-1}(1+r_t)+c$ | §3 model controls |
    | Step 2: probability distributions of inputs | historical data or analyst judgment | §3 return painter (anywidget) |
    | Step 3: simulate repeatedly | draw inputs → compute output → repeat | §3 fan chart & histogram |
    | Measure spread with variance / std | $s^2$, $s$ (world) vs $s/\sqrt N$ (estimate) | §4 running mean & std |

    *"While you may not be able to travel into the future, with Monte Carlo simulation you'll have a much better idea of the possibilities the future holds."*
    """)
    return


if __name__ == "__main__":
    app.run()

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
        mo.md("# Markov Chain Monte Carlo, explained by touching it"),
        mo.Html('<iframe width="100%" height="380" src="https://www.youtube.com/embed/nndtTssgtZE" title="MCMC - Explained" frameborder="0" allowfullscreen></iframe>'),
        mo.md(r"""
    *Picture a walker in complete darkness exploring a mountain range, step by step, where each footstep depends only on where they stand right now. Somehow, the walker ends up mapping the whole landscape.* This notebook follows the video (DataMListic) chapter by chapter: **accept-reject sampling → Markov chains → Monte Carlo → the stationary distribution → MCMC in action → burn-in → detailed balance**.

    **Color language** (same as the video):
    <span style="color:#2a9d8f">■ green</span> = accepted / kept samples,
    <span style="color:#d62828">■ red</span> = rejected samples or burn-in,
    <span style="color:#e9a23b">■ gold</span> = the target distribution / true value,
    <span style="color:#457b9d">■ blue</span> = independent samples or proposals.
    """),
    ])
    return


@app.cell
def target(np, pd):
    # Shared target: the video's two-peak landscape (30% left peak, 70% right peak)
    W_MIX = np.array([0.3, 0.7]); MU_MIX = np.array([-2.0, 2.0]); SD_MIX = np.array([0.6, 0.8])

    def target_pdf(x):
        x = np.asarray(x, dtype=float)[..., None]
        return np.sum(W_MIX * np.exp(-0.5 * ((x - MU_MIX) / SD_MIX) ** 2) / (SD_MIX * np.sqrt(2 * np.pi)), axis=-1)

    X_GRID = np.linspace(-5, 5, 201)
    PDF_GRID = target_pdf(X_GRID)
    TRUE_MEAN = float(W_MIX @ MU_MIX)  # 0.3*(-2) + 0.7*2 = 0.8
    target_df = pd.DataFrame({"x": X_GRID, "density": PDF_GRID})
    return (
        MU_MIX,
        PDF_GRID,
        SD_MIX,
        TRUE_MEAN,
        W_MIX,
        X_GRID,
        target_df,
        target_pdf,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Why not just throw darts? (accept-reject sampling)

    The naive way to sample from a weird density $p(x)$: draw a **bounding box** around it, throw points uniformly
    inside, keep the ones **under the curve** (green), throw away the ones above (red).

    $$\text{acceptance rate} = \frac{\text{area under } p(x)}{\text{area of the box}}$$

    It works, but in the video *"out of 30 samples, we're rejecting most of them."* And the box gets relatively
    huge as dimensions grow. Press **▶ throw** below.
    """)
    return


@app.cell
def DartBox_cls(anywidget, traitlets):
    class DartBox(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 640, H = 260, PAD = 30, XMIN = -5, XMAX = 5;
      const NS = "http://www.w3.org/2000/svg";
      const ctl = document.createElement("div"); ctl.style.marginBottom = "6px"; el.appendChild(ctl);
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none" });
      el.appendChild(svg);
      const info = document.createElement("div"); info.style.fontFamily = "monospace"; el.appendChild(info);
      const mk = (tag, a, p = svg) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
      let pts = [], timer = null;
      const ymax = () => model.get("box_height");
      const sx = v => PAD + (v - XMIN) / (XMAX - XMIN) * (W - 2 * PAD);
      const sy = v => H - PAD - v / ymax() * (H - 2 * PAD);
      const f = x => { const xs = model.get("xs"), ys = model.get("ys");
        const t = (x - xs[0]) / (xs[xs.length - 1] - xs[0]) * (xs.length - 1); const i = Math.max(0, Math.min(xs.length - 2, Math.floor(t)));
        return ys[i] + (ys[i + 1] - ys[i]) * (t - i); };
      function draw() {
        svg.innerHTML = "";
        mk("rect", { x: sx(XMIN), y: sy(ymax()), width: sx(XMAX) - sx(XMIN), height: sy(0) - sy(ymax()), fill: "none", stroke: "#2a9d8f", "stroke-width": 2, "stroke-dasharray": "6 4" });
        const xs = model.get("xs"), ys = model.get("ys");
        mk("path", { d: xs.map((x, i) => (i ? "L" : "M") + sx(x) + "," + sy(ys[i])).join(" "), fill: "rgba(233,162,59,0.25)", stroke: "#e9a23b", "stroke-width": 3 });
        for (const p of pts) mk("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 3.2, fill: p[2] ? "#2a9d8f" : "#d62828", opacity: 0.85 });
        const acc = pts.filter(p => p[2]).length;
        info.textContent = `thrown: ${pts.length}   accepted: ${acc}   rejected: ${pts.length - acc}   acceptance: ${pts.length ? (100 * acc / pts.length).toFixed(1) : "-"}%`;
      }
      function throwOne() {
        const x = XMIN + Math.random() * (XMAX - XMIN), y = Math.random() * ymax();
        pts.push([x, y, y < f(x)]); if (pts.length > 3000) pts.shift();
        draw();
        if (pts.length % 20 === 0) sync();
      }
      function sync() { model.set("thrown", pts.length); model.set("accepted", pts.filter(p => p[2]).length); model.save_changes(); }
      const btn = (t, fn) => { const b = document.createElement("button"); b.textContent = t; b.onclick = fn; b.style.marginRight = "6px"; ctl.appendChild(b); return b; };
      const play = btn("▶ throw", () => { if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ throw"; sync(); }
        else { timer = setInterval(throwOne, 30); play.textContent = "⏸ pause"; } });
      btn("+30 darts", () => { for (let i = 0; i < 30; i++) throwOne(); sync(); });
      btn("↺ reset", () => { pts = []; draw(); sync(); });
      model.on("change:box_height", () => { pts = []; draw(); sync(); });
      draw();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        xs = traitlets.List(traitlets.Float()).tag(sync=True)
        ys = traitlets.List(traitlets.Float()).tag(sync=True)
        box_height = traitlets.Float(0.4).tag(sync=True)
        thrown = traitlets.Int(0).tag(sync=True)
        accepted = traitlets.Int(0).tag(sync=True)

    return (DartBox,)


@app.cell
def box_slider(mo):
    box_slider = mo.ui.slider(0.36, 1.5, step=0.02, value=0.4, label="Box height (must cover the peak ≈ 0.35)", show_value=True)
    box_slider
    return (box_slider,)


@app.cell
def dartbox(DartBox, PDF_GRID, X_GRID, box_slider, mo):
    dartbox = mo.ui.anywidget(DartBox(xs=X_GRID.tolist(), ys=PDF_GRID.tolist(), box_height=box_slider.value))
    dartbox
    return (dartbox,)


@app.cell
def dart_readout(box_slider, dartbox, mo):
    _n, _a = dartbox.value["thrown"], dartbox.value["accepted"]
    _theory = 1.0 / (10 * box_slider.value)   # area under p = 1, box area = width 10 × height
    mo.hstack([
        mo.stat(f"{_n}", label="darts thrown"),
        mo.stat(f"{(_a / _n if _n else 0):.1%}", label="observed acceptance"),
        mo.stat(f"{_theory:.1%}", label="theory: 1 / box area", caption="area under p(x) = 1"),
        mo.stat(f"{(_n - _a)}", label="wasted (red) darts"),
    ], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The curse of dimensionality

    Even for the *friendliest* target, a ball inside its bounding cube, the acceptance rate is
    $\dfrac{\text{vol(ball)}}{\text{vol(cube)}} = \dfrac{\pi^{d/2}}{\Gamma(d/2+1)\,2^d}$, which collapses as dimension $d$ grows.
    This is the video's *"acceptance rates that drop to less than 1%"*.
    """)
    return


@app.cell
def dim_slider(mo):
    dim_slider = mo.ui.slider(1, 30, value=10, label="dimension d", show_value=True)
    dim_slider
    return (dim_slider,)


@app.cell
def curse_chart(alt, dim_slider, mo, np, pd):
    from math import gamma as _gamma, pi as _pi
    _d = np.arange(1, 31)
    _rate = np.array([_pi ** (k / 2) / _gamma(k / 2 + 1) / 2 ** k for k in _d])
    ball_rate = dict(zip(_d.tolist(), _rate.tolist()))
    _df = pd.DataFrame({"d": _d, "acceptance": _rate, "now": _d == dim_slider.value})
    _line = alt.Chart(_df).mark_line(color="#d62828").encode(
        x=alt.X("d:Q", title="dimension d"),
        y=alt.Y("acceptance:Q", scale=alt.Scale(type="log"), title="acceptance rate (log)"))
    _pt = alt.Chart(_df[_df.now]).mark_point(size=150, filled=True, color="#e9a23b").encode(x="d:Q", y="acceptance:Q")
    mo.vstack([
        (_line + _pt).properties(width="container", height=220),
        mo.md(f"At **d = {dim_slider.value}**, only **{ball_rate[dim_slider.value]:.2e}** of darts land inside: "
              f"you'd throw ~**{1 / ball_rate[dim_slider.value]:,.0f}** darts per useful sample."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Throw ~300 darts with the box just covering the peak (0.36). Acceptance is around 28%.
    2. Drag the box height up to 1.5. Acceptance drops to ~7%, because the box is "too big for the peaks".
    3. Move to d = 20 in the chart. Fewer than one dart in 40 million is accepted. Throwing samples independently just doesn't scale.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Key insight: let each sample remember where we just were

    Independent sampling (blue) throws away everything it learned. A **Markov chain** creates dependence instead:

    $$x_{n+1} \sim T(\,\cdot \mid x_n)$$

    The next state depends **only on the present**, not the past (Andrey Markov's *memoryless* property).
    In the video, a yellow particle hops around a **circular network of states**, deciding where to go based only on its
    current position. Below is that ring. Each state $i$ stays put, steps clockwise, or steps counter-clockwise.
    You control how much it prefers clockwise moves. Watch where the particle spends its time.
    """)
    return


@app.cell
def ring_controls(mo):
    ring_k = mo.ui.slider(4, 10, value=6, label="number of states", show_value=True)
    ring_cw = mo.ui.slider(0.05, 0.9, step=0.05, value=0.5, label="pull toward state 0 (P of stepping homeward)", show_value=True)
    mo.hstack([ring_k, ring_cw], justify="start")
    return ring_cw, ring_k


@app.cell
def ring_math(np, ring_cw, ring_k):
    def ring_matrix(k, pull, stay=0.2):
        """Ring transition matrix: stay with prob `stay`, otherwise step to a neighbour.
        The step goes toward state 0 (along the shorter arc) with prob `pull`; state 0 itself is symmetric."""
        T = np.zeros((k, k))
        for _i in range(k):
            _cw_toward = (k - _i) < _i            # is clockwise the shorter way home?
            _p_cw = 0.5 if _i == 0 or 2 * _i == k else (pull if _cw_toward else 1 - pull)
            T[_i, _i] += stay
            T[_i, (_i + 1) % k] += (1 - stay) * _p_cw
            T[_i, (_i - 1) % k] += (1 - stay) * (1 - _p_cw)
        return T

    def stationary(T):
        _vals, _vecs = np.linalg.eig(T.T)
        _v = np.real(_vecs[:, np.argmin(np.abs(_vals - 1))])
        return _v / _v.sum()

    RING_T = ring_matrix(ring_k.value, ring_cw.value)
    RING_PI = stationary(RING_T)
    return RING_PI, RING_T


@app.cell
def RingWalker_cls(anywidget, traitlets):
    class RingWalker(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 640, H = 300, NS = "http://www.w3.org/2000/svg";
      const ctl = document.createElement("div"); el.appendChild(ctl);
      const svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none" }); el.appendChild(svg);
      const mk = (tag, a) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); svg.appendChild(e); return e; };
      const txt = (x, y, s, a = {}) => { const t = mk("text", Object.assign({ x, y, "text-anchor": "middle", "font-size": 12, fill: "currentColor" }, a)); t.textContent = s; return t; };
      let cur = 0, visits = [], n = 0, timer = null, prev = null;
      const reset = () => { const k = model.get("T").length; visits = Array(k).fill(0); n = 0; cur = 0; prev = null; };
      reset();
      const CX = 150, CY = 150, R = 110;
      const pos = i => { const k = model.get("T").length, a = -Math.PI / 2 + 2 * Math.PI * i / k; return [CX + R * Math.cos(a), CY + R * Math.sin(a)]; };
      function draw() {
        svg.innerHTML = "";
        const T = model.get("T"), k = T.length, pi = model.get("pi");
        for (let i = 0; i < k; i++) { const [x1, y1] = pos(i), [x2, y2] = pos((i + 1) % k);
          mk("line", { x1, y1, x2, y2, stroke: "#888", "stroke-width": 2, opacity: 0.5 }); }
        for (let i = 0; i < k; i++) { const [x, y] = pos(i);
          mk("circle", { cx: x, cy: y, r: 20, fill: i === cur ? "#e9a23b" : "rgba(69,123,157,0.25)", stroke: "#457b9d", "stroke-width": 2 });
          txt(x, y + 4, String(i)); }
        if (prev !== null && prev !== cur) { const [x1, y1] = pos(prev), [x2, y2] = pos(cur);
          mk("line", { x1, y1, x2, y2, stroke: "#e9a23b", "stroke-width": 5, opacity: 0.8 }); }
        txt(CX, CY + 4, `step ${n}`, { "font-size": 14 });
        // bars: visit fraction (green) vs stationary (gold tick)
        const BX = 320, BW = 300, BH = 220, BY = 40, bw = BW / k;
        txt(BX + BW / 2, 22, "fraction of time in each state  (gold tick = stationary π)");
        for (let i = 0; i < k; i++) {
          const f = n ? visits[i] / n : 0, h = f / 0.5 * BH, hp = pi[i] / 0.5 * BH;
          mk("rect", { x: BX + i * bw + 4, y: BY + BH - h, width: bw - 8, height: h, fill: "#2a9d8f" });
          mk("line", { x1: BX + i * bw + 2, x2: BX + (i + 1) * bw - 2, y1: BY + BH - hp, y2: BY + BH - hp, stroke: "#e9a23b", "stroke-width": 4 });
          txt(BX + i * bw + bw / 2, BY + BH + 16, String(i));
        }
      }
      function step() {
        const row = model.get("T")[cur]; let u = Math.random(), j = 0;
        while (j < row.length - 1 && u > row[j]) { u -= row[j]; j++; }
        prev = cur; cur = j; visits[cur]++; n++;
        draw();
        if (n % 50 === 0) sync();
      }
      function sync() { model.set("visits", visits.slice()); model.save_changes(); }
      const btn = (t, fn) => { const b = document.createElement("button"); b.textContent = t; b.onclick = fn; b.style.marginRight = "6px"; ctl.appendChild(b); return b; };
      const speed = () => model.get("fast") ? 2 : 250;
      const play = btn("▶ walk", () => { if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ walk"; sync(); }
        else { timer = setInterval(step, speed()); play.textContent = "⏸ pause"; } });
      btn("step", () => { step(); sync(); });
      btn("+1000 steps", () => { for (let i = 0; i < 1000; i++) step(); sync(); });
      btn("↺ reset", () => { reset(); draw(); sync(); });
      model.on("change:T", () => { reset(); draw(); });
      model.on("change:pi", draw);
      draw();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        T = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)
        pi = traitlets.List(traitlets.Float()).tag(sync=True)
        visits = traitlets.List(traitlets.Int()).tag(sync=True)
        fast = traitlets.Bool(False).tag(sync=True)

    return (RingWalker,)


@app.cell
def ring(RING_PI, RING_T, RingWalker, mo):
    ring = mo.ui.anywidget(RingWalker(T=RING_T.tolist(), pi=RING_PI.tolist()))
    ring
    return (ring,)


@app.cell
def ring_readout(RING_PI, RING_T, mo, np, pd, ring):
    _v = np.array(ring.value["visits"] or [0] * len(RING_PI), dtype=float)
    _n = _v.sum()
    _emp = _v / _n if _n else np.zeros_like(_v)
    _tv = 0.5 * np.abs(_emp - RING_PI).sum() if _n else float("nan")
    mo.vstack([
        mo.hstack([
            mo.stat(f"{int(_n)}", label="steps synced"),
            mo.stat(f"{_tv:.3f}" if _n else "–", label="distance to π (total variation)", caption="→ 0 as steps grow"),
        ], justify="start"),
        mo.md("**Transition matrix T** (row = current state, column = next state). Stationary π solves π T = π:"),
        mo.ui.table(pd.DataFrame(RING_T, columns=[f"→{_j}" for _j in range(len(RING_PI))]).round(3)
                    .assign(**{"π (stationary)": RING_PI.round(3)}), selection=None, show_column_summaries=False),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. With pull = 0.5, the walk is symmetric. Press **+1000 steps** a few times. Every state gets the same share of time.
    2. Set pull = 0.8. Every state nudges the particle toward state 0, so the gold ticks pile up there. Watch the green bars converge to them, wherever the particle started.
    3. Hit **step** slowly. Each hop uses only the current state's row of T: the particle is memoryless, yet its long-run time shares are completely predictable.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Monte Carlo: estimate by averaging random samples

    *"Imagine trying to find the average of this distribution. Analytically, we need complex integrals."*
    Monte Carlo swaps the integral for an average:

    $$\mathbb{E}[x] = \int x\,p(x)\,dx \;\approx\; \frac{1}{N}\sum_{n=1}^{N} x_n, \qquad x_n \sim p$$

    Our two-peak target has true mean $0.3\cdot(-2) + 0.7\cdot 2 = 0.8$ (gold line). The **running average** (red) of the
    green samples converges to it, with error shrinking like $1/\sqrt{N}$.
    """)
    return


@app.cell
def mc_controls(mo):
    mc_n = mo.ui.slider(10, 5000, step=10, value=200, label="number of samples N", show_value=True)
    mc_reroll = mo.ui.button(label="🎲 new samples", value=0, on_click=lambda v: v + 1)
    mo.hstack([mc_n, mc_reroll], justify="start")
    return mc_n, mc_reroll


@app.cell
def mc_chart(
    MU_MIX,
    SD_MIX,
    TRUE_MEAN,
    W_MIX,
    alt,
    mc_n,
    mc_reroll,
    mo,
    np,
    pd,
):
    _rng = np.random.default_rng(100 + mc_reroll.value)
    _comp = _rng.random(mc_n.value) < W_MIX[1]
    mc_samples = np.where(_comp, _rng.normal(MU_MIX[1], SD_MIX[1], mc_n.value), _rng.normal(MU_MIX[0], SD_MIX[0], mc_n.value))
    _run = np.cumsum(mc_samples) / np.arange(1, mc_n.value + 1)
    _step = max(1, mc_n.value // 600)
    _df = pd.DataFrame({"N": np.arange(1, mc_n.value + 1), "running mean": _run, "sample": mc_samples})[::_step]
    _pts = alt.Chart(_df).mark_circle(size=12, color="#2a9d8f", opacity=0.35).encode(x=alt.X("N:Q", title="sample number N"), y=alt.Y("sample:Q", title="value"))
    _ln = alt.Chart(_df).mark_line(color="#d62828", strokeWidth=3).encode(x="N:Q", y="running mean:Q")
    _truth = alt.Chart(pd.DataFrame({"y": [TRUE_MEAN]})).mark_rule(color="#e9a23b", strokeWidth=3, strokeDash=[6, 4]).encode(y="y:Q")
    mo.vstack([
        (_pts + _ln + _truth).properties(width="container", height=260),
        mo.hstack([
            mo.stat(f"{_run[-1]:.3f}", label="Monte Carlo estimate"),
            mo.stat(f"{TRUE_MEAN:.3f}", label="true mean"),
            mo.stat(f"{abs(_run[-1] - TRUE_MEAN):.3f}", label="error"),
            mo.stat(f"{mc_samples.std() / np.sqrt(mc_n.value):.3f}", label="expected error ≈ σ/√N"),
        ], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Set N = 20 and re-roll several times. The estimate jumps around a lot.
    2. Raise N to 2000. To halve the error you need **4×** the samples (the 1/√N law).
    3. Here we *cheated*: we could draw independent samples of p directly. The rest of the lesson is about what to do when you **can't**, and that's where Markov chains come in.
    """), kind="warn")
    return


@app.cell
def metropolis_fn(np):
    def metropolis(pdf, x0, n_steps, step, seed=0):
        """Random-walk Metropolis: propose y = x + N(0, step²), accept with prob min(1, p(y)/p(x)).
        Returns (chain, accepted_flags). Only *ratios* of pdf are used, so pdf may be unnormalized."""
        _rng = np.random.default_rng(seed)
        _x, _px = float(x0), float(pdf(x0))
        _chain = np.empty(n_steps); _acc = np.zeros(n_steps, dtype=bool)
        for _t in range(n_steps):
            _y = _x + step * _rng.normal()
            _py = float(pdf(_y))
            if _rng.random() < (1.0 if _px == 0 else min(1.0, _py / _px)):
                _x, _px, _acc[_t] = _y, _py, True
            _chain[_t] = _x
        return _chain, _acc

    return (metropolis,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · The stationary-distribution trick

    Section 2 showed that a Markov chain run long enough spends a **fixed fraction of time** in each state, its
    stationary distribution π. MCMC *engineers* the transition rule so that **π equals the target p**.

    *"No matter where we started, the red, blue, or green chains all converge to spending time proportional to the height of
    our target. After enough steps, 30% of samples are in this peak, 70% in that peak."*

    Below, three chains start at very different places. Track the **fraction of time each spends in the right peak (x > 0)**.
    """)
    return


@app.cell
def stat_controls(mo):
    stat_steps = mo.ui.slider(50, 20000, step=50, value=2000, label="steps per chain", show_value=True)
    stat_step = mo.ui.slider(0.2, 5.0, step=0.1, value=1.5, label="proposal step size", show_value=True)
    mo.hstack([stat_steps, stat_step], justify="start")
    return stat_step, stat_steps


@app.cell
def stat_chart(alt, metropolis, mo, np, pd, stat_step, stat_steps, target_pdf):
    _starts = {"red (start −4.5)": -4.5, "blue (start 0)": 0.0, "green (start +4.5)": 4.5}
    _rows = []
    for _k, (_lab, _x0) in enumerate(_starts.items()):
        _c, _ = metropolis(target_pdf, _x0, stat_steps.value, stat_step.value, seed=10 + _k)
        _frac = np.cumsum(_c > 0) / np.arange(1, len(_c) + 1)
        _idx = np.unique(np.geomspace(1, len(_c), 300).astype(int)) - 1
        _rows.append(pd.DataFrame({"step": _idx + 1, "fraction in right peak": _frac[_idx], "chain": _lab}))
    stat_df = pd.concat(_rows)
    _lines = alt.Chart(stat_df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("step:Q", scale=alt.Scale(type="log"), title="step (log scale)"),
        y=alt.Y("fraction in right peak:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("chain:N", scale=alt.Scale(domain=list(_starts), range=["#d62828", "#457b9d", "#2a9d8f"])))
    _truth = alt.Chart(pd.DataFrame({"y": [0.7]})).mark_rule(color="#e9a23b", strokeWidth=3, strokeDash=[6, 4]).encode(y="y:Q")
    _final = stat_df.groupby("chain")["fraction in right peak"].last()
    mo.vstack([
        (_lines + _truth).properties(width="container", height=260),
        mo.hstack([mo.stat(f"{_v:.1%}", label=_k) for _k, _v in _final.items()] + [mo.stat("70.0%", label="target (gold)")], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. With 50 steps, the three chains disagree wildly: each still "remembers" where it started.
    2. Slide to 20,000 steps. All three land near **70%**, matching the target.
    3. Set the step size to 0.2. The chains rarely cross the valley between peaks, so convergence is much slower. *Mixing* depends on the proposal.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · MCMC in action: the Metropolis walker

    The algorithm the video animates (random-walk **Metropolis**):

    1. From the current point $x$, **propose** $y = x + \varepsilon$, with $\varepsilon \sim \mathcal N(0, s^2)$ (blue dot)
    2. **Accept** with probability $\alpha = \min\!\left(1, \dfrac{p(y)}{p(x)}\right)$ and move there (green), otherwise **reject** and stay put (red flash; $x$ is recorded again)

    Uphill moves are always accepted. Downhill moves are accepted *sometimes*, which keeps the walker from getting stuck on one
    summit. Only the **ratio** $p(y)/p(x)$ is needed, so the normalizing constant never matters. That's why MCMC works for Bayesian posteriors.
    """)
    return


@app.cell
def MetropolisWalker_cls(anywidget, traitlets):
    class MetropolisWalker(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 640, H = 330, PAD = 30, XMIN = -5, XMAX = 5, NS = "http://www.w3.org/2000/svg";
      const ctl = document.createElement("div"); el.appendChild(ctl);
      const lab = document.createElement("label"); lab.style.marginLeft = "10px";
      lab.innerHTML = 'step size s <input type="range" min="0.1" max="5" step="0.1" style="vertical-align:middle"> <span></span>';
      const rng = lab.querySelector("input"), rv = lab.querySelector("span");
      const svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none" }); el.appendChild(svg);
      const info = document.createElement("div"); info.style.fontFamily = "monospace"; el.appendChild(info);
      const mk = (tag, a) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); svg.appendChild(e); return e; };
      const sx = v => PAD + (v - XMIN) / (XMAX - XMIN) * (W - 2 * PAD);
      const inv = px => XMIN + (px - PAD) / (W - 2 * PAD) * (XMAX - XMIN);
      const toSvg = e => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const f = x => { if (x <= XMIN || x >= XMAX) return 0; const xs = model.get("xs"), ys = model.get("ys");
        const t = (x - xs[0]) / (xs[xs.length - 1] - xs[0]) * (xs.length - 1); const i = Math.max(0, Math.min(xs.length - 2, Math.floor(t)));
        return ys[i] + (ys[i + 1] - ys[i]) * (t - i); };
      const NB = 40, CURVE_Y = 150, HIST_Y = 310;
      let x = model.get("x0"), samples = [], props = 0, accs = 0, last = null, timer = null;
      const ymax = () => Math.max(...model.get("ys"));
      function draw() {
        svg.innerHTML = "";
        const xs = model.get("xs"), ys = model.get("ys"), ym = ymax();
        const cy = v => CURVE_Y - v / ym * 120;
        mk("path", { d: xs.map((u, i) => (i ? "L" : "M") + sx(u) + "," + cy(ys[i])).join(" "), fill: "rgba(233,162,59,0.2)", stroke: "#e9a23b", "stroke-width": 3 });
        mk("line", { x1: sx(XMIN), x2: sx(XMAX), y1: CURVE_Y, y2: CURVE_Y, stroke: "#888" });
        // histogram of samples, scaled as density
        const counts = Array(NB).fill(0), bw = (XMAX - XMIN) / NB;
        for (const s of samples) { const k = Math.floor((s - XMIN) / bw); if (k >= 0 && k < NB) counts[k]++; }
        const hy = v => HIST_Y - v / ym * 120;
        for (let k = 0; k < NB; k++) { const d = samples.length ? counts[k] / samples.length / bw : 0;
          mk("rect", { x: sx(XMIN + k * bw) + 1, y: hy(d), width: sx(XMIN + bw) - sx(XMIN) - 2, height: HIST_Y - hy(d), fill: "#2a9d8f", opacity: 0.75 }); }
        mk("path", { d: xs.map((u, i) => (i ? "L" : "M") + sx(u) + "," + hy(ys[i])).join(" "), fill: "none", stroke: "#e9a23b", "stroke-width": 2, "stroke-dasharray": "5 4" });
        mk("line", { x1: sx(XMIN), x2: sx(XMAX), y1: HIST_Y, y2: HIST_Y, stroke: "#888" });
        // recent trail
        const tr = samples.slice(-25);
        tr.forEach((s, i) => mk("circle", { cx: sx(s), cy: cy(f(s)), r: 3, fill: "#2a9d8f", opacity: 0.15 + 0.6 * i / tr.length }));
        if (last) {
          mk("line", { x1: sx(last.from), x2: sx(last.y), y1: cy(f(last.from)), y2: cy(f(last.y)), stroke: "#457b9d", "stroke-dasharray": "3 3" });
          mk("circle", { cx: sx(last.y), cy: cy(f(last.y)), r: 6, fill: "#457b9d", opacity: 0.8 });
        }
        mk("circle", { cx: sx(x), cy: cy(f(x)), r: 9, fill: last && !last.ok ? "#d62828" : "#2a9d8f", stroke: "#fff", "stroke-width": 2 });
        const t = mk("text", { x: W - PAD, y: 18, "text-anchor": "end", "font-size": 12, fill: "currentColor" });
        t.textContent = "click anywhere on the curve to teleport the walker";
        const a = last ? Math.min(1, f(last.y) / Math.max(f(last.from), 1e-300)) : null;
        info.textContent = `steps: ${props}   accepted: ${accs} (${props ? (100 * accs / props).toFixed(0) : "-"}%)   x = ${x.toFixed(2)}` +
          (last ? `   last α = min(1, ${f(last.y).toFixed(3)}/${f(last.from).toFixed(3)}) = ${a.toFixed(2)} → ${last.ok ? "ACCEPT" : "reject"}` : "");
      }
      function step() {
        const s = model.get("step");
        const g = Math.sqrt(-2 * Math.log(Math.random() + 1e-12)) * Math.cos(2 * Math.PI * Math.random());
        const y = x + s * g, px = f(x), py = f(y);
        const ok = Math.random() < (px === 0 ? 1 : Math.min(1, py / px));
        last = { from: x, y, ok };
        if (ok) { x = y; accs++; }
        props++; samples.push(x);
        draw();
        if (props % 50 === 0) sync();
      }
      function sync() { model.set("samples", samples.slice(-5000)); model.set("proposed", props); model.set("accepted", accs); model.save_changes(); }
      const btn = (t, fn) => { const b = document.createElement("button"); b.textContent = t; b.onclick = fn; b.style.marginRight = "6px"; ctl.appendChild(b); return b; };
      const play = btn("▶ run", () => { if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ run"; sync(); }
        else { timer = setInterval(step, 60); play.textContent = "⏸ pause"; } });
      btn("step", () => { step(); sync(); });
      btn("+500 fast", () => { for (let i = 0; i < 500; i++) step(); sync(); });
      btn("↺ reset", () => { samples = []; props = 0; accs = 0; last = null; x = model.get("x0"); draw(); sync(); });
      ctl.appendChild(lab);
      rng.value = model.get("step"); rv.textContent = model.get("step").toFixed(1);
      rng.oninput = () => { rv.textContent = (+rng.value).toFixed(1); model.set("step", +rng.value); };
      rng.onchange = () => model.save_changes();
      svg.addEventListener("pointerdown", e => { const p = toSvg(e); if (p.y < CURVE_Y + 5) { x = Math.max(XMIN + 0.01, Math.min(XMAX - 0.01, inv(p.x))); last = null; draw(); } });
      model.on("change:step", () => { rng.value = model.get("step"); rv.textContent = model.get("step").toFixed(1); });
      draw();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        xs = traitlets.List(traitlets.Float()).tag(sync=True)
        ys = traitlets.List(traitlets.Float()).tag(sync=True)
        x0 = traitlets.Float(-4.0).tag(sync=True)
        step = traitlets.Float(1.0).tag(sync=True)
        samples = traitlets.List(traitlets.Float()).tag(sync=True)
        proposed = traitlets.Int(0).tag(sync=True)
        accepted = traitlets.Int(0).tag(sync=True)

    return (MetropolisWalker,)


@app.cell
def walker(MetropolisWalker, PDF_GRID, X_GRID, mo):
    walker = mo.ui.anywidget(MetropolisWalker(xs=X_GRID.tolist(), ys=PDF_GRID.tolist(), x0=-4.0, step=1.0))
    walker
    return (walker,)


@app.cell
def walker_readout(TRUE_MEAN, alt, mo, np, pd, target_df, walker):
    _s = np.array(walker.value["samples"])
    _p, _a = walker.value["proposed"], walker.value["accepted"]
    if len(_s) < 20:
        _out = mo.md("_Run the walker (or press **+500 fast**) to collect samples. This readout updates every 50 steps._")
    else:
        _h, _e = np.histogram(_s, bins=40, range=(-5, 5), density=True)
        _hd = pd.DataFrame({"x": (_e[:-1] + _e[1:]) / 2, "density": _h})
        _bars = alt.Chart(_hd).mark_bar(color="#2a9d8f", opacity=0.7).encode(x=alt.X("x:Q", title="x"), y=alt.Y("density:Q"))
        _curve = alt.Chart(target_df).mark_line(color="#e9a23b", strokeWidth=3).encode(x="x:Q", y="density:Q")
        _out = mo.vstack([
            (_bars + _curve).properties(width="container", height=200),
            mo.hstack([
                mo.stat(f"{_p}", label="proposals"),
                mo.stat(f"{_a / _p:.0%}", label="acceptance rate", caption="sweet spot for 1-D ≈ 40–50%"),
                mo.stat(f"{np.mean(_s > 0):.1%}", label="time in right peak", caption="target 70%"),
                mo.stat(f"{_s.mean():.2f}", label="sample mean", caption=f"true {TRUE_MEAN}"),
            ], justify="start"),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Press **step** about ten times and read the α line. Uphill proposals give α = 1. Downhill proposals are accepted only some of the time.
    2. Set **s = 0.1**: almost every proposal is accepted (~95%), but the walker crawls and stays on one peak. **s = 5**: most proposals land in empty space and get rejected. Somewhere in between works best.
    3. Click on the far left tail to teleport the walker, then hit **+500 fast** a few times. The green histogram still converges to the gold curve.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Burn-in: throw away the warm-up

    *"Those first samples, the red ones, are still finding their way to the high-probability regions... like warming up an
    engine on a cold morning."* Early samples still depend on the arbitrary starting point, so we **discard the first B samples**.
    The trace plot below shows sample value against iteration. Start the chain far away and choose where to cut.
    """)
    return


@app.cell
def burnin_controls(mo):
    bi_start = mo.ui.slider(-30, 30, step=1, value=25, label="starting point x₀", show_value=True)
    bi_step = mo.ui.slider(0.1, 3.0, step=0.1, value=0.5, label="step size", show_value=True)
    bi_cut = mo.ui.slider(0, 500, step=5, value=0, label="burn-in B (discard first B)", show_value=True)
    mo.vstack([mo.hstack([bi_start, bi_step], justify="start"), bi_cut])
    return bi_cut, bi_start, bi_step


@app.cell
def burnin_chart(
    TRUE_MEAN,
    alt,
    bi_cut,
    bi_start,
    bi_step,
    metropolis,
    mo,
    np,
    pd,
    target_df,
    target_pdf,
):
    burn_chain, _ = metropolis(target_pdf, bi_start.value, 1500, bi_step.value, seed=7)
    _df = pd.DataFrame({"iteration": np.arange(len(burn_chain)), "x": burn_chain,
                        "phase": np.where(np.arange(len(burn_chain)) < bi_cut.value, "burn-in (discarded)", "kept")})
    _trace = alt.Chart(_df).mark_line(strokeWidth=1.2).encode(
        x="iteration:Q", y=alt.Y("x:Q", title="sample value"),
        color=alt.Color("phase:N", scale=alt.Scale(domain=["burn-in (discarded)", "kept"], range=["#d62828", "#2a9d8f"])),
        detail="phase:N")
    _rule = alt.Chart(pd.DataFrame({"b": [bi_cut.value]})).mark_rule(color="#e9a23b", strokeWidth=3).encode(x="b:Q")
    _kept = burn_chain[bi_cut.value:]
    _h, _e = np.histogram(_kept, bins=40, range=(-5, 5), density=True)
    _hist = alt.Chart(pd.DataFrame({"x": (_e[:-1] + _e[1:]) / 2, "density": _h})).mark_bar(color="#2a9d8f", opacity=0.7).encode(x="x:Q", y="density:Q")
    _curve = alt.Chart(target_df).mark_line(color="#e9a23b", strokeWidth=3).encode(x="x:Q", y="density:Q")
    mo.vstack([
        alt.vconcat((_trace + _rule).properties(width=620, height=200, title="trace plot"),
                    (_hist + _curve).properties(width=620, height=160, title="histogram of kept samples vs target")),
        mo.hstack([
            mo.stat(f"{_kept.mean():.2f}", label="mean of kept samples", caption=f"true {TRUE_MEAN}"),
            mo.stat(f"{burn_chain.mean():.2f}", label="mean with NO burn-in"),
            mo.stat(f"{np.mean(_kept > 0):.1%}", label="kept: time in right peak", caption="target 70%"),
        ], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. With x₀ = 25 and B = 0, the long red slide down from 25 drags the mean way above 0.8.
    2. Increase B until the gold line sits just past the point where the trace "settles". The kept mean snaps toward 0.8.
    3. Raise the step size to 2.5. The chain reaches the peaks much faster, so a smaller burn-in is enough.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Why it works: detailed balance

    *"For any two states X and Y, the flow of probability from X to Y, $P(X)\,T(Y\mid X)$, must exactly equal the flow from Y back to X, $P(Y)\,T(X\mid Y)$."*

    $$\underbrace{p(x)\,T(y\mid x)}_{\text{flow } x\to y} \;=\; \underbrace{p(y)\,T(x\mid y)}_{\text{flow } y\to x}\quad\text{for all } x,y \;\;\Longrightarrow\;\; p \text{ is stationary.}$$

    (Sum both sides over $x$: $\sum_x p(x)T(y\mid x) = p(y)$.) Metropolis gets this for free: with a symmetric proposal,
    $p(x)\min\!\big(1,\tfrac{p(y)}{p(x)}\big) = \min(p(x),p(y))$, which is symmetric in $x,y$.

    Below is a **6-state ring**. Set the target weights, pick an acceptance rule, and check the flows.
    """)
    return


@app.cell
def db_controls(mo):
    db_weights = mo.ui.array([mo.ui.slider(1, 10, value=_v, label=f"p({_i}) weight", show_value=True) for _i, _v in enumerate([1, 3, 8, 4, 2, 6])])
    db_rule = mo.ui.dropdown(["Metropolis: α = min(1, p(y)/p(x))", "Accept everything: α = 1", "Greedy: accept only uphill"],
                             value="Metropolis: α = min(1, p(y)/p(x))", label="acceptance rule")
    mo.hstack([db_weights, db_rule], justify="start", gap=2)
    return db_rule, db_weights


@app.cell
def db_math(db_rule, db_weights, np):
    def ring_metropolis_T(p, rule):
        """Proposal: left or right neighbour with prob 1/2 each (symmetric). Rejected mass stays on the diagonal."""
        _k = len(p); _T = np.zeros((_k, _k))
        for _x in range(_k):
            for _y in ((_x - 1) % _k, (_x + 1) % _k):
                if rule.startswith("Metropolis"):
                    _a = min(1.0, p[_y] / p[_x])
                elif rule.startswith("Accept"):
                    _a = 1.0
                else:
                    _a = 1.0 if p[_y] >= p[_x] else 0.0
                _T[_x, _y] += 0.5 * _a
            _T[_x, _x] = 1 - _T[_x].sum()
        return _T

    db_p = np.array(db_weights.value, dtype=float); db_p /= db_p.sum()
    db_T = ring_metropolis_T(db_p, db_rule.value)
    db_flow = db_p[:, None] * db_T            # flow[x, y] = p(x) T(y|x)
    db_imbalance = np.abs(db_flow - db_flow.T).max()
    _P = np.linalg.matrix_power(db_T, 2000)
    db_pi = 0.5 * (_P + _P @ db_T)[0]   # long-run time share from state 0 (averaging 2 steps handles period-2 chains)
    return db_flow, db_imbalance, db_p, db_pi


@app.cell
def db_chart(alt, db_flow, db_imbalance, db_p, db_pi, mo, np, pd):
    _k = len(db_p)
    _pairs = pd.DataFrame([{"pair": f"{_x}↔{(_x + 1) % _k}", "direction": _d, "flow": _f}
        for _x in range(_k)
        for _d, _f in ((f"forward", db_flow[_x, (_x + 1) % _k]), ("backward", db_flow[(_x + 1) % _k, _x]))])
    _flows = alt.Chart(_pairs).mark_bar().encode(
        x=alt.X("direction:N", title=None, axis=alt.Axis(labels=False, ticks=False)), y=alt.Y("flow:Q", title="probability flow"),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["forward", "backward"], range=["#457b9d", "#e9a23b"])),
        column=alt.Column("pair:N", title="neighbouring pair: forward p(x)T(y|x) vs backward p(y)T(x|y)")).properties(width=70, height=160)
    _dist = pd.DataFrame({"state": np.tile(np.arange(_k), 2), "prob": np.r_[db_p, db_pi],
                          "which": ["target p"] * _k + ["chain's long-run π"] * _k})
    _cmp = alt.Chart(_dist).mark_bar().encode(
        x=alt.X("which:N", title=None, axis=alt.Axis(labels=False, ticks=False)), y="prob:Q",
        color=alt.Color("which:N", scale=alt.Scale(domain=["target p", "chain's long-run π"], range=["#e9a23b", "#2a9d8f"])),
        column=alt.Column("state:N", title="target p vs chain's long-run π")).properties(width=70, height=160)
    _ok = db_imbalance < 1e-12
    mo.vstack([
        mo.callout(mo.md(f"**Max flow imbalance |p(x)T(y|x) − p(y)T(x|y)| = {db_imbalance:.2e}**: "
                         + ("detailed balance holds ✅" if _ok else "detailed balance is violated ❌")), kind="success" if _ok else "danger"),
        _flows, _cmp,
        mo.md(f"Total variation between target and long-run π: **{0.5 * np.abs(db_p - db_pi).sum():.4f}**"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Under Metropolis, move any weight slider. Forward and backward bars always match, and the green π equals gold p.
    2. Switch to **Accept everything**. The chain becomes a plain random walk: π goes **uniform** (1/6 each) and ignores the target.
    3. Switch to **Greedy**. The chain climbs and gets stuck on a local peak, so π collapses onto one state. The occasional downhill acceptance is essential.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Sandbox: sample any landscape

    Pick a target (none of them normalized; MCMC doesn't care), tune the sampler, and see whether you can recover the shape.
    """)
    return


@app.cell
def sandbox_controls(mo, np, target_pdf):
    SANDBOX_TARGETS = {
        "Two peaks 30/70 (video)": lambda x: target_pdf(x),
        "Three peaks": lambda x: np.exp(-0.5 * (x + 3) ** 2 / 0.25) + 2 * np.exp(-0.5 * x ** 2 / 0.25) + np.exp(-0.5 * (x - 3) ** 2 / 0.25),
        "Far-apart islands": lambda x: np.exp(-0.5 * (x + 4) ** 2 / 0.09) + np.exp(-0.5 * (x - 4) ** 2 / 0.09),
        "Heavy tail (Cauchy)": lambda x: 1 / (1 + x ** 2),
        "Skewed (gamma-like)": lambda x: np.where(x > -5, (x + 5) ** 2 * np.exp(-(x + 5)), 0.0),
    }
    sb_target = mo.ui.dropdown(list(SANDBOX_TARGETS), value="Three peaks", label="target")
    sb_step = mo.ui.slider(0.05, 8.0, step=0.05, value=1.0, label="step size", show_value=True)
    sb_n = mo.ui.slider(500, 30000, step=500, value=5000, label="iterations", show_value=True)
    sb_burn = mo.ui.slider(0, 5000, step=100, value=500, label="burn-in", show_value=True)
    sb_x0 = mo.ui.slider(-8, 8, step=0.5, value=0, label="start x₀", show_value=True)
    sb_seed = mo.ui.number(0, 999, value=1, label="seed")
    mo.vstack([mo.hstack([sb_target, sb_seed], justify="start"), mo.hstack([sb_step, sb_n], justify="start"), mo.hstack([sb_burn, sb_x0], justify="start")])
    return SANDBOX_TARGETS, sb_burn, sb_n, sb_seed, sb_step, sb_target, sb_x0


@app.cell
def sandbox_chart(
    SANDBOX_TARGETS,
    alt,
    metropolis,
    mo,
    np,
    pd,
    sb_burn,
    sb_n,
    sb_seed,
    sb_step,
    sb_target,
    sb_x0,
):
    _f = SANDBOX_TARGETS[sb_target.value]
    _chain, _acc = metropolis(_f, sb_x0.value, sb_n.value, sb_step.value, seed=int(sb_seed.value))
    _kept = _chain[min(sb_burn.value, len(_chain) - 1):]
    _xg = np.linspace(-8, 8, 400); _yg = _f(_xg); _yg = _yg / np.trapezoid(_yg, _xg)
    _h, _e = np.histogram(_kept, bins=80, range=(-8, 8)); _h = _h / (len(_kept) * (_e[1] - _e[0]))
    _hist = alt.Chart(pd.DataFrame({"x": (_e[:-1] + _e[1:]) / 2, "density": _h})).mark_bar(color="#2a9d8f", opacity=0.7).encode(x=alt.X("x:Q", scale=alt.Scale(domain=[-8, 8])), y="density:Q")
    _curve = alt.Chart(pd.DataFrame({"x": _xg, "density": _yg})).mark_line(color="#e9a23b", strokeWidth=3).encode(x="x:Q", y="density:Q")
    _stride = max(1, len(_chain) // 1500)
    _tdf = pd.DataFrame({"iteration": np.arange(len(_chain)), "x": _chain})[::_stride]
    _tdf["phase"] = np.where(_tdf.iteration < sb_burn.value, "burn-in", "kept")
    _trace = alt.Chart(_tdf).mark_line(strokeWidth=1).encode(x="iteration:Q", y=alt.Y("x:Q", scale=alt.Scale(domain=[-8, 8], clamp=True)),
        color=alt.Color("phase:N", scale=alt.Scale(domain=["burn-in", "kept"], range=["#d62828", "#2a9d8f"])), detail="phase:N")
    _ac1 = np.corrcoef(_kept[:-1], _kept[1:])[0, 1] if len(_kept) > 2 and _kept.std() > 0 else 1.0
    _ess = len(_kept) * (1 - _ac1) / (1 + _ac1) if _ac1 < 1 else 0
    mo.vstack([
        alt.vconcat((_hist + _curve).properties(width=620, height=200), _trace.properties(width=620, height=150)),
        mo.hstack([
            mo.stat(f"{_acc.mean():.0%}", label="acceptance rate"),
            mo.stat(f"{_ac1:.2f}", label="lag-1 autocorrelation", caption="dependence cost"),
            mo.stat(f"{_ess:,.0f}", label="≈ effective sample size", caption=f"of {len(_kept):,} kept"),
        ], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Challenges**

    - **Far-apart islands**: with step 0.5 the chain never leaves its island. Find a step size that visits both islands in a ~50/50 split.
    - **Heavy tail**: get the acceptance rate between 30% and 50% and watch the occasional excursions far out.
    - Find the step size that **maximizes effective sample size** for "Three peaks". That's the price of dependent samples: 10,000 correlated draws can be worth only a few hundred independent ones.
    """), kind="neutral")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 9 · Quiz: check your intuition
    """)
    return


@app.cell
def quiz(mo):
    QUIZ = {
        "q1": ("Why does accept-reject sampling break down in high dimensions?",
               ["The bounding box's volume dwarfs the target's, so acceptance collapses", "Random numbers become less random", "Rejected samples become correlated"], 0),
        "q2": ("In a Markov chain, x₍ₙ₊₁₎ depends on…",
               ["all previous samples", "only the current state xₙ", "only the starting point x₀"], 1),
        "q3": ("A Metropolis proposal has p(y)/p(x) = 0.25. The move is accepted…",
               ["never", "with probability 0.25", "always"], 1),
        "q4": ("Why discard burn-in samples?",
               ["They are correlated with each other", "They still reflect the arbitrary starting point, not the target", "They have low acceptance"], 1),
        "q5": ("Which condition guarantees p is the chain's stationary distribution?",
               ["p(x)T(y|x) = p(y)T(x|y) for all x, y", "T(y|x) = T(x|y)", "acceptance rate = 50%"], 0),
        "q6": ("Why can MCMC sample a Bayesian posterior without its normalizing constant?",
               ["It only uses ratios p(y)/p(x)", "It estimates the constant by burn-in", "Posteriors are always normalized"], 0),
    }
    quiz = mo.ui.dictionary({_k: mo.ui.radio(_opts, label=f"**{_k[1:]}.** {_q}") for _k, (_q, _opts, _a) in QUIZ.items()})
    quiz_reveal = mo.ui.switch(label="Reveal answers")
    mo.vstack([quiz, quiz_reveal])
    return QUIZ, quiz, quiz_reveal


@app.cell
def quiz_score(QUIZ, mo, quiz, quiz_reveal):
    _score = sum(quiz.value[_k] == _opts[_a] for _k, (_q, _opts, _a) in QUIZ.items())
    _answered = sum(_v is not None for _v in quiz.value.values())
    _lines = []
    if quiz_reveal.value:
        for _k, (_q, _opts, _a) in QUIZ.items():
            _mark = "✅" if quiz.value[_k] == _opts[_a] else "❌"
            _lines.append(f"- {_mark} **{_k[1:]}.** {_opts[_a]}")
    mo.vstack([
        mo.stat(f"{_score} / {len(QUIZ)}", label="score", caption=f"{_answered} answered"),
        mo.md("\n".join(_lines)) if _lines else mo.md("_Flip the switch to see the correct answers._"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 · Recap

    | Idea (video chapter) | One-liner | Formula | Where you touched it |
    |---|---|---|---|
    | Accept-reject sampling | Darts in a box: simple but wasteful, and hopeless in high-d | rate = area(p) / area(box) | Dart box + curse-of-dimensionality chart |
    | Key insight / Markov chain | Next sample depends only on the current one | $x_{n+1}\sim T(\cdot\mid x_n)$ | Ring walker |
    | Monte Carlo | Replace integrals by sample averages | $\mathbb E[f]\approx\frac1N\sum f(x_n)$, error $\propto 1/\sqrt N$ | Running-mean chart |
    | Stationary distribution | Long-run time shares are fixed, whatever the start | $\pi T = \pi$ | Three chains → 70% |
    | MCMC in action (Metropolis) | Propose, then accept with the density ratio | $\alpha=\min(1, p(y)/p(x))$ | Metropolis walker |
    | Burn-in | Drop the warm-up that remembers $x_0$ | keep $x_{B+1},\dots$ | Trace plot |
    | Detailed balance | Flow x→y equals flow y→x ⇒ p is stationary | $p(x)T(y\mid x)=p(y)T(x\mid y)$ | 6-state flow check |
    | The trade-off | Dependent samples are cheap to get but worth less each | ESS < N | Sandbox |
    """)
    return


if __name__ == "__main__":
    app.run()

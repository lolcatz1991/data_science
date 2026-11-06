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
    from scipy import stats as scipy_stats

    return alt, anywidget, mo, np, pd, scipy_stats, traitlets


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
    # Bootstrap resampling: one sample, many universes
    *An interactive companion to DataMListic's video **"Bootstrap Resampling - Explained"**.*
    """),
        mo.Html('<iframe width="560" height="315" src="https://www.youtube.com/embed/41ST4JdMrpA" title="Bootstrap Resampling - Explained" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'),
        mo.md(r"""
    > You measured **30 trees** in a forest and the average came out to **14.2 m**. How sure are you about that number?

    The bootstrap answers this using nothing but the sample you already have.
    We follow the video's chapters: **one sample** → **the imagined multiverse** → **the bootstrap trick** →
    **the bootstrap distribution** → **standard error & CI** → **why it works**.

    **Color language used throughout:**
    <span style="color:#e4572e">■ red = the truth (population, true mean μ)</span> ·
    <span style="color:#6c5b7b">■ purple = your one sample</span> ·
    <span style="color:#2e86ab">■ blue = the true sampling distribution (other universes)</span> ·
    <span style="color:#f18f01">■ orange = bootstrap resamples</span>
    """),
    ])
    return


@app.cell
def population():
    # The "forest": a population of tree heights we normally never see (gamma, mean 14 m, sd 4 m)
    MU_TRUE, SD_TRUE = 14.0, 4.0
    _shape = (MU_TRUE / SD_TRUE) ** 2
    POP_SHAPE, POP_SCALE = _shape, MU_TRUE / _shape

    def draw_trees(n, rng):
        """Measure n random trees from the forest."""
        return rng.gamma(POP_SHAPE, POP_SCALE, n)

    COLORS = {"truth": "#e4572e", "sample": "#6c5b7b", "multiverse": "#2e86ab", "boot": "#f18f01"}
    return COLORS, MU_TRUE, POP_SCALE, POP_SHAPE, SD_TRUE, draw_trees


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · One sample, one number

    You walk into the forest, measure $n = 30$ trees and compute the sample mean

    $$\bar X = \frac{1}{n}\sum_{i=1}^{n} X_i .$$

    That one number is *all you know*. If you came back tomorrow and measured 30 **different** trees, you would not get the
    same number again. **The number would wiggle. The real question is: by how much?**

    Press the button to go back to the forest and measure another 30 trees.
    """)
    return


@app.cell
def remeasure_btn(mo):
    remeasure = mo.ui.button(value=0, on_click=lambda v: v + 1, label="🌲 Measure another 30 trees")
    remeasure
    return (remeasure,)


@app.cell
def one_sample(COLORS, MU_TRUE, alt, draw_trees, mo, np, pd, remeasure):
    _rng = np.random.default_rng(2024 + remeasure.value)
    day_sample = draw_trees(30, _rng)
    _df = pd.DataFrame({"height": day_sample, "y": _rng.uniform(0, 1, 30)})
    _pts = alt.Chart(_df).mark_circle(size=90, color=COLORS["sample"], opacity=0.8).encode(
        x=alt.X("height:Q", scale=alt.Scale(domain=[0, 30]), title="tree height (m)"),
        y=alt.Y("y:Q", axis=None))
    _mean = alt.Chart(pd.DataFrame({"m": [day_sample.mean()]})).mark_rule(color=COLORS["sample"], size=3).encode(x="m:Q")
    _truth = alt.Chart(pd.DataFrame({"m": [MU_TRUE]})).mark_rule(color=COLORS["truth"], strokeDash=[6, 4], size=2).encode(x="m:Q")
    mo.vstack([
        mo.hstack([
            mo.stat(f"{day_sample.mean():.2f} m", label="your sample mean X̄", caption=f"day #{remeasure.value + 1}"),
            mo.stat(f"{MU_TRUE:.1f} m", label="true mean μ (secret!)", caption="you never get to see this"),
            mo.stat(f"{day_sample.mean() - MU_TRUE:+.2f} m", label="error this time"),
        ]),
        (_pts + _mean + _truth).properties(width="container", height=140, title="Your 30 trees (purple line = X̄, dashed red = μ)"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - Click the button 5–10 times and write down the sample means. How far do they stray from 14?
    - Notice: in real life you only ever get **one** click. The wiggle is invisible to you.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · The imagined multiverse

    Imagine repeating the whole experiment in **parallel universes**. Each universe collects its own 30 trees and hands you
    a slightly different $\bar X$. Put all those means on a number line and they form a little **cloud** around the true
    average $\mu$. That cloud is the **sampling distribution** of $\bar X$, *the dream object*.
    Its width is exactly the uncertainty you want, the **standard error**:

    $$\operatorname{SE}(\bar X) = \frac{\sigma}{\sqrt{n}}$$

    Press **▶ play** and watch universes appear. Each one measures $n$ trees (small dots), computes its mean, and drops it onto the pile.
    """)
    return


@app.cell
def multiverse_class(anywidget, traitlets):
    class MultiverseSim(anywidget.AnyWidget):
        """Animated parallel universes: each draws n trees from the gamma forest and stacks its mean."""
        _esm = r"""
    function render({ model, el }) {
      const W = 700, H = 320;
      const ctl = document.createElement("div");
      Object.assign(ctl.style, { display: "flex", gap: "6px", alignItems: "center", font: "13px system-ui, sans-serif", marginBottom: "6px" });
      el.appendChild(ctl);
      const canvas = document.createElement("canvas"); canvas.width = W * 2; canvas.height = H * 2;
      Object.assign(canvas.style, { width: "100%", maxWidth: W + "px", borderRadius: "8px", background: "rgba(127,127,127,0.06)" });
      el.appendChild(canvas);
      const ctx = canvas.getContext("2d"); ctx.scale(2, 2);
      const btn = (t, fn) => { const b = document.createElement("button"); b.textContent = t; b.onclick = fn;
        Object.assign(b.style, { padding: "3px 10px", borderRadius: "12px", border: "1px solid #bbb", cursor: "pointer" }); ctl.appendChild(b); return b; };
      const label = document.createElement("span"); label.style.marginLeft = "8px";

      // gamma sampler (Marsaglia-Tsang, shape >= 1)
      const randn = () => { let u = 0, v = 0; while (!u) u = Math.random(); while (!v) v = Math.random();
        return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
      const rgamma = (k, th) => { const d = k - 1 / 3, c = 1 / Math.sqrt(9 * d);
        while (true) { let x, v; do { x = randn(); v = 1 + c * x; } while (v <= 0); v = v * v * v;
          const u = Math.random(); if (Math.log(u) < 0.5 * x * x + d - d * v + d * Math.log(v)) return d * v * th; } };

      const XMIN = 8, XMAX = 20, NB = 60, BASE = H - 30;
      const sx = v => 30 + (v - XMIN) / (XMAX - XMIN) * (W - 60);
      let means = [], last = [], timer = null;
      function universe() {
        const n = model.get("n"), k = model.get("shape"), th = model.get("scale");
        last = Array.from({ length: n }, () => rgamma(k, th));
        means.push(last.reduce((a, b) => a + b, 0) / n);
      }
      function sync() { model.set("means", means.slice(-3000)); model.save_changes(); }
      function draw() {
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px system-ui"; ctx.fillStyle = "#888";
        ctx.fillText("latest universe: its trees (top) and the mean it contributes", 30, 16);
        // latest trees (axis 0..30 on top strip)
        const tx = v => 30 + v / 30 * (W - 60);
        ctx.fillStyle = "rgba(108,91,123,0.6)";
        last.forEach(v => { ctx.beginPath(); ctx.arc(tx(v), 40 + Math.random() * 20, 3, 0, 7); ctx.fill(); });
        if (means.length) { const m = means[means.length - 1]; ctx.strokeStyle = "#6c5b7b"; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.moveTo(tx(m), 30); ctx.lineTo(tx(m), 70); ctx.stroke(); }
        // histogram of means
        const counts = new Array(NB).fill(0), bw = (XMAX - XMIN) / NB;
        means.forEach(m => { const b = Math.floor((m - XMIN) / bw); if (b >= 0 && b < NB) counts[b]++; });
        const mx = Math.max(10, ...counts), ph = BASE - 95;
        ctx.fillStyle = "rgba(46,134,171,0.75)";
        counts.forEach((c, b) => { const h = c / mx * ph; ctx.fillRect(sx(XMIN + b * bw) + 1, BASE - h, (W - 60) / NB - 2, h); });
        // axis
        ctx.strokeStyle = "#999"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(30, BASE); ctx.lineTo(W - 30, BASE); ctx.stroke();
        ctx.fillStyle = "#888";
        for (let v = XMIN; v <= XMAX; v += 2) ctx.fillText(v, sx(v) - 6, BASE + 16);
        // true mean
        const mu = model.get("shape") * model.get("scale");
        ctx.strokeStyle = "#e4572e"; ctx.setLineDash([6, 4]); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(sx(mu), 85); ctx.lineTo(sx(mu), BASE); ctx.stroke(); ctx.setLineDash([]);
        let sd = 0; if (means.length > 1) { const a = means.reduce((x, y) => x + y, 0) / means.length;
          sd = Math.sqrt(means.reduce((x, y) => x + (y - a) ** 2, 0) / (means.length - 1)); }
        label.textContent = `universes: ${means.length} · width of the cloud (sd of means): ${sd.toFixed(3)} m`;
      }
      const play = btn("▶ play", () => {
        if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; sync(); }
        else { timer = setInterval(() => { universe(); draw(); if (means.length % 20 === 0) sync(); }, 60); play.textContent = "⏸ pause"; } });
      btn("+1 universe", () => { universe(); draw(); sync(); });
      btn("⏩ +1000", () => { for (let i = 0; i < 1000; i++) universe(); draw(); sync(); });
      btn("↺ reset", () => { means = []; last = []; draw(); sync(); });
      ctl.appendChild(label);
      model.on("change:n", () => { means = []; last = []; draw(); sync(); });
      draw();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        n = traitlets.Int(30).tag(sync=True)
        shape = traitlets.Float(12.25).tag(sync=True)
        scale = traitlets.Float(14 / 12.25).tag(sync=True)
        means = traitlets.List(traitlets.Float()).tag(sync=True)

    return (MultiverseSim,)


@app.cell
def multiverse_n_slider(mo):
    multiverse_n = mo.ui.slider(5, 120, value=30, step=5, label="trees per universe (n)", show_value=True)
    multiverse_n
    return (multiverse_n,)


@app.cell
def multiverse_widget(MultiverseSim, POP_SCALE, POP_SHAPE, mo, multiverse_n):
    multiverse = mo.ui.anywidget(MultiverseSim(n=multiverse_n.value, shape=POP_SHAPE, scale=POP_SCALE))
    multiverse
    return (multiverse,)


@app.cell
def multiverse_readout(MU_TRUE, SD_TRUE, mo, multiverse, multiverse_n, np):
    _m = np.array(multiverse.value["means"])
    _n = multiverse_n.value
    _theory = SD_TRUE / np.sqrt(_n)
    mo.hstack([
        mo.stat(f"{len(_m)}", label="universes simulated"),
        mo.stat(f"{_m.mean():.3f} m" if len(_m) else "–", label="center of the cloud", caption=f"true μ = {MU_TRUE}"),
        mo.stat(f"{_m.std(ddof=1):.3f} m" if len(_m) > 1 else "–", label="observed SE (sd of X̄'s)"),
        mo.stat(f"{_theory:.3f} m", label="theory σ/√n", caption=f"σ = {SD_TRUE}, n = {_n}"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - Press **⏩ +1000**. The blue cloud is centered on the red μ, and its width matches $\sigma/\sqrt{n}$.
    - Change **n** to 120: the cloud gets **half** as wide as at n = 30 (√4 = 2). Change it to 5: wide and slightly skewed.
    - **The catch:** you only get **one** universe. This whole picture is forever out of reach… unless you use the bootstrap.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · The bootstrap trick

    *The magical idea:* take your one sample and **treat it as if it were the entire population**. Then draw a new sample
    from it, the **same size** $n$, but **with replacement**. Some trees get picked twice, some get skipped, some show up three times.

    Formally you draw $X^*_1, \dots, X^*_n$ from the **empirical distribution** $\hat F_n$, which is a fancy way of saying
    *pick uniformly from your original sample, with replacement*. Each resample gives a new mean $\bar X^*_b$.

    Below are your 30 trees as cards (height in m). Press **🎲 resample**: the badge shows how many times each tree was picked.
    Grey cards were skipped this time.
    """)
    return


@app.cell
def resample_class(anywidget, traitlets):
    class ResampleCards(anywidget.AnyWidget):
        """Shows the original sample as cards; 'resample' draws n indices with replacement and shows pick counts."""
        _esm = r"""
    function render({ model, el }) {
      const ctl = document.createElement("div");
      Object.assign(ctl.style, { display: "flex", gap: "6px", alignItems: "center", font: "13px system-ui, sans-serif", marginBottom: "8px", flexWrap: "wrap" });
      const grid = document.createElement("div");
      Object.assign(grid.style, { display: "grid", gridTemplateColumns: "repeat(10, 1fr)", gap: "6px", maxWidth: "700px" });
      const cap = document.createElement("div"); Object.assign(cap.style, { font: "13px system-ui", marginTop: "8px" });
      el.append(ctl, grid, cap);
      const btn = (t, fn) => { const b = document.createElement("button"); b.textContent = t; b.onclick = fn;
        Object.assign(b.style, { padding: "3px 10px", borderRadius: "12px", border: "1px solid #bbb", cursor: "pointer" }); ctl.appendChild(b); return b; };
      function resample(times) {
        const xs = model.get("sample"), n = xs.length, hist = [...model.get("boot_means")];
        let counts = null;
        for (let t = 0; t < times; t++) {
          counts = new Array(n).fill(0); let s = 0;
          for (let i = 0; i < n; i++) { const j = Math.floor(Math.random() * n); counts[j]++; s += xs[j]; }
          hist.push(s / n);
        }
        model.set("counts", counts); model.set("boot_means", hist); model.save_changes();
      }
      btn("🎲 resample", () => resample(1));
      btn("🎲 ×100", () => resample(100));
      btn("↺ reset", () => { model.set("counts", []); model.set("boot_means", []); model.save_changes(); });
      function draw() {
        const xs = model.get("sample"), counts = model.get("counts"), hist = model.get("boot_means");
        grid.innerHTML = "";
        xs.forEach((v, i) => {
          const c = counts.length ? counts[i] : 1;
          const d = document.createElement("div");
          const bg = !counts.length ? "rgba(108,91,123,0.18)" : c === 0 ? "rgba(127,127,127,0.12)" : `rgba(241,143,1,${0.25 + 0.2 * c})`;
          Object.assign(d.style, { position: "relative", padding: "10px 0", textAlign: "center", borderRadius: "8px", background: bg,
            font: "13px system-ui", opacity: c === 0 ? 0.45 : 1, border: c === 0 ? "1px dashed #999" : "1px solid transparent" });
          d.textContent = v.toFixed(1);
          if (counts.length && c > 1) { const b = document.createElement("span"); b.textContent = "×" + c;
            Object.assign(b.style, { position: "absolute", top: "-6px", right: "-4px", background: "#f18f01", color: "white",
              borderRadius: "10px", padding: "0 5px", fontSize: "11px", fontWeight: "bold" }); d.appendChild(b); }
          grid.appendChild(d);
        });
        if (!counts.length) { cap.innerHTML = "Original sample, X̄ = <b>" + (xs.reduce((a, b) => a + b, 0) / xs.length).toFixed(2) + "</b>. Press resample."; return; }
        const skipped = counts.filter(c => c === 0).length, multi = counts.filter(c => c > 1).length;
        cap.innerHTML = `Resample #${hist.length}: <b>${skipped}</b> trees skipped, <b>${multi}</b> picked more than once · ` +
          `X̄* = <b style="color:#f18f01">${hist[hist.length - 1].toFixed(2)}</b>`;
      }
      model.on("change:counts", draw); model.on("change:sample", draw); draw();
    }
    export default { render };
    """
        sample = traitlets.List(traitlets.Float()).tag(sync=True)
        counts = traitlets.List(traitlets.Int()).tag(sync=True)
        boot_means = traitlets.List(traitlets.Float()).tag(sync=True)

    return (ResampleCards,)


@app.cell
def cards_widget(ResampleCards, draw_trees, mo, np):
    # Our ONE sample for the rest of the lesson (fixed seed so the story is reproducible)
    my_sample = draw_trees(30, np.random.default_rng(7))
    cards = mo.ui.anywidget(ResampleCards(sample=[round(float(v), 1) for v in my_sample]))
    cards
    return cards, my_sample


@app.cell
def cards_readout(COLORS, alt, cards, mo, np, pd):
    _bm = np.array(cards.value["boot_means"])
    _xs = np.array(cards.value["sample"])
    _n = len(_xs)
    _df = pd.DataFrame({"mean": _bm})
    _base = alt.Chart(_df).mark_bar(color=COLORS["boot"], opacity=0.8).encode(
        x=alt.X("mean:Q", bin=alt.Bin(extent=[10, 18], step=0.2), title="bootstrap mean X̄*"),
        y=alt.Y("count():Q", title="resamples"))
    _rule = alt.Chart(pd.DataFrame({"m": [_xs.mean()]})).mark_rule(color=COLORS["sample"], size=3).encode(x="m:Q")
    mo.vstack([
        mo.hstack([
            mo.stat(f"{len(_bm)}", label="resamples so far (B)"),
            mo.stat(f"{(1 - 1/_n) ** _n:.1%}", label="P(a given tree is skipped)", caption="(1 − 1/n)ⁿ → 1/e ≈ 36.8%"),
            mo.stat(f"{_bm.std(ddof=1):.3f} m" if len(_bm) > 1 else "–", label="spread of X̄*"),
        ]),
        (_base + _rule).properties(width="container", height=180, title="Your pile of bootstrap means (purple = original X̄)")
        if len(_bm) else mo.md("_Resample to start building the pile._"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - Resample a few times and count the grey cards. About **11 of 30** (≈ 36.8%) are skipped each time.
    - Why **with** replacement? Without it you would just shuffle the same 30 trees, and $\bar X^*$ would always equal $\bar X$. No wiggle!
    - Press **×100** a few times and watch the orange pile take shape. That pile is the next section.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · The bootstrap distribution

    Now do it not once but **B** times: $\bar X^*_1, \bar X^*_2, \dots, \bar X^*_B$. Stack them into a histogram and something
    remarkable happens: this histogram, built **entirely from your single sample**, has almost exactly the same **shape and
    width** as the true sampling distribution you could never see.

    Below we cheat (we are the simulation's god): **blue** is the real multiverse (5,000 universes),
    **orange** is the bootstrap from *one* purple sample. Note the bootstrap is centered at your $\bar X$, not at $\mu$.
    The bootstrap gets the **wiggle** right; it can't fix where your sample happened to land.
    """)
    return


@app.cell
def boot_controls(mo):
    boot_n = mo.ui.slider(5, 200, value=30, step=5, label="sample size n", show_value=True)
    boot_B = mo.ui.slider(10, 5000, value=1000, step=10, label="bootstrap resamples B", show_value=True)
    boot_center = mo.ui.switch(value=False, label="center both at 0 (compare shape & width only)")
    boot_new = mo.ui.button(value=0, on_click=lambda v: v + 1, label="🌲 new original sample")
    mo.vstack([mo.hstack([boot_n, boot_B]), mo.hstack([boot_center, boot_new], justify="start")])
    return boot_B, boot_center, boot_n, boot_new


@app.cell
def boot_dist(
    COLORS,
    MU_TRUE,
    alt,
    boot_B,
    boot_center,
    boot_n,
    boot_new,
    draw_trees,
    np,
    pd,
):
    def bootstrap(x, stat=np.mean, B=1000, rng=None):
        """Return B bootstrap replicates of stat(x): resample x with replacement, same size."""
        rng = rng or np.random.default_rng()
        _idx = rng.integers(0, len(x), size=(B, len(x)))
        return stat(x[_idx], axis=1)

    _rng = np.random.default_rng(100 + boot_new.value)
    boot_sample = draw_trees(boot_n.value, _rng)
    boot_means = bootstrap(boot_sample, np.mean, boot_B.value, _rng)
    true_means = draw_trees((5000, boot_n.value), np.random.default_rng(1)).mean(axis=1)

    _shift_b = boot_sample.mean() if boot_center.value else 0.0
    _shift_t = MU_TRUE if boot_center.value else 0.0
    _df = pd.concat([
        pd.DataFrame({"v": true_means - _shift_t, "kind": "true sampling distribution (5,000 universes)"}),
        pd.DataFrame({"v": boot_means - _shift_b, "kind": f"bootstrap distribution (B = {boot_B.value})"}),
    ])
    _dens = alt.Chart(_df).transform_density("v", groupby=["kind"], as_=["v", "density"]).mark_area(opacity=0.45, interpolate="monotone").encode(
        x=alt.X("v:Q", title="mean − center" if boot_center.value else "mean height (m)"),
        y=alt.Y("density:Q"),
        color=alt.Color("kind:N", scale=alt.Scale(range=[COLORS["boot"], COLORS["multiverse"]]), legend=alt.Legend(orient="top", title=None)))
    _rules = alt.Chart(pd.DataFrame({"m": [0.0 if boot_center.value else MU_TRUE, 0.0 if boot_center.value else boot_sample.mean()],
                                     "c": [COLORS["truth"], COLORS["sample"]]})).mark_rule(size=2, strokeDash=[6, 3]).encode(
        x="m:Q", color=alt.Color("c:N", scale=None))
    boot_chart = (_dens + _rules).properties(width="container", height=260)
    boot_chart
    return boot_means, boot_sample, bootstrap, true_means


@app.cell
def boot_readout(MU_TRUE, boot_means, boot_sample, mo, true_means):
    mo.hstack([
        mo.stat(f"{boot_sample.mean():.2f} m", label="your X̄ (purple)", caption=f"μ = {MU_TRUE}"),
        mo.stat(f"{true_means.std(ddof=1):.3f}", label="true SE (blue width)"),
        mo.stat(f"{boot_means.std(ddof=1):.3f}", label="bootstrap SE (orange width)"),
        mo.stat(f"{boot_means.std(ddof=1) / true_means.std(ddof=1):.2f}", label="ratio boot / true", caption="≈ 1 means it worked"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - Flip **center both** on. The two humps nearly coincide: same shape, same width.
    - Click **new original sample** several times. The orange hump *slides around* (it follows your $\bar X$), but its **width** stays close to blue.
    - Set **n = 5**: the bootstrap gets noisier (5 trees is a tiny window on the forest). Set **B = 10**: a ragged histogram. B is cheap, so use thousands.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Standard error & confidence interval

    Once you have the bootstrap distribution, the rest is easy:

    - **Standard error** = standard deviation of the bootstrap means: $\widehat{\operatorname{SE}} = \operatorname{sd}(\bar X^*_1, \dots, \bar X^*_B)$
    - **95% confidence interval (percentile method)**: throw away the bottom 2.5% and the top 2.5% of bootstrap means.
      What's left is your interval: $\big[\, q_{0.025}(\bar X^*),\; q_{0.975}(\bar X^*) \,\big]$

    No normal assumption, no z-scores, no formulas. And it works for **any statistic**: try the median, which has no simple SE formula.
    """)
    return


@app.cell
def ci_controls(mo):
    ci_level = mo.ui.slider(50, 99, value=95, step=1, label="confidence level (%)", show_value=True)
    ci_stat = mo.ui.dropdown(["mean", "median", "std", "90th percentile"], value="mean", label="statistic")
    mo.hstack([ci_level, ci_stat], justify="start")
    return ci_level, ci_stat


@app.cell
def ci_chart(COLORS, alt, bootstrap, ci_level, ci_stat, mo, my_sample, np, pd):
    STATS = {
        "mean": np.mean,
        "median": np.median,
        "std": lambda a, axis=None: np.std(a, axis=axis, ddof=1),
        "90th percentile": lambda a, axis=None: np.percentile(a, 90, axis=axis),
    }

    def percentile_ci(reps, level):
        """Keep the middle `level`% of bootstrap replicates."""
        _a = (100 - level) / 2
        return np.percentile(reps, [_a, 100 - _a])

    _f = STATS[ci_stat.value]
    ci_reps = bootstrap(my_sample, _f, 4000, np.random.default_rng(3))
    ci_lo, ci_hi = percentile_ci(ci_reps, ci_level.value)
    _est = float(_f(my_sample))
    _df = pd.DataFrame({"v": ci_reps})
    _df["inside"] = np.where((_df.v >= ci_lo) & (_df.v <= ci_hi), "kept", "thrown away")
    _bars = alt.Chart(_df).mark_bar().encode(
        x=alt.X("v:Q", bin=alt.Bin(maxbins=60), title=f"bootstrap {ci_stat.value} of 30 trees"),
        y=alt.Y("count():Q", stack=True, title="resamples"),
        color=alt.Color("inside:N", scale=alt.Scale(domain=["kept", "thrown away"], range=[COLORS["boot"], "#bbbbbb"]),
                        legend=alt.Legend(orient="top", title=None)))
    _r = alt.Chart(pd.DataFrame({"m": [_est]})).mark_rule(color=COLORS["sample"], size=3).encode(x="m:Q")
    _se = ci_reps.std(ddof=1)
    _z = {90: 1.645, 95: 1.96, 99: 2.576}.get(ci_level.value)
    mo.vstack([
        (_bars + _r).properties(width="container", height=220),
        mo.hstack([
            mo.stat(f"{_est:.2f}", label=f"sample {ci_stat.value}"),
            mo.stat(f"{_se:.3f}", label="bootstrap SE"),
            mo.stat(f"[{ci_lo:.2f}, {ci_hi:.2f}]", label=f"{ci_level.value}% percentile CI"),
            mo.stat(f"[{my_sample.mean() - 1.96 * my_sample.std(ddof=1) / np.sqrt(30):.2f}, {my_sample.mean() + 1.96 * my_sample.std(ddof=1) / np.sqrt(30):.2f}]",
                    label="textbook 95% for the mean", caption="X̄ ± 1.96·s/√n, for comparison"),
        ]),
    ])
    return STATS, percentile_ci


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Does a 95% interval really catch μ 95% of the time?** Let's check in the multiverse: each universe draws its own 30
    trees, builds its own bootstrap CI, and we count how often it contains the true $\mu = 14$.
    """)
    return


@app.cell
def coverage_btn(mo):
    coverage_run = mo.ui.run_button(label="🌌 Run 300 universes (B = 1000 each)")
    coverage_run
    return (coverage_run,)


@app.cell
def coverage_result(
    COLORS,
    MU_TRUE,
    alt,
    ci_level,
    coverage_run,
    draw_trees,
    mo,
    np,
    pd,
):
    mo.stop(not coverage_run.value, mo.md("_Press the button to run the coverage experiment._"))
    _rng = np.random.default_rng()
    _K, _B, _n = 300, 1000, 30
    _samples = draw_trees((_K, _n), _rng)
    _idx = _rng.integers(0, _n, size=(_K, _B, _n))
    _reps = np.take_along_axis(_samples[:, None, :].repeat(_B, 1), _idx, axis=2).mean(axis=2)
    _a = (100 - ci_level.value) / 2
    _lo, _hi = np.percentile(_reps, _a, axis=1), np.percentile(_reps, 100 - _a, axis=1)
    _hit = (_lo <= MU_TRUE) & (MU_TRUE <= _hi)
    _df = pd.DataFrame({"u": np.arange(_K), "lo": _lo, "hi": _hi, "hit": np.where(_hit, "contains μ", "misses μ")})
    _ch = alt.Chart(_df.head(100)).mark_rule(size=2).encode(
        y=alt.Y("u:O", axis=None, title="universe"), x=alt.X("lo:Q", title="CI for mean height (m)", scale=alt.Scale(zero=False)), x2="hi:Q",
        color=alt.Color("hit:N", scale=alt.Scale(domain=["contains μ", "misses μ"], range=[COLORS["boot"], COLORS["truth"]]), legend=alt.Legend(orient="top", title=None)))
    _mu = alt.Chart(pd.DataFrame({"m": [MU_TRUE]})).mark_rule(color=COLORS["truth"], strokeDash=[6, 4]).encode(x="m:Q")
    mo.vstack([
        mo.stat(f"{_hit.mean():.1%}", label=f"coverage of {ci_level.value}% bootstrap CIs over {_K} universes", caption="should be close to the nominal level (slightly under for small n)"),
        (_ch + _mu).properties(width="container", height=320, title="First 100 universes' intervals"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - Slide the level from 95 to 50: the grey tails grow, the interval shrinks. Then rerun coverage: about half the intervals miss.
    - Switch the statistic to **median** or **90th percentile**: the same two lines of code give you an SE and a CI, and no formula is needed.
    - For the **mean**, compare the percentile CI to the textbook $\bar X \pm 1.96\,s/\sqrt n$. They nearly agree, a good sanity check.
    """), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Why it works: the empirical distribution

    Why on earth should resampling from your own data mimic new draws from the forest? Because of the **empirical CDF**

    $$\hat F_n(x) = \frac{1}{n}\,\#\{\,i : X_i \le x\,\}.$$

    With 10 data points $\hat F_n$ is a jagged staircase barely resembling the truth. With 100 it tightens up, and with 1,000 it's
    almost indistinguishable from the real $F$. Resampling from the data **is** sampling from $\hat F_n$, so if $\hat F_n \approx F$,
    bootstrap resamples behave like fresh samples from the forest.
    """)
    return


@app.cell
def ecdf_controls(mo):
    ecdf_n = mo.ui.slider(steps=[10, 30, 100, 300, 1000, 3000], value=10, label="number of trees measured (n)", show_value=True)
    ecdf_seed = mo.ui.button(value=0, on_click=lambda v: v + 1, label="🎲 new sample")
    mo.hstack([ecdf_n, ecdf_seed], justify="start")
    return ecdf_n, ecdf_seed


@app.cell
def ecdf_chart(
    COLORS,
    POP_SCALE,
    POP_SHAPE,
    alt,
    draw_trees,
    ecdf_n,
    ecdf_seed,
    mo,
    np,
    pd,
    scipy_stats,
):
    _x = np.sort(draw_trees(ecdf_n.value, np.random.default_rng(50 + ecdf_seed.value)))
    _F = np.arange(1, len(_x) + 1) / len(_x)
    _grid = np.linspace(0, 32, 300)
    _true = scipy_stats.gamma.cdf(_grid, POP_SHAPE, scale=POP_SCALE)
    ecdf_ks = float(np.max(np.maximum(np.abs(_F - scipy_stats.gamma.cdf(_x, POP_SHAPE, scale=POP_SCALE)),
                                       np.abs(_F - 1 / len(_x) - scipy_stats.gamma.cdf(_x, POP_SHAPE, scale=POP_SCALE)))))
    _e = pd.DataFrame({"x": np.r_[0, _x, 32], "F": np.r_[0, _F, 1]})
    _t = pd.DataFrame({"x": _grid, "F": _true})
    _c1 = alt.Chart(_t).mark_line(color=COLORS["truth"], strokeDash=[6, 4], size=2).encode(x=alt.X("x:Q", title="tree height (m)"), y=alt.Y("F:Q", title="P(height ≤ x)"))
    _c2 = alt.Chart(_e).mark_line(color=COLORS["sample"], interpolate="step-after", size=2).encode(x="x:Q", y="F:Q")
    mo.vstack([
        (_c1 + _c2).properties(width="container", height=260, title="purple staircase = empirical CDF F̂ₙ · dashed red = true CDF F"),
        mo.hstack([
            mo.stat(f"{ecdf_ks:.3f}", label="largest gap |F̂ₙ − F|  (KS distance)"),
            mo.stat(f"{1.36 / np.sqrt(ecdf_n.value):.3f}", label="typical 95% bound ≈ 1.36/√n"),
        ]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - Step n from 10 → 100 → 1000. The staircase hugs the red curve, and the gap shrinks like $1/\sqrt n$ (Glivenko–Cantelli / DKW).
    - At n = 10, resample a few times: the staircase is wildly different each time. That's why bootstrap with tiny samples is shaky.
    - **Limits:** the bootstrap struggles with extremes (e.g. the *maximum*: a resample can never exceed your tallest tree) and with dependent data (time series need block bootstrap).
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Sandbox: bootstrap your own data

    Your sample is now a set of **draggable trees** on a number line. **Drag** a point, **double-click** empty space to add one,
    **double-click** a point to remove it. While you drag, the purple line ($\bar X$) and median follow instantly in JS. On release, Python
    bootstraps your data and draws the orange bootstrap distribution with its CI.
    """)
    return


@app.cell
def drag_class(anywidget, traitlets):
    class DragSample(anywidget.AnyWidget):
        """Draggable 1-D data points. save_changes() only on pointer-up."""
        _esm = r"""
    function render({ model, el }) {
      const W = 700, H = 170, PAD = 30, XMIN = 0, XMAX = 40, Y0 = 110;
      const NS = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none",
        background: "rgba(127,127,127,0.06)", borderRadius: "8px" });
      const bar = document.createElement("div"); Object.assign(bar.style, { display: "flex", gap: "6px", marginBottom: "6px" });
      el.append(bar, svg);
      const mk = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); svg.appendChild(e); return e; };
      const toSvg = e => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const sx = v => PAD + (v - XMIN) / (XMAX - XMIN) * (W - 2 * PAD);
      const inv = px => XMIN + (px - PAD) / (W - 2 * PAD) * (XMAX - XMIN);
      const snap = v => Math.max(XMIN, Math.min(XMAX, Math.round(v * 2) / 2));
      const presets = {
        "Tree sample": () => [9,10.5,11,12,12,12.5,13,13.5,14,14,14.5,15,15.5,16,17,18,19.5],
        "Add an outlier 🌳": () => [...model.get("values"), 38],
        "Tiny (n=4)": () => [11, 13, 15, 17],
        "Two clumps": () => [6,6.5,7,7.5,8,8,25,25.5,26,27,27.5,28],
      };
      for (const name in presets) { const b = document.createElement("button"); b.textContent = name;
        Object.assign(b.style, { padding: "3px 10px", borderRadius: "12px", border: "1px solid #bbb", cursor: "pointer" });
        b.onclick = () => { model.set("values", presets[name]()); model.save_changes(); }; bar.appendChild(b); }
      let drag = -1;
      function draw() {
        const xs = model.get("values");
        svg.innerHTML = "";
        mk("line", { x1: sx(XMIN), x2: sx(XMAX), y1: Y0, y2: Y0, stroke: "#999", "stroke-width": 2 });
        for (let v = XMIN; v <= XMAX; v += 5) { const t = mk("text", { x: sx(v), y: Y0 + 22, "text-anchor": "middle", "font-size": 11, fill: "#888" }); t.textContent = v; }
        const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
        const s = [...xs].sort((a, b) => a - b), med = s.length % 2 ? s[(s.length - 1) / 2] : (s[s.length / 2 - 1] + s[s.length / 2]) / 2;
        mk("line", { x1: sx(mean), x2: sx(mean), y1: 20, y2: Y0, stroke: "#6c5b7b", "stroke-width": 3 });
        const t1 = mk("text", { x: sx(mean) + 4, y: 30, "font-size": 12, fill: "#6c5b7b" }); t1.textContent = `X̄ = ${mean.toFixed(2)}`;
        mk("line", { x1: sx(med), x2: sx(med), y1: 40, y2: Y0, stroke: "#6c5b7b", "stroke-width": 2, "stroke-dasharray": "4 3" });
        const t2 = mk("text", { x: sx(med) + 4, y: 50, "font-size": 12, fill: "#6c5b7b" }); t2.textContent = `median = ${med.toFixed(2)}`;
        // stack duplicates visually
        const seen = {};
        xs.forEach((v, i) => { const k = v.toFixed(1); seen[k] = (seen[k] || 0) + 1;
          const c = mk("circle", { cx: sx(v), cy: Y0 - 10 - (seen[k] - 1) * 14, r: 7, fill: "#6c5b7b", stroke: "white", "stroke-width": 1.5 });
          c.style.cursor = "grab";
          c.addEventListener("pointerdown", e => { e.stopPropagation(); drag = i; svg.setPointerCapture(e.pointerId); });
          c.addEventListener("dblclick", e => { e.stopPropagation(); const a = [...model.get("values")];
            if (a.length > 2) { a.splice(i, 1); model.set("values", a); model.save_changes(); } });
        });
        const t3 = mk("text", { x: W - PAD, y: 16, "text-anchor": "end", "font-size": 12, fill: "#888" }); t3.textContent = `n = ${xs.length}`;
      }
      svg.addEventListener("dblclick", e => { model.set("values", [...model.get("values"), snap(inv(toSvg(e).x))]); model.save_changes(); });
      svg.addEventListener("pointermove", e => { if (drag < 0) return; const a = [...model.get("values")]; a[drag] = snap(inv(toSvg(e).x)); model.set("values", a); });
      const stop = () => { if (drag >= 0) { drag = -1; model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointerleave", stop);
      model.on("change:values", draw); draw();
    }
    export default { render };
    """
        values = traitlets.List(traitlets.Float()).tag(sync=True)

    return (DragSample,)


@app.cell
def sandbox_widget(DragSample, STATS, mo):
    sandbox = mo.ui.anywidget(DragSample(values=[9, 10.5, 11, 12, 12, 12.5, 13, 13.5, 14, 14, 14.5, 15, 15.5, 16, 17, 18, 19.5]))
    sandbox_stat = mo.ui.dropdown(list(STATS) + ["max"], value="mean", label="statistic")
    sandbox_B = mo.ui.slider(100, 10000, value=2000, step=100, label="B", show_value=True)
    mo.vstack([sandbox, mo.hstack([sandbox_stat, sandbox_B], justify="start")])
    return sandbox, sandbox_B, sandbox_stat


@app.cell
def sandbox_readout(
    COLORS,
    STATS,
    alt,
    bootstrap,
    mo,
    np,
    pd,
    percentile_ci,
    sandbox,
    sandbox_B,
    sandbox_stat,
):
    _x = np.array(sandbox.value["values"], dtype=float)
    _f = STATS.get(sandbox_stat.value, lambda a, axis=None: np.max(a, axis=axis))
    _reps = bootstrap(_x, _f, sandbox_B.value, np.random.default_rng(11))
    _lo, _hi = percentile_ci(_reps, 95)
    _df = pd.DataFrame({"v": _reps})
    _h = alt.Chart(_df).mark_bar(color=COLORS["boot"], opacity=0.85).encode(
        x=alt.X("v:Q", bin=alt.Bin(maxbins=50), title=f"bootstrap {sandbox_stat.value}"), y=alt.Y("count():Q", title="resamples"))
    _band = alt.Chart(pd.DataFrame({"lo": [_lo], "hi": [_hi]})).mark_rect(color=COLORS["boot"], opacity=0.12).encode(x="lo:Q", x2="hi:Q")
    _r = alt.Chart(pd.DataFrame({"m": [float(_f(_x))]})).mark_rule(color=COLORS["sample"], size=3).encode(x="m:Q")
    mo.vstack([
        (_band + _h + _r).properties(width="container", height=200),
        mo.hstack([
            mo.stat(f"{len(_x)}", label="n"),
            mo.stat(f"{float(_f(_x)):.2f}", label=f"sample {sandbox_stat.value}"),
            mo.stat(f"{_reps.std(ddof=1):.3f}", label="bootstrap SE"),
            mo.stat(f"[{_lo:.2f}, {_hi:.2f}]", label="95% percentile CI"),
            mo.stat(f"{len(np.unique(_reps))}", label="distinct values of the statistic"),
        ]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Challenges**
    1. Press **Add an outlier 🌳**. Which statistic's CI blows up more, **mean** or **median**? (Robustness, visible.)
    2. Choose **max**. The bootstrap histogram piles up on your tallest tree and can never go higher. This is a known bootstrap failure.
    3. Use **Tiny (n=4)** with **median**: count the distinct values. A 4-point sample has only a handful of possible resamples, so the "distribution" is a few spikes.
    4. Load **Two clumps** and look at the mean: the bootstrap reveals a lumpy, non-normal sampling distribution that a ±1.96·SE formula would hide.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Quiz: check your intuition
    Answer all, then flip the reveal switch.
    """)
    return


@app.cell
def quiz(mo):
    quiz = mo.ui.dictionary({
        "q1": mo.ui.radio(["Without replacement, same size", "With replacement, same size", "With replacement, half the size"],
                          label="**1.** A bootstrap resample is drawn…"),
        "q2": mo.ui.radio(["Its mean equals μ", "Its width approximates the true sampling distribution's width", "It is always normal"],
                          label="**2.** What does the bootstrap distribution get right?"),
        "q3": mo.ui.radio(["sd of the bootstrap means", "sd of the original sample", "mean of the bootstrap means"],
                          label="**3.** The bootstrap standard error is…"),
        "q4": mo.ui.radio(["X̄ ± 1.96·SE", "the 2.5% and 97.5% percentiles of the bootstrap means", "min and max of the bootstrap means"],
                          label="**4.** The 95% percentile CI is…"),
        "q5": mo.ui.radio(["≈ 0%", "≈ 37%", "≈ 63%"], label="**5.** For n = 30, roughly what share of original points are missing from a given resample?"),
        "q6": mo.ui.radio(["The empirical CDF F̂ₙ approaches the true F as n grows", "Resampling creates new information", "The population is always normal"],
                          label="**6.** Why does the bootstrap work?"),
    })
    quiz_reveal = mo.ui.switch(label="reveal answers")
    mo.vstack([quiz, quiz_reveal])
    return quiz, quiz_reveal


@app.cell
def quiz_score(mo, quiz, quiz_reveal):
    QUIZ_ANSWERS = {
        "q1": ("With replacement, same size", "Same n, with replacement, so some trees repeat and some are skipped."),
        "q2": ("Its width approximates the true sampling distribution's width", "It is centered at your X̄, not μ, but its shape and spread match."),
        "q3": ("sd of the bootstrap means", "SE = sd(X̄*₁ … X̄*_B)."),
        "q4": ("the 2.5% and 97.5% percentiles of the bootstrap means", "Throw away 2.5% on each side. No z-scores needed."),
        "q5": ("≈ 37%", f"(1 − 1/30)³⁰ = {(1 - 1/30) ** 30:.3f} → 1/e."),
        "q6": ("The empirical CDF F̂ₙ approaches the true F as n grows", "Sampling from your data = sampling from F̂ₙ ≈ F."),
    }
    _score = sum(quiz.value[k] == a for k, (a, _) in QUIZ_ANSWERS.items())
    _answered = sum(v is not None for v in quiz.value.values())
    if not quiz_reveal.value:
        _out = mo.md(f"Answered **{_answered}/6**. Flip *reveal answers* when ready.")
    else:
        _rows = "\n".join(
            f"| {k[1:]} | {'✅' if quiz.value[k] == a else '❌'} | {a} | {why} |" for k, (a, why) in QUIZ_ANSWERS.items())
        _out = mo.vstack([
            mo.callout(mo.md(f"### Score: {_score} / 6"), kind="success" if _score >= 5 else "warn"),
            mo.md("| # | | correct answer | why |\n|---|---|---|---|\n" + _rows),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap

    | Chapter | Idea | Formula / rule | Where you touched it |
    |---|---|---|---|
    | One sample, one number | $\bar X$ wiggles from sample to sample | $\bar X = \frac1n\sum X_i$ | 🌲 re-measure button |
    | Imagined multiverse | The sampling distribution is the "dream object" | $\operatorname{SE} = \sigma/\sqrt n$ | animated multiverse simulator |
    | Bootstrap trick | Treat the sample as the population; resample **with replacement** | $X^*_i \sim \hat F_n$; ~36.8% skipped | resample cards |
    | Bootstrap distribution | B resample means mimic the true sampling distribution's shape & width | $\bar X^*_1,\dots,\bar X^*_B$ | blue vs orange overlay |
    | SE & CI | SE = sd of $\bar X^*$; CI = drop the tails | $[q_{\alpha/2},\, q_{1-\alpha/2}]$ | CI chart + coverage experiment |
    | Why it works | $\hat F_n \to F$ as n grows | $\sup|\hat F_n - F| \sim 1/\sqrt n$ | ECDF staircase |
    | Limits | Tiny n, extremes (max), dependent data | | sandbox challenges |

    *"Your sample, as small and humble as it looks, is actually the best window you have onto the population."*
    """)
    return


if __name__ == "__main__":
    app.run()

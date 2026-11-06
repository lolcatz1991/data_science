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
    import warnings
    from scipy import stats
    import statsmodels.formula.api as smf

    return alt, anywidget, mo, np, pd, smf, stats, traitlets, warnings


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
    # Mixed models, explained by touching them
    *An interactive companion to Simplistics (QuantPsych): **"Simple Explanation of Mixed Models (Hierarchical Linear Models, Multilevel Models)"**.*

    Mixed models, hierarchical linear models (HLM), multilevel models, and mixed-effects models are **the same thing**. They are a tool for data that comes in **clusters**: clients of the same therapist, students of the same teacher, or the same person measured again and again.
    """),
        mo.Html('<iframe width="560" height="315" style="max-width:100%" src="https://www.youtube.com/embed/5tOifM51ZOk" title="Simple Explanation of Mixed Models" frameborder="0" allowfullscreen></iframe>'),
        mo.md(r"""
    **Route through this notebook (the video's order):**

    | # | Idea | You will... |
    |---|---|---|
    | 1 | Duplicating your data | make copies of a thesis dataset and watch *p* fall |
    | 2 | Clustering is partial duplication | run thousands of fake classroom studies in an animated simulator |
    | 3 | What mixed models do geometrically | drag hospitals around and break a regression line |
    | 4 | The technical caveat: lines are pulled toward the fixed effect | see estimates shrink toward the average |
    | 5 | Fixed vs random intercepts and slopes | build the four model types by dragging lines |
    | 6 | Sandbox, quiz, recap | experiment freely, then test yourself |

    **Color language (used throughout):** <span style="color:#999">**gray dashed** = naive model that ignores clusters</span> ·
    **black** = the **fixed effect** (the average line across clusters) · <span style="color:#3a86ff">**colors**</span> = individual clusters (their **random effects**) ·
    <span style="color:#e4572e">**red**</span> = danger: a false positive or a misleading result.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The duplicate trick: why independence matters

    The video opens with a story. An undergraduate honors thesis has **67 participants** and a *p*-value just above .05. The tempting idea: *"If sample size is the only thing stopping me... why not just duplicate my data?"*

    The *p*-value asks **"how likely is data like this, if there is no effect?"** Getting the exact same 67 scores again from 67 *new* people has a probability of essentially zero. A copy has a probability of **one**. A copy adds **no new information**, but the formula still counts it as new evidence:

    $$t = r\sqrt{\frac{n-2}{1-r^2}} \qquad\Rightarrow\qquad n \uparrow \;\Rightarrow\; t\uparrow \;\Rightarrow\; p \downarrow$$

    Every test that uses probabilities assumes the rows are **independent**. Use the slider to copy the dataset.
    """)
    return


@app.cell
def dup_controls(mo, np, pd, stats):
    _rng = np.random.default_rng(67)
    _x = _rng.normal(size=67)
    _z = _rng.normal(size=67)
    _x = (_x - _x.mean()) / _x.std()
    _z = _z - _z.mean()
    _z = _z - (_z @ _x) / (_x @ _x) * _x
    _z = _z / _z.std()
    thesis_r = 0.21  # the observed correlation, fixed exactly
    thesis_df = pd.DataFrame({
        "predictor": 50 + 10 * _x,
        "outcome": 50 + 10 * (thesis_r * _x + np.sqrt(1 - thesis_r**2) * _z),
    })

    def p_for_r(r, n):
        """Two-sided p-value of a Pearson correlation r with n rows."""
        _t = r * np.sqrt((n - 2) / (1 - r**2))
        return 2 * stats.t.sf(abs(_t), n - 2)

    dup_copies = mo.ui.slider(1, 10, value=1, label="Copies of the 67 participants", show_value=True)
    dup_copies
    return dup_copies, p_for_r, thesis_df, thesis_r


@app.cell
def dup_readout(alt, dup_copies, mo, np, p_for_r, pd, thesis_df, thesis_r):
    _k = dup_copies.value
    _n = 67 * _k
    _p = p_for_r(thesis_r, _n)
    _scatter = alt.Chart(pd.concat([thesis_df] * _k)).mark_circle(size=60, opacity=min(1.0, 0.25 * _k), color="#3a86ff").encode(
        x=alt.X("predictor:Q", scale=alt.Scale(zero=False)), y=alt.Y("outcome:Q", scale=alt.Scale(zero=False)),
    ).properties(width=300, height=260, title=f"The data with {_k} copies (it looks identical)")
    _curve = pd.DataFrame({"copies": np.arange(1, 11)})
    _curve["p"] = [p_for_r(thesis_r, 67 * c) for c in _curve["copies"]]
    _line = alt.Chart(_curve).mark_line(point=True, color="#222").encode(
        x=alt.X("copies:O", title="copies"), y=alt.Y("p:Q", scale=alt.Scale(type="log"), title="p-value (log scale)"))
    _now = alt.Chart(_curve[_curve.copies == _k]).mark_circle(size=220, color="#e4572e").encode(x="copies:O", y="p:Q")
    _rule = alt.Chart(pd.DataFrame({"y": [0.05]})).mark_rule(strokeDash=[5, 4], color="#e4572e").encode(y="y:Q")
    mo.vstack([
        mo.hstack([
            mo.stat(f"{_n}", label="rows the test thinks it has"),
            mo.stat("67", label="independent people you actually have"),
            mo.stat(f"{thesis_r:.2f}", label="correlation r (unchanged)"),
            mo.stat(f"{_p:.4f}", label="p-value", caption="'significant'!" if _p < 0.05 else "not significant"),
        ]),
        alt.hconcat(_scatter, (_line + _now + _rule).properties(width=300, height=260, title="p-value vs number of copies")),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Go from 1 to 2 copies. The picture and *r* do not change at all, but *p* drops below .05. Nothing about the world changed.
    2. Go to 10 copies. *p* becomes tiny. This is **certainty you did not earn**.
    3. Keep this in mind for section 2: your brother's opinions, or your own answers 6 months from now, are *partial* copies of you.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Clustering is partial duplication

    Nobody duplicates data on purpose. But if you ask a stranger about a Supreme Court decision, you cannot predict their answer from the narrator's answer. If you ask **his brother** (same house, same parents, same genes), the answer is *almost a copy*. The same thing happens with repeated measures (you now vs. you in 6 months), family members, and **students taught by the same teacher**. Scores that are correlated within a group are **clustered**.

    The amount of copying is the **intraclass correlation**, ρ (ICC): the share of the variance that is shared by everyone in a cluster.

    $$y_{ij} = \mu + u_j + \varepsilon_{ij},\qquad \rho = \frac{\operatorname{Var}(u_j)}{\operatorname{Var}(u_j) + \operatorname{Var}(\varepsilon_{ij})}$$

    A cluster of *m* people is worth only $m / \text{DE}$ independent people, where $\text{DE} = 1 + (m-1)\rho$ is the **design effect**.

    **The simulator:** each study gives a teaching intervention to half the classrooms. The intervention **does nothing** (the null is true). Each dot is one study's *t*-statistic:
    * **top row (naive):** treat every student as independent, which is a plain *t*-test on all students.
    * **bottom row (cluster-aware):** compare classroom means, which is a *t*-test on *J* clusters.

    A <span style="color:#e4572e">**red**</span> dot is a false positive. An honest test should produce red dots about 5% of the time.
    """)
    return


@app.cell
def ClusterSim_class(anywidget, traitlets):
    class ClusterSim(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 640, H = 230, PAD = 20, TMAX = 8;
      let timer = null, dots = [], n = 0, fpN = 0, fpC = 0;
      const ctl = document.createElement("div");
      ctl.style.cssText = "display:flex;gap:6px;align-items:center;margin-bottom:6px;flex-wrap:wrap";
      el.appendChild(ctl);
      const btn = (label, fn) => { const b = document.createElement("button"); b.textContent = label; b.onclick = fn;
        b.style.cssText = "padding:4px 10px;border-radius:6px;border:1px solid #888;background:transparent;color:inherit;cursor:pointer"; ctl.appendChild(b); return b; };
      const info = document.createElement("span"); info.style.marginLeft = "10px";
      const canvas = document.createElement("canvas"); canvas.width = W; canvas.height = H;
      canvas.style.cssText = "width:100%;max-width:" + W + "px";
      el.appendChild(canvas);
      const cx = canvas.getContext("2d");
      const randn = () => { let u = 0; while (!u) u = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random()); };
      const mean = a => a.reduce((s, v) => s + v, 0) / a.length;
      const tstat = (a, b) => { const ma = mean(a), mb = mean(b);
        const ss = a.reduce((s, v) => s + (v - ma) ** 2, 0) + b.reduce((s, v) => s + (v - mb) ** 2, 0);
        const sp = ss / (a.length + b.length - 2); return (mb - ma) / Math.sqrt(sp * (1 / a.length + 1 / b.length)); };
      function study() {
        const J = model.get("J"), m = model.get("m"), rho = model.get("rho");
        const su = Math.sqrt(rho), se = Math.sqrt(1 - rho);
        const g0 = [], g1 = [], c0 = [], c1 = [];
        for (let j = 0; j < J; j++) {
          const u = su * randn(); const vals = [];
          for (let i = 0; i < m; i++) vals.push(u + se * randn());
          if (j < J / 2) { g0.push(...vals); c0.push(mean(vals)); } else { g1.push(...vals); c1.push(mean(vals)); }
        }
        const tn = tstat(g0, g1), tc = tstat(c0, c1);
        const rn = Math.abs(tn) > model.get("crit_naive"), rc = Math.abs(tc) > model.get("crit_cluster");
        n++; if (rn) fpN++; if (rc) fpC++;
        dots.push([tn, rn, tc, rc]); if (dots.length > 400) dots.shift();
      }
      const sx = t => PAD + (Math.max(-TMAX, Math.min(TMAX, t)) + TMAX) / (2 * TMAX) * (W - 2 * PAD);
      const cssColor = getComputedStyle(el).color || "#888";
      function draw() {
        cx.clearRect(0, 0, W, H);
        cx.font = "13px sans-serif"; cx.fillStyle = cssColor;
        const rows = [[60, "naive (students as independent)", model.get("crit_naive"), 0],
                      [160, "cluster-aware (classroom means)", model.get("crit_cluster"), 2]];
        for (const [y, label, crit, k] of rows) {
          cx.fillStyle = cssColor; cx.fillText(label, PAD, y - 34);
          cx.strokeStyle = "#999"; cx.setLineDash([]); cx.beginPath(); cx.moveTo(PAD, y); cx.lineTo(W - PAD, y); cx.stroke();
          cx.strokeStyle = "#e4572e"; cx.setLineDash([4, 3]);
          for (const s of [-1, 1]) { cx.beginPath(); cx.moveTo(sx(s * crit), y - 28); cx.lineTo(sx(s * crit), y + 28); cx.stroke(); }
          cx.setLineDash([]);
          dots.forEach((d, i) => { cx.globalAlpha = 0.15 + 0.85 * i / dots.length;
            cx.fillStyle = d[k + 1] ? "#e4572e" : "#3a86ff";
            cx.beginPath(); cx.arc(sx(d[k]), y + ((i * 37) % 41) - 20, 3, 0, 7); cx.fill(); });
          cx.globalAlpha = 1;
          const rate = n ? (k === 0 ? fpN : fpC) / n : 0;
          cx.fillStyle = rate > 0.08 ? "#e4572e" : cssColor;
          cx.fillText(`false positives: ${(100 * rate).toFixed(1)}%`, W - PAD - 170, y - 34);
        }
        cx.fillStyle = cssColor; cx.fillText("t-statistic →  (dashed red = critical value)", W / 2 - 130, H - 6);
        info.textContent = `${n} studies run`;
      }
      const push = () => { model.set("n_studies", n); model.set("fp_naive", fpN); model.set("fp_cluster", fpC); model.save_changes(); };
      const tick = () => { for (let r = 0; r < model.get("batch"); r++) study(); draw(); if (n % 50 < model.get("batch")) push(); };
      const play = btn("▶ play", () => {
        if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; push(); }
        else { timer = setInterval(tick, 40); play.textContent = "⏸ pause"; } });
      btn("+1 study", () => { study(); draw(); push(); });
      btn("+500 studies", () => { for (let r = 0; r < 500; r++) study(); draw(); push(); });
      btn("↺ reset", () => { dots = []; n = 0; fpN = 0; fpC = 0; draw(); push(); });
      ctl.appendChild(info);
      draw();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        J = traitlets.Int(10).tag(sync=True)
        m = traitlets.Int(20).tag(sync=True)
        rho = traitlets.Float(0.2).tag(sync=True)
        crit_naive = traitlets.Float(1.97).tag(sync=True)
        crit_cluster = traitlets.Float(2.3).tag(sync=True)
        batch = traitlets.Int(5).tag(sync=True)
        n_studies = traitlets.Int(0).tag(sync=True)
        fp_naive = traitlets.Int(0).tag(sync=True)
        fp_cluster = traitlets.Int(0).tag(sync=True)

    return (ClusterSim,)


@app.cell
def sim_controls(mo):
    sim_rho = mo.ui.slider(0.0, 0.6, step=0.05, value=0.2, label="ICC ρ (how much students in a class are 'copies')", show_value=True)
    sim_J = mo.ui.slider(4, 30, step=2, value=10, label="classrooms J", show_value=True)
    sim_m = mo.ui.slider(2, 60, value=20, label="students per classroom m", show_value=True)
    mo.vstack([sim_rho, sim_J, sim_m])
    return sim_J, sim_m, sim_rho


@app.cell
def cluster_sim(ClusterSim, mo, sim_J, sim_m, sim_rho, stats):
    cluster_sim = mo.ui.anywidget(ClusterSim(
        J=sim_J.value, m=sim_m.value, rho=sim_rho.value,
        crit_naive=float(stats.t.ppf(0.975, sim_J.value * sim_m.value - 2)),
        crit_cluster=float(stats.t.ppf(0.975, sim_J.value - 2)),
    ))
    cluster_sim
    return (cluster_sim,)


@app.cell
def sim_readout(alt, cluster_sim, mo, np, pd, sim_J, sim_m, sim_rho, stats):
    _v = cluster_sim.value
    _N = sim_J.value * sim_m.value
    _de = 1 + (sim_m.value - 1) * sim_rho.value
    _ns = max(_v["n_studies"], 1)
    _fn, _fc = _v["fp_naive"] / _ns, _v["fp_cluster"] / _ns
    _df = pd.DataFrame({"m": np.arange(2, 61)})
    _df["design effect"] = 1 + (_df["m"] - 1) * sim_rho.value
    _df["naive false-positive rate (theory)"] = 2 * stats.norm.sf(stats.norm.ppf(0.975) / np.sqrt(_df["design effect"]))
    _chart = alt.Chart(_df).mark_line(color="#e4572e").encode(
        x=alt.X("m:Q", title="students per classroom m"),
        y=alt.Y("naive false-positive rate (theory):Q", axis=alt.Axis(format="%"), title="naive false-positive rate (approx.)"),
    ) + alt.Chart(pd.DataFrame({"y": [0.05]})).mark_rule(strokeDash=[5, 4], color="#222").encode(y="y:Q") + alt.Chart(
        pd.DataFrame({"m": [sim_m.value], "r": [2 * stats.norm.sf(stats.norm.ppf(0.975) / np.sqrt(_de))]})
    ).mark_circle(size=160, color="#e4572e").encode(x="m:Q", y="r:Q")
    mo.vstack([
        mo.hstack([
            mo.stat(f"{_N}", label="students measured"),
            mo.stat(f"{_de:.2f}", label="design effect 1+(m−1)ρ"),
            mo.stat(f"{_N / _de:.0f}", label="effective independent students"),
            mo.stat(f"{100 * _fn:.1f}%" if _v["n_studies"] else "–", label="naive false positives (sim)"),
            mo.stat(f"{100 * _fc:.1f}%" if _v["n_studies"] else "–", label="cluster-aware false positives (sim)"),
        ]),
        _chart.properties(width="container", height=220, title="Bigger classrooms make the naive test more wrong (black = nominal 5%)"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Press **+500 studies** with ρ = 0.2. The naive row is covered in red dots (often 30% or more false positives). The cluster-aware row stays near 5%.
    2. Set ρ = 0. Both rows become honest. When there is no clustering, there is nothing to fix.
    3. Keep ρ = 0.05 (small!) and raise *m* to 60. Even a little within-class similarity, multiplied by big clusters, ruins the naive *p*-value. This is the video's **consequence #1: artificially low *p*-values, so more Type I errors.**
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · What mixed models do geometrically: the hospital example

    The video's **second consequence** of ignoring clusters is that the *relationship itself* can be deceiving. On the *x*-axis is **symptom severity**, and on the *y*-axis is the **survival rate**. If you fit one line through everything, the result seems to say *"the more severe your symptoms, the more likely you are to survive"*. That seems weird.

    Now color the dots by hospital: <span style="color:#e4572e">**In-N-Out**</span>, <span style="color:#2a9d8f">**Cyprus**</span>, <span style="color:#3a86ff">**University Medical**</span>. University Medical takes only severe patients, and In-N-Out takes only mild ones. **Inside each hospital, the relationship is negative.**

    > Mixed models (roughly) fit **a separate line for each cluster** (the **random effects**), then average them into **one line** (the **fixed effect**).

    $$\text{survival}_{ij} = \underbrace{\gamma_0 + \gamma_1\,\text{severity}_{ij}}_{\text{fixed effect (black)}} + \underbrace{u_{0j}}_{\text{hospital } j\text{'s own shift}} + \varepsilon_{ij}$$

    **Drag each hospital (the big circle)** to change who it treats and how well it does. The points move with it. The gray dashed line ignores clusters. The black line is the average of the colored hospital lines.
    """)
    return


@app.cell
def HospitalCanvas_class(anywidget, traitlets):
    class HospitalCanvas(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 640, H = 400, PADL = 48, PADB = 40, PADT = 14, PADR = 16, XMAX = 10, YMAX = 100;
      const NS = "http://www.w3.org/2000/svg";
      const bar = document.createElement("div"); bar.style.marginBottom = "6px"; el.appendChild(bar);
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none" });
      el.appendChild(svg);
      const mk = (tag, attrs, parent = svg) => { const e = document.createElementNS(NS, tag);
        for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
      const sx = v => PADL + v / XMAX * (W - PADL - PADR);
      const sy = v => H - PADB - v / YMAX * (H - PADB - PADT);
      const ix = p => (p - PADL) / (W - PADL - PADR) * XMAX;
      const iy = p => (H - PADB - p) / (H - PADB - PADT) * YMAX;
      const toSvg = e => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const ols = pts => { const n = pts.length, mx = pts.reduce((s, p) => s + p[0], 0) / n, my = pts.reduce((s, p) => s + p[1], 0) / n;
        let sxy = 0, sxx = 0; pts.forEach(p => { sxy += (p[0] - mx) * (p[1] - my); sxx += (p[0] - mx) ** 2; });
        const b = sxx > 0 ? sxy / sxx : 0; return [my - b * mx, b]; };
      const seg = (a, b, x0, x1, attrs) => mk("line", Object.assign({ x1: sx(x0), x2: sx(x1), y1: sy(a + b * x0), y2: sy(a + b * x1) }, attrs));
      const readout = document.createElement("div"); readout.style.cssText = "font:13px sans-serif;margin-top:4px"; el.appendChild(readout);
      let drag = -1;
      const presets = { "Video layout (paradox)": [[2.5, 30], [5, 55], [7.5, 80]], "Same survival level": [[2.5, 60], [5, 60], [7.5, 60]],
                        "Reverse it": [[2.5, 80], [5, 55], [7.5, 30]] };
      for (const name in presets) { const b = document.createElement("button"); b.textContent = name;
        b.style.cssText = "margin-right:6px;padding:3px 9px;border-radius:6px;border:1px solid #888;background:transparent;color:inherit;cursor:pointer";
        b.onclick = () => { model.set("centers", presets[name]); model.save_changes(); }; bar.appendChild(b); }
      function draw() {
        const C = model.get("centers"), O = model.get("offsets"), cols = model.get("colors"), names = model.get("names");
        svg.innerHTML = "";
        mk("rect", { x: PADL, y: PADT, width: W - PADL - PADR, height: H - PADB - PADT, fill: "none", stroke: "#8884" });
        for (let t = 0; t <= 10; t += 2) { const tx = mk("text", { x: sx(t), y: H - PADB + 16, "text-anchor": "middle", "font-size": 11, fill: "currentColor" }); tx.textContent = t; }
        for (let t = 0; t <= 100; t += 25) { const ty = mk("text", { x: PADL - 6, y: sy(t) + 4, "text-anchor": "end", "font-size": 11, fill: "currentColor" }); ty.textContent = t + "%"; }
        mk("text", { x: (W + PADL) / 2, y: H - 6, "text-anchor": "middle", "font-size": 12, fill: "currentColor" }).textContent = "symptom severity";
        const all = [], lines = [];
        C.forEach((c, h) => {
          const pts = O[h].map(o => [c[0] + o[0], c[1] + o[1]]); all.push(...pts);
          pts.forEach(p => mk("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 4, fill: cols[h], opacity: 0.55 }));
          const [a, b] = ols(pts); lines.push([a, b]);
          const xs = pts.map(p => p[0]); seg(a, b, Math.min(...xs), Math.max(...xs), { stroke: cols[h], "stroke-width": 3 });
        });
        const [pa, pb] = ols(all);
        seg(pa, pb, 0, XMAX, { stroke: "#999", "stroke-width": 3, "stroke-dasharray": "8 6" });
        const fa = lines.reduce((s, l) => s + l[0], 0) / lines.length, fb = lines.reduce((s, l) => s + l[1], 0) / lines.length;
        seg(fa, fb, 0, XMAX, { stroke: "currentColor", "stroke-width": 3 });
        C.forEach((c, h) => {
          const g = mk("circle", { cx: sx(c[0]), cy: sy(c[1]), r: 15, fill: cols[h], stroke: "#fff", "stroke-width": 3 });
          g.style.cursor = "grab";
          g.addEventListener("pointerdown", e => { drag = h; svg.setPointerCapture(e.pointerId); });
          mk("text", { x: sx(c[0]), y: sy(c[1]) - 20, "text-anchor": "middle", "font-size": 12, "font-weight": "bold", fill: cols[h] }).textContent = names[h];
        });
        readout.innerHTML = `<span style="color:#999">ignore clusters: slope <b>${pb >= 0 ? "+" : ""}${pb.toFixed(1)}</b> %/pt</span> &nbsp;·&nbsp; <b>average of hospital lines: slope ${fb >= 0 ? "+" : ""}${fb.toFixed(1)} %/pt</b>`;
      }
      svg.addEventListener("pointermove", e => { if (drag < 0) return; const p = toSvg(e);
        const C = model.get("centers").map(c => [...c]);
        C[drag] = [Math.max(1.5, Math.min(8.5, Math.round(ix(p.x) * 10) / 10)), Math.max(10, Math.min(90, Math.round(iy(p.y))))];
        model.set("centers", C); });
      const stop = () => { if (drag >= 0) { drag = -1; model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointerleave", stop);
      model.on("change:centers", draw); draw();
    }
    export default { render };
    """
        centers = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)
        offsets = traitlets.List(traitlets.List(traitlets.List(traitlets.Float()))).tag(sync=True)
        names = traitlets.List(traitlets.Unicode()).tag(sync=True)
        colors = traitlets.List(traitlets.Unicode()).tag(sync=True)

    return (HospitalCanvas,)


@app.cell
def hospital_canvas(HospitalCanvas, mo, np):
    hospital_names = ["In-N-Out", "Cyprus", "University Medical"]
    hospital_colors = ["#e4572e", "#2a9d8f", "#3a86ff"]
    _rng = np.random.default_rng(3)
    _offsets = []
    for _h in range(3):
        _dx = _rng.uniform(-1.3, 1.3, 25)
        _dy = -6 * _dx + _rng.normal(0, 3, 25)  # within-hospital slope: -6 % per severity point
        _offsets.append([[float(a - _dx.mean()), float(b - _dy.mean())] for a, b in zip(_dx, _dy)])
    hospital_canvas = mo.ui.anywidget(HospitalCanvas(
        centers=[[2.5, 30.0], [5.0, 55.0], [7.5, 80.0]], offsets=_offsets,
        names=hospital_names, colors=hospital_colors,
    ))
    hospital_canvas
    return hospital_canvas, hospital_colors, hospital_names


@app.cell
def hospital_readout(
    alt,
    hospital_canvas,
    hospital_colors,
    hospital_names,
    mo,
    pd,
    smf,
    warnings,
):
    _v = hospital_canvas.value
    hospital_df = pd.DataFrame([
        {"hospital": hospital_names[h], "severity": c[0] + o[0], "survival": c[1] + o[1]}
        for h, c in enumerate(_v["centers"]) for o in _v["offsets"][h]
    ])
    _naive = smf.ols("survival ~ severity", hospital_df).fit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _mixed = smf.mixedlm("survival ~ severity", hospital_df, groups=hospital_df["hospital"]).fit(reml=True)
    _g0, _g1 = _mixed.fe_params["Intercept"], _mixed.fe_params["severity"]
    _lines = []
    for _h, _name in enumerate(hospital_names):
        _xs = hospital_df.loc[hospital_df.hospital == _name, "severity"]
        _a = _g0 + _mixed.random_effects[_name].iloc[0]
        for _x in (_xs.min(), _xs.max()):
            _lines.append({"hospital": _name, "severity": _x, "survival": _a + _g1 * _x})
    _color = alt.Color("hospital:N", scale=alt.Scale(domain=hospital_names, range=hospital_colors))
    _pts = alt.Chart(hospital_df).mark_circle(opacity=0.5).encode(x="severity:Q", y=alt.Y("survival:Q", title="survival (%)"), color=_color)
    _cl = alt.Chart(pd.DataFrame(_lines)).mark_line(strokeWidth=3).encode(x="severity:Q", y="survival:Q", color=_color, detail="hospital:N")
    _grid = pd.DataFrame({"severity": [0.0, 10.0]})
    _fix = alt.Chart(_grid.assign(survival=_g0 + _g1 * _grid.severity)).mark_line(color="#222", strokeWidth=3).encode(x="severity:Q", y="survival:Q")
    _nv = alt.Chart(_grid.assign(survival=_naive.params["Intercept"] + _naive.params["severity"] * _grid.severity)).mark_line(
        color="#999", strokeDash=[8, 6], strokeWidth=3).encode(x="severity:Q", y="survival:Q")
    mo.vstack([
        mo.md("**After release, Python fits the real models** (`statsmodels`): naive OLS vs. a random-intercept mixed model."),
        mo.hstack([
            mo.stat(f"{_naive.params['severity']:+.2f}", label="naive OLS slope (gray)", caption=f"p = {_naive.pvalues['severity']:.2g}"),
            mo.stat(f"{_g1:+.2f}", label="mixed model fixed slope (black)", caption=f"p = {_mixed.pvalues['severity']:.2g}"),
            mo.stat("−6.00", label="true within-hospital slope"),
        ]),
        (_pts + _cl + _nv + _fix).properties(width="container", height=300, title="statsmodels MixedLM: colored = hospital lines, black = fixed effect"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Use the **Video layout** first. The gray line slopes *up* (severity seems to help), and the black line slopes *down*. This is the video's paradox.
    2. Click **Same survival level**. Now the gray line also points down, because the hospitals no longer confound the picture.
    3. Drag University Medical to the bottom right. The gray line becomes *steeper* than the truth. Ignoring clusters can flip, inflate, or hide a relationship. This is **consequence #2: you can miss important relationships.**
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · The technical caveat: cluster lines are pulled toward the fixed effect

    The video adds a caveat: *"We don't actually fit separate regression lines for each cluster. The cluster lines are biased toward the fixed effect."* This is called **partial pooling** or **shrinkage**. Each cluster's estimate is a weighted compromise between its own data and the overall average:

    $$\hat\mu_j^{\text{mixed}} = \lambda_j\,\bar y_j + (1-\lambda_j)\,\hat\gamma_0, \qquad \lambda_j = \frac{\tau^2}{\tau^2 + \sigma^2/n_j}$$

    Here, $\tau$ is the spread *between* clusters, $\sigma$ is the noise *within* clusters, and $n_j$ is the cluster size. Small, noisy clusters do not get much say, so they are pulled toward the average. Large clusters keep their own value.

    Below are 8 schools with very different numbers of students. A hollow circle is the school's raw mean (no pooling). A filled circle is the mixed-model estimate. A tick is the true value.
    """)
    return


@app.cell
def pp_controls(mo):
    pp_tau = mo.ui.slider(0.5, 15, step=0.5, value=5, label="τ: true spread between schools", show_value=True)
    pp_sigma = mo.ui.slider(1, 40, step=1, value=15, label="σ: noise within a school", show_value=True)
    pp_roll = mo.ui.button(label="🎲 new sample", value=0, on_click=lambda v: v + 1)
    mo.hstack([mo.vstack([pp_tau, pp_sigma]), pp_roll], justify="start", gap=2)
    return pp_roll, pp_sigma, pp_tau


@app.cell
def pp_readout(alt, mo, np, pd, pp_roll, pp_sigma, pp_tau, smf, warnings):
    _rng = np.random.default_rng(100 + pp_roll.value)
    _sizes = [3, 5, 8, 12, 20, 30, 50, 80]
    _true = 50 + _rng.normal(0, pp_tau.value, len(_sizes))
    _rows = []
    for _j, (_n, _mu) in enumerate(zip(_sizes, _true)):
        for _y in _mu + _rng.normal(0, pp_sigma.value, _n):
            _rows.append({"school": f"school {_j + 1} (n={_n})", "y": _y})
    pp_df = pd.DataFrame(_rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _fit = smf.mixedlm("y ~ 1", pp_df, groups=pp_df["school"]).fit(reml=True)
    _g0 = _fit.fe_params["Intercept"]
    _order = [f"school {_j + 1} (n={_n})" for _j, _n in enumerate(_sizes)]
    _raw = pp_df.groupby("school")["y"].mean()
    pp_est = pd.DataFrame({
        "school": _order,
        "n": _sizes,
        "no pooling (raw mean)": [_raw[s] for s in _order],
        "mixed model": [_g0 + _fit.random_effects[s].iloc[0] for s in _order],
        "truth": _true,
    })
    _tau2, _s2 = float(_fit.cov_re.iloc[0, 0]), float(_fit.scale)
    pp_est["λ (weight on own data)"] = _tau2 / (_tau2 + _s2 / pp_est["n"])
    _y = alt.Y("school:N", sort=_order, title=None)
    _rule = alt.Chart(pp_est).mark_rule(color="#888").encode(y=_y, x="no pooling (raw mean):Q", x2="mixed model:Q")
    _raw_pt = alt.Chart(pp_est).mark_point(size=110, color="#3a86ff", filled=False, strokeWidth=2).encode(
        y=_y, x=alt.X("no pooling (raw mean):Q", title="estimated school mean", scale=alt.Scale(zero=False)),
        tooltip=list(pp_est.columns))
    _mix_pt = alt.Chart(pp_est).mark_circle(size=110, color="#3a86ff").encode(y=_y, x="mixed model:Q")
    _tru = alt.Chart(pp_est).mark_tick(color="#2a9d8f", thickness=3, size=18).encode(y=_y, x="truth:Q")
    _fix = alt.Chart(pd.DataFrame({"x": [_g0]})).mark_rule(color="#222", strokeWidth=2).encode(x="x:Q")
    _err_raw = np.sqrt(np.mean((pp_est["no pooling (raw mean)"] - _true) ** 2))
    _err_mix = np.sqrt(np.mean((pp_est["mixed model"] - _true) ** 2))
    mo.vstack([
        (_fix + _rule + _raw_pt + _mix_pt + _tru).properties(width="container", height=280,
            title="hollow = raw mean → filled = mixed estimate · black = fixed effect · green tick = truth"),
        mo.hstack([
            mo.stat(f"{_err_raw:.2f}", label="RMS error: separate means"),
            mo.stat(f"{_err_mix:.2f}", label="RMS error: mixed model", caption="usually smaller"),
            mo.stat(f"{np.sqrt(_tau2):.2f}", label="estimated τ"),
            mo.stat(f"{np.sqrt(_s2):.2f}", label="estimated σ"),
        ]),
        pp_est.round(2),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Look at school 1 (n = 3) versus school 8 (n = 80). The small school's arrow is long, and the large school's arrow barely moves.
    2. Raise σ to 40. Everything collapses toward the black line, because noisy data earns less trust.
    3. Raise τ to 15 and lower σ to 3. The schools really differ, so the model mostly trusts each school (λ → 1). Re-roll a few times and compare the two RMS errors.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Fixed vs. random effects: the four model shapes

    * **Random**: each cluster gets its **own** value (its own intercept and/or slope).
    * **Fixed**: **everyone gets the same** value, which is the average across clusters.

    The video's examples:
    * **Random intercepts, fixed slopes** (*very common*): calories → weight gain has the *same* slope in every school, but each school starts at its own level. Treatment effects in therapist or school studies, and the male–female gap across schools, are other examples. Result: **parallel lines**.
    * **Random intercepts and random slopes** (*probably the most common*): every cluster gets its own line.
    * **Fixed intercepts, random slopes** (*rare and weird*): all lines start at the same point and **fan out**. *"I've never seen this in practice."*

    $$y_{ij} = (\gamma_0 + u_{0j}) + (\gamma_1 + u_{1j})\,x_{ij} + \varepsilon_{ij}$$
    Remove $u_{0j}$ to make the intercept fixed, and remove $u_{1j}$ to make the slope fixed.

    **Build the shapes yourself:** drag a **circle** (at x = 0) to move the intercept of a line, and drag a **square** (at x = 10) to tilt the line. The black dashed line is the fixed effect (the average line).
    """)
    return


@app.cell
def LineBuilder_class(anywidget, traitlets):
    class LineBuilder(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 640, H = 340, PADL = 40, PADR = 30, PADT = 14, PADB = 30;
      const NS = "http://www.w3.org/2000/svg";
      const bar = document.createElement("div"); bar.style.marginBottom = "6px"; el.appendChild(bar);
      const svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none" }); el.appendChild(svg);
      const mk = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); svg.appendChild(e); return e; };
      const sx = v => PADL + v / 10 * (W - PADL - PADR), sy = v => H - PADB - v / 100 * (H - PADB - PADT);
      const iy = p => (H - PADB - p) / (H - PADB - PADT) * 100;
      const toSvg = e => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const presets = {
        "Random intercepts + slopes": [[15, 6], [35, 2.5], [55, 3.5], [25, 4.5]],
        "Random intercepts (fixed slope)": [[10, 4], [25, 4], [40, 4], [55, 4]],
        "Random slopes (fixed intercept)": [[30, 1], [30, 3], [30, 5], [30, 6.5]],
        "Everything fixed": [[30, 4], [30, 4], [30, 4], [30, 4]],
      };
      for (const name in presets) { const b = document.createElement("button"); b.textContent = name;
        b.style.cssText = "margin:0 6px 4px 0;padding:3px 9px;border-radius:6px;border:1px solid #888;background:transparent;color:inherit;cursor:pointer";
        b.onclick = () => { model.set("lines", presets[name]); model.save_changes(); }; bar.appendChild(b); }
      let drag = null;
      function draw() {
        const L = model.get("lines"), cols = model.get("colors"); svg.innerHTML = "";
        mk("rect", { x: PADL, y: PADT, width: W - PADL - PADR, height: H - PADB - PADT, fill: "none", stroke: "#8884" });
        mk("text", { x: (W + PADL) / 2, y: H - 8, "text-anchor": "middle", "font-size": 12, fill: "currentColor" }).textContent = "x (predictor)";
        L.forEach(([a, b], k) => {
          mk("line", { x1: sx(0), y1: sy(a), x2: sx(10), y2: sy(a + 10 * b), stroke: cols[k], "stroke-width": 3 });
        });
        const fa = L.reduce((s, l) => s + l[0], 0) / L.length, fb = L.reduce((s, l) => s + l[1], 0) / L.length;
        mk("line", { x1: sx(0), y1: sy(fa), x2: sx(10), y2: sy(fa + 10 * fb), stroke: "currentColor", "stroke-width": 3, "stroke-dasharray": "9 6" });
        L.forEach(([a, b], k) => {
          const c = mk("circle", { cx: sx(0), cy: sy(a), r: 9, fill: cols[k], stroke: "#fff", "stroke-width": 2 });
          c.style.cursor = "ns-resize"; c.addEventListener("pointerdown", e => { drag = [k, "a"]; svg.setPointerCapture(e.pointerId); });
          const s = mk("rect", { x: sx(10) - 8, y: sy(a + 10 * b) - 8, width: 16, height: 16, fill: cols[k], stroke: "#fff", "stroke-width": 2 });
          s.style.cursor = "ns-resize"; s.addEventListener("pointerdown", e => { drag = [k, "b"]; svg.setPointerCapture(e.pointerId); });
        });
      }
      svg.addEventListener("pointermove", e => { if (!drag) return; const y = Math.max(0, Math.min(100, iy(toSvg(e).y)));
        const L = model.get("lines").map(l => [...l]); const [k, which] = drag;
        if (which === "a") L[k][0] = Math.round(y); else L[k][1] = Math.round((y - L[k][0])) / 10;
        model.set("lines", L); });
      const stop = () => { if (drag) { drag = null; model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointerleave", stop);
      model.on("change:lines", draw); draw();
    }
    export default { render };
    """
        lines = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)
        colors = traitlets.List(traitlets.Unicode()).tag(sync=True)

    return (LineBuilder,)


@app.cell
def line_builder(LineBuilder, mo):
    line_builder = mo.ui.anywidget(LineBuilder(
        lines=[[15.0, 6.0], [35.0, 2.5], [55.0, 3.5], [25.0, 4.5]],
        colors=["#e4572e", "#2a9d8f", "#3a86ff", "#f4a261"],
    ))
    line_builder
    return (line_builder,)


@app.cell
def line_readout(line_builder, mo, np):
    def classify_lines(lines, tol_a=1.5, tol_b=0.15):
        """Name the model type from a set of cluster lines [(intercept, slope), ...]."""
        _L = np.asarray(lines, dtype=float)
        _ra, _rb = np.ptp(_L[:, 0]) > tol_a, np.ptp(_L[:, 1]) > tol_b
        return {
            (True, True): ("Random intercepts + random slopes", 'smf.mixedlm("y ~ x", df, groups="g", re_formula="~x")', "lme4: y ~ x + (1 + x | g)", "Very common: every cluster gets its own line."),
            (True, False): ("Random intercepts, fixed slope", 'smf.mixedlm("y ~ x", df, groups="g")', "lme4: y ~ x + (1 | g)", "Very common: parallel lines (calories → weight gain)."),
            (False, True): ("Fixed intercept, random slopes", 'smf.mixedlm("y ~ x", df, groups="g", re_formula="0 + x")', "lme4: y ~ x + (0 + x | g)", "Rare and weird: lines fan out from one point."),
            (False, False): ("Everything fixed (ordinary regression)", 'smf.ols("y ~ x", df)', "lm: y ~ x", "No random effects: only appropriate if clusters truly do not differ."),
        }[(_ra, _rb)]

    _L = np.asarray(line_builder.value["lines"])
    _name, _sm, _r, _note = classify_lines(_L)
    mo.vstack([
        mo.hstack([
            mo.stat(f"{_L[:, 0].mean():.1f}", label="fixed intercept γ₀ (average)"),
            mo.stat(f"{_L[:, 0].std():.1f}", label="SD of intercepts (τ₀)"),
            mo.stat(f"{_L[:, 1].mean():.2f}", label="fixed slope γ₁ (average)"),
            mo.stat(f"{_L[:, 1].std():.2f}", label="SD of slopes (τ₁)"),
        ]),
        mo.callout(mo.md(f"### {_name}\n{_note}\n\n`{_sm}`  \n`{_r}`"), kind="success" if "Fixed intercept" not in _name else "warn"),
    ])
    return (classify_lines,)


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Click **Random intercepts (fixed slope)**, then drag a circle up and down. The lines stay parallel, so the model type does not change.
    2. Tilt just one square. Now one cluster has its own slope, and the model becomes **random intercepts + slopes**.
    3. Click **Random slopes (fixed intercept)**. All lines start at one point. The video says to *"almost never fix the intercept"*. The question to ask for each new predictor is: **"Should this slope be the same in every cluster?"**
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Sandbox: generate clustered data and fit any model

    Choose the true data-generating process, choose a model, and compare the estimated fixed slope (and its standard error) with the truth. The **between-cluster confound** shifts clusters that have high *x* upward or downward, as in the hospital example.
    """)
    return


@app.cell
def sb_controls(mo):
    sb_J = mo.ui.slider(3, 30, value=10, label="clusters", show_value=True)
    sb_n = mo.ui.slider(3, 50, value=15, label="observations per cluster", show_value=True)
    sb_tau0 = mo.ui.slider(0, 20, step=1, value=8, label="τ₀ intercept SD", show_value=True)
    sb_tau1 = mo.ui.slider(0.0, 3.0, step=0.1, value=0.8, label="τ₁ slope SD", show_value=True)
    sb_g1 = mo.ui.slider(-5.0, 5.0, step=0.5, value=-2.0, label="true fixed slope γ₁", show_value=True)
    sb_conf = mo.ui.slider(-15.0, 15.0, step=1.0, value=10.0, label="between-cluster confound", show_value=True)
    sb_sigma = mo.ui.slider(1, 15, value=4, label="σ residual noise", show_value=True)
    sb_model = mo.ui.dropdown(["Naive OLS (ignore clusters)", "Random intercepts", "Random intercepts + slopes"],
                              value="Random intercepts + slopes", label="Model to fit")
    sb_roll = mo.ui.button(label="🎲 re-roll data", value=0, on_click=lambda v: v + 1)
    mo.hstack([mo.vstack([sb_J, sb_n, sb_tau0, sb_tau1]), mo.vstack([sb_g1, sb_conf, sb_sigma, sb_model, sb_roll])])
    return (
        sb_J,
        sb_conf,
        sb_g1,
        sb_model,
        sb_n,
        sb_roll,
        sb_sigma,
        sb_tau0,
        sb_tau1,
    )


@app.cell
def sb_readout(
    alt,
    mo,
    np,
    pd,
    sb_J,
    sb_conf,
    sb_g1,
    sb_model,
    sb_n,
    sb_roll,
    sb_sigma,
    sb_tau0,
    sb_tau1,
    smf,
    warnings,
):
    _rng = np.random.default_rng(7 + sb_roll.value)
    _rows = []
    for _j in range(sb_J.value):
        _c = _rng.uniform(2, 8)
        _u0, _u1 = _rng.normal(0, sb_tau0.value), _rng.normal(0, sb_tau1.value)
        _x = _c + _rng.normal(0, 1.2, sb_n.value)
        _y = 50 + _u0 + sb_conf.value * (_c - 5) + (sb_g1.value + _u1) * (_x - 5) + _rng.normal(0, sb_sigma.value, sb_n.value)
        _rows += [{"g": f"c{_j:02d}", "x": a, "y": b} for a, b in zip(_x, _y)]
    sb_df = pd.DataFrame(_rows)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if sb_model.value.startswith("Naive"):
            _fit = smf.ols("y ~ x", sb_df).fit()
            _b0, _b1, _se = _fit.params["Intercept"], _fit.params["x"], _fit.bse["x"]
            _cl = {g: (_b0, _b1) for g in sb_df.g.unique()}
        else:
            _re = "~x" if "slopes" in sb_model.value else None
            _fit = smf.mixedlm("y ~ x", sb_df, groups=sb_df["g"], re_formula=_re).fit(reml=True)
            _b0, _b1, _se = _fit.fe_params["Intercept"], _fit.fe_params["x"], _fit.bse_fe["x"]
            _cl = {g: (_b0 + r.iloc[0], _b1 + (r.iloc[1] if len(r) > 1 else 0.0)) for g, r in _fit.random_effects.items()}
    _seg = []
    for _g, (_a, _b) in _cl.items():
        _xs = sb_df.loc[sb_df.g == _g, "x"]
        _seg += [{"g": _g, "x": v, "y": _a + _b * v} for v in (_xs.min(), _xs.max())]
    _col = alt.Color("g:N", legend=None, scale=alt.Scale(scheme="tableau20"))
    _pts = alt.Chart(sb_df).mark_circle(opacity=0.45, size=30).encode(x="x:Q", y=alt.Y("y:Q", scale=alt.Scale(zero=False)), color=_col)
    _lin = alt.Chart(pd.DataFrame(_seg)).mark_line(strokeWidth=2).encode(x="x:Q", y="y:Q", color=_col, detail="g:N")
    _grid = pd.DataFrame({"x": [0.0, 10.0]})
    _fx = alt.Chart(_grid.assign(y=_b0 + _b1 * _grid.x)).mark_line(
        color="#999" if sb_model.value.startswith("Naive") else "#222", strokeWidth=4,
        strokeDash=[8, 6] if sb_model.value.startswith("Naive") else [1, 0]).encode(x="x:Q", y="y:Q")
    _cover = abs(_b1 - sb_g1.value) < 1.96 * _se
    mo.vstack([
        mo.hstack([
            mo.stat(f"{sb_g1.value:+.2f}", label="true fixed slope"),
            mo.stat(f"{_b1:+.2f}", label="estimated slope"),
            mo.stat(f"{_se:.3f}", label="standard error"),
            mo.stat("yes ✅" if _cover else "no ❌", label="95% CI covers truth?"),
        ]),
        (_pts + _lin + _fx).properties(width="container", height=340, title=sb_model.value),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Challenges**
    1. Use a confound of +10 and a true slope of −2. Switch between **Naive OLS** and a mixed model. Which one gets the sign right?
    2. Set the confound to 0 and τ₁ = 2. Compare the standard error from **Random intercepts** with the one from **Random intercepts + slopes**. A model that ignores varying slopes is overconfident. Re-roll and watch the CI coverage.
    3. Set τ₀ = τ₁ = 0 (no clustering). All three models agree, so the mixed model costs you almost nothing.
    """), kind="neutral")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Quiz: test your knowledge

    Like the end of the video, first identify the model type from each picture, then answer a few concept questions. Turn on **Reveal** to score yourself.
    """)
    return


@app.cell
def quiz_plots(alt, classify_lines, pd):
    _cols = ["#e4572e", "#2a9d8f", "#3a86ff", "#f4a261"]
    _quiz_lines = {
        "Plot A": [(20, 4), (35, 4), (50, 4), (10, 4)],
        "Plot B": [(10, 6), (40, 2), (25, 5.5), (55, 1)],
        "Plot C": [(30, 0.5), (30, 2.5), (30, 4.5), (30, 6.5)],
    }
    _charts = []
    for _t, _ls in _quiz_lines.items():
        _d = pd.DataFrame([{"k": str(_k), "x": x, "y": a + b * x} for _k, (a, b) in enumerate(_ls) for x in (0, 10)])
        _charts.append(alt.Chart(_d).mark_line(strokeWidth=3).encode(
            x=alt.X("x:Q", title=None), y=alt.Y("y:Q", title=None, scale=alt.Scale(domain=[0, 100])),
            color=alt.Color("k:N", legend=None, scale=alt.Scale(range=_cols))).properties(width=190, height=150, title=_t))
    quiz_truth_plots = {t: classify_lines(ls)[0] for t, ls in _quiz_lines.items()}
    alt.hconcat(*_charts)
    return (quiz_truth_plots,)


@app.cell
def quiz(mo):
    _types = ["Random intercepts + random slopes", "Random intercepts, fixed slope", "Fixed intercept, random slopes", "Everything fixed (ordinary regression)"]
    quiz = mo.ui.dictionary({
        "Plot A": mo.ui.dropdown(_types, label="Plot A is"),
        "Plot B": mo.ui.dropdown(_types, label="Plot B is"),
        "Plot C": mo.ui.dropdown(_types, label="Plot C is"),
        "q_dup": mo.ui.dropdown(["gets smaller", "stays the same", "gets larger"], label="Duplicating your dataset makes the p-value..."),
        "q_icc": mo.ui.dropdown(["0", "0.5", "1"], label="At which ICC ρ is there no clustering problem?"),
        "q_black": mo.ui.dropdown(["a random effect", "the fixed effect", "the naive OLS line"], label="The average line across all cluster lines is..."),
        "q_rare": mo.ui.dropdown(["random intercepts", "fixed slopes", "fixed intercepts"], label="What does the video say we almost never do?"),
        "q_names": mo.ui.dropdown(["HLM is a special case", "they are the same thing", "only multilevel models handle clustering"], label="HLM vs mixed vs multilevel models:"),
    })
    quiz_reveal = mo.ui.switch(label="Reveal answers")
    mo.vstack([quiz, quiz_reveal])
    return quiz, quiz_reveal


@app.cell
def quiz_score(mo, quiz, quiz_reveal, quiz_truth_plots):
    _answers = dict(quiz_truth_plots, q_dup="gets smaller", q_icc="0", q_black="the fixed effect", q_rare="fixed intercepts", q_names="they are the same thing")
    _why = {
        "Plot A": "The lines are parallel, so the slope is fixed and the intercepts vary.",
        "Plot B": "The lines have different starts and different tilts.",
        "Plot C": "The lines fan out from one point. This is the rare model.",
        "q_dup": "n grows, but no information is added, so you get false certainty.",
        "q_icc": "With ρ = 0, people in the same cluster are no more alike than strangers are.",
        "q_black": "Fixed = the same for everyone, which is the average across clusters.",
        "q_rare": "'We almost never do a fixed intercept.'",
        "q_names": "These are all names for the same model.",
    }
    _score = sum(quiz.value[k] == v for k, v in _answers.items())
    if not quiz_reveal.value:
        _out = mo.md(f"_Answered {sum(v is not None for v in quiz.value.values())}/{len(_answers)}. Flip **Reveal** to check._")
    else:
        _rows = "\n".join(f"| {k} | {quiz.value[k] or '–'} | {v} | {'✅' if quiz.value[k] == v else '❌'} | {_why[k]} |" for k, v in _answers.items())
        _out = mo.vstack([
            mo.stat(f"{_score}/{len(_answers)}", label="score"),
            mo.md("| question | yours | answer | | why |\n|---|---|---|---|---|\n" + _rows),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap (the video's learning objectives)

    | Objective | Key idea | Where you touched it |
    |---|---|---|
    | Independence and "duplicating" data | Statistical tests assume independent rows. A copy adds *n* but no information | §1: copies slider |
    | Two consequences of violating it | ① *p*-values are too low (Type I errors) ② relationships can be missed or reversed | §2 simulator, §3 hospitals |
    | HLM = mixed = multilevel = mixed-effects | These are all the same model. They need **long-format** data (one row per observation) | Quiz |
    | What mixed models do geometrically | A line per cluster, averaged into one line. Cluster lines are **pulled toward the average** | §3 drag, §4 shrinkage |
    | Fixed vs. random effects | Fixed = the same for everyone (the average). Random = unique to each cluster | §5 line builder |
    | Model shapes | Random intercepts + slopes (most common) · random intercepts / fixed slope (parallel) · fixed intercept / random slopes (rare) | §5, Quiz |
    | Practical rule | Almost always let intercepts vary. For each predictor, ask: "Is this slope the same in every cluster?" | §5, §6 |

    **Formula cheat sheet (statsmodels):** `smf.mixedlm("y ~ x", df, groups="g")` = random intercepts · add `re_formula="~x"` for random slopes too.
    """)
    return


if __name__ == "__main__":
    app.run()

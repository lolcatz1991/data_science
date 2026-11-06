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
    # The Cox proportional hazards model, explained interactively
    *An explorable companion to TileStats' video **"The Cox proportional hazards model explained"**.*

    Cox regression answers one question: **how much does a variable multiply your risk of dying *right now*?**
    It never says *when* anyone dies. It only compares people who are still alive at the same moment.
    """),
        mo.Html('<iframe width="560" height="315" src="https://www.youtube.com/embed/-g6YLWG2yYo" title="The Cox proportional hazards model explained" frameborder="0" allowfullscreen></iframe>'),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Roadmap (same order as the video).**
    1. From the Kaplan–Meier curve to the **hazard** $h(t)$ and the **cumulative hazard** $H(t)$
    2. Two groups, one **hazard ratio**: $h_A(t) = h_0(t)\,e^{\beta}$
    3. How Cox actually fits $\beta$: the **partial likelihood**, plus the R-style output (HR, CI, p-value, $1/\text{HR}$)
    4. Beyond the log-rank test: **adjusting for age**
    5. Checking the **proportional hazards assumption** with a log-minus-log plot

    Then a sandbox (a live survival race), a quiz and a recap.

    **Color language used everywhere:**
    <span style="color:#e4572e">■ group A (old treatment, $x=1$)</span> ·
    <span style="color:#2e86ab">■ group B (new treatment, the baseline, $x=0$)</span> ·
    <span style="color:#6c5b7b">■ the key quantity ($\beta$, the hazard ratio, the fitted curve)</span> ·
    <span style="color:#999">■ censored / not at risk</span>

    *A separate notebook (`survival_analysis.py`) covers Kaplan–Meier basics; here we focus on Cox.*
    """)
    return


@app.cell
def helpers(np, pd, stats):
    def km_table(times, events):
        """Interval table from the video: one row per distinct death time.
        hazard = deaths / (at risk * interval length); H accumulates hazard * length."""
        _t = np.asarray(times, float); _e = np.asarray(events, int)
        _rows, _prev, _S, _H = [], 0.0, 1.0, 0.0
        for _u in np.unique(_t[_e == 1]):
            _n = int((_t >= _u).sum()); _d = int(((_t == _u) & (_e == 1)).sum())
            _dt = _u - _prev
            _h = _d / (_n * _dt) if _dt > 0 else np.nan
            _S *= 1 - _d / _n
            _H += _d / _n
            _rows.append(dict(start=_prev, end=_u, at_risk=_n, deaths=_d, dt=_dt,
                              hazard=_h, S=_S, H=_H))
            _prev = _u
        return pd.DataFrame(_rows)


    def km_curve(times, events):
        """Step-function points (t, S) for plotting a Kaplan-Meier curve."""
        _tab = km_table(times, events)
        _pts, _S = [(0.0, 1.0)], 1.0
        for _r in _tab.itertuples():
            _pts.append((_r.end, _S)); _S = _r.S; _pts.append((_r.end, _S))
        _pts.append((float(np.max(times)), _S))
        return pd.DataFrame(_pts, columns=["t", "S"])


    def cox_partial(time, event, X, beta):
        """Breslow log partial likelihood, gradient and Hessian."""
        _eta = X @ beta; _w = np.exp(_eta)
        _ll = 0.0; _g = np.zeros(X.shape[1]); _Hs = np.zeros((X.shape[1],) * 2)
        for _u in np.unique(time[event == 1]):
            _D = (time == _u) & (event == 1); _R = time >= _u; _d = _D.sum()
            _s0 = _w[_R].sum(); _s1 = _w[_R] @ X[_R]; _s2 = (X[_R] * _w[_R, None]).T @ X[_R]
            _ll += _eta[_D].sum() - _d * np.log(_s0)
            _g += X[_D].sum(0) - _d * _s1 / _s0
            _Hs -= _d * (_s2 / _s0 - np.outer(_s1, _s1) / _s0**2)
        return _ll, _g, _Hs


    def cox_fit(time, event, X, names=None):
        """Newton-Raphson Cox fit. Returns a coefficient table (R-style) and log-likelihoods."""
        time = np.asarray(time, float); event = np.asarray(event, int)
        X = np.asarray(X, float).reshape(len(time), -1)
        _b = np.zeros(X.shape[1])
        _ll0 = cox_partial(time, event, X, _b)[0]
        _ll, _g, _Hs = cox_partial(time, event, X, _b)
        for _ in range(60):
            _step = np.linalg.solve(_Hs, _g)
            _new = _b - _step; _s = 1.0
            while cox_partial(time, event, X, _new)[0] < _ll - 1e-10 and _s > 1e-4:
                _s /= 2; _new = _b - _s * _step
            _b = _new
            _ll_new, _g, _Hs = cox_partial(time, event, X, _b)
            if abs(_ll_new - _ll) < 1e-10: _ll = _ll_new; break
            _ll = _ll_new
        _se = np.sqrt(np.diag(np.linalg.inv(-_Hs)))
        _z = _b / _se
        _tab = pd.DataFrame({
            "coef": _b, "exp(coef)": np.exp(_b), "se(coef)": _se, "z": _z,
            "p": 2 * stats.norm.sf(np.abs(_z)),
            "exp(-coef)": np.exp(-_b),
            "lower .95": np.exp(_b - 1.96 * _se), "upper .95": np.exp(_b + 1.96 * _se),
        }, index=names or [f"x{_k+1}" for _k in range(X.shape[1])])
        return dict(table=_tab, beta=_b, se=_se, loglik=_ll, loglik0=_ll0)

    COL_A, COL_B, COL_KEY, COL_GREY = "#e4572e", "#2e86ab", "#6c5b7b", "#999999"
    return (
        COL_A,
        COL_B,
        COL_GREY,
        COL_KEY,
        cox_fit,
        cox_partial,
        km_curve,
        km_table,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · From the Kaplan–Meier curve to the hazard

    The video starts with **six subjects**. One dies after 1 year ($S = 5/6 \approx 0.83$), the next after 6 years ($S=4/6\approx0.67$), and so on.

    The **hazard** is the risk of dying in the next small time step, *given you survived until $t$*:

    $$h(t) = \lim_{\Delta t\to 0}\frac{P(t \le T < t+\Delta t \mid T \ge t)}{\Delta t} \;=\; -\frac{S'(t)}{S(t)}$$

    With yearly steps, as in the video, we estimate it per interval as
    $\hat h = \dfrac{\text{deaths}}{\text{at risk}\times\text{interval length}}$.
    Between year 1 and year 6, one of five people died over **five** years, so $\hat h = 1/(5\cdot5)=0.04$ per year.

    The **cumulative hazard** adds up hazard × time: $H(t)=\int_0^t h(u)\,du$.
    *(The video's description corrects an error at 4:12: you must multiply each hazard by its interval length,
    giving 0.1667, 0.3667, 0.6167, 1.2833, 2.2833.)*

    **Drag a subject's end marker** left/right to change when they die. **Click** a marker to toggle death (●) ↔ censored (○).
    """)
    return


@app.cell
def DeathTimeline_cls(anywidget, traitlets):
    class DeathTimeline(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 640, H = 360, PADL = 70, PADR = 20, XMAX = 15;
      const NS = "http://www.w3.org/2000/svg";
      const bar = document.createElement("div"); bar.style.marginBottom = "4px"; el.appendChild(bar);
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none", fontFamily: "sans-serif" });
      el.appendChild(svg);
      const mk = (tag, a, p = svg) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
      const toSvg = (evt) => { const p = svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const sx = v => PADL + v / XMAX * (W - PADL - PADR);
      const inv = px => (px - PADL) / (W - PADL - PADR) * XMAX;
      const KT = 20, KB = 140, sy = s => KB - s * (KB - KT);
      const btn = (label, fn) => { const b = document.createElement("button"); b.textContent = label; b.onclick = fn; b.style.marginRight = "6px"; bar.appendChild(b); };
      const push = (t, e) => { model.set("times", t); model.set("events", e); model.save_changes(); };
      btn("+ subject", () => { const t = [...model.get("times")], e = [...model.get("events")]; if (t.length < 10) { t.push(8); e.push(1); push(t, e); } });
      btn("- subject", () => { const t = [...model.get("times")], e = [...model.get("events")]; if (t.length > 2) { t.pop(); e.pop(); push(t, e); } });
      btn("reset to video", () => push([1, 6, 10, 11, 11, 12], [1, 1, 1, 1, 1, 1]));
      let drag = -1, moved = false;
      function draw() {
        const ts = model.get("times"), es = model.get("events");
        svg.innerHTML = "";
        // KM curve computed in JS for instant feedback
        mk("text", { x: 4, y: KT + 4, "font-size": 11, fill: "currentColor" }).textContent = "S(t) = 1";
        mk("text", { x: 4, y: KB + 4, "font-size": 11, fill: "currentColor" }).textContent = "S(t) = 0";
        mk("line", { x1: PADL, x2: W - PADR, y1: KB, y2: KB, stroke: "#bbb" });
        mk("line", { x1: PADL, x2: PADL, y1: KT, y2: KB, stroke: "#bbb" });
        const ut = [...new Set(ts.filter((t, i) => es[i] === 1))].sort((a, b) => a - b);
        let S = 1, prev = 0, d = `M ${sx(0)} ${sy(1)}`;
        for (const u of ut) {
          const n = ts.filter(t => t >= u).length, dd = ts.filter((t, i) => t === u && es[i] === 1).length;
          d += ` L ${sx(u)} ${sy(S)}`; S *= 1 - dd / n; d += ` L ${sx(u)} ${sy(S)}`;
          mk("text", { x: sx(u) + 3, y: sy(S) - 4, "font-size": 10, fill: "#6c5b7b" }).textContent = S.toFixed(2);
        }
        d += ` L ${sx(Math.max(...ts))} ${sy(S)}`;
        mk("path", { d, fill: "none", stroke: "#6c5b7b", "stroke-width": 3 });
        // time axis
        const AX = H - 18;
        mk("line", { x1: PADL, x2: W - PADR, y1: AX, y2: AX, stroke: "#bbb" });
        for (let k = 0; k <= XMAX; k += 1) {
          mk("line", { x1: sx(k), x2: sx(k), y1: AX, y2: AX + 4, stroke: "#bbb" });
          if (k % 2 === 0) mk("text", { x: sx(k), y: AX + 15, "font-size": 10, "text-anchor": "middle", fill: "currentColor" }).textContent = k;
        }
        mk("text", { x: W - PADR, y: AX - 4, "font-size": 10, "text-anchor": "end", fill: "currentColor" }).textContent = "years";
        // lanes
        ts.forEach((t, i) => {
          const y = 165 + i * 17;
          mk("text", { x: 4, y: y + 4, "font-size": 11, fill: "currentColor" }).textContent = "subject " + (i + 1);
          mk("line", { x1: sx(0), x2: sx(t), y1: y, y2: y, stroke: es[i] ? "#6c5b7b" : "#999", "stroke-width": 4, "stroke-linecap": "round" });
          const c = mk("circle", { cx: sx(t), cy: y, r: 7, fill: es[i] ? "#6c5b7b" : "white", stroke: es[i] ? "white" : "#999", "stroke-width": 2 });
          c.style.cursor = "grab";
          c.addEventListener("pointerdown", e => { drag = i; moved = false; svg.setPointerCapture(e.pointerId); });
        });
      }
      svg.addEventListener("pointermove", e => {
        if (drag < 0) return;
        const t = [...model.get("times")];
        const v = Math.max(0.5, Math.min(XMAX, Math.round(inv(toSvg(e).x) * 2) / 2));
        if (v !== t[drag]) { t[drag] = v; moved = true; model.set("times", t); }
      });
      const stop = () => {
        if (drag < 0) return;
        if (!moved) { const e = [...model.get("events")]; e[drag] = 1 - e[drag]; model.set("events", e); }
        drag = -1; model.save_changes();
      };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointerleave", stop);
      model.on("change:times", draw); model.on("change:events", draw); draw();
    }
    export default { render };
    """
        times = traitlets.List(traitlets.Float()).tag(sync=True)
        events = traitlets.List(traitlets.Int()).tag(sync=True)

    return (DeathTimeline,)


@app.cell
def timeline(DeathTimeline, mo):
    timeline = mo.ui.anywidget(DeathTimeline(times=[1.0, 6.0, 10.0, 11.0, 11.0, 12.0], events=[1] * 6))
    timeline
    return (timeline,)


@app.cell
def hazard_readout(COL_KEY, alt, km_table, mo, np, pd, timeline):
    _t = timeline.value["times"]; _e = timeline.value["events"]
    mo.stop(sum(_e) == 0, mo.callout(mo.md("Every subject is censored: no deaths, so no hazard can be estimated. Click a marker to make it a death."), kind="warn"))
    hazard_tab = km_table(_t, _e)
    _steps = []
    for _r in hazard_tab.itertuples():
        _steps += [dict(t=_r.start, h=_r.hazard), dict(t=_r.end, h=_r.hazard)]
    _hz = pd.DataFrame(_steps)
    _Hc = pd.concat([pd.DataFrame([dict(t=0.0, H=0.0)]), hazard_tab.rename(columns={"end": "t"})[["t", "H"]]])
    _hchart = alt.Chart(_hz).mark_line(color=COL_KEY, strokeWidth=3).encode(
        x=alt.X("t:Q", title="years", scale=alt.Scale(domain=[0, 15])), y=alt.Y("h:Q", title="hazard h(t) per year")
    ).properties(width=300, height=200, title="Hazard (piecewise constant)")
    _Hchart = alt.Chart(_Hc).mark_line(color=COL_KEY, strokeWidth=3, point=True, interpolate="linear").encode(
        x=alt.X("t:Q", title="years", scale=alt.Scale(domain=[0, 15])), y=alt.Y("H:Q", title="cumulative hazard H(t)")
    ).properties(width=300, height=200, title="Cumulative hazard H(t) = Σ h·Δt")
    mo.vstack([
        mo.hstack([
            mo.stat(f"{hazard_tab.hazard.iloc[0]:.4f}", label="hazard in first interval", caption="video: 0.1667"),
            mo.stat(f"{hazard_tab.H.iloc[-1]:.4f}", label="H at last death", caption="video (corrected): 2.2833"),
            mo.stat(f"{np.exp(-hazard_tab.H.iloc[-1]):.3f}", label="exp(-H)", caption=f"KM S = {hazard_tab.S.iloc[-1]:.3f}"),
        ], justify="start"),
        alt.hconcat(_hchart, _Hchart),
        mo.ui.table(hazard_tab.round(4), selection=None, label="The video's interval table"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - With the video's data, read off $\hat h$ for 1–6 years (0.04) and 10–11 years (0.667). Two of three people died within one year, which is why the hazard spikes.
    - Drag subject 2 from year 6 to year 2. The *same* death now happens over a short interval, so the hazard **jumps from 0.04 to 0.2**, while $H$ at the end hardly changes.
    - Click a marker to **censor** it. That person still counts in the "at risk" denominator until they leave, but adds no death.
    - Note that $H$ here equals the **Nelson–Aalen** sum $\sum d/n$, because $h\cdot\Delta t = d/n$. Compare $e^{-H}$ with the KM value.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Two groups and the hazard ratio

    Now 12 subjects: group **<span style="color:#e4572e">A</span>** (old treatment) and group **<span style="color:#2e86ab">B</span>** (new treatment).
    B is the **baseline** ($x=0$), A has $x=1$.

    The video's rough intuition: take the **log** of each group's estimated hazards, and fit two straight lines that are
    forced to have the **same slope**. Then the lines differ only by a vertical shift $\beta$:

    $$\log h_A(t) = \underbrace{a + b\,t}_{\log h_0(t)} + \beta\cdot 1 \quad\Longrightarrow\quad h_A(t) = h_0(t)\,e^{\beta}$$

    so $e^{\beta}$ is the **hazard ratio (HR)**: at every time point, A's hazard is $e^\beta$ times B's.
    That is the whole Cox model: $\;h(t\mid x) = h_0(t)\,e^{\beta x}$. **Parallel on the log scale = proportional on the raw scale.**

    *(The real Cox fit doesn't do this regression; it uses the partial likelihood of section 3. The video's simplified fit gives β≈1.1 and the true Cox fit gives 1.25. Our made-up data lands in a similar spot.)*
    """)
    return


@app.cell
def two_groups(cox_fit, km_table, np, pd):
    two_groups = pd.DataFrame({
        "time":  [1, 2, 4, 5, 7, 9,   1, 6, 10, 11, 11, 12],
        "event": [1] * 12,
        "x":     [1] * 6 + [0] * 6,
    })
    two_groups["group"] = np.where(two_groups.x == 1, "A", "B")

    _pts = []
    for _g in ["A", "B"]:
        _d = two_groups[two_groups.group == _g]
        _tab = km_table(_d.time, _d.event)
        _pts.append(_tab.assign(group=_g, x=int(_g == "A"))[["end", "hazard", "group", "x"]])
    hazard_points = pd.concat(_pts).rename(columns={"end": "t"}).reset_index(drop=True)

    # the video's "simplified" model: OLS on log hazards with a shared slope
    _Xd = np.column_stack([np.ones(len(hazard_points)), hazard_points.t, hazard_points.x])
    ols_coef = np.linalg.lstsq(_Xd, np.log(hazard_points.hazard), rcond=None)[0]
    cox_two = cox_fit(two_groups.time, two_groups.event, two_groups[["x"]].values, names=["groupA"])
    hazard_points
    return cox_two, hazard_points, ols_coef, two_groups


@app.cell
def beta_hr(mo):
    beta_hr = mo.ui.slider(-1.0, 3.0, step=0.05, value=0.0, label="your β (shift of group A's log-hazard line)", show_value=True)
    log_scale = mo.ui.switch(value=True, label="log scale for hazard")
    mo.hstack([beta_hr, log_scale], justify="start")
    return beta_hr, log_scale


@app.cell
def hr_chart(
    COL_A,
    COL_B,
    COL_KEY,
    alt,
    beta_hr,
    cox_two,
    hazard_points,
    log_scale,
    mo,
    np,
    ols_coef,
    pd,
):
    _a, _b, _ = ols_coef
    _tt = np.linspace(0.5, 12.5, 60)
    _lines = pd.concat([
        pd.DataFrame(dict(t=_tt, h=np.exp(_a + _b * _tt), curve="B: baseline h₀(t)")),
        pd.DataFrame(dict(t=_tt, h=np.exp(_a + _b * _tt + beta_hr.value), curve="A: h₀(t)·e^β (your β)")),
    ])
    _yscale = alt.Scale(type="log") if log_scale.value else alt.Scale()
    _cscale = alt.Scale(domain=["A", "B"], range=[COL_A, COL_B])
    _pts = alt.Chart(hazard_points).mark_circle(size=110).encode(
        x=alt.X("t:Q", title="years"), y=alt.Y("hazard:Q", title="hazard", scale=_yscale),
        color=alt.Color("group:N", scale=_cscale), tooltip=["group", "t", alt.Tooltip("hazard:Q", format=".3f")])
    _ln = alt.Chart(_lines).mark_line(strokeWidth=3).encode(
        x="t:Q", y=alt.Y("h:Q", scale=_yscale),
        color=alt.Color("curve:N", scale=alt.Scale(range=[COL_KEY, COL_B]), title="fitted"),
        strokeDash=alt.StrokeDash("curve:N", legend=None))
    _err = np.mean((np.log(hazard_points.hazard) - (_a + _b * hazard_points.t + beta_hr.value * hazard_points.x))[hazard_points.x == 1] ** 2)
    mo.vstack([
        (_pts + _ln).resolve_scale(color="independent").properties(width="container", height=300,
            title="Estimated hazards per interval, with the shared-slope model"),
        mo.hstack([
            mo.stat(f"{beta_hr.value:.2f}", label="your β"),
            mo.stat(f"{np.exp(beta_hr.value):.2f}", label="your HR = e^β"),
            mo.stat(f"{_err:.3f}", label="mean squared log-error (group A)"),
            mo.stat(f"{ols_coef[2]:.2f}", label="OLS β (video's shortcut)", caption=f"HR {np.exp(ols_coef[2]):.2f}"),
            mo.stat(f"{cox_two['beta'][0]:.2f}", label="real Cox β", caption=f"HR {np.exp(cox_two['beta'][0]):.2f}"),
        ], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - Move β until the purple line runs through the orange points. You've done the video's "linear regression on log hazards" by eye. Compare with the OLS β.
    - Switch the log scale **off**. The two curves are no longer parallel; they **fan out**. On the raw scale "proportional" means *one is a constant multiple of the other*, not a constant gap.
    - Set β = 0 (HR = 1, no group effect), then β = −0.7 (HR ≈ 0.5): group A would then be the **safer** group.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · How Cox really fits β: the partial likelihood

    The video skips this ("it's quite complicated"), but the idea is a simple game. **At every death, ask: of everyone still at risk, why *this* person?**
    Under $h_i(t)=h_0(t)e^{\beta x_i}$, the probability that the death hit subject $i$ is

    $$P(i \mid \text{one death at } t) = \frac{h_0(t)\,e^{\beta x_i}}{\sum_{j\in R(t)} h_0(t)\,e^{\beta x_j}} = \frac{e^{\beta x_i}}{\sum_{j\in R(t)} e^{\beta x_j}}$$

    The **baseline hazard $h_0(t)$ cancels**. That's why Cox never needs to know its shape. Multiply these probabilities over all deaths to get the **partial likelihood** $L(\beta)$; the $\hat\beta$ that maximises it is the Cox estimate.

    **Step through the deaths.** Bars are subjects (still at risk = solid, gone = faded). The number at the right of each at-risk bar is that subject's chance of being "the one" under your β.
    """)
    return


@app.cell
def RiskSetStepper_cls(anywidget, traitlets):
    class RiskSetStepper(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 640, PADL = 60, PADR = 90, XMAX = 13, ROW = 20, TOP = 16;
      const NS = "http://www.w3.org/2000/svg";
      const ctl = document.createElement("div"), cap = document.createElement("div");
      cap.style.minHeight = "3em"; cap.style.fontFamily = "sans-serif"; cap.style.fontSize = "14px";
      const svg = document.createElementNS(NS, "svg");
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", fontFamily: "sans-serif" });
      el.append(ctl, svg, cap);
      const mk = (tag, a, p = svg) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
      const sx = v => PADL + v / XMAX * (W - PADL - PADR);
      const go = d => { const n = model.get("states").length; model.set("i", Math.max(0, Math.min(n - 1, model.get("i") + d))); model.save_changes(); };
      for (const [t, d] of [["⏮ first", -1e9], ["◀ prev death", -1], ["next death ▶", 1], ["last ⏭", 1e9]]) {
        const b = document.createElement("button"); b.textContent = t; b.style.marginRight = "6px"; b.onclick = () => go(d); ctl.appendChild(b); }
      function draw() {
        const subs = model.get("subjects"), st = model.get("states")[model.get("i")];
        const H = TOP + subs.length * ROW + 30;
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.innerHTML = "";
        const AX = TOP + subs.length * ROW + 6;
        for (let k = 0; k <= XMAX; k++) { if (k % 2 === 0) mk("text", { x: sx(k), y: AX + 16, "font-size": 10, "text-anchor": "middle", fill: "currentColor" }).textContent = k; }
        mk("line", { x1: sx(0), x2: sx(XMAX), y1: AX, y2: AX, stroke: "#bbb" });
        mk("line", { x1: sx(st.t), x2: sx(st.t), y1: TOP - 10, y2: AX, stroke: "#6c5b7b", "stroke-width": 2, "stroke-dasharray": "5 4" });
        mk("text", { x: sx(st.t) + 4, y: TOP - 2, "font-size": 11, fill: "#6c5b7b" }).textContent = "t = " + st.t;
        subs.forEach((s, j) => {
          const y = TOP + j * ROW + ROW / 2, risk = st.at_risk.includes(j), died = st.died.includes(j);
          const col = s.group === "A" ? "#e4572e" : "#2e86ab";
          mk("text", { x: 4, y: y + 4, "font-size": 11, fill: col }).textContent = s.label;
          mk("line", { x1: sx(0), x2: sx(s.time), y1: y, y2: y, stroke: col, "stroke-width": 8, "stroke-linecap": "round", opacity: risk ? 0.9 : 0.15 });
          if (died) mk("circle", { cx: sx(s.time), cy: y, r: 8, fill: "#6c5b7b", stroke: "white", "stroke-width": 2 });
          if (risk) mk("text", { x: W - PADR + 8, y: y + 4, "font-size": 12, fill: died ? "#6c5b7b" : "currentColor", "font-weight": died ? "bold" : "normal" }).textContent = "p = " + st.probs[j].toFixed(3);
        });
        cap.innerHTML = `<b>death ${model.get("i") + 1}/${model.get("states").length}</b>: ${st.caption}`;
      }
      model.on("change:i", draw); model.on("change:states", draw); draw();
    }
    export default { render };
    """
        subjects = traitlets.List(traitlets.Dict()).tag(sync=True)
        states = traitlets.List(traitlets.Dict()).tag(sync=True)
        i = traitlets.Int(0).tag(sync=True)

    return (RiskSetStepper,)


@app.cell
def beta_pl(mo):
    beta_pl = mo.ui.slider(-1.0, 3.0, step=0.05, value=0.0, label="β used in the partial likelihood", show_value=True)
    beta_pl
    return (beta_pl,)


@app.cell
def stepper(RiskSetStepper, beta_pl, mo, np, two_groups):
    _d = two_groups.sort_values(["time", "x"], ascending=[True, False]).reset_index(drop=True)
    _tm, _ev, _x = _d.time.values, _d.event.values, _d.x.values
    _w = np.exp(beta_pl.value * _x)
    _states, _cum = [], 0.0
    for _u in np.unique(_tm[_ev == 1]):
        _R = np.where(_tm >= _u)[0]; _D = np.where((_tm == _u) & (_ev == 1))[0]
        _p = {int(_j): float(_w[_j] / _w[_R].sum()) for _j in _R}
        _contrib = float(sum(np.log(_p[int(_j)]) for _j in _D))
        _cum += _contrib
        _who = ", ".join(f"{_d.group[_j]}{_j+1}" for _j in _D)
        _states.append(dict(
            t=float(_u), at_risk=[int(_j) for _j in _R], died=[int(_j) for _j in _D],
            probs={str(_k): _v for _k, _v in _p.items()},
            caption=(f"{len(_R)} at risk ({int(_x[_R].sum())} A, {int((1 - _x[_R]).sum())} B). Died: <b>{_who}</b>. "
                     f"log-contribution = {_contrib:.3f}; running log L(β) = <b style='color:#6c5b7b'>{_cum:.3f}</b>"),
        ))
    _states = [dict(_s, probs=[_s["probs"].get(str(_j), 0.0) for _j in range(len(_d))]) for _s in _states]
    _subs = [dict(label=f"{_g}{_j+1}", time=float(_t), group=_g) for _j, (_g, _t) in enumerate(zip(_d.group, _d.time))]
    stepper = mo.ui.anywidget(RiskSetStepper(subjects=_subs, states=_states, i=0))
    stepper
    return


@app.cell
def pl_readout(
    COL_A,
    COL_KEY,
    alt,
    beta_pl,
    cox_partial,
    cox_two,
    mo,
    np,
    pd,
    stats,
    two_groups,
):
    _X = two_groups[["x"]].values.astype(float); _tm = two_groups.time.values.astype(float); _ev = two_groups.event.values
    _grid = np.linspace(-1, 3, 161)
    _ll = pd.DataFrame(dict(beta=_grid, loglik=[cox_partial(_tm, _ev, _X, np.array([_b]))[0] for _b in _grid]))
    _cur = cox_partial(_tm, _ev, _X, np.array([beta_pl.value]))[0]
    _bh = cox_two["beta"][0]
    _curve = alt.Chart(_ll).mark_line(color=COL_KEY, strokeWidth=3).encode(x=alt.X("beta:Q", title="β"), y=alt.Y("loglik:Q", title="log partial likelihood", scale=alt.Scale(zero=False)))
    _mle = alt.Chart(pd.DataFrame(dict(beta=[_bh]))).mark_rule(color=COL_KEY, strokeDash=[4, 4]).encode(x="beta:Q")
    _me = alt.Chart(pd.DataFrame(dict(beta=[beta_pl.value], loglik=[_cur]))).mark_circle(size=200, color=COL_A).encode(x="beta:Q", y="loglik:Q")
    _tab = cox_two["table"]
    _score_g = cox_partial(_tm, _ev, _X, np.zeros(1))
    _score = float(_score_g[1][0] ** 2 / -_score_g[2][0, 0])
    _lr = 2 * (cox_two["loglik"] - cox_two["loglik0"])
    mo.vstack([
        (_curve + _mle + _me).properties(width="container", height=260, title="Your β (orange dot) on the log partial-likelihood curve; dashed = Cox estimate β̂"),
        mo.hstack([
            mo.stat(f"{_bh:.3f}", label="β̂ (coef)"),
            mo.stat(f"{np.exp(_bh):.2f}", label="HR = exp(coef)"),
            mo.stat(f"[{_tab['lower .95'].iloc[0]:.2f}, {_tab['upper .95'].iloc[0]:.2f}]", label="95% CI for HR"),
            mo.stat(f"{_tab['p'].iloc[0]:.3f}", label="Wald p-value"),
            mo.stat(f"{np.exp(-_bh):.3f}", label="exp(-coef) = 1/HR", caption=f"B has {100*(1-np.exp(-_bh)):.0f}% lower hazard"),
        ], justify="start"),
        mo.md(f"**R-style `coxph` output** (`Surv(time, event) ~ group`). Likelihood-ratio test χ² = {_lr:.2f} (p = {stats.chi2.sf(_lr, 1):.3f}); "
              f"score (= **log-rank**) test χ² = {_score:.2f} (p = {stats.chi2.sf(_score, 1):.3f})."),
        mo.ui.table(_tab.round(4), selection=None),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - At β = 0 everyone at risk is equally likely to die: every p = 1/(number at risk). Step through. Group A keeps "losing" early, which is evidence for β > 0.
    - Slide β to ≈ β̂. The orange A-bars get the higher probabilities, and the orange dot sits at the top of the curve.
    - Push β to 3: A deaths become very likely, but the **B deaths at years 1 and 6** become very *unlikely*, so log L falls again. The maximum balances both.
    - Read the output like the video: the CI for HR is wide and includes **1**, and p > 0.05. With 12 people, we can't reject "no difference". `exp(-coef)` is the HR with A as baseline.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Why not just a log-rank test? Adding age

    To compare two groups, the **log-rank test** is enough (above, it's literally Cox's *score test*).
    Cox's advantage is that it's a **regression**: add more variables and each gets its own hazard ratio, *controlling for the others*.

    $$h(t\mid x, \text{age}) = h_0(t)\,\exp(\beta_1 x + \beta_2\,\text{age})$$

    The video's example: $\hat\beta_2 = 0.095 \Rightarrow e^{0.095}\approx 1.10$. **Each extra year of age raises the hazard by ~10%**, holding group fixed.

    Below we simulate a bigger trial where group A also tends to be **older**. Watch what happens to the group HR when we forget to adjust for age.
    """)
    return


@app.cell
def sim_controls(mo):
    sim_n = mo.ui.slider(50, 600, step=50, value=300, label="subjects per group", show_value=True)
    sim_b1 = mo.ui.slider(-1.0, 2.0, step=0.05, value=0.5, label="true β₁ (group A)", show_value=True)
    sim_b2 = mo.ui.slider(0.0, 0.15, step=0.005, value=0.095, label="true β₂ (per year of age)", show_value=True)
    sim_gap = mo.ui.slider(0, 15, step=1, value=8, label="group A is older by (years)", show_value=True)
    sim_roll = mo.ui.run_button(label="🎲 new trial")
    mo.vstack([mo.hstack([sim_n, sim_b1], justify="start"), mo.hstack([sim_b2, sim_gap, sim_roll], justify="start")])
    return sim_b1, sim_b2, sim_gap, sim_n, sim_roll


@app.cell
def age_trial(cox_fit, np, pd, sim_b1, sim_b2, sim_gap, sim_n, sim_roll):
    sim_roll
    _rng = np.random.default_rng()
    _n = sim_n.value
    _x = np.r_[np.ones(_n), np.zeros(_n)]
    _age = np.r_[_rng.normal(60 + sim_gap.value, 8, _n), _rng.normal(60, 8, _n)]
    _rate = 0.05 * np.exp(sim_b1.value * _x + sim_b2.value * (_age - 60))
    _T = _rng.exponential(1 / _rate)
    _C = _rng.uniform(5, 20, 2 * _n)
    age_trial = pd.DataFrame(dict(time=np.minimum(_T, _C).round(3), event=(_T <= _C).astype(int), x=_x, age=_age.round(1)))
    fit_unadj = cox_fit(age_trial.time, age_trial.event, age_trial[["x"]].values, names=["groupA"])
    fit_adj = cox_fit(age_trial.time, age_trial.event, age_trial[["x", "age"]].values, names=["groupA", "age"])
    return age_trial, fit_adj, fit_unadj


@app.cell
def age_readout(
    COL_A,
    COL_GREY,
    COL_KEY,
    age_trial,
    alt,
    fit_adj,
    fit_unadj,
    mo,
    np,
    pd,
    sim_b1,
    sim_b2,
    stats,
):
    _rows = []
    for _model, _f in [("unadjusted: ~ group", fit_unadj), ("adjusted: ~ group + age", fit_adj)]:
        for _var, _r in _f["table"].iterrows():
            _rows.append(dict(model=_model, term=_var, HR=_r["exp(coef)"], lo=_r["lower .95"], hi=_r["upper .95"]))
    _forest = pd.DataFrame(_rows)
    _truth = pd.DataFrame(dict(term=["groupA", "age"], HR=[np.exp(sim_b1.value), np.exp(sim_b2.value)]))
    def _panel(_term):
        _f = _forest[_forest.term == _term]
        _base = alt.Chart(_f).encode(y=alt.Y("model:N", title=None), color=alt.Color("model:N", scale=alt.Scale(range=[COL_GREY, COL_KEY]), legend=None))
        return alt.layer(
            _base.mark_rule(strokeWidth=3).encode(x=alt.X("lo:Q", title=f"HR for {_term} (log scale)", scale=alt.Scale(type="log", zero=False)), x2="hi:Q"),
            _base.mark_circle(size=140).encode(x="HR:Q", tooltip=["model", "term", alt.Tooltip("HR:Q", format=".3f")]),
            alt.Chart(_truth[_truth.term == _term]).mark_rule(color=COL_A, strokeDash=[5, 3], strokeWidth=2).encode(x="HR:Q"),
        ).properties(width=520, height=70)
    _ch = alt.vconcat(_panel("groupA"), _panel("age"))
    _lr = 2 * (fit_adj["loglik"] - fit_unadj["loglik"])
    mo.vstack([
        mo.md(f"**{len(age_trial)} subjects, {age_trial.event.sum()} deaths.** Orange dashed line = true HR. Mean age: A {age_trial.age[age_trial.x==1].mean():.1f}, B {age_trial.age[age_trial.x==0].mean():.1f}."),
        _ch,
        mo.hstack([
            mo.stat(f"{fit_unadj['table']['exp(coef)'].iloc[0]:.2f}", label="group HR, unadjusted"),
            mo.stat(f"{fit_adj['table']['exp(coef)'].iloc[0]:.2f}", label="group HR, adjusted for age"),
            mo.stat(f"{np.exp(sim_b1.value):.2f}", label="true group HR"),
            mo.stat(f"{fit_adj['table']['exp(coef)'].iloc[1]:.3f}", label="HR per year of age", caption=f"true {np.exp(sim_b2.value):.3f}"),
            mo.stat(f"{_lr:.1f}", label="likelihood-ratio χ² (add age)", caption=f"p = {stats.chi2.sf(_lr, 1):.2g}"),
        ], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - Set **true β₁ = 0** (treatment does nothing) and age gap = 15. The unadjusted model "finds" a strong treatment effect that is pure **confounding by age**. The adjusted model recovers HR ≈ 1.
    - Set the age gap to 0. Both models now agree on the group HR, because adjusting doesn't matter when the groups are balanced.
    - The video's likelihood-ratio test compares the two nested models: a big χ² means adding age improves the fit.
    - An HR per year of 1.10 compounds: 10 years older ⇒ $1.1^{10}\approx e^{0.95}\approx 2.6\times$ the hazard.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Checking the proportional hazards assumption

    Cox assumes the HR is **constant over time**. If the Kaplan–Meier curves **cross**, that's hard to believe: A is riskier early, B is riskier later.

    The video's check is the **log-minus-log plot**. Since $S(t)=e^{-H(t)}$ and under PH $H_A(t)=e^\beta H_0(t)$:

    $$\log\!\big(-\log S_A(t)\big) = \beta + \log\!\big(-\log S_B(t)\big)$$

    so plotted against $\log t$, the two curves should be **parallel**, a constant vertical distance $\beta$ apart.
    """)
    return


@app.cell
def ph_scenario(mo):
    ph_scenario = mo.ui.dropdown(
        ["Proportional (HR = 2 always)", "Crossing hazards", "Effect wears off after year 5"],
        value="Proportional (HR = 2 always)", label="true hazards")
    ph_scenario
    return (ph_scenario,)


@app.cell
def ph_readout(
    COL_A,
    COL_B,
    alt,
    cox_fit,
    km_curve,
    km_table,
    mo,
    np,
    pd,
    ph_scenario,
):
    _rng = np.random.default_rng(7)
    _tg = np.linspace(0, 40, 4001)
    _HB = {"Proportional (HR = 2 always)": (_tg / 10) ** 1.5,
           "Crossing hazards": (_tg / 9) ** 2.5,
           "Effect wears off after year 5": 0.1 * _tg}[ph_scenario.value]
    _HA = {"Proportional (HR = 2 always)": 2 * (_tg / 10) ** 1.5,
           "Crossing hazards": (_tg / 6) ** 0.6,
           "Effect wears off after year 5": 0.3 * np.minimum(_tg, 5) + 0.1 * np.maximum(_tg - 5, 0)}[ph_scenario.value]
    _parts = []
    for _g, _H, _xv in [("A", _HA, 1), ("B", _HB, 0)]:
        _T = np.interp(_rng.exponential(size=400), _H, _tg)
        _parts.append(pd.DataFrame(dict(time=np.minimum(_T, 20).round(3), event=(_T < 20).astype(int), x=_xv, group=_g)))
    ph_data = pd.concat(_parts, ignore_index=True)
    ph_fit = cox_fit(ph_data.time, ph_data.event, ph_data[["x"]].values, names=["groupA"])

    _km, _lml = [], []
    for _g in ["A", "B"]:
        _d = ph_data[ph_data.group == _g]
        _km.append(km_curve(_d.time, _d.event).assign(group=_g))
        _tab = km_table(_d.time, _d.event)
        _tab = _tab[(_tab.S > 0) & (_tab.S < 1)]
        _lml.append(pd.DataFrame(dict(logt=np.log(_tab.end), lml=np.log(-np.log(_tab.S)), group=_g)))
    _km = pd.concat(_km); _lml = pd.concat(_lml)
    _cs = alt.Scale(domain=["A", "B"], range=[COL_A, COL_B])
    _c1 = alt.Chart(_km).mark_line(interpolate="linear").encode(x=alt.X("t:Q", title="years"), y=alt.Y("S:Q", title="S(t)"), color=alt.Color("group:N", scale=_cs)
            ).properties(width=300, height=240, title="Kaplan–Meier")
    _c2 = alt.Chart(_lml).mark_line().encode(x=alt.X("logt:Q", title="log t"), y=alt.Y("lml:Q", title="log(−log S(t))"), color=alt.Color("group:N", scale=_cs)
            ).properties(width=300, height=240, title="Log-minus-log: parallel ⇔ PH")
    _gap = (np.interp(np.log([2, 5, 10, 15]), _lml[_lml.group == "A"].logt, _lml[_lml.group == "A"].lml)
            - np.interp(np.log([2, 5, 10, 15]), _lml[_lml.group == "B"].logt, _lml[_lml.group == "B"].lml))
    mo.vstack([
        alt.hconcat(_c1, _c2),
        mo.hstack([
            mo.stat(f"{ph_fit['table']['exp(coef)'].iloc[0]:.2f}", label="Cox HR (one number for all t)"),
            *[mo.stat(f"{np.exp(_v):.2f}", label=f"implied HR of cum. hazards at t={_t}") for _t, _v in zip([2, 5, 10, 15], _gap)],
        ], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - **Proportional**: the log-minus-log curves are parallel, the gap is ≈ log 2, and every "implied HR" is ≈ 2, matching Cox.
    - **Crossing hazards**: the KM curves cross and the log-minus-log curves **cross** too. Cox still prints a single HR, but it's an average of "A worse early" and "B worse late", and describes neither period.
    - **Effect wears off**: the curves *converge*. The one-number HR understates the early benefit of B and overstates the late one. (Fixes: stratify, time-varying coefficients, or report HRs per period.)
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sandbox · The survival race

    Two groups of people live through time. At each tick, everyone still alive in group B dies with probability
    $h_0(t)\,\Delta t$; in group A with $h_0(t)\,e^{\beta}\,\Delta t$. The baseline $h_0(t)$ is a Weibull hazard whose **shape** you choose
    (shape < 1: risk falls over time; = 1: constant; > 1: risk rises).

    Press **▶ race**. When the race ends at $t=20$ (or when you pause), the results go to Python, which fits a Cox model.
    **Can Cox recover your β, whatever the baseline shape?**
    """)
    return


@app.cell
def SurvivalRace_cls(anywidget, traitlets):
    class SurvivalRace(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 640, H = 330, TMAX = 20, DT = 0.05;
      const ctl = document.createElement("div"); ctl.style.marginBottom = "4px"; el.appendChild(ctl);
      const canvas = document.createElement("canvas"); canvas.width = W; canvas.height = H;
      Object.assign(canvas.style, { width: "100%", maxWidth: W + "px" }); el.appendChild(canvas);
      const ctx = canvas.getContext("2d");
      const info = document.createElement("span"); info.style.marginLeft = "10px"; info.style.fontFamily = "sans-serif";
      let timer = null, st;
      const reset = () => {
        const n = model.get("n");
        st = { t: 0, people: [], kmA: [[0, 1]], kmB: [[0, 1]] };
        for (let g = 0; g < 2; g++) for (let k = 0; k < n; k++) st.people.push({ g, k, alive: true, time: TMAX });
        draw();
      };
      const H0 = t => Math.pow(t / 10, model.get("shape"));
      const push = () => {
        model.set("records", st.people.map(p => [p.time, p.alive ? 0 : 1, p.g === 0 ? 1 : 0]));
        model.save_changes();
      };
      function step() {
        if (st.t >= TMAX - 1e-9) { pause(); push(); return; }
        const dH = H0(st.t + DT) - H0(st.t), m = Math.exp(model.get("beta"));
        for (const p of st.people) if (p.alive) {
          const pr = 1 - Math.exp(-dH * (p.g === 0 ? m : 1));
          if (Math.random() < pr) { p.alive = false; p.time = +(st.t + DT).toFixed(2); }
        }
        st.t += DT;
        for (const g of [0, 1]) {
          const grp = st.people.filter(p => p.g === g), s = grp.filter(p => p.alive).length / grp.length;
          (g === 0 ? st.kmA : st.kmB).push([st.t, s]);
        }
        draw();
      }
      function draw() {
        ctx.clearRect(0, 0, W, H);
        const n = model.get("n"), cols = Math.ceil(n / 4);
        for (const p of st.people) {
          const x = 10 + (p.k % cols) * (300 / cols) + (p.g === 1 ? 320 : 0), y = 20 + Math.floor(p.k / cols) * 14;
          ctx.beginPath(); ctx.arc(x + 4, y, 4.5, 0, 2 * Math.PI);
          ctx.fillStyle = p.alive ? (p.g === 0 ? "#e4572e" : "#2e86ab") : "rgba(150,150,150,0.3)"; ctx.fill();
        }
        ctx.fillStyle = "#e4572e"; ctx.font = "12px sans-serif"; ctx.fillText("group A (hazard × e^β)", 10, 10);
        ctx.fillStyle = "#2e86ab"; ctx.fillText("group B (baseline)", 330, 10);
        const top = 90, bot = H - 20, sx = t => 40 + t / TMAX * (W - 60), sy = s => bot - s * (bot - top);
        ctx.strokeStyle = "#bbb"; ctx.beginPath(); ctx.moveTo(40, top); ctx.lineTo(40, bot); ctx.lineTo(W - 20, bot); ctx.stroke();
        ctx.fillStyle = "#888"; ctx.fillText("S(t)", 5, top + 10); ctx.fillText("t = " + st.t.toFixed(1) + " / 20", W - 110, top + 10);
        for (const [km, c] of [[st.kmA, "#e4572e"], [st.kmB, "#2e86ab"]]) {
          ctx.strokeStyle = c; ctx.lineWidth = 2.5; ctx.beginPath();
          km.forEach(([t, s], i) => i ? ctx.lineTo(sx(t), sy(s)) : ctx.moveTo(sx(t), sy(s))); ctx.stroke();
        }
        ctx.lineWidth = 1;
        const dA = st.people.filter(p => p.g === 0 && !p.alive).length, dB = st.people.filter(p => p.g === 1 && !p.alive).length;
        info.textContent = `deaths: A ${dA}, B ${dB}`;
      }
      const pause = () => { if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ race"; } };
      const play = document.createElement("button"); play.textContent = "▶ race";
      play.onclick = () => { if (timer) { pause(); push(); } else { timer = setInterval(step, 30); play.textContent = "⏸ pause"; } };
      const rs = document.createElement("button"); rs.textContent = "↺ reset"; rs.style.marginLeft = "6px";
      rs.onclick = () => { pause(); reset(); model.set("records", []); model.save_changes(); };
      const fast = document.createElement("button"); fast.textContent = "⏩ finish instantly"; fast.style.marginLeft = "6px";
      fast.onclick = () => { pause(); while (st.t < TMAX - 1e-9) step(); push(); };
      ctl.append(play, rs, fast, info);
      model.on("change:n", reset); reset();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        beta = traitlets.Float(1.0).tag(sync=True)
        shape = traitlets.Float(1.0).tag(sync=True)
        n = traitlets.Int(60).tag(sync=True)
        records = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)

    return (SurvivalRace,)


@app.cell
def race_controls(mo):
    race_beta = mo.ui.slider(-1.5, 2.0, step=0.1, value=1.0, label="true β (group A)", show_value=True)
    race_shape = mo.ui.slider(0.5, 3.0, step=0.1, value=1.0, label="baseline Weibull shape", show_value=True)
    race_n = mo.ui.slider(20, 120, step=20, value=60, label="people per group", show_value=True)
    mo.hstack([race_beta, race_shape, race_n], justify="start")
    return race_beta, race_n, race_shape


@app.cell
def race(SurvivalRace, mo, race_beta, race_n, race_shape):
    race = mo.ui.anywidget(SurvivalRace(beta=race_beta.value, shape=race_shape.value, n=race_n.value))
    race
    return (race,)


@app.cell
def race_readout(cox_fit, mo, np, race, race_beta):
    _rec = np.array(race.value["records"], dtype=float).reshape(-1, 3)
    mo.stop(len(_rec) == 0 or _rec[:, 1].sum() == 0,
            mo.callout(mo.md("Press **▶ race** (or ⏩ finish instantly) above. Results appear here when the race ends or pauses."), kind="neutral"))
    _mask = _rec[:, 0] > 0
    race_fit = cox_fit(_rec[_mask, 0], _rec[_mask, 1].astype(int), _rec[_mask, 2:3], names=["groupA"])
    _r = race_fit["table"].iloc[0]
    _hit = _r["lower .95"] <= np.exp(race_beta.value) <= _r["upper .95"]
    mo.vstack([
        mo.hstack([
            mo.stat(f"{race_beta.value:.2f}", label="true β", caption=f"HR {np.exp(race_beta.value):.2f}"),
            mo.stat(f"{_r['coef']:.2f}", label="Cox β̂", caption=f"HR {_r['exp(coef)']:.2f}"),
            mo.stat(f"[{_r['lower .95']:.2f}, {_r['upper .95']:.2f}]", label="95% CI for HR"),
            mo.stat(f"{_r['p']:.3g}", label="p-value"),
            mo.stat(f"{int(_rec[:, 1].sum())}", label="deaths observed", caption=f"of {len(_rec)}"),
        ], justify="start"),
        mo.callout(mo.md("✅ The CI covers the true HR." if _hit else "❌ The CI missed the true HR. That should happen about 5% of the time; re-run and see."),
                   kind="success" if _hit else "danger"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Challenges**
    1. Set β = 1 and try shapes 0.5, 1 and 3. Cox's estimate stays near 1 every time, because $h_0(t)$ cancels in the partial likelihood.
    2. With 20 per group and β = 0.5, how often is p < 0.05? Now try 120 per group. That's the "wide CI from a small sample" point from the video.
    3. Set β = 0 and race 20 times. Roughly one race in 20 should give a "significant" p-value by chance.
    4. Pause halfway (around t = 8). Survivors are **censored** at that time. Does β̂ change much?
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Quiz · Check yourself
    """)
    return


@app.cell
def quiz(mo):
    quiz = mo.ui.dictionary({
        "q1": mo.ui.radio(["A's survival time is 3.5× shorter", "At any time, A's hazard is 3.5× B's", "3.5× more people in A die in total"],
                          label="**1.** Cox gives HR = 3.5 for group A vs baseline B. This means…"),
        "q2": mo.ui.radio(["≈ 1.1×", "≈ 2.0×", "≈ 2.6×", "≈ 9.5×"],
                          label="**2.** Age has β = 0.095 per year. Compare someone 10 years older (same group): their hazard is…"),
        "q3": mo.ui.radio(["71% lower hazard", "29% lower hazard", "3.5× lower survival time"],
                          label="**3.** With A as the baseline, HR(B) = 0.2861. B has…"),
        "q4": mo.ui.radio(["It's assumed to be exponential", "It cancels out in the partial likelihood", "It is estimated first by Kaplan–Meier"],
                          label="**4.** Why does Cox not need a formula for the baseline hazard h₀(t)?"),
        "q5": mo.ui.radio(["KM curves that never cross", "Parallel curves in log(−log S) vs log t", "A p-value < 0.05"],
                          label="**5.** What supports the proportional hazards assumption?"),
        "q6": mo.ui.radio(["0.6667", "1.0", "2.2833", "3.0"],
                          label="**6.** Six subjects die at 1, 6, 10, 11, 11, 12 years. What is the cumulative hazard H at year 12?"),
    })
    quiz
    return (quiz,)


@app.cell
def reveal(mo):
    reveal = mo.ui.switch(label="reveal answers")
    reveal
    return (reveal,)


@app.cell
def quiz_score(km_table, mo, np, quiz, reveal):
    _ans = {
        "q1": ("At any time, A's hazard is 3.5× B's", "HR compares instantaneous risk, not survival times or totals."),
        "q2": ("≈ 2.6×", f"e^(10·0.095) = {np.exp(0.95):.2f}; effects multiply."),
        "q3": ("71% lower hazard", "1 − 0.2861 = 0.7139, as in the video."),
        "q4": ("It cancels out in the partial likelihood", "Every term is e^{βxᵢ}/Σ e^{βxⱼ}; h₀(t) is in both numerator and denominator."),
        "q5": ("Parallel curves in log(−log S) vs log t", "Parallel ⇔ constant HR. (Non-crossing KM is necessary but not sufficient.)"),
        "q6": ("2.2833", f"Σ h·Δt = {km_table([1,6,10,11,11,12],[1]*6).H.iloc[-1]:.4f} (the video's description correction)."),
    }
    _score = sum(quiz.value[_k] == _v[0] for _k, _v in _ans.items())
    _answered = sum(quiz.value[_k] is not None for _k in _ans)
    _lines = [f"**Score: {_score} / {len(_ans)}** ({_answered} answered)"]
    if reveal.value:
        for _k, (_a, _why) in _ans.items():
            _ok = quiz.value[_k] == _a
            _lines.append(f"- {'✅' if _ok else '❌'} **{_k[1:]}.** {_a}: {_why}")
    mo.callout(mo.md("\n".join(_lines)), kind="success" if _score == len(_ans) else "neutral")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap

    | Idea | Formula | What you saw |
    |---|---|---|
    | Hazard | $h(t) = -S'(t)/S(t)$, estimated as $\dfrac{d}{n\,\Delta t}$ | the same death spread over a longer interval means a lower hazard |
    | Cumulative hazard | $H(t) = \int_0^t h = \sum h\,\Delta t$, $\;S = e^{-H}$ | the video's corrected 0.1667 … 2.2833 |
    | Cox model | $h(t\mid x) = h_0(t)\,e^{\beta x}$ | parallel lines on the log-hazard scale |
    | Hazard ratio | $\text{HR} = e^{\beta}$; reversing the baseline gives $1/\text{HR}$ | $1-0.2861 = 71\%$ lower risk |
    | Partial likelihood | $L(\beta) = \prod_{\text{deaths}} \dfrac{e^{\beta x_i}}{\sum_{R(t_i)} e^{\beta x_j}}$ | $h_0$ cancels; β̂ maximises the curve |
    | Inference | CI for HR includes 1 ⇔ p > 0.05; score test = log-rank | a wide CI with 12 subjects |
    | Covariates | $e^{\beta_2}$ per unit, holding others fixed; LR test for nested models | adjusting for age removes confounding |
    | PH assumption | $\log(-\log S)$ vs $\log t$ should be parallel | crossing curves make one HR misleading |
    """)
    return


if __name__ == "__main__":
    app.run()

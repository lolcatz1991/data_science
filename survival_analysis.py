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
def _(mo):
    mo.vstack([
        mo.md(r"""
    # Survival analysis: how long until it happens?
    *An interactive companion to two videos: **DataMListic, "Survival Analysis - Explained"** (a subscription app
    asking "how long does a user stay?") and **Biostatistics, "Mini Lecture: Survival Analysis"** (a cancer trial
    asking "does the new treatment help people survive longer?").*

    Both videos tell the same story with different examples. **Some of your stories haven't ended yet**, and the
    whole field exists to use those unfinished stories without throwing them away or making up their endings.
    """),
        mo.hstack([
            mo.Html('<iframe width="100%" height="260" src="https://www.youtube.com/embed/Aq5CIwBcxnE" title="Survival Analysis - Explained" frameborder="0" allowfullscreen></iframe>'),
            mo.Html('<iframe width="100%" height="260" src="https://www.youtube.com/embed/t6vdjwhauF8" title="Mini Lecture: Survival Analysis" frameborder="0" allowfullscreen></iframe>'),
        ], widths="equal", gap=1),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Color language used throughout**

    | color | meaning |
    |---|---|
    | <span style="color:#e4572e">■ orange-red</span> | an **event**: a cancellation or a death (we saw it happen) |
    | <span style="color:#2e86ab">■ blue</span> | **censored**: still going when we stopped watching (we only know "at least this long") |
    | <span style="color:#6c5b7b">■ purple</span> | the **Kaplan-Meier estimate** $\hat S(t)$ |
    | <span style="color:#555">■ dashed grey</span> | the **true** curve (only available when we simulate) |
    | <span style="color:#f18f01">■ amber</span> | **naive** answers that ignore censoring |

    **Roadmap:** 1. Censoring and the two wrong answers → 2. Survival, density and hazard → 3. Kaplan-Meier step by step →
    4. Checking against the truth (and when it breaks) → 5. What the hazard's shape tells you → 6. Comparing groups with the log-rank test →
    sandbox, quiz, recap. The **Cox proportional hazards model** gets only a teaser here; it has its own notebook, `cox_proportional_hazards.py`.
    """)
    return


@app.cell
def helpers(np, pd, stats):
    def km_estimate(durations, events):
        """Kaplan-Meier table at every distinct observed time, with Greenwood standard errors."""
        _t = np.asarray(durations, dtype=float)
        _e = np.asarray(events, dtype=int)
        _rows = []
        _S, _gw = 1.0, 0.0
        for _tj in np.unique(_t):
            _n = int((_t >= _tj).sum())
            _d = int(((_t == _tj) & (_e == 1)).sum())
            _c = int(((_t == _tj) & (_e == 0)).sum())
            if _d > 0:
                _S *= 1 - _d / _n
                if _n > _d:
                    _gw += _d / (_n * (_n - _d))
            _se = _S * np.sqrt(_gw)
            _rows.append(dict(t=_tj, n=_n, d=_d, c=_c, S=_S, se=_se,
                              lo=max(0.0, _S - 1.96 * _se), hi=min(1.0, _S + 1.96 * _se)))
        _df = pd.DataFrame(_rows)
        _start = pd.DataFrame([dict(t=0.0, n=len(_t), d=0, c=0, S=1.0, se=0.0, lo=1.0, hi=1.0)])
        return pd.concat([_start, _df], ignore_index=True)


    def km_median(table):
        """First time the KM curve reaches 0.5 or below (NaN if it never does)."""
        _hit = table[table["S"] <= 0.5]
        return float(_hit["t"].iloc[0]) if len(_hit) else float("nan")


    def km_at(table, t):
        """Value of the step function at time t."""
        return float(table[table["t"] <= t]["S"].iloc[-1])


    def logrank(durations, events, group):
        """Two-group log-rank test. group is boolean (True = group 1). Returns stats and per-time table."""
        _t = np.asarray(durations, float); _e = np.asarray(events, int); _g = np.asarray(group, bool)
        _rows = []
        for _tj in np.unique(_t[_e == 1]):
            _at = _t >= _tj
            _n, _n1 = _at.sum(), (_at & _g).sum()
            _dmask = (_t == _tj) & (_e == 1)
            _d, _d1 = _dmask.sum(), (_dmask & _g).sum()
            _E1 = _d * _n1 / _n
            _V = _d * (_n1 / _n) * (1 - _n1 / _n) * (_n - _d) / (_n - 1) if _n > 1 else 0.0
            _rows.append(dict(t=_tj, n=_n, n1=_n1, d=_d, O1=_d1, E1=_E1, V=_V))
        _tab = pd.DataFrame(_rows)
        _O, _E, _Vs = _tab["O1"].sum(), _tab["E1"].sum(), _tab["V"].sum()
        _chi = (_O - _E) ** 2 / _Vs
        return dict(O1=_O, E1=_E, O2=_tab["d"].sum() - _O, E2=_tab["d"].sum() - _E,
                    chi2=_chi, p=float(stats.chi2.sf(_chi, 1)), table=_tab)


    def step_df(table, label):
        """KM table -> rows for an altair step line, extended to the last observed time."""
        _df = table[["t", "S", "lo", "hi"]].copy()
        _df["curve"] = label
        return _df

    return km_estimate, km_median, logrank, step_df


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Censoring: the stories that haven't ended

    **The app.** Your manager asks *"how long does a user stay before they cancel?"*. You pull up eight users.
    Four of them canceled (●, orange-red). Four are **still subscribed** (○ with a dashed arrow, blue). One of them
    only joined 3 months ago. Their missing endings are what statisticians call **censoring**.

    **The trial.** It's the same problem in the cancer study. You can't run a study until everyone has died, so it stops at a fixed date.
    Patients still alive at the end, plus the [often 30%+] who drop out for reasons like travel, work or moving house, are
    **right-censored**. We know their event lies *somewhere to the right* of the last time we saw them.

    Two tempting fixes, both **too low**:

    1. **Drop the active users** and take the median of the people who left. But the loyal 9-month user is exactly who you threw out.
    2. **Pretend the active users cancel today.** This invents a cancellation that never happened.

    Instead, use what a censored user really tells us: **"my stay is *at least* this long."**

    **Drag** a bar's end to change its length. **Click** an end marker to switch between canceled and still active.
    """)
    return


@app.cell
def CohortTimeline_cls(anywidget, traitlets):
    class CohortTimeline(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 660, H = 320, PADL = 70, PADR = 40, TOP = 14, BOT = 40, XMAX = 13;
      const NS = "http://www.w3.org/2000/svg";
      const bar = document.createElement("div");
      bar.style.marginBottom = "6px";
      el.appendChild(bar);
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none" });
      el.appendChild(svg);
      const mk = (tag, attrs, parent = svg) => { const e = document.createElementNS(NS, tag);
        for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
      const toSvg = (evt) => { const p = svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY;
        return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const sx = v => PADL + v / XMAX * (W - PADL - PADR);
      const inv = px => (px - PADL) / (W - PADL - PADR) * XMAX;
      const presets = {
        "Video example": [[2,3,4,5,6,8,9,12],[1,0,1,1,0,1,0,0]],
        "Nobody censored": [[2,3,4,5,6,8,9,12],[1,1,1,1,1,1,1,1]],
        "Loyal actives": [[2,4,5,8,11,12,12,12],[1,1,1,1,0,0,0,0]],
      };
      for (const name in presets) {
        const b = document.createElement("button"); b.textContent = name; b.style.marginRight = "6px";
        b.onclick = () => { model.set("durations", [...presets[name][0]]); model.set("events", [...presets[name][1]]); model.save_changes(); };
        bar.appendChild(b);
      }
      let drag = -1, startX = 0, moved = false;
      function draw() {
        const d = model.get("durations"), e = model.get("events"), n = d.length;
        const rowH = (H - TOP - BOT) / n;
        svg.innerHTML = "";
        for (let t = 0; t <= 12; t++) {
          mk("line", { x1: sx(t), x2: sx(t), y1: TOP, y2: H - BOT, stroke: "currentColor", "stroke-opacity": 0.12 });
          const tx = mk("text", { x: sx(t), y: H - BOT + 16, "text-anchor": "middle", "font-size": 11, fill: "currentColor" }); tx.textContent = t;
        }
        const lab = mk("text", { x: (sx(0) + sx(12)) / 2, y: H - 6, "text-anchor": "middle", "font-size": 12, fill: "currentColor" });
        lab.textContent = "months since sign-up (users lined up by tenure)";
        d.forEach((v, i) => {
          const y = TOP + i * rowH + rowH / 2, col = e[i] ? "#e4572e" : "#2e86ab";
          const t = mk("text", { x: 8, y: y + 4, "font-size": 12, fill: "currentColor" }); t.textContent = "user " + (i + 1);
          mk("line", { x1: sx(0), x2: sx(v), y1: y, y2: y, stroke: col, "stroke-width": 9, "stroke-linecap": "round", "stroke-opacity": 0.45 });
          if (!e[i]) {
            mk("line", { x1: sx(v) + 10, x2: sx(v) + 34, y1: y, y2: y, stroke: col, "stroke-width": 2, "stroke-dasharray": "4 3" });
            mk("path", { d: `M${sx(v)+34},${y-5} L${sx(v)+40},${y} L${sx(v)+34},${y+5}`, fill: "none", stroke: col, "stroke-width": 2 });
          }
          const c = mk("circle", { cx: sx(v), cy: y, r: 9, fill: e[i] ? col : "white", stroke: col, "stroke-width": 3 });
          c.style.cursor = "ew-resize";
          c.addEventListener("pointerdown", ev => { drag = i; startX = ev.clientX; moved = false; svg.setPointerCapture(ev.pointerId); });
        });
      }
      svg.addEventListener("pointermove", ev => {
        if (drag < 0) return;
        if (Math.abs(ev.clientX - startX) > 4) moved = true;
        if (!moved) return;
        const a = [...model.get("durations")];
        a[drag] = Math.max(1, Math.min(12, Math.round(inv(toSvg(ev).x))));
        model.set("durations", a);
      });
      const stop = () => {
        if (drag < 0) return;
        if (!moved) { const e = [...model.get("events")]; e[drag] = e[drag] ? 0 : 1; model.set("events", e); }
        drag = -1; model.save_changes();
      };
      svg.addEventListener("pointerup", stop);
      svg.addEventListener("pointercancel", stop);
      model.on("change:durations", draw); model.on("change:events", draw); draw();
    }
    export default { render };
    """
        durations = traitlets.List(traitlets.Float()).tag(sync=True)
        events = traitlets.List(traitlets.Int()).tag(sync=True)

    return (CohortTimeline,)


@app.cell
def cohort(CohortTimeline, mo):
    cohort = mo.ui.anywidget(CohortTimeline(durations=[2, 3, 4, 5, 6, 8, 9, 12], events=[1, 0, 1, 1, 0, 1, 0, 0]))
    cohort
    return (cohort,)


@app.cell
def cohort_readout(alt, cohort, km_estimate, km_median, mo, np, pd):
    cohort_t = np.array(cohort.value["durations"], dtype=float)
    cohort_e = np.array(cohort.value["events"], dtype=int)
    cohort_km = km_estimate(cohort_t, cohort_e)

    _drop = np.median(cohort_t[cohort_e == 1]) if cohort_e.sum() else float("nan")
    _today = np.median(cohort_t)
    _km_med = km_median(cohort_km)

    _naive = pd.DataFrame({"estimate": ["① drop active users", "② pretend they cancel today", "Kaplan-Meier"],
                           "median": [_drop, _today, _km_med]})
    _line = (alt.Chart(cohort_km).mark_line(interpolate="step-after", color="#6c5b7b", strokeWidth=3)
             .encode(x=alt.X("t:Q", title="months", scale=alt.Scale(domain=[0, 13])),
                     y=alt.Y("S:Q", title="Ŝ(t) = share still subscribed", scale=alt.Scale(domain=[0, 1.02]))))
    _ticks = (alt.Chart(cohort_km[cohort_km["c"] > 0]).mark_tick(color="#2e86ab", thickness=2, size=14)
              .encode(x="t:Q", y="S:Q", tooltip=["t", "n", "c"]))
    _half = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
    _rules = (alt.Chart(_naive.dropna()).mark_rule(strokeWidth=2)
              .encode(x="median:Q",
                      color=alt.Color("estimate:N", scale=alt.Scale(range=["#f18f01", "#f6c26b", "#6c5b7b"]),
                                      legend=alt.Legend(orient="bottom", title=None)),
                      tooltip=["estimate", "median"]))
    mo.vstack([
        mo.hstack([
            mo.stat(f"{_drop:.1f} mo", label="① drop active users", caption="biased toward early leavers"),
            mo.stat(f"{_today:.1f} mo", label="② pretend they cancel today", caption="invents cancellations"),
            mo.stat("never reaches ½" if np.isnan(_km_med) else f"{_km_med:.0f} mo", label="Kaplan-Meier median",
                    caption="uses 'at least' facts", bordered=True),
        ], justify="space-around"),
        (_line + _ticks + _half + _rules).properties(width="container", height=240,
            title="The survival curve built from your eight users (blue ticks = censored)"),
    ])
    return cohort_e, cohort_km, cohort_t


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    - Start from **Video example**: the two naive medians are 4.5 and 5.5 months, but Kaplan-Meier says **8**, almost double.
    - Click **Nobody censored**. All three answers agree, because censoring is the *only* reason they differed.
    - Click **Loyal actives**, then drag the blue users further right. The naive ① answer **doesn't move at all**, since it ignores them. KM can stop above ½ and refuses to make up a median.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Not just right-censoring: the Mini Lecture's other ways to have incomplete stories

    | phenomenon | what we know | example from the lecture |
    |---|---|---|
    | **Right-censored** | the event is *after* the last time we looked | still alive at study end; dropped out; GFP signal above the detection limit; too many bacteria to count |
    | **Left-censored** | the event happened *before* some time, but we don't know how long before | infection first seen at a positive test; the egg is already there in the morning |
    | **Interval-censored** | only that the event fell *between* two times | infected during a holiday, tested afterwards; COVID infection between exposure and positive test |
    | **Left-truncated** | people who had the event *before entry* never appear | delayed entry into the trial: you had to survive long enough to be diagnosed |
    | **Right-truncated** | only people who *already* had the event appear | historical death records: you must have died to be in the data |
    | **Competing risk** | a *different* event happened at a known time and prevents ours | a cancer patient dies in a traffic accident (handled with multi-state models) |

    Censored values come with a **bound**. Truncated units are **not in the data at all**. Neither is the same as ordinary
    **missing data**, where the row exists but a value is blank. Right-censoring and left-truncation are the most common.
    The rest of this notebook deals with right-censoring.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · The survival curve, the density and the hazard

    DataMListic's first move is to **stop chasing a single number**. Let $T$ be the length of a stay. Then

    $$S(t) = P(T > t)$$

    is the share still subscribed (or alive) at time $t$. It starts at 1 and can only go down. The **median** is where it crosses ½.
    People quote the median rather than the mean because the curve often stops before reaching 0, and the mean would need the part we never saw.

    *Why a curve and not a histogram?* A histogram needs a stay to be **finished** before it can go in a bin. $S(t)$ only asks
    *"did you make it past $t$?"*, and an active user can answer that for every $t$ up to today.

    Three views of one distribution:

    | | formula | reads as |
    |---|---|---|
    | density | $f(t)$ | when do events happen, overall? |
    | survival | $S(t) = 1-F(t)=\int_t^\infty f$ | who is still here? |
    | **hazard** | $h(t) = \dfrac{f(t)}{S(t)}$ | the risk of leaving *right now*, **given you've lasted this long** |

    The hazard is the continuous-time version of the $d/n$ that Kaplan-Meier multiplies (section 3). For a tiny step, $1-x \approx e^{-x}$, so
    a long product of one-step survivals becomes a sum of hazards in an exponent:

    $$S(t) = \prod (1 - h\,\Delta t) \;\longrightarrow\; S(t) = \exp\!\Big(-\int_0^t h(u)\,du\Big).$$

    With a constant hazard $\lambda$ you get $S(t)=e^{-\lambda t}$, the exponential distribution. The **Weibull** with shape $k$ has
    $h(t) = \frac{k}{\lambda}\big(\frac{t}{\lambda}\big)^{k-1}$: falling if $k<1$, constant at $k=1$, rising (wear-out) if $k>1$.
    """)
    return


@app.cell
def shape_controls(mo):
    shape_family = mo.ui.radio(["Weibull", "Bathtub"], value="Weibull", label="hazard family", inline=True)
    weib_k = mo.ui.slider(0.3, 4.0, step=0.05, value=0.8, label="Weibull shape k", show_value=True)
    weib_scale = mo.ui.slider(2, 30, step=0.5, value=13.0, label="scale λ (months)", show_value=True)
    mo.hstack([shape_family, weib_k, weib_scale], justify="start", gap=2)
    return shape_family, weib_k, weib_scale


@app.cell
def shape_chart(alt, mo, np, pd, shape_family, weib_k, weib_scale):
    _t = np.linspace(0.01, 36, 600)
    if shape_family.value == "Weibull":
        _k, _lam = weib_k.value, weib_scale.value
        _h = (_k / _lam) * (_t / _lam) ** (_k - 1)
        _H = (_t / _lam) ** _k
    else:
        _u = _t / (weib_scale.value / 13.0)
        _h = (0.35 * np.exp(-_u / 1.5) + 0.03 + 0.25 * (_u / 30) ** 6) / (weib_scale.value / 13.0)
        _H = np.concatenate([[0], np.cumsum(0.5 * (_h[1:] + _h[:-1]) * np.diff(_t))]) + _h[0] * _t[0]
    _S = np.exp(-_H)
    _f = _h * _S
    shape_df = pd.DataFrame({"t": _t, "f(t) density": _f, "S(t) survival": _S, "h(t) hazard": _h})
    shape_median = float(_t[np.argmax(_S <= 0.5)]) if (_S <= 0.5).any() else float("nan")

    def _panel(col, color, title):
        _base = alt.Chart(shape_df).mark_line(color=color, strokeWidth=3).encode(
            x=alt.X("t:Q", title="months"), y=alt.Y(f"{col}:Q", title=None))
        return _base.properties(width=210, height=190, title=title)

    _Sp = _panel("S(t) survival", "#6c5b7b", "S(t) = P(T > t)")
    _Sp = _Sp + alt.Chart(pd.DataFrame({"m": [shape_median]})).mark_rule(color="#e4572e", strokeDash=[4, 3]).encode(x="m:Q")
    _hmax = float(np.quantile(_h, 0.98)) * 1.1
    _hp = alt.Chart(shape_df).mark_line(color="#e4572e", strokeWidth=3).encode(
        x=alt.X("t:Q", title="months"),
        y=alt.Y("h(t) hazard:Q", title=None, scale=alt.Scale(domain=[0, _hmax], clamp=True))).properties(
        width=210, height=190, title="h(t) = f / S")
    _trend = "falling (early leavers / infant mortality)" if _h[-1] < _h[50] * 0.9 else (
        "rising (wear-out)" if _h[-1] > _h[50] * 1.1 else "roughly constant (memoryless)")
    mo.vstack([
        alt.hconcat(_panel("f(t) density", "#888", "f(t)"), _Sp, _hp),
        mo.md(f"Median stay: **{shape_median:.1f} months** (red dashed line). Overall the hazard is **{_trend}**. "
              f"Months 0 to 36: S(36) = {_S[-1]:.2f}."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Where does the *e* come from?** Take a constant hazard of 0.3 per month. Chop each month into $m$ slices and survive each
    slice with probability $1-0.3/m$. As $m$ grows, the product of many factors close to 1 turns into $e^{-0.3t}$.
    """)
    return


@app.cell
def slices(mo):
    slices = mo.ui.slider(1, 50, step=1, value=1, label="slices per month m", show_value=True)
    slices
    return (slices,)


@app.cell
def e_demo(alt, mo, np, pd, slices):
    _lam = 0.3
    _tt = np.arange(0, 12 * slices.value + 1) / slices.value
    _prod = (1 - _lam / slices.value) ** (_tt * slices.value)
    _df = pd.concat([
        pd.DataFrame({"t": _tt, "S": _prod, "curve": f"product of (1 − 0.3/{slices.value})"}),
        pd.DataFrame({"t": np.linspace(0, 12, 200), "S": np.exp(-_lam * np.linspace(0, 12, 200)), "curve": "exp(−0.3 t)"}),
    ])
    _c = alt.Chart(_df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("t:Q", title="months"), y=alt.Y("S:Q", title="S(t)"),
        color=alt.Color("curve:N", scale=alt.Scale(range=["#555", "#6c5b7b"]), legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("curve:N", legend=None))
    mo.vstack([_c.properties(width="container", height=200),
               mo.md(f"At t = 12 the product gives **{_prod[-1]:.4f}**, and e^(−3.6) = **{np.exp(-3.6):.4f}**.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    - Set **k = 1**. The hazard becomes a flat line, which is the exponential, "memoryless" case.
    - Move **k below 1**, like the video's simulated users. The hazard falls: the longer you've stayed, the less likely you are to leave next month.
    - Switch to **Bathtub**. The hazard is high at the start, flat for a long time, and rises at the end, which is common for real products and machines. Notice how much harder it is to read that story from $S(t)$ alone.
    - Slide **m** from 1 to 50 and watch the product converge to the exponential.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Kaplan-Meier, one month at a time

    Surviving past month 8 means surviving month 1, *then* month 2 given you made it through month 1, and so on. That is just the
    chain rule of conditional probability. At every time $t_i$ with at least one event:

    - $n_i$ = the **risk set**: everyone **still here *and* still observed**,
    - $d_i$ = how many have the event at $t_i$,

    $$\hat S(t) = \prod_{t_i \le t}\left(1 - \frac{d_i}{n_i}\right).$$

    A **censored** person counts as a survivor in every step up to when we stopped seeing them, then leaves the denominators.
    As the Mini Lecture puts it: *"the only thing that changes with censored observations is the number at risk."*
    The censoring is drawn as a small tick, not a drop.

    The stepper below replays the eight users **from section 1**, so edit them up there and the replay follows.
    """)
    return


@app.cell
def KMStepper_cls(anywidget, traitlets):
    class KMStepper(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 660, H = 430, PADL = 70, PADR = 30, TOP = 10, ROWS_H = 200, CTOP = 245, CH = 150;
      const NS = "http://www.w3.org/2000/svg";
      const ctl = document.createElement("div"); el.appendChild(ctl);
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none" });
      el.appendChild(svg);
      const cap = document.createElement("div"); cap.style.marginTop = "4px"; cap.style.minHeight = "2.6em"; el.appendChild(cap);
      const mk = (tag, attrs) => { const e = document.createElementNS(NS, tag);
        for (const k in attrs) e.setAttribute(k, attrs[k]); svg.appendChild(e); return e; };
      const go = d => { const n = model.get("states").length;
        model.set("i", Math.max(0, Math.min(n - 1, model.get("i") + d))); model.save_changes(); };
      for (const [t, d] of [["⏮ start", -1e9], ["◀ back", -1], ["next ▶", 1], ["end ⏭", 1e9]]) {
        const b = document.createElement("button"); b.textContent = t; b.style.marginRight = "6px"; b.onclick = () => go(d); ctl.appendChild(b); }
      const XMAX = 13;
      const sx = v => PADL + v / XMAX * (W - PADL - PADR);
      const sy = s => CTOP + (1 - s) * CH;
      const COL = { risk: "#6c5b7b", event: "#e4572e", cens: "#2e86ab", out_event: "#e4572e", out_cens: "#2e86ab" };
      function draw() {
        const states = model.get("states"); if (!states.length) return;
        const s = states[model.get("i")], rows = s.rows, n = rows.length, rh = ROWS_H / n;
        svg.innerHTML = "";
        mk("rect", { x: sx(s.t) - 2, y: TOP, width: 4, height: ROWS_H, fill: "#f18f01", opacity: 0.6 });
        rows.forEach((r, j) => {
          const y = TOP + j * rh + rh / 2, gone = r.status.startsWith("out");
          const t = mk("text", { x: 8, y: y + 4, "font-size": 11, fill: "currentColor", opacity: gone ? 0.35 : 1 });
          t.textContent = "user " + (r.id + 1) + (r.status === "risk" || r.status === "event" || r.status === "cens" ? " ★" : "");
          mk("line", { x1: sx(0), x2: sx(r.dur), y1: y, y2: y, stroke: COL[r.status], "stroke-width": 8,
            "stroke-linecap": "round", "stroke-opacity": gone ? 0.15 : (r.status === "risk" ? 0.35 : 0.8) });
          mk("circle", { cx: sx(r.dur), cy: y, r: r.status === "event" || r.status === "cens" ? 9 : 6,
            fill: r.ev ? "#e4572e" : "white", stroke: r.ev ? "#e4572e" : "#2e86ab", "stroke-width": 3, opacity: gone ? 0.3 : 1 });
        });
        for (let t = 0; t <= 12; t += 1) {
          const tx = mk("text", { x: sx(t), y: CTOP + CH + 16, "text-anchor": "middle", "font-size": 11, fill: "currentColor" }); tx.textContent = t; }
        for (const v of [0, 0.5, 1]) {
          mk("line", { x1: PADL, x2: W - PADR, y1: sy(v), y2: sy(v), stroke: "currentColor", "stroke-opacity": v === 0.5 ? 0.35 : 0.12, "stroke-dasharray": v === 0.5 ? "4 4" : "" });
          const tx = mk("text", { x: PADL - 8, y: sy(v) + 4, "text-anchor": "end", "font-size": 11, fill: "currentColor" }); tx.textContent = v; }
        const lab = mk("text", { x: 8, y: CTOP + 10, "font-size": 12, fill: "#6c5b7b" }); lab.textContent = "Ŝ(t)";
        let d = `M${sx(0)},${sy(1)}`, prev = 1;
        s.curve.forEach(([t, S, c]) => { d += ` H${sx(t)} V${sy(S)}`; prev = S; });
        d += ` H${sx(s.t)}`;
        mk("path", { d, fill: "none", stroke: "#6c5b7b", "stroke-width": 3 });
        s.curve.forEach(([t, S, c]) => { if (c) mk("line", { x1: sx(t), x2: sx(t), y1: sy(S) - 7, y2: sy(S) + 7, stroke: "#2e86ab", "stroke-width": 2.5 }); });
        mk("circle", { cx: sx(s.t), cy: sy(prev), r: 5, fill: "#6c5b7b" });
        cap.innerHTML = `<b>step ${model.get("i")}/${states.length - 1}</b>: ${s.caption}`;
      }
      model.on("change:i", draw); model.on("change:states", draw); draw();
    }
    export default { render };
    """
        states = traitlets.List(traitlets.Dict()).tag(sync=True)
        i = traitlets.Int(0).tag(sync=True)

    return (KMStepper,)


@app.cell
def km_stepper(KMStepper, cohort_e, cohort_t, mo, np):
    def km_states(durations, events):
        _states = [dict(t=0.0, curve=[], caption="Day 0: everyone is subscribed, Ŝ(0) = 1. ★ marks the risk set.",
                        rows=[dict(id=j, dur=float(v), ev=int(e), status="risk") for j, (v, e) in enumerate(zip(durations, events))])]
        _S, _curve = 1.0, []
        for _tj in np.unique(durations):
            _rows = []
            for j, (v, e) in enumerate(zip(durations, events)):
                _st = ("out_event" if e else "out_cens") if v < _tj else (("event" if e else "cens") if v == _tj else "risk")
                _rows.append(dict(id=j, dur=float(v), ev=int(e), status=_st))
            _n = sum(r["status"] in ("risk", "event", "cens") for r in _rows)
            _d = sum(r["status"] == "event" for r in _rows)
            _c = sum(r["status"] == "cens" for r in _rows)
            _prev = _S
            _S = _S * (1 - _d / _n)
            _curve = _curve + [[float(_tj), _S, int(_c > 0)]]
            if _d:
                _cap = (f"Month {_tj:g}: <b>n = {_n}</b> at risk, <b style='color:#e4572e'>d = {_d}</b> cancel ⇒ "
                        f"Ŝ = {_prev:.3f} × (1 − {_d}/{_n}) = <b style='color:#6c5b7b'>{_S:.3f}</b>")
            else:
                _cap = f"Month {_tj:g}: nobody cancels, so the curve stays at {_S:.3f}."
            if _c:
                _cap += f" <span style='color:#2e86ab'>{_c} censored</span> leave the risk set (tick mark, no drop)."
            _states.append(dict(t=float(_tj), rows=_rows, curve=_curve, caption=_cap))
        return _states

    km_stepper = mo.ui.anywidget(KMStepper(states=km_states(cohort_t, cohort_e), i=0))
    km_stepper
    return (km_stepper,)


@app.cell
def km_table(cohort_km, km_stepper, mo):
    _i = km_stepper.value["i"]
    _tab = cohort_km.iloc[1:_i + 1][["t", "n", "d", "c", "S"]].rename(
        columns={"t": "month t_i", "n": "at risk n_i", "d": "events d_i", "c": "censored", "S": "Ŝ(t_i)"})
    _tab["1 − d/n"] = 1 - _tab["events d_i"] / _tab["at risk n_i"]
    mo.vstack([
        mo.md("**Life table so far.** It's the same numbers the stepper shows, one row per distinct time:"),
        mo.ui.table(_tab.round(3), selection=None, show_column_summaries=False) if len(_tab) else mo.md("_Press **next ▶**._"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    - Step to **month 8** in the video example: only **3** users are still at risk, so one cancellation knocks the curve down by a third (0.583 → 0.389).
    - Go back to section 1 and **turn user 2 (the 3-month user) into a cancellation**. Step through again: the drop at month 3 appears, and every later fraction has a smaller denominator.
    - Notice what censored users do *while they're in the risk set*: each cancellation is divided among **more** people, so the curve falls more slowly.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Checking it against the truth (and how to break it)

    Eight users can't prove anything, so DataMListic simulates **a thousand**. Each stay comes from a Weibull distribution with a
    **true median of 8.3 months** (a falling hazard, $k = 0.8$). Users sign up at random times during a two-year study, so many are
    still active when it ends. In the video, dropping the actives gives 2.9 months, "cancel today" gives 4.8, and Kaplan-Meier lands at 8.5,
    right on top of the truth.

    **The hidden condition: non-informative censoring.** KM works if *being observed tells you nothing about when someone will cancel*.
    Here censoring depends only on the sign-up date, so the condition holds. But if users who are about to leave **stop opening the app and you lose
    track of them early**, the risk set keeps the loyal ones and the curve **overestimates** how long people stay.

    The shaded band is a 95% confidence interval from **Greenwood's** standard error, the uncertainty the Mini Lecture mentions.
    """)
    return


@app.cell
def sim_controls(mo):
    sim_n = mo.ui.slider(50, 3000, step=50, value=1000, label="users", show_value=True)
    sim_study = mo.ui.slider(6, 60, step=1, value=24, label="study length (months)", show_value=True)
    sim_informative = mo.ui.switch(value=False, label="informative censoring (leavers go silent first)")
    sim_reroll = mo.ui.button(label="🎲 re-roll users", value=0, on_click=lambda v: v + 1)
    mo.hstack([sim_n, sim_study, sim_informative, sim_reroll], justify="start", gap=1.5, wrap=True)
    return sim_informative, sim_n, sim_reroll, sim_study


@app.cell
def sim_data(
    km_estimate,
    km_median,
    np,
    sim_informative,
    sim_n,
    sim_reroll,
    sim_study,
):
    SIM_K = 0.8
    SIM_LAM = 8.3 / np.log(2) ** (1 / SIM_K)        # Weibull scale so that the true median is 8.3 months

    _rng = np.random.default_rng(7 + sim_reroll.value)
    _T = SIM_LAM * _rng.weibull(SIM_K, sim_n.value)                 # true stay lengths
    _signup = _rng.uniform(0, sim_study.value, sim_n.value)
    _C = sim_study.value - _signup                                  # time until the study ends
    if sim_informative.value:
        _silent = (_rng.random(sim_n.value) < 0.5) & (_T < _C)
        _C = np.where(_silent, _T * _rng.uniform(0.3, 0.9, sim_n.value), _C)
    sim_t = np.minimum(_T, _C)
    sim_e = (_T <= _C).astype(int)
    sim_km = km_estimate(sim_t, sim_e)

    sim_naive_drop = float(np.median(sim_t[sim_e == 1]))
    sim_naive_today = float(np.median(sim_t))
    sim_km_median = km_median(sim_km)
    return (
        SIM_K,
        SIM_LAM,
        sim_e,
        sim_km,
        sim_km_median,
        sim_naive_drop,
        sim_naive_today,
        sim_t,
    )


@app.cell
def sim_chart(
    SIM_K,
    SIM_LAM,
    alt,
    mo,
    np,
    pd,
    sim_e,
    sim_km,
    sim_km_median,
    sim_naive_drop,
    sim_naive_today,
    sim_t,
):
    _km = sim_km.iloc[np.unique(np.linspace(0, len(sim_km) - 1, 400).astype(int))]
    _tmax = float(sim_t.max())
    _grid = np.linspace(0, _tmax, 200)
    _truth = pd.DataFrame({"t": _grid, "S": np.exp(-(_grid / SIM_LAM) ** SIM_K)})
    _x = alt.X("t:Q", title="months since sign-up")
    _band = alt.Chart(_km).mark_area(interpolate="step-after", color="#6c5b7b", opacity=0.2).encode(_x, y="lo:Q", y2="hi:Q")
    _line = alt.Chart(_km).mark_line(interpolate="step-after", color="#6c5b7b", strokeWidth=2.5).encode(
        _x, y=alt.Y("S:Q", title="S(t)", scale=alt.Scale(domain=[0, 1])))
    _tr = alt.Chart(_truth).mark_line(color="#555", strokeDash=[6, 4], strokeWidth=2).encode(_x, y="S:Q")
    _rules = alt.Chart(pd.DataFrame({"m": [sim_naive_drop, sim_naive_today], "what": ["drop actives", "cancel today"]})).mark_rule(
        color="#f18f01", strokeWidth=2).encode(x="m:Q", tooltip=["what", "m"])
    _true_med = 8.3
    _err = sim_km_median - _true_med
    mo.vstack([
        mo.hstack([
            mo.stat(f"{100 * (1 - sim_e.mean()):.0f}%", label="censored", caption="still active / lost"),
            mo.stat(f"{sim_naive_drop:.1f}", label="① drop actives", caption="months"),
            mo.stat(f"{sim_naive_today:.1f}", label="② cancel today", caption="months"),
            mo.stat("—" if np.isnan(sim_km_median) else f"{sim_km_median:.1f}", label="Kaplan-Meier median",
                    caption="never crosses ½" if np.isnan(sim_km_median) else f"{_err:+.1f} vs truth", bordered=True),
            mo.stat("8.3", label="true median", caption="Weibull k = 0.8"),
        ], justify="space-around"),
        (_band + _line + _tr + _rules).properties(width="container", height=280,
            title="Kaplan-Meier (purple, 95% band) vs the truth (dashed); amber = naive medians"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    - With the defaults, KM sits on the dashed truth while both amber lines are far to the left. **Re-roll** a few times: the naive answers are *consistently* wrong, not just noisy.
    - Shrink the **study to 6 months**. The curve simply stops early, and the band widens at the end where few users are at risk. KM won't extrapolate.
    - Turn on **informative censoring**. Now the purple curve floats **above** the truth: the leavers vanished as "censored", so the risk set is full of loyal users.
    - Drop **users to 50** and watch the Greenwood band grow. Uncertainty is part of the answer.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · What the shape of the hazard tells you (and what it doesn't)

    DataMListic finds the **hazard shape** more revealing than the survival curve: **rising** means wear-out,
    **falling** means early churn, and a **bathtub** is high, then flat, then rising again.

    But be careful reading a falling hazard as *"users get more loyal over time."* Their example: **40% of users are flaky** with a 30% chance of
    leaving each month, and **the rest are loyal** at 3%. *Every individual's* risk is constant, month after month, yet the group's hazard
    falls from about **14% toward 3%**, because the flaky users leave first:

    $$h_{\text{pop}}(t) = \frac{p\,q_f(1-q_f)^{t-1} + (1-p)\,q_\ell(1-q_\ell)^{t-1}}{p(1-q_f)^{t-1} + (1-p)(1-q_\ell)^{t-1}}$$

    The hazard describes **whoever is still in the risk set at that moment**.
    """)
    return


@app.cell
def mix_controls(mo):
    mix_p = mo.ui.slider(0, 1, step=0.05, value=0.4, label="share flaky p", show_value=True)
    mix_qf = mo.ui.slider(0.01, 0.6, step=0.01, value=0.30, label="flaky monthly hazard", show_value=True)
    mix_ql = mo.ui.slider(0.0, 0.3, step=0.005, value=0.03, label="loyal monthly hazard", show_value=True)
    mo.hstack([mix_p, mix_qf, mix_ql], justify="start", gap=1.5)
    return mix_p, mix_qf, mix_ql


@app.cell
def mix_chart(alt, mix_p, mix_qf, mix_ql, mo, np, pd):
    _t = np.arange(1, 37)
    _sf = mix_p.value * (1 - mix_qf.value) ** (_t - 1)
    _sl = (1 - mix_p.value) * (1 - mix_ql.value) ** (_t - 1)
    mix_hazard = (_sf * mix_qf.value + _sl * mix_ql.value) / (_sf + _sl)
    _df = pd.concat([
        pd.DataFrame({"month": _t, "hazard": mix_hazard, "who": "whole population"}),
        pd.DataFrame({"month": _t, "hazard": mix_qf.value, "who": "each flaky user"}),
        pd.DataFrame({"month": _t, "hazard": mix_ql.value, "who": "each loyal user"}),
    ])
    _h = alt.Chart(_df).mark_line(point=True).encode(
        x=alt.X("month:Q"), y=alt.Y("hazard:Q", title="monthly hazard", axis=alt.Axis(format="%")),
        color=alt.Color("who:N", scale=alt.Scale(domain=["whole population", "each flaky user", "each loyal user"],
                                                 range=["#e4572e", "#aaa", "#2e86ab"]), legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.condition(alt.datum.who == "whole population", alt.value([1, 0]), alt.value([5, 4])),
    ).properties(width=600, height=220, title="Constant individual hazards, falling population hazard")
    _comp = alt.Chart(pd.DataFrame({"month": _t, "flaky share of risk set": _sf / (_sf + _sl)})).mark_area(
        color="#aaa", opacity=0.6).encode(x="month:Q", y=alt.Y("flaky share of risk set:Q", axis=alt.Axis(format="%"),
                                                               scale=alt.Scale(domain=[0, 1]))).properties(
        width=600, height=110, title="Who is still in the risk set?")
    mo.vstack([alt.vconcat(_h, _comp),
               mo.md(f"Population hazard in month 1: **{mix_hazard[0]:.1%}**; in month 36: **{mix_hazard[-1]:.1%}**.")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    - At the defaults, month 1 is ≈ 13.8% (0.4·30% + 0.6·3%), and it slides down toward 3% as the grey area empties.
    - Set the **two hazards equal**. The population hazard goes flat, and the "loyalty" illusion disappears.
    - Set **p = 0** or **p = 1**: a single homogeneous group has a constant hazard.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Comparing two groups: the log-rank test

    Now there are two curves: women vs men in a lung-cancer cohort (the Mini Lecture's example) or **annual vs monthly** subscribers
    (DataMListic's example). Is the gap real or just noise?

    The **log-rank test** walks through every event time $t_j$. If the groups had the same survival, then the $d_j$ events would be split
    in proportion to who's at risk. Group 1's **expected** share is

    $$E_{1j} = d_j \frac{n_{1j}}{n_j}, \qquad V_j = d_j\frac{n_{1j}}{n_j}\Big(1-\frac{n_{1j}}{n_j}\Big)\frac{n_j-d_j}{n_j-1}$$

    and we compare it with the **observed** $O_{1j}$, summed over time:

    $$\chi^2 = \frac{\big(\sum_j O_{1j} - \sum_j E_{1j}\big)^2}{\sum_j V_j} \;\sim\; \chi^2_1 \text{ under } H_0.$$

    The lung data is the classic NCCTG `lung` dataset from R's **survival** package, the package the Mini Lecture recommends.
    It has 228 patients, with time in days, and loads from a public Rdatasets mirror.
    """)
    return


@app.cell
def lung(mo, pd):
    lung = pd.read_csv("https://vincentarelbundock.github.io/Rdatasets/csv/survival/cancer.csv")
    lung = lung.assign(event=(lung["status"] == 2).astype(int), female=lung["sex"] == 2)[["time", "event", "female", "age"]]
    mo.md(f"Loaded **{len(lung)}** lung-cancer patients, **{lung['event'].sum()}** deaths, "
          f"**{lung['female'].sum()}** women.")
    return (lung,)


@app.cell
def grp_controls(mo):
    grp_source = mo.ui.dropdown(["Lung cancer: women vs men", "Simulated: annual vs monthly plan"],
                                value="Lung cancer: women vs men", label="data")
    grp_hr = mo.ui.slider(0.2, 2.0, step=0.05, value=0.5, label="true hazard ratio annual / monthly (simulation)", show_value=True)
    grp_n = mo.ui.slider(20, 1000, step=10, value=200, label="users per group (simulation)", show_value=True)
    mo.hstack([grp_source, grp_hr, grp_n], justify="start", gap=1.5, wrap=True)
    return grp_hr, grp_n, grp_source


@app.cell
def grp_data(
    SIM_K,
    SIM_LAM,
    grp_hr,
    grp_n,
    grp_source,
    km_estimate,
    logrank,
    lung,
    np,
):
    if grp_source.value.startswith("Lung"):
        grp_t, grp_e, grp_g = lung["time"].to_numpy(float), lung["event"].to_numpy(), lung["female"].to_numpy()
        grp_names, grp_unit = ("women", "men"), "days"
    else:
        _rng = np.random.default_rng(11)
        _g = np.repeat([True, False], grp_n.value)
        _lam = np.where(_g, SIM_LAM * grp_hr.value ** (-1 / SIM_K), SIM_LAM)   # PH Weibull: scale ∝ HR^(-1/k)
        _T = _lam * _rng.weibull(SIM_K, _g.size)
        _C = _rng.uniform(0, 24, _g.size)
        grp_t, grp_e, grp_g = np.minimum(_T, _C), (_T <= _C).astype(int), _g
        grp_names, grp_unit = ("annual", "monthly"), "months"
    grp_test = logrank(grp_t, grp_e, grp_g)
    grp_km = {grp_names[0]: km_estimate(grp_t[grp_g], grp_e[grp_g]),
              grp_names[1]: km_estimate(grp_t[~grp_g], grp_e[~grp_g])}
    return grp_km, grp_names, grp_test, grp_unit


@app.cell
def grp_chart(
    alt,
    grp_km,
    grp_names,
    grp_test,
    grp_unit,
    km_median,
    mo,
    np,
    pd,
    step_df,
):
    _df = pd.concat([step_df(_k, _n) for _n, _k in grp_km.items()])
    _col = alt.Color("curve:N", scale=alt.Scale(domain=list(grp_names), range=["#6c5b7b", "#17bebb"]),
                     legend=alt.Legend(orient="top", title=None))
    _x = alt.X("t:Q", title=grp_unit)
    _km_chart = (alt.Chart(_df).mark_area(interpolate="step-after", opacity=0.15).encode(_x, y="lo:Q", y2="hi:Q", color=_col)
                 + alt.Chart(_df).mark_line(interpolate="step-after", strokeWidth=2.5).encode(
                     _x, y=alt.Y("S:Q", title="Ŝ(t)", scale=alt.Scale(domain=[0, 1])), color=_col)
                 ).properties(width=600, height=260, title="Kaplan-Meier per group with 95% bands")
    _tab = grp_test["table"].assign(cum_O_minus_E=lambda d: (d["O1"] - d["E1"]).cumsum())
    _oe = alt.Chart(_tab).mark_line(color="#e4572e", interpolate="step-after").encode(
        x=alt.X("t:Q", title=grp_unit), y=alt.Y("cum_O_minus_E:Q", title=f"cumulative O − E ({grp_names[0]})")
    ) + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="gray").encode(y="y:Q")
    _meds = {n: km_median(k) for n, k in grp_km.items()}
    _p = grp_test["p"]
    mo.vstack([
        mo.hstack([
            mo.stat(f"{grp_test['O1']:.0f} / {grp_test['E1']:.1f}", label=f"{grp_names[0]}: observed / expected"),
            mo.stat(f"{grp_test['O2']:.0f} / {grp_test['E2']:.1f}", label=f"{grp_names[1]}: observed / expected"),
            mo.stat(f"{grp_test['chi2']:.2f}", label="log-rank χ²"),
            mo.stat(f"{_p:.2g}", label="p-value", caption="significant at 5%" if _p < 0.05 else "not significant", bordered=True),
        ], justify="space-around"),
        mo.md(" · ".join(f"median {n}: **{'not reached' if np.isnan(m) else f'{m:.0f} {grp_unit}'}**" for n, m in _meds.items())),
        alt.vconcat(_km_chart, _oe.properties(width=600, height=140,
            title="Where the evidence piles up: cumulative observed − expected events")),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    - **Lung data:** women have fewer deaths than expected (O < E) and a higher curve. The log-rank χ² ≈ 10.3, p ≈ 0.001, the same as R's `survdiff`.
    - **Simulation:** set the hazard ratio to **1.0**. The curves overlap and the p-value bounces around. Then try **0.5** with only **20 users per group**: the effect is real, but is it detectable?
    - Watch the **O − E** line: a steady slope means a consistent difference over time. That is the "proportional hazards" situation the log-rank test is most powerful for.

    **Next step: the Cox model.** The log-rank test says *whether* the groups differ. The **Cox proportional hazards model**,
    $h(t\mid x) = h_0(t)\,e^{\beta x}$, says *by how much*: $e^\beta$ is the **hazard ratio**, it handles several covariates at once, and,
    as DataMListic shows, the baseline $h_0(t)$ **cancels out** of the partial likelihood and never has to be estimated. On simulated plans
    with a true ratio of ½, it recovers 0.51. That gets its own notebook: **`cox_proportional_hazards.py`**.
    """), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Sandbox: paint a hazard, watch everything else follow

    Paint the **monthly hazard** $h_t$, the chance of leaving during month $t$ given you were still here at its start. The purple line
    is the survival curve it implies, $S(t)=\prod_{u\le t}(1-h_u)$, computed live in your browser. When you release the mouse,
    Python simulates a cohort with **your** hazard, censors it with a study window, and checks Kaplan-Meier against the truth.
    """)
    return


@app.cell
def HazardPainter_cls(anywidget, traitlets):
    class HazardPainter(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 660, H = 280, PADL = 46, PADR = 46, TOP = 14, BOT = 34, HMAX = 0.4;
      const NS = "http://www.w3.org/2000/svg";
      const bar = document.createElement("div"); bar.style.marginBottom = "6px"; el.appendChild(bar);
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none", cursor: "crosshair" });
      el.appendChild(svg);
      const mk = (tag, attrs) => { const e = document.createElementNS(NS, tag);
        for (const k in attrs) e.setAttribute(k, attrs[k]); svg.appendChild(e); return e; };
      const toSvg = (evt) => { const p = svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY;
        return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const plotW = W - PADL - PADR, plotH = H - TOP - BOT;
      const presets = {
        "Constant 10%": (k, n) => 0.10,
        "Early churn": (k, n) => 0.03 + 0.30 * Math.exp(-k / 2.5),
        "Wear-out": (k, n) => 0.01 + 0.25 * Math.pow(k / n, 3),
        "Bathtub": (k, n) => 0.03 + 0.25 * Math.exp(-k / 1.8) + 0.30 * Math.pow(k / n, 6),
        "Free trial cliff": (k, n) => (k === 0 ? 0.35 : (k === 11 ? 0.30 : 0.02)),
      };
      for (const name in presets) {
        const b = document.createElement("button"); b.textContent = name; b.style.marginRight = "6px";
        b.onclick = () => { const n = model.get("hazard").length;
          model.set("hazard", Array.from({ length: n }, (_, k) => Math.min(HMAX, presets[name](k, n)))); model.save_changes(); };
        bar.appendChild(b);
      }
      let painting = false;
      function draw() {
        const h = model.get("hazard"), n = h.length, bw = plotW / n;
        svg.innerHTML = "";
        for (const v of [0, 0.1, 0.2, 0.3, 0.4]) {
          const y = TOP + plotH * (1 - v / HMAX);
          mk("line", { x1: PADL, x2: W - PADR, y1: y, y2: y, stroke: "currentColor", "stroke-opacity": 0.1 });
          const t = mk("text", { x: PADL - 6, y: y + 4, "text-anchor": "end", "font-size": 10, fill: "#e4572e" }); t.textContent = (v * 100) + "%";
          const t2 = mk("text", { x: W - PADR + 6, y: y + 4, "font-size": 10, fill: "#6c5b7b" }); t2.textContent = (v / HMAX).toFixed(2);
        }
        h.forEach((v, k) => {
          const bh = plotH * v / HMAX;
          mk("rect", { x: PADL + k * bw + 1, y: TOP + plotH - bh, width: bw - 2, height: bh, fill: "#e4572e", opacity: 0.55, rx: 2 });
          if (k % 3 === 2) { const t = mk("text", { x: PADL + (k + 0.5) * bw, y: H - BOT + 14, "text-anchor": "middle", "font-size": 10, fill: "currentColor" }); t.textContent = k + 1; }
        });
        let S = 1, d = `M${PADL},${TOP}`, med = null;
        h.forEach((v, k) => { S *= 1 - v; const x = PADL + (k + 1) * bw, y = TOP + plotH * (1 - S);
          d += ` H${x} V${y}`; if (med === null && S <= 0.5) med = k + 1; });
        mk("path", { d, fill: "none", stroke: "#6c5b7b", "stroke-width": 3 });
        const lab = mk("text", { x: PADL + 4, y: H - 4, "font-size": 11, fill: "currentColor" });
        lab.textContent = `month →   bars: hazard h_t (left axis)   line: S(t) (right axis)   median: ${med === null ? "not reached" : med + " mo"}   S(24) = ${S.toFixed(2)}`;
      }
      function paint(evt) {
        const p = toSvg(evt), h = [...model.get("hazard")], n = h.length, bw = plotW / n;
        const k = Math.floor((p.x - PADL) / bw); if (k < 0 || k >= n) return;
        h[k] = Math.max(0, Math.min(HMAX, (TOP + plotH - p.y) / plotH * HMAX));
        model.set("hazard", h);
      }
      svg.addEventListener("pointerdown", e => { painting = true; svg.setPointerCapture(e.pointerId); paint(e); });
      svg.addEventListener("pointermove", e => { if (painting) paint(e); });
      const stop = () => { if (painting) { painting = false; model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointercancel", stop);
      model.on("change:hazard", draw); draw();
    }
    export default { render };
    """
        hazard = traitlets.List(traitlets.Float()).tag(sync=True)

    return (HazardPainter,)


@app.cell
def painter(HazardPainter, mo, np):
    painter = mo.ui.anywidget(HazardPainter(hazard=[0.03 + 0.25 * np.exp(-k / 1.8) + 0.30 * (k / 24) ** 6 for k in range(24)]))
    sand_n = mo.ui.slider(50, 3000, step=50, value=500, label="cohort size", show_value=True)
    sand_study = mo.ui.slider(3, 24, step=1, value=18, label="max follow-up (months)", show_value=True)
    mo.vstack([painter, mo.hstack([sand_n, sand_study], justify="start", gap=2)])
    return painter, sand_n, sand_study


@app.cell
def sandbox_readout(
    alt,
    km_estimate,
    km_median,
    mo,
    np,
    painter,
    pd,
    sand_n,
    sand_study,
):
    _h = np.clip(np.array(painter.value["hazard"]), 0, 1)
    _months = np.arange(1, len(_h) + 1)
    sand_true_S = np.cumprod(1 - _h)
    _rng = np.random.default_rng(3)
    _U = _rng.random((sand_n.value, len(_h)))
    _left = _U < _h
    _T = np.where(_left.any(1), _left.argmax(1) + 1, len(_h) + 1)
    _C = _rng.integers(1, sand_study.value + 1, sand_n.value)
    _t, _e = np.minimum(_T, _C), (_T <= _C).astype(int)
    sand_km = km_estimate(_t, _e)
    _true_med = float(_months[np.argmax(sand_true_S <= 0.5)]) if (sand_true_S <= 0.5).any() else float("nan")
    _km_med = km_median(sand_km)
    _truth = pd.DataFrame({"t": np.r_[0, _months], "S": np.r_[1, sand_true_S]})
    _x = alt.X("t:Q", title="month", scale=alt.Scale(domain=[0, 24]))
    _chart = (alt.Chart(sand_km).mark_area(interpolate="step-after", color="#6c5b7b", opacity=0.2).encode(_x, y="lo:Q", y2="hi:Q")
              + alt.Chart(sand_km).mark_line(interpolate="step-after", color="#6c5b7b", strokeWidth=3).encode(
                  _x, y=alt.Y("S:Q", scale=alt.Scale(domain=[0, 1]), title="S(t)"))
              + alt.Chart(_truth).mark_line(interpolate="step-after", color="#555", strokeDash=[6, 4]).encode(_x, y="S:Q"))
    _fmt = lambda m: "not reached" if np.isnan(m) else f"{m:.0f} mo"
    mo.vstack([
        mo.hstack([
            mo.stat(_fmt(_true_med), label="true median"),
            mo.stat(_fmt(_km_med), label="KM median", bordered=True),
            mo.stat(f"{100 * (1 - _e.mean()):.0f}%", label="censored"),
            mo.stat(f"{np.median(_t[_e == 1]) if _e.sum() else float('nan'):.0f} mo", label="naive ① (drop actives)"),
        ], justify="space-around"),
        _chart.properties(width="container", height=240, title="Simulated cohort: Kaplan-Meier (purple) vs your painted truth (dashed)"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Challenges**

    1. Paint a hazard whose **median is exactly 6 months**. The JS readout updates while you paint.
    2. Make the **naive ① median** look *great* (close to the truth) while censoring is still above 40%. *Hint: which hazard shapes make early leavers representative?*
    3. Use **Free trial cliff**: why is there a big drop right after month 12? Now set follow-up to **10 months** and see what KM can and can't tell you.
    4. Paint a hazard of **0.4 in every month**. Check that $S(t) = 0.6^t$, so the median is 2 months.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Quiz

    Answer everything, then flip the switch to see your score.
    """)
    return


@app.cell
def quiz(mo):
    QUIZ = {
        "q1": ("A user joined 3 months ago and is still subscribed. Their stay is…",
               ["missing data", "right-censored at 3 months", "left-censored at 3 months", "left-truncated"],
               "right-censored at 3 months"),
        "q2": ("You find the bird's egg already laid when you check in the morning. The laying time is…",
               ["right-censored", "left-censored", "right-truncated", "a competing risk"], "left-censored"),
        "q3": ("A study uses only historical death records, so you have to have died to be in it. That is…",
               ["left-censoring", "interval-censoring", "right-truncation", "missing data"], "right-truncation"),
        "q4": ("Users about to quit stop opening the app and get lost from follow-up early. Kaplan-Meier will…",
               ["underestimate survival", "be unaffected", "overestimate survival"], "overestimate survival"),
        "q5": ("Ŝ was 0.80 just before a month in which 5 users are at risk and 1 cancels. The new Ŝ is…",
               ["0.60", "0.64", "0.75", "0.80"], f"{0.80 * (1 - 1 / 5):.2f}"),
        "q6": ("A constant hazard λ gives S(t) =", ["1 − λt", "λ e^(−λt)", "e^(−λt)", "λt"], "e^(−λt)"),
        "q7": ("Each user has a constant personal hazard, yet the population hazard falls over time. Why?",
               ["users become more loyal", "high-risk users leave first, so the risk set changes", "censoring bias", "the log-rank test"],
               "high-risk users leave first, so the risk set changes"),
        "q8": ("The log-rank test compares…",
               ["the two medians", "observed vs expected events at each event time", "the means of the observed times", "the hazard ratio e^β"],
               "observed vs expected events at each event time"),
    }
    quiz = mo.ui.dictionary({k: mo.ui.radio(opts, label=f"**{i + 1}.** {q}") for i, (k, (q, opts, _)) in enumerate(QUIZ.items())})
    quiz_reveal = mo.ui.switch(label="reveal answers & score")
    mo.vstack([quiz, quiz_reveal])
    return QUIZ, quiz, quiz_reveal


@app.cell
def quiz_score(QUIZ, mo, quiz, quiz_reveal):
    mo.stop(not quiz_reveal.value, mo.md("_Flip the switch when you're done._"))
    _rows, _score = [], 0
    for _k, (_q, _opts, _ans) in QUIZ.items():
        _got = quiz.value[_k]
        _ok = _got == _ans
        _score += _ok
        _rows.append(f"| {'✅' if _ok else '❌'} | {_q} | {_got or '—'} | **{_ans}** |")
    mo.vstack([
        mo.callout(mo.md(f"### Score: {_score} / {len(QUIZ)}"), kind="success" if _score == len(QUIZ) else "warn"),
        mo.md("| | question | yours | correct |\n|---|---|---|---|\n" + "\n".join(_rows)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap

    | idea | formula / rule | the aha | where you touched it |
    |---|---|---|---|
    | **Censoring** | we know only $T > c$ | an unfinished story is a real **lower bound**, not missing data | §1 drag-the-cohort timeline |
    | Naive fixes | drop actives / "cancel today" | both are **too low**, and dropping actives throws out the loyal users | §1, §4 amber lines |
    | Other incompleteness | left/interval-censoring, truncation, competing risks | bound vs never-in-the-data vs missing value | §1 table |
    | **Survival function** | $S(t)=P(T>t)$, median at ½ | a curve, because actives can answer "past $t$?" | §2 |
    | **Hazard** | $h=f/S$, $S=e^{-\int h}$ | the risk *right now* given you've lasted; the $e$ is a long product of near-1 factors | §2 slices demo |
    | **Kaplan-Meier** | $\hat S(t)=\prod(1-d_i/n_i)$ | censored people stay in the risk set until they're no longer observed | §3 stepper |
    | Assumption | non-informative censoring | if leavers vanish first, KM overestimates | §4 switch |
    | Uncertainty | Greenwood SE → 95% band | few at risk means a wide band | §4, §6 |
    | Hazard shape | rising / falling / bathtub | a falling population hazard can come from **selection**, not loyalty | §5 flaky vs loyal |
    | **Log-rank test** | $\chi^2 = (O-E)^2/V$ | observed vs expected events, summed over time | §6 lung data |
    | Cox PH (preview) | $h_0(t)e^{\beta x}$ | the hazard ratio $e^\beta$; $h_0$ cancels | → `cox_proportional_hazards.py` |
    """)
    return


if __name__ == "__main__":
    app.run()

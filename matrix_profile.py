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
    import stumpy

    return alt, anywidget, mo, np, pd, stumpy, traitlets


@app.cell(hide_code=True)
def title(mo):
    mo.vstack([
        mo.md(r"""
    # STUMPY & matrix profiles: find the repeats and the weirdos in any time series
    *An interactive companion to Sean Law's SciPy 2024 talk **"STUMPY: Modern Time Series Analysis with Matrix Profiles"**.*
    """),
        mo.Html('<iframe width="100%" height="390" src="https://www.youtube.com/embed/0O6dlq6a4rA" title="STUMPY talk" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'),
        mo.md(r"""
    > Someone drops a time series in your lap. *Is there any signal, or is it just noise?*

    Sean's answer in one sentence: **compare every subsequence with every other one using plain Pythagoras, keep only the distance to each
    subsequence's single nearest neighbour, and that one vector — the _matrix profile_ — tells you where behaviour is conserved (low values)
    and where anomalies hide (high values).** Each section below lets you rebuild that idea by hand, in the talk's order.
    """),
    ])
    return


@app.cell(hide_code=True)
def colors(mo):
    mo.md(r"""
    ### Colour language (used everywhere below)
    <span style="color:#4ea8de">■ **light blue**</span> = the *query* subsequence you are asking about &nbsp;·&nbsp;
    <span style="color:#1d3557">■ **dark blue**</span> = the subsequence it is compared with / its nearest neighbour &nbsp;·&nbsp;
    <span style="color:#e4572e">■ **red**</span> = distances & **discords** (anomalies) &nbsp;·&nbsp;
    <span style="color:#f18f01">■ **orange**</span> = **motifs** (conserved patterns) &nbsp;·&nbsp;
    <span style="color:#999">■ **grey**</span> = the raw series / the excluded "trivial match" zone.

    Notation from the talk: time series $T$ of length $n$, window (subsequence length) $m$, subsequence $T_{i,m} = T[i : i+m]$.
    """)
    return


@app.cell
def helpers(np):
    COLORS = {"query": "#4ea8de", "nn": "#1d3557", "dist": "#e4572e", "motif": "#f18f01", "series": "#999999"}


    def subsequences(T, m):
        """All length-m windows of T, one per row (n - m + 1 rows)."""
        return np.lib.stride_tricks.sliding_window_view(np.asarray(T, dtype=float), m)


    def znorm(S):
        """z-normalise each row: subtract its mean, divide by its std (constant rows -> zeros)."""
        _mu = S.mean(axis=-1, keepdims=True)
        _sd = S.std(axis=-1, keepdims=True)
        _ok = _sd > 1e-12
        return np.where(_ok, (S - _mu) / np.where(_ok, _sd, 1.0), 0.0)


    def exclusion_zone(m):
        """STUMPY's default trivial-match zone: |i - j| <= ceil(m / 4)."""
        return int(np.ceil(m / 4))


    def distance_profile(T, i, m, normalize=True):
        """Brute force: Euclidean distance from T[i:i+m] to every other window."""
        S = subsequences(T, m)
        if normalize:
            S = znorm(S)
        return np.sqrt(((S - S[i]) ** 2).sum(axis=1))


    def distance_matrix(T, m, normalize=True):
        """Stack every distance profile: the full (n-m+1) x (n-m+1) matrix."""
        S = subsequences(T, m)
        if normalize:
            S = znorm(S)
        _sq = (S ** 2).sum(axis=1)
        return np.sqrt(np.clip(_sq[:, None] + _sq[None, :] - 2 * S @ S.T, 0, None))


    def naive_matrix_profile(T, m, normalize=True):
        """Matrix profile P and index I: row-wise minimum of the distance matrix outside the trivial zone."""
        D = distance_matrix(T, m, normalize)
        _k = np.arange(len(D))
        D = np.where(np.abs(_k[:, None] - _k[None, :]) <= exclusion_zone(m), np.inf, D)
        return D.min(axis=1), D.argmin(axis=1)

    return (
        COLORS,
        distance_matrix,
        distance_profile,
        exclusion_zone,
        naive_matrix_profile,
        znorm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Subsequences and the most boring distance in the world

    The talk starts with a tiny series of **13 numbers**. Stare at it: do you see a pattern? Probably not yet.

    **Subsequence:** just a short section of the full series, $T_{i,m} = (t_i, t_{i+1}, \dots, t_{i+m-1})$.

    **Comparing two subsequences:** Sean picks the simplest thing he knows, the straight-line (Euclidean) distance.
    For length 2 it *is* Pythagoras; for longer windows you just keep adding squared legs:

    $$h = \sqrt{a^2 + b^2} \quad\longrightarrow\quad d(T_{i,m}, T_{j,m}) = \sqrt{a^2 + b^2 + c^2 + d^2 + \dots}, \qquad a = t_i - t_j,\; b = t_{i+1}-t_{j+1}, \dots$$

    Identical windows → distance **0**; anything else → positive. No magic, no training data, one interpretable parameter $m$.

    **Drag the light-blue and dark-blue windows** below. The lower panel overlays them; each red dashed leg is one term in the sum.
    """)
    return


@app.cell
def toy(mo, np):
    T_toy = np.array([0, 1, 3, 2, 9, 1, 14, 15, 1, 2, 2, 10, 7], dtype=float)
    m_pair = mo.ui.slider(2, 6, value=4, label="window length m", show_value=True)
    m_pair
    return T_toy, m_pair


@app.cell
def window_pair_widget(anywidget, traitlets):
    class WindowPair(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const NS = "http://www.w3.org/2000/svg", W = 680, H = 330;
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none", fontFamily: "sans-serif" });
      el.appendChild(svg);
      const mk = (t, a, p = svg) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
      const toSvg = (e) => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const LB = "#4ea8de", DB = "#1d3557", RED = "#e4572e";
      let drag = null, grab = 0;
      const stepOf = () => (W - 60) / (model.get("series").length - 1);
      function draw() {
        svg.innerHTML = "";
        const T = model.get("series"), n = T.length, m = model.get("m"), a = model.get("a"), b = model.get("b");
        const step = stepOf(), lo0 = Math.min(...T), hi0 = Math.max(...T), pad = (hi0 - lo0) * 0.12 || 1;
        const sx = i => 40 + i * step, sy = v => 150 - (v - (lo0 - pad)) / ((hi0 + pad) - (lo0 - pad)) * 125;
        mk("text", { x: 8, y: 13, "font-size": 12, fill: "currentColor" }).textContent = "Drag the shaded windows along the series";
        for (const [p, c, key] of [[a, LB, "a"], [b, DB, "b"]]) {
          const r = mk("rect", { x: sx(p) - step * 0.45, y: 20, width: step * (m - 1) + step * 0.9, height: 138, fill: c, "fill-opacity": 0.18, stroke: c, "stroke-width": 2, rx: 6 });
          r.style.cursor = "grab";
          r.addEventListener("pointerdown", e => { drag = key; grab = Math.round((toSvg(e).x - 40) / step) - model.get(key); svg.setPointerCapture(e.pointerId); });
        }
        mk("polyline", { points: T.map((v, i) => `${sx(i)},${sy(v)}`).join(" "), fill: "none", stroke: "#999", "stroke-width": 2, "pointer-events": "none" });
        T.forEach((v, i) => {
          const inA = i >= a && i < a + m, inB = i >= b && i < b + m;
          mk("circle", { cx: sx(i), cy: sy(v), r: 4.5, fill: inB ? DB : inA ? LB : "#999", "pointer-events": "none" });
          mk("text", { x: sx(i), y: sy(v) - 8, "text-anchor": "middle", "font-size": 11, fill: "currentColor", "pointer-events": "none" }).textContent = +v.toFixed(2);
        });
        const A = T.slice(a, a + m), B = T.slice(b, b + m), all = A.concat(B);
        const l2 = Math.min(...all), h2 = Math.max(...all), p2 = (h2 - l2) * 0.15 || 1;
        const ox = k => 140 + k * (W - 300) / Math.max(m - 1, 1), oy = v => 318 - (v - (l2 - p2)) / ((h2 + p2) - (l2 - p2)) * 128;
        mk("text", { x: 8, y: 180, "font-size": 12, fill: "currentColor" }).textContent = "Overlay (red legs = differences a, b, c, ...)";
        let s = 0;
        A.forEach((v, k) => {
          const d = v - B[k]; s += d * d;
          mk("line", { x1: ox(k), x2: ox(k), y1: oy(v), y2: oy(B[k]), stroke: RED, "stroke-width": 3, "stroke-dasharray": "5 3" });
          mk("text", { x: ox(k) + 6, y: (oy(v) + oy(B[k])) / 2 + 4, "font-size": 11, fill: RED }).textContent = Math.abs(d) > 1e-9 ? "abcdefgh"[k] + "=" + +d.toFixed(2) : "";
        });
        for (const [S, c] of [[A, LB], [B, DB]]) {
          mk("polyline", { points: S.map((v, k) => `${ox(k)},${oy(v)}`).join(" "), fill: "none", stroke: c, "stroke-width": 3 });
          S.forEach((v, k) => mk("circle", { cx: ox(k), cy: oy(v), r: 5, fill: c }));
        }
        mk("text", { x: W - 8, y: 215, "text-anchor": "end", "font-size": 17, "font-weight": "bold", fill: RED }).textContent = `d = √${+s.toFixed(2)} = ${Math.sqrt(s).toFixed(2)}`;
      }
      svg.addEventListener("pointermove", e => {
        if (!drag) return;
        const n = model.get("series").length, m = model.get("m");
        const p = Math.max(0, Math.min(n - m, Math.round((toSvg(e).x - 40) / stepOf()) - grab));
        if (p !== model.get(drag)) model.set(drag, p);
      });
      const stop = () => { if (drag) { drag = null; model.save_changes(); } };
      svg.addEventListener("pointerup", stop);
      svg.addEventListener("pointerleave", stop);
      for (const t of ["series", "m", "a", "b"]) model.on("change:" + t, draw);
      draw();
    }
    export default { render };
    """
        series = traitlets.List(traitlets.Float()).tag(sync=True)
        m = traitlets.Int(4).tag(sync=True)
        a = traitlets.Int(1).tag(sync=True)
        b = traitlets.Int(8).tag(sync=True)

    return (WindowPair,)


@app.cell
def pair(T_toy, WindowPair, m_pair, mo):
    pair = mo.ui.anywidget(WindowPair(series=T_toy.tolist(), m=m_pair.value, a=1, b=len(T_toy) - m_pair.value - 1))
    pair
    return (pair,)


@app.cell
def pair_readout(T_toy, distance_profile, mo, np, pair, pd):
    _a, _b, _m = pair.value["a"], pair.value["b"], pair.value["m"]
    _A, _B = T_toy[_a:_a + _m], T_toy[_b:_b + _m]
    _diff = _A - _B
    _pair_df = pd.DataFrame({
        "leg": list("abcdefgh"[:_m]),
        f"light blue T[{_a}+k]": _A,
        f"dark blue T[{_b}+k]": _B,
        "difference": _diff,
        "squared": _diff ** 2,
    })
    _d = float(np.sqrt((_diff ** 2).sum()))
    mo.hstack([
        mo.ui.table(_pair_df, selection=None, show_column_summaries=False),
        mo.vstack([
            mo.stat(value=f"{_d:.3f}", label="Euclidean distance", caption=f"√{(_diff ** 2).sum():g}  (sum of squared legs)"),
            mo.stat(value=f"{float(distance_profile(T_toy, _a, _m, normalize=False)[_b]):.3f}", label="same, via distance_profile()", caption="the helper agrees"),
        ]),
    ], widths=[3, 2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. With $m=2$, put the windows on `[1, 3]` and `[2, 2]` (i = 1 and i = 9). The talk's example: $h=\sqrt{1^2+1^2}=\sqrt 2$.
    2. With $m=4$, find the pair whose distance is $\sqrt{2}\approx 1.41$. Hint: one starts at **1**. Remember it — it comes back in section 3.
    3. Put both windows on the *same* spot. Distance is 0: that is the **trivial self-match** we'll have to ignore later.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Brute force: the distance profile

    To ask *"is this subsequence conserved — does it repeat?"*, Sean copies the series, pins the **light-blue** query in place and
    slides the **dark-blue** window from the start to the end, writing each Euclidean distance into a vector.
    That vector is the **distance profile** of subsequence $i$:

    $$D_i[j] = d\big(T_{i,m},\, T_{j,m}\big), \qquad j = 0, \dots, n-m.$$

    It is *not* an approximation: every comparison is made. Two things to read off it:

    * At $j=i$ the distance is **0** — the **trivial match** (a window matches itself). Neighbouring, overlapping windows are
      nearly trivial too, so STUMPY ignores an **exclusion zone** $|i-j| \le \lceil m/4 \rceil$ (grey).
    * The smallest value *outside* that zone is the query's **1-nearest neighbour** (dark blue bar).

    Pick a query, then press **▶ play** and watch the profile being built one slide at a time.
    """)
    return


@app.cell
def dp_query(T_toy, mo):
    dp_query = mo.ui.slider(0, len(T_toy) - 4, value=1, label="query subsequence i (m = 4)", show_value=True)
    dp_query
    return (dp_query,)


@app.cell
def slide_animator_widget(anywidget, traitlets):
    class SlideAnimator(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const NS = "http://www.w3.org/2000/svg", W = 680, H = 340;
      const ctl = document.createElement("div"); ctl.style.marginBottom = "4px"; el.appendChild(ctl);
      const btn = (label, fn) => { const b = document.createElement("button"); b.textContent = label; b.onclick = fn;
        Object.assign(b.style, { marginRight: "6px", padding: "3px 10px", cursor: "pointer" }); ctl.appendChild(b); return b; };
      const info = document.createElement("span"); info.style.marginLeft = "8px";
      const svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", fontFamily: "sans-serif" }); el.appendChild(svg);
      const mk = (t, a) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); svg.appendChild(e); return e; };
      const LB = "#4ea8de", DB = "#1d3557", RED = "#e4572e";
      let timer = null;
      function draw() {
        svg.innerHTML = "";
        const T = model.get("series"), n = T.length, m = model.get("m"), q = model.get("query"), P = model.get("profile"), z = model.get("excl");
        const k = model.get("k"), L = P.length;
        const step = (W - 60) / (n - 1), sx = i => 40 + i * step;
        const lo = Math.min(...T), hi = Math.max(...T), pad = (hi - lo) * 0.12 || 1;
        const sy = v => 135 - (v - (lo - pad)) / ((hi + pad) - (lo - pad)) * 115;
        mk("rect", { x: sx(q) - step * 0.45, y: 12, width: step * (m - 1) + step * 0.9, height: 130, fill: LB, "fill-opacity": 0.2, stroke: LB, "stroke-width": 2, rx: 6 });
        if (k >= 0) mk("rect", { x: sx(k) - step * 0.45, y: 16, width: step * (m - 1) + step * 0.9, height: 122, fill: DB, "fill-opacity": 0.15, stroke: DB, "stroke-width": 2, "stroke-dasharray": "6 3", rx: 6 });
        mk("polyline", { points: T.map((v, i) => `${sx(i)},${sy(v)}`).join(" "), fill: "none", stroke: "#999", "stroke-width": 2 });
        T.forEach((v, i) => mk("circle", { cx: sx(i), cy: sy(v), r: 4, fill: (k >= 0 && i >= k && i < k + m) ? DB : (i >= q && i < q + m) ? LB : "#999" }));
        // distance profile bars
        const pmax = Math.max(...P.filter(Number.isFinite), 1e-9), base = 320, ph = 140;
        let best = -1;
        for (let j = 0; j <= Math.min(k, L - 1); j++) if (Math.abs(j - q) > z && (best < 0 || P[j] < P[best])) best = j;
        mk("line", { x1: 30, x2: W - 10, y1: base, y2: base, stroke: "#999" });
        mk("text", { x: 4, y: 172, "font-size": 12, fill: "currentColor" }).textContent = `distance profile D_${q}`;
        for (let j = 0; j < L; j++) {
          const x = sx(j) - step * 0.35, w = step * 0.7;
          const trivial = Math.abs(j - q) <= z;
          if (trivial) mk("rect", { x, y: base - ph, width: w, height: ph, fill: "#999", "fill-opacity": 0.15 });
          if (j > k) continue;
          const h = P[j] / pmax * ph;
          const fill = j === best ? DB : trivial ? "#bbb" : RED;
          mk("rect", { x, y: base - h, width: w, height: h, fill, "fill-opacity": j === best ? 1 : 0.75 });
          mk("text", { x: sx(j), y: base - h - 4, "text-anchor": "middle", "font-size": 10, fill: "currentColor" }).textContent = P[j].toFixed(1);
          mk("text", { x: sx(j), y: base + 13, "text-anchor": "middle", "font-size": 10, fill: "currentColor" }).textContent = j;
        }
        info.textContent = k < 0 ? "press play or step" : (best >= 0 ? `slid to j=${k}; nearest neighbour so far: j=${best}, d=${P[best].toFixed(2)}` : `slid to j=${k}; only trivial matches so far`);
      }
      const stepFn = () => { const L = model.get("profile").length, k = model.get("k");
        if (k >= L - 1) { stopTimer(); return; } model.set("k", k + 1); };
      const stopTimer = () => { if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; model.save_changes(); } };
      const play = btn("▶ play", () => { if (timer) { stopTimer(); return; }
        if (model.get("k") >= model.get("profile").length - 1) model.set("k", -1);
        timer = setInterval(stepFn, 450); play.textContent = "⏸ pause"; });
      btn("step", () => { stepFn(); model.save_changes(); });
      btn("⏭ all", () => { stopTimer(); model.set("k", model.get("profile").length - 1); model.save_changes(); });
      btn("↺ reset", () => { stopTimer(); model.set("k", -1); model.save_changes(); });
      ctl.appendChild(info);
      for (const t of ["series", "m", "query", "profile", "excl", "k"]) model.on("change:" + t, draw);
      draw();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        series = traitlets.List(traitlets.Float()).tag(sync=True)
        m = traitlets.Int(4).tag(sync=True)
        query = traitlets.Int(1).tag(sync=True)
        profile = traitlets.List(traitlets.Float()).tag(sync=True)
        excl = traitlets.Int(1).tag(sync=True)
        k = traitlets.Int(-1).tag(sync=True)

    return (SlideAnimator,)


@app.cell
def slide(
    SlideAnimator,
    T_toy,
    distance_profile,
    dp_query,
    exclusion_zone,
    mo,
):
    toy_profile = distance_profile(T_toy, dp_query.value, 4, normalize=False)
    slide = mo.ui.anywidget(SlideAnimator(series=T_toy.tolist(), m=4, query=dp_query.value,
                                          profile=toy_profile.tolist(), excl=exclusion_zone(4)))
    slide
    return slide, toy_profile


@app.cell
def slide_readout(T_toy, dp_query, exclusion_zone, mo, np, slide, toy_profile):
    _q, _z = dp_query.value, exclusion_zone(4)
    _masked = np.where(np.abs(np.arange(len(toy_profile)) - _q) <= _z, np.inf, toy_profile)
    _nn = int(_masked.argmin())
    _done = slide.value["k"] >= len(toy_profile) - 1
    mo.vstack([
        mo.hstack([
            mo.stat(value=f"{len(toy_profile)}", label="distances computed", caption="one per window: n - m + 1"),
            mo.stat(value=(f"j = {_nn}" if _done else "?"), label="1-nearest neighbour", caption="finish the slide to reveal"),
            mo.stat(value=(f"{_masked[_nn]:.2f}" if _done else "?"), label="its distance", caption="smallest non-trivial bar"),
        ], justify="start"),
        mo.md(f"Query `{T_toy[_q:_q + 4].astype(int).tolist()}` ↔ nearest neighbour `{T_toy[_nn:_nn + 4].astype(int).tolist()}`" if _done else "_Press **⏭ all** (or let it play) to reveal the readout._"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Query $i=1$: before playing, guess which $j$ wins. Then play it. (It's the $\sqrt2$ pair from section 1.)
    2. Query $i=6$ (`[14, 15, 1, 2]`): its nearest neighbour is far away — this window is *unusual*. Keep it in mind for "discords".
    3. Count the work: one query costs $n-m+1$ distances, each of $m$ squared legs. Doing this for **every** query is the next section.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Stack the profiles: distance matrix → **matrix profile**

    Repeat section 2 for every query $i$ and stack the profiles: you get the $(n-m+1)\times(n-m+1)$ **distance matrix**.
    Simple, intuitive, exact… and hopeless at scale (see the thermostat calculator below).

    The 2016 insight (Keogh's lab at UC Riverside & Mueen at UNM): you don't need to *keep* the matrix. Keep only each row's minimum
    outside the trivial zone:

    $$P[i] = \min_{|j-i| > \lceil m/4\rceil} D_i[j] \qquad\text{(matrix profile)}, \qquad I[i] = \arg\min_j D_i[j] \qquad\text{(matrix profile index)}.$$

    Still $O(n^2)$ time, but only $O(n)$ memory — and in STUMPY, with sliding-window tricks, it's fast.
    Each cell below is one distance; grey = trivial zone; dark-blue outline = the row minimum that survives into $P$.
    """)
    return


@app.cell
def mx_row(T_toy, mo):
    mx_row = mo.ui.slider(0, len(T_toy) - 4, value=1, label="highlight row (query) i", show_value=True)
    mx_row
    return (mx_row,)


@app.cell
def matrix(
    COLORS,
    T_toy,
    alt,
    distance_matrix,
    exclusion_zone,
    mo,
    mx_row,
    naive_matrix_profile,
    np,
    pd,
):
    _m = 4
    _D = distance_matrix(T_toy, _m, normalize=False)
    toy_P, toy_I = naive_matrix_profile(T_toy, _m, normalize=False)
    _L, _z = len(_D), exclusion_zone(_m)
    _rows = [{"i": _i, "j": _j, "d": round(float(_D[_i, _j]), 1),
              "trivial": abs(_i - _j) <= _z, "nn": (_j == toy_I[_i]) and abs(_i - _j) > _z,
              "row": _i == mx_row.value} for _i in range(_L) for _j in range(_L)]
    _df = pd.DataFrame(_rows)
    _base = alt.Chart(_df).encode(x=alt.X("j:O", title="compared window j"), y=alt.Y("i:O", title="query i"))
    _heat = _base.mark_rect().encode(
        color=alt.condition("datum.trivial", alt.value("#dddddd"), alt.Color("d:Q", scale=alt.Scale(scheme="reds"), title="distance")),
        opacity=alt.condition("datum.row", alt.value(1.0), alt.value(0.55)),
        tooltip=["i", "j", "d"],
    )
    _nnbox = _base.transform_filter("datum.nn").mark_rect(fill=None, stroke=COLORS["nn"], strokeWidth=3)
    _txt = _base.mark_text(fontSize=10).encode(text="d:Q", color=alt.condition("datum.d > 9", alt.value("white"), alt.value("black")))
    _mat = (_heat + _nnbox + _txt).properties(width=380, height=380, title="distance matrix (raw Euclidean, m = 4)")
    _mp_df = pd.DataFrame({"i": np.arange(_L), "P": toy_P, "row": np.arange(_L) == mx_row.value})
    _mp = alt.Chart(_mp_df).mark_bar().encode(
        x=alt.X("i:O", title="i"), y=alt.Y("P:Q", title="matrix profile P[i]"),
        color=alt.condition("datum.row", alt.value(COLORS["query"]), alt.value(COLORS["dist"])), tooltip=["i", alt.Tooltip("P:Q", format=".2f")],
    ).properties(width=380, height=110, title="matrix profile = row minima")
    matrix_chart = alt.vconcat(_mat, _mp)
    _i = mx_row.value
    mo.hstack([matrix_chart, mo.vstack([
        mo.md(f"**Row {_i}:** query `{T_toy[_i:_i + 4].astype(int).tolist()}`"),
        mo.md(f"nearest neighbour **j = {toy_I[_i]}** `{T_toy[toy_I[_i]:toy_I[_i] + 4].astype(int).tolist()}`"),
        mo.stat(value=f"{toy_P[_i]:.2f}", label=f"P[{_i}]", caption="distance to 1-NN"),
        mo.md(f"**Global minimum** of P: {toy_P.min():.2f} at i = {int(toy_P.argmin())} and {int(toy_I[toy_P.argmin()])} → a **motif** pair.<br>"
              f"**Global maximum** of P: {toy_P.max():.2f} at i = {int(toy_P.argmax())} → the most **anomalous** window."),
    ])], justify="start", gap=2)
    return (toy_P,)


@app.cell
def stumpy_check_toy(T_toy, mo, np, stumpy, toy_P):
    _stumpy_toy = stumpy.stump(T_toy, 4, normalize=False)
    _agree = np.allclose(_stumpy_toy[:, 0].astype(float), toy_P)
    mo.md(f"""**Cross-check with STUMPY** — `stumpy.stump(T_toy, m=4, normalize=False)` gives
    `{np.round(_stumpy_toy[:, 0].astype(float), 2).tolist()}`
    → {"✅ identical to our brute-force row minima" if _agree else "❌ differs from the brute force"}. (`normalize=False` because the toy example uses plain distances; section 5 explains the default.)""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why nobody computed the full matrix (until now)

    Sean's thermostat: one reading every 3 s for 5 years ≈ **52 million points**. A naive full distance matrix needs ~$n^2$ numbers.
    Play with the inputs; throughput is set so that the defaults reproduce the talk's *≈ 4.4 years, ≈ 11 PB*.
    """)
    return


@app.cell
def calc_ui(mo):
    calc_years = mo.ui.slider(1, 10, value=5, label="years of recording", show_value=True)
    calc_rate = mo.ui.slider(1, 60, value=20, label="readings per minute", show_value=True)
    mo.hstack([calc_years, calc_rate], justify="start")
    return calc_rate, calc_years


@app.cell
def calc(calc_rate, calc_years, mo):
    def _human_bytes(b):
        for _u in ["B", "KB", "MB", "GB", "TB", "PB", "EB"]:
            if b < 1000:
                return f"{b:.1f} {_u}"
            b /= 1000
        return f"{b:.1f} ZB"

    _n = calc_years.value * 365 * 24 * 60 * calc_rate.value
    _pairs = float(_n) ** 2
    _secs = _pairs / 2e7          # ~2e7 distances / second (single naive core)
    mo.hstack([
        mo.stat(value=f"{_n / 1e6:.1f} M", label="data points n"),
        mo.stat(value=f"{_secs / 3.156e7:.1f} years", label="naive full matrix time", caption="n² distances"),
        mo.stat(value=_human_bytes(_pairs * 4), label="distance matrix memory", caption="n² float32"),
        mo.stat(value=_human_bytes(_n * 8), label="matrix profile memory", caption="n float64 — O(n)"),
    ], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Move the row slider to 1 and 8: they point at **each other** with the smallest distance $\sqrt2 \approx 1.41$. Mutual nearest neighbours at the global minimum = the **top motif**.
    2. Find the tallest bar of $P$. That window's *best* match is still far away: nothing like it exists elsewhere → a **discord**.
    3. In the calculator, halve the readings per minute: time & matrix memory drop by **4×** (quadratic), the profile only by 2× (linear).
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Motifs and discords for free

    Amir Mueen & Eamonn Keogh's claim (quoted in the talk): *"given a matrix profile, most time series data mining problems are trivial
    to solve in a few lines of code."* The two most important ones:

    * **Motif** (conserved behaviour) = the **global minimum** of $P$. Its index $I$ points at the matching window.
    * **Discord** (potential anomaly) = the **global maximum** of $P$: the window whose *best* match is still bad.

    Below is a 1,000-point series, the kind that "lands in your lap". A pattern was hidden in it **twice**, plus one glitch.
    **Predict first:** where does a repeated pattern start? Lock your guess, then STUMPY reveals the answer.
    """)
    return


@app.cell
def demo_data(mo, np):
    def make_demo_series(seed=7, n=1000, plant=(180, 690), glitch=460):
        _rng = np.random.default_rng(seed)
        _e = _rng.normal(0, 0.45, n)
        _T = np.zeros(n)
        for _k in range(1, n):                       # AR(1) background noise
            _T[_k] = 0.6 * _T[_k - 1] + _e[_k]
        _t = np.arange(48)
        _shape = 3.0 * np.exp(-((_t - 14) / 2.5) ** 2) - 1.8 * np.exp(-((_t - 22) / 3.0) ** 2) + 1.2 * np.sin(_t / 47 * 2 * np.pi)
        for _p in plant:
            _T[_p:_p + 48] += _shape + _rng.normal(0, 0.15, 48)
        _T[glitch] += 4.0
        _T[glitch + 1] -= 3.0
        return _T

    PLANTED = (180, 690)
    GLITCH = 460
    T_demo = make_demo_series(plant=PLANTED, glitch=GLITCH)
    mo.md(f"`T_demo`: n = {len(T_demo)} points (seeded, generated in-notebook).")
    return GLITCH, PLANTED, T_demo


@app.cell
def demo_guess(mo):
    demo_guess = mo.ui.slider(0, 950, value=500, label="your guess: a repeated pattern starts at i ≈", show_value=True)
    demo_lock = mo.ui.switch(value=False, label="🔒 lock in my guess & reveal")
    m_demo = mo.ui.slider(8, 120, value=48, step=4, label="window m", show_value=True)
    mo.vstack([mo.hstack([demo_guess, demo_lock], justify="start"), m_demo])
    return demo_guess, demo_lock, m_demo


@app.cell
def demo_mp(T_demo, demo_lock, m_demo, np, stumpy):
    demo_locked = demo_lock.value
    mp_demo = stumpy.stump(T_demo, m_demo.value)                 # <- the one line from the talk
    P_demo = mp_demo[:, 0].astype(float)
    I_demo = mp_demo[:, 1].astype(int)
    motif_idx = int(np.argmin(P_demo))
    motif_nn = int(I_demo[motif_idx])
    discord_idx = int(np.argmax(P_demo))
    return P_demo, demo_locked, discord_idx, motif_idx, motif_nn


@app.cell
def demo_chart(
    COLORS,
    P_demo,
    T_demo,
    alt,
    demo_guess,
    demo_locked,
    discord_idx,
    m_demo,
    motif_idx,
    motif_nn,
    np,
    pd,
):
    _m = m_demo.value
    _series = pd.DataFrame({"i": np.arange(len(T_demo)), "T": T_demo})
    _line = alt.Chart(_series).mark_line(color=COLORS["series"], strokeWidth=1).encode(x=alt.X("i:Q", title="time i"), y=alt.Y("T:Q", title="T"))
    _guess = alt.Chart(pd.DataFrame({"i": [demo_guess.value]})).mark_rule(color="black", strokeDash=[4, 3], strokeWidth=2).encode(x="i:Q")
    _layers = [_line, _guess]
    if demo_locked:
        _boxes = pd.DataFrame({
            "start": [motif_idx, motif_nn, discord_idx],
            "end": [motif_idx + _m, motif_nn + _m, discord_idx + _m],
            "kind": ["motif", "motif", "discord"],
        })
        _layers.insert(0, alt.Chart(_boxes).mark_rect(opacity=0.3).encode(
            x="start:Q", x2="end:Q",
            color=alt.Color("kind:N", scale=alt.Scale(domain=["motif", "discord"], range=[COLORS["motif"], COLORS["dist"]]), title=None)))
    _top = alt.layer(*_layers).properties(width=680, height=170, title="T_demo (dashed = your guess)")
    if demo_locked:
        _mp_df = pd.DataFrame({"i": np.arange(len(P_demo)), "P": P_demo})
        _marks = pd.DataFrame({"i": [motif_idx, motif_nn, discord_idx], "P": [P_demo[motif_idx], P_demo[motif_nn], P_demo[discord_idx]],
                               "kind": ["motif", "motif", "discord"]})
        _bottom = (alt.Chart(_mp_df).mark_line(color=COLORS["nn"], strokeWidth=1).encode(x=alt.X("i:Q", title="window start i"), y=alt.Y("P:Q", title="matrix profile P"))
                   + alt.Chart(_marks).mark_point(size=120, filled=True).encode(x="i:Q", y="P:Q",
                       color=alt.Color("kind:N", scale=alt.Scale(domain=["motif", "discord"], range=[COLORS["motif"], COLORS["dist"]]), legend=None))
                   ).properties(width=680, height=130, title=f"matrix profile, m = {_m}")
        demo_chart = alt.vconcat(_top, _bottom)
    else:
        demo_chart = _top
    demo_chart
    return


@app.cell
def demo_readout(
    COLORS,
    GLITCH,
    PLANTED,
    P_demo,
    T_demo,
    alt,
    demo_guess,
    demo_locked,
    discord_idx,
    m_demo,
    mo,
    motif_idx,
    motif_nn,
    naive_matrix_profile,
    np,
    pd,
    znorm,
):
    mo.stop(not demo_locked, mo.callout(mo.md("_Make a guess and flip **lock in** to reveal the matrix profile._"), kind="neutral"))
    _err = min(abs(demo_guess.value - _p) for _p in PLANTED)
    _P_naive, _ = naive_matrix_profile(T_demo, m_demo.value)
    _m = m_demo.value
    _zoom = pd.DataFrame({"k": np.tile(np.arange(_m), 2),
                          "z": np.concatenate([znorm(T_demo[motif_idx:motif_idx + _m]), znorm(T_demo[motif_nn:motif_nn + _m])]),
                          "which": [f"motif i={motif_idx}"] * _m + [f"its 1-NN j={motif_nn}"] * _m})
    _zchart = alt.Chart(_zoom).mark_line(strokeWidth=2.5).encode(
        x=alt.X("k:Q", title="offset within window"), y=alt.Y("z:Q", title="z-normalised value"),
        color=alt.Color("which:N", scale=alt.Scale(range=[COLORS["motif"], COLORS["nn"]]), title=None),
    ).properties(width=330, height=170, title="zoom: the motif pair overlaid")
    mo.hstack([
        mo.vstack([
            mo.stat(value=f"{motif_idx} ↔ {motif_nn}", label="top motif pair", caption=f"P = {P_demo[motif_idx]:.2f}; planted at {PLANTED} (both windows equally offset = same alignment)"),
            mo.stat(value=f"{discord_idx}", label="top discord", caption=f"P = {P_demo[discord_idx]:.2f}; glitch at {GLITCH}"),
            mo.stat(value=f"{_err}", label="your guess was off by", caption="🎯 nailed it" if _err < 15 else "points from nearest planted start"),
            mo.md(f"Naive numpy vs STUMPY max |ΔP| = `{np.abs(_P_naive - P_demo).max():.1e}` ✅"),
        ]),
        _zchart,
    ], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Shrink $m$ to 8–12: the motif shrinks to a fragment of the pattern, and the glitch dominates more.
    2. Grow $m$ to 100+: the windows now contain the pattern *plus* random noise, so the motif's $P$ value rises — the match gets blurrier.
    3. Notice the nearest neighbour can be in the **past or the future** — $I$ points both ways.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · The asterisk: z-normalisation

    From the Q&A: the matrix profile definition had a small asterisk — **z-normalised**. Before measuring, STUMPY rescales each window:

    $$\hat T_{i,m} = \frac{T_{i,m} - \mu_i}{\sigma_i} \qquad\Rightarrow\qquad d_z(i,j) = \sqrt{2m\,(1-\rho_{ij})},$$

    where $\rho_{ij}$ is the Pearson correlation of the two windows. So the distance cares about **shape**, not level or amplitude —
    no detrending or differencing needed. It's on by default; `stumpy.stump(T, m, normalize=False)` turns it off.

    Below, the pattern from the query window (light blue, i = 80) reappears later **3× taller on a rising trend**. Which mode finds it?
    """)
    return


@app.cell
def zn_data(mo, np):
    def make_scale_series(seed=3, n=500):
        _rng = np.random.default_rng(seed)
        _t = np.arange(48)
        _shape = 3.0 * np.exp(-((_t - 14) / 2.5) ** 2) - 1.8 * np.exp(-((_t - 22) / 3.0) ** 2) + 1.2 * np.sin(_t / 47 * 2 * np.pi)
        _T = 0.02 * np.arange(n) + _rng.normal(0, 0.3, n)
        _T[80:128] += _shape
        _T[330:378] += 3 * _shape
        return _T

    T_scale = make_scale_series()
    ZN_QUERY, ZN_M = 80, 48
    zn_switch = mo.ui.switch(value=True, label="z-normalise (STUMPY default)")
    zn_switch
    return T_scale, ZN_M, ZN_QUERY, zn_switch


@app.cell
def zn_chart(
    COLORS,
    T_scale,
    ZN_M,
    ZN_QUERY,
    alt,
    distance_profile,
    exclusion_zone,
    np,
    pd,
    zn_switch,
):
    _z = exclusion_zone(ZN_M)
    zn_profile = distance_profile(T_scale, ZN_QUERY, ZN_M, normalize=zn_switch.value)
    _masked = np.where(np.abs(np.arange(len(zn_profile)) - ZN_QUERY) <= _z, np.inf, zn_profile)
    zn_nn = int(_masked.argmin())
    _s = pd.DataFrame({"i": np.arange(len(T_scale)), "T": T_scale})
    _boxes = pd.DataFrame({"start": [ZN_QUERY, zn_nn], "end": [ZN_QUERY + ZN_M, zn_nn + ZN_M], "kind": ["query", "nearest neighbour"]})
    _cs = alt.Scale(domain=["query", "nearest neighbour"], range=[COLORS["query"], COLORS["nn"]])
    _top = (alt.Chart(_boxes).mark_rect(opacity=0.3).encode(x="start:Q", x2="end:Q", color=alt.Color("kind:N", scale=_cs, title=None))
            + alt.Chart(_s).mark_line(color=COLORS["series"]).encode(x=alt.X("i:Q", title="time"), y="T:Q")).properties(width=680, height=160)
    _p = pd.DataFrame({"j": np.arange(len(zn_profile)), "d": zn_profile})
    _bot = (alt.Chart(_p).mark_line(color=COLORS["dist"]).encode(x=alt.X("j:Q", title="window j"), y=alt.Y("d:Q", title="distance to query"))
            + alt.Chart(pd.DataFrame({"j": [zn_nn], "d": [zn_profile[zn_nn]]})).mark_point(size=150, filled=True, color=COLORS["nn"]).encode(x="j:Q", y="d:Q")
            ).properties(width=680, height=120, title=f"distance profile of i = {ZN_QUERY} ({'z-normalised' if zn_switch.value else 'raw'})")
    alt.vconcat(_top, _bot)
    return zn_nn, zn_profile


@app.cell
def zn_readout(T_scale, ZN_M, ZN_QUERY, mo, np, zn_nn, zn_profile):
    _q = T_scale[ZN_QUERY:ZN_QUERY + ZN_M]
    _c = T_scale[zn_nn:zn_nn + ZN_M]
    _rho = float(np.corrcoef(_q, _c)[0, 1])
    mo.hstack([
        mo.stat(value=f"j = {zn_nn}", label="nearest neighbour found", caption="✅ the scaled copy (330)" if abs(zn_nn - 330) < 10 else "❌ not the scaled copy at 330"),
        mo.stat(value=f"{zn_profile[zn_nn]:.2f}", label="distance d"),
        mo.stat(value=f"{np.sqrt(2 * ZN_M * (1 - _rho)):.2f}", label="√(2m(1−ρ))", caption=f"ρ = {_rho:.3f} — equals d only when z-normalised"),
    ], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Flip the switch off. Raw Euclidean distance picks a nearby flat stretch at a similar *level* instead of the true copy.
    2. Compare the last two numbers: with z-normalisation, $d = \sqrt{2m(1-\rho)}$ exactly. Raw mode breaks that identity.
    3. When is raw better? When *amplitude itself* matters (e.g. a temperature exceeding 40 °C is the event). Because STUMPY is fast, Sean's advice is: try both.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Which window size $m$? The pan matrix profile

    The most common question to the researchers: *how do I choose $m$?* Because computing a profile is cheap, compute it for **many** $m$,
    divide each by its maximum possible z-normalised distance $2\sqrt m$ so they're comparable, and stack them: the **pan matrix profile**.
    Paint every value below a threshold black. Dark "triangles" mark where conserved behaviour lives and over which range of window lengths.
    """)
    return


@app.cell
def pan(T_demo, mo, np, stumpy):
    pan_ms = list(range(8, 129, 8))
    pan = np.full((len(pan_ms), len(T_demo)), np.nan)
    for _r, _mm in enumerate(pan_ms):
        _P = stumpy.stump(T_demo, _mm)[:, 0].astype(float)
        pan[_r, :len(_P)] = _P / (2 * np.sqrt(_mm))
    pan_thr = mo.ui.slider(0.05, 0.6, value=0.2, step=0.01, label="black if normalised P <", show_value=True)
    pan_thr
    return pan, pan_ms, pan_thr


@app.cell
def pan_chart(alt, mo, np, pan, pan_ms, pan_thr, pd):
    _bin = 5
    _nb = pan.shape[1] // _bin
    _binned = np.min(np.nan_to_num(pan, nan=np.inf)[:, :_nb * _bin].reshape(len(pan_ms), _nb, _bin), axis=2)
    _df = pd.DataFrame([{"i": _b * _bin, "m": _mm, "P": float(_binned[_r, _b])}
                        for _r, _mm in enumerate(pan_ms) for _b in range(_nb) if np.isfinite(_binned[_r, _b])])
    _df["below"] = _df["P"] < pan_thr.value
    pan_chart = alt.Chart(_df).mark_rect().encode(
        x=alt.X("i:Q", bin=alt.Bin(step=_bin), title="window start i"),
        y=alt.Y("m:O", sort="descending", title="window length m"),
        color=alt.condition("datum.below", alt.value("black"), alt.Color("P:Q", scale=alt.Scale(scheme="reds"), title="P / 2√m")),
        tooltip=["i", "m", alt.Tooltip("P:Q", format=".3f")],
    ).properties(width="container", height=260, title="pan matrix profile of T_demo")
    _best = {_mm: float(np.nanmin(pan[_r])) for _r, _mm in enumerate(pan_ms)}
    mo.vstack([pan_chart,
               mo.md("Lowest normalised P per m: " + " · ".join(f"m={_k}: {_v:.2f}" for _k, _v in _best.items()))])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. At threshold 0.20 you should see **two dark triangles** around i ≈ 160–200 and 660–700 — the planted copies (180, 690) — spanning m ≈ 16–72.
    2. Raise it to 0.25: the triangles grow up to m ≈ 100, while m = 8 turns almost fully black — tiny windows match *something* everywhere, so very short $m$ is uninformative.
    3. The triangle peak tells you the longest window over which the behaviour is still conserved — a data-driven choice of $m$.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Sandbox: draw your own time series

    **Paint** a series with the mouse (click-drag across the canvas), or start from a preset. On release, STUMPY computes the matrix
    profile and marks the top **motif** pair (orange) and the top **discord** (red). Your job: *fool it*, or *make it find what you meant*.
    """)
    return


@app.cell
def series_painter_widget(anywidget, traitlets):
    class SeriesPainter(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const NS = "http://www.w3.org/2000/svg", W = 680, H = 230, PADL = 30, TOP = 10, PH = 210, YMIN = -3, YMAX = 3;
      const bar = document.createElement("div"); bar.style.marginBottom = "4px"; el.appendChild(bar);
      const svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none", cursor: "crosshair", border: "1px solid #9994", borderRadius: "6px" });
      el.appendChild(svg);
      const mk = (t, a) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); svg.appendChild(e); return e; };
      const toSvg = (e) => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const n = () => model.get("values").length;
      const sx = i => PADL + i * (W - PADL - 10) / (n() - 1);
      const sy = v => TOP + (YMAX - v) / (YMAX - YMIN) * PH;
      const g = () => { let s = 0; for (let k = 0; k < 6; k++) s += Math.random(); return s - 3; };   // ~N(0, 0.7)
      const bump = (k, c) => { const t = k - c; return (t >= 0 && t < 30) ? 2.2 * Math.exp(-((t - 9) / 2.5) ** 2) - 1.3 * Math.exp(-((t - 15) / 3) ** 2) : 0; };
      const presets = {
        "Two hidden bumps": N => Array.from({ length: N }, (_, k) => 0.35 * g() + bump(k, 30) + bump(k, 140)),
        "Sine + glitch": N => Array.from({ length: N }, (_, k) => Math.sin(k / 8) + 0.15 * g() + (k === 110 ? 2.5 : 0)),
        "Random walk": N => { let x = 0; return Array.from({ length: N }, () => (x = Math.max(-2.8, Math.min(2.8, x + 0.25 * g())))); },
        "Pure noise": N => Array.from({ length: N }, () => 0.8 * g()),
        "Flat": N => Array(N).fill(0),
      };
      for (const name in presets) {
        const b = document.createElement("button"); b.textContent = name;
        Object.assign(b.style, { marginRight: "6px", padding: "3px 10px", cursor: "pointer" });
        b.onclick = () => { model.set("values", presets[name](n()).map(v => Math.max(YMIN, Math.min(YMAX, v)))); model.save_changes(); };
        bar.appendChild(b);
      }
      function draw() {
        svg.innerHTML = "";
        const V = model.get("values"), hl = model.get("highlights");
        mk("line", { x1: PADL, x2: W - 10, y1: sy(0), y2: sy(0), stroke: "#9996" });
        for (const h of hl) mk("rect", { x: sx(h.start), y: TOP, width: sx(Math.min(h.start + h.len - 1, V.length - 1)) - sx(h.start), height: PH, fill: h.color, "fill-opacity": 0.22 });
        mk("polyline", { points: V.map((v, i) => `${sx(i)},${sy(v)}`).join(" "), fill: "none", stroke: "#888", "stroke-width": 2 });
        mk("text", { x: PADL + 4, y: TOP + 12, "font-size": 11, fill: "currentColor" }).textContent = "click-drag to paint";
      }
      let painting = false, last = -1;
      function paint(evt) {
        const p = toSvg(evt), N = n();
        const k = Math.round((p.x - PADL) / ((W - PADL - 10) / (N - 1)));
        if (k < 0 || k >= N) return;
        const v = Math.max(YMIN, Math.min(YMAX, YMAX - (p.y - TOP) / PH * (YMAX - YMIN)));
        const V = [...model.get("values")];
        if (last >= 0 && last !== k) {                       // interpolate so fast strokes leave no gaps
          const a = Math.min(last, k), b = Math.max(last, k), va = V[last];
          for (let j = a; j <= b; j++) V[j] = va + (v - va) * (j - last) / (k - last);
        }
        V[k] = v; last = k;
        model.set("values", V);
      }
      svg.addEventListener("pointerdown", e => { painting = true; last = -1; svg.setPointerCapture(e.pointerId); paint(e); });
      svg.addEventListener("pointermove", e => painting && paint(e));
      const stop = () => { if (painting) { painting = false; model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointerleave", stop);
      model.on("change:values", draw); model.on("change:highlights", draw);
      draw();
    }
    export default { render };
    """
        values = traitlets.List(traitlets.Float()).tag(sync=True)
        highlights = traitlets.List(traitlets.Dict()).tag(sync=True)

    return (SeriesPainter,)


@app.cell
def painter(SeriesPainter, mo, np):
    def _initial_paint(n=200, seed=11):
        _rng = np.random.default_rng(seed)
        _k = np.arange(n)
        def _bump(c):
            _t = _k - c
            return np.where((_t >= 0) & (_t < 30), 2.2 * np.exp(-((_t - 9) / 2.5) ** 2) - 1.3 * np.exp(-((_t - 15) / 3) ** 2), 0)
        return np.clip(0.35 * _rng.normal(0, 0.7, n) + _bump(30) + _bump(140), -3, 3)

    painter = mo.ui.anywidget(SeriesPainter(values=_initial_paint().tolist()))
    m_sand = mo.ui.slider(4, 60, value=24, label="window m", show_value=True)
    mo.vstack([m_sand, painter])
    return m_sand, painter


@app.cell
def sandbox(COLORS, alt, m_sand, mo, np, painter, pd, stumpy):
    T_sand = np.asarray(painter.value["values"], dtype=float)
    _m = m_sand.value
    if np.ptp(T_sand) < 1e-9:
        painter.widget.highlights = []
        _out = mo.callout(mo.md("A perfectly flat line: every z-normalised window is identical, so *everything* is a motif and nothing is a discord. Paint something!"), kind="warn")
    else:
        _mp = stumpy.stump(T_sand, _m)
        _P = _mp[:, 0].astype(float)
        _mi, _di = int(np.argmin(_P)), int(np.argmax(_P))
        _mj = int(_mp[_mi, 1])
        painter.widget.highlights = [
            {"start": _mi, "len": _m, "color": COLORS["motif"]},
            {"start": _mj, "len": _m, "color": COLORS["motif"]},
            {"start": _di, "len": _m, "color": COLORS["dist"]},
        ]
        _chart = alt.Chart(pd.DataFrame({"i": np.arange(len(_P)), "P": _P})).mark_area(
            color=COLORS["nn"], opacity=0.6, line=True).encode(x=alt.X("i:Q", title="window start i"), y=alt.Y("P:Q", title="matrix profile P")
        ).properties(width="container", height=130)
        _out = mo.vstack([
            mo.hstack([
                mo.stat(value=f"{_mi} ↔ {_mj}", label="top motif", caption=f"P = {_P[_mi]:.2f}"),
                mo.stat(value=f"{_di}", label="top discord", caption=f"P = {_P[_di]:.2f}"),
                mo.stat(value=f"{_P[_di] / max(_P[_mi], 1e-9):.1f}×", label="discord / motif ratio", caption="large = clear structure"),
                mo.stat(value=f"{2 * np.sqrt(_m):.1f}", label="max possible distance", caption="2√m (perfect anti-correlation)"),
            ], justify="start"),
            _chart,
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Challenges**
    1. **Pure noise**: the motif/discord ratio collapses toward ~1–2 — the matrix profile is telling you *"no conserved behaviour here"* (Sean's first question!).
    2. **Sine + glitch**: the discord lands on the spike at i = 110. Now paint a *second* identical spike elsewhere — the discord moves, because a repeated anomaly is no longer anomalous.
    3. **Two hidden bumps**: set $m$ to the bump width (~30) and check the motif pair lands at 30 ↔ 140. Then draw a third bump *upside-down*: z-normalisation can't call that a match.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Quiz: check yourself
    Answer all questions, then flip the switch to score.
    """)
    return


@app.cell
def quiz(mo):
    quiz_answers = {
        "q1": ("P[i] (the matrix profile at i) stores…", ["the distance from subsequence i to its 1-nearest neighbour", "the average distance from i to all subsequences", "the value of T at time i", "the index of the best match"], 0),
        "q2": ("The global *minimum* of the matrix profile marks…", ["a discord (anomaly)", "a motif (conserved pattern)", "the trivial match", "the best window size"], 1),
        "q3": ("The global *maximum* of the matrix profile marks…", ["a motif", "the start of the series", "a discord (potential anomaly)", "a perfect match"], 2),
        "q4": ("Why is there an exclusion zone around j = i?", ["to save memory", "to skip the trivial (self / overlapping) matches", "because of z-normalisation", "to speed up the FFT"], 1),
        "q5": ("Z-normalised distance is invariant to a window's…", ["shape", "length", "offset and amplitude", "position in time"], 2),
        "q6": ("Memory: full distance matrix vs matrix profile?", ["O(n) vs O(n²)", "O(n²) vs O(n)", "both O(n²)", "both O(n)"], 1),
        "q7": ("Raw distance between [1, 3, 2, 9] and [1, 2, 2, 10]?", ["0", "√2 ≈ 1.41", "2", "√67 ≈ 8.19"], 1),
    }
    quiz = mo.ui.dictionary({_k: mo.ui.radio(options=_v[1], label=f"**{_k[1:]}. {_v[0]}**") for _k, _v in quiz_answers.items()})
    quiz_reveal = mo.ui.switch(label="score my answers")
    mo.vstack([quiz, quiz_reveal])
    return quiz, quiz_answers, quiz_reveal


@app.cell
def quiz_score(mo, np, quiz, quiz_answers, quiz_reveal):
    mo.stop(not quiz_reveal.value, mo.md("_Flip **score my answers** when you're done._"))
    _rows, _score = [], 0
    for _k, (_q, _opts, _ans) in quiz_answers.items():
        _ok = quiz.value[_k] == _opts[_ans]
        _score += _ok
        _rows.append(f"| {_k[1:]} | {'✅' if _ok else '❌'} | {_opts[_ans]} |")
    # sanity: q7 answer is what our own helper computes
    assert np.isclose(np.sqrt(((np.array([1, 3, 2, 9]) - np.array([1, 2, 2, 10])) ** 2).sum()), np.sqrt(2))
    mo.vstack([
        mo.stat(value=f"{_score} / {len(quiz_answers)}", label="score", caption="🏆 matrix-profile master" if _score == len(quiz_answers) else "review the sections marked ❌"),
        mo.md("| # | | correct answer |\n|---|---|---|\n" + "\n".join(_rows)),
    ])
    return


@app.cell(hide_code=True)
def recap(mo):
    mo.md(r"""
    ## Recap

    | Concept | One-liner | Where you touched it |
    |---|---|---|
    | Subsequence $T_{i,m}$ | a length-$m$ window of the series | §1 draggable windows |
    | Euclidean distance | Pythagoras with more legs: $\sqrt{a^2+b^2+c^2+\dots}$ | §1 red legs |
    | Distance profile $D_i$ | slide the query over everything, record each distance | §2 slide animator |
    | Trivial match / exclusion zone | ignore $\lvert i-j\rvert \le \lceil m/4\rceil$ | §2–3 grey cells |
    | Matrix profile $P$, index $I$ | row minima of the distance matrix: $O(n)$ memory | §3 heatmap |
    | Motif | global min of $P$ → conserved behaviour | §4, §7 orange |
    | Discord | global max of $P$ → potential anomaly | §4, §7 red |
    | Z-normalisation | compare *shape*: $d=\sqrt{2m(1-\rho)}$ | §5 switch |
    | Pan matrix profile | many $m$ at once → pick the window size | §6 heatmap |
    | STUMPY | `stumpy.stump(T, m)` (CPU) · `stumpy.gpu_stump` · `stumpy.stumped` (Dask) | everywhere |

    **Beyond this notebook** (all in STUMPY): AB-joins between two series, streaming updates (`stumpy.stumpi`), fast pattern matching
    (`stumpy.match`), top-k nearest neighbours (`k=`), approximate profiles (`scrump`), and multi-dimensional motifs (`mstump`).
    Sean's call to action: put those three lines into your exploratory data analysis *before* you reach for a model.
    """)
    return


if __name__ == "__main__":
    app.run()

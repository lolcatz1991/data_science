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
    mo.md(r"""
    # Mean, Variance, Skewness & Kurtosis
    *An interactive companion to DeepLearning.ai's **Math for ML** lesson.*

    A distribution has a **shape**, and four numbers summarize most of it. Each one comes from a **moment**, which is the expected value of a power of the data:

    | # | Statistic | Formula | Tells you about… |
    |---|---|---|---|
    | 1 | **Mean** $\mu$ | $E[X]$ | where the centre of mass is |
    | 2 | **Variance** $\sigma^2$ | $E[(X-\mu)^2]$ | how spread out it is |
    | 3 | **Skewness** | $E\left[\left(\frac{X-\mu}{\sigma}\right)^3\right]$ | asymmetry (which tail is longer) |
    | 4 | **Kurtosis** | $E\left[\left(\frac{X-\mu}{\sigma}\right)^4\right]$ | how heavy the tails are |

    The odd powers (1, 3) keep the **sign** of a deviation, so they measure *direction*.
    The even powers (2, 4) discard the sign, so they measure *size*. Higher powers also give
    much more weight to points far from the mean.

    Drag, click and slide everything below. 👇
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The mean is a balance point

    Think of each data point as a 1 kg weight on a seesaw. The seesaw only balances when the
    pivot sits at the **mean**, because that is where the deviations to the left and the right cancel out:

    $$\sum_i (x_i-\mu) = 0$$

    **Drag the weights** (double-click the beam to add a weight, double-click a weight to remove it).
    """)
    return


@app.cell
def balance_beam_widget(anywidget, traitlets):
    class BalanceBeam(anywidget.AnyWidget):
        """Draggable weights on a seesaw; the pivot sits at the mean."""

        _esm = r"""
    function render({ model, el }) {
      const W = 680, H = 240, PAD = 30, XMIN = -10, XMAX = 10, BEAM_Y = 160;
      const sx = v => PAD + (v - XMIN) / (XMAX - XMIN) * (W - 2 * PAD);
      const inv = px => XMIN + (px - PAD) / (W - 2 * PAD) * (XMAX - XMIN);
      const NS = "http://www.w3.org/2000/svg";
      const mk = (tag, attrs, parent) => {
        const e = document.createElementNS(NS, tag);
        for (const k in attrs) e.setAttribute(k, attrs[k]);
        (parent || svg).appendChild(e);
        return e;
      };
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none", cursor: "default" });
      el.appendChild(svg);
      const info = document.createElement("div");
      Object.assign(info.style, { font: "14px system-ui, sans-serif", marginTop: "6px" });
      el.appendChild(info);

      let dragging = -1;
      const toData = (evt) => {
        const pt = svg.createSVGPoint();
        pt.x = evt.clientX; pt.y = evt.clientY;
        const p = pt.matrixTransform(svg.getScreenCTM().inverse());
        return Math.max(XMIN, Math.min(XMAX, Math.round(inv(p.x) * 2) / 2));
      };

      function draw() {
        const vals = model.get("values");
        const showDev = model.get("show_deviations");
        const n = vals.length;
        const mean = n ? vals.reduce((a, b) => a + b, 0) / n : 0;
        const vari = n ? vals.reduce((a, b) => a + (b - mean) ** 2, 0) / n : 0;
        svg.innerHTML = "";

        // axis ticks
        for (let t = XMIN; t <= XMAX; t += 2) {
          mk("line", { x1: sx(t), x2: sx(t), y1: BEAM_Y + 4, y2: BEAM_Y + 10, stroke: "#999" });
          const tx = mk("text", { x: sx(t), y: BEAM_Y + 24, "text-anchor": "middle", "font-size": 11, fill: "#666" });
          tx.textContent = t;
        }
        // beam
        const beam = mk("rect", { x: sx(XMIN), y: BEAM_Y - 3, width: sx(XMAX) - sx(XMIN), height: 6, rx: 3, fill: "#555" });
        beam.style.cursor = "copy";
        beam.addEventListener("dblclick", (e) => {
          model.set("values", [...model.get("values"), toData(e)]);
          model.save_changes();
        });
        // fulcrum
        const fx = sx(mean);
        mk("polygon", { points: `${fx},${BEAM_Y + 3} ${fx - 16},${BEAM_Y + 40} ${fx + 16},${BEAM_Y + 40}`, fill: "#e4572e" });
        const ml = mk("text", { x: fx, y: BEAM_Y + 56, "text-anchor": "middle", "font-size": 13, "font-weight": "bold", fill: "#e4572e" });
        ml.textContent = `μ = ${mean.toFixed(2)}`;

        // stack weights that share a position
        const order = vals.map((v, i) => [v, i]).sort((a, b) => a[0] - b[0]);
        const level = new Array(n).fill(0);
        const seen = {};
        for (const [v, i] of order) { const k = v.toFixed(1); level[i] = seen[k] || 0; seen[k] = level[i] + 1; }

        vals.forEach((v, i) => {
          const cy = BEAM_Y - 16 - level[i] * 24;
          if (showDev) {
            const col = v < mean ? "#2e86ab" : "#f18f01";
            mk("line", { x1: fx, x2: sx(v), y1: cy, y2: cy, stroke: col, "stroke-width": 2, "stroke-dasharray": "4 3", opacity: 0.8 });
            const lab = mk("text", { x: (fx + sx(v)) / 2, y: cy - 4, "text-anchor": "middle", "font-size": 10, fill: col });
            lab.textContent = (v - mean >= 0 ? "+" : "") + (v - mean).toFixed(1);
          }
          const c = mk("circle", { cx: sx(v), cy, r: 11, fill: i === dragging ? "#3b1f2b" : "#6c5b7b", stroke: "white", "stroke-width": 2 });
          c.style.cursor = "grab";
          c.addEventListener("pointerdown", (e) => { dragging = i; svg.setPointerCapture(e.pointerId); draw(); });
          c.addEventListener("dblclick", (e) => {
            e.stopPropagation();
            const cur = [...model.get("values")];
            if (cur.length > 2) { cur.splice(i, 1); model.set("values", cur); model.save_changes(); }
          });
        });

        const left = vals.filter(v => v < mean).reduce((a, v) => a + (mean - v), 0);
        const right = vals.filter(v => v > mean).reduce((a, v) => a + (v - mean), 0);
        info.innerHTML = `n = <b>${n}</b> &nbsp;·&nbsp; mean = <b>${mean.toFixed(2)}</b> &nbsp;·&nbsp; variance = <b>${vari.toFixed(2)}</b> &nbsp;·&nbsp; std = <b>${Math.sqrt(vari).toFixed(2)}</b>
          <br><span style="color:#2e86ab">left pull Σ|x−μ| = ${left.toFixed(2)}</span> &nbsp;⚖️&nbsp; <span style="color:#f18f01">right pull Σ|x−μ| = ${right.toFixed(2)}</span>`;
      }

      svg.addEventListener("pointermove", (e) => {
        if (dragging < 0) return;
        const cur = [...model.get("values")];
        const nv = toData(e);
        if (cur[dragging] !== nv) { cur[dragging] = nv; model.set("values", cur); }
      });
      const stop = () => { if (dragging >= 0) { dragging = -1; model.save_changes(); draw(); } };
      svg.addEventListener("pointerup", stop);
      svg.addEventListener("pointerleave", stop);

      model.on("change:values", draw);
      model.on("change:show_deviations", draw);
      draw();
    }
    export default { render };
    """
        values = traitlets.List(traitlets.Float()).tag(sync=True)
        show_deviations = traitlets.Bool(True).tag(sync=True)

    return (BalanceBeam,)


@app.cell
def beam(BalanceBeam, mo):
    beam = mo.ui.anywidget(BalanceBeam(values=[-6.0, -2.0, -1.0, 0.0, 1.5, 2.0, 7.5]))
    beam
    return (beam,)


@app.cell
def beam_stats(beam, np, pd):
    beam_x = np.array(beam.value["values"], dtype=float)
    beam_mu = beam_x.mean()
    beam_sigma = beam_x.std()
    beam_z = (beam_x - beam_mu) / beam_sigma if beam_sigma > 0 else np.zeros_like(beam_x)
    beam_df = pd.DataFrame(
        {
            "point": [f"x{i+1}={v:g}" for i, v in enumerate(beam_x)],
            "x": beam_x,
            "z": beam_z,
            "1 · deviation (x−μ)": beam_x - beam_mu,
            "2 · squared (x−μ)²": (beam_x - beam_mu) ** 2,
            "3 · z³": beam_z**3,
            "4 · z⁴": beam_z**4,
        }
    )
    return (beam_df,)


@app.cell
def moment_picker(mo):
    moment_picker = mo.ui.radio(
        options=["1 · deviation (x−μ)", "2 · squared (x−μ)²", "3 · z³", "4 · z⁴"],
        value="1 · deviation (x−μ)",
        inline=True,
        label="Contribution of each weight to moment:",
    )
    moment_picker
    return (moment_picker,)


@app.cell
def beam_moment_chart(alt, beam_df, mo, moment_picker, pd):
    _col = moment_picker.value
    _avg = beam_df[_col].mean()
    _bars = (
        alt.Chart(beam_df)
        .mark_bar()
        .encode(
            x=alt.X("point:N", sort=alt.SortField("x"), title=None),
            y=alt.Y(f"{_col}:Q", title=_col),
            color=alt.condition(alt.datum[_col] >= 0, alt.value("#f18f01"), alt.value("#2e86ab")),
            tooltip=["point", alt.Tooltip(_col, format=".3f")],
        )
    )
    _rule = alt.Chart(pd.DataFrame({"avg": [_avg]})).mark_rule(color="#e4572e", strokeWidth=2, strokeDash=[6, 3]).encode(y="avg:Q")
    _names = {
        "1 · deviation (x−μ)": "always 0, which is why μ is the balance point",
        "2 · squared (x−μ)²": "the variance σ²",
        "3 · z³": "the skewness",
        "4 · z⁴": "the kurtosis",
    }
    mo.vstack(
        [
            (_bars + _rule).properties(width="container", height=260, title="Per-point contribution (dashed line = average)"),
            mo.md(f"Average of this column = **{_avg:.3f}**: {_names[_col]}. Drag a weight far out on the beam above and watch the higher moments react."),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **Try this:**
    1. Drag a single weight far to the right. The mean moves only a little, but the **z³** bar for that weight grows much faster than the others. That one outlier makes the skewness positive.
    2. Put the weights in a mirror-symmetric pattern. Skewness drops to **0**.
    3. Pick **z⁴** and compare a few far outliers with many moderate points. The 4th power is dominated by the extremes.
    """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Variance: same mean, different risk

    The lecture uses a game. You can **win or lose \$1** on a coin flip, or **win or lose \$100**.
    Both games have an expected value of **0**, yet they feel completely different. Variance measures that difference:

    $$\operatorname{Var}(X) = E[(X-\mu)^2] = E[X^2] - \mu^2 \qquad \sigma = \sqrt{\operatorname{Var}(X)}$$

    We square the deviations so that negative ones cannot cancel positive ones. Use the sliders to compare two games.
    """)
    return


@app.cell
def game_controls(mo):
    stake_a = mo.ui.slider(1, 100, value=1, label="Game A: stake $", show_value=True)
    stake_b = mo.ui.slider(1, 100, value=20, label="Game B: stake $", show_value=True)
    p_win = mo.ui.slider(0.05, 0.95, step=0.05, value=0.5, label="P(win)", show_value=True)
    n_rounds = mo.ui.slider(10, 500, step=10, value=100, label="rounds played", show_value=True)
    resample = mo.ui.button(label="🎲 play again", value=0, on_click=lambda v: v + 1)
    mo.hstack([mo.vstack([stake_a, stake_b]), mo.vstack([p_win, n_rounds]), resample], justify="start", gap=2)
    return n_rounds, p_win, resample, stake_a, stake_b


@app.cell
def game_chart(alt, mo, n_rounds, np, p_win, pd, resample, stake_a, stake_b):
    resample.value
    _rng = np.random.default_rng()
    _p = p_win.value
    _rows = []
    for _name, _s in [("A", stake_a.value), ("B", stake_b.value)]:
        _outcomes = np.where(_rng.random(n_rounds.value) < _p, _s, -_s)
        _bank = np.cumsum(_outcomes)
        _mu = _s * _p - _s * (1 - _p)
        _var = _s**2 - _mu**2  # E[X^2] - mu^2
        for _i, _b in enumerate(_bank):
            _rows.append({"game": f"{_name} (stake ${_s}): μ={_mu:+.1f}, σ={np.sqrt(_var):.1f}", "round": _i + 1, "bankroll": _b})
    game_df = pd.DataFrame(_rows)

    _zoom = alt.selection_interval(bind="scales")
    _lines = (
        alt.Chart(game_df)
        .mark_line(strokeWidth=2)
        .encode(
            x="round:Q",
            y=alt.Y("bankroll:Q", title="cumulative winnings ($)"),
            color=alt.Color("game:N", scale=alt.Scale(range=["#2e86ab", "#f18f01"]), legend=alt.Legend(orient="top", title=None)),
            tooltip=["game", "round", "bankroll"],
        )
        .add_params(_zoom)
    )
    _zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#999").encode(y="y:Q")
    mo.vstack(
        [
            (_zero + _lines).properties(width="container", height=300, title="Same expected value, very different journeys (scroll to zoom)"),
            mo.md(
                "Both lines drift at the **same expected rate** of μ per round. The width of the band they wander in grows like "
                "$\\sigma\\sqrt{n}$. Move **P(win)** away from 0.5 and you will see that a larger stake scales the mean **and** the risk together."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Skewness: which tail is longer?

    Standardize first, $z = (x-\mu)/\sigma$, then average the **cubes**:

    $$\text{skew} = E[z^3]$$

    Cubing keeps the sign and inflates large values. A long right tail produces a few huge positive cubes,
    so the skew is **> 0**. A long left tail does the opposite. For a symmetric shape the cubes cancel, so the skew is **0**.

    A useful rule of thumb: in a right-skewed distribution, **mean > median > mode**. The mean is pulled toward the tail.
    """)
    return


@app.cell
def skew_controls(mo):
    skew_shape = mo.ui.slider(-12, 12, step=0.5, value=5, label="shape α (skew-normal)", show_value=True, full_width=True)
    skew_family = mo.ui.dropdown(
        options=["skew-normal", "log-normal (income-like)", "exponential (waiting times)", "beta (exam scores)"],
        value="skew-normal",
        label="Distribution",
    )
    mo.hstack([skew_family, skew_shape], widths=[1, 2])
    return skew_family, skew_shape


@app.cell
def skew_chart(alt, mo, np, pd, skew_family, skew_shape, stats):
    _rng = np.random.default_rng(42)
    _a = skew_shape.value
    if skew_family.value == "skew-normal":
        _dist = stats.skewnorm(_a)
    elif skew_family.value.startswith("log-normal"):
        _dist = stats.lognorm(s=0.1 + abs(_a) / 12)
    elif skew_family.value.startswith("exponential"):
        _dist = stats.expon()
    else:
        # alpha slider controls asymmetry: negative -> left skew (easy exam)
        _dist = stats.beta(2 + max(-_a, 0), 2 + max(_a, 0))
    skew_sample = _dist.rvs(size=20_000, random_state=_rng)
    if skew_family.value.startswith("log-normal") and _a < 0:
        skew_sample = -skew_sample  # mirror for a left tail

    _mu, _sd = skew_sample.mean(), skew_sample.std()
    _z = (skew_sample - _mu) / _sd
    skew_value = np.mean(_z**3)
    _median = np.median(skew_sample)
    _counts, _edges = np.histogram(skew_sample, bins=60)
    _mode = (_edges[np.argmax(_counts)] + _edges[np.argmax(_counts) + 1]) / 2
    _hist_df = pd.DataFrame({"lo": _edges[:-1], "hi": _edges[1:], "count": _counts})
    _hist_df["z_mid"] = ((_hist_df.lo + _hist_df.hi) / 2 - _mu) / _sd
    _hist_df["cube_contrib"] = _hist_df["count"] / _counts.sum() * _hist_df["z_mid"] ** 3

    _lines_df = pd.DataFrame({"stat": ["mean", "median", "mode"], "value": [_mu, _median, _mode]})
    _stat_colors = alt.Scale(domain=["mean", "median", "mode"], range=["#e4572e", "#29335c", "#76b041"])
    _hist = alt.Chart(_hist_df).mark_bar(color="#6c5b7b", opacity=0.75).encode(
        x=alt.X("lo:Q", title="x", bin="binned"), x2="hi:Q", y=alt.Y("count:Q")
    )
    _rules = alt.Chart(_lines_df).mark_rule(strokeWidth=3).encode(
        x="value:Q", color=alt.Color("stat:N", scale=_stat_colors, legend=alt.Legend(orient="top", title=None)),
        tooltip=["stat", alt.Tooltip("value:Q", format=".3f")],
    )
    _contrib = alt.Chart(_hist_df).mark_bar().encode(
        x=alt.X("lo:Q", title="x", bin="binned"), x2="hi:Q",
        y=alt.Y("cube_contrib:Q", title="share of E[z³]"),
        color=alt.condition(alt.datum.cube_contrib >= 0, alt.value("#f18f01"), alt.value("#2e86ab")),
        tooltip=[alt.Tooltip("z_mid:Q", format=".2f"), alt.Tooltip("cube_contrib:Q", format=".4f")],
    )
    _dir = "right ➡️" if skew_value > 0.1 else ("⬅️ left" if skew_value < -0.1 else "≈ symmetric")
    mo.vstack(
        [
            mo.hstack(
                [
                    mo.stat(f"{skew_value:+.3f}", label="sample skewness E[z³]", caption=_dir, bordered=True),
                    mo.stat(f"{float(_dist.stats(moments='s')) * (-1 if (skew_family.value.startswith('log') and _a < 0) else 1):+.3f}", label="theoretical skewness", bordered=True),
                    mo.stat(f"{_mu:.2f} / {_median:.2f} / {_mode:.2f}", label="mean / median / mode", bordered=True),
                ],
                justify="start",
            ),
            alt.vconcat(
                (_hist + _rules).properties(width=620, height=220, title="Histogram with mean, median and mode"),
                _contrib.properties(width=620, height=150, title="Where does the skewness come from? (bars sum to the skewness)"),
            ).resolve_scale(x="shared"),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Kurtosis: how heavy are the tails?

    $$\text{kurtosis} = E[z^4] \qquad \text{excess kurtosis} = E[z^4] - 3$$

    The normal distribution has kurtosis **3**, so people often report the *excess* above it.
    Every curve below has been standardized to **mean 0 and variance 1**, so it has the same centre and the same spread. Kurtosis is
    the remaining difference: how much probability sits far out in the tails.

    Click a legend entry to highlight a curve. Turn on the log scale to compare the tails directly.
    """)
    return


@app.cell
def kurt_controls(mo):
    kurt_dists = mo.ui.multiselect(
        options=["Uniform", "Normal", "Logistic", "Laplace", "Student t"],
        value=["Uniform", "Normal", "Laplace", "Student t"],
        label="Distributions",
    )
    t_df = mo.ui.slider(4.5, 30, step=0.5, value=5, label="Student t degrees of freedom ν", show_value=True)
    log_y = mo.ui.switch(label="log-scale y (see the tails)")
    mo.hstack([kurt_dists, t_df, log_y], justify="start", gap=2)
    return kurt_dists, log_y, t_df


@app.cell
def kurt_chart(alt, kurt_dists, log_y, mo, np, pd, stats, t_df):
    def _standardized(name):
        if name == "Uniform":
            return stats.uniform(loc=-np.sqrt(3), scale=2 * np.sqrt(3))
        if name == "Normal":
            return stats.norm()
        if name == "Logistic":
            return stats.logistic(scale=np.sqrt(3) / np.pi)
        if name == "Laplace":
            return stats.laplace(scale=1 / np.sqrt(2))
        _nu = t_df.value
        return stats.t(_nu, scale=np.sqrt((_nu - 2) / _nu))

    _x = np.linspace(-6, 6, 601)
    _rows, _summary = [], []
    for _n in kurt_dists.value:
        _d = _standardized(_n)
        _k = float(_d.stats(moments="k")) + 3  # scipy returns excess
        _label = f"{_n}{' ν=' + str(t_df.value) if _n == 'Student t' else ''}  (kurt={_k:.2f})"
        _pdf = _d.pdf(_x)
        _rows.append(pd.DataFrame({"x": _x, "density": np.maximum(_pdf, 1e-7), "z4_density": _x**4 * _pdf, "dist": _label}))
        _summary.append({"distribution": _label, "kurtosis": round(_k, 3), "excess": round(_k - 3, 3),
                         "P(|z|>3)": f"{2 * _d.sf(3):.4%}", "tail label": "platykurtic" if _k < 2.9 else ("leptokurtic" if _k > 3.1 else "mesokurtic")})
    kurt_df = pd.concat(_rows) if _rows else pd.DataFrame(columns=["x", "density", "z4_density", "dist"])
    kurt_summary = pd.DataFrame(_summary)

    _sel = alt.selection_point(fields=["dist"], bind="legend")
    _base = alt.Chart(kurt_df).encode(
        x=alt.X("x:Q", title="standardized value z"),
        color=alt.Color("dist:N", legend=alt.Legend(orient="top", columns=2, title=None)),
        opacity=alt.condition(_sel, alt.value(1), alt.value(0.12)),
        tooltip=["dist", alt.Tooltip("x:Q", format=".2f"), alt.Tooltip("density:Q", format=".5f")],
    )
    _dens = _base.mark_line(strokeWidth=2.5).encode(
        y=alt.Y("density:Q", scale=alt.Scale(type="log" if log_y.value else "linear"), title="density")
    ).add_params(_sel).properties(width=620, height=260, title="Same mean, same variance, different tails")
    _z4 = _base.mark_area(opacity=0.35, line=True).encode(
        y=alt.Y("z4_density:Q", title="z⁴ · f(z)", stack=None)
    ).properties(width=620, height=180, title="Integrand of kurtosis: the area under each curve is its kurtosis")
    mo.vstack([alt.vconcat(_dens, _z4), mo.ui.table(kurt_summary, selection=None)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(
        mo.md(
            r"""
    **Look at the lower panel.** The z⁴·f(z) curves are almost **zero near the centre**, because z⁴ is tiny when |z| < 1.
    All of the kurtosis comes from |z| > 1.5 or so, which means **kurtosis is mostly a measure of the tails**. Despite a common
    claim, it says little about how "peaked" the distribution is. Lower ν for the Student t and watch its tail mass, and its kurtosis, grow.
    In finance, this is why stock returns with "fat tails" produce more extreme crashes than a normal model predicts.
    """
        ),
        kind="warn",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Sandbox: draw your own distribution

    **Click and drag across the bars** to paint a probability distribution over the outcomes 0–20.
    Choose a preset to start from. The four moments are recomputed in Python every time you release the mouse.
    """)
    return


@app.cell
def histogram_painter_widget(anywidget, traitlets):
    class HistogramPainter(anywidget.AnyWidget):
        """Paint bar heights with the mouse; synced back to Python as `weights`."""

        _esm = r"""
    function render({ model, el }) {
      const W = 680, H = 260, PAD_L = 30, PAD_B = 26, TOP = 10;
      const NS = "http://www.w3.org/2000/svg";
      const wrap = document.createElement("div");
      wrap.style.font = "13px system-ui, sans-serif";
      el.appendChild(wrap);

      const bar = document.createElement("div");
      Object.assign(bar.style, { display: "flex", gap: "6px", flexWrap: "wrap", marginBottom: "6px" });
      wrap.appendChild(bar);
      const presets = {
        "Uniform": (k, n) => 1,
        "Bell": (k, n) => Math.exp(-0.5 * ((k - 10) / 3) ** 2),
        "Right-skewed": (k, n) => Math.pow(k + 0.5, 1.5) * Math.exp(-(k + 0.5) / 2.5),
        "Left-skewed": (k, n) => Math.pow(n - k - 0.5, 1.5) * Math.exp(-(n - k - 0.5) / 2.5),
        "Bimodal": (k, n) => Math.exp(-0.5 * ((k - 5) / 1.8) ** 2) + Math.exp(-0.5 * ((k - 15) / 1.8) ** 2),
        "Heavy tails": (k, n) => 1 / (1 + ((k - 10) / 1.2) ** 2),
        "Clear": (k, n) => 0,
      };
      for (const name in presets) {
        const b = document.createElement("button");
        b.textContent = name;
        Object.assign(b.style, { padding: "3px 10px", borderRadius: "12px", border: "1px solid #bbb", background: "#f6f4f8", cursor: "pointer" });
        b.onclick = () => {
          const n = model.get("weights").length;
          const w = Array.from({ length: n }, (_, k) => presets[name](k, n));
          const m = Math.max(...w, 1e-9);
          model.set("weights", w.map(v => name === "Clear" ? 0 : v / m));
          model.save_changes();
        };
        bar.appendChild(b);
      }

      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none", cursor: "crosshair", background: "#fcfbfd", borderRadius: "8px" });
      wrap.appendChild(svg);
      const info = document.createElement("div");
      info.style.marginTop = "4px";
      wrap.appendChild(info);

      const mk = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); svg.appendChild(e); return e; };
      let painting = false;

      function geom() {
        const n = model.get("weights").length;
        const bw = (W - PAD_L - 10) / n;
        return { n, bw, plotH: H - PAD_B - TOP };
      }
      function draw() {
        const w = model.get("weights");
        const { n, bw, plotH } = geom();
        const tot = w.reduce((a, b) => a + b, 0);
        let mu = 0, v = 0;
        if (tot > 0) {
          w.forEach((p, k) => mu += k * p / tot);
          w.forEach((p, k) => v += (k - mu) ** 2 * p / tot);
        }
        svg.innerHTML = "";
        w.forEach((p, k) => {
          const h = p * plotH;
          mk("rect", { x: PAD_L + k * bw + 1, y: TOP + plotH - h, width: bw - 2, height: Math.max(h, 0), rx: 2, fill: "#6c5b7b" });
          const t = mk("text", { x: PAD_L + k * bw + bw / 2, y: H - 8, "text-anchor": "middle", "font-size": 10, fill: "#666" });
          t.textContent = k;
        });
        mk("line", { x1: PAD_L, x2: W - 10, y1: TOP + plotH, y2: TOP + plotH, stroke: "#999" });
        if (tot > 0) {
          const xm = PAD_L + (mu + 0.5) * bw, sd = Math.sqrt(v);
          mk("rect", { x: PAD_L + (mu - sd + 0.5) * bw, y: TOP, width: 2 * sd * bw, height: plotH, fill: "#e4572e", opacity: 0.08 });
          mk("line", { x1: xm, x2: xm, y1: TOP, y2: TOP + plotH, stroke: "#e4572e", "stroke-width": 2.5, "stroke-dasharray": "6 3" });
          const lab = mk("text", { x: xm + 4, y: TOP + 12, "font-size": 12, fill: "#e4572e", "font-weight": "bold" });
          lab.textContent = `μ=${mu.toFixed(2)}  (shaded: μ±σ)`;
        }
        info.textContent = tot > 0 ? "Release the mouse to sync the moments to Python." : "Paint some bars!";
      }
      function paint(evt) {
        const pt = svg.createSVGPoint(); pt.x = evt.clientX; pt.y = evt.clientY;
        const p = pt.matrixTransform(svg.getScreenCTM().inverse());
        const { n, bw, plotH } = geom();
        const k = Math.floor((p.x - PAD_L) / bw);
        if (k < 0 || k >= n) return;
        const val = Math.max(0, Math.min(1, (TOP + plotH - p.y) / plotH));
        const w = [...model.get("weights")];
        w[k] = val;
        model.set("weights", w);
      }
      svg.addEventListener("pointerdown", (e) => { painting = true; svg.setPointerCapture(e.pointerId); paint(e); });
      svg.addEventListener("pointermove", (e) => { if (painting) paint(e); });
      const stop = () => { if (painting) { painting = false; model.save_changes(); } };
      svg.addEventListener("pointerup", stop);
      svg.addEventListener("pointerleave", stop);
      model.on("change:weights", draw);
      draw();
    }
    export default { render };
    """
        weights = traitlets.List(traitlets.Float()).tag(sync=True)

    return (HistogramPainter,)


@app.cell
def painter(HistogramPainter, mo, np):
    _k = np.arange(21)
    painter = mo.ui.anywidget(HistogramPainter(weights=list(np.exp(-0.5 * ((_k - 7) / 3) ** 2) * (1 + 0.6 * (_k > 7)))))
    painter
    return (painter,)


@app.cell
def moments_fn(np):
    def moments(xs, ps):
        """Mean, variance, skewness, kurtosis of a discrete distribution P(X=xs)=ps."""
        ps = np.asarray(ps, float)
        ps = ps / ps.sum()
        mu = np.sum(xs * ps)
        var = np.sum((xs - mu) ** 2 * ps)
        sd = np.sqrt(var)
        z = (xs - mu) / sd if sd > 0 else np.zeros_like(xs, dtype=float)
        return {"mean": mu, "variance": var, "skewness": np.sum(z**3 * ps), "kurtosis": np.sum(z**4 * ps)}

    return (moments,)


@app.cell
def painter_stats(mo, moments, np, painter):
    _w = np.array(painter.value["weights"], dtype=float)
    if _w.sum() <= 0:
        _out = mo.callout(mo.md("Paint at least one bar to compute moments."), kind="neutral")
    else:
        _m = moments(np.arange(len(_w)), _w)
        _skew_txt = "right tail ➡️" if _m["skewness"] > 0.2 else ("⬅️ left tail" if _m["skewness"] < -0.2 else "roughly symmetric")
        _kurt_txt = "heavier tails than normal" if _m["kurtosis"] > 3.3 else ("lighter tails than normal" if _m["kurtosis"] < 2.7 else "normal-like tails")
        _out = mo.hstack(
            [
                mo.stat(f"{_m['mean']:.2f}", label="Mean  E[X]", bordered=True),
                mo.stat(f"{_m['variance']:.2f}", label="Variance  E[(X−μ)²]", caption=f"σ = {np.sqrt(_m['variance']):.2f}", bordered=True),
                mo.stat(f"{_m['skewness']:+.3f}", label="Skewness  E[z³]", caption=_skew_txt, bordered=True),
                mo.stat(f"{_m['kurtosis']:.3f}", label="Kurtosis  E[z⁴]", caption=f"excess {_m['kurtosis'] - 3:+.2f} · {_kurt_txt}", bordered=True),
            ],
            widths="equal",
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "🎯 Challenges (click to open)": mo.md(
                r"""
    1. Draw a distribution with **skewness ≈ 0** that is *not* symmetric. (Hint: it is possible!)
    2. Make the **kurtosis as small as you can**. Which shape wins? (Two equal spikes at the ends, which gives kurtosis 1, the theoretical minimum.)
    3. Keep the variance roughly constant and push the kurtosis above 6.
    4. Use **Bimodal**. Why is its kurtosis *lower* than the bell curve's, even though it has two peaks?
    """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Quiz: read the shape

    Each histogram below has a hidden skewness and kurtosis. Make your guesses, then reveal the answers.
    """)
    return


@app.cell
def quiz_plot(alt, np, pd):
    _rng = np.random.default_rng(7)
    quiz_samples = {
        "A": _rng.exponential(1, 4000),
        "B": _rng.uniform(0, 1, 4000),
        "C": -_rng.gamma(2, 1, 4000),
        "D": _rng.standard_t(4, 4000),
    }
    _frames = []
    for _key, _s in quiz_samples.items():
        _c, _e = np.histogram(_s, bins=40)
        _frames.append(pd.DataFrame({"lo": _e[:-1], "hi": _e[1:], "count": _c, "mystery": _key}))
    _qdf = pd.concat(_frames)
    alt.Chart(_qdf).mark_bar(color="#6c5b7b").encode(
        x=alt.X("lo:Q", bin="binned", title=None), x2="hi:Q", y=alt.Y("count:Q", title=None)
    ).properties(width=150, height=120).facet(facet="mystery:N", columns=4).resolve_scale(x="independent", y="independent")
    return (quiz_samples,)


@app.cell
def quiz_inputs(mo):
    quiz_guesses = mo.ui.dictionary(
        {
            _k: mo.ui.dictionary(
                {
                    "skew": mo.ui.dropdown(["negative", "≈ 0", "positive"], label=f"{_k} skew"),
                    "tails": mo.ui.dropdown(["light (<3)", "normal (≈3)", "heavy (>3)"], label="tails"),
                }
            )
            for _k in ["A", "B", "C", "D"]
        }
    )
    reveal = mo.ui.switch(label="Reveal answers")
    mo.vstack([mo.hstack([quiz_guesses[_k] for _k in "ABCD"], widths="equal"), reveal])
    return quiz_guesses, reveal


@app.cell
def quiz_result(mo, moments, np, pd, quiz_guesses, quiz_samples, reveal):
    def _truth(s):
        m = moments(s, np.ones_like(s))
        sk = "positive" if m["skewness"] > 0.3 else ("negative" if m["skewness"] < -0.3 else "≈ 0")
        tl = "heavy (>3)" if m["kurtosis"] > 3.5 else ("light (<3)" if m["kurtosis"] < 2.6 else "normal (≈3)")
        return m, sk, tl

    _rows = []
    _score = 0
    for _k, _s in quiz_samples.items():
        _m, _sk, _tl = _truth(_s)
        _g = quiz_guesses.value[_k]
        _ok = (_g["skew"] == _sk) + (_g["tails"] == _tl)
        _score += _ok
        _rows.append({"mystery": _k, "your skew": _g["skew"], "your tails": _g["tails"],
                      "skewness": round(_m["skewness"], 2) if reveal.value else "?",
                      "kurtosis": round(_m["kurtosis"], 2) if reveal.value else "?",
                      "✓": ("✅" * _ok + "❌" * (2 - _ok)) if reveal.value else ""})
    mo.vstack([
        mo.ui.table(pd.DataFrame(_rows), selection=None),
        mo.md(f"### Score: {_score} / 8 {'🏆' if _score == 8 else ''}" if reveal.value else "_Answers hidden._"),
        mo.md("_Answers: A = exponential, B = uniform, C = mirrored gamma, D = Student t with ν=4 (its true kurtosis is infinite!)._") if reveal.value else mo.md(""),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap

    | Moment | Power | Sign kept? | Unit | Normal value |
    |---|---|---|---|---|
    | Mean | 1 | ✅ | same as X | μ |
    | Variance | 2 | ❌ | X² | σ² |
    | Skewness | 3 | ✅ | unitless | 0 |
    | Kurtosis | 4 | ❌ | unitless | 3 (excess 0) |

    * **Odd** moments measure *direction*. **Even** moments measure *magnitude*.
    * Skewness and kurtosis are computed after standardizing, so they don't depend on units or scale and only describe **shape**.
    * Higher powers are more sensitive to outliers. This makes them informative about tails, and also noisy to estimate from small samples.
    """)
    return


if __name__ == "__main__":
    app.run()

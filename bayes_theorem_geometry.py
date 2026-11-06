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
    # Bayes' theorem, the geometry of changing beliefs
    *An interactive companion to 3Blue1Brown's video. Evidence should not **determine** your beliefs; it should **update** them.*
    """),
        mo.Html('<iframe width="100%" style="aspect-ratio:16/9;max-width:720px" src="https://www.youtube.com/embed/HZGCoVF3YvM" title="Bayes theorem" frameborder="0" allowfullscreen></iframe>'),
        mo.md(r"""
    **Colour language used throughout**

    | colour | meaning |
    |---|---|
    | <span style="color:#3b82f6">■ blue</span> | the hypothesis $H$ (Steve is a **librarian**) |
    | <span style="color:#16a34a">■ green</span> | not the hypothesis $\neg H$ (Steve is a **farmer**) |
    | <span style="color:#f59e0b">■ amber</span> | people / area that **fit the evidence** $E$ ("meek and tidy soul") |
    | <span style="color:#e4572e">■ red</span> | the **posterior** $P(H\mid E)$, the number we want |

    *(The neighbouring `bayesian_statistics.py` notebook covers Bayesian regression. This one stays with the basic picture: one hypothesis, one piece of evidence, one square.)*
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0 · Meet Steve

    > *Steve is very shy and withdrawn, invariably helpful but with very little interest in people or the world of reality.
    > A meek and tidy soul, he has a need for order and structure, and a passion for detail.*

    This description comes from a study by Kahneman and Tversky. **Commit to a guess before you scroll on.**
    """)
    return


@app.cell
def steve_guess(mo):
    steve_guess = mo.ui.slider(0, 100, step=1, value=50, label="How likely is it (in %) that Steve is a **librarian** rather than a farmer?", show_value=True, full_width=True)
    steve_guess
    return (steve_guess,)


@app.cell(hide_code=True)
def _(mo, steve_guess):
    mo.callout(mo.md(
        f"You said **{steve_guess.value}%**. Most people say the librarian is more likely, because the description matches the stereotype. "
        "Keep your number in mind. Section 1 works out the answer under the video's assumptions."
    ), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Think with a representative sample

    Kahneman and Tversky's point is not about whether your stereotypes are right. It is that **almost nobody thinks about how many farmers there are compared with librarians**.
    In their paper that ratio was about **20 : 1**.

    The video's recipe:
    1. Picture a representative sample: **10 librarians** and **200 farmers**.
    2. Your gut says **40%** of librarians fit the description and **10%** of farmers do.
    3. That gives **4** librarians and **20** farmers who fit, so $P(\text{librarian}\mid\text{description}) = \frac{4}{4+20} = 16.7\%$.

    Use the sliders, then press **👁 hear the description** inside the widget. Evidence **restricts the space of possibilities**: everyone who doesn't fit fades out.
    """)
    return


@app.cell
def people_grid_class(anywidget, traitlets):
    class PeopleGrid(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const NS = "http://www.w3.org/2000/svg";
      const ctl = document.createElement("div");
      Object.assign(ctl.style, { display: "flex", gap: "8px", alignItems: "center", marginBottom: "6px", flexWrap: "wrap" });
      el.appendChild(ctl);
      const btn = document.createElement("button");
      Object.assign(btn.style, { padding: "4px 10px", borderRadius: "6px", border: "1px solid #999", cursor: "pointer" });
      ctl.appendChild(btn);
      const info = document.createElement("span"); ctl.appendChild(info);
      const svg = document.createElementNS(NS, "svg");
      el.appendChild(svg);
      Object.assign(svg.style, { width: "100%", maxWidth: "720px", userSelect: "none" });
      const mk = (tag, a, p = svg) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
      btn.onclick = () => { model.set("show_evidence", !model.get("show_evidence")); model.save_changes(); };
      function draw() {
        const nL = model.get("n_lib"), nF = model.get("n_farm");
        const kL = Math.round(nL * model.get("p_e_lib")), kF = Math.round(nF * model.get("p_e_farm"));
        const ev = model.get("show_evidence");
        btn.textContent = ev ? "↺ forget the description" : "👁 hear the description";
        const cols = 30, r = 9, gap = 22, PAD = 14;
        const rowsL = Math.ceil(nL / 5) , colsL = Math.min(5, nL);
        const rowsF = Math.ceil(nF / cols);
        const xF0 = PAD + 5 * gap + 40;
        const W = xF0 + cols * gap + PAD, H = PAD * 2 + Math.max(rowsL, rowsF) * gap + 40;
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
        svg.innerHTML = "";
        mk("text", { x: PAD, y: 18, "font-size": 14, fill: "#3b82f6", "font-weight": "bold" }).textContent = `Librarians (${nL})`;
        mk("text", { x: xF0, y: 18, "font-size": 14, fill: "#16a34a", "font-weight": "bold" }).textContent = `Farmers (${nF})`;
        const person = (cx, cy, color, fits) => {
          const op = ev && !fits ? 0.12 : 1;
          const g = mk("g", { opacity: op });
          if (fits) mk("circle", { cx, cy, r: r + 2.5, fill: "none", stroke: "#f59e0b", "stroke-width": 3 }, g);
          mk("circle", { cx, cy, r: r - 1, fill: color }, g);
          g.style.transition = "opacity 0.6s";
        };
        for (let i = 0; i < nL; i++) person(PAD + r + (i % 5) * gap, 40 + Math.floor(i / 5) * gap, "#3b82f6", i < kL);
        for (let i = 0; i < nF; i++) person(xF0 + r + (i % cols) * gap, 40 + Math.floor(i / cols) * gap, "#16a34a", i < kF);
        const post = kL + kF > 0 ? kL / (kL + kF) : NaN;
        info.innerHTML = ev
          ? `Fit the description: <b style="color:#3b82f6">${kL}</b> librarians + <b style="color:#16a34a">${kF}</b> farmers. Among them, P(librarian) = ${kL}/${kL + kF} = <b style="color:#e4572e">${(100 * post).toFixed(1)}%</b>`
          : `Before the evidence: P(librarian) = ${nL}/${nL + nF} = <b>${(100 * nL / (nL + nF)).toFixed(1)}%</b>`;
      }
      ["n_lib", "n_farm", "p_e_lib", "p_e_farm", "show_evidence"].forEach(t => model.on("change:" + t, draw));
      draw();
    }
    export default { render };
    """
        n_lib = traitlets.Int(10).tag(sync=True)
        n_farm = traitlets.Int(200).tag(sync=True)
        p_e_lib = traitlets.Float(0.4).tag(sync=True)
        p_e_farm = traitlets.Float(0.1).tag(sync=True)
        show_evidence = traitlets.Bool(False).tag(sync=True)

    return (PeopleGrid,)


@app.cell
def sample_controls(mo):
    n_lib_slider = mo.ui.slider(1, 50, value=10, label="Librarians in the sample", show_value=True)
    n_farm_slider = mo.ui.slider(10, 300, step=10, value=200, label="Farmers in the sample", show_value=True)
    p_lib_slider = mo.ui.slider(0, 1, step=0.05, value=0.4, label="% of librarians who fit the description", show_value=True)
    p_farm_slider = mo.ui.slider(0, 1, step=0.05, value=0.1, label="% of farmers who fit the description", show_value=True)
    mo.hstack([mo.vstack([n_lib_slider, n_farm_slider]), mo.vstack([p_lib_slider, p_farm_slider])], justify="start", gap=2)
    return n_farm_slider, n_lib_slider, p_farm_slider, p_lib_slider


@app.cell
def people(
    PeopleGrid,
    mo,
    n_farm_slider,
    n_lib_slider,
    p_farm_slider,
    p_lib_slider,
):
    people = mo.ui.anywidget(PeopleGrid(
        n_lib=n_lib_slider.value, n_farm=n_farm_slider.value,
        p_e_lib=p_lib_slider.value, p_e_farm=p_farm_slider.value,
    ))
    people
    return (people,)


@app.cell
def sample_readout(mo, people, steve_guess):
    _nL, _nF = people.value["n_lib"], people.value["n_farm"]
    _kL, _kF = round(_nL * people.value["p_e_lib"]), round(_nF * people.value["p_e_farm"])
    sample_posterior = _kL / (_kL + _kF) if _kL + _kF else float("nan")
    mo.hstack([
        mo.stat(f"{_nL / (_nL + _nF):.1%}", label="Before: P(librarian)", caption=f"{_nL} of {_nL + _nF}"),
        mo.stat(f"{_kL} + {_kF}", label="People who fit", caption="librarians + farmers"),
        mo.stat(f"{sample_posterior:.1%}", label="After: P(librarian | fits)", caption=f"{_kL} of {_kL + _kF}"),
        mo.stat(f"{steve_guess.value}%", label="Your gut guess", caption=f"off by {abs(steve_guess.value - 100 * sample_posterior):.0f} points"),
    ], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - Keep the defaults (10 vs 200, 40% vs 10%). A librarian is **4× as likely** to fit the description, yet the answer is only **16.7%**. There are just too many farmers.
    - Push *% of librarians who fit* to **100%**. Can the librarian answer get past 50% while farmers outnumber librarians 20 : 1 and 10% of them fit?
    - Set both percentages equal. The "after" number matches the "before" number: **evidence that fits everyone equally changes nothing.**
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · From counts to a formula

    Generalize: a **hypothesis** $H$ ("Steve is a librarian") and some **evidence** $E$ (the description). We want $P(H\mid E)$. The bar means *given that*: restrict your view to the cases where the evidence holds.

    | name | symbol | Steve example |
    |---|---|---|
    | **prior** | $P(H)$ | $1/21$, the share of librarians |
    | **likelihood** | $P(E\mid H)$ | 40% of librarians fit |
    | (other likelihood) | $P(E\mid \neg H)$ | 10% of farmers fit |
    | **posterior** | $P(H\mid E)$ | what we want |

    Counting with a sample of size $N$ (the video uses 210), then cancelling $N$, gives:

    $$
    P(H\mid E) \;=\; \frac{N\,P(H)\,P(E\mid H)}{N\,P(H)\,P(E\mid H) \;+\; N\,P(\neg H)\,P(E\mid \neg H)}
    \;=\; \frac{P(H)\,P(E\mid H)}{P(E)}
    $$

    The sample size $N$ was an arbitrary choice, so it has to cancel. Pick any $N$ below and watch the counts change while the posterior stays the same.
    """)
    return


@app.cell
def formula_N(mo):
    formula_N = mo.ui.slider(21, 2100, step=21, value=210, label="Sample size N (arbitrary)", show_value=True, full_width=True)
    formula_N
    return (formula_N,)


@app.cell
def formula_readout(
    formula_N,
    mo,
    n_farm_slider,
    n_lib_slider,
    p_farm_slider,
    p_lib_slider,
):
    _pH = n_lib_slider.value / (n_lib_slider.value + n_farm_slider.value)
    _a, _b = p_lib_slider.value, p_farm_slider.value
    _N = formula_N.value
    _num = _N * _pH * _a
    _rest = _N * (1 - _pH) * _b
    mo.md(rf"""
    Using the Section 1 sliders: $P(H) = {_pH:.4f}$, $P(E\mid H) = {_a:.2f}$, $P(E\mid\neg H) = {_b:.2f}$.

    $$
    P(H\mid E) = \frac{{\color{{#3b82f6}}{{{_N}}}\cdot{_pH:.4f}\cdot{_a:.2f}}}{{\color{{#3b82f6}}{{{_N}}}\cdot{_pH:.4f}\cdot{_a:.2f} + \color{{#16a34a}}{{{_N}}}\cdot{1-_pH:.4f}\cdot{_b:.2f}}}
    = \frac{{{_num:.1f}}}{{{_num:.1f} + {_rest:.1f}}}
    = \color{{#e4572e}}{{\mathbf{{{(_num / (_num + _rest) if _num + _rest else float("nan")):.4f}}}}}
    $$

    $P(E)$, the total share of people who fit the description, is $({_num:.1f} + {_rest:.1f})/{_N} = {(_num + _rest) / _N:.4f}$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - Drag $N$ from 21 to 2100. The counts grow, the red posterior **doesn't move**.
    - At $N = 210$ with the defaults you should read exactly $4/(4+20)$.
    - The denominator $P(E)$ is almost never known directly. You **compute** it by splitting into "$H$ true" and "$H$ false". That split is the second term.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · The picture to keep in your head: probability as area

    The video's advice: **don't memorize the formula, draw this diagram.** Replace the 210 people with a **1 × 1 square** of all possibilities. Probability is **area**.

    - The **left strip**, width $P(H)$, holds the librarians. The right strip, width $P(\neg H)$, holds the farmers.
    - In each strip, the **amber bottom part** is where the evidence holds. Its height is the likelihood: $P(E\mid H)$ on the left, $P(E\mid\neg H)$ on the right.
    - The left amber rectangle has area $P(H)\,P(E\mid H) = P(H \cap E)$.

    Seeing the evidence **throws away everything that isn't amber**. The posterior is the share of the remaining amber area that is blue:

    $$P(H\mid E) = \frac{\text{left amber area}}{\text{total amber area}}$$

    **Drag** the vertical divider (prior) and the tops of the two amber bars (likelihoods). Toggle **restrict to evidence** to see the uneven, "wonky" shape that is left.
    """)
    return


@app.cell
def bayes_square_class(anywidget, traitlets):
    class BayesSquare(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const NS = "http://www.w3.org/2000/svg";
      const S = 360, PAD = 40, BARX = PAD + S + 50, W = BARX + 120, H = S + 2 * PAD + 10;
      const ctl = document.createElement("div");
      Object.assign(ctl.style, { display: "flex", gap: "10px", alignItems: "center", marginBottom: "4px" });
      el.appendChild(ctl);
      const tbtn = document.createElement("button");
      Object.assign(tbtn.style, { padding: "4px 10px", borderRadius: "6px", border: "1px solid #999", cursor: "pointer" });
      ctl.appendChild(tbtn);
      tbtn.onclick = () => { model.set("restrict", !model.get("restrict")); model.save_changes(); };
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none" });
      el.appendChild(svg);
      const mk = (tag, a, p = svg) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
      const toSvg = (evt) => { const p = svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const X = v => PAD + v * S, Y = v => PAD + (1 - v) * S;
      const clamp = v => Math.max(0.005, Math.min(0.995, v));
      let drag = null;
      function draw() {
        const p = model.get("prior"), a = model.get("lik_h"), b = model.get("lik_nh"), r = model.get("restrict");
        tbtn.textContent = r ? "↺ show the whole square" : "👁 restrict to evidence";
        svg.innerHTML = "";
        const fade = r ? 0.08 : 1;
        // non-evidence parts
        mk("rect", { x: X(0), y: Y(1), width: p * S, height: (1 - a) * S, fill: "#3b82f6", opacity: 0.25 * fade + (r ? 0 : 0) });
        mk("rect", { x: X(p), y: Y(1), width: (1 - p) * S, height: (1 - b) * S, fill: "#16a34a", opacity: 0.25 * fade });
        // evidence parts
        mk("rect", { x: X(0), y: Y(a), width: p * S, height: a * S, fill: "#3b82f6", opacity: 0.85 });
        mk("rect", { x: X(p), y: Y(b), width: (1 - p) * S, height: b * S, fill: "#16a34a", opacity: 0.85 });
        mk("rect", { x: X(0), y: Y(a), width: p * S, height: a * S, fill: "none", stroke: "#f59e0b", "stroke-width": 4 });
        mk("rect", { x: X(p), y: Y(b), width: (1 - p) * S, height: b * S, fill: "none", stroke: "#f59e0b", "stroke-width": 4 });
        mk("rect", { x: X(0), y: Y(1), width: S, height: S, fill: "none", stroke: "currentColor", "stroke-width": 1.5, opacity: 0.6 });
        // handles
        const hdiv = mk("line", { x1: X(p), x2: X(p), y1: Y(1) - 12, y2: Y(0) + 12, stroke: "currentColor", "stroke-width": 3 });
        const hdivHit = mk("rect", { x: X(p) - 9, y: Y(1) - 14, width: 18, height: S + 28, fill: "transparent", cursor: "ew-resize" });
        hdivHit.addEventListener("pointerdown", e => { drag = "prior"; svg.setPointerCapture(e.pointerId); });
        const bar = (x0, w, v, key) => {
          mk("line", { x1: X(x0), x2: X(x0 + w), y1: Y(v), y2: Y(v), stroke: "#f59e0b", "stroke-width": 5 });
          const k = mk("circle", { cx: X(x0 + w / 2), cy: Y(v), r: 8, fill: "#f59e0b", stroke: "#fff", "stroke-width": 2, cursor: "ns-resize" });
          const hit = mk("rect", { x: X(x0), y: Y(v) - 9, width: Math.max(w * S, 18), height: 18, fill: "transparent", cursor: "ns-resize" });
          [k, hit].forEach(n => n.addEventListener("pointerdown", e => { drag = key; svg.setPointerCapture(e.pointerId); }));
        };
        bar(0, p, a, "lik_h"); bar(p, 1 - p, b, "lik_nh");
        // labels
        const t = (x, y, s, attrs = {}) => { const e = mk("text", Object.assign({ x, y, "font-size": 13, fill: "currentColor", "text-anchor": "middle" }, attrs)); e.textContent = s; return e; };
        t(X(p / 2), Y(0) + 18, `P(H)=${p.toFixed(2)}`, { fill: "#3b82f6" });
        t(X(p + (1 - p) / 2), Y(0) + 18, `P(¬H)=${(1 - p).toFixed(2)}`, { fill: "#16a34a" });
        t(X(0) - 4, Y(a) + 4, `${a.toFixed(2)}`, { "text-anchor": "end", fill: "#f59e0b" });
        t(X(1) + 4, Y(b) + 4, `${b.toFixed(2)}`, { "text-anchor": "start", fill: "#f59e0b" });
        t(X(0.5), Y(1) - 18, "P(E|H) on the left · P(E|¬H) on the right (heights)", { "font-size": 12, opacity: 0.7 });
        // posterior bar: amber region squashed into a column
        const ah = p * a, an = (1 - p) * b, tot = ah + an, post = tot > 0 ? ah / tot : 0;
        mk("rect", { x: BARX, y: Y(1), width: 40, height: S, fill: "none", stroke: "currentColor", opacity: 0.4 });
        mk("rect", { x: BARX, y: Y(post), width: 40, height: post * S, fill: "#3b82f6" });
        mk("rect", { x: BARX, y: Y(1), width: 40, height: (1 - post) * S, fill: "#16a34a" });
        mk("line", { x1: BARX - 6, x2: BARX + 46, y1: Y(p), y2: Y(p), stroke: "currentColor", "stroke-dasharray": "4 3" });
        t(BARX + 50, Y(p) + 4, "prior", { "text-anchor": "start", "font-size": 11 });
        t(BARX + 20, Y(1) - 18, "P(H|E)", { fill: "#e4572e", "font-weight": "bold" });
        t(BARX + 20, Y(0) + 18, `${(100 * post).toFixed(1)}%`, { fill: "#e4572e", "font-weight": "bold", "font-size": 16 });
      }
      svg.addEventListener("pointermove", e => {
        if (!drag) return;
        const q = toSvg(e);
        if (drag === "prior") model.set("prior", Math.round(clamp((q.x - PAD) / S) * 200) / 200);
        else model.set(drag, Math.round(clamp(1 - (q.y - PAD) / S) * 100) / 100);
        draw();
      });
      const stop = () => { if (drag) { drag = null; model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointerleave", stop);
      ["prior", "lik_h", "lik_nh", "restrict"].forEach(k => model.on("change:" + k, draw));
      draw();
    }
    export default { render };
    """
        prior = traitlets.Float(1 / 21).tag(sync=True)
        lik_h = traitlets.Float(0.4).tag(sync=True)
        lik_nh = traitlets.Float(0.1).tag(sync=True)
        restrict = traitlets.Bool(False).tag(sync=True)

    return (BayesSquare,)


@app.cell
def bayes_fn():
    def bayes_posterior(prior, lik_h, lik_nh):
        """P(H|E) = P(H)P(E|H) / [P(H)P(E|H) + P(¬H)P(E|¬H)]"""
        _num = prior * lik_h
        _den = _num + (1 - prior) * lik_nh
        return _num / _den if _den > 0 else float("nan")

    assert abs(bayes_posterior(1 / 21, 0.4, 0.1) - 4 / 24) < 1e-12
    return (bayes_posterior,)


@app.cell
def square(BayesSquare, mo):
    square = mo.ui.anywidget(BayesSquare())
    square
    return (square,)


@app.cell
def square_readout(bayes_posterior, mo, square):
    _p, _a, _b = square.value["prior"], square.value["lik_h"], square.value["lik_nh"]
    square_post = bayes_posterior(_p, _a, _b)
    _lr = _a / _b if _b > 0 else float("inf")
    mo.vstack([
        mo.hstack([
            mo.stat(f"{_p:.3f}", label="prior P(H)", caption="width of the blue strip"),
            mo.stat(f"{_p * _a:.4f}", label="P(H ∩ E)", caption="left amber area"),
            mo.stat(f"{_p * _a + (1 - _p) * _b:.4f}", label="P(E)", caption="total amber area"),
            mo.stat(f"{square_post:.1%}", label="posterior P(H|E)", caption="left share of amber"),
            mo.stat(f"{_lr:.2f}×", label="likelihood ratio", caption="P(E|H) / P(E|¬H)"),
        ], justify="start"),
        mo.md(f"Belief moved from **{_p:.1%}** to **{square_post:.1%}**: "
              + ("**up**." if square_post > _p + 1e-9 else "**down**." if square_post < _p - 1e-9 else "**not at all**: the evidence was irrelevant.")),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - Drag the two amber bars to the **same height**. The posterior bar lines up with the dashed prior line. **Irrelevant evidence doesn't change your beliefs.**
    - Make the librarian bar very tall and the farmer bar very short. *Very different* likelihoods are what move your belief a lot.
    - Keep the likelihoods at 0.4 and 0.1 and drag the divider. The same evidence takes a 5% prior to 17%, and a 50% prior to 80%. **Evidence updates the prior. It doesn't replace it.**
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Why "40 out of 100" beats "40%": Linda

    In another Kahneman and Tversky experiment, people read about **Linda**: *31, single, outspoken, very bright, majored in philosophy, concerned with discrimination and social justice, joined anti-nuclear demonstrations.* Then they were asked which is more likely:

    1. Linda is a bank teller.
    2. Linda is a bank teller **and** is active in the feminist movement.

    **85%** chose (2). But feminist bank tellers are a **subset** of bank tellers, so (2) can never be more likely than (1).
    When the question was reframed as *"of 100 people who fit this description, how many are…"*, the error dropped to **0%**. Try it both ways.
    """)
    return


@app.cell
def linda_guess(mo):
    linda_guess = mo.ui.radio(["(1) bank teller", "(2) bank teller AND feminist"], label="**Abstract version:** which is more likely?")
    linda_guess
    return (linda_guess,)


@app.cell
def linda_tellers(mo):
    linda_tellers = mo.ui.slider(0, 100, value=10, label="**Counting version:** of 100 people like Linda, how many are bank tellers?", show_value=True, full_width=True)
    linda_tellers
    return (linda_tellers,)


@app.cell
def linda_fem(mo):
    linda_fem = mo.ui.slider(0, 100, value=5, label="…and how many are bank tellers **active in the feminist movement**?", show_value=True, full_width=True)
    linda_fem
    return (linda_fem,)


@app.cell
def linda_chart(alt, linda_fem, linda_guess, linda_tellers, mo, np, pd):
    _t, _f = linda_tellers.value, linda_fem.value
    _grid = pd.DataFrame({"i": range(100)})
    _grid["col"], _grid["row"] = _grid.i % 10, _grid.i // 10
    _grid["group"] = np.where(_grid.i < min(_f, _t), "feminist bank teller",
                     np.where(_grid.i < _t, "bank teller (other)", "not a bank teller"))
    _grid.loc[(_grid.i >= _t) & (_grid.i < _f), "group"] = "impossible: feminist teller who is not a teller"
    _chart = alt.Chart(_grid).mark_circle(size=260).encode(
        x=alt.X("col:O", axis=None), y=alt.Y("row:O", axis=None),
        color=alt.Color("group:N", scale=alt.Scale(
            domain=["feminist bank teller", "bank teller (other)", "not a bank teller", "impossible: feminist teller who is not a teller"],
            range=["#e4572e", "#f59e0b", "#d1d5db", "#7c3aed"]), legend=alt.Legend(orient="bottom", columns=2)),
        tooltip=["group"],
    ).properties(width="container", height=260, title="100 people who fit Linda's description")
    _ok = _f <= _t
    _msg = (mo.callout(mo.md(f"With counts, the subset is obvious: **{_f} ≤ {_t}**. Every red dot is also a bank teller."), kind="success") if _ok
            else mo.callout(mo.md(f"You have **{_f} feminist bank tellers** but only **{_t} bank tellers**. The purple dots can't exist. That is the conjunction fallacy."), kind="danger"))
    _abs = ("" if linda_guess.value is None else
            "Your abstract answer was **(2)**, the conjunction fallacy. Compare it with your counts." if linda_guess.value.startswith("(2)")
            else "Your abstract answer was **(1)**, which is correct.")
    mo.vstack([_chart, _msg, mo.md(_abs)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - Set feminist tellers **above** bank tellers. The counting picture shows straight away that this is impossible.
    - This is why Section 1 used **210 people**. Counts trigger our intuition. Area (Section 3) does the same job and also handles continuous probabilities and pencil sketches.
    - The math of probability is **the math of proportions**. Bayes' theorem says: *look at the cases where the evidence is true, then find the proportion of those where the hypothesis is also true.*
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Unfinished business: who *is* Steve?

    Some psychologists dispute the "irrational" verdict. **Who is Steve?** A random American? A friend of the psychologists? Someone *you* are likely to meet? (Grant says he runs into many more librarians than farmers.)
    How likely each group is to fit the description is also open to interpretation.

    The diagram absorbs the whole debate:
    - arguments about **context** move the **prior** (the divider),
    - arguments about **personalities and stereotypes** move the **likelihoods** (the amber bars).

    Choose a framing for Steve and set the likelihoods. The heat map shows the posterior for **every** prior × likelihood ratio, and the ★ marks your current choice.
    """)
    return


@app.cell
def steve_controls(mo):
    steve_context = mo.ui.dropdown(
        {"Random American (K&T: 20 farmers per librarian)": 1 / 21,
         "Modern US stats (~60 : 1)": 1 / 61,
         "Someone a city-dweller meets (1 : 1)": 0.5,
         "A friend of two academic psychologists (5 : 1 librarians)": 5 / 6},
        value="Random American (K&T: 20 farmers per librarian)", label="Who is Steve? (sets the prior)")
    steve_lh = mo.ui.slider(0.05, 1, step=0.05, value=0.4, label="P(description | librarian)", show_value=True)
    steve_lnh = mo.ui.slider(0.01, 1, step=0.01, value=0.1, label="P(description | farmer)", show_value=True)
    mo.vstack([steve_context, mo.hstack([steve_lh, steve_lnh], justify="start")])
    return steve_context, steve_lh, steve_lnh


@app.cell
def steve_heatmap(
    alt,
    bayes_posterior,
    mo,
    np,
    pd,
    steve_context,
    steve_lh,
    steve_lnh,
):
    _priors = np.logspace(-2.5, np.log10(0.99), 60)
    _lrs = np.logspace(-1, 2, 60)
    _P, _L = np.meshgrid(_priors, _lrs)
    _post = _P * _L / (_P * _L + (1 - _P))   # divide num & den by P(E|¬H): only the ratio matters
    _df = pd.DataFrame({"prior": _P.ravel(), "lr": _L.ravel(), "posterior": _post.ravel()})
    _pr, _lr = steve_context.value, steve_lh.value / steve_lnh.value
    steve_post = bayes_posterior(_pr, steve_lh.value, steve_lnh.value)
    _heat = alt.Chart(_df).mark_point(filled=True, shape="square", size=60).encode(
        x=alt.X("prior:Q", scale=alt.Scale(type="log"), title="prior P(librarian)  (log scale)"),
        y=alt.Y("lr:Q", scale=alt.Scale(type="log"), title="likelihood ratio P(E|H)/P(E|¬H)  (log)"),
        color=alt.Color("posterior:Q", scale=alt.Scale(scheme="greenblue", domain=[0, 1]), title="P(H|E)"),
        tooltip=[alt.Tooltip("prior:Q", format=".3f"), alt.Tooltip("lr:Q", format=".2f"), alt.Tooltip("posterior:Q", format=".2f")],
    )
    _half = alt.Chart(pd.DataFrame({"prior": _priors, "lr": (1 - _priors) / _priors})).transform_filter("datum.lr >= 0.1 && datum.lr <= 100").mark_line(color="white", strokeDash=[5, 4]).encode(x="prior:Q", y="lr:Q")
    _star = alt.Chart(pd.DataFrame({"prior": [_pr], "lr": [_lr]})).mark_point(shape="diamond", size=300, color="#e4572e", filled=True, stroke="white", strokeWidth=2).encode(x="prior:Q", y="lr:Q")
    mo.vstack([
        (_heat + _half + _star).properties(width="container", height=360, title="Posterior for every (prior, likelihood ratio). Dashed: posterior = 50%"),
        mo.hstack([mo.stat(f"{_pr:.3f}", label="prior"), mo.stat(f"{_lr:.1f}×", label="likelihood ratio"),
                   mo.stat(f"{steve_post:.1%}", label="P(librarian | description)")], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - Switch to *"Someone a city-dweller meets"*. With the same stereotypes, Steve is now **80%** likely to be a librarian. The context argument is entirely an argument about the prior.
    - Under the *Modern US stats* prior, how big does the likelihood ratio have to be to reach 50%? (Find the dashed line: the ratio has to equal the prior odds against.)
    - Contours run **diagonally**: doubling the prior odds and doubling the likelihood ratio have the same effect.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Sandbox: draw your own square

    Pick a scenario to load into a fresh square, then drag freely. Below the square, the **posterior becomes the new prior**: the chart shows what happens if you keep seeing *independent* pieces of evidence with the same likelihoods. Each update is just a new square that starts from the last posterior.
    """)
    return


@app.cell
def sandbox_pick(mo):
    sandbox_scenarios = {
        "Steve (K&T)": (1 / 21, 0.40, 0.10),
        "Medical test: 1% prevalence, 90% sensitivity, 9% false positives": (0.01, 0.90, 0.09),
        "Irrelevant evidence (Steve likes pizza)": (1 / 21, 0.60, 0.60),
        "Smoking gun: rare but damning": (0.02, 0.95, 0.001),
        "Evidence against H": (0.5, 0.1, 0.6),
    }
    sandbox_pick = mo.ui.dropdown(list(sandbox_scenarios), value="Medical test: 1% prevalence, 90% sensitivity, 9% false positives", label="Scenario")
    sandbox_pick
    return sandbox_pick, sandbox_scenarios


@app.cell
def sandbox_square(BayesSquare, mo, sandbox_pick, sandbox_scenarios):
    _p, _a, _b = sandbox_scenarios[sandbox_pick.value]
    sandbox_square = mo.ui.anywidget(BayesSquare(prior=_p, lik_h=_a, lik_nh=_b, restrict=True))
    sandbox_square
    return (sandbox_square,)


@app.cell
def sandbox_chain(alt, bayes_posterior, mo, pd, sandbox_square):
    _p, _a, _b = (sandbox_square.value[k] for k in ("prior", "lik_h", "lik_nh"))
    _beliefs = [_p]
    for _ in range(8):
        _beliefs.append(bayes_posterior(_beliefs[-1], _a, _b))
    _df = pd.DataFrame({"pieces of evidence seen": range(9), "P(H)": _beliefs})
    _line = alt.Chart(_df).mark_line(point=True, color="#e4572e").encode(
        x=alt.X("pieces of evidence seen:O"), y=alt.Y("P(H):Q", scale=alt.Scale(domain=[0, 1])),
        tooltip=[alt.Tooltip("P(H):Q", format=".4f")])
    _rule = alt.Chart(pd.DataFrame({"y": [0.5]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
    mo.vstack([
        mo.hstack([mo.stat(f"{_p:.3f}", label="prior"), mo.stat(f"{_beliefs[1]:.1%}", label="after 1 piece of evidence"),
                   mo.stat(f"{_beliefs[2]:.1%}", label="after 2"), mo.stat(f"{_beliefs[-1]:.1%}", label="after 8")], justify="start"),
        (_line + _rule).properties(width="container", height=240, title="Repeated independent evidence: yesterday's posterior is today's prior"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Challenges**
    - **Medical test:** a positive result from a "90% accurate" test leaves you at only **~9%**. Drag the divider to see how prevalence decides this. How many positive tests in a row before it passes 50%?
    - **Pizza:** equal likelihoods give a flat line. You can watch irrelevant evidence forever and learn nothing.
    - **Smoking gun:** the farmer bar is almost invisible. The likelihood *ratio* (950×), not the likelihood itself, is what overwhelms a tiny prior.
    - Find settings where one piece of evidence takes you from **below 10% to above 90%**. What ratio did you need?
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Quiz: read the square

    Answer all four, then flip the switch. Draw the square in your head; you don't need the formula.
    """)
    return


@app.cell
def quiz(mo):
    quiz = mo.ui.dictionary({
        "q1": mo.ui.radio(["~9%", "~50%", "~90%"], label="1. 1% of people have a disease. The test catches 90% of cases and gives 9% false positives. You test positive. P(disease)?"),
        "q2": mo.ui.radio(["goes up", "stays the same", "goes down"], label="2. Librarians and farmers are equally likely to own a cat. Steve owns a cat. P(librarian)…"),
        "q3": mo.ui.radio(["P(H) · P(E|H)", "P(E|H) only", "P(H|E) · P(E|¬H)"], label="3. In the unit square, the left amber rectangle has area…"),
        "q4": mo.ui.radio(["the prior", "the likelihoods", "the sample size N"], label="4. Arguing about who Steve is likely to be (random American vs. a friend of yours) changes…"),
    })
    quiz_reveal = mo.ui.switch(label="Reveal answers")
    mo.vstack([quiz, quiz_reveal])
    return quiz, quiz_reveal


@app.cell
def quiz_feedback(bayes_posterior, mo, quiz, quiz_reveal):
    _truth_q1 = bayes_posterior(0.01, 0.9, 0.09)
    _bucket = "~9%" if _truth_q1 < 0.3 else "~50%" if _truth_q1 < 0.7 else "~90%"
    _answers = {
        "q1": (_bucket, f"Amber area: 0.01·0.9 = 0.009 on the left vs 0.99·0.09 ≈ 0.089 on the right, so {_truth_q1:.1%}."),
        "q2": ("stays the same", "Equal likelihoods mean equal-height amber bars, so the proportion doesn't change."),
        "q3": ("P(H) · P(E|H)", "Width P(H) × height P(E|H) = P(H and E)."),
        "q4": ("the prior", "Context moves the divider. Stereotypes move the bars. N always cancels."),
    }
    _got = {k: quiz.value.get(k) for k in _answers}
    _score = sum(_got[k] == v[0] for k, v in _answers.items())
    _hidden = mo.md(f"_Answered {sum(v is not None for v in _got.values())}/4. Flip the switch to check._")
    _shown = mo.vstack([
        mo.stat(f"{_score}/4", label="score"),
        *[mo.callout(mo.md(f"**{k.upper()}**: {'✅' if _got[k] == a else '❌'} answer: **{a}**. {why}"), kind="success" if _got[k] == a else "danger")
          for k, (a, why) in _answers.items()],
    ])
    _shown if quiz_reveal.value else _hidden
    return


@app.cell(hide_code=True)
def recap(mo):
    mo.md(r"""
    ## Recap

    | idea | in the video | in this notebook |
    |---|---|---|
    | Base rates matter | Steve: 20 farmers per librarian. "Rationality is recognizing which facts are relevant." | §1 people grid |
    | Evidence **restricts** the space | Keep only those who fit; 4 of 24 = 16.7% | §1 👁 button, §3 restrict toggle |
    | Bayes' theorem | $P(H\mid E)=\dfrac{P(H)P(E\mid H)}{P(H)P(E\mid H)+P(\neg H)P(E\mid\neg H)}$ | §2 live formula, $N$ cancels |
    | Prior / likelihood / posterior | names for $P(H)$, $P(E\mid H)$, $P(H\mid E)$ | §2 table |
    | Probability as **area** | 1 × 1 square; posterior = blue share of the amber region | §3 draggable square |
    | Irrelevant evidence | equal likelihoods leave the belief unchanged | §3, sandbox "pizza" |
    | Counts beat percentages | Linda: 85% → 0% errors with "out of 100" | §4 |
    | The debate fits the diagram | context → prior, stereotypes → likelihoods | §5 heat map |
    | Beliefs keep updating | today's posterior is tomorrow's prior | §6 sandbox |

    **Mantra:** *evidence should not determine beliefs, it should update them.*
    """)
    return


if __name__ == "__main__":
    app.run()

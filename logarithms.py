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
        mo.md("# Logarithms: turning multiplication into addition"),
        mo.Html('<iframe width="100%" height="380" src="https://www.youtube.com/embed/xMygCJ1FoMQ" title="This Is the Logarithm Explanation You Never Got" frameborder="0" allowfullscreen></iframe>'),
        mo.md("_Based on **Eigen: \"This Is the Logarithm Explanation You Never Got\"**. One idea runs through the whole lesson: **write down how many times a number was multiplied by itself, and add those counts instead of multiplying the numbers.**_"),
    ])
    return


@app.cell(hide_code=True)
def colors(mo):
    BLUE, ORANGE, RED, GREY = "#2e86ab", "#f18f01", "#e4572e", "#9e9e9e"
    mo.callout(mo.md(f"""
    **Colour language used throughout**

    - <span style="color:{BLUE}">**blue = numbers**</span>: the top row, the world where we *multiply*
    - <span style="color:{ORANGE}">**orange = counting numbers / exponents / logarithms**</span>: the bottom row, the world where we *add*
    - <span style="color:{RED}">**red = the answer**</span> you are hunting for
    """), kind="info")
    return BLUE, GREY, ORANGE, RED


@app.cell(hide_code=True)
def s1_intro(mo):
    mo.md(r"""
    ## 1. The doubling trick

    400 years ago, multiplying two big numbers by hand could take most of a day. The question was: *can multiplication be turned into addition?*

    Start with powers of two: $2, 4, 8, 16, 32, 64, \dots$ Each number is the last one times two. Under each one, write **how many twos** were multiplied to build it, the *counting number*:

    $$\underbrace{2\cdot2\cdot2}_{3\text{ twos}} \cdot \underbrace{2\cdot2\cdot2\cdot2}_{4\text{ twos}} = \underbrace{2\cdot2\cdot2\cdot2\cdot2\cdot2\cdot2}_{3+4=7\text{ twos}} \quad\Longrightarrow\quad 8\times16 = 128$$

    **Multiplying on top is the same as adding on the bottom.** Click any two numbers on the top row below.
    """)
    return


@app.cell
def NapierTable(anywidget, traitlets):
    class NapierTable(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const BLUE = "#2e86ab", ORANGE = "#f18f01", RED = "#e4572e";
      const wrap = document.createElement("div");
      wrap.style.fontFamily = "system-ui, sans-serif";
      el.appendChild(wrap);
      function draw() {
        const b = model.get("base"), n = model.get("n"), sel = model.get("selected");
        const sum = sel.length === 2 ? sel[0] + sel[1] : null;
        let h = '<div style="overflow-x:auto"><table style="border-collapse:separate;border-spacing:4px;text-align:center">';
        h += '<tr><td style="color:' + BLUE + ';font-weight:600;padding-right:8px">number</td>';
        for (let k = 1; k <= n; k++) {
          const on = sel.includes(k), ans = k === sum;
          const bg = on ? BLUE : ans ? RED : "rgba(46,134,171,0.12)";
          const fg = on || ans ? "white" : "inherit";
          h += '<td data-k="' + k + '" style="cursor:pointer;min-width:44px;padding:8px 6px;border-radius:8px;background:' + bg + ';color:' + fg + ';font-weight:600">' + Math.pow(b, k).toLocaleString() + '</td>';
        }
        h += '</tr><tr><td style="color:' + ORANGE + ';font-weight:600;padding-right:8px">count</td>';
        for (let k = 1; k <= n; k++) {
          const on = sel.includes(k), ans = k === sum;
          const bg = on ? ORANGE : ans ? RED : "rgba(241,143,1,0.12)";
          const fg = on || ans ? "white" : "inherit";
          h += '<td style="padding:6px;border-radius:8px;background:' + bg + ';color:' + fg + '">' + k + '</td>';
        }
        h += '</tr></table></div>';
        let msg = "Click two numbers on the top row.";
        if (sel.length === 1) msg = "Picked " + Math.pow(b, sel[0]).toLocaleString() + ". Pick one more.";
        if (sel.length === 2) {
          const [i, j] = sel, p = Math.pow(b, i) * Math.pow(b, j);
          msg = '<span style="color:' + BLUE + '">' + Math.pow(b, i).toLocaleString() + ' &times; ' + Math.pow(b, j).toLocaleString() + '</span>'
            + ' &nbsp;&harr;&nbsp; <span style="color:' + ORANGE + '">' + i + ' + ' + j + ' = ' + sum + '</span>'
            + ' &nbsp;&rArr;&nbsp; <span style="color:' + RED + ';font-weight:700">' + p.toLocaleString() + '</span>'
            + (sum > n ? ' <i>(off the end of the table!)</i>' : '');
        }
        wrap.innerHTML = h + '<div style="margin-top:8px;font-size:1.05em">' + msg + '</div>';
        wrap.querySelectorAll("td[data-k]").forEach(td => td.onclick = () => {
          const k = +td.dataset.k; let s = [...model.get("selected")];
          if (s.length >= 2) s = [];
          s.push(k); model.set("selected", s); model.save_changes();
        });
      }
      model.on("change:selected", draw); model.on("change:base", draw); model.on("change:n", draw);
      draw();
    }
    export default { render };
    """
        base = traitlets.Int(2).tag(sync=True)
        n = traitlets.Int(10).tag(sync=True)
        selected = traitlets.List(traitlets.Int()).tag(sync=True)

    return (NapierTable,)


@app.cell
def napier_base_ui(mo):
    napier_base = mo.ui.dropdown({"2": 2, "3": 3, "10": 10}, value="2", label="Base of the table")
    napier_base
    return (napier_base,)


@app.cell
def napier(NapierTable, mo, napier_base):
    napier = mo.ui.anywidget(NapierTable(base=napier_base.value, n=12 if napier_base.value < 10 else 7, selected=[2, 3]))
    napier
    return (napier,)


@app.cell
def napier_readout(mo, napier):
    _sel = napier.value["selected"]
    _b = napier.value["base"]
    if len(_sel) == 2:
        _i, _j = _sel
        _out = mo.hstack([
            mo.stat(f"{_b**_i:,} × {_b**_j:,}", label="multiply (top row)", caption="the slow way"),
            mo.stat(f"{_i} + {_j} = {_i+_j}", label="add (bottom row)", caption="the fast way"),
            mo.stat(f"{_b**(_i+_j):,}", label=f"{_b}^{_i+_j}", caption="✅ matches" if _b**_i * _b**_j == _b**(_i+_j) else "❌"),
        ], justify="start")
    else:
        _out = mo.md("_Pick two numbers to see the readout._")
    _out
    return


@app.cell(hide_code=True)
def s1_try(mo):
    mo.callout(mo.md("""
    **Try this**

    1. Reproduce the video: click **4** and **8** (2 + 3 = 5 → 32), then **8** and **16** (3 + 4 = 7 → 128).
    2. Click the same number twice: you just *squared* it by *doubling* its counting number.
    3. Switch the base to **10**: 100 × 1,000 ↔ 2 + 3 = 5 → 100,000. The trick does not care about the base.
    """), kind="success")
    return


@app.cell(hide_code=True)
def s2_intro(mo):
    mo.md(r"""
    ## 2. The problem with the gaps

    The top row $2, 4, 8, 16, 32$ is spread far apart. What if we want $7 \times 12$? Neither is on the row; they fall **in the gaps**.

    So we loosen what "counting number" means. $7$ sits between $8 = 2^3$ and $4 = 2^2$, close to $8$, so its counting number should be a bit under $3$. The video says *"something like 2.8"*, and indeed $2^{2.8} \approx 6.96$.

    **Predict first:** what counting number sits under 7?
    """)
    return


@app.cell
def gap_guess_ui(mo):
    gap_target = mo.ui.dropdown(["7", "12", "50", "100", "1000"], value="7", label="Number in the gap")
    gap_guess = mo.ui.slider(0.0, 10.0, step=0.05, value=2.5, label="Your guess for its counting number", show_value=True)
    gap_lock = mo.ui.run_button(label="Lock in guess")
    mo.hstack([gap_target, gap_guess, gap_lock], justify="start")
    return gap_guess, gap_lock, gap_target


@app.cell
def gap_reveal(gap_guess, gap_lock, gap_target, mo, np):
    mo.stop(not gap_lock.value, mo.md("_Make your guess, then lock it in._"))
    _N = int(gap_target.value)
    _true = np.log2(_N)
    _lo = int(np.floor(_true))
    _err = abs(gap_guess.value - _true)
    mo.vstack([
        mo.md(f"$2^{{{_lo}}} = {2**_lo} \\;<\\; {_N} \\;<\\; 2^{{{_lo+1}}} = {2**(_lo+1)}$, so the answer is between **{_lo}** and **{_lo+1}**."),
        mo.hstack([
            mo.stat(f"{gap_guess.value:.2f}", label="your guess", caption=f"2^guess = {2**gap_guess.value:.2f}"),
            mo.stat(f"{_true:.3f}", label=f"true counting number of {_N}", caption=f"2^{_true:.3f} = {_N}"),
            mo.stat(f"{_err:.2f}", label="off by", caption="🎯 nailed it" if _err < 0.1 else "close" if _err < 0.4 else "keep going"),
        ], justify="start"),
    ])
    return


@app.cell
def frac_ui(mo):
    frac_exp = mo.ui.slider(0.0, 6.0, step=0.01, value=2.8, label="Exponent x (fractional allowed)", show_value=True)
    frac_exp
    return (frac_exp,)


@app.cell
def frac_chart(BLUE, GREY, ORANGE, RED, alt, frac_exp, mo, np, pd):
    _x = frac_exp.value
    _ks = np.arange(0, 7)
    _dots = pd.DataFrame({"x": _ks, "value": 2.0**_ks, "kind": "whole-number powers"})
    _line = pd.DataFrame({"x": np.linspace(0, 6, 300)})
    _line["value"] = 2**_line["x"]
    _base = alt.Chart(_dots).mark_circle(size=120, color=BLUE).encode(
        x=alt.X("x:Q", title="counting number (exponent)"), y=alt.Y("value:Q", title="number = 2^x"))
    _curve = alt.Chart(_line).mark_line(color=GREY, strokeDash=[4, 4]).encode(x="x:Q", y="value:Q")
    _pt = alt.Chart(pd.DataFrame({"x": [_x], "value": [2**_x]})).mark_point(size=250, color=RED, filled=True).encode(
        x="x:Q", y="value:Q", tooltip=["x", "value"])
    _rule = alt.Chart(pd.DataFrame({"x": [_x]})).mark_rule(color=ORANGE).encode(x="x:Q")
    mo.vstack([
        (_curve + _base + _rule + _pt).properties(width="container", height=280),
        mo.md(f"$2^{{{_x:.2f}}} = $ <span style='color:{RED}'>**{2**_x:.3f}**</span>: the gaps between the blue dots fill in with fractional exponents."),
    ])
    return


@app.cell(hide_code=True)
def s2_try(mo):
    mo.callout(mo.md("""
    **Try this**

    1. Set the slider to **2.8**: you land on ≈ 6.96, the video's "almost exactly seven".
    2. Find the exponent that gives **12** (hint: it is between 3 and 4, closer to 4).
    3. Set it to **0.5**: $2^{0.5} \\approx 1.414 = \\sqrt 2$. A half-count of twos is a square root.
    """), kind="success")
    return


@app.cell(hide_code=True)
def s3_intro(mo):
    mo.md(r"""
    ## 3. Turning it into a curve, then flipping it

    Plot exponent → number and you get the **exponential curve** $y = 2^x$: slow at first, then faster and faster.
    But we want the *opposite*: hand it a number (say 7) and get its exponent back. So we **swap the axes**. Every point $(x, 2^x)$ becomes $(2^x, x)$, a mirror image across the diagonal $y = x$.

    That flipped curve is *Napier's table drawn with no gaps*. Its name is the **logarithm**:

    $$\log_2(N) = \text{the exponent you raise 2 to in order to reach } N, \qquad \log_2 32 = 5.$$

    **Drag the blue point** along the exponential curve; its orange mirror rides along the logarithm.
    """)
    return


@app.cell
def MirrorCurve(anywidget, traitlets):
    class MirrorCurve(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const BLUE = "#2e86ab", ORANGE = "#f18f01", GREY = "#9e9e9e";
      const W = 520, H = 520, PAD = 36, MIN = -1, MAX = 9;
      const NS = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none" });
      el.appendChild(svg);
      const mk = (tag, a, p = svg) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
      const sx = v => PAD + (v - MIN) / (MAX - MIN) * (W - 2 * PAD);
      const sy = v => H - PAD - (v - MIN) / (MAX - MIN) * (H - 2 * PAD);
      const ix = px => MIN + (px - PAD) / (W - 2 * PAD) * (MAX - MIN);
      const toSvg = e => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const path = f => { let d = ""; for (let i = 0; i <= 400; i++) { const t = MIN + i / 400 * (MAX - MIN); const v = f(t);
          if (!isFinite(v.x) || !isFinite(v.y) || v.y > MAX + 1 || v.y < MIN - 1 || v.x > MAX + 1 || v.x < MIN - 1) continue;
          d += (d ? "L" : "M") + sx(v.x).toFixed(1) + "," + sy(v.y).toFixed(1); } return d; };
      let drag = false;
      function draw() {
        const b = model.get("base"), x = model.get("x"), y = Math.pow(b, x);
        svg.innerHTML = "";
        for (let g = MIN; g <= MAX; g++) {
          mk("line", { x1: sx(g), x2: sx(g), y1: sy(MIN), y2: sy(MAX), stroke: "currentColor", "stroke-opacity": g === 0 ? 0.5 : 0.08 });
          mk("line", { y1: sy(g), y2: sy(g), x1: sx(MIN), x2: sx(MAX), stroke: "currentColor", "stroke-opacity": g === 0 ? 0.5 : 0.08 });
          if (g > MIN) { const t = mk("text", { x: sx(g) + 2, y: sy(0) + 13, "font-size": 10, fill: "currentColor", "fill-opacity": 0.6 }); t.textContent = g; }
        }
        mk("line", { x1: sx(MIN), y1: sy(MIN), x2: sx(MAX), y2: sy(MAX), stroke: GREY, "stroke-dasharray": "6 5" });
        mk("path", { d: path(t => ({ x: t, y: Math.pow(b, t) })), fill: "none", stroke: BLUE, "stroke-width": 3 });
        mk("path", { d: path(t => ({ x: Math.pow(b, t), y: t })), fill: "none", stroke: ORANGE, "stroke-width": 3 });
        if (y <= MAX) {
          mk("line", { x1: sx(x), y1: sy(y), x2: sx(y), y2: sy(x), stroke: GREY, "stroke-dasharray": "2 4" });
          const m = mk("circle", { cx: sx(y), cy: sy(x), r: 9, fill: ORANGE, stroke: "white", "stroke-width": 2 });
        }
        const c = mk("circle", { cx: sx(x), cy: sy(Math.min(y, MAX + 0.5)), r: 11, fill: BLUE, stroke: "white", "stroke-width": 2 });
        c.style.cursor = "grab";
        c.addEventListener("pointerdown", e => { drag = true; svg.setPointerCapture(e.pointerId); });
        const lab = mk("text", { x: PAD + 6, y: PAD + 4, "font-size": 15, fill: "currentColor" });
        lab.innerHTML = `<tspan fill="${BLUE}">${b}^${x.toFixed(2)} = ${y.toFixed(2)}</tspan>   <tspan fill="${ORANGE}">log${b}(${y.toFixed(2)}) = ${x.toFixed(2)}</tspan>`;
        const t1 = mk("text", { x: sx(2.6), y: sy(8.4), "font-size": 13, fill: BLUE }); t1.textContent = `y = ${b}^x`;
        const t2 = mk("text", { x: sx(7.0), y: sy(Math.log(7) / Math.log(b)) - 10, "font-size": 13, fill: ORANGE }); t2.textContent = `y = log${b}(x)`;
        const t3 = mk("text", { x: sx(7.6), y: sy(8.2), "font-size": 11, fill: GREY }); t3.textContent = "y = x";
      }
      svg.addEventListener("pointermove", e => { if (!drag) return;
        const b = model.get("base"), xmax = Math.log(MAX) / Math.log(b);
        model.set("x", Math.max(MIN, Math.min(xmax, Math.round(ix(toSvg(e).x) * 100) / 100))); draw(); });
      const stop = () => { if (drag) { drag = false; model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointerleave", stop);
      model.on("change:x", draw); model.on("change:base", draw); draw();
    }
    export default { render };
    """
        base = traitlets.Float(2.0).tag(sync=True)
        x = traitlets.Float(2.8).tag(sync=True)

    return (MirrorCurve,)


@app.cell
def mirror_base_ui(mo, np):
    mirror_base = mo.ui.dropdown({"2": 2.0, "e ≈ 2.718": float(np.e), "3": 3.0, "10": 10.0}, value="2", label="Base")
    mirror_base
    return (mirror_base,)


@app.cell
def mirror(MirrorCurve, mirror_base, mo):
    mirror = mo.ui.anywidget(MirrorCurve(base=mirror_base.value, x=2.8 if mirror_base.value < 3 else 0.8))
    mirror
    return (mirror,)


@app.cell
def mirror_readout(mirror, mo, np):
    _b, _x = mirror.value["base"], mirror.value["x"]
    _N = _b**_x
    mo.hstack([
        mo.stat(f"{_x:.2f}", label="exponent (orange)", caption="vertical on the log curve"),
        mo.stat(f"{_N:.3f}", label="number (blue)", caption=f"= {_b:.3g}^{_x:.2f}"),
        mo.stat(f"{np.log(_N)/np.log(_b):.3f}", label=f"log_{_b:.3g}({_N:.3f})", caption="recovers the exponent ✅"),
    ], justify="start")
    return


@app.cell(hide_code=True)
def s3_try(mo):
    mo.callout(mo.md("""
    **Try this**

    1. Drag the blue point to x = 3: blue says 2³ = 8, orange says log₂ 8 = 3. Same pair, coordinates swapped.
    2. Look at the orange curve's shape: from 2→4→8 the number doubles each time while the log only goes up by **1**. Huge numbers get squeezed into small heights.
    3. Drag to x = 0: every base gives 1, so **log(1) = 0** for any base. Drag below 0: numbers between 0 and 1 have *negative* logs.
    """), kind="success")
    return


@app.cell(hide_code=True)
def s4_intro(mo):
    mo.md(r"""
    ## 4. Multiply by just adding

    The rule that is "the entire reason the logarithm exists":

    $$\log(A\cdot B) = \log A + \log B$$

    and the ones that fall out for free:

    $$\log\frac{A}{B} = \log A - \log B, \qquad \log(A^n) = n\,\log A \;\;(\text{so } \sqrt[3]{A} \text{ becomes } \tfrac{1}{3}\log A).$$

    Napier's readers did it like this: **look up** two logs, **add**, then **look up backwards** which number has that sum. Walk through it step by step:
    """)
    return


@app.cell
def rule_ui(mo, np):
    rule_a = mo.ui.number(start=0.001, stop=1e9, value=32, label="A")
    rule_b = mo.ui.number(start=0.001, stop=1e9, value=8, label="B")
    rule_op = mo.ui.radio(["A × B", "A ÷ B", "A ^ B (power)"], value="A × B", label="Operation", inline=True)
    rule_base = mo.ui.dropdown({"2": 2.0, "10": 10.0, "e": float(np.e)}, value="2", label="Base of the table")
    mo.hstack([rule_a, rule_b, rule_op, rule_base], justify="start")
    return rule_a, rule_b, rule_base, rule_op


@app.cell
def rule_steps(BLUE, ORANGE, RED, mo, np, rule_a, rule_b, rule_base, rule_op):
    def log_b(x, b):
        """log of x in base b, the one function this whole notebook teaches."""
        return np.log(x) / np.log(b)

    _A, _B, _b = rule_a.value, rule_b.value, rule_base.value
    _bs = "e" if abs(_b - np.e) < 1e-9 else f"{_b:g}"
    _la, _lb = log_b(_A, _b), log_b(_B, _b)
    if rule_op.value == "A × B":
        _s, _sym, _true, _desc = _la + _lb, "+", _A * _B, "add"
        _step2 = f"{_la:.4f} + {_lb:.4f} = **{_la+_lb:.4f}**"
    elif rule_op.value == "A ÷ B":
        _s, _sym, _true, _desc = _la - _lb, "−", _A / _B, "subtract"
        _step2 = f"{_la:.4f} − {_lb:.4f} = **{_la-_lb:.4f}**"
    else:
        _s, _true, _desc = _B * _la, _A**_B, "multiply by B"
        _step2 = f"{_B:g} × {_la:.4f} = **{_B*_la:.4f}**"
    mo.md(f"""
    | step | what you do | result |
    |---|---|---|
    | 1. look up | <span style="color:{ORANGE}">log<sub>{_bs}</sub>({_A:g}) and log<sub>{_bs}</sub>({_B:g})</span> | {_la:.4f} and {_lb:.4f} |
    | 2. {_desc} | <span style="color:{ORANGE}">{_step2}</span> | {_s:.4f} |
    | 3. look up backwards | <span style="color:{BLUE}">{_bs}<sup>{_s:.4f}</sup></span> | <span style="color:{RED}">**{_b**_s:,.6g}**</span> |
    | check | direct calculation | {_true:,.6g} {"✅" if np.isclose(_b**_s, _true) else "❌"} |
    """)
    return (log_b,)


@app.cell(hide_code=True)
def s4_try(mo):
    mo.callout(mo.md("""
    **Try this**

    1. The video's example: **32 × 8** in base 2 → 5 + 3 = 8 → 2⁸ = 256. "We never once multiplied."
    2. Change the base to **10** or **e**: the logs change, the final answer doesn't. The base is just the ruler's units.
    3. Cube root: pick **A = 1000**, **B = 1/3 ≈ 0.3333**, power. Multiplying the log by ⅓ is "a cube root as a plain division".
    """), kind="success")
    return


@app.cell(hide_code=True)
def s5_intro(mo):
    mo.md(r"""
    ## 5. The slide rule: adding lengths multiplies numbers

    Take a ruler, but **space the marks by their logarithm**: 1, 2, 4, 8 sit equally far apart, because each is one step up in exponent. Put two such rulers next to each other and slide one along the other.

    To multiply $3 \times 4$: put the **1** of the top ruler over **3** on the bottom ruler, then read the bottom ruler under **4** on the top ruler. The lengths add,
    $\log 3 + \log 4 = \log 12$, so the reading is 12. For ~300 years, bridges, ships' tables and early rockets were worked out this way.

    **Drag the top ruler** left/right, and **drag the red hairline** to read off products.
    """)
    return


@app.cell
def SlideRule(anywidget, traitlets):
    class SlideRule(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const BLUE = "#2e86ab", ORANGE = "#f18f01", RED = "#e4572e";
      const W = 760, H = 200, L = 30, LEN = 340;   // one decade = LEN px, bottom ruler spans 2 decades
      const NS = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none" });
      el.appendChild(svg);
      const info = document.createElement("div"); info.style.fontSize = "1.05em"; info.style.marginTop = "4px"; el.appendChild(info);
      const mk = (tag, a, p = svg) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
      const toSvg = e => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const lg = Math.log10;
      const marks = [];
      for (let d = 0; d < 2; d++) for (let k = 1; k < 10; k++) for (let s = 0; s < 10; s += (k < 5 ? 1 : 5)) marks.push({ v: (k + s / 10) * Math.pow(10, d), major: s === 0 });
      marks.push({ v: 100, major: true });
      let drag = null, start = 0, start0 = 0;
      function ruler(x0, y, up, color, maxv) {
        const g = mk("g", {});
        mk("rect", { x: x0, y: up ? y - 34 : y, width: LEN * lg(maxv), height: 34, fill: color, "fill-opacity": 0.13, stroke: color, rx: 4 }, g);
        for (const m of marks) { if (m.v > maxv + 1e-9) continue;
          const x = x0 + LEN * lg(m.v), h = m.major ? 14 : 7;
          mk("line", { x1: x, x2: x, y1: y, y2: up ? y - h : y + h, stroke: color, "stroke-width": m.major ? 1.6 : 0.8 }, g);
          if (m.major) { const t = mk("text", { x: x, y: up ? y - 18 : y + 28, "text-anchor": "middle", "font-size": 11, fill: color, "font-weight": 600 }, g);
            t.textContent = m.v >= 10 ? m.v : m.v; } }
        return g;
      }
      function draw() {
        const off = model.get("offset"), cur = model.get("cursor");
        svg.innerHTML = "";
        const top = ruler(L + LEN * off, 90, true, BLUE, 10);
        top.style.cursor = "grab";
        top.addEventListener("pointerdown", e => { drag = "top"; start = toSvg(e).x; start0 = model.get("offset"); svg.setPointerCapture(e.pointerId); });
        ruler(L, 96, false, ORANGE, 100);
        const lab1 = mk("text", { x: 4, y: 70, "font-size": 11, fill: BLUE }); lab1.textContent = "C";
        const lab2 = mk("text", { x: 4, y: 120, "font-size": 11, fill: ORANGE }); lab2.textContent = "D";
        // length arrows
        mk("line", { x1: L, x2: L + LEN * off, y1: 168, y2: 168, stroke: ORANGE, "stroke-width": 4 });
        mk("line", { x1: L + LEN * off, x2: L + LEN * (off + cur), y1: 168, y2: 168, stroke: BLUE, "stroke-width": 4 });
        const t = mk("text", { x: L, y: 188, "font-size": 11, fill: "currentColor" });
        t.textContent = `length log(A) + length log(B) = log(A·B)`;
        const hx = L + LEN * (off + cur);
        const hl = mk("line", { x1: hx, x2: hx, y1: 30, y2: 150, stroke: RED, "stroke-width": 2.5 });
        const hit = mk("rect", { x: hx - 8, y: 20, width: 16, height: 135, fill: "transparent" });
        hit.style.cursor = "ew-resize";
        hit.addEventListener("pointerdown", e => { e.stopPropagation(); drag = "cur"; start = toSvg(e).x; start0 = model.get("cursor"); svg.setPointerCapture(e.pointerId); });
        const A = Math.pow(10, off), B = Math.pow(10, cur), P = A * B;
        info.innerHTML = `top-ruler 1 sits over <b style="color:${ORANGE}">A = ${A.toFixed(2)}</b>; hairline on top ruler at <b style="color:${BLUE}">B = ${B.toFixed(2)}</b>; read below: <b style="color:${RED}">A×B ≈ ${P.toFixed(P < 10 ? 2 : 1)}</b>`;
      }
      svg.addEventListener("pointermove", e => { if (!drag) return;
        const dx = (toSvg(e).x - start) / LEN;
        if (drag === "top") model.set("offset", Math.max(0, Math.min(1, start0 + dx)));
        else model.set("cursor", Math.max(0, Math.min(1, start0 + dx)));
        draw(); });
      const stop = () => { if (drag) { drag = null; model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointerleave", stop);
      model.on("change:offset", draw); model.on("change:cursor", draw); draw();
    }
    export default { render };
    """
        offset = traitlets.Float(0.47712125).tag(sync=True)   # log10(3)
        cursor = traitlets.Float(0.60205999).tag(sync=True)   # log10(4)

    return (SlideRule,)


@app.cell
def slide(SlideRule, mo):
    slide = mo.ui.anywidget(SlideRule())
    slide
    return (slide,)


@app.cell
def slide_readout(mo, np, slide):
    _A, _B = 10 ** slide.value["offset"], 10 ** slide.value["cursor"]
    mo.hstack([
        mo.stat(f"{np.log10(_A):.3f} + {np.log10(_B):.3f}", label="lengths added (log₁₀ units)"),
        mo.stat(f"{_A:.3f} × {_B:.3f}", label="numbers multiplied"),
        mo.stat(f"{_A*_B:.3f}", label="exact product", caption=f"10^{np.log10(_A)+np.log10(_B):.3f}"),
    ], justify="start")
    return


@app.cell(hide_code=True)
def s5_try(mo):
    mo.callout(mo.md("""
    **Try this**

    1. The default is the video's **3 × 4**: the hairline lands on **12**.
    2. Slide the top ruler so its **1** sits on **2**. Now every top mark *n* sits above **2n** below: a whole times-two table at once.
    3. Measure: the gap from 1 to 2 equals the gap from 2 to 4 and from 4 to 8. Equal ratios = equal distances.
    """), kind="success")
    return


@app.cell(hide_code=True)
def s6_intro(mo):
    mo.md(r"""
    ## 6. Where logarithms show up today

    Some things grow by **multiplying**: populations, money earning interest, sound energy, earthquake shaking. On a normal axis the curve "shoots off the top of the page". Plot the **logarithm** instead, and runaway growth straightens into a line, with even, readable steps.
    """)
    return


@app.cell
def growth_ui(mo):
    growth_rate = mo.ui.slider(1, 50, value=7, label="Growth per year (%)", show_value=True)
    growth_years = mo.ui.slider(10, 150, value=100, label="Years", show_value=True)
    growth_log = mo.ui.switch(value=False, label="Logarithmic y-axis")
    mo.hstack([growth_rate, growth_years, growth_log], justify="start")
    return growth_log, growth_rate, growth_years


@app.cell
def growth_chart(
    BLUE,
    ORANGE,
    RED,
    alt,
    growth_log,
    growth_rate,
    growth_years,
    mo,
    np,
    pd,
):
    _t = np.arange(growth_years.value + 1)
    _df = pd.DataFrame({"year": _t, "value": 100 * (1 + growth_rate.value / 100) ** _t})
    _df["log10(value)"] = np.log10(_df["value"])
    _scale = alt.Scale(type="log") if growth_log.value else alt.Scale(type="linear")
    _ch = alt.Chart(_df).mark_line(color=BLUE, strokeWidth=3).encode(
        x=alt.X("year:Q"), y=alt.Y("value:Q", scale=_scale, title="amount (start = 100)"),
        tooltip=["year", alt.Tooltip("value:Q", format=",.0f")]).properties(width="container", height=300)
    _slope = np.log10(1 + growth_rate.value / 100)
    mo.vstack([_ch, mo.md(
        f"On the log axis the line rises by a constant <span style='color:{ORANGE}'>**{_slope:.4f}**</span> in log₁₀ per year, "
        f"i.e. it gains one power of ten every <span style='color:{RED}'>**{1/_slope:.1f} years**</span>. "
        f"Final value: {_df['value'].iloc[-1]:,.0f}.")])
    return


@app.cell(hide_code=True)
def db_intro(mo):
    mo.md(r"""
    ### Sound: why "twice as loud" is about ten times the energy

    Decibels are a log scale: $\text{dB} = 10\log_{10}\!\left(\frac{E}{E_0}\right)$. Every **+10 dB = ×10 energy**, and our ears hear roughly +10 dB as "twice as loud". Earthquake magnitudes work the same way.
    """)
    return


@app.cell
def db_ui(mo):
    db_level = mo.ui.slider(0, 130, step=1, value=60, label="Sound level (dB)", show_value=True)
    db_level
    return (db_level,)


@app.cell
def db_readout(BLUE, RED, alt, db_level, mo, pd):
    _ref = [(0, "threshold of hearing"), (30, "whisper"), (60, "conversation"), (85, "busy traffic"), (100, "jackhammer"), (120, "rock concert"), (130, "pain")]
    _name = [n for d, n in _ref if d <= db_level.value][-1]
    _E = 10 ** (db_level.value / 10)
    _df = pd.DataFrame({"sound": [n for _, n in _ref], "dB": [d for d, _ in _ref]})
    _df["energy (× threshold)"] = 10.0 ** (_df["dB"] / 10)
    _bars = alt.Chart(_df).mark_bar(color=BLUE).encode(
        y=alt.Y("sound:N", sort=None, title=None), x=alt.X("energy (× threshold):Q", scale=alt.Scale(type="log")))
    _rule = alt.Chart(pd.DataFrame({"E": [_E]})).mark_rule(color=RED, strokeWidth=3).encode(x="E:Q")
    mo.vstack([
        mo.hstack([
            mo.stat(f"{db_level.value} dB", label="level", caption=f"≈ {_name}"),
            mo.stat(f"{_E:.3g}×", label="energy vs. threshold"),
            mo.stat(f"{2**(db_level.value/10):.3g}×", label="perceived loudness vs. threshold", caption="doubles every +10 dB"),
        ], justify="start"),
        (_bars + _rule).properties(width="container", height=220),
    ])
    return


@app.cell(hide_code=True)
def s6_try(mo):
    mo.callout(mo.md("""
    **Try this**

    1. Growth at 7% for 100 years on a linear axis: nothing happens for 60 years, then a wall. Flip the log switch: a straight line.
    2. Raise the growth rate: on the log axis only the **slope** changes. Straight line on a log axis means constant *multiplicative* growth.
    3. Move the dB slider from 60 to 70: perceived loudness doubles, energy goes ×10. From 60 to 120: energy ×1,000,000.
    """), kind="success")
    return


@app.cell(hide_code=True)
def sb_intro(mo):
    mo.md(r"""
    ## 7. Sandbox: be Napier for a minute

    Type any list of positive numbers and pick a base. The notebook multiplies them **only by adding logs**, like a 17th-century calculator, and plots the log curves of several bases. Notice they are all the *same shape*, just rescaled: $\log_b x = \dfrac{\log_{10} x}{\log_{10} b}$. That is why the base is a choice of units, not a different idea.
    """)
    return


@app.cell
def sb_ui(mo):
    sb_numbers = mo.ui.text(value="7, 12, 3.5, 250", label="Numbers to multiply", full_width=True)
    sb_base = mo.ui.slider(1.1, 20, step=0.1, value=2.0, label="Base", show_value=True)
    mo.vstack([sb_numbers, sb_base])
    return sb_base, sb_numbers


@app.cell
def sb_out(alt, log_b, mo, np, pd, sb_base, sb_numbers):
    try:
        _nums = [float(s) for s in sb_numbers.value.replace(";", ",").split(",") if s.strip()]
        assert _nums and all(v > 0 for v in _nums)
    except Exception:
        _nums = None
    mo.stop(_nums is None, mo.callout(mo.md("Enter a comma-separated list of **positive** numbers (logs of 0 or negatives don't exist)."), kind="warn"))
    _b = sb_base.value
    _tab = pd.DataFrame({"number": _nums, f"log_{_b:g}(number)": [log_b(v, _b) for v in _nums]})
    _s = _tab.iloc[:, 1].sum()
    _xs = np.geomspace(0.05, 100, 300)
    _curves = pd.concat([pd.DataFrame({"x": _xs, "log": log_b(_xs, bb), "base": lbl})
                         for bb, lbl in [(2, "2"), (np.e, "e"), (10, "10"), (_b, f"your base {_b:g}")]])
    _ch = alt.Chart(_curves).mark_line().encode(
        x=alt.X("x:Q", title="number"), y=alt.Y("log:Q", title="log_b(number)"),
        color=alt.Color("base:N", legend=alt.Legend(orient="top", title="base")),
        strokeDash=alt.condition(alt.datum.base == f"your base {_b:g}", alt.value([1, 0]), alt.value([4, 3])),
    ).properties(width="container", height=260)
    mo.vstack([
        mo.hstack([mo.ui.table(_tab.round(5), selection=None), mo.vstack([
            mo.stat(f"{_s:.5f}", label="sum of logs (orange world)"),
            mo.stat(f"{_b**_s:,.6g}", label=f"{_b:g}^sum (red answer)", caption=f"direct product: {np.prod(_nums):,.6g}"),
        ])], widths=[2, 1]),
        _ch,
    ])
    return


@app.cell(hide_code=True)
def sb_try(mo):
    mo.callout(mo.md("""
    **Challenges**

    1. Multiply 20 numbers between 1 and 2. The sum of logs stays small while the product blows up.
    2. Set the base to exactly **10** and multiply 1000, 100: logs 3 + 2 = 5. That's Briggs' "round, clean steps".
    3. Find a base where log of 2 is 1 and log of 8 is 3. (Then try e ≈ 2.7, the "natural" base the video leaves hanging.)
    """), kind="success")
    return


@app.cell(hide_code=True)
def quiz_intro(mo):
    mo.md("""
    ## 8. Quiz: check yourself
    Answer all five, then flip the reveal switch.
    """)
    return


@app.cell
def quiz(mo):
    quiz = mo.ui.dictionary({
        "q1": mo.ui.dropdown(["4", "5", "6", "32"], label="1. log₂(32) = ?"),
        "q2": mo.ui.dropdown(["between 2 and 3", "between 3 and 4", "exactly 3.5", "between 7 and 8"], label="2. log₂(12) lies…"),
        "q3": mo.ui.dropdown(["log A × log B", "log A + log B", "log(A + B)", "A · log B"], label="3. log(A·B) = ?"),
        "q4": mo.ui.dropdown(["100", "1,000", "10", "300"], label="4. A +30 dB change multiplies sound energy by?"),
        "q5": mo.ui.dropdown(["divide log A by 3", "subtract 3 from log A", "multiply A by ⅓", "take log of log A"], label="5. To take the cube root with logs you…"),
    })
    quiz_reveal = mo.ui.switch(label="Reveal answers")
    mo.vstack([quiz, quiz_reveal])
    return quiz, quiz_reveal


@app.cell
def quiz_score(log_b, mo, quiz, quiz_reveal):
    _l12 = log_b(12, 2)
    _truth = {
        "q1": f"{log_b(32, 2):.0f}",
        "q2": "between 3 and 4" if 3 < _l12 < 4 else "between 2 and 3",
        "q3": "log A + log B",
        "q4": f"{10**(30/10):,.0f}",
        "q5": "divide log A by 3",
    }
    _why = {
        "q1": "2·2·2·2·2 = 32: five twos.",
        "q2": f"8 < 12 < 16, so it's between 3 and 4 (≈ {_l12:.3f}).",
        "q3": "multiplication inside becomes addition outside.",
        "q4": "every +10 dB is ×10 energy: 10·10·10.",
        "q5": "log(A^(1/3)) = (1/3)·log A, a plain division.",
    }
    _score = sum(quiz.value[k] == v for k, v in _truth.items())
    if not quiz_reveal.value:
        _out = mo.md(f"Answered **{sum(v is not None for v in quiz.value.values())}/5**. Flip the switch when ready.")
    else:
        _rows = "\n".join(f"| {k[1]} | {quiz.value[k] or '—'} | {_truth[k]} | {'✅' if quiz.value[k] == _truth[k] else '❌'} | {_why[k]} |" for k in _truth)
        _out = mo.vstack([
            mo.stat(f"{_score}/5", label="score", caption="🏆 Napier-grade" if _score == 5 else "review the sections marked ❌"),
            mo.md("| # | yours | correct | | why |\n|---|---|---|---|---|\n" + _rows),
        ])
    _out
    return


@app.cell(hide_code=True)
def recap(mo):
    mo.md(r"""
    ## Recap

    | Idea | In one line | Where you touched it |
    |---|---|---|
    | Doubling trick | multiply on top ↔ add the counting numbers below | Napier table widget |
    | Filling the gaps | fractional exponents give every number a counting number ($2^{2.8}\approx 6.96$) | guess-the-gap + exponent slider |
    | The curve & its flip | $\log_b N$ = the exponent that reaches $N$; mirror of $b^x$ across $y=x$ | draggable mirror curve |
    | The rules | $\log AB=\log A+\log B$, $\log\frac AB=\log A-\log B$, $\log A^n = n\log A$ | look-up / add / look-back table |
    | Base is a choice | base 2, 10 (Briggs), e (natural): same curve rescaled | base dropdowns, sandbox |
    | Slide rule | log-spaced rulers: adding lengths multiplies | draggable slide rule |
    | Log scales | multiplicative growth → straight line; +10 dB = ×10 energy | growth chart, dB slider |

    **Open thread from the video:** why does the strange base $e \approx 2.718$ show up "on its own"? That's the next lesson.
    """)
    return


if __name__ == "__main__":
    app.run()

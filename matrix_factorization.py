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
    import sympy as sp

    return alt, anywidget, mo, np, pd, sp, traitlets


@app.cell(hide_code=True)
def title(mo):
    mo.vstack([
        mo.md(r"""
    # Matrix Factorization: factoring the unfactorable

    *An interactive companion to* **[Matrix Factorization – Numberphile](https://www.youtube.com/watch?v=wTUSz-HSaBg)** *with Prof. David Eisenbud.*

    Some polynomials can't be factored. Mathematicians have handled stuck problems like this before: they **enlarge the domain**. Negative numbers, fractions, and *i* were all added this way.
    Dirac enlarged it to **matrices**, and Eisenbud proved that almost every polynomial can then be factored.
    """),
        mo.Html('<iframe width="560" height="315" src="https://www.youtube.com/embed/wTUSz-HSaBg" title="Matrix Factorization - Numberphile" frameborder="0" allowfullscreen></iframe>'),
        mo.callout(mo.md(r"""
    **Color language used throughout:**
    <span style="color:#2e86de">**blue = the first factor A**</span>,
    <span style="color:#f18f01">**orange = the second factor B**</span>,
    <span style="color:#e4572e">**red = the product / the polynomial f**</span>.
    Gray means zero or unknown.
    """), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def s1_hook(mo):
    mo.md(r"""
    ## 1. Factoring, and why you sometimes need a bigger toolbox

    The video opens with middle-school factoring:

    $$x^2-4=(x-2)(x+2),\qquad x^2-y^2=(x-y)(x+y),\qquad x^2+2xy+y^2=(x+y)^2 .$$

    $x^2+3$ will not factor over the rationals, because there is no square root of $-3$. **"If you enlarge the domain of things you accept, there's a factorization."**
    Allow complex numbers and $x^2+3=(x-i\sqrt3)(x+i\sqrt3)$.
    This is the same move that gave us fractions (to solve $3x-1=0$) and negative numbers (to solve $3x+1=0$).

    Pick a polynomial and a number system. Watch which factorizations appear and which never do.
    """)
    return


@app.cell
def s1_controls(mo):
    poly_pick = mo.ui.dropdown(
        options=["x**2 - 4", "x**2 - y**2", "x**2 + 2*x*y + y**2", "x**2 + 3", "x**2 - 3",
                 "x**2 + y**2", "x*y - u*v", "t**2 + x**2 + y**2 + z**2"],
        value="x**2 + 3", label="Polynomial",
    )
    domain_pick = mo.ui.radio(
        options=["integers / rationals ℚ", "add square roots (√3)", "add i as well (complex)"],
        value="integers / rationals ℚ", label="Allowed numbers",
    )
    mo.hstack([poly_pick, domain_pick], justify="start", gap=2)
    return domain_pick, poly_pick


@app.cell
def s1_readout(domain_pick, mo, poly_pick, sp):
    def factor_in(expr, domain):
        """Factor a sympy expression after allowing more numbers into the domain."""
        if domain.startswith("integers"):
            return sp.factor(expr)
        if domain.startswith("add square"):
            return sp.factor(expr, extension=[sp.sqrt(3)])
        return sp.factor(expr, extension=[sp.I, sp.sqrt(3)])

    _f = sp.sympify(poly_pick.value)
    _rows = []
    for _d in domain_pick.options:
        _fac = factor_in(_f, _d)
        _ok = sum(int(_a.exp) if _a.is_Pow else 1 for _a in sp.Mul.make_args(_fac) if not _a.is_number) >= 2
        _rows.append(f"| {_d} | ${sp.latex(_fac)}$ | {'✅ factors' if _ok else '❌ stuck'} |")
    _chosen = factor_in(_f, domain_pick.value)
    mo.vstack([
        mo.md(f"### <span style='color:#e4572e'>f = ${sp.latex(_f)}$</span>  →  ${sp.latex(_chosen)}$"),
        mo.md("| number system | best factorization | |\n|---|---|---|\n" + "\n".join(_rows)),
    ])
    return


@app.cell(hide_code=True)
def s1_try(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Move $x^2+3$ from ℚ to *complex*. It only splits once you allow $i$.
    2. $x^2+y^2$ also needs $i$: $(x-iy)(x+iy)$.
    3. Now pick **Dirac's polynomial** $t^2+x^2+y^2+z^2$ (and its cousin $xy-uv$). They stay stuck **even with complex numbers**. No number system rescues them. That is the cliffhanger the rest of the lesson resolves.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def s2_hook(mo):
    mo.md(r"""
    ## 2. How do you multiply matrices? Rows times columns

    To factor *into* matrices we first need a matrix multiplication. Eisenbud writes it out for $2\times2$:

    $$\begin{pmatrix}a&b\\c&d\end{pmatrix}\begin{pmatrix}a'&b'\\c'&d'\end{pmatrix}
    =\begin{pmatrix}aa'+bc' & ab'+bd'\\ ca'+dc' & cb'+dd'\end{pmatrix}.$$

    Each entry of the product is a **row of the first matrix** combined with a **column of the second**. Because rows and columns play different roles, **order matters**:
    in general $AB\neq BA$. "They don't commute."

    Drag the <span style="color:#2e86de">blue</span> and <span style="color:#f18f01">orange</span> numbers up or down. Hover a <span style="color:#e4572e">red</span> entry to see the row and column that produce it.
    """)
    return


@app.cell
def matmul_class(anywidget, traitlets):
    class MatMulWidget(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const COL = { A: "#2e86de", B: "#f18f01", P: "#e4572e" };
      const root = document.createElement("div");
      root.style.cssText = "font-family:sans-serif;display:flex;flex-direction:column;gap:10px;user-select:none;touch-action:none";
      el.appendChild(root);
      const cap = document.createElement("div"); cap.style.cssText = "min-height:2.6em;font-size:15px";
      let hover = null, drag = null;
      const mul = (X, Y) => [X[0]*Y[0]+X[1]*Y[2], X[0]*Y[1]+X[1]*Y[3], X[2]*Y[0]+X[3]*Y[2], X[2]*Y[1]+X[3]*Y[3]];
      function makeGrid(parent, label, color, editKey) {
        const box = document.createElement("div");
        box.style.cssText = `display:inline-grid;grid-template-columns:repeat(2,54px);gap:4px;padding:6px;border-left:3px solid ${color};border-right:3px solid ${color};border-radius:6px`;
        const cells = [];
        for (let k = 0; k < 4; k++) {
          const d = document.createElement("div");
          d.style.cssText = `height:40px;display:flex;align-items:center;justify-content:center;border-radius:6px;font-size:18px;font-weight:600;color:${color}`;
          if (editKey) { d.style.cursor = "ns-resize"; d.title = "drag up/down";
            d.addEventListener("pointerdown", e => startDrag(e, editKey, k)); }
          box.appendChild(d); cells.push(d);
        }
        const w = document.createElement("div"); w.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:2px";
        const l = document.createElement("div"); l.textContent = label; l.style.cssText = `font-size:13px;color:${color};font-weight:600`;
        w.append(box, l); parent.appendChild(w); return cells;
      }
      function row(spec) {
        const r = document.createElement("div"); r.style.cssText = "display:flex;align-items:center;gap:12px";
        const sym = t => { const s = document.createElement("span"); s.textContent = t; s.style.fontSize = "22px"; r.appendChild(s); };
        const g1 = makeGrid(r, ...spec[0]); sym("×"); const g2 = makeGrid(r, ...spec[1]); sym("="); const g3 = makeGrid(r, ...spec[2]);
        root.appendChild(r); return [g1, g2, g3];
      }
      const [a1, b1, p1] = row([["A", COL.A, "A"], ["B", COL.B, "B"], ["A·B", COL.P, null]]);
      const [b2, a2, p2] = row([["B", COL.B, "B"], ["A", COL.A, "A"], ["B·A", COL.P, null]]);
      root.appendChild(cap);
      [[p1, "AB"], [p2, "BA"]].forEach(([g, which]) => g.forEach((d, k) => {
        d.style.cursor = "help";
        d.addEventListener("pointerenter", () => { hover = [which, k]; draw(); });
        d.addEventListener("pointerleave", () => { hover = null; draw(); });
      }));
      function startDrag(e, key, k) {
        e.preventDefault(); drag = { key, k, y0: e.clientY, v0: model.get(key)[k] };
        window.addEventListener("pointermove", onMove); window.addEventListener("pointerup", onUp);
      }
      function onMove(e) {
        if (!drag) return;
        const v = Math.max(-9, Math.min(9, drag.v0 + Math.round((drag.y0 - e.clientY) / 14)));
        const arr = [...model.get(drag.key)];
        if (arr[drag.k] !== v) { arr[drag.k] = v; model.set(drag.key, arr); draw(); }
      }
      function onUp() {
        window.removeEventListener("pointermove", onMove); window.removeEventListener("pointerup", onUp);
        if (drag) { drag = null; model.save_changes(); }
      }
      const fmt = v => (Math.round(v * 100) / 100).toString();
      const paint = (cells, vals, hl, color) => cells.forEach((d, k) => {
        d.textContent = fmt(vals[k]);
        d.style.background = hl.includes(k) ? color + "40" : "rgba(127,127,127,.08)";
      });
      function draw() {
        const A = model.get("A"), B = model.get("B"), AB = mul(A, B), BA = mul(B, A);
        let hA1 = [], hB1 = [], hB2 = [], hA2 = [], hP1 = [], hP2 = [];
        if (hover) {
          const [w, k] = hover, r = k >> 1, c = k & 1, ri = [2*r, 2*r+1], ci = [c, c+2];
          const [X, Y, nx, ny, cx, cy, P] = w === "AB" ? [A, B, "A", "B", COL.A, COL.B, AB] : [B, A, "B", "A", COL.B, COL.A, BA];
          if (w === "AB") { hA1 = ri; hB1 = ci; hP1 = [k]; } else { hB2 = ri; hA2 = ci; hP2 = [k]; }
          cap.innerHTML = `(${nx}·${ny})<sub>${r+1}${c+1}</sub> = row ${r+1} of <b style="color:${cx}">${nx}</b> · column ${c+1} of <b style="color:${cy}">${ny}</b> = `
            + `<span style="color:${cx}">${fmt(X[2*r])}</span>·<span style="color:${cy}">${fmt(Y[c])}</span> + `
            + `<span style="color:${cx}">${fmt(X[2*r+1])}</span>·<span style="color:${cy}">${fmt(Y[c+2])}</span> = <b style="color:${COL.P}">${fmt(P[k])}</b>`;
        } else {
          const same = AB.every((v, i) => v === BA[i]);
          cap.innerHTML = `Hover a red entry to see its row × column. Drag a blue/orange number to change it. Right now A·B ${same ? "<b>=</b>" : "<b>≠</b>"} B·A.`;
        }
        paint(a1, A, hA1, COL.A); paint(b1, B, hB1, COL.B); paint(p1, AB, hP1, COL.P);
        paint(b2, B, hB2, COL.B); paint(a2, A, hA2, COL.A); paint(p2, BA, hP2, COL.P);
      }
      model.on("change:A", draw); model.on("change:B", draw); draw();
    }
    export default { render };
    """
        A = traitlets.List(traitlets.Float(), default_value=[1, 2, 3, 4]).tag(sync=True)
        B = traitlets.List(traitlets.Float(), default_value=[0, 1, 1, 0]).tag(sync=True)

    return (MatMulWidget,)


@app.cell
def matmul_widget(MatMulWidget, mo):
    matmul = mo.ui.anywidget(MatMulWidget(A=[1, 2, 3, 4], B=[0, 1, 1, 0]))
    matmul
    return (matmul,)


@app.cell
def matmul_readout(matmul, mo, np):
    _A = np.array(matmul.value["A"]).reshape(2, 2)
    _B = np.array(matmul.value["B"]).reshape(2, 2)
    _AB, _BA = _A @ _B, _B @ _A
    _comm = _AB - _BA
    mo.hstack([
        mo.stat(value="yes" if np.allclose(_AB, _BA) else "no", label="AB = BA ?", bordered=True),
        mo.stat(value=f"{np.abs(_comm).max():.0f}", label="largest entry of AB − BA", bordered=True),
        mo.stat(value=f"{np.linalg.det(_AB):.0f} = {np.linalg.det(_BA):.0f}", label="det(AB) = det(BA), always", bordered=True),
        mo.stat(value=f"{np.trace(_AB):.0f} = {np.trace(_BA):.0f}", label="trace(AB) = trace(BA), always", bordered=True),
    ], justify="start")
    return


@app.cell(hide_code=True)
def s2_try(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. The default <span style="color:#f18f01">B</span> is the *swap* matrix. On the right, $AB$ swaps A's **columns**. On the left, $BA$ swaps A's **rows**. Same two matrices, different products.
    2. Make B the identity (1, 0, 0, 1). Now $AB=BA=A$. The identity behaves like the number 1, and that will matter in Section 3.
    3. Even when $AB\neq BA$, the determinant and trace of the two products always agree. Order changes the matrix but not every quantity derived from it.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def s3_hook(mo):
    mo.md(r"""
    ## 3. Dirac's trick: factor $xy-uv$ with matrices

    Dirac's polynomial $t^2+x^2+y^2+z^2$ becomes, after a complex change of coordinates, the simpler-looking $xy-uv$.
    Ordinary polynomials can't factor it. With matrices it is one line:

    $$
    \color{#2e86de}{\underbrace{\begin{pmatrix}x&u\\v&y\end{pmatrix}}_{A}}\;
    \color{#f18f01}{\underbrace{\begin{pmatrix}y&-u\\-v&x\end{pmatrix}}_{B}}
    =\color{#e4572e}{\begin{pmatrix}xy-uv&0\\0&xy-uv\end{pmatrix}} = (xy-uv)\cdot I .
    $$

    "This equality is wrong" if you read it literally, because the left side is a matrix and the right side a polynomial.
    The polynomial is **personified** as $(xy-uv)$ times the identity matrix $I$, which behaves like the number 1. In that sense, $A\cdot B$ factors $xy-uv$.

    Plug in numbers and check that the product is always $f$ on the diagonal and zero elsewhere, in both orders.
    """)
    return


@app.cell
def s3_sliders(mo):
    sx = mo.ui.slider(-5, 5, step=1, value=3, label="x", show_value=True)
    sy = mo.ui.slider(-5, 5, step=1, value=2, label="y", show_value=True)
    su = mo.ui.slider(-5, 5, step=1, value=1, label="u", show_value=True)
    sv = mo.ui.slider(-5, 5, step=1, value=4, label="v", show_value=True)
    mo.hstack([sx, sy, su, sv], justify="start")
    return su, sv, sx, sy


@app.cell
def s3_readout(mo, sp, su, sv, sx, sy):
    x, y, u, v = sp.symbols("x y u v")
    A_dirac = sp.Matrix([[x, u], [v, y]])
    B_dirac = sp.Matrix([[y, -u], [-v, x]])
    f_dirac = x * y - u * v

    _sub = {x: sx.value, y: sy.value, u: su.value, v: sv.value}
    _An, _Bn = A_dirac.subs(_sub), B_dirac.subs(_sub)
    _L = sp.latex
    mo.vstack([
        mo.md(
            rf"$$\color{{#2e86de}}{{{_L(_An)}}}\color{{#f18f01}}{{{_L(_Bn)}}} = \color{{#e4572e}}{{{_L(_An * _Bn)}}}"
            rf"\qquad \color{{#f18f01}}{{{_L(_Bn)}}}\color{{#2e86de}}{{{_L(_An)}}} = \color{{#e4572e}}{{{_L(_Bn * _An)}}}$$"
        ),
        mo.md(
            rf"<span style='color:#e4572e'>$f = xy-uv = {sx.value}\cdot{sy.value} - {su.value}\cdot{sv.value} = {f_dirac.subs(_sub)}$</span>. "
            rf"Symbolically, $AB - fI = {_L(sp.expand(A_dirac * B_dirac - f_dirac * sp.eye(2)))}$ and "
            rf"$BA - fI = {_L(sp.expand(B_dirac * A_dirac - f_dirac * sp.eye(2)))}$."
        ),
    ])
    return A_dirac, B_dirac, f_dirac, u, v, x, y


@app.cell(hide_code=True)
def s3_try(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Choose values where $xy=uv$, for example x=2, y=3, u=1, v=6. The product becomes the **zero matrix**, even though neither A nor B is zero. Ordinary numbers can't do that ($ab=0$ forces $a=0$ or $b=0$).
    2. Notice that $B$ is just $A$ with the diagonal swapped and the off-diagonal negated, its **adjugate**. And $xy-uv=\det A$. Hold on to that; the sandbox uses it.
    3. $AB$ and $BA$ agree here, unlike in Section 2. A matrix factorization requires **both** $AB=fI$ and $BA=fI$.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def s4_hook(mo):
    mo.md(r"""
    ## 4. Dirac wanted a *square root*: go up to 4×4

    Dirac didn't just want *a* factorization. He wanted a **square root**, a single matrix $M$ with $M\cdot M = f\cdot I$.
    "This requires bigger matrices, no problem." Stack the two factors from Section 3 into a $4\times4$ block matrix:

    $$M=\begin{pmatrix}0&\color{#2e86de}{A}\\ \color{#f18f01}{B}&0\end{pmatrix},\qquad
    M^2=\begin{pmatrix}\color{#2e86de}{A}\color{#f18f01}{B}&0\\0&\color{#f18f01}{B}\color{#2e86de}{A}\end{pmatrix}=\color{#e4572e}{(xy-uv)\,I_4}.$$

    So $M$ is a square root of $xy-uv$. Here is the change of coordinates that turns Dirac's $t^2+x^2+y^2+z^2$ into this form:
    """)
    return


@app.cell
def s4_coords(f_dirac, mo, sp, u, v, x, y):
    t_, x_, y_, z_ = sp.symbols("t x y z")
    _coords = {x: t_ + sp.I * z_, y: t_ - sp.I * z_, u: sp.I * x_ + y_, v: sp.I * x_ - y_}
    _back = sp.expand(f_dirac.subs(_coords, simultaneous=True))
    mo.md(
        r"With $x\to t+iz,\; y\to t-iz,\; u\to ix+y,\; v\to ix-y$: "
        rf"$\quad xy-uv \;\longrightarrow\; \color{{#e4572e}}{{{sp.latex(_back)}}}$. "
        "Dirac's operator in disguise. (This is where complex numbers come in, as the video says.)"
    )
    return


@app.cell
def s4_states(A_dirac, B_dirac, sp):
    M_dirac = sp.BlockMatrix([[sp.zeros(2), A_dirac], [B_dirac, sp.zeros(2)]]).as_explicit()
    M2_dirac = sp.expand(M_dirac * M_dirac)

    def _short(e):
        return str(e).replace("*", "").replace(" ", "").replace("-", "−")

    _blocks = []
    for _r in range(4):
        for _c in range(4):
            _blocks.append("0" if M_dirac[_r, _c] == 0 else ("A" if _c >= 2 else "B"))
    _caps = ["Press ▶ to compute M² one entry at a time: row r of the left M times column c of the right M."]
    for _r in range(4):
        for _c in range(4):
            _terms = [(M_dirac[_r, _k], M_dirac[_k, _c]) for _k in range(4)]
            _nz = [f"({_short(_a)})·({_short(_b)})" for _a, _b in _terms if _a != 0 and _b != 0]
            _caps.append(
                f"M²<sub>{_r+1}{_c+1}</sub> = row {_r+1} · column {_c+1} = "
                + (" + ".join(_nz) if _nz else "every product has a 0 in it")
                + f" = <b style='color:#e4572e'>{_short(M2_dirac[_r, _c])}</b>"
                + (" ← on the diagonal: f!" if _r == _c else "")
            )
    dirac_states = dict(
        M=[_short(e) for e in M_dirac], M2=[_short(e) for e in M2_dirac], blocks=_blocks, captions=_caps
    )
    return M2_dirac, dirac_states


@app.cell
def stepper_class(anywidget, traitlets):
    class BlockStepper(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const COL = { A: "#2e86de", B: "#f18f01", P: "#e4572e", "0": "#999" };
      const root = document.createElement("div"); root.style.cssText = "font-family:sans-serif;display:flex;flex-direction:column;gap:8px";
      el.appendChild(root);
      const ctl = document.createElement("div"); ctl.style.cssText = "display:flex;gap:6px;align-items:center";
      const view = document.createElement("div"); view.style.cssText = "display:flex;gap:10px;align-items:center;flex-wrap:wrap";
      const cap = document.createElement("div"); cap.style.cssText = "min-height:3em;font-size:14px";
      root.append(ctl, view, cap);
      let timer = null;
      const n = () => model.get("captions").length;
      const stopPlay = () => { if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; } };
      const go = d => { stopPlay(); model.set("k", Math.max(0, Math.min(n() - 1, model.get("k") + d))); model.save_changes(); };
      for (const [t, d] of [["⏮", -1e9], ["◀", -1], ["▶", 1], ["⏭", 1e9]]) {
        const b = document.createElement("button"); b.textContent = t; b.onclick = () => go(d); ctl.appendChild(b);
      }
      const play = document.createElement("button"); play.textContent = "▶ play"; ctl.appendChild(play);
      play.onclick = () => {
        if (timer) { stopPlay(); model.save_changes(); return; }
        if (model.get("k") >= n() - 1) model.set("k", 0);
        play.textContent = "⏸ pause";
        timer = setInterval(() => {
          if (model.get("k") >= n() - 1) { stopPlay(); model.save_changes(); return; }
          model.set("k", model.get("k") + 1);
        }, 650);
      };
      function grid(label) {
        const w = document.createElement("div"); w.style.cssText = "display:flex;flex-direction:column;align-items:center;gap:2px";
        const g = document.createElement("div");
        g.style.cssText = "display:grid;grid-template-columns:repeat(4,76px);gap:3px;padding:5px;border-left:2px solid #888;border-right:2px solid #888;border-radius:5px";
        const cells = [];
        for (let q = 0; q < 16; q++) {
          const d = document.createElement("div");
          d.style.cssText = "height:30px;display:flex;align-items:center;justify-content:center;border-radius:4px;font-size:12px;font-family:monospace;font-weight:600";
          g.appendChild(d); cells.push(d);
        }
        const l = document.createElement("div"); l.innerHTML = label; l.style.fontSize = "13px";
        w.append(g, l); view.appendChild(w); return cells;
      }
      const sym = t => { const s = document.createElement("span"); s.textContent = t; s.style.fontSize = "20px"; view.appendChild(s); };
      const L = grid("M"); sym("×"); const R = grid("M"); sym("="); const P = grid("M²");
      function draw() {
        const k = model.get("k"), M = model.get("M"), blk = model.get("blocks"), M2 = model.get("M2");
        const idx = k - 1, r = idx >> 2, c = idx & 3;
        for (let q = 0; q < 16; q++) {
          const col = COL[blk[q]];
          for (const [cells, hl] of [[L, k > 0 && (q >> 2) === r], [R, k > 0 && (q & 3) === c]]) {
            cells[q].textContent = M[q]; cells[q].style.color = col;
            cells[q].style.background = hl ? "rgba(228,87,46,.2)" : "rgba(127,127,127,.07)";
          }
          const shown = q <= idx;
          P[q].textContent = shown ? M2[q] : "?";
          P[q].style.color = shown ? (M2[q] === "0" ? "#999" : COL.P) : "#bbb";
          P[q].style.background = q === idx ? "rgba(228,87,46,.28)" : "rgba(127,127,127,.07)";
        }
        cap.innerHTML = `<b>step ${k}/${n() - 1}</b>: ${model.get("captions")[k]}`;
      }
      model.on("change:k", draw); draw();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        M = traitlets.List(traitlets.Unicode()).tag(sync=True)
        M2 = traitlets.List(traitlets.Unicode()).tag(sync=True)
        blocks = traitlets.List(traitlets.Unicode()).tag(sync=True)
        captions = traitlets.List(traitlets.Unicode()).tag(sync=True)
        k = traitlets.Int(0).tag(sync=True)

    return (BlockStepper,)


@app.cell
def dirac_stepper(BlockStepper, dirac_states, mo):
    dirac_stepper = mo.ui.anywidget(BlockStepper(**dirac_states))
    dirac_stepper
    return (dirac_stepper,)


@app.cell
def stepper_readout(M2_dirac, dirac_stepper, f_dirac, mo, sp):
    _k = dirac_stepper.value["k"]
    _done = max(_k, 0)
    _diag = sum(1 for _q in range(_done) if _q // 4 == _q % 4)
    _ok = sp.simplify(M2_dirac - f_dirac * sp.eye(4)) == sp.zeros(4)
    mo.hstack([
        mo.stat(value=f"{_done}/16", label="entries of M² computed", bordered=True),
        mo.stat(value=f"{_diag}/4", label="diagonal entries = xy − uv", bordered=True),
        mo.stat(value=f"{_done - _diag}", label="off-diagonal entries (all 0)", bordered=True),
        mo.stat(value="✅" if _ok else "❌", label="M² = (xy − uv)·I₄ (sympy)", bordered=True),
    ], justify="start")
    return


@app.cell(hide_code=True)
def s4_try(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Step to entry (1,1). Row 1 of M meets column 1 of M through the <span style="color:#2e86de">A</span> block and the <span style="color:#f18f01">B</span> block. You get $(AB)_{11}=xy-uv$.
    2. Watch the off-diagonal entries. Each one is either "a 0 hits everything" or a cancellation like $xu-ux$.
    3. Brady asked: *"Have you changed sports mid-game?"* Eisenbud: *"Nature said, really you should be working with matrices to begin with."* Dirac's $M$ (the gamma matrices) is the foundation of the Dirac equation for the electron.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def s5_hook(mo):
    mo.md(r"""
    ## 5. Eisenbud's theorem: (almost) everything factors

    "f = 1 · f" factors anything, but you learn nothing from it. So **ban constants**: no numbers alone in any matrix entry.
    Then degree counting blocks some polynomials. A product of entries with no constant term has degree ≥ 2, so a polynomial with a **linear term** like $x$ or $x+y^2$ is out.

    > **Theorem (Eisenbud, 1980).** Any polynomial with **no constant and no linear term** has a matrix factorization $A\,B = B\,A = f\cdot I$ in which no entry of $A$ or $B$ has a constant term.

    "I can write down an algorithm which would produce the matrix factorization of whatever you want, or the machine will do it for me." Here is a simple version of such a machine:

    1. Every monomial has degree ≥ 2, so write $f = x_1 g_1 + x_2 g_2 + \dots + x_n g_n$, where each $g_i$ has no constant term.
    2. Start with the $1\times1$ factorization $[x_1]\,[g_1]$.
    3. Add one term $x\,g$ at a time with the block rule
    $A' = \begin{pmatrix}A & xI\\ -gI & B\end{pmatrix},\; B' = \begin{pmatrix}B & -xI\\ gI & A\end{pmatrix}$, which doubles the size. Section 3 is exactly this rule applied to $xy + u(-v)$.

    Type a polynomial, including Brady's $x^3+79y^2+34xz$:
    """)
    return


@app.cell
def eisenbud_algo(sp):
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

    _TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

    def parse_poly(text):
        return sp.expand(parse_expr(text, transformations=_TRANSFORMS))

    def eisenbud_mf(f, max_terms=5):
        """Return (A, B, pairs, None) with A*B = B*A = f*I and no constant entries, or (None, None, None, reason)."""
        f = sp.expand(f)
        gens = sorted(f.free_symbols, key=str)
        if f == 0 or not gens:
            return None, None, None, "f is a constant: nothing to factor"
        P = sp.Poly(f, *gens)
        for monom, coeff in P.terms():
            if sum(monom) == 0:
                return None, None, None, f"it has a **constant term** {coeff}"
            if sum(monom) == 1:
                return None, None, None, f"it has a **linear term** ${sp.latex(coeff * gens[monom.index(1)])}$; nothing without constants multiplies to give degree 1"
        groups = {}
        for monom, coeff in P.terms():
            i = next(k for k, e in enumerate(monom) if e > 0)
            term = coeff * sp.Mul(*[g**e for g, e in zip(gens, monom)])
            groups[gens[i]] = groups.get(gens[i], 0) + sp.expand(term / gens[i])
        pairs = [(xx, gg) for xx, gg in groups.items() if gg != 0]
        if len(pairs) > max_terms:
            return None, None, pairs, f"too many variables for a readable display ({len(pairs)} → {2**(len(pairs)-1)}×{2**(len(pairs)-1)} matrices)"
        A, B = sp.Matrix([[pairs[0][0]]]), sp.Matrix([[pairs[0][1]]])
        for xx, gg in pairs[1:]:
            I_n = sp.eye(A.shape[0])
            A, B = (
                sp.BlockMatrix([[A, xx * I_n], [-gg * I_n, B]]).as_explicit(),
                sp.BlockMatrix([[B, -xx * I_n], [gg * I_n, A]]).as_explicit(),
            )
        return A, B, pairs, None

    def has_constant_entry(Mx):
        return any(sp.expand(e).as_coeff_add()[0] != 0 for e in Mx)

    return eisenbud_mf, has_constant_entry, parse_poly


@app.cell
def mf_preset(mo):
    mf_preset = mo.ui.dropdown(
        options={
            "Brady's: x³ + 79y² + 34xz": "x^3 + 79y^2 + 34xz",
            "Dirac's: xy − uv": "xy - uv",
            "Sum of three squares: x² + y² + z²": "x^2 + y^2 + z^2",
            "A cusp: x³ − y²": "x^3 - y^2",
            "Has a linear term: x + y²": "x + y^2",
            "Has a constant: x² + 1": "x^2 + 1",
        },
        value="Brady's: x³ + 79y² + 34xz", label="Preset",
    )
    mf_preset
    return (mf_preset,)


@app.cell
def mf_text(mf_preset, mo):
    mf_text = mo.ui.text(value=mf_preset.value, label="f =", full_width=True)
    mf_text
    return (mf_text,)


@app.cell
def mf_readout(eisenbud_mf, has_constant_entry, mf_text, mo, parse_poly, sp):
    try:
        mf_f = parse_poly(mf_text.value)
        mf_A, mf_B, mf_pairs, mf_reason = eisenbud_mf(mf_f)
        _err = None
    except Exception as _e:
        mf_f, mf_A, mf_B, mf_pairs, mf_reason, _err = None, None, None, None, None, str(_e)

    if _err:
        _out = mo.callout(mo.md(f"Couldn't parse that: `{_err}`"), kind="danger")
    elif mf_A is None:
        _out = mo.callout(mo.md(f"**No (non-trivial) matrix factorization** for ${sp.latex(mf_f)}$: {mf_reason}."), kind="danger")
    else:
        _n = mf_A.shape[0]
        _AB_ok = sp.expand(mf_A * mf_B - mf_f * sp.eye(_n)) == sp.zeros(_n)
        _BA_ok = sp.expand(mf_B * mf_A - mf_f * sp.eye(_n)) == sp.zeros(_n)
        _split = " + ".join(rf"\color{{#2e86de}}{{{sp.latex(a)}}}\cdot\color{{#f18f01}}{{({sp.latex(g)})}}" for a, g in mf_pairs)
        _out = mo.vstack([
            mo.md(rf"**Step 1, split:** $\color{{#e4572e}}{{f}} = {_split}$"),
            mo.md(rf"**Result ({_n}×{_n}):** $$\color{{#2e86de}}{{A={sp.latex(mf_A)}}}\qquad \color{{#f18f01}}{{B={sp.latex(mf_B)}}}$$"),
            mo.hstack([
                mo.stat(value=f"{_n}×{_n}", label="matrix size", bordered=True),
                mo.stat(value="✅" if _AB_ok else "❌", label="A·B = f·I", bordered=True),
                mo.stat(value="✅" if _BA_ok else "❌", label="B·A = f·I", bordered=True),
                mo.stat(value="none" if not (has_constant_entry(mf_A) or has_constant_entry(mf_B)) else "some!", label="constant terms in entries", bordered=True),
            ], justify="start"),
        ])
    _out
    return


@app.cell(hide_code=True)
def s5_try(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Brady's $x^3+79y^2+34xz$ splits as $x(x^2+34z) + y(79y)$, giving a $2\times2$ factorization. "Just anything, doesn't matter."
    2. Type `x + y^2`. *"You can't do x plus anything."* Then type `x*z + y^2`. Multiplying the lone $x$ by another variable is enough, just as Brady guessed.
    3. Type a long polynomial like `xy + zw + ab + cd`. Each new variable group **doubles** the matrix size. Eisenbud's actual theorem, via periodic free resolutions, often gives smaller factorizations; this machine is the simple (Koszul-style) one.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def s6_hook(mo):
    mo.md(r"""
    ## 6. Another "matrix factorization": the recommender grid

    The video ends with real-world uses. Outside pure maths, *matrix factorization* most often means something else, and you'd be using it every time Netflix suggests a film.
    Take a **ratings matrix** $R$ (users × movies, mostly blank) and approximate it by a product of two **thin** matrices:

    $$\color{#e4572e}{R}\;\approx\;\color{#2e86de}{U}\,\color{#f18f01}{V^\top},\qquad U\in\mathbb{R}^{\text{users}\times k},\; V\in\mathbb{R}^{\text{movies}\times k}.$$

    | | Eisenbud / Dirac | Recommender |
    |---|---|---|
    | entries | polynomials | real numbers |
    | equality | **exact**: $AB=f\,I$ | **approximate**, fit only on the known cells |
    | goal | understand $f$ | **fill in the blanks**: predict unseen ratings |

    Click cells to rate. The factors $U$ and $V$ are refit (alternating least squares) every time you click.
    """)
    return


@app.cell
def rating_class(anywidget, traitlets):
    class RatingGrid(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const root = document.createElement("div"); root.style.fontFamily = "sans-serif"; el.appendChild(root);
      const tb = document.createElement("div"); tb.style.cssText = "display:flex;gap:6px;margin-bottom:6px;flex-wrap:wrap;align-items:center";
      const tbl = document.createElement("table"); tbl.style.cssText = "border-collapse:separate;border-spacing:3px;user-select:none";
      const help = document.createElement("div"); help.style.cssText = "font-size:13px;opacity:.75;margin-top:4px";
      help.textContent = "Click a cell: ★1 → ★5 → blank. Right-click or shift-click blanks it. Blank = unknown; the model must predict it.";
      root.append(tb, tbl, help);
      const lab = document.createElement("span"); lab.textContent = "Presets:"; lab.style.fontSize = "13px"; tb.appendChild(lab);
      const presets = model.get("presets");
      for (const name in presets) {
        const b = document.createElement("button"); b.textContent = name;
        b.onclick = () => { model.set("ratings", presets[name].map(r => [...r])); model.save_changes(); };
        tb.appendChild(b);
      }
      const shade = v => v === 0 ? "rgba(127,127,127,.08)" : `rgba(46,134,222,${0.12 + 0.16 * v})`;
      function set(i, j, v) {
        const R = model.get("ratings").map(r => [...r]); R[i][j] = v;
        model.set("ratings", R); model.save_changes();
      }
      function draw() {
        const R = model.get("ratings"), U = model.get("users"), I = model.get("items");
        tbl.innerHTML = "";
        const h = tbl.insertRow(); h.insertCell();
        I.forEach(t => { const c = h.insertCell(); c.textContent = t; c.style.cssText = "font-size:12px;text-align:center;padding:2px 4px"; });
        R.forEach((row, i) => {
          const tr = tbl.insertRow(); const nm = tr.insertCell(); nm.textContent = U[i];
          nm.style.cssText = "font-size:13px;padding-right:6px;text-align:right";
          row.forEach((v, j) => {
            const c = tr.insertCell(); c.textContent = v ? "★" + v : "·";
            c.style.cssText = `width:70px;height:34px;text-align:center;border-radius:6px;cursor:pointer;font-weight:600;background:${shade(v)}`;
            c.onclick = e => set(i, j, e.shiftKey ? 0 : (v + 1) % 6);
            c.oncontextmenu = e => { e.preventDefault(); set(i, j, 0); };
          });
        });
      }
      model.on("change:ratings", draw); draw();
    }
    export default { render };
    """
        ratings = traitlets.List(traitlets.List(traitlets.Int())).tag(sync=True)
        users = traitlets.List(traitlets.Unicode()).tag(sync=True)
        items = traitlets.List(traitlets.Unicode()).tag(sync=True)
        presets = traitlets.Dict().tag(sync=True)

    return (RatingGrid,)


@app.cell
def rating_widget(RatingGrid, mo):
    rec_users = ["Ana", "Ben", "Cara", "Dev", "Eli", "Fay"]
    rec_items = ["Alien", "Terminator", "Star Wars", "Notting Hill", "Titanic", "Love Actually"]
    _two_tribes = [
        [5, 5, 0, 1, 0, 1],
        [4, 0, 5, 0, 1, 2],
        [5, 4, 4, 0, 2, 0],
        [1, 0, 2, 5, 4, 0],
        [0, 1, 0, 4, 5, 5],
        [2, 0, 3, 3, 0, 3],
    ]
    rec_presets = {
        "Two tribes (sci-fi vs romance)": _two_tribes,
        "Everyone agrees": [[4, 0, 5, 2, 0, 3], [0, 4, 5, 0, 3, 3], [4, 4, 0, 2, 3, 0], [4, 0, 5, 0, 3, 3], [0, 4, 5, 2, 0, 3], [4, 4, 0, 2, 3, 0]],
        "Blank": [[0] * 6 for _ in range(6)],
    }
    rating_grid = mo.ui.anywidget(RatingGrid(ratings=[r[:] for r in _two_tribes], users=rec_users, items=rec_items, presets=rec_presets))
    rec_rank = mo.ui.slider(1, 4, value=1, label="rank k (number of hidden 'taste' dimensions)", show_value=True)
    rec_lambda = mo.ui.slider(0.0, 2.0, step=0.05, value=1.0, label="regularization λ", show_value=True)
    mo.vstack([rating_grid, mo.hstack([rec_rank, rec_lambda], justify="start")])
    return rating_grid, rec_items, rec_lambda, rec_rank, rec_users


@app.cell
def rec_fit(np, rating_grid, rec_lambda, rec_rank):
    def als_factorize(R, mask, k, lam, iters=60, seed=0):
        """Alternating least squares on observed cells: R - mu ≈ U V^T. Returns (U, V, mu)."""
        _rng = np.random.default_rng(seed)
        n_u, n_i = R.shape
        mu = R[mask].mean() if mask.any() else 3.0
        X = np.where(mask, R - mu, 0.0)
        U = 0.3 * _rng.standard_normal((n_u, k))
        V = 0.3 * _rng.standard_normal((n_i, k))
        eye = lam * np.eye(k) + 1e-9 * np.eye(k)
        for _ in range(iters):
            for a in range(n_u):
                m = mask[a]
                U[a] = np.linalg.solve(V[m].T @ V[m] + eye, V[m].T @ X[a, m])
            for b in range(n_i):
                m = mask[:, b]
                V[b] = np.linalg.solve(U[m].T @ U[m] + eye, U[m].T @ X[m, b])
        return U, V, mu

    rec_R = np.array(rating_grid.value["ratings"], dtype=float)
    rec_mask = rec_R > 0
    rec_U, rec_V, rec_mu = als_factorize(rec_R, rec_mask, rec_rank.value, rec_lambda.value)
    rec_pred = np.clip(rec_mu + rec_U @ rec_V.T, 1, 5)
    return rec_R, rec_U, rec_V, rec_mask, rec_pred


@app.cell
def rec_heatmap(
    alt,
    mo,
    np,
    pd,
    rec_R,
    rec_items,
    rec_mask,
    rec_pred,
    rec_rank,
    rec_users,
):
    _df = pd.DataFrame(
        [dict(user=rec_users[a], movie=rec_items[b], predicted=float(rec_pred[a, b]),
              observed=int(rec_R[a, b]), kind="your rating" if rec_mask[a, b] else "prediction")
         for a in range(len(rec_users)) for b in range(len(rec_items))]
    )
    _base = alt.Chart(_df).encode(
        x=alt.X("movie:N", sort=rec_items, title=None, axis=alt.Axis(labelAngle=-30)),
        y=alt.Y("user:N", sort=rec_users, title=None),
    )
    _heat = _base.mark_rect(cornerRadius=4).encode(
        color=alt.Color("predicted:Q", scale=alt.Scale(domain=[1, 5], scheme="orangered"), title="U·Vᵀ"),
        stroke=alt.condition("datum.kind == 'your rating'", alt.value("#2e86de"), alt.value(None)),
        strokeWidth=alt.value(3),
        tooltip=["user", "movie", alt.Tooltip("predicted:Q", format=".2f"), "observed", "kind"],
    )
    _txt = _base.mark_text(fontSize=13, fontWeight="bold").encode(
        text=alt.Text("predicted:Q", format=".1f"),
        color=alt.condition("datum.kind == 'your rating'", alt.value("#1b4f8a"), alt.value("#222")),
    )
    _rmse = float(np.sqrt(np.mean((rec_pred[rec_mask] - rec_R[rec_mask]) ** 2))) if rec_mask.any() else float("nan")
    mo.vstack([
        mo.md("**Reconstructed grid** $\\mu + UV^\\top$. Blue outline = a cell you rated; all others are **predictions**."),
        (_heat + _txt).properties(width="container", height=260),
        mo.hstack([
            mo.stat(value=f"{int(rec_mask.sum())}/36", label="known ratings", bordered=True),
            mo.stat(value=f"{_rmse:.2f}", label="RMSE on known cells", bordered=True),
            mo.stat(value=f"{rec_rank.value * 12} vs 36", label="numbers in U,V vs in R", bordered=True),
        ], justify="start"),
    ])
    return


@app.cell
def rec_factors(alt, mo, pd, rec_U, rec_V, rec_items, rec_rank, rec_users):
    _fu = pd.DataFrame([dict(name=rec_users[a], factor=f"taste {d+1}", value=float(rec_U[a, d]), who="user (U)")
                        for a in range(len(rec_users)) for d in range(rec_rank.value)])
    _fv = pd.DataFrame([dict(name=rec_items[b], factor=f"taste {d+1}", value=float(rec_V[b, d]), who="movie (V)")
                        for b in range(len(rec_items)) for d in range(rec_rank.value)])
    def _bars(df, color, order, title):
        return alt.Chart(df).mark_bar(color=color).encode(
            y=alt.Y("name:N", sort=order, title=None),
            x=alt.X("value:Q", title=None),
            column=alt.Column("factor:N", title=None),
            tooltip=["name", "factor", alt.Tooltip("value:Q", format=".2f")],
        ).properties(width=110, height=150, title=title)
    mo.hstack([
        _bars(_fu, "#2e86de", rec_users, "U: each user's taste"),
        _bars(_fv, "#f18f01", rec_items, "V: each movie's traits"),
    ], justify="start", wrap=True)
    return


@app.cell(hide_code=True)
def s6_try(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. With "Two tribes" at the default **k = 1**, taste 1 separates the sci-fi fans from the romance fans, with opposite signs on the two movie groups. One hidden number per person explains most of the grid.
    2. At k = 1, give **Fay** ★5 for *Alien*, then watch her predicted *Terminator* and *Star Wars* rise. The factors moved her toward the sci-fi tribe.
    3. Push **k to 4** with λ = 0. The known cells are fit almost perfectly (RMSE → 0) but the predictions get wild. That is overfitting. Unlike Eisenbud's exact factorizations, a recommender must not reproduce its data *too* well.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def sandbox_hook(mo):
    mo.md(r"""
    ## 7. Sandbox: every matrix is half of a factorization

    The pattern behind Section 3: for **any** square matrix $A$, the adjugate $\operatorname{adj}(A)$ satisfies

    $$A\cdot\operatorname{adj}(A)=\operatorname{adj}(A)\cdot A=\det(A)\,I .$$

    So **any** matrix of polynomials gives a matrix factorization of its determinant. Fill in a $2\times2$ or $3\times3$ matrix and see which polynomial you have factored.
    Matrices whose entries are all linear (as in Dirac's case) give the most interesting results.
    """)
    return


@app.cell
def sb_size(mo):
    sb_size = mo.ui.radio(["2×2", "3×3"], value="2×2", label="size", inline=True)
    sb_size
    return (sb_size,)


@app.cell
def sb_entries(mo, sb_size):
    _defaults = {
        "2×2": ["x", "u", "v", "y"],
        "3×3": ["0", "x", "y", "-x", "0", "z", "-y", "-z", "w"],
    }[sb_size.value]
    sb_entries = mo.ui.array([mo.ui.text(value=_d, full_width=True) for _d in _defaults])
    _n = 2 if sb_size.value == "2×2" else 3
    mo.vstack([
        mo.md("**Your matrix A** (any polynomial in each box):"),
        mo.vstack([mo.hstack([sb_entries[_r * _n + _c] for _c in range(_n)], justify="start", widths=[1] * _n)
                   for _r in range(_n)], align="start"),
    ])
    return (sb_entries,)


@app.cell
def sb_readout(has_constant_entry, mo, parse_poly, sb_entries, sb_size, sp):
    _n = 2 if sb_size.value == "2×2" else 3
    try:
        _A = sp.Matrix(_n, _n, [parse_poly(_t or "0") for _t in sb_entries.value])
        _adj = _A.adjugate()
        _det = sp.factor(_A.det())
        _ok = sp.expand(_A * _adj - _A.det() * sp.eye(_n)) == sp.zeros(_n)
        _const = has_constant_entry(_A) or has_constant_entry(_adj)
        _verdict = (
            "f = 0: the factorization is degenerate. Try a different matrix."
            if sp.expand(_A.det()) == 0 else
            ("Some entry has a constant term, so this is the boring kind (closer to f = 1·f)."
             if _const else "No constant terms anywhere: a genuine matrix factorization in Eisenbud's sense.")
        )
        _out = mo.vstack([
            mo.md(rf"$$\color{{#2e86de}}{{{sp.latex(_A)}}}\;\color{{#f18f01}}{{{sp.latex(_adj)}}}"
                  rf"= \color{{#e4572e}}{{\left({sp.latex(_det)}\right)}}\cdot I_{_n}$$"),
            mo.hstack([
                mo.stat(value="✅" if _ok else "❌", label="A·adj(A) = det(A)·I", bordered=True),
                mo.stat(value=str(sp.Poly(_A.det(), *sorted(_A.free_symbols, key=str)).total_degree()) if _A.free_symbols else "0",
                        label="degree of f = det A", bordered=True),
            ], justify="start"),
            mo.callout(mo.md(_verdict), kind="success" if not _const and sp.expand(_A.det()) != 0 else "neutral"),
        ])
    except Exception as _e:
        _out = mo.callout(mo.md(f"Couldn't parse: `{_e}`"), kind="danger")
    _out
    return


@app.cell(hide_code=True)
def sandbox_try(mo):
    mo.callout(mo.md(r"""
    **Challenges**

    1. Find a $2\times2$ matrix of *linear* entries whose determinant is $x^2+y^2$. (Hint: use $i$, written `I`.)
    2. The default $3\times3$ is skew-symmetric in its top-left block. What happens when you set $w=0$? Why is the determinant of an odd-sized skew-symmetric matrix always 0?
    3. Make a $2\times2$ with determinant $x^3-y^2$ (the cusp) and no constant entries. Compare it to the machine's answer in Section 5.
    """), kind="info")
    return


@app.cell(hide_code=True)
def quiz_hook(mo):
    mo.md("""
    ## 8. Quiz: check yourself
    """)
    return


@app.cell
def quiz(mo):
    quiz_items = {
        "1. x² + 3 factors over the rationals.": ("False", "Needs √−3 = i√3, so you have to enlarge the domain to complex numbers."),
        "2. For 2×2 matrices, AB = BA always.": ("False", "Rows of the first times columns of the second; order matters (Section 2)."),
        "3. t² + x² + y² + z² factors into polynomials once complex numbers are allowed.": ("False", "That's why Dirac had to jump to matrices."),
        "4. [[x, u], [v, y]]·[[y, −u], [−v, x]] = (xy − uv)·I.": ("True", "Dirac's factorization; B is the adjugate of A."),
        "5. x + y² has a matrix factorization with no constant entries.": ("False", "A linear term can't come from products of entries with no constants (degree ≥ 2)."),
        "6. x³ + 79y² + 34xz has a matrix factorization with no constant entries.": ("True", "No constant or linear term, so Eisenbud's theorem applies: x(x² + 34z) + y(79y)."),
        "7. The square of [[0, A], [B, 0]] is [[AB, 0], [0, BA]].": ("True", "That's how a factorization becomes a square root (Section 4)."),
        "8. A recommender's U·Vᵀ must reproduce the known ratings exactly.": ("False", "It's an approximation; fitting known cells perfectly usually means overfitting."),
    }
    quiz = mo.ui.dictionary({_q: mo.ui.radio(["True", "False"], inline=True) for _q in quiz_items})
    quiz_reveal = mo.ui.switch(label="Reveal answers")
    mo.vstack([quiz, quiz_reveal])
    return quiz, quiz_items, quiz_reveal


@app.cell
def quiz_check(mo, quiz, quiz_items, quiz_reveal):
    _answered = {q: a for q, a in quiz.value.items() if a is not None}
    _score = sum(quiz_items[q][0] == a for q, a in _answered.items())
    mo.stop(not quiz_reveal.value, mo.md(f"_{len(_answered)}/{len(quiz_items)} answered. Flip the switch to check._"))
    _rows = "\n".join(
        f"| {'✅' if quiz.value[q] == ans else ('❌' if quiz.value[q] else '·')} | {q} | **{ans}** | {why} |"
        for q, (ans, why) in quiz_items.items()
    )
    mo.vstack([
        mo.stat(value=f"{_score} / {len(quiz_items)}", label="score", bordered=True),
        mo.md("| | statement | answer | why |\n|---|---|---|---|\n" + _rows),
    ])
    return


@app.cell(hide_code=True)
def recap(mo):
    mo.md(r"""
    ## Recap

    | Idea | In one line | Where you touched it |
    |---|---|---|
    | Enlarge the domain | Stuck equations get unstuck with new numbers: fractions, negatives, $i$ | §1 factor-in-a-domain picker |
    | Matrix multiplication | Entry $(r,c)$ = row $r$ · column $c$; generally $AB\neq BA$ | §2 drag-and-hover multiplier |
    | Matrix factorization | $AB = BA = f\cdot I$: the identity "personifies" the polynomial | §3 Dirac's $xy-uv$ with sliders |
    | Square root | $M=\begin{pmatrix}0&A\\B&0\end{pmatrix}$ gives $M^2=f\cdot I$ | §4 4×4 stepper |
    | Eisenbud's theorem | No constant and no linear term ⇒ a matrix factorization without constants exists | §5 factorization machine |
    | Recommender MF | $R\approx UV^\top$: approximate, low-rank, predicts blanks | §6 click-to-rate grid |
    | Adjugate trick | $A\cdot\operatorname{adj}A=\det A\cdot I$ for any square $A$ | §7 sandbox |

    *Further reading:* D. Eisenbud, "Homological algebra on a complete intersection, with an application to group representations", *Trans. AMS* 260 (1980). Matrix factorizations later became the boundary conditions (D-branes) of Landau–Ginzburg models in string theory (Kapustin–Li, 2003), which is why the paper is so widely cited.
    """)
    return


if __name__ == "__main__":
    app.run()

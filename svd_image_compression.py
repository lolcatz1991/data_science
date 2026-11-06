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
    import zlib
    import struct

    return alt, anywidget, mo, np, pd, struct, traitlets, zlib


@app.cell(hide_code=True)
def title(mo):
    mo.vstack([
        mo.md(r"""
    # Singular Value Decomposition (SVD) & Image Compression
    *An explorable companion to Luis Serrano's video. **Every matrix is a rotation, a stretch, and another rotation.** Drop the smallest stretches and you get compression.*
    """),
        mo.Html('<iframe width="560" height="315" src="https://www.youtube.com/embed/DG7YTlGnCEo" title="SVD and Image Compression" frameborder="0" allowfullscreen></iframe>'),
    ])
    return


@app.cell(hide_code=True)
def roadmap(mo):
    mo.md(r"""
    **Roadmap (follows the video's order)**

    | # | Section | You get to... |
    |---|---|---|
    | 1 | Transformations & the puzzle | drag the columns of a 2×2 matrix and watch the coloured unit circle become an ellipse |
    | 2 | $A = U\Sigma V^\dagger$ | play the movie *rotate → stretch → rotate* bead by bead |
    | 3 | Dimensionality reduction | squash the small singular value to 0 and watch the ellipse become a line |
    | 4 | Sums of rank-1 matrices | add outer products $\sigma_i u_i v_i^\dagger$ one at a time |
    | 5 | The heart | paint a pixel image and decompose it into components |
    | 6 | Image compression | slide the rank $k$ on a picture; read error vs storage |

    **Colour language (used everywhere):** <span style="color:#2e86ab">**blue** = first basis direction $e_1$ / first column</span>,
    <span style="color:#e4572e">**red** = second direction $e_2$ / second column</span>,
    <span style="color:#f18f01">**orange** = singular values & singular directions (the key quantity)</span>.
    Beads on circles are rainbow-coloured by angle so you can follow *each point*, not just the shape (the "harder puzzle").
    In heatmaps: orange = positive, blue = negative. In images: dark = 1 (ink), light = 0, as in the video.
    """)
    return


@app.cell
def helpers(alt, mo, np, pd, struct, zlib):
    def to_png(img, scale=1, ink=False):
        """Encode a 2-D array in [0,1] as grayscale PNG bytes (no PIL needed).
        ink=True: 1 -> black (the video's convention for the heart)."""
        _a = np.clip(np.asarray(img, dtype=float), 0, 1)
        if ink:
            _a = 1 - _a
        _a = (_a * 255).round().astype(np.uint8)
        if scale > 1:
            _a = np.kron(_a, np.ones((scale, scale), dtype=np.uint8))
        _h, _w = _a.shape
        _raw = b"".join(b"\x00" + _a[_r].tobytes() for _r in range(_h))
        def _chunk(tag, data):
            return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        return (b"\x89PNG\r\n\x1a\n" + _chunk(b"IHDR", struct.pack(">IIBBBBB", _w, _h, 8, 0, 0, 0, 0))
                + _chunk(b"IDAT", zlib.compress(_raw, 9)) + _chunk(b"IEND", b""))

    def show_img(img, width=240, ink=False, caption=None):
        _a = np.asarray(img)
        _s = max(1, int(np.ceil(width / _a.shape[1])))
        return mo.image(to_png(_a, scale=_s, ink=ink), width=width, caption=caption,
                        style={"image-rendering": "pixelated", "border": "1px solid #ccc"})

    def mat_tex(M, d=2):
        _rows = [" & ".join(f"{v:.{d}f}" for v in row) for row in np.atleast_2d(M)]
        return r"\begin{bmatrix}" + r" \\ ".join(_rows) + r"\end{bmatrix}"

    def rotation_form(M):
        """Write a 2x2 M as R(theta_U) diag(s1, s2) R(theta_V) with both outer factors proper rotations.
        s2 may come out negative: that is a reflection (video: 'stretch followed by a reflection')."""
        _U, _S, _Vt = np.linalg.svd(np.asarray(M, dtype=float))
        _s = _S.copy()
        if np.linalg.det(_Vt) < 0:
            _Vt[1] *= -1
            _U[:, 1] *= -1
        if np.linalg.det(_U) < 0:
            _U[:, 1] *= -1
            _s[1] *= -1
        _thV, _thU = float(np.arctan2(_Vt[1, 0], _Vt[0, 0])), float(np.arctan2(_U[1, 0], _U[0, 0]))
        if abs(_thV) > np.pi / 2:  # R(pi) = -I commutes with diag: shift both angles by pi for the smallest first rotation
            _thV -= np.sign(_thV) * np.pi
            _thU = float((_thU + 2 * np.pi) % (2 * np.pi) - np.pi)
        return _thV, _s, _thU

    def heat(M, title, w=150):
        _M = np.asarray(M, dtype=float)
        _df = pd.DataFrame([(i, j, _M[i, j]) for i in range(_M.shape[0]) for j in range(_M.shape[1])],
                           columns=["row", "col", "value"])
        _m = max(float(np.abs(_M).max()), 1e-9)
        _base = alt.Chart(_df).encode(x=alt.X("col:O", axis=None), y=alt.Y("row:O", axis=None))
        _r = _base.mark_rect().encode(color=alt.Color("value:Q", scale=alt.Scale(scheme="blueorange", domain=[-_m, _m]), legend=None),
                                      tooltip=["row", "col", alt.Tooltip("value:Q", format=".2f")])
        _t = _base.mark_text(fontSize=11).encode(text=alt.Text("value:Q", format=".1f"))
        return (_r + _t).properties(width=w, height=w, title=title)

    return heat, mat_tex, rotation_form, show_img


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Transformations: every matrix turns the circle into an ellipse

    The video starts with a puzzle: turn a circle into a tilted ellipse using only **rotations** and
    **horizontal / vertical scalings** (stretches and compressions). Then the *harder* puzzle: do it so every
    coloured bead lands in the right spot.

    A 2×2 matrix is exactly such a point-to-point map. With the video's

    $$A=\begin{bmatrix}3&0\\4&5\end{bmatrix},\qquad \begin{bmatrix}p\\q\end{bmatrix}\mapsto\begin{bmatrix}3p+0q\\4p+5q\end{bmatrix},$$

    the blue point $(1,0)$ goes to $(3,4)$ (the **first column**), the red point $(0,1)$ goes to $(0,5)$ (the **second column**).
    The whole unit circle lands on an ellipse centred at the origin.

    **Drag the blue and red handles on the right**: they *are* the columns of $A$. The orange arrows are the ellipse's
    long and short half-axes, of lengths $\sigma_1 \ge \sigma_2$. The dashed orange arrows on the left show which input
    directions get sent there.
    """)
    return


@app.cell
def MatrixDrag_class(anywidget, traitlets):
    class MatrixDrag(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const NS = "http://www.w3.org/2000/svg", W = 720, H = 360;
      const bar = document.createElement("div"); bar.style.marginBottom = "6px"; el.appendChild(bar);
      const presets = { "Video A = [[3,0],[4,5]]": [3,0,4,5], "Near-singular": [1.8,1.2,4.4,4.6],
        "Rotation 30°": [0.866,-0.5,0.5,0.866], "Pure scaling": [3,0,0,1.5], "Shear": [1,2,0,1],
        "Reflection": [0,1,1,0], "Identity": [1,0,0,1] };
      for (const k in presets) {
        const b = document.createElement("button"); b.textContent = k; b.style.margin = "0 4px 4px 0";
        b.onclick = () => { const [a,bb,c,d] = presets[k]; model.set("a",a); model.set("b",bb); model.set("c",c); model.set("d",d); model.save_changes(); draw(); };
        bar.appendChild(b);
      }
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none" });
      el.appendChild(svg);
      const mk = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); svg.appendChild(e); return e; };
      const toSvg = (evt) => { const p = svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const L = { cx: 180, cy: 180, s: 110 }, R = { cx: 540, cy: 180, s: 20 };
      const tL = (x, y) => [L.cx + x * L.s, L.cy - y * L.s], tR = (x, y) => [R.cx + x * R.s, R.cy - y * R.s];
      const arrow = (P, x, y, col, w, dash) => { const [x0, y0] = P(0, 0), [x1, y1] = P(x, y);
        mk("line", { x1: x0, y1: y0, x2: x1, y2: y1, stroke: col, "stroke-width": w, "stroke-dasharray": dash || "" });
        mk("circle", { cx: x1, cy: y1, r: w + 1, fill: col }); };
      const N = 36; let drag = -1;
      function svd(a, b, c, d) {
        const p = a*a + c*c, q = a*b + c*d, r = b*b + d*d, tr = p + r, det = p*r - q*q;
        const disc = Math.sqrt(Math.max(0, tr*tr/4 - det)); const l1 = tr/2 + disc, l2 = Math.max(0, tr/2 - disc);
        let v1 = Math.abs(q) > 1e-9 ? [l1 - r, q] : (p >= r ? [1, 0] : [0, 1]);
        const n = Math.hypot(v1[0], v1[1]); v1 = [v1[0]/n, v1[1]/n]; const v2 = [-v1[1], v1[0]];
        return { s1: Math.sqrt(l1), s2: Math.sqrt(l2), v1, v2 };
      }
      function draw() {
        const a = model.get("a"), b = model.get("b"), c = model.get("c"), d = model.get("d");
        svg.innerHTML = "";
        mk("rect", { x: 2, y: 2, width: 356, height: 356, fill: "rgba(128,128,128,0.06)", rx: 8 });
        mk("rect", { x: 362, y: 2, width: 356, height: 356, fill: "rgba(128,128,128,0.06)", rx: 8 });
        for (let g = -8; g <= 8; g++) { const [gx] = tR(g, 0), [, gy] = tR(0, g);
          mk("line", { x1: gx, y1: 22, x2: gx, y2: 338, stroke: "rgba(128,128,128,0.15)" });
          mk("line", { x1: 382, y1: gy, x2: 698, y2: gy, stroke: "rgba(128,128,128,0.15)" }); }
        for (const [P, ext] of [[tL, 1.5], [tR, 8]]) {
          const [x0, y0] = P(-ext, 0), [x1] = P(ext, 0), [, yt] = P(0, ext), [, yb] = P(0, -ext);
          mk("line", { x1: x0, y1: y0, x2: x1, y2: y0, stroke: "#999" });
          const [xc] = P(0, 0); mk("line", { x1: xc, y1: yt, x2: xc, y2: yb, stroke: "#999" }); }
        const lab = (x, y, t, col, size) => { const e = mk("text", { x, y, fill: col || "currentColor", "font-size": size || 14, "font-family": "sans-serif" }); e.textContent = t; };
        lab(14, 24, "before: unit circle", null, 13); lab(374, 24, "after: A × circle", null, 13);
        const S = svd(a, b, c, d);
        // left: circle, beads, singular input directions
        let pts = []; for (let i = 0; i <= 120; i++) { const t = 2*Math.PI*i/120; pts.push(tL(Math.cos(t), Math.sin(t)).join(",")); }
        mk("polyline", { points: pts.join(" "), fill: "none", stroke: "#888", "stroke-width": 1.5 });
        arrow(tL, S.v1[0], S.v1[1], "#f18f01", 2, "5 4"); arrow(tL, S.v2[0], S.v2[1], "#f18f01", 1.5, "2 4");
        for (let i = 0; i < N; i++) { const t = 2*Math.PI*i/N; const [x, y] = tL(Math.cos(t), Math.sin(t));
          mk("circle", { cx: x, cy: y, r: 5, fill: `hsl(${360*i/N},80%,55%)` }); }
        arrow(tL, 1, 0, "#2e86ab", 3); arrow(tL, 0, 1, "#e4572e", 3);
        // right: ellipse, beads, singular axes, columns
        pts = []; for (let i = 0; i <= 120; i++) { const t = 2*Math.PI*i/120, x = Math.cos(t), y = Math.sin(t); pts.push(tR(a*x + b*y, c*x + d*y).join(",")); }
        mk("polyline", { points: pts.join(" "), fill: "rgba(241,143,1,0.08)", stroke: "#888", "stroke-width": 1.5 });
        for (let i = 0; i < N; i++) { const t = 2*Math.PI*i/N, x = Math.cos(t), y = Math.sin(t); const [px, py] = tR(a*x + b*y, c*x + d*y);
          mk("circle", { cx: px, cy: py, r: 4, fill: `hsl(${360*i/N},80%,55%)` }); }
        const Av = (v) => [a*v[0] + b*v[1], c*v[0] + d*v[1]];
        const A1 = Av(S.v1), A2 = Av(S.v2);
        arrow(tR, A1[0], A1[1], "#f18f01", 4); arrow(tR, A2[0], A2[1], "#f18f01", 2.5);
        arrow(tR, a, c, "#2e86ab", 2.5); arrow(tR, b, d, "#e4572e", 2.5);
        [[a, c, "#2e86ab", 0], [b, d, "#e4572e", 1]].forEach(([x, y, col, k]) => {
          const [hx, hy] = tR(x, y);
          const h = mk("circle", { cx: hx, cy: hy, r: 10, fill: col, stroke: "white", "stroke-width": 2, "fill-opacity": 0.85 });
          h.style.cursor = "grab";
          h.addEventListener("pointerdown", e => { drag = k; svg.setPointerCapture(e.pointerId); }); });
        const f = v => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2);
        lab(374, 344, `A = [[${f(a)}, ${f(b)}], [${f(c)}, ${f(d)}]]`, null, 13);
        lab(560, 24, `σ1 = ${S.s1.toFixed(2)}  σ2 = ${S.s2.toFixed(2)}`, "#f18f01", 13);
        lab(560, 42, `|det| = σ1·σ2 = ${(S.s1*S.s2).toFixed(2)}`, null, 12);
      }
      svg.addEventListener("pointermove", e => {
        if (drag < 0) return;
        const p = toSvg(e); let x = (p.x - R.cx) / R.s, y = (R.cy - p.y) / R.s;
        x = Math.max(-8, Math.min(8, Math.round(x * 10) / 10)); y = Math.max(-8, Math.min(8, Math.round(y * 10) / 10));
        if (drag === 0) { model.set("a", x); model.set("c", y); } else { model.set("b", x); model.set("d", y); }
        draw();
      });
      const stop = () => { if (drag >= 0) { drag = -1; model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointerleave", stop);
      for (const t of ["a", "b", "c", "d"]) model.on("change:" + t, draw);
      draw();
    }
    export default { render };
    """
        a = traitlets.Float(3.0).tag(sync=True)
        b = traitlets.Float(0.0).tag(sync=True)
        c = traitlets.Float(4.0).tag(sync=True)
        d = traitlets.Float(5.0).tag(sync=True)

    return (MatrixDrag,)


@app.cell
def matrix_widget(MatrixDrag, mo):
    matrix_widget = mo.ui.anywidget(MatrixDrag(a=3.0, b=0.0, c=4.0, d=5.0))
    matrix_widget
    return (matrix_widget,)


@app.cell
def matrix_readout(mat_tex, matrix_widget, mo, np):
    A_drag = np.array([[matrix_widget.value["a"], matrix_widget.value["b"]],
                       [matrix_widget.value["c"], matrix_widget.value["d"]]])
    _U, _S, _Vt = np.linalg.svd(A_drag)
    mo.vstack([
        mo.hstack([
            mo.stat(f"{_S[0]:.3f}", label="σ₁ (long half-axis)"),
            mo.stat(f"{_S[1]:.3f}", label="σ₂ (short half-axis)"),
            mo.stat(f"{np.linalg.det(A_drag):.3f}", label="det A  (= ±σ₁σ₂)"),
            mo.stat(f"{np.linalg.matrix_rank(A_drag, tol=1e-9)}", label="rank"),
        ], justify="start"),
        mo.md(rf"""
    numpy's `np.linalg.svd(A)` for the matrix you dragged:
    $$A={mat_tex(A_drag)} = \underbrace{{{mat_tex(_U, 3)}}}_{{U}}\;\underbrace{{{mat_tex(np.diag(_S), 3)}}}_{{\Sigma}}\;\underbrace{{{mat_tex(_Vt, 3)}}}_{{V^\dagger}}$$
    Check: $A\,e_1 = {mat_tex(A_drag[:, :1])}$ is the blue handle, $A\,e_2 = {mat_tex(A_drag[:, 1:])}$ is the red one.
    """),
    ])
    return (A_drag,)


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Click **Video A**: the singular values should read $3\sqrt5 \approx 6.71$ and $\sqrt5\approx 2.24$, exactly as in the video.
    2. Click **Rotation 30°**: the ellipse is still a circle, so $\sigma_1=\sigma_2=1$. Rotations don't stretch anything.
    3. Drag the red handle on top of the blue one. The ellipse collapses into a **line**, $\sigma_2 \to 0$, and det $\to 0$ (rank 1). Remember this for Section 3.
    4. Watch the beads: the *shape* is set by $\sigma_1,\sigma_2$, but *where each colour lands* is the harder puzzle. Section 2 solves it.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · The main equation: $A = U\,\Sigma\,V^\dagger$

    Two special kinds of matrix from the video:

    $$\underbrace{\begin{bmatrix}\cos\theta&-\sin\theta\\ \sin\theta&\cos\theta\end{bmatrix}}_{\text{rotation by }\theta}
    \qquad
    \underbrace{\begin{bmatrix}\sigma_1&0\\0&\sigma_2\end{bmatrix}}_{\text{scale horizontally by }\sigma_1,\ \text{vertically by }\sigma_2}$$

    **The moral of the video:** any matrix is *rotation → horizontal/vertical scaling → rotation*:

    $$A = U\,\Sigma\,V^\dagger .$$

    Read it **right to left**, because the vector is multiplied by the right-most matrix first:
    $V^\dagger$ rotates the circle so the right beads sit on the axes, $\Sigma$ stretches along the axes only,
    and $U$ rotates the result into place. ($V^\dagger$ is the conjugate transpose; for real matrices it is just $V^T$.)

    The movie below uses the matrix you set in Section 1. Press **▶ play**, or scrub the slider.
    """)
    return


@app.cell
def SVDMovie_class(anywidget, traitlets):
    class SVDMovie(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const NS = "http://www.w3.org/2000/svg", W = 420, H = 420, C = 210;
      const ctl = document.createElement("div"); ctl.style.marginBottom = "6px"; el.appendChild(ctl);
      const btn = (t, fn) => { const b = document.createElement("button"); b.textContent = t; b.style.marginRight = "4px"; b.onclick = fn; ctl.appendChild(b); return b; };
      const play = btn("▶ play", () => animate());
      const stages = ["start: unit circle", "1. rotate by V† ", "2. scale by Σ", "3. rotate by U  → A·circle"];
      [0, 1, 2, 3].forEach(k => btn(String(k), () => { model.set("t", k); model.save_changes(); draw(); }));
      const sl = document.createElement("input"); Object.assign(sl, { type: "range", min: 0, max: 3, step: 0.01 }); sl.style.width = "200px"; sl.style.verticalAlign = "middle";
      ctl.appendChild(sl);
      sl.oninput = () => { model.set("t", +sl.value); draw(); };
      sl.onchange = () => model.save_changes();
      const svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none" }); el.appendChild(svg);
      const mk = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); svg.appendChild(e); return e; };
      let raf = null;
      function M(t) {
        const tv = model.get("thV"), tu = model.get("thU"), s1 = model.get("s1"), s2 = model.get("s2");
        const f1 = Math.min(1, Math.max(0, t)), f2 = Math.min(1, Math.max(0, t - 1)), f3 = Math.min(1, Math.max(0, t - 2));
        const R = th => [[Math.cos(th), -Math.sin(th)], [Math.sin(th), Math.cos(th)]];
        const mul = (X, Y) => [[X[0][0]*Y[0][0] + X[0][1]*Y[1][0], X[0][0]*Y[0][1] + X[0][1]*Y[1][1]],
                               [X[1][0]*Y[0][0] + X[1][1]*Y[1][0], X[1][0]*Y[0][1] + X[1][1]*Y[1][1]]];
        const D = [[1 + (s1 - 1)*f2, 0], [0, 1 + (s2 - 1)*f2]];
        return mul(R(tu*f3), mul(D, R(tv*f1)));
      }
      function draw() {
        const t = model.get("t"); sl.value = t;
        const ext = Math.max(1.2, Math.abs(model.get("s1")) * 1.15), s = 190 / ext;
        const P = (x, y) => [C + x*s, C - y*s];
        svg.innerHTML = "";
        mk("rect", { x: 0, y: 0, width: W, height: H, fill: "rgba(128,128,128,0.06)", rx: 8 });
        mk("line", { x1: 10, y1: C, x2: W - 10, y2: C, stroke: "#999" }); mk("line", { x1: C, y1: 10, x2: C, y2: H - 10, stroke: "#999" });
        const curve = (Mx, col, w, dash) => { const pts = [];
          for (let i = 0; i <= 120; i++) { const a = 2*Math.PI*i/120, x = Math.cos(a), y = Math.sin(a);
            pts.push(P(Mx[0][0]*x + Mx[0][1]*y, Mx[1][0]*x + Mx[1][1]*y).join(",")); }
          mk("polyline", { points: pts.join(" "), fill: "none", stroke: col, "stroke-width": w, "stroke-dasharray": dash || "" }); };
        curve([[1, 0], [0, 1]], "#aaa", 1, "3 3");
        curve(M(3), "#f18f01", 1.5, "6 4");
        const Mt = M(t); curve(Mt, "#666", 2);
        const N = 36;
        for (let i = 0; i < N; i++) { const a = 2*Math.PI*i/N, x = Math.cos(a), y = Math.sin(a);
          const [px, py] = P(Mt[0][0]*x + Mt[0][1]*y, Mt[1][0]*x + Mt[1][1]*y);
          mk("circle", { cx: px, cy: py, r: 5, fill: `hsl(${360*i/N},80%,55%)` }); }
        const ar = (x, y, col) => { const [x0, y0] = P(0, 0), [x1, y1] = P(x, y);
          mk("line", { x1: x0, y1: y0, x2: x1, y2: y1, stroke: col, "stroke-width": 3 }); mk("circle", { cx: x1, cy: y1, r: 4, fill: col }); };
        ar(Mt[0][0], Mt[1][0], "#2e86ab"); ar(Mt[0][1], Mt[1][1], "#e4572e");
        const k = Math.min(3, Math.round(t) === t ? t : Math.ceil(t));
        const tx = mk("text", { x: 12, y: 22, fill: "currentColor", "font-size": 14, "font-family": "sans-serif" });
        tx.textContent = `t = ${t.toFixed(2)}   ${stages[k]}`;
        const tx2 = mk("text", { x: 12, y: H - 12, fill: "#f18f01", "font-size": 12, "font-family": "sans-serif" });
        tx2.textContent = "dashed orange = final ellipse A·circle";
      }
      function animate() {
        if (raf) { cancelAnimationFrame(raf); raf = null; play.textContent = "▶ play"; model.save_changes(); return; }
        if (model.get("t") >= 3) model.set("t", 0);
        play.textContent = "⏸ pause"; let last = performance.now();
        const step = now => { const t = Math.min(3, model.get("t") + (now - last) / 1200); last = now;
          model.set("t", t); draw();
          if (t < 3) raf = requestAnimationFrame(step); else { raf = null; play.textContent = "▶ play"; model.save_changes(); } };
        raf = requestAnimationFrame(step);
      }
      for (const k of ["t", "thV", "thU", "s1", "s2"]) model.on("change:" + k, draw);
      draw();
      return () => { if (raf) cancelAnimationFrame(raf); };
    }
    export default { render };
    """
        thV = traitlets.Float(0.0).tag(sync=True)
        s1 = traitlets.Float(1.0).tag(sync=True)
        s2 = traitlets.Float(1.0).tag(sync=True)
        thU = traitlets.Float(0.0).tag(sync=True)
        t = traitlets.Float(0.0).tag(sync=True)

    return (SVDMovie,)


@app.cell
def svd_movie(A_drag, SVDMovie, mo, rotation_form):
    movie_params = rotation_form(A_drag)
    svd_movie = mo.ui.anywidget(SVDMovie(thV=movie_params[0], s1=float(movie_params[1][0]),
                                         s2=float(movie_params[1][1]), thU=movie_params[2], t=0.0))
    svd_movie
    return movie_params, svd_movie


@app.cell
def movie_readout(A_drag, mat_tex, mo, movie_params, np, svd_movie):
    _thV, _s, _thU = movie_params
    _R = lambda th: np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    _rebuilt = _R(_thU) @ np.diag(_s) @ _R(_thV)
    _t = svd_movie.value["t"]
    _stage = ["the unit circle", "after $V^\\dagger$ (a rotation)", "after $\\Sigma V^\\dagger$ (stretched along the axes)", "after $U\\Sigma V^\\dagger = A$"][min(3, int(np.ceil(_t - 1e-9)))]
    mo.md(rf"""
    | step | factor | what it does |
    |---|---|---|
    | 1 | $V^\dagger = R({np.degrees(_thV):.2f}^\circ)$ | rotate by **{np.degrees(_thV):.2f}°** ({'clockwise' if _thV < 0 else 'counter-clockwise'}) |
    | 2 | $\Sigma = \operatorname{{diag}}({_s[0]:.3f},\ {_s[1]:.3f})$ | stretch horizontally ×{_s[0]:.3f}, vertically ×{abs(_s[1]):.3f}{' and **reflect** (negative ⇒ det A < 0)' if _s[1] < 0 else ''} |
    | 3 | $U = R({np.degrees(_thU):.2f}^\circ)$ | rotate by **{np.degrees(_thU):.2f}°** |

    Rebuilt $R_U\,\Sigma\,R_V = {mat_tex(_rebuilt, 3)}$ vs $A = {mat_tex(A_drag, 3)}$; max difference **{np.abs(_rebuilt - A_drag).max():.1e}**.
    Movie position (after release): **{_stage}**.

    *numpy may return $U$ or $V^\dagger$ as reflections (det = −1). Here signs are flipped in matching pairs so both are pure rotations, which is the video's picture; the product doesn't change.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. With **Video A**: step 1 should be a rotation by **−45°** (clockwise), step 2 stretches by $3\sqrt5$ and $\sqrt5$, step 3 rotates by $\arctan 3 \approx 71.6^\circ$. Same numbers as the video.
    2. Follow one bead (say the yellow one) through the movie and check it lands where it does in Section 1. That is the "harder puzzle" solved.
    3. Pick **Reflection** in Section 1, then come back. $\sigma_2$ shows up negative here: a stretch combined with a flip, which no rotation can undo.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Dimensionality reduction: when an ellipse is almost a line

    The video's next matrix is

    $$B=\begin{bmatrix}1.8&1.2\\4.4&4.6\end{bmatrix}.$$

    It still turns the circle into an ellipse, but a **very skinny** one: the points almost lie on a line.
    In $\Sigma$ the horizontal stretch is big ($\sigma_1\approx 6.71$) and the vertical one is tiny ($\sigma_2\approx 0.44$).
    All of that skinniness lives in the small number.

    *"Who are we kidding? 0.44 is really small, let's turn it into a zero."* Use the slider to shrink $\sigma_2$
    while keeping $U$ and $V^\dagger$ fixed, and watch the matrix $U\,\Sigma'\,V^\dagger$ change.
    """)
    return


@app.cell
def sigma2_slider(mo, np):
    B_near = np.array([[1.8, 1.2], [4.4, 4.6]])
    B_U, B_S, B_Vt = np.linalg.svd(B_near)
    sigma2_slider = mo.ui.slider(0.0, round(float(B_S[1]), 2), value=round(float(B_S[1]), 2), step=0.01,
                                 label="σ₂ (the small singular value)", show_value=True, full_width=True)
    sigma2_slider
    return B_S, B_U, B_Vt, B_near, sigma2_slider


@app.cell
def sigma2_chart(
    B_S,
    B_U,
    B_Vt,
    B_near,
    alt,
    mat_tex,
    mo,
    np,
    pd,
    sigma2_slider,
):
    _S_new = np.array([B_S[0], sigma2_slider.value])
    B_trunc = B_U @ np.diag(_S_new) @ B_Vt
    _th = np.linspace(0, 2 * np.pi, 200)
    _circle = np.vstack([np.cos(_th), np.sin(_th)])
    _df = pd.concat([
        pd.DataFrame({"x": (B_near @ _circle)[0], "y": (B_near @ _circle)[1], "i": range(200), "matrix": "original B (σ₂ = 0.44)"}),
        pd.DataFrame({"x": (B_trunc @ _circle)[0], "y": (B_trunc @ _circle)[1], "i": range(200), "matrix": f"modified (σ₂ = {sigma2_slider.value:.2f})"}),
    ])
    _chart = alt.Chart(_df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[-7, 7])), y=alt.Y("y:Q", scale=alt.Scale(domain=[-7, 7])),
        order="i:Q",
        color=alt.Color("matrix:N", scale=alt.Scale(range=["#999999", "#f18f01"]), legend=alt.Legend(orient="top", title=None)),
        strokeDash=alt.StrokeDash("matrix:N", scale=alt.Scale(range=[[4, 4], [1, 0]]), legend=None),
    ).properties(width="container", height=360, title="Unit circle under B (grey) and under U Σ' V† (orange)")
    mo.vstack([
        _chart,
        mo.hstack([
            mo.stat(f"{np.linalg.det(B_trunc):.3f}", label="det"),
            mo.stat(f"{np.linalg.matrix_rank(B_trunc, tol=1e-6)}", label="rank"),
            mo.stat(f"{np.linalg.norm(B_near - B_trunc):.3f}", label="‖B − B'‖ (= 0.44 − σ₂)"),
        ], justify="start"),
        mo.md(rf"$$U\,\operatorname{{diag}}({_S_new[0]:.2f},\ {_S_new[1]:.2f})\,V^\dagger = {mat_tex(B_trunc)}\qquad\text{{(original }}B={mat_tex(B_near)})$$"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Drag $\sigma_2$ all the way to **0**. The matrix becomes $\begin{bmatrix}1.5&1.5\\4.5&4.5\end{bmatrix}$: two **identical columns**, rank 1, det 0. The ellipse is now a line (a *degenerate* transformation).
    2. Note how small the change was: the matrix moved by exactly 0.44, the value you deleted. A tiny change bought a much simpler matrix: 2 numbers (1.5, 4.5) plus "both columns equal" instead of 4 numbers.
    3. Bigger picture: $\sigma_2$ measures how far $B$ is from the nearest rank-1 matrix. That is the whole idea behind compression.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Rank-1 pieces: a matrix as a sum of outer products

    Look at the video's 4×4 example. Every row is a multiple of $(1,2,3,4)$:

    $$\begin{bmatrix}1&2&3&4\\-1&-2&-3&-4\\2&4&6&8\\10&20&30&40\end{bmatrix}
    =\begin{bmatrix}1\\-1\\2\\10\end{bmatrix}\begin{bmatrix}1&2&3&4\end{bmatrix}
    \quad\Rightarrow\quad 16 \text{ numbers} \to 8.$$

    That is a **rank-1** matrix (an *outer product*). For a 1000×1000 rank-1 matrix: $10^6 \to 2000$ numbers.

    A random matrix is not rank 1, but the SVD writes **any** matrix as a sum of rank-1 pieces,
    ordered from most to least important:

    $$A = U\Sigma V^\dagger = \sigma_1\,u_1 v_1^\dagger + \sigma_2\,u_2 v_2^\dagger + \sigma_3\,u_3 v_3^\dagger + \sigma_4\,u_4 v_4^\dagger,
    \qquad \sigma_1\ge\sigma_2\ge\sigma_3\ge\sigma_4 \ge 0.$$

    Keep only the first $k$ terms and you get $A_k$, the **closest rank-$k$ matrix to $A$**
    (the Eckart–Young theorem: it beats every other rank-$k$ matrix).
    """)
    return


@app.cell
def rank_k4(mat_tex, mo, np):
    _rng = np.random.default_rng(7)
    M4 = np.round(np.outer([2, 3, 4, 5], [1, 2, 2, 3]) + 2 * np.outer([1, -1, 0, 1], [1, 0, -1, 1]) + _rng.normal(0, 0.6, (4, 4)), 0)
    M4_U, M4_S, M4_Vt = np.linalg.svd(M4)
    rank_k4 = mo.ui.slider(1, 4, value=1, step=1, label="number of rank-1 terms k", show_value=True)
    mo.vstack([mo.md(rf"Our 4×4 matrix $A = {mat_tex(M4, 0)}$ has singular values **{', '.join(f'{s:.2f}' for s in M4_S)}**."), rank_k4])
    return M4, M4_S, M4_U, M4_Vt, rank_k4


@app.cell
def rank_k4_view(M4, M4_S, M4_U, M4_Vt, alt, heat, mo, np, pd, rank_k4):
    _k = rank_k4.value
    _terms = [M4_S[i] * np.outer(M4_U[:, i], M4_Vt[i]) for i in range(4)]
    _Ak = sum(_terms[:_k])
    _err = np.linalg.norm(M4 - _Ak) / np.linalg.norm(M4)
    _row1 = alt.hconcat(*[heat(_terms[i], f"σ{i+1} u{i+1} v{i+1}†  (σ={M4_S[i]:.2f})", w=130).properties(
        title=alt.TitleParams(f"σ{i+1}·u{i+1}v{i+1}† (σ={M4_S[i]:.2f})", color="#f18f01" if i < _k else "#aaaaaa")) for i in range(4)])
    _row2 = alt.hconcat(heat(M4, "original A", w=170), heat(_Ak, f"A_{_k} = sum of first {_k}", w=170), heat(M4 - _Ak, "error A − A_k", w=170))
    _sv = alt.Chart(pd.DataFrame({"i": [f"σ{i+1}" for i in range(4)], "σ": M4_S, "kept": ["kept" if i < _k else "dropped" for i in range(4)]})).mark_bar().encode(
        x=alt.X("i:N", title=None), y="σ:Q",
        color=alt.Color("kept:N", scale=alt.Scale(domain=["kept", "dropped"], range=["#f18f01", "#cccccc"]), legend=None)).properties(width=170, height=170, title="singular values")
    mo.vstack([
        mo.md("**The four rank-1 components** (orange title = included in the sum):"),
        _row1,
        alt.hconcat(_row2, _sv),
        mo.hstack([
            mo.stat(f"{_err:.1%}", label="relative error ‖A − A_k‖ / ‖A‖"),
            mo.stat(f"{np.sqrt(np.sum(M4_S[_k:] ** 2)):.3f}", label="√(σ²_{k+1}+…) = ‖A − A_k‖"),
            mo.stat(f"{_k * (4 + 4 + 1)}", label="numbers stored k·(m+n+1)"),
            mo.stat("16", label="numbers in A"),
        ], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. At $k=1$ the first component already looks like a blurry copy of $A$: same pattern of big and small entries. It is the best you can do with a single outer product.
    2. Step $k = 2, 3, 4$. Each new term is "lighter" (smaller $\sigma$) and fixes less. At $k=4$ the error is 0, because a 4×4 matrix has at most rank 4.
    3. The error norm equals $\sqrt{\sigma_{k+1}^2+\dots}$, the dropped singular values. Small tail means a cheap approximation.
    4. Storage: here $k(m+n+1)$ only beats 16 at $k=1$. The pay-off comes with big matrices such as images.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · The heart: an image *is* a matrix

    The video's finale: write a pixel image as a matrix, **1 for black, 0 for white**, and take its SVD.
    Each component $\sigma_i\,u_i v_i^\dagger$ is itself a little image, a rank-1 "plaid" (each row is a multiple of one pattern).
    Its strength is the singular value. Adding components from the left makes the heart sharper and sharper,
    and once the singular values hit **0** you can stop. The heart is exact.

    **Paint your own image** below (click/drag toggles pixels, presets on top). Then pick $k$.
    """)
    return


@app.cell
def PixelPainter_class(anywidget, traitlets):
    class PixelPainter(anywidget.AnyWidget):
        _esm = r"""
    const HEART = [
      "............", "..##....##..", ".####..####.", "############", "############", "############",
      ".##########.", "..########..", "...######...", "....####....", ".....##.....", "............"];
    function render({ model, el }) {
      const NS = "http://www.w3.org/2000/svg";
      const bar = document.createElement("div"); bar.style.marginBottom = "6px"; el.appendChild(bar);
      const n = () => model.get("n");
      const presets = {
        "Heart": (i, j) => (HEART[i] || "")[j] === "#" ? 1 : 0,
        "Cross": (i, j, N) => (Math.abs(i - (N-1)/2) < 1.5 || Math.abs(j - (N-1)/2) < 1.5) ? 1 : 0,
        "Square": (i, j, N) => (i > 2 && i < N-3 && j > 2 && j < N-3) ? 1 : 0,
        "Diagonal": (i, j) => Math.abs(i - j) < 1.5 ? 1 : 0,
        "Checker": (i, j) => (Math.floor(i/3) + Math.floor(j/3)) % 2,
        "Random": () => Math.random() < 0.4 ? 1 : 0,
        "Invert": null, "Clear": () => 0 };
      for (const k in presets) {
        const b = document.createElement("button"); b.textContent = k; b.style.margin = "0 4px 4px 0";
        b.onclick = () => { const N = n(); let px;
          if (k === "Invert") px = model.get("pixels").map(v => 1 - v);
          else px = Array.from({ length: N*N }, (_, idx) => presets[k](Math.floor(idx / N), idx % N, N));
          model.set("pixels", px); model.save_changes(); draw(); };
        bar.appendChild(b);
      }
      const CS = 26;
      const svg = document.createElementNS(NS, "svg");
      Object.assign(svg.style, { width: "100%", maxWidth: "320px", userSelect: "none", touchAction: "none" });
      el.appendChild(svg);
      const toSvg = (evt) => { const p = svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      let painting = false, val = 1;
      function draw() {
        const N = n(), px = model.get("pixels");
        svg.setAttribute("viewBox", `0 0 ${N*CS} ${N*CS}`); svg.innerHTML = "";
        for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
          const r = document.createElementNS(NS, "rect");
          const v = px[i*N + j];
          Object.entries({ x: j*CS, y: i*CS, width: CS - 1, height: CS - 1, rx: 3,
            fill: v > 0.5 ? "#222" : "rgba(128,128,128,0.12)" }).forEach(([a, b]) => r.setAttribute(a, b));
          svg.appendChild(r);
        }
      }
      function cell(e) { const p = toSvg(e), N = n(); const i = Math.floor(p.y / CS), j = Math.floor(p.x / CS);
        return (i < 0 || j < 0 || i >= N || j >= N) ? -1 : i*N + j; }
      function paint(e) { const idx = cell(e); if (idx < 0) return; const px = [...model.get("pixels")];
        if (px[idx] !== val) { px[idx] = val; model.set("pixels", px); draw(); } }
      svg.addEventListener("pointerdown", e => { const idx = cell(e); if (idx < 0) return;
        val = model.get("pixels")[idx] > 0.5 ? 0 : 1; painting = true; svg.setPointerCapture(e.pointerId); paint(e); });
      svg.addEventListener("pointermove", e => painting && paint(e));
      const stop = () => { if (painting) { painting = false; model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointerleave", stop);
      model.on("change:pixels", draw); draw();
    }
    export default { render };
    """
        n = traitlets.Int(12).tag(sync=True)
        pixels = traitlets.List(traitlets.Float()).tag(sync=True)

    return (PixelPainter,)


@app.cell
def painter(PixelPainter, mo):
    _heart = ["............", "..##....##..", ".####..####.", "############", "############", "############",
              ".##########.", "..########..", "...######...", "....####....", ".....##.....", "............"]
    painter = mo.ui.anywidget(PixelPainter(n=12, pixels=[1.0 if ch == "#" else 0.0 for row in _heart for ch in row]))
    heart_k = mo.ui.slider(1, 12, value=1, step=1, label="components kept k", show_value=True)
    mo.hstack([painter, mo.vstack([mo.md("**Your image** (dark = 1). Paint, then pick how many components to keep:"), heart_k])],
              justify="start", gap=2)
    return heart_k, painter


@app.cell
def heart_view(alt, heart_k, mo, np, painter, pd, show_img):
    P_img = np.array(painter.value["pixels"], dtype=float).reshape(painter.value["n"], painter.value["n"])
    P_U, P_S, P_Vt = np.linalg.svd(P_img)
    _k = heart_k.value
    _rank = int(np.sum(P_S > 1e-9))
    _Pk = (P_U[:, :_k] * P_S[:_k]) @ P_Vt[:_k]
    _comps = [mo.vstack([show_img(P_S[i] * np.outer(P_U[:, i], P_Vt[i]), width=90, ink=True),
                         mo.md(f"<span style='color:{'#f18f01' if i < _k else '#aaa'}'>σ{i+1} = {P_S[i]:.2f}</span>")])
              for i in range(min(6, len(P_S)))]
    _sums = [mo.vstack([show_img(np.clip((P_U[:, :j] * P_S[:j]) @ P_Vt[:j], 0, 1), width=90, ink=True), mo.md(f"first {j}")])
             for j in range(1, min(6, len(P_S)) + 1)]
    _sv = alt.Chart(pd.DataFrame({"i": np.arange(1, len(P_S) + 1), "σ": P_S, "kept": np.where(np.arange(len(P_S)) < _k, "kept", "dropped")})).mark_bar().encode(
        x=alt.X("i:O", title="component i"), y=alt.Y("σ:Q", title="singular value σᵢ"),
        color=alt.Color("kept:N", scale=alt.Scale(domain=["kept", "dropped"], range=["#f18f01", "#cccccc"]), legend=None),
    ).properties(width="container", height=160)
    mo.vstack([
        mo.hstack([show_img(P_img, 180, ink=True, caption="original"), show_img(np.clip(_Pk, 0, 1), 180, ink=True, caption=f"sum of first {_k} components"),
                   mo.vstack([mo.stat(str(_rank), label="rank (non-zero σ's)"),
                              mo.stat(f"{np.linalg.norm(P_img - _Pk) / max(np.linalg.norm(P_img), 1e-12):.1%}", label="relative error"),
                              mo.stat(f"{_k * (2 * P_img.shape[0] + 1)} vs {P_img.size}", label="numbers stored vs pixels")])], justify="start", gap=2),
        mo.md("**Components** $\\sigma_i u_i v_i^\\dagger$ (lighter = smaller σ; negative entries clipped to white):"), mo.hstack(_comps, justify="start"),
        mo.md("**Running sums** (what the video adds up from the left):"), mo.hstack(_sums, justify="start"),
        _sv,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. **Heart** preset: component 1 is already a fuzzy heart. With a few components it is perfect, and every σ after the rank is (numerically) 0.
    2. Paint a **Square** or a **Cross**: how many non-zero σ's do they need? A filled rectangle is one outer product, so rank 1. A cross needs 2.
    3. The **Diagonal** looks simple but is *full-rank-ish*: SVD can only build things out of horizontal × vertical plaids, so diagonal edges are expensive. **Random** is the worst case: no structure to exploit.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Image compression: choose your sharpness, pay in storage

    A real image is just a bigger matrix ($m\times n$ grey values in $[0,1]$). Keep $k$ components:

    $$A_k=\sum_{i=1}^{k}\sigma_i\,u_i v_i^\dagger,\qquad
    \text{storage}=k\,(m+n+1)\ \text{numbers instead of}\ m\,n,\qquad
    \frac{\lVert A-A_k\rVert}{\lVert A\rVert}=\sqrt{\frac{\sum_{i>k}\sigma_i^2}{\sum_i\sigma_i^2}}.$$

    The image below is generated in the notebook (a little landscape, so no downloads are needed). *"The sharper we want the image,
    the more storage we're going to need."*
    """)
    return


@app.cell
def scene_k(mo, np):
    def make_image(kind="Landscape scene", noise=0.0, seed=0, m=150, n=200):
        _y, _x = np.mgrid[0:m, 0:n]
        _y, _x = _y / (m - 1), _x / (n - 1)
        if kind == "Landscape scene":
            _img = 0.92 - 0.35 * _y                                              # sky gradient
            _img = np.where((_x - 0.22) ** 2 + ((_y - 0.2) * m / n) ** 2 < 0.07 ** 2, 1.0, _img)   # sun
            _ridge = 0.5 + 0.08 * np.sin(9 * _x) + 0.04 * np.sin(23 * _x + 1)
            _img = np.where(_y > _ridge, 0.3 + 0.15 * _x + 0.05 * np.sin(40 * _x) * np.sin(30 * _y), _img)  # mountains
            _img = np.where(_y > 0.82, 0.55 + 0.06 * np.sin(70 * _y), _img)      # lake ripples
            _house = (_x > 0.6) & (_x < 0.8) & (_y > 0.62) & (_y < 0.82)
            _img = np.where(_house, 0.12, _img)
            _roof = (_y <= 0.62) & (_y > 0.5) & (np.abs(_x - 0.7) < (_y - 0.5) * 1.0)
            _img = np.where(_roof, 0.05, _img)
            _win = (((_x > 0.63) & (_x < 0.68)) | ((_x > 0.72) & (_x < 0.77))) & (_y > 0.66) & (_y < 0.72)
            _img = np.where(_win, 0.95, _img)
            _img = np.where(((_x - 0.4) ** 2 + ((_y - 0.62) * m / n) ** 2) < 0.05 ** 2, 0.2, _img)  # tree
        elif kind == "Checkerboard":
            _img = ((np.floor(_x * 8) + np.floor(_y * 6)) % 2).astype(float)
        elif kind == "Plaid (rank 2)":
            _img = 0.25 * np.outer(np.sin(np.linspace(0, 6, m)), np.cos(np.linspace(0, 9, n))) + np.outer(0.5 + 0.2 * np.linspace(-1, 1, m), np.ones(n))
        elif kind == "Diagonal stripes":
            _img = (np.sin(25 * (_x + _y)) > 0).astype(float)  # hard edges; a smooth sine stripe would be rank 2!
        elif kind == "Circle":
            _img = (((_x - 0.5) * n / m) ** 2 + (_y - 0.5) ** 2 < 0.35 ** 2).astype(float)
        else:  # pure noise
            _img = np.random.default_rng(seed).uniform(0, 1, (m, n))
        if noise > 0:
            _img = _img + np.random.default_rng(seed + 1).normal(0, noise, (m, n))
        return np.clip(_img, 0, 1)

    scene = make_image("Landscape scene", noise=0.02)
    scene_U, scene_S, scene_Vt = np.linalg.svd(scene, full_matrices=False)
    scene_k = mo.ui.slider(1, 150, value=10, step=1, label="rank k", show_value=True, full_width=True)
    scene_k
    return make_image, scene, scene_S, scene_U, scene_Vt, scene_k


@app.cell
def scene_view(
    alt,
    mo,
    np,
    pd,
    scene,
    scene_S,
    scene_U,
    scene_Vt,
    scene_k,
    show_img,
):
    _m, _n = scene.shape
    _k = scene_k.value
    _Ak = (scene_U[:, :_k] * scene_S[:_k]) @ scene_Vt[:_k]
    _ks = np.arange(1, len(scene_S) + 1)
    _energy = np.cumsum(scene_S ** 2) / np.sum(scene_S ** 2)
    _curve = pd.DataFrame({"k": np.concatenate([_ks, _ks]),
                           "value": np.concatenate([np.sqrt(np.clip(1 - _energy, 0, None)), _ks * (_m + _n + 1) / (_m * _n)]),
                           "curve": ["relative error"] * len(_ks) + ["storage / original"] * len(_ks)})
    _lines = alt.Chart(_curve).mark_line(strokeWidth=2.5).encode(
        x=alt.X("k:Q", title="rank k"), y=alt.Y("value:Q", title=None, scale=alt.Scale(domain=[0, 1.2])),
        color=alt.Color("curve:N", scale=alt.Scale(domain=["relative error", "storage / original"], range=["#e4572e", "#2e86ab"]), legend=alt.Legend(orient="top", title=None)))
    _rule = alt.Chart(pd.DataFrame({"k": [_k]})).mark_rule(color="#f18f01", strokeWidth=3).encode(x="k:Q")
    _stored = _k * (_m + _n + 1)
    mo.vstack([
        mo.hstack([show_img(scene, 400, caption=f"original {_m}×{_n} = {_m*_n:,} numbers"),
                   show_img(np.clip(_Ak, 0, 1), 400, caption=f"rank {_k}: {_stored:,} numbers")], justify="start", gap=1),
        mo.hstack([
            mo.stat(f"{np.linalg.norm(scene - _Ak) / np.linalg.norm(scene):.2%}", label="relative error"),
            mo.stat(f"{_energy[_k-1]:.2%}", label="energy kept Σσ²"),
            mo.stat(f"{_stored / (_m * _n):.1%}", label="storage vs original"),
            mo.stat(f"{_m * _n // (_m + _n + 1)}", label="break-even k (storage = 100%)"),
        ], justify="start"),
        (_lines + _rule).properties(width="container", height=240, title="Error falls fast, storage grows linearly"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Start at $k=1$: horizontal × vertical bands only, a "plaid" sky. By $k\approx 10$ the house and the mountains are recognisable at about 12% of the storage.
    2. Find the smallest $k$ where you can't tell the difference (typically 25–40). Note the error there and the storage.
    3. Push past the **break-even** $k$: storing $U_k,\Sigma_k,V_k$ now costs *more* than the raw pixels. Compression only pays when the tail of σ's is small.
    4. The last error gains come from the 2% sensor noise we added. Noise has no structure, so it spreads over **all** singular values.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Sandbox: what makes an image compressible?

    Pick an image, add noise, and choose $k$. The right panel shows the **residual** $A-A_k$ (grey = 0), which is
    exactly what you threw away. The log-scale chart shows how fast the singular values decay. A fast drop means cheap compression.
    """)
    return


@app.cell
def sandbox_controls(mo):
    sb_kind = mo.ui.dropdown(["Landscape scene", "Checkerboard", "Plaid (rank 2)", "Diagonal stripes", "Circle", "Pure noise"],
                              value="Plaid (rank 2)", label="image")
    sb_noise = mo.ui.slider(0.0, 0.3, value=0.0, step=0.01, label="added noise σ", show_value=True)
    sb_k = mo.ui.slider(1, 100, value=2, step=1, label="rank k", show_value=True)
    mo.hstack([sb_kind, sb_noise, sb_k], justify="start", gap=2)
    return sb_k, sb_kind, sb_noise


@app.cell
def sandbox_view(
    alt,
    make_image,
    mo,
    np,
    pd,
    sb_k,
    sb_kind,
    sb_noise,
    show_img,
):
    _img = make_image(sb_kind.value, noise=sb_noise.value, seed=3)
    _U, _S, _Vt = np.linalg.svd(_img, full_matrices=False)
    _k = sb_k.value
    _Ak = (_U[:, :_k] * _S[:_k]) @ _Vt[:_k]
    _m, _n = _img.shape
    _df = pd.DataFrame({"i": np.arange(1, len(_S) + 1), "σ": np.maximum(_S, 1e-12), "kept": np.where(np.arange(len(_S)) < _k, "kept", "dropped")})
    _chart = alt.Chart(_df).mark_circle(size=30).encode(
        x=alt.X("i:Q", title="index i"), y=alt.Y("σ:Q", scale=alt.Scale(type="log", domainMin=1e-3), title="σᵢ (log)"),
        color=alt.Color("kept:N", scale=alt.Scale(domain=["kept", "dropped"], range=["#f18f01", "#cccccc"]), legend=None),
    ).properties(width="container", height=200)
    _need = int(np.searchsorted(np.cumsum(_S ** 2) / np.sum(_S ** 2), 0.99) + 1)
    mo.vstack([
        mo.hstack([show_img(_img, 260, caption="original"), show_img(np.clip(_Ak, 0, 1), 260, caption=f"rank {_k}"),
                   show_img(np.clip(0.5 + (_img - _Ak), 0, 1), 260, caption="residual (grey = 0)")], justify="start"),
        mo.hstack([mo.stat(f"{np.linalg.norm(_img - _Ak) / np.linalg.norm(_img):.2%}", label="relative error"),
                   mo.stat(f"{_k * (_m + _n + 1) / (_m * _n):.1%}", label="storage vs original"),
                   mo.stat(str(_need), label="k needed for 99% energy"),
                   mo.stat(str(np.linalg.matrix_rank(_img)), label="numerical rank")], justify="start"),
        _chart,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Challenges**

    1. **Plaid** with no noise: show that $k=2$ is perfect, then explain why with the formula $\sigma_1 u_1v_1^\dagger+\sigma_2 u_2v_2^\dagger$.
    2. **Checkerboard** looks complicated, but what rank is it? (Hint: every row is one of two patterns.)
    3. **Circle** vs **Diagonal stripes**: which needs more components for 99% energy, and why do curved and diagonal edges cost more than horizontal ones?
    4. **Pure noise**: the σ's barely decay. With no structure there is nothing to compress. Then add noise to the plaid and watch the flat "noise floor" appear in the log chart.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Quiz: check yourself

    Answer all, then flip **reveal answers**.
    """)
    return


@app.cell
def quiz(mo):
    quiz = mo.ui.dictionary({
        "q1": mo.ui.dropdown(["U", "Σ", "V†"], label="1. In A = UΣV†, which matrix acts on the vector **first**?"),
        "q2": mo.ui.dropdown(["a circle of radius 2", "an ellipse 4 wide and 1 tall", "an ellipse 2 wide and 0.5 tall", "a line"],
                             label="2. diag(2, 0.5) turns the unit circle into…"),
        "q3": mo.ui.dropdown(["0 and 0", "1 and 1", "cos θ and sin θ", "θ and −θ"], label="3. The singular values of a rotation matrix are…"),
        "q4": mo.ui.dropdown(["the ellipse becomes a circle", "the matrix gets rank 1 and the ellipse becomes a line", "nothing changes", "the matrix becomes the identity"],
                             label="4. Setting the smaller σ of a 2×2 matrix to 0 means…"),
        "q5": mo.ui.dropdown(["1,000,000", "40,020", "20,000", "2,000"], label="5. Numbers needed for a rank-20 approximation of a 1000×1000 image (k·(m+n+1))?"),
        "q6": mo.ui.dropdown(["a filled rectangle", "random noise", "a diagonal line", "a checkerboard of random colours"],
                             label="6. Which image compresses best with SVD?"),
        "q7": mo.ui.dropdown(["it's the closest rank-k matrix to A", "it has the same determinant as A", "it is always exact", "it only works for square matrices"],
                             label="7. What is special about keeping the top k terms σᵢuᵢvᵢ†?"),
    })
    reveal = mo.ui.switch(label="reveal answers")
    mo.vstack([quiz, reveal])
    return quiz, reveal


@app.cell
def quiz_score(mo, quiz, reveal):
    _truth = {
        "q1": ("V†", "Matrices act right-to-left: rotate by V† first."),
        "q2": ("an ellipse 4 wide and 1 tall", "Half-axes are 2 and 0.5, so the full width is 4 and the height is 1."),
        "q3": ("1 and 1", "A rotation doesn't stretch; the circle stays a unit circle."),
        "q4": ("the matrix gets rank 1 and the ellipse becomes a line", "The video's 1.8/1.2/4.4/4.6 → 1.5/1.5/4.5/4.5 example."),
        "q5": ("40,020", "20 × (1000 + 1000 + 1) = 40,020, about 4% of 10⁶."),
        "q6": ("a filled rectangle", "A rectangle is a single outer product (rank 1)."),
        "q7": ("it's the closest rank-k matrix to A", "Eckart–Young: no other rank-k matrix is closer."),
    }
    _score = sum(quiz.value[q] == a for q, (a, _) in _truth.items())
    _rows = "\n".join(f"| {q[1:]} | {quiz.value[q] or '—'} | {'✅' if quiz.value[q] == a else '❌'} | {a} — {why} |" for q, (a, why) in _truth.items())
    _out = [mo.stat(f"{_score} / {len(_truth)}", label="score")]
    if reveal.value:
        _out.append(mo.md("| # | your answer | | correct answer |\n|---|---|---|---|\n" + _rows))
    mo.vstack(_out)
    return


@app.cell(hide_code=True)
def recap(mo):
    mo.md(r"""
    ## Recap

    | Idea | Formula / picture | Where you played with it |
    |---|---|---|
    | A matrix moves points; the unit circle becomes an ellipse | columns of $A$ = images of $e_1, e_2$ | §1 draggable matrix |
    | Every matrix = rotate → scale on the axes → rotate | $A = U\Sigma V^\dagger$ (read right to left) | §2 SVD movie |
    | Singular values are the ellipse's half-axes | $\sigma_1 \ge \sigma_2 \ge \dots \ge 0$, $\lvert\det A\rvert = \prod \sigma_i$ | §1, §2 |
    | A small σ means the matrix is nearly degenerate; zero it to reduce rank | $B \to \begin{bmatrix}1.5&1.5\\4.5&4.5\end{bmatrix}$ | §3 σ₂ slider |
    | Rank 1 = outer product; SVD = sum of rank-1 pieces | $A=\sum_i \sigma_i u_i v_i^\dagger$ | §4 component heatmaps |
    | Truncate to $k$ terms → best rank-$k$ approximation | $\lVert A-A_k\rVert=\sqrt{\sum_{i>k}\sigma_i^2}$ | §4, §5 heart painter |
    | Image compression trades sharpness for storage | $k(m+n+1)$ vs $mn$ | §6 rank-k slider, §7 sandbox |

    **Further from the video:** PCA and matrix factorization (Netflix recommendations) are the same idea: keep the big singular directions.
    """)
    return


if __name__ == "__main__":
    app.run()

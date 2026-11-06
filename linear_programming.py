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
    from scipy.optimize import linprog

    return alt, anywidget, linprog, mo, np, pd, traitlets


@app.cell(hide_code=True)
def title(mo):
    mo.vstack([
        mo.md(r"""
    # Intro to Linear Programming: the carpenter's problem
    *An interactive companion to Dr. Trefor Bazett's video **"Intro to Linear Programming"**.*

    A carpenter can build **tables** or **bookcases**, with a limited supply of wood and time.
    How many of each should they make to earn the most money? We answer this with no calculus and no matrices,
    only a picture: shade a region, slide a line across it, and look where the line touches last.
    """),
        mo.Html('<iframe width="560" height="315" src="https://www.youtube.com/embed/K7TL5NMlKIk" title="Intro to Linear Programming" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'),
    ])
    return


@app.cell(hide_code=True)
def colors(mo):
    mo.md(r"""
    ### Colour language used throughout
    | colour | meaning |
    |---|---|
    | <span style="color:#e76f51">■ orange</span> | the **labour** constraint $5x+4y\le 80$ (hours) |
    | <span style="color:#8d6e63">■ brown</span> | the **lumber** constraint $10x+20y\le 200$ (wood) |
    | <span style="color:#2a9d8f">■ green</span> | the **feasible region**, i.e. plans that break no rule (green, like in the video) |
    | <span style="color:#1f77b4">■ blue</span> | **iso-value lines**, where profit $180x+200y$ is constant |
    | <span style="color:#d62828">■ red</span> | the **optimum** |
    | <span style="color:#6c5b7b">■ purple</span> | *your* plan or guess |
    """)
    return


@app.cell
def lp_helpers(linprog, np, pd):
    def lp_vertices(cons, obj, box=None):
        """Corner points of {a*x + b*y <= r for each constraint, x >= 0, y >= 0}.

        cons: list of dicts with keys a, b, r (and optional label).
        box:  optional (xmax, ymax) used only to clip unbounded regions for drawing.
        Returns a DataFrame of vertices ordered around the polygon, with the objective value.
        """
        _rows = [(c["a"], c["b"], c["r"], c.get("label", "")) for c in cons]
        _rows += [(-1.0, 0.0, 0.0, "x ≥ 0"), (0.0, -1.0, 0.0, "y ≥ 0")]
        if box is not None:
            _rows += [(1.0, 0.0, box[0], "x ≤ edge"), (0.0, 1.0, box[1], "y ≤ edge")]
        A = np.array([r[:2] for r in _rows], float)
        rhs = np.array([r[2] for r in _rows], float)
        pts = {}
        for i in range(len(_rows)):
            for j in range(i + 1, len(_rows)):
                M = A[[i, j]]
                if abs(np.linalg.det(M)) < 1e-12:
                    continue
                p = np.linalg.solve(M, rhs[[i, j]])
                if np.all(A @ p <= rhs + 1e-7):
                    key = (round(p[0], 6), round(p[1], 6))
                    tight = [r[3] for r, lhs, rr in zip(_rows, A @ p, rhs) if abs(lhs - rr) < 1e-7]
                    pts[key] = (p + 0.0, tight)
        if not pts:
            return pd.DataFrame(columns=["x", "y", "value", "tight"])
        df = pd.DataFrame([(v[0][0], v[0][1], " & ".join(v[1])) for v in pts.values()], columns=["x", "y", "tight"])
        cx, cy = df.x.mean(), df.y.mean()
        df = df.assign(_ang=np.arctan2(df.y - cy, df.x - cx)).sort_values("_ang").drop(columns="_ang")
        df["value"] = obj[0] * df.x + obj[1] * df.y
        return df.reset_index(drop=True)[["x", "y", "value", "tight"]]


    def lp_solve(cons, obj, sense="max"):
        """Reference answer from scipy's simplex-family solver (HiGHS)."""
        _sign = -1 if sense == "max" else 1
        res = linprog(
            _sign * np.array(obj, float),
            A_ub=[[c["a"], c["b"]] for c in cons] or None,
            b_ub=[c["r"] for c in cons] or None,
            bounds=[(0, None), (0, None)],
            method="highs",
        )
        status = {0: "optimal", 2: "infeasible", 3: "unbounded"}.get(res.status, res.message)
        return {"status": status, "x": res.x, "value": (_sign * res.fun) if res.status == 0 else None}

    return lp_solve, lp_vertices


@app.cell
def lp_js_base():
    LP_JS_BASE = r"""
    const NS = "http://www.w3.org/2000/svg";
    function niceStep(m) { const raw = m / 8, p = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / p;
      return (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * p; }
    function plane(el, model, W = 560, H = 420) {
      const PADL = 46, PADB = 40, PADT = 12, PADR = 14;
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none", display: "block", color: "inherit" });
      el.appendChild(svg);
      const mk = (tag, attrs, parent = svg) => { const e = document.createElementNS(NS, tag);
        for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
      const X = () => model.get("xmax"), Y = () => model.get("ymax");
      const sx = v => PADL + v / X() * (W - PADL - PADR), sy = v => H - PADB - v / Y() * (H - PADB - PADT);
      const ix = px => (px - PADL) / (W - PADL - PADR) * X(), iy = py => (H - PADB - py) / (H - PADB - PADT) * Y();
      const toSvg = e => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      function clipLine(a, b, r) {
        const xm = X(), ym = Y(), pts = [], eps = 1e-9;
        if (Math.abs(b) > 1e-12) for (const x of [0, xm]) { const y = (r - a * x) / b; if (y >= -eps && y <= ym + eps) pts.push([x, y]); }
        if (Math.abs(a) > 1e-12) for (const y of [0, ym]) { const x = (r - b * y) / a; if (x >= -eps && x <= xm + eps) pts.push([x, y]); }
        if (pts.length < 2) return null;
        pts.sort((p, q) => p[0] - q[0] || p[1] - q[1]); return [pts[0], pts[pts.length - 1]];
      }
      const text = (x, y, s, attrs = {}) => { const t = mk("text", Object.assign({ x, y, "font-size": 12, fill: "currentColor" }, attrs)); t.textContent = s; return t; };
      function base(opts = {}) {
        svg.innerHTML = "";
        const xm = X(), ym = Y(), stx = niceStep(xm), sty = niceStep(ym);
        for (let v = 0; v <= xm + 1e-9; v += stx) { mk("line", { x1: sx(v), x2: sx(v), y1: sy(0), y2: sy(ym), stroke: "currentColor", "stroke-opacity": .1 });
          text(sx(v), H - PADB + 15, +v.toFixed(2), { "text-anchor": "middle", "font-size": 11 }); }
        for (let v = 0; v <= ym + 1e-9; v += sty) { mk("line", { x1: sx(0), x2: sx(xm), y1: sy(v), y2: sy(v), stroke: "currentColor", "stroke-opacity": .1 });
          text(PADL - 6, sy(v) + 4, +v.toFixed(2), { "text-anchor": "end", "font-size": 11 }); }
        mk("line", { x1: sx(0), x2: sx(xm), y1: sy(0), y2: sy(0), stroke: "currentColor", "stroke-opacity": .6 });
        mk("line", { x1: sx(0), x2: sx(0), y1: sy(0), y2: sy(ym), stroke: "currentColor", "stroke-opacity": .6 });
        text(sx(xm), H - 6, "x = tables →", { "text-anchor": "end" });
        text(8, 14, "↑ y = bookcases", {});
        const poly = model.get("poly");
        if (poly.length >= 3) mk("polygon", { points: poly.map(p => `${sx(p[0])},${sy(p[1])}`).join(" "),
          fill: "#2a9d8f", "fill-opacity": .22, stroke: "#2a9d8f", "stroke-width": 1.5 });
        for (const c of model.get("cons")) {
          const seg = clipLine(c.a, c.b, c.r); if (!seg) continue;
          mk("line", { x1: sx(seg[0][0]), y1: sy(seg[0][1]), x2: sx(seg[1][0]), y2: sy(seg[1][1]),
            stroke: c.color || "#888", "stroke-width": opts.faded ? 1.2 : 2.5, "stroke-opacity": opts.faded ? .35 : .95 });
          const lx = seg[0][0] + 0.06 * (seg[1][0] - seg[0][0]), ly = seg[0][1] + 0.06 * (seg[1][1] - seg[0][1]);
          text(sx(lx) + 4, sy(ly) - 6, c.label || "", { fill: c.color || "#888", "font-weight": 600 });
        }
      }
      return { svg, mk, sx, sy, ix, iy, toSvg, base, text, clipLine, W, H, X, Y };
    }
    """
    return (LP_JS_BASE,)


@app.cell(hide_code=True)
def s1_md(mo):
    mo.md(r"""
    ## 1 · What is a linear program?

    Despite the name, *programming* here means **planning**, not coding. A linear program is

    $$\text{optimize}\quad 180x + 200y \qquad\text{subject to}\qquad
    \begin{cases}5x + 4y \le 80\\ 10x + 20y \le 200\\ x \ge 0,\; y \ge 0\end{cases}$$

    Everything is **linear**: no $x^2$, no $\sin x$, only numbers times variables, added up.
    "Optimize" can mean maximize *or* minimize.

    ### The carpenter problem
    | | lumber (units) | labour (hours) | profit |
    |---|---|---|---|
    | **table** ($x$) | 10 | 5 | \$180 |
    | **bookcase** ($y$) | 20 | 4 | \$200 |
    | **available** | 200 | 80 | maximize |

    So $x$ tables use $10x$ lumber and $5x$ hours, and $y$ bookcases use $20y$ lumber and $4y$ hours.
    **Before reading on, pick a production plan. Can you beat \$3,000?**
    """)
    return


@app.cell
def plan_sliders(mo):
    plan_x = mo.ui.slider(0, 20, step=1, value=8, label="tables x", show_value=True)
    plan_y = mo.ui.slider(0, 12, step=1, value=4, label="bookcases y", show_value=True)
    mo.hstack([plan_x, plan_y], justify="start", gap=2)
    return plan_x, plan_y


@app.cell
def plan_readout(alt, mo, pd, plan_x, plan_y):
    _x, _y = plan_x.value, plan_y.value
    _use = pd.DataFrame({
        "resource": ["labour (hours)", "lumber (units)"],
        "used": [5 * _x + 4 * _y, 10 * _x + 20 * _y],
        "available": [80, 200],
        "color": ["#e76f51", "#8d6e63"],
    })
    _use["share"] = _use.used / _use.available
    _bars = alt.Chart(_use).mark_bar(height=26).encode(
        x=alt.X("share:Q", title="share of what's available", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, max(1.3, _use.share.max())])),
        y=alt.Y("resource:N", title=None),
        color=alt.Color("color:N", scale=None),
        tooltip=["resource", "used", "available"],
    )
    _rule = alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(color="#d62828", strokeDash=[5, 3], size=2).encode(x="x:Q")
    _ok = bool((_use.used <= _use.available).all())
    _profit = 180 * _x + 200 * _y
    mo.vstack([
        mo.hstack([
            mo.stat(f"${_profit:,}", label="profit 180x + 200y", bordered=True),
            mo.stat(f"{_use.used[0]} / 80", label="hours used", bordered=True),
            mo.stat(f"{_use.used[1]} / 200", label="lumber used", bordered=True),
        ], justify="start"),
        (_bars + _rule).properties(width="container", height=110, title="Resource use (red line = 100%)"),
        mo.callout(mo.md("✅ **Feasible**: this plan breaks no rule." if _ok else
                         "❌ **Infeasible**: you ran out of something. This plan is not allowed, however much it would pay."),
                   kind="success" if _ok else "danger"),
    ])
    return


@app.cell(hide_code=True)
def s1_try(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Make only tables (y = 0). Which resource runs out first, and at how many tables?
    2. Make only bookcases (x = 0). Now which resource is the bottleneck?
    3. Hunt for the best whole-number plan. Notice that the good plans use up *nearly all* of **both** resources.
    """), kind="info")
    return


@app.cell(hide_code=True)
def s2_md(mo):
    mo.md(r"""
    ## 2 · Graphing inequalities: the feasible region

    Each inequality is a **line plus a side**. The line $5x+4y=80$ is "all 80 hours used exactly", and
    $5x+4y\le 80$ shades everything *beneath* it. Do this for every constraint: the spot where **all** shadings
    overlap (dark green in the video) is the **feasible region**.

    $$\underbrace{5x+4y\le 80}_{\text{labour}}\ \cap\ \underbrace{10x+20y\le 200}_{\text{lumber}}\ \cap\ \underbrace{x\ge0,\ y\ge0}_{\text{no negative furniture}}$$

    Four edges give a **quadrilateral with four vertices**. Any point inside it is a legal plan.
    **Drag the purple dot** (or click anywhere) to test plans. Use the sliders to change the carpenter's budgets.
    """)
    return


@app.cell
def budget_sliders(mo):
    hours_cap = mo.ui.slider(40, 120, step=5, value=80, label="labour hours available", show_value=True)
    lumber_cap = mo.ui.slider(100, 300, step=10, value=200, label="lumber available", show_value=True)
    mo.hstack([hours_cap, lumber_cap], justify="start", gap=2)
    return hours_cap, lumber_cap


@app.cell
def carpenter_model(hours_cap, lp_vertices, lumber_cap):
    PROFIT = (180, 200)
    carpenter_cons = [
        {"a": 5, "b": 4, "r": hours_cap.value, "label": f"labour 5x+4y≤{hours_cap.value}", "color": "#e76f51"},
        {"a": 10, "b": 20, "r": lumber_cap.value, "label": f"lumber 10x+20y≤{lumber_cap.value}", "color": "#8d6e63"},
    ]
    PLOT_BOX = (25, 18)
    carpenter_poly = lp_vertices(carpenter_cons, PROFIT, box=PLOT_BOX)[["x", "y"]].values.tolist()
    return PLOT_BOX, PROFIT, carpenter_cons, carpenter_poly


@app.cell
def LPPlane_class(LP_JS_BASE, anywidget, traitlets):
    class LPPlane(anywidget.AnyWidget):
        """Drag a plan point around the constraint plane; JS gives instant feedback, Python gets the point on release."""
        _esm = LP_JS_BASE + r"""
    function render({ model, el }) {
      const P = plane(el, model);
      const info = document.createElement("div");
      Object.assign(info.style, { fontFamily: "monospace", fontSize: "13px", margin: "4px 0 0 46px" });
      el.appendChild(info);
      let drag = false;
      function feasible(x, y) { return x >= -1e-9 && y >= -1e-9 && model.get("cons").every(c => c.a * x + c.b * y <= c.r + 1e-9); }
      function draw() {
        P.base();
        // whole-number plans inside the region, as faint dots
        for (let i = 0; i <= P.X(); i++) for (let j = 0; j <= P.Y(); j++)
          if (feasible(i, j)) P.mk("circle", { cx: P.sx(i), cy: P.sy(j), r: 1.8, fill: "currentColor", "fill-opacity": .35 });
        const [x, y] = model.get("point"), ok = feasible(x, y), [p, q] = model.get("obj");
        P.mk("line", { x1: P.sx(x), x2: P.sx(x), y1: P.sy(y), y2: P.sy(0), stroke: "#6c5b7b", "stroke-dasharray": "3 3" });
        P.mk("line", { x1: P.sx(0), x2: P.sx(x), y1: P.sy(y), y2: P.sy(y), stroke: "#6c5b7b", "stroke-dasharray": "3 3" });
        const c = P.mk("circle", { cx: P.sx(x), cy: P.sy(y), r: 10, fill: ok ? "#6c5b7b" : "none", stroke: ok ? "white" : "#d62828", "stroke-width": ok ? 2 : 3 });
        c.style.cursor = "grab";
        if (!ok) P.text(P.sx(x) - 5, P.sy(y) + 5, "✗", { fill: "#d62828", "font-weight": 700 });
        const parts = model.get("cons").map(k => `${k.a}x+${k.b}y = ${(k.a * x + k.b * y).toFixed(1)} ${k.a * x + k.b * y <= k.r + 1e-9 ? "≤" : ">"} ${k.r}`);
        info.innerHTML = `(x, y) = (${x.toFixed(2)}, ${y.toFixed(2)}) · profit $${(p * x + q * y).toFixed(0)} · ` +
          `<b style="color:${ok ? "#2a9d8f" : "#d62828"}">${ok ? "feasible" : "infeasible"}</b><br>` + parts.join(" · ");
      }
      function move(e) { const s = P.toSvg(e), snap = v => Math.round(v * 4) / 4;
        model.set("point", [Math.max(0, Math.min(P.X(), snap(P.ix(s.x)))), Math.max(0, Math.min(P.Y(), snap(P.iy(s.y))))]); }
      P.svg.addEventListener("pointerdown", e => { drag = true; P.svg.setPointerCapture(e.pointerId); move(e); });
      P.svg.addEventListener("pointermove", e => { if (drag) move(e); });
      const stop = () => { if (drag) { drag = false; model.save_changes(); } };
      P.svg.addEventListener("pointerup", stop); P.svg.addEventListener("pointercancel", stop);
      for (const t of ["point", "cons", "poly", "obj"]) model.on("change:" + t, draw);
      draw();
    }
    export default { render };
    """
        cons = traitlets.List(traitlets.Dict()).tag(sync=True)
        poly = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)
        xmax = traitlets.Float(25).tag(sync=True)
        ymax = traitlets.Float(18).tag(sync=True)
        point = traitlets.List(traitlets.Float(), default_value=[8.0, 4.0]).tag(sync=True)
        obj = traitlets.List(traitlets.Float(), default_value=[180.0, 200.0]).tag(sync=True)

    return (LPPlane,)


@app.cell
def plane_widget(
    LPPlane,
    PLOT_BOX,
    PROFIT,
    carpenter_cons,
    carpenter_poly,
    mo,
):
    plane_widget = mo.ui.anywidget(LPPlane(
        cons=carpenter_cons, poly=carpenter_poly, xmax=PLOT_BOX[0], ymax=PLOT_BOX[1],
        point=[8.0, 4.0], obj=list(map(float, PROFIT)),
    ))
    plane_widget
    return (plane_widget,)


@app.cell
def plane_readout(carpenter_cons, mo, np, pd, plane_widget):
    _x, _y = plane_widget.value["point"]
    _rows = [(c["label"].split(" ")[0], c["a"] * _x + c["b"] * _y, c["r"]) for c in carpenter_cons]
    _rows += [("x ≥ 0", -_x, 0), ("y ≥ 0", -_y, 0)]
    _tab = pd.DataFrame(_rows, columns=["constraint", "LHS", "RHS"])
    _tab["slack (unused)"] = _tab.RHS - _tab.LHS
    _tab["status"] = np.where(_tab["slack (unused)"] < -1e-9, "❌ violated", np.where(_tab["slack (unused)"] < 1e-9, "🟠 tight (on the line)", "✅ satisfied"))
    mo.vstack([
        mo.md(f"**Plan:** {_x:g} tables, {_y:g} bookcases, profit **${180*_x + 200*_y:,.0f}**"),
        mo.ui.table(_tab.round(2), selection=None, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def s2_try(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Drag the dot exactly onto the orange line. Its *slack* becomes 0: every hour is used. Then slide along that line and watch the lumber slack.
    2. Find the one point where **both** slacks are 0 at once. That is the video's "middle vertex".
    3. Increase the labour budget until the orange line no longer cuts the region. Now labour stops mattering and only lumber binds.
    """), kind="info")
    return


@app.cell(hide_code=True)
def s3_md(mo):
    mo.md(r"""
    ## 3 · Computing the maximum: check the corners

    The video's intuition: if you finish with **unused wood or unused hours**, you could have built more and earned more.
    So the best plan should use **all** of both, which is the vertex where the two lines cross. Turn both inequalities into equations:

    $$5x+4y=80 \;\Rightarrow\; y = 20-\tfrac54x \qquad\qquad 10x+20y=200 \;\Rightarrow\; y = 10-\tfrac{x}{2}$$

    $$20-\tfrac54x = 10-\tfrac x2 \;\Rightarrow\; x=\tfrac{40}{3}\approx 13.3,\quad y=\tfrac{10}{3}\approx 3.33,\quad 180x+200y \approx \$3{,}066.67$$

    The general recipe: **intersect every pair of boundary lines, throw away crossings that break a rule, and evaluate the profit
    at the survivors.** Step through it below. Green dots are kept, red ✗s are infeasible crossings, and the red star is the best so far.

    <small>*Note: one slide in the video writes the lumber rule as 10x + 10y ≤ 200 and says the all-bookcase corner pays \$1,800.
    The setup (20 lumber per bookcase) gives 10x + 20y ≤ 200 and 200·10 = **\$2,000** there. The optimum (\$3,066.67) is unaffected.*</small>
    """)
    return


@app.cell
def lp_steps(np):
    def lp_steps(cons, obj):
        """All pairwise-intersection states for the vertex-enumeration stepper."""
        _lines = [dict(c) for c in cons] + [
            {"a": -1, "b": 0, "r": 0, "label": "x = 0", "color": "#888888"},
            {"a": 0, "b": -1, "r": 0, "label": "y = 0", "color": "#888888"},
        ]
        def _ok(x, y):
            return all(l["a"] * x + l["b"] * y <= l["r"] + 1e-7 for l in _lines)
        states = [{"pair": [], "pt": [], "ok": False, "best": [], "kept": [], "rejected": [],
                   "caption": "Start: we have %d boundary lines, so %d pairs to try." % (len(_lines), len(_lines) * (len(_lines) - 1) // 2)}]
        kept, rejected, best, best_v = [], [], [], -np.inf
        for i in range(len(_lines)):
            for j in range(i + 1, len(_lines)):
                L1, L2 = _lines[i], _lines[j]
                M = np.array([[L1["a"], L1["b"]], [L2["a"], L2["b"]]], float)
                name = f"{L1['label'].split(' ')[0]} ∩ {L2['label'].split(' ')[0]}"
                if abs(np.linalg.det(M)) < 1e-12:
                    states.append({"pair": [i, j], "pt": [], "ok": False, "best": list(best), "kept": list(kept), "rejected": list(rejected),
                                   "caption": f"{name}: parallel lines, no crossing."})
                    continue
                x, y = np.linalg.solve(M, [L1["r"], L2["r"]]) + 0.0
                v = obj[0] * x + obj[1] * y
                if _ok(x, y):
                    kept.append([x, y])
                    if v > best_v + 1e-9:
                        best, best_v = [x, y], v
                    cap = f"{name} at ({x:.2f}, {y:.2f}): <b style='color:#2a9d8f'>feasible</b> vertex, profit ${v:,.2f}. Best so far ${best_v:,.2f}."
                else:
                    rejected.append([x, y])
                    cap = f"{name} at ({x:.2f}, {y:.2f}): <b style='color:#d62828'>breaks a constraint</b>, discarded (it would pay ${v:,.2f}, but it is not allowed)."
                states.append({"pair": [i, j], "pt": [x, y], "ok": bool(_ok(x, y)), "best": list(best), "kept": list(kept), "rejected": list(rejected), "caption": cap})
        states.append({"pair": [], "pt": [], "ok": False, "best": list(best), "kept": list(kept), "rejected": list(rejected),
                       "caption": f"Done. The <b style='color:#d62828'>maximum ${best_v:,.2f}</b> is at ({best[0]:.2f}, {best[1]:.2f})."})
        return _lines, states

    return (lp_steps,)


@app.cell
def VertexStepper_class(LP_JS_BASE, anywidget, traitlets):
    class VertexStepper(anywidget.AnyWidget):
        """Step through pairwise line intersections. Python computes the states; JS only renders states[i]."""
        _esm = LP_JS_BASE + r"""
    function render({ model, el }) {
      const ctl = document.createElement("div"); ctl.style.margin = "0 0 6px 46px"; el.appendChild(ctl);
      const P = plane(el, model);
      const cap = document.createElement("div"); Object.assign(cap.style, { margin: "6px 0 0 46px", minHeight: "2.6em" }); el.appendChild(cap);
      let timer = null;
      const go = d => { const n = model.get("states").length; model.set("i", Math.max(0, Math.min(n - 1, model.get("i") + d))); model.save_changes(); };
      const btn = (t, f) => { const b = document.createElement("button"); b.textContent = t; b.onclick = f; b.style.marginRight = "4px"; ctl.appendChild(b); return b; };
      btn("⏮", () => go(-1e9)); btn("◀ back", () => go(-1)); btn("next ▶", () => go(1));
      const play = btn("▶▶ play", () => { if (timer) { clearInterval(timer); timer = null; play.textContent = "▶▶ play"; return; }
        if (model.get("i") >= model.get("states").length - 1) { model.set("i", 0); model.save_changes(); }
        play.textContent = "⏸ pause";
        timer = setInterval(() => { if (model.get("i") >= model.get("states").length - 1) { clearInterval(timer); timer = null; play.textContent = "▶▶ play"; } else go(1); }, 900); });
      btn("⏭", () => go(1e9));
      function draw() {
        const s = model.get("states")[model.get("i")], lines = model.get("lines");
        P.base({ faded: s.pair.length > 0 });
        for (const k of s.pair) { const L = lines[k], seg = P.clipLine(L.a, L.b, L.r); if (!seg) continue;
          P.mk("line", { x1: P.sx(seg[0][0]), y1: P.sy(seg[0][1]), x2: P.sx(seg[1][0]), y2: P.sy(seg[1][1]), stroke: L.color, "stroke-width": 5, "stroke-opacity": .9 }); }
        const inBox = p => p[0] >= -1e-9 && p[1] >= -1e-9 && p[0] <= P.X() && p[1] <= P.Y();
        for (const p of s.rejected) if (inBox(p)) P.text(P.sx(p[0]) - 5, P.sy(p[1]) + 5, "✗", { fill: "#d62828", "fill-opacity": .5, "font-weight": 700 });
        for (const p of s.kept) P.mk("circle", { cx: P.sx(p[0]), cy: P.sy(p[1]), r: 6, fill: "#2a9d8f" });
        if (s.pt.length) { if (inBox(s.pt)) P.mk("circle", { cx: P.sx(s.pt[0]), cy: P.sy(s.pt[1]), r: 12, fill: "none", stroke: s.ok ? "#2a9d8f" : "#d62828", "stroke-width": 3 });
          else P.text(P.W - 200, 30, "(crossing is off-screen)", { fill: "#d62828" }); }
        if (s.best.length) P.text(P.sx(s.best[0]) - 9, P.sy(s.best[1]) + 7, "★", { fill: "#d62828", "font-size": 22 });
        cap.innerHTML = `<b>step ${model.get("i")}/${model.get("states").length - 1}</b>: ${s.caption}`;
      }
      model.on("change:i", draw); model.on("change:states", draw); draw();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        lines = traitlets.List(traitlets.Dict()).tag(sync=True)
        states = traitlets.List(traitlets.Dict()).tag(sync=True)
        cons = traitlets.List(traitlets.Dict()).tag(sync=True)
        poly = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)
        xmax = traitlets.Float(25).tag(sync=True)
        ymax = traitlets.Float(18).tag(sync=True)
        i = traitlets.Int(0).tag(sync=True)

    return (VertexStepper,)


@app.cell
def stepper(
    PLOT_BOX,
    PROFIT,
    VertexStepper,
    carpenter_cons,
    carpenter_poly,
    lp_steps,
    mo,
):
    _lines, _states = lp_steps(carpenter_cons, PROFIT)
    stepper = mo.ui.anywidget(VertexStepper(
        lines=_lines, states=_states, cons=carpenter_cons, poly=carpenter_poly,
        xmax=PLOT_BOX[0], ymax=PLOT_BOX[1],
    ))
    stepper
    return (stepper,)


@app.cell
def vertex_readout(
    PROFIT,
    alt,
    carpenter_cons,
    lp_solve,
    lp_vertices,
    mo,
    np,
    stepper,
):
    carpenter_vertices = lp_vertices(carpenter_cons, PROFIT)
    _best = carpenter_vertices.value.max()
    _v = carpenter_vertices.assign(
        plan=lambda d: [f"({x:.2f}, {y:.2f})" for x, y in zip(d.x, d.y)],
        is_best=lambda d: np.isclose(d.value, _best),
    )
    _chart = alt.Chart(_v).mark_bar().encode(
        x=alt.X("value:Q", title="profit 180x + 200y ($)"),
        y=alt.Y("plan:N", sort="-x", title="vertex (tables, bookcases)"),
        color=alt.condition("datum.is_best", alt.value("#d62828"), alt.value("#2a9d8f")),
        tooltip=["plan", alt.Tooltip("value:Q", format=",.2f"), "tight"],
    ).properties(width="container", height=150, title="Profit at each vertex of the feasible region")
    _ref = lp_solve(carpenter_cons, PROFIT)
    mo.vstack([
        _chart,
        mo.md(f"Stepper is at step **{stepper.value['i']}**. Vertex check: max **${_best:,.2f}**. "
              f"scipy `linprog` (HiGHS) agrees: **${_ref['value']:,.2f}** at ({_ref['x'][0]:.3f}, {_ref['x'][1]:.3f})."),
    ])
    return (carpenter_vertices,)


@app.cell(hide_code=True)
def s3_try(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Press **▶▶ play** and count the pairs. With $m$ lines there are $\binom{m}{2}$ pairs: 6 here, but 4,950 for 100 constraints. That's why real solvers (the **simplex method**) walk from vertex to neighbouring vertex instead of checking them all.
    2. Go back to section 2 and raise the lumber budget to 300. Re-run the stepper: which crossing switches from ✗ to a green dot?
    3. Look at the *rejected* crossings: some would pay more than \$3,067. Being profitable is not enough; a plan must also be feasible.
    """), kind="info")
    return


@app.cell(hide_code=True)
def s4_md(mo):
    mo.md(r"""
    ## 4 · Iso-value lines: why the answer is a corner

    That still leaves a question: *why* must the best plan be a vertex? The video's argument uses **iso-value lines**,
    which are all plans that earn the *same* amount $c$:

    $$180x + 200y = c$$

    Changing $c$ slides the line **parallel to itself**. A bigger $c$ moves it up and to the right, in the direction of the arrow
    $(180, 200)$, which points where profit grows fastest.

    * Is $c$ too big? The line misses the green region, so **no feasible plan** earns that much.
    * Is $c$ too small? Lots of feasible plans earn it, but you're leaving money on the table.
    * The best $c$ is the largest one where the line still *"comes down and kisses"* the region. For a polygon, that kiss happens at a **vertex**.

    **Drag on the plane** to sweep the blue line (it follows your pointer). **Drag the arrow tip** to change the profit per table or bookcase.
    """)
    return


@app.cell
def IsoSweeper_class(LP_JS_BASE, anywidget, traitlets):
    class IsoSweeper(anywidget.AnyWidget):
        """Sweep an iso-value line p*x + q*y = c across the feasible region; drag the arrow tip to rotate the objective."""
        _esm = LP_JS_BASE + r"""
    function render({ model, el }) {
      const ctl = document.createElement("div"); ctl.style.margin = "0 0 6px 46px"; el.appendChild(ctl);
      const P = plane(el, model);
      const info = document.createElement("div"); Object.assign(info.style, { margin: "6px 0 0 46px", fontFamily: "monospace", fontSize: "13px" }); el.appendChild(info);
      const AX = P.W - 95, AY = 95, K = 4;              // arrow anchor (px) and $ per px
      let mode = null;
      const vals = () => { const [p, q] = model.get("obj"); return model.get("verts").map(v => p * v[0] + q * v[1]); };
      const best = () => { const vs = vals(); if (!vs.length) return 0; return model.get("sense") === "min" ? Math.min(...vs) : Math.max(...vs); };
      const btn = (t, f) => { const b = document.createElement("button"); b.textContent = t; b.onclick = f; b.style.marginRight = "4px"; ctl.appendChild(b); };
      btn("snap to optimum", () => { model.set("c", best()); model.save_changes(); });
      btn("c = 0", () => { model.set("c", 0); model.save_changes(); });
      function draw() {
        P.base();
        const [p, q] = model.get("obj"), c = model.get("c"), vs = vals(), poly = model.get("poly");
        const lo = vs.length ? Math.min(...vs) : 0, hi = vs.length ? Math.max(...vs) : 0, tol = 1e-6 * Math.max(1, Math.abs(hi));
        // ghost iso-lines
        for (const f of [0.25, 0.5, 0.75, 1.25]) { const seg = P.clipLine(p, q, f * best() || f * 1000); if (!seg) continue;
          P.mk("line", { x1: P.sx(seg[0][0]), y1: P.sy(seg[0][1]), x2: P.sx(seg[1][0]), y2: P.sy(seg[1][1]), stroke: "#1f77b4", "stroke-opacity": .18, "stroke-dasharray": "4 4" }); }
        const touches = c >= lo - tol && c <= hi + tol;
        const seg = P.clipLine(p, q, c);
        if (seg) P.mk("line", { x1: P.sx(seg[0][0]), y1: P.sy(seg[0][1]), x2: P.sx(seg[1][0]), y2: P.sy(seg[1][1]),
          stroke: "#1f77b4", "stroke-width": 2.5, "stroke-dasharray": touches ? "" : "8 5" });
        // part of the iso-line inside the region
        if (touches && poly.length >= 3) { const hits = [];
          for (let k = 0; k < poly.length; k++) { const A = poly[k], B = poly[(k + 1) % poly.length];
            const fa = p * A[0] + q * A[1] - c, fb = p * B[0] + q * B[1] - c;
            if (Math.abs(fa) < tol) hits.push(A);
            if ((fa < -tol && fb > tol) || (fa > tol && fb < -tol)) { const t = fa / (fa - fb); hits.push([A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1])]); } }
          if (hits.length) { hits.sort((u, v) => u[0] - v[0] || u[1] - v[1]); const a = hits[0], b = hits[hits.length - 1];
            P.mk("line", { x1: P.sx(a[0]), y1: P.sy(a[1]), x2: P.sx(b[0]), y2: P.sy(b[1]), stroke: "#1f77b4", "stroke-width": 7, "stroke-linecap": "round" }); } }
        // vertices
        const opt = best(), atOpt = Math.abs(c - opt) <= Math.max(tol, 0.003 * Math.max(1, Math.abs(opt)));
        model.get("verts").forEach((v, k) => { if (v[0] > P.X() || v[1] > P.Y()) return;
          const isOpt = Math.abs(vs[k] - opt) <= tol;
          P.mk("circle", { cx: P.sx(v[0]), cy: P.sy(v[1]), r: isOpt && atOpt ? 9 : 5, fill: isOpt && atOpt ? "#d62828" : "#2a9d8f", stroke: "white", "stroke-width": 1.5 });
          P.text(P.sx(v[0]) + 8, P.sy(v[1]) - 8, "$" + vs[k].toFixed(0), { "font-size": 11, fill: isOpt && atOpt ? "#d62828" : "currentColor" }); });
        // objective arrow
        let dx = p / K, dy = -q / K; const L = Math.hypot(dx, dy), s = L > 80 ? 80 / L : 1; dx *= s; dy *= s;
        P.mk("circle", { cx: AX, cy: AY, r: 3, fill: "#1f77b4" });
        P.mk("line", { x1: AX, y1: AY, x2: AX + dx, y2: AY + dy, stroke: "#1f77b4", "stroke-width": 3 });
        const tip = P.mk("circle", { cx: AX + dx, cy: AY + dy, r: 9, fill: "#1f77b4", "fill-opacity": .85, stroke: "white", "stroke-width": 2 }); tip.style.cursor = "move";
        P.text(AX - 60, AY + 70, `profit = ${p}x + ${q}y`, { fill: "#1f77b4", "font-size": 12 });
        const msg = !touches ? (c > hi ? "too greedy: no feasible plan earns this" : "below every feasible plan") :
          atOpt ? `<b style="color:#d62828">just kissing the region: ${model.get("sense") === "min" ? "minimum" : "maximum"}!</b>` : "cuts through the region (feasible, but not optimal)";
        info.innerHTML = `c = $${c.toFixed(2)} · ${msg}<br>vertex values range ${lo.toFixed(2)} … ${hi.toFixed(2)}`;
      }
      P.svg.addEventListener("pointerdown", e => { const s = P.toSvg(e), [p, q] = model.get("obj");
        let dx = p / K, dy = -q / K; const L = Math.hypot(dx, dy), k = L > 80 ? 80 / L : 1;
        mode = Math.hypot(s.x - (AX + dx * k), s.y - (AY + dy * k)) < 16 ? "rot" : "sweep";
        P.svg.setPointerCapture(e.pointerId); move(e); });
      function move(e) { const s = P.toSvg(e);
        if (mode === "rot") { const p = Math.round((s.x - AX) * K / 10) * 10, q = Math.round(-(s.y - AY) * K / 10) * 10;
          if (p !== 0 || q !== 0) model.set("obj", [p, q]); }
        else if (mode === "sweep") { const [p, q] = model.get("obj"); model.set("c", p * P.ix(s.x) + q * P.iy(s.y)); } }
      P.svg.addEventListener("pointermove", e => { if (mode) move(e); });
      const stop = () => { if (mode) { mode = null; model.save_changes(); } };
      P.svg.addEventListener("pointerup", stop); P.svg.addEventListener("pointercancel", stop);
      for (const t of ["obj", "c", "cons", "poly", "verts", "sense"]) model.on("change:" + t, draw);
      draw();
    }
    export default { render };
    """
        cons = traitlets.List(traitlets.Dict()).tag(sync=True)
        poly = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)
        verts = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)
        xmax = traitlets.Float(25).tag(sync=True)
        ymax = traitlets.Float(18).tag(sync=True)
        obj = traitlets.List(traitlets.Float(), default_value=[180.0, 200.0]).tag(sync=True)
        c = traitlets.Float(1500.0).tag(sync=True)
        sense = traitlets.Unicode("max").tag(sync=True)

    return (IsoSweeper,)


@app.cell
def sweeper(
    IsoSweeper,
    PLOT_BOX,
    PROFIT,
    carpenter_cons,
    carpenter_poly,
    carpenter_vertices,
    mo,
):
    sweeper = mo.ui.anywidget(IsoSweeper(
        cons=carpenter_cons, poly=carpenter_poly,
        verts=carpenter_vertices[["x", "y"]].values.tolist(),
        xmax=PLOT_BOX[0], ymax=PLOT_BOX[1], obj=list(map(float, PROFIT)), c=1500.0,
    ))
    sweeper
    return (sweeper,)


@app.cell
def sweeper_readout(carpenter_vertices, mo, np, sweeper):
    _p, _q = sweeper.value["obj"]
    _c = sweeper.value["c"]
    _v = carpenter_vertices.assign(value=_p * carpenter_vertices.x + _q * carpenter_vertices.y)
    _mx = _v.value.max()
    _winners = _v[np.isclose(_v.value, _mx)]
    _where = " and ".join(f"({x:.2f}, {y:.2f})" for x, y in zip(_winners.x, _winners.y))
    _status = ("above the maximum: unreachable" if _c > _mx + 1e-6 else
               "at the maximum" if np.isclose(_c, _mx, rtol=3e-3) else
               "below the maximum: money left on the table" if _c >= _v.value.min() else "below every plan")
    mo.vstack([
        mo.hstack([
            mo.stat(f"${_c:,.0f}", label="your iso-value c", caption=_status, bordered=True),
            mo.stat(f"${_mx:,.2f}", label=f"max of {_p:g}x + {_q:g}y", caption=f"at {_where}", bordered=True),
            mo.stat(f"{_p / _q:.2f}" if _q else "∞", label="price ratio p/q", caption="labour line 1.25 · lumber line 0.50", bordered=True),
        ], justify="start"),
        mo.md("**Tie!** Your objective is parallel to an edge, so every point on that edge is optimal, including both end vertices."
              if len(_winners) > 1 else ""),
    ])
    return


@app.cell(hide_code=True)
def s4_try(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Sweep the line from the origin outward. Watch the thick in-region segment **shrink to a single point** just before it leaves the region. That point is the vertex (40/3, 10/3).
    2. Rotate the arrow so tables pay much more (e.g. p = 400, q = 100). The kiss jumps to the **all-tables** corner (16, 0). The optimum moves *between vertices* and never into the middle of an edge.
    3. Set the ratio p/q to exactly **1.25** (e.g. 250 and 200). The iso-line is now parallel to the labour line, and a whole edge ties. The vertices are still optimal, just not uniquely.
    """), kind="info")
    return


@app.cell(hide_code=True)
def s5_md(mo):
    mo.md(r"""
    ## 5 · The big idea, and when it breaks

    > **If the feasible region is a bounded, convex polygon, a linear function reaches its maximum and its minimum at a vertex.**

    So the method is: **list the vertices, evaluate, pick the best.** The video adds two caveats, and there is one more worth knowing:

    * **Ties:** if the iso-line is parallel to an edge, the whole edge is optimal (its end vertices included).
    * **Minimum:** the same picture works. Slide the line *down* until it last touches. For the carpenter that's $c = 0$ at $(0,0)$.
    * **Unbounded:** if the region runs off to infinity in the arrow's direction, no maximum exists. The "vertices only" rule needs the region to be **bounded**.

    Pick a case. Each one uses the same iso-line widget; press **snap to optimum**.
    """)
    return


@app.cell
def edge_case(mo):
    edge_case = mo.ui.dropdown(
        ["Unique vertex (carpenter)", "Tie along an edge", "Minimize instead", "Unbounded region"],
        value="Unique vertex (carpenter)", label="case",
    )
    edge_case
    return (edge_case,)


@app.cell
def edge_widget(IsoSweeper, PLOT_BOX, edge_case, lp_vertices, mo):
    _base = [
        {"a": 5, "b": 4, "r": 80, "label": "labour", "color": "#e76f51"},
        {"a": 10, "b": 20, "r": 200, "label": "lumber", "color": "#8d6e63"},
    ]
    EDGE_CASES = {
        "Unique vertex (carpenter)": (_base, (180, 200), "max"),
        "Tie along an edge": (_base, (250, 200), "max"),
        "Minimize instead": (_base, (180, 200), "min"),
        "Unbounded region": ([
            {"a": -1, "b": -1, "r": -6, "label": "x+y≥6 (min order)", "color": "#e76f51"},
            {"a": -1, "b": 2, "r": 16, "label": "-x+2y≤16", "color": "#8d6e63"},
        ], (180, 200), "max"),
    }
    edge_cons, edge_obj, edge_sense = EDGE_CASES[edge_case.value]
    _poly = lp_vertices(edge_cons, edge_obj, box=PLOT_BOX)
    _verts = lp_vertices(edge_cons, edge_obj)
    _show = _poly if edge_case.value == "Unbounded region" else _verts
    edge_widget = mo.ui.anywidget(IsoSweeper(
        cons=edge_cons, poly=_poly[["x", "y"]].values.tolist(), verts=_show[["x", "y"]].values.tolist(),
        xmax=PLOT_BOX[0], ymax=PLOT_BOX[1], obj=list(map(float, edge_obj)), c=1500.0, sense=edge_sense,
    ))
    edge_widget
    return edge_cons, edge_obj, edge_sense


@app.cell
def edge_readout(
    edge_case,
    edge_cons,
    edge_obj,
    edge_sense,
    lp_solve,
    lp_vertices,
    mo,
):
    _r = lp_solve(edge_cons, edge_obj, edge_sense)
    _v = lp_vertices(edge_cons, edge_obj)
    _notes = {
        "Unique vertex (carpenter)": "One vertex wins outright: **(40/3, 10/3)** at **$3,066.67**.",
        "Tie along an edge": "Profit 250x + 200y = 50·(5x + 4y), which is parallel to the labour line. Every plan using all 80 hours earns "
                             "**$4,000**, so the whole edge from (40/3, 10/3) to (16, 0) ties. A solver just reports one of the vertices.",
        "Minimize instead": "Slide the line down instead of up. The last contact is the vertex **(0, 0)** with **$0**, which matches the video: the minimum is also at a vertex.",
        "Unbounded region": "The green region keeps going up and to the right (it's only clipped by the picture frame). The arrow points into that open end, so "
                            "profit can grow forever: **no maximum exists**. The 'vertex values' shown are artefacts of the frame, not true corners.",
    }
    mo.vstack([
        mo.md(_notes[edge_case.value]),
        mo.md(f"scipy `linprog` says: **{_r['status']}**" + (f", optimum ${_r['value']:,.2f} at ({_r['x'][0]:.2f}, {_r['x'][1]:.2f})" if _r["value"] is not None else "")),
        mo.ui.table(_v.round(2), selection=None, pagination=False, label="true vertices of the region"),
    ])
    return


@app.cell(hide_code=True)
def s5_try(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. In **Tie along an edge**, sweep slowly. At $4,000 the thick in-region segment is the *whole* orange edge, not just a point.
    2. In **Unbounded region**, rotate the arrow to point down-left (negative p and q, so you are minimizing cost). Now the line *does* get a last touch. An unbounded region can still have an optimum if the arrow points away from the open end.
    3. In **Minimize instead**, compare with the carpenter case. The same four vertices are checked; only "best" flips from largest to smallest.
    """), kind="info")
    return


@app.cell(hide_code=True)
def s6_md(mo):
    mo.md(r"""
    ## 6 · Sandbox: write your own linear program

    Edit the constraint table (each row means $a\,x + b\,y \le r$; for a "≥" rule, multiply by −1), set the objective,
    and the region, vertices and iso-line all update. Rows with a = b = 0 are ignored.
    """)
    return


@app.cell
def sandbox_inputs(mo, pd):
    sandbox_table = mo.ui.data_editor(
        pd.DataFrame({
            "a": [5.0, 10.0, 1.0],
            "b": [4.0, 20.0, 0.0],
            "r": [80.0, 200.0, 12.0],
            "label": ["labour", "lumber", "max 12 tables"],
        }),
        label="constraints  a·x + b·y ≤ r",
    )
    sandbox_p = mo.ui.number(-1000, 1000, step=10, value=180, label="p (profit per x)")
    sandbox_q = mo.ui.number(-1000, 1000, step=10, value=200, label="q (profit per y)")
    sandbox_sense = mo.ui.dropdown(["max", "min"], value="max", label="sense")
    mo.vstack([sandbox_table, mo.hstack([sandbox_p, sandbox_q, sandbox_sense], justify="start")])
    return sandbox_p, sandbox_q, sandbox_sense, sandbox_table


@app.cell
def sandbox_widget(
    IsoSweeper,
    lp_vertices,
    mo,
    np,
    pd,
    sandbox_p,
    sandbox_q,
    sandbox_sense,
    sandbox_table,
):
    _df = pd.DataFrame(sandbox_table.value).dropna(subset=["a", "b", "r"])
    _df = _df[(_df.a != 0) | (_df.b != 0)]
    _palette = ["#e76f51", "#8d6e63", "#f4a261", "#9c6644", "#e9c46a", "#b5838d"]
    sandbox_cons = [
        {"a": float(r.a), "b": float(r.b), "r": float(r.r), "label": str(r.label), "color": _palette[k % len(_palette)]}
        for k, r in enumerate(_df.itertuples())
    ]
    sandbox_obj = (float(sandbox_p.value), float(sandbox_q.value))
    _ints_x = [c["r"] / c["a"] for c in sandbox_cons if c["a"] > 0 and c["r"] > 0]
    _ints_y = [c["r"] / c["b"] for c in sandbox_cons if c["b"] > 0 and c["r"] > 0]
    sandbox_box = (float(np.clip(1.3 * max(_ints_x, default=20), 5, 500)), float(np.clip(1.3 * max(_ints_y, default=15), 5, 500)))
    sandbox_vertices = lp_vertices(sandbox_cons, sandbox_obj)
    _poly = lp_vertices(sandbox_cons, sandbox_obj, box=sandbox_box)
    sandbox_widget = mo.ui.anywidget(IsoSweeper(
        cons=sandbox_cons, poly=_poly[["x", "y"]].values.tolist(), verts=sandbox_vertices[["x", "y"]].values.tolist(),
        xmax=sandbox_box[0], ymax=sandbox_box[1], obj=list(sandbox_obj), c=0.0, sense=sandbox_sense.value,
    ))
    sandbox_widget
    return sandbox_cons, sandbox_obj, sandbox_vertices


@app.cell
def sandbox_readout(
    lp_solve,
    mo,
    np,
    sandbox_cons,
    sandbox_obj,
    sandbox_sense,
    sandbox_vertices,
):
    _r = lp_solve(sandbox_cons, sandbox_obj, sandbox_sense.value)
    _v = sandbox_vertices
    if len(_v):
        _best = _v.value.max() if sandbox_sense.value == "max" else _v.value.min()
        _v = _v.assign(optimal=np.where(np.isclose(_v.value, _best), "★", ""))
    _agree = (_r["value"] is not None and len(_v) and np.isclose(_r["value"], _best))
    mo.vstack([
        mo.md(f"**{len(_v)} vertices.** Solver status: **{_r['status']}**" +
              (f", optimum **{_r['value']:,.2f}** at ({_r['x'][0]:.2f}, {_r['x'][1]:.2f}). "
               + ("✅ matches the best vertex." if _agree else "") if _r["value"] is not None else
               (" (the region is unbounded in the objective's direction, so vertices can't settle it)" if _r["status"] == "unbounded" else ""))),
        mo.ui.table(_v.round(3), selection=None, pagination=False) if len(_v) else mo.md("_No feasible region: the constraints contradict each other._"),
    ])
    return


@app.cell(hide_code=True)
def s6_try(mo):
    mo.callout(mo.md(r"""
    **Challenges**
    1. Add a third rule *"at most 12 tables"* (already there: a = 1, b = 0, r = 12). It cuts off the old optimum (40/3, 10/3). Which constraint stops binding, and what's the new best vertex?
    2. Make the region **empty** (e.g. add −x − y ≤ −40, meaning "build at least 40 items"). What does the solver say?
    3. Find an objective where the optimum is the all-bookcases corner. *Hint:* the ratio p/q must drop below the lumber line's slope ratio 0.5.
    """), kind="info")
    return


@app.cell(hide_code=True)
def s7_md(mo):
    mo.md(r"""
    ## 7 · Quiz: check your intuition
    """)
    return


@app.cell
def quiz(mo):
    QUIZ = {
        "q1": ("Where can the maximum of a linear objective over a bounded polygon occur?",
               ["Only at the centre of the region", "At a vertex (possibly a whole edge in a tie)", "Anywhere inside the region", "Only on the x-axis"], 1),
        "q2": ("In the carpenter problem, what is the profit at the all-tables corner (16, 0)?",
               ["$1,800", "$2,000", "$2,880", "$3,067"], 2),
        "q3": ("An iso-value line 180x + 200y = c lies completely outside the feasible region, above it. This means…",
               ["c is the optimum", "no feasible plan earns c", "every plan earns c", "the region is unbounded"], 1),
        "q4": ("The profit changes to 250x + 200y. What happens?",
               ["Unique optimum at (0, 10)", "The whole labour edge ties for the maximum", "No maximum exists", "The optimum moves to (0, 0)"], 1),
        "q5": ("Why does the carpenter's optimum sit where both lines cross?",
               ["Because x = y there", "Any leftover wood or hours could have been turned into more profit", "Because it's the centre of mass", "Coincidence"], 1),
        "q6": ("What can go wrong with 'just check the vertices' if the region is unbounded?",
               ["Nothing", "The objective may increase forever, so there is no maximum", "There are no vertices at all, ever", "The minimum is always 0"], 1),
    }
    quiz = mo.ui.dictionary({k: mo.ui.radio(opts, label=f"**{k[1:]}.** {q}") for k, (q, opts, _) in QUIZ.items()})
    quiz_reveal = mo.ui.switch(label="reveal answers")
    mo.vstack([quiz, quiz_reveal])
    return QUIZ, quiz, quiz_reveal


@app.cell
def quiz_score(QUIZ, mo, quiz, quiz_reveal):
    _ans = quiz.value
    _score = sum(_ans[k] == QUIZ[k][1][QUIZ[k][2]] for k in QUIZ)
    _answered = sum(v is not None for v in _ans.values())
    _rows = []
    for _k, (_q, _opts, _i) in QUIZ.items():
        _mark = "—" if _ans[_k] is None else ("✅" if _ans[_k] == _opts[_i] else "❌")
        _rows.append(f"| {_k[1:]} | {_mark} | {_opts[_i] if quiz_reveal.value else '🔒'} |")
    mo.vstack([
        mo.stat(f"{_score} / {len(QUIZ)}", label="score", caption=f"{_answered} answered", bordered=True),
        mo.md("| # | yours | correct answer |\n|---|---|---|\n" + "\n".join(_rows)) if (_answered or quiz_reveal.value) else mo.md("_Answer some questions to see your marks._"),
        mo.md("🏆 **Perfect!** You can now solve any 2-variable LP with a ruler and a pencil." if _score == len(QUIZ) else ""),
    ])
    return


@app.cell(hide_code=True)
def recap(mo):
    mo.md(r"""
    ## Recap

    | idea | in symbols | what you did |
    |---|---|---|
    | linear program | optimize $px+qy$ subject to $a_ix+b_iy\le r_i$ | turned the carpenter story into $180x+200y$, $5x+4y\le80$, $10x+20y\le200$ |
    | feasible region | intersection of half-planes (a convex polygon) | dragged a plan point and read off slack |
    | vertices | crossings of two boundary lines that satisfy every rule | stepped through all $\binom{m}{2}$ crossings |
    | iso-value line | $px+qy=c$, slides parallel as $c$ changes | swept $c$ until the line *kissed* the region |
    | big idea | bounded region ⇒ max & min at a vertex | carpenter optimum $(40/3,\,10/3)$, **\$3,066.67** |
    | caveats | ties (parallel edge), minimize, unbounded | explored each in section 5 |

    **Next steps:** the **simplex method** (hop vertex → better neighbour rather than checking all of them), **integer programming**
    (the carpenter can't sell a third of a table: the best whole-number plan is (12, 4) for \$2,960, not the rounded (13, 3) at \$2,940, and you can check it in section 1), and **duality**
    (what is one more hour of labour worth?).
    """)
    return


if __name__ == "__main__":
    app.run()

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
def _(mo):
    mo.vstack([
        mo.md(r"""
    # How DAGs change the eight steps of data analysis
    *An interactive companion to **Simplistics (QuantPsych)**'s video "How DAGs Change the Eight Steps of Data Analysis".*

    > **DAGs tell us *what to adjust for*. The eight steps tell us *what model to fit*. We need both.**

    The video walks through the eight steps (hypothesis, measurement, univariate plots, plot the model,
    residuals, estimates, decision, replication) and asks of each one: *does a DAG help here?*
    We follow the same running example: **does fertilizer make plants grow, controlling for rain?**

    $$\text{Growth} = \beta_0 + \beta_1\,\text{Rain} + \beta_2\,\text{Fertilizer} + \varepsilon \qquad (\beta_2 \text{ is the parameter of interest})$$
    """),
        mo.Html('<iframe width="100%" height="380" src="https://www.youtube.com/embed/aiDuqoNoyNI" title="How DAGs Change the Eight Steps of Data Analysis" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Colour language (used everywhere below)

    | colour | meaning |
    |---|---|
    | <span style="color:#3b82f6">**■ blue**</span> | the **exposure / treatment** X (fertilizer) and estimates of its effect |
    | <span style="color:#f18f01">**■ orange**</span> | the **outcome** Y (plant growth) |
    | <span style="color:#2a9d8f">**■ green**</span> | **causal** paths and the **true** effect |
    | <span style="color:#e4572e">**■ red**</span> | **open non-causal (biasing)** paths and biased estimates |
    | **▢ square node** | a variable we **adjust for** (condition on), the usual DAG convention |
    | <span style="color:#7b61ff">**■ purple**</span> | the **direct** effect (when a mediator is adjusted for) |

    **Map of the lesson:** §1 steps 1–2 (build the DAG, find the adjustment set) · §2 confounder/collider/mediator simulation ·
    §3 measurement · §4 steps 3–5 (plots and residuals, where the DAG is silent) · §5 steps 6–7 (estimand and decision) · §6 step 8 (replication) · sandbox · quiz · recap.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Steps 1–2 (revised): construct the DAG, then read off the adjustment set

    The original step 1 was "state the hypothesis in words, and be specific". The video argues that **DAGs force even more specificity**: you must name the
    **exposure**, the **outcome**, and your hypothesis about the *whole* causal structure. That structure tells you what to adjust for and what **not** to adjust for.

    The video's fertilizer DAG:

    * **Fertilizer → Root growth → Growth**: root growth is a **mediator**.
    * **Rain** causes both fertilizer use and growth, so it is a **confounder** and opens a *back-door path* Fertilizer ← Rain → Growth.
    * **Soil nitrogen** is caused by fertilizer *and* soil quality, so it is a **collider**. *"We wouldn't want to condition on that."*

    **The back-door criterion.** A set $Z$ identifies the total effect of $X$ on $Y$ if (i) no member of $Z$ is a descendant of $X$, and
    (ii) $Z$ blocks every path between $X$ and $Y$ that starts with an arrow *into* $X$. A path is **blocked** by a non-collider in $Z$, or by a collider
    that is *not* in $Z$ and has no descendant in $Z$.

    **Editor controls.** Drag nodes in *Move* mode. Pick a mode, then click nodes: *Edge* (click A, then B, to toggle A→B), *Exposure*, *Outcome*,
    *Adjust* (toggle the square), *Delete*. Type a name and press **＋ Node** to add a variable. Paths are coloured live in JS; Python computes the
    authoritative adjustment sets underneath.
    """)
    return


@app.cell
def dag_tools(np):
    def _graph(nodes, edges):
        ids = [n["id"] for n in nodes]
        ch = {i: [] for i in ids}
        pa = {i: [] for i in ids}
        for a, b in edges:
            if a in ch and b in ch:
                ch[a].append(b)
                pa[b].append(a)
        return ids, ch, pa

    def _descendants(ch, n):
        seen, stack = {n}, [n]
        while stack:
            u = stack.pop()
            for v in ch[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    def dag_paths(nodes, edges, X, Y, Z=(), max_paths=400):
        """All simple paths X ... Y in the skeleton, each classified as causal / back-door / other, open / blocked given Z."""
        ids, ch, pa = _graph(nodes, edges)
        Z = set(Z)
        if X not in ch or Y not in ch or X == Y:
            return []
        found = []
        def dfs(u, path):
            if len(found) >= max_paths:
                return
            if u == Y:
                found.append(list(path))
                return
            for v in dict.fromkeys(ch[u] + pa[u]):
                if v not in path:
                    path.append(v)
                    dfs(v, path)
                    path.pop()
        dfs(X, [X])
        out = []
        for p in found:
            is_open, blockers = True, []
            for i in range(1, len(p) - 1):
                a, m, b = p[i - 1], p[i], p[i + 1]
                collider = m in ch[a] and m in ch[b]
                if collider:
                    if not (_descendants(ch, m) & Z):
                        is_open = False
                        blockers.append(f"{m} (collider)")
                elif m in Z:
                    is_open = False
                    blockers.append(m)
            causal = all(p[i] in ch[p[i - 1]] for i in range(1, len(p)))
            kind = "causal" if causal else ("back-door" if p[1] in pa[X] else "non-causal")
            arrows = p[0]
            for i in range(1, len(p)):
                arrows += (" → " if p[i] in ch[p[i - 1]] else " ← ") + p[i]
            out.append(dict(path=arrows, nodes=p, kind=kind, open=is_open, blocked_by=", ".join(blockers)))
        return out

    def backdoor_ok(nodes, edges, X, Y, Z):
        ids, ch, pa = _graph(nodes, edges)
        if set(Z) & (_descendants(ch, X) - {X}):
            return False
        return not any(p["open"] for p in dag_paths(nodes, edges, X, Y, Z) if p["kind"] == "back-door")

    def adjustment_sets(nodes, edges, X, Y, max_candidates=12):
        """Minimal back-door adjustment sets (brute force; fine for lesson-sized DAGs)."""
        from itertools import combinations
        ids, ch, pa = _graph(nodes, edges)
        if X not in ch or Y not in ch or X == Y:
            return []
        cand = [i for i in ids if i not in (X, Y) and i not in _descendants(ch, X)][:max_candidates]
        minimal = []
        for k in range(len(cand) + 1):
            for Zs in combinations(cand, k):
                if any(set(m) <= set(Zs) for m in minimal):
                    continue
                if backdoor_ok(nodes, edges, X, Y, Zs):
                    minimal.append(Zs)
        return [list(m) for m in minimal]

    def classify_set(nodes, edges, X, Y, Z):
        """What does adjusting for Z estimate? 'total', 'direct', or 'biased' (path-based, linear world)."""
        ps = dag_paths(nodes, edges, X, Y, Z)
        if any(p["open"] and p["kind"] != "causal" for p in ps):
            return "biased"
        causal = [p for p in ps if p["kind"] == "causal"]
        if all(p["open"] for p in causal):
            return "total"
        if [p for p in causal if p["open"]] == [p for p in causal if len(p["nodes"]) == 2] and any(len(p["nodes"]) == 2 for p in causal):
            return "direct"
        return "partial"

    def ols(y, cols):
        """OLS with intercept. cols: dict name -> array. Returns (coef dict, se dict)."""
        names = list(cols)
        A = np.column_stack([np.ones(len(y))] + [np.asarray(cols[c], float) for c in names])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        s2 = resid @ resid / max(len(y) - A.shape[1], 1)
        cov = s2 * np.linalg.pinv(A.T @ A)
        se = np.sqrt(np.diag(cov))
        return dict(zip(["const"] + names, beta)), dict(zip(["const"] + names, se))

    return adjustment_sets, backdoor_ok, classify_set, dag_paths, ols


@app.cell
def dag_editor_class(anywidget, traitlets):
    class DagEditor(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 680, H = 380, NS = "http://www.w3.org/2000/svg";
      const C = { X: "#3b82f6", Y: "#f18f01", causal: "#2a9d8f", bias: "#e4572e", idle: "#9aa0a6" };
      const wrap = document.createElement("div"); wrap.style.cssText = "font:13px system-ui,sans-serif";
      el.appendChild(wrap);
      const bar = document.createElement("div"); bar.style.cssText = "display:flex;flex-wrap:wrap;gap:4px;margin-bottom:4px;align-items:center";
      const bar2 = document.createElement("div"); bar2.style.cssText = bar.style.cssText;
      wrap.append(bar, bar2);
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none",
        border: "1px solid rgba(128,128,128,.35)", borderRadius: "8px" });
      wrap.appendChild(svg);
      const info = document.createElement("div"); info.style.cssText = "margin-top:6px;line-height:1.5"; wrap.appendChild(info);
      const mk = (tag, a, p = svg) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
      const toSvg = (e) => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      let mode = "move", pending = null, drag = null, msg = "";
      const btn = (parent, label, fn) => { const b = document.createElement("button"); b.textContent = label; b.onclick = fn;
        b.style.cssText = "padding:3px 8px;border-radius:6px;border:1px solid #999;cursor:pointer;background:transparent;color:inherit"; parent.appendChild(b); return b; };
      const modeBtns = {};
      [["move", "✋ Move"], ["edge", "➜ Edge"], ["exposure", "X Exposure"], ["outcome", "Y Outcome"], ["adjust", "▢ Adjust"], ["delete", "🗑 Delete"]]
        .forEach(([m, l]) => { modeBtns[m] = btn(bar, l, () => { mode = m; pending = null; msg = ""; draw(); }); });
      const inp = document.createElement("input"); inp.placeholder = "new node name"; inp.size = 12;
      inp.style.cssText = "padding:3px;border-radius:6px;border:1px solid #999;background:transparent;color:inherit"; bar.appendChild(inp);
      const addNode = () => { const name = inp.value.trim(); if (!name) return; const ns = [...model.get("nodes")];
        if (ns.some(n => n.id === name)) { msg = "That name already exists."; draw(); return; }
        ns.push({ id: name, x: 60 + Math.random() * (W - 120), y: 40 + Math.random() * 60 }); inp.value = "";
        model.set("nodes", ns); model.save_changes(); draw(); };
      btn(bar, "＋ Node", addNode); inp.addEventListener("keydown", e => { if (e.key === "Enter") addNode(); });
      const presets = model.get("presets") || {};
      if (Object.keys(presets).length) { const s = document.createElement("span"); s.textContent = "Presets:"; s.style.opacity = .7; bar2.appendChild(s); }
      for (const name in presets) btn(bar2, name, () => { const p = presets[name];
        for (const k of ["nodes", "edges", "exposure", "outcome", "adjusted"]) model.set(k, JSON.parse(JSON.stringify(p[k])));
        model.save_changes(); pending = null; msg = ""; draw(); });

      function graph() {
        const ids = model.get("nodes").map(n => n.id), ch = {}, pa = {};
        ids.forEach(n => { ch[n] = []; pa[n] = []; });
        model.get("edges").forEach(([a, b]) => { if (a in ch && b in ch) { ch[a].push(b); pa[b].push(a); } });
        return { ids, ch, pa };
      }
      const desc = (ch, n) => { const s = new Set([n]), st = [n]; while (st.length) { const u = st.pop(); ch[u].forEach(v => { if (!s.has(v)) { s.add(v); st.push(v); } }); } return s; };
      function analyze() {
        const { ch, pa } = graph(), X = model.get("exposure"), Y = model.get("outcome"), Z = new Set(model.get("adjusted"));
        const paths = [];
        if (!(X in ch) || !(Y in ch) || X === Y) return paths;
        const dfs = (u, path) => { if (paths.length > 300) return; if (u === Y) { paths.push([...path]); return; }
          for (const v of new Set([...ch[u], ...pa[u]])) { if (path.includes(v)) continue; path.push(v); dfs(v, path); path.pop(); } };
        dfs(X, [X]);
        return paths.map(p => { let open = true;
          for (let i = 1; i < p.length - 1; i++) { const a = p[i - 1], m = p[i], b = p[i + 1];
            if (ch[a].includes(m) && ch[b].includes(m)) { if (![...desc(ch, m)].some(z => Z.has(z))) open = false; }
            else if (Z.has(m)) open = false; }
          const causal = p.every((n, i) => i === 0 || ch[p[i - 1]].includes(n));
          let txt = p[0]; for (let i = 1; i < p.length; i++) txt += (ch[p[i - 1]].includes(p[i]) ? " → " : " ← ") + p[i];
          return { p, open, causal, txt }; });
      }
      const nodeW = id => Math.max(64, id.length * 8 + 22), NH = 34;
      function draw() {
        for (const m in modeBtns) modeBtns[m].style.fontWeight = m === mode ? "700" : "400",
          modeBtns[m].style.background = m === mode ? "rgba(59,130,246,.18)" : "transparent";
        svg.innerHTML = "";
        const nodes = model.get("nodes"), pos = {}; nodes.forEach(n => pos[n.id] = n);
        const X = model.get("exposure"), Y = model.get("outcome"), Z = new Set(model.get("adjusted"));
        const res = analyze(), edgeCol = {};
        res.forEach(r => { if (!r.open) return; for (let i = 1; i < r.p.length; i++) {
          const k = [r.p[i - 1], r.p[i]].sort().join("|");
          if (!r.causal) edgeCol[k] = C.bias; else if (!edgeCol[k]) edgeCol[k] = C.causal; } });
        model.get("edges").forEach(([a, b]) => { const A = pos[a], B = pos[b]; if (!A || !B) return;
          const col = edgeCol[[a, b].sort().join("|")] || C.idle;
          const dx = B.x - A.x, dy = B.y - A.y, L = Math.hypot(dx, dy) || 1, ux = dx / L, uy = dy / L;
          const clip = (n, s) => { const hw = nodeW(n.id) / 2 + 3, hh = NH / 2 + 3; const t = Math.min(Math.abs(ux) > 1e-6 ? hw / Math.abs(ux) : 1e9, Math.abs(uy) > 1e-6 ? hh / Math.abs(uy) : 1e9); return [n.x + s * ux * t, n.y + s * uy * t]; };
          const [x1, y1] = clip(A, 1), [x2, y2] = clip(B, -1);
          mk("line", { x1, y1, x2: x2 - ux * 8, y2: y2 - uy * 8, stroke: col, "stroke-width": col === C.idle ? 2 : 3.5 });
          mk("polygon", { points: `${x2},${y2} ${x2 - ux * 13 - uy * 6},${y2 - uy * 13 + ux * 6} ${x2 - ux * 13 + uy * 6},${y2 - uy * 13 - ux * 6}`, fill: col });
        });
        nodes.forEach(n => { const w = nodeW(n.id), isZ = Z.has(n.id);
          const fill = n.id === X ? C.X : n.id === Y ? C.Y : "rgba(150,150,150,.18)";
          const g = mk("g", { transform: `translate(${n.x},${n.y})` }); g.style.cursor = mode === "move" ? "grab" : "pointer";
          mk("rect", { x: -w / 2, y: -NH / 2, width: w, height: NH, rx: isZ ? 2 : NH / 2, fill,
            stroke: pending === n.id ? "#f5c518" : isZ ? "currentColor" : "rgba(128,128,128,.6)", "stroke-width": pending === n.id ? 4 : isZ ? 3 : 1.5 }, g);
          const t = mk("text", { "text-anchor": "middle", "dominant-baseline": "central", "font-size": 13, "font-weight": isZ ? 700 : 500,
            fill: (n.id === X || n.id === Y) ? "white" : "currentColor" }, g); t.textContent = n.id;
          g.addEventListener("pointerdown", e => { e.stopPropagation(); onNode(n.id, e); });
        });
        const nOpenBias = res.filter(r => r.open && !r.causal).length;
        info.innerHTML = (msg ? `<div style="color:${C.bias}">${msg}</div>` : "") +
          `<b>X</b> = <span style="color:${C.X}">${X || "?"}</span>, <b>Y</b> = <span style="color:${C.Y}">${Y || "?"}</span>, ` +
          `<b>adjusted</b> = {${[...Z].join(", ")}} · <b style="color:${nOpenBias ? C.bias : C.causal}">${nOpenBias} open biasing path(s)</b>` +
          (mode === "edge" ? ` · <i>${pending ? "now click the child of " + pending : "click the parent node"}</i>` : "") +
          "<br>" + res.map(r => `<span style="color:${!r.open ? C.idle : r.causal ? C.causal : C.bias};${r.open ? "" : "text-decoration:line-through"}">${r.txt}</span>`).join(" &nbsp;|&nbsp; ");
      }
      function onNode(id, e) {
        msg = "";
        if (mode === "move") { drag = id; svg.setPointerCapture(e.pointerId); return; }
        if (mode === "exposure") { model.set("exposure", id); if (model.get("outcome") === id) model.set("outcome", ""); }
        else if (mode === "outcome") { model.set("outcome", id); if (model.get("exposure") === id) model.set("exposure", ""); }
        else if (mode === "adjust") { const z = new Set(model.get("adjusted")); z.has(id) ? z.delete(id) : z.add(id); model.set("adjusted", [...z]); }
        else if (mode === "delete") { model.set("nodes", model.get("nodes").filter(n => n.id !== id));
          model.set("edges", model.get("edges").filter(([a, b]) => a !== id && b !== id));
          model.set("adjusted", model.get("adjusted").filter(z => z !== id));
          if (model.get("exposure") === id) model.set("exposure", ""); if (model.get("outcome") === id) model.set("outcome", ""); }
        else if (mode === "edge") {
          if (!pending) { pending = id; draw(); return; }
          if (pending !== id) { const a = pending, b = id, E = model.get("edges");
            const has = E.findIndex(([p, q]) => p === a && q === b);
            if (has >= 0) model.set("edges", E.filter((_, i) => i !== has));
            else if (desc(graph().ch, b).has(a)) msg = `Refused: ${a} → ${b} would create a cycle (DAGs are acyclic).`;
            else model.set("edges", [...E.filter(([p, q]) => !(p === b && q === a)), [a, b]]); }
          pending = null; }
        model.save_changes(); draw();
      }
      svg.addEventListener("pointermove", e => { if (!drag) return; const p = toSvg(e);
        model.set("nodes", model.get("nodes").map(n => n.id === drag ? { ...n, x: Math.max(40, Math.min(W - 40, p.x)), y: Math.max(20, Math.min(H - 20, p.y)) } : n)); draw(); });
      const stop = () => { if (drag) { drag = null; model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointerleave", stop);
      svg.addEventListener("pointerdown", () => { if (pending) { pending = null; draw(); } });
      for (const k of ["nodes", "edges", "exposure", "outcome", "adjusted"]) model.on("change:" + k, draw);
      draw();
    }
    export default { render };
    """
        nodes = traitlets.List(traitlets.Dict()).tag(sync=True)
        edges = traitlets.List(traitlets.List(traitlets.Unicode())).tag(sync=True)
        exposure = traitlets.Unicode("").tag(sync=True)
        outcome = traitlets.Unicode("").tag(sync=True)
        adjusted = traitlets.List(traitlets.Unicode()).tag(sync=True)
        presets = traitlets.Dict().tag(sync=True)

    return (DagEditor,)


@app.cell
def dag_presets():
    def _p(nodes, edges, X, Y, Z=()):
        return dict(nodes=[dict(id=i, x=float(x), y=float(y)) for i, x, y in nodes],
                    edges=[list(e) for e in edges], exposure=X, outcome=Y, adjusted=list(Z))

    DAG_PRESETS = {
        "Fertilizer (video)": _p(
            [("Rain", 340, 50), ("Fertilizer", 110, 190), ("Root", 340, 190), ("Growth", 570, 190),
             ("Soil", 450, 320), ("Nitrogen", 220, 320)],
            [("Rain", "Fertilizer"), ("Rain", "Growth"), ("Fertilizer", "Root"), ("Root", "Growth"),
             ("Fertilizer", "Growth"), ("Fertilizer", "Nitrogen"), ("Soil", "Nitrogen"), ("Soil", "Growth")],
            "Fertilizer", "Growth"),
        "Confounder": _p([("Rain", 340, 60), ("Fertilizer", 140, 250), ("Growth", 540, 250)],
                         [("Rain", "Fertilizer"), ("Rain", "Growth"), ("Fertilizer", "Growth")], "Fertilizer", "Growth"),
        "Mediator": _p([("Fertilizer", 120, 190), ("Root", 340, 80), ("Growth", 560, 190)],
                       [("Fertilizer", "Root"), ("Root", "Growth"), ("Fertilizer", "Growth")], "Fertilizer", "Growth"),
        "Collider": _p([("Fertilizer", 140, 100), ("Growth", 540, 100), ("Nitrogen", 340, 300), ("Soil", 540, 300)],
                       [("Fertilizer", "Nitrogen"), ("Soil", "Nitrogen"), ("Soil", "Growth"), ("Fertilizer", "Growth")],
                       "Fertilizer", "Growth"),
        "Proxy rainfall (step 2)": _p([("Rain", 340, 60), ("Rain_measured", 560, 60), ("Fertilizer", 140, 260), ("Growth", 540, 260)],
                         [("Rain", "Fertilizer"), ("Rain", "Growth"), ("Fertilizer", "Growth"), ("Rain", "Rain_measured")],
                         "Fertilizer", "Growth", ["Rain_measured"]),
        "M-bias": _p([("U1", 120, 60), ("U2", 560, 60), ("M", 340, 170), ("Fertilizer", 120, 310), ("Growth", 560, 310)],
                     [("U1", "M"), ("U2", "M"), ("U1", "Fertilizer"), ("U2", "Growth"), ("Fertilizer", "Growth")],
                     "Fertilizer", "Growth"),
    }
    return (DAG_PRESETS,)


@app.cell
def dag_editor(DAG_PRESETS, DagEditor, mo):
    dag_editor = mo.ui.anywidget(DagEditor(**DAG_PRESETS["Fertilizer (video)"], presets=DAG_PRESETS))
    dag_editor
    return (dag_editor,)


@app.cell
def dag_readout(
    adjustment_sets,
    backdoor_ok,
    classify_set,
    dag_editor,
    dag_paths,
    mo,
    pd,
):
    _v = dag_editor.value
    _X, _Y, _Z = _v["exposure"], _v["outcome"], _v["adjusted"]
    _paths = dag_paths(_v["nodes"], _v["edges"], _X, _Y, _Z)
    _sets = adjustment_sets(_v["nodes"], _v["edges"], _X, _Y)
    if not _paths:
        _out = mo.callout(mo.md("Pick an **exposure** and an **outcome** connected by at least one path."), kind="warn")
    else:
        _df = pd.DataFrame([{"path": p["path"], "type": p["kind"], "status": "🟢 open" if p["open"] else "⛔ blocked",
                             "blocked by": p["blocked_by"]} for p in _paths])
        _ok = backdoor_ok(_v["nodes"], _v["edges"], _X, _Y, _Z)
        _what = classify_set(_v["nodes"], _v["edges"], _X, _Y, _Z)
        _verdict = {
            "total": ("success", f"Adjusting for {{{', '.join(_Z)}}} identifies the **total effect** of {_X} on {_Y}."),
            "direct": ("info", f"Adjusting for {{{', '.join(_Z)}}} blocks the mediated path(s): you are estimating the **direct effect**, not the total effect."),
            "partial": ("warn", "Some causal paths are blocked and others are open: the estimate is a mix of direct and indirect effects."),
            "biased": ("danger", f"At least one **non-causal path is open** given {{{', '.join(_Z)}}}: the estimate is **biased**."),
        }[_what]
        _sets_md = ", ".join("{" + ", ".join(s) + "}" if s else "∅ (no adjustment needed)" for s in _sets) or "none: the effect is **not identifiable** by back-door adjustment with observed variables"
        _out = mo.vstack([
            mo.hstack([
                mo.stat(len(_paths), label="paths X…Y"),
                mo.stat(sum(p["open"] and p["kind"] != "causal" for p in _paths), label="open biasing paths"),
                mo.stat("yes" if _ok else "no", label="back-door criterion met?"),
            ], justify="start"),
            mo.md(f"**Minimal back-door adjustment sets:** {_sets_md}"),
            mo.callout(mo.md(_verdict[1]), kind=_verdict[0]),
            mo.ui.table(_df, selection=None, show_column_summaries=False),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. In the fertilizer preset, **Adjust** for *Rain*. The red back-door path turns grey and the minimal set {Rain} is satisfied.
    2. Now also adjust for *Nitrogen*. A new red path appears, **Fertilizer → Nitrogen ← Soil → Growth**, because conditioning on a collider *opens* it. Then adjust for *Soil* as well and watch it close again.
    3. Adjust for *Root*. The verdict switches to **direct effect**: the video's "control for root growth *if we want the direct effect and not the total effect*".
    4. Load **M-bias**. *M* looks like a pre-treatment "control variable", but adjusting for it creates bias.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · Confounder, mediator, collider: what each choice does to $\hat\beta_2$

    A DAG is a claim about how the data were generated, so we can **simulate** the fertilizer world and fit
    $\text{Growth} \sim \text{Fertilizer} + Z$ for **every** possible adjustment set $Z \subseteq \{\text{Rain, Root, Nitrogen, Soil}\}$.

    Ground truth in this simulated world (growth in **%**):

    * direct effect Fertilizer → Growth = **15 %**
    * indirect effect via roots = 1.0 × 20 = **20 %**
    * **total effect = 35 %**. The video's decision threshold is **30 %**, the growth needed to pay for the fertilizer.

    The bar for **your current adjustment set in the §1 editor** is outlined in black, so toggling ▢ *Adjust* up there moves the highlight here.
    (Only Rain, Root, Nitrogen and Soil exist in this simulated world.)
    """)
    return


@app.cell
def fert_controls(mo):
    conf_fert = mo.ui.slider(0, 3, step=0.25, value=1.5, label="Rain → Fertilizer (log-odds per SD)", show_value=True)
    conf_growth = mo.ui.slider(-20, 20, step=1, value=12, label="Rain → Growth (% per SD)", show_value=True)
    collider_strength = mo.ui.slider(0, 30, step=1, value=15, label="Soil → Growth (% per SD)", show_value=True)
    reroll_fert = mo.ui.button(label="🎲 new sample", value=0, on_click=lambda v: v + 1)
    mo.hstack([mo.vstack([conf_fert, conf_growth, collider_strength]), reroll_fert], justify="start", gap=2)
    return collider_strength, conf_fert, conf_growth, reroll_fert


@app.cell
def fert_sim(collider_strength, conf_fert, conf_growth, np, pd, reroll_fert):
    def simulate_fertilizer(n=3000, a_rain_f=1.5, b_rain_g=12.0, b_soil_g=15.0, seed=0):
        """The video's fertilizer DAG as a linear structural model (growth in %)."""
        _rng = np.random.default_rng(seed)
        rain = _rng.normal(size=n)
        soil = _rng.normal(size=n)
        fert = (_rng.random(n) < 1 / (1 + np.exp(-a_rain_f * rain))).astype(float)
        root = 1.0 * fert + _rng.normal(0, 0.7, n)
        nitrogen = 1.0 * fert + 1.0 * soil + _rng.normal(0, 0.5, n)
        growth = 50 + 15 * fert + 20 * root + b_rain_g * rain + b_soil_g * soil + _rng.normal(0, 10, n)
        return pd.DataFrame(dict(Rain=rain, Soil=soil, Fertilizer=fert, Root=root, Nitrogen=nitrogen, Growth=growth))

    fert_df = simulate_fertilizer(a_rain_f=conf_fert.value, b_rain_g=conf_growth.value,
                                  b_soil_g=collider_strength.value, seed=reroll_fert.value)
    fert_df.head()
    return fert_df, simulate_fertilizer


@app.cell
def fert_chart(
    DAG_PRESETS,
    alt,
    classify_set,
    dag_editor,
    fert_df,
    mo,
    ols,
    pd,
):
    from itertools import combinations as _comb
    _G = DAG_PRESETS["Fertilizer (video)"]
    _rows = []
    for _k in range(5):
        for _Zs in _comb(["Rain", "Root", "Nitrogen", "Soil"], _k):
            _coef, _se = ols(fert_df["Growth"].to_numpy(), {c: fert_df[c] for c in ("Fertilizer",) + _Zs})
            _cls = classify_set(_G["nodes"], _G["edges"], "Fertilizer", "Growth", _Zs)
            _rows.append(dict(Z="{" + ", ".join(_Zs) + "}" if _Zs else "∅ (naive)", estimate=_coef["Fertilizer"],
                              lo=_coef["Fertilizer"] - 1.96 * _se["Fertilizer"], hi=_coef["Fertilizer"] + 1.96 * _se["Fertilizer"],
                              identifies={"total": "total effect", "direct": "direct effect"}.get(_cls, "biased"),
                              zset=frozenset(_Zs)))
    fert_estimates = pd.DataFrame(_rows)
    _current = frozenset(z for z in dag_editor.value["adjusted"] if z in {"Rain", "Root", "Nitrogen", "Soil"})
    fert_estimates["selected"] = fert_estimates["zset"] == _current
    _plot = fert_estimates.drop(columns="zset")
    _scale = alt.Scale(domain=["total effect", "direct effect", "biased"], range=["#2a9d8f", "#7b61ff", "#e4572e"])
    _base = alt.Chart(_plot).encode(y=alt.Y("Z:N", sort=None, title="adjustment set Z"))
    _bars = _base.mark_bar(height=14).encode(
        x=alt.X("estimate:Q", title="estimated effect of Fertilizer on Growth (%)"),
        color=alt.Color("identifies:N", scale=_scale, title="this Z identifies"),
        stroke=alt.condition("datum.selected", alt.value("black"), alt.value(None)),
        strokeWidth=alt.condition("datum.selected", alt.value(3), alt.value(0)),
        tooltip=["Z", alt.Tooltip("estimate:Q", format=".1f"), alt.Tooltip("lo:Q", format=".1f"), alt.Tooltip("hi:Q", format=".1f"), "identifies"])
    _err = _base.mark_rule().encode(x="lo:Q", x2="hi:Q")
    _truth = alt.Chart(pd.DataFrame(dict(v=[35, 15], lab=["total = 35", "direct = 15"]))).mark_rule(strokeDash=[6, 4], size=2).encode(
        x="v:Q", color=alt.Color("lab:N", scale=alt.Scale(domain=["total = 35", "direct = 15"], range=["#2a9d8f", "#7b61ff"]), title="truth"))
    _sel = fert_estimates[fert_estimates.selected]
    mo.vstack([
        (_bars + _err + _truth).properties(width="container", height=420).resolve_scale(color="independent"),
        mo.md(f"**Your §1 adjustment set** {_sel.Z.iloc[0]} → β̂₂ = **{_sel.estimate.iloc[0]:.1f} %** ({_sel.identifies.iloc[0]})." if len(_sel)
              else "_Your §1 adjustment set uses variables that are not in this simulated world._"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Find the **naive** bar (∅). With the default confounding it overshoots 35 %. Set *Rain → Growth* to **0** and the naive estimate becomes fine, because a confounder needs *both* arrows.
    2. Compare **{Rain}** with **{Rain, Nitrogen}**. Conditioning on the collider drags β̂₂ away from the truth. Raise *Soil → Growth* to make the collider bias bigger.
    3. Every **purple** bar contains *Root*. Those bars estimate the 15 % direct effect, which sits well below the 30 % threshold even though the total effect (35 %) clears it. *Which estimand you choose changes the business decision.*
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Step 2: measurement. Which variable is measured badly matters

    The original step 2 said: check that your measures are reliable and valid. The video's example is *weighing the pot*: you also weigh the water in the soil.
    What it did not say is **how bad measurement changes causal estimates**. DAGs can tell us:

    * **Noisy outcome** (Fertilizer → *True growth* → *Measured growth*): no back-door path opens. The estimate stays **unbiased** but gets **less precise**.
    * **Noisy confounder** (Rain → *Measured rainfall*, e.g. regional historical averages): adjusting for the proxy **cannot close** the back door Fertilizer ← Rain → Growth.
      Some confounding leaks through (*residual confounding*). Load the *Proxy rainfall* preset in §1: the tool marks the set as fine only because it assumes *Rain* itself is observed.

    The **reliability** of the rain proxy is $\rho = \operatorname{Var}(\text{Rain}) / \operatorname{Var}(\text{Rain}_{\text{measured}})$.
    """)
    return


@app.cell
def measurement_controls(mo):
    rel_rain = mo.ui.slider(0.05, 1.0, step=0.05, value=0.5, label="reliability of measured rainfall ρ", show_value=True)
    growth_noise = mo.ui.slider(0, 60, step=5, value=20, label="extra noise in measured growth (SD, %)", show_value=True)
    mo.hstack([rel_rain, growth_noise], justify="start", gap=2)
    return growth_noise, rel_rain


@app.cell
def measurement_chart(
    alt,
    growth_noise,
    mo,
    np,
    ols,
    pd,
    rel_rain,
    simulate_fertilizer,
):
    def _measure(rho, noise_sd, n=4000, seed=7):
        _d = simulate_fertilizer(n=n, seed=seed)
        _rng = np.random.default_rng(seed + 1)
        _rain_m = _d["Rain"] + _rng.normal(0, np.sqrt(1 / rho - 1), n)
        _g_m = _d["Growth"] + _rng.normal(0, noise_sd, n)
        _c1, _s1 = ols(_g_m.to_numpy(), dict(Fertilizer=_d["Fertilizer"], Rain=_d["Rain"]))
        _c2, _s2 = ols(_d["Growth"].to_numpy(), dict(Fertilizer=_d["Fertilizer"], Rain=_rain_m))
        return _c1["Fertilizer"], _s1["Fertilizer"], _c2["Fertilizer"], _s2["Fertilizer"]

    _grid = []
    for _r in np.arange(0.05, 1.0001, 0.05):
        _a, _sa, _b, _sb = _measure(_r, growth_noise.value)
        _grid.append(dict(rho=_r, est=_b, lo=_b - 1.96 * _sb, hi=_b + 1.96 * _sb))
    _grid = pd.DataFrame(_grid)
    _noise = []
    for _ns in range(0, 61, 5):
        _a, _sa, _b, _sb = _measure(rel_rain.value, _ns)
        _noise.append(dict(noise=_ns, est=_a, lo=_a - 1.96 * _sa, hi=_a + 1.96 * _sa))
    _noise = pd.DataFrame(_noise)
    _a, _sa, _b, _sb = _measure(rel_rain.value, growth_noise.value)

    _truth = alt.Chart(pd.DataFrame(dict(v=[35]))).mark_rule(color="#2a9d8f", strokeDash=[6, 4], size=2).encode(y="v:Q")
    _y = alt.Y("est:Q", title="β̂₂ (%)", scale=alt.Scale(domain=[20, 55]))
    _left = (alt.Chart(_noise).mark_area(opacity=.2, color="#3b82f6").encode(x=alt.X("noise:Q", title="noise added to measured growth (SD)"), y=alt.Y("lo:Q", scale=alt.Scale(domain=[20, 55])), y2="hi:Q")
             + alt.Chart(_noise).mark_line(color="#3b82f6").encode(x="noise:Q", y=_y)
             + alt.Chart(pd.DataFrame(dict(noise=[growth_noise.value], est=[_a]))).mark_point(size=120, filled=True, color="black").encode(x="noise:Q", y="est:Q")
             + _truth).properties(width=330, height=260, title="Noisy OUTCOME: unbiased, wider CI")
    _right = (alt.Chart(_grid).mark_area(opacity=.2, color="#e4572e").encode(x=alt.X("rho:Q", title="reliability ρ of measured rainfall"), y=alt.Y("lo:Q", scale=alt.Scale(domain=[20, 55])), y2="hi:Q")
              + alt.Chart(_grid).mark_line(color="#e4572e").encode(x="rho:Q", y=_y)
              + alt.Chart(pd.DataFrame(dict(rho=[rel_rain.value], est=[_b]))).mark_point(size=120, filled=True, color="black").encode(x="rho:Q", y="est:Q")
              + _truth).properties(width=330, height=260, title="Noisy CONFOUNDER: biased, back door stays ajar")
    mo.vstack([
        alt.hconcat(_left, _right),
        mo.hstack([
            mo.stat(f"{_a:.1f} ± {1.96*_sa:.1f}", label="noisy growth, true rain"),
            mo.stat(f"{_b:.1f} ± {1.96*_sb:.1f}", label=f"true growth, rain proxy ρ={rel_rain.value:.2f}"),
            mo.stat("35.0", label="true total effect"),
        ], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Push the growth noise to 60. The left line wobbles around 35 and only the band widens: *"unbiased, though less precise"*.
    2. Drag ρ toward 0.05. The right estimate slides toward the **naive** estimate from §2, because a useless proxy is like not adjusting at all.
    3. At ρ = 1 the two panels agree. Measurement error in a *confounder* is a causal problem. Measurement error in the *outcome* is mostly a precision problem.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Steps 3–5: plot distributions, plot the model, study residuals. Here the DAG is (almost) silent

    The video's verdict: DAGs say **nothing about functional form**: magnitude, direction, outliers, nonlinearity, interactions, or whether residuals behave.
    There is also a *danger*. A DAG can make you **myopic**. It says "control for rain", so you look only at an *added-variable plot* of fertilizer
    and conclude all is well. Meanwhile rain could **interact** with fertilizer, or bend the growth curve.

    > *"If we merely condition on our confounders without actually visualizing them … we run a major risk of really messing up our model, like big time."*
    > Best advice for this step: **forget the DAG** and visualise every component of the model.

    The world below is the same DAG (Rain → Fertilizer, Rain → Growth, Fertilizer → Growth), with
    $\text{Growth} = 50 + (35 + k\cdot\text{Rain})\,F + 10\,\text{Rain} + q\,\text{Rain}^2 + \varepsilon$. We fit the analyst's
    main-effects model $\text{Growth} \sim F + \text{Rain}$.
    """)
    return


@app.cell
def form_controls(mo):
    interaction_k = mo.ui.slider(-20, 20, step=1, value=12, label="k: Rain × Fertilizer interaction", show_value=True)
    curvature_q = mo.ui.slider(-10, 10, step=1, value=0, label="q: curvature in Rain (Rain²)", show_value=True)
    mo.hstack([interaction_k, curvature_q], justify="start", gap=2)
    return curvature_q, interaction_k


@app.cell
def form_charts(alt, curvature_q, interaction_k, mo, np, ols, pd):
    def simulate_rain_world(n=600, k=0.0, q=0.0, rain_mean=0.0, seed=3):
        _rng = np.random.default_rng(seed)
        rain = _rng.normal(rain_mean, 1, n)
        fert = (_rng.random(n) < 1 / (1 + np.exp(-1.5 * rain))).astype(float)
        growth = 50 + (35 + k * rain) * fert + 10 * rain + q * rain**2 + _rng.normal(0, 10, n)
        return pd.DataFrame(dict(Rain=rain, Fertilizer=fert, Growth=growth))

    form_df = simulate_rain_world(k=interaction_k.value, q=curvature_q.value)
    _c, _s = ols(form_df["Growth"].to_numpy(), dict(Fertilizer=form_df["Fertilizer"], Rain=form_df["Rain"]))
    _d = form_df.assign(
        fitted=_c["const"] + _c["Fertilizer"] * form_df["Fertilizer"] + _c["Rain"] * form_df["Rain"],
        group=np.where(form_df["Fertilizer"] == 1, "fertilized", "not fertilized"),
    )
    _d["residual"] = _d["Growth"] - _d["fitted"]
    _cr, _ = ols(_d["Growth"].to_numpy(), dict(Rain=_d["Rain"]))
    _cf, _ = ols(_d["Fertilizer"].to_numpy(), dict(Rain=_d["Rain"]))
    _d["Growth | Rain"] = _d["Growth"] - (_cr["const"] + _cr["Rain"] * _d["Rain"])
    _d["Fertilizer | Rain"] = _d["Fertilizer"] - (_cf["const"] + _cf["Rain"] * _d["Rain"])
    _col = alt.Color("group:N", scale=alt.Scale(domain=["fertilized", "not fertilized"], range=["#3b82f6", "#9aa0a6"]), title=None)
    _av = (alt.Chart(_d).mark_circle(size=14, opacity=.5).encode(x="Fertilizer | Rain:Q", y="Growth | Rain:Q", color=_col)
           + alt.Chart(_d).transform_regression("Fertilizer | Rain", "Growth | Rain").mark_line(color="black")
           .encode(x="Fertilizer | Rain:Q", y="Growth | Rain:Q")).properties(width=220, height=230, title="① added-variable plot (DAG-myopic)")
    _raw = (alt.Chart(_d).mark_circle(size=14, opacity=.45).encode(x="Rain:Q", y="Growth:Q", color=_col)
            + alt.Chart(_d).mark_line(size=3).encode(x="Rain:Q", y="fitted:Q", color=_col)).properties(
            width=220, height=230, title="② raw data + main-effects fit")
    _res = (alt.Chart(_d).mark_circle(size=14, opacity=.45).encode(x="Rain:Q", y="residual:Q", color=_col)
            + alt.Chart(_d).transform_loess("Rain", "residual", groupby=["group"], bandwidth=0.5).mark_line(size=3)
            .encode(x="Rain:Q", y="residual:Q", color=_col)
            + alt.Chart(pd.DataFrame(dict(z=[0]))).mark_rule(strokeDash=[4, 4]).encode(y="z:Q")).properties(
            width=220, height=230, title="③ residuals vs Rain (step 5)")
    mo.vstack([
        alt.hconcat(_av, _raw, _res),
        mo.md(f"Main-effects fit: Growth = {_c['const']:.1f} + **{_c['Fertilizer']:.1f}**·F + {_c['Rain']:.1f}·Rain. "
              f"Residual SD = {_d['residual'].std():.1f} (the noise SD is 10)."),
    ])
    return (simulate_rain_world,)


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. With **k = 12**, plot ① looks like a tidy treatment effect. Plot ② shows the two groups are *not* parallel, and the residual smooths in ③ fan out in opposite directions.
    2. Set **k = 0, q = 8**. Plot ② still looks roughly linear because the strong linear trend *masks* the curve, which is the video's step 5 example. Plot ③ subtracts the fit and shows the smile.
    3. Set both to 0. Now every plot is boring, and that is what a well-specified model looks like. The DAG was identical in all three cases.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Steps 6–7: interpret the *right* quantity, then decide against a *meaningful* threshold

    **Step 6 (original):** don't jump from the model to the p-values; interpret estimates and effect sizes first.
    **Step 6 (with DAGs):** *"don't jump from the model to the coefficient either."* The causal estimand, e.g. the **average treatment effect**

    $$\text{ATE} = E\big[\text{Growth}(F{=}1) - \text{Growth}(F{=}0)\big],$$

    need not equal $\beta_2$. With a Rain × Fertilizer interaction, $\beta_2$ from the main-effects model is a variance-weighted average that over-weights rain levels
    where fertilizer use is a coin flip. **G-computation** (fit the interaction model, predict everyone under $F{=}1$ and $F{=}0$, average the difference) targets the ATE.

    **Step 7:** decide only if you must, and against a **substantive** threshold (*30 % pays for the fertilizer*), not $p<.05$.
    If the interval straddles the threshold, the honest answer is **"ambiguous, gather more data"**.
    """)
    return


@app.cell
def estimand_controls(curvature_q, interaction_k, mo):
    rain_mean = mo.ui.slider(-1.5, 1.5, step=0.1, value=0.8, label="average rain in the study region (SD units)", show_value=True)
    study_n = mo.ui.slider(50, 2000, step=50, value=300, label="sample size n", show_value=True)
    threshold = mo.ui.slider(0, 60, step=1, value=30, label="decision threshold (% growth that pays for fertilizer)", show_value=True)
    mo.vstack([mo.md(f"Using **k = {interaction_k.value}**, **q = {curvature_q.value}** from §4."), mo.hstack([rain_mean, study_n, threshold], justify="start", gap=1)])
    return rain_mean, study_n, threshold


@app.cell
def estimand_chart(
    alt,
    curvature_q,
    interaction_k,
    mo,
    np,
    ols,
    pd,
    rain_mean,
    simulate_rain_world,
    study_n,
    threshold,
):
    def _fit_both(d):
        _y = d["Growth"].to_numpy()
        _c1, _ = ols(_y, dict(F=d["Fertilizer"], R=d["Rain"]))
        _c2, _ = ols(_y, dict(F=d["Fertilizer"], R=d["Rain"], R2=d["Rain"] ** 2, FR=d["Fertilizer"] * d["Rain"]))
        return _c1["F"], _c2["F"] + _c2["FR"] * d["Rain"].mean()

    estimand_df = simulate_rain_world(n=study_n.value, k=interaction_k.value, q=curvature_q.value, rain_mean=rain_mean.value, seed=11)
    _b2, _ate = _fit_both(estimand_df)
    _rng = np.random.default_rng(0)
    _boot = np.array([_fit_both(estimand_df.iloc[_rng.integers(0, len(estimand_df), len(estimand_df))]) for _ in range(300)])
    _lo, _hi = np.percentile(_boot, [2.5, 97.5], axis=0)
    true_ate = 35 + interaction_k.value * rain_mean.value
    estimand_table = pd.DataFrame(dict(
        quantity=["β₂ (main-effects coefficient)", "ATE via g-computation", "true ATE"],
        estimate=[_b2, _ate, true_ate], lo=[_lo[0], _lo[1], true_ate], hi=[_hi[0], _hi[1], true_ate],
        kind=["coefficient", "estimand", "truth"]))

    def _decide(lo, hi, t):
        if lo > t:
            return "✅ adopt: the whole interval clears the threshold"
        if hi < t:
            return "❌ don't adopt: the whole interval is below the threshold"
        return "🤷 ambiguous: the interval straddles the threshold, so gather more data"

    _ch = alt.Chart(estimand_table)
    _y = alt.Y("quantity:N", sort=None, title=None)
    _colr = alt.Color("kind:N", scale=alt.Scale(domain=["coefficient", "estimand", "truth"], range=["#e4572e", "#3b82f6", "#2a9d8f"]), legend=None)
    _chart = (_ch.mark_rule(size=3).encode(x=alt.X("lo:Q", title="effect of fertilizer on growth (%)"), x2="hi:Q", y=_y, color=_colr)
              + _ch.mark_point(size=160, filled=True).encode(x="estimate:Q", y=_y, color=_colr,
                    tooltip=["quantity", alt.Tooltip("estimate:Q", format=".1f"), alt.Tooltip("lo:Q", format=".1f"), alt.Tooltip("hi:Q", format=".1f")])
              + alt.Chart(pd.DataFrame(dict(t=[threshold.value]))).mark_rule(strokeDash=[6, 4], size=2, color="black").encode(x="t:Q")
              ).properties(width="container", height=170)
    mo.vstack([
        _chart,
        mo.hstack([mo.stat(f"{_b2:.1f}", label="β₂"), mo.stat(f"{_ate:.1f}", label="ATE (g-comp)"), mo.stat(f"{true_ate:.1f}", label="true ATE"),
                   mo.stat(f"{threshold.value} %", label="threshold")], justify="start"),
        mo.callout(mo.md(f"**Decision from the ATE** (95 % bootstrap CI {_lo[1]:.1f} to {_hi[1]:.1f}): {_decide(_lo[1], _hi[1], threshold.value)}<br>"
                         f"Decision from β₂ ({_lo[0]:.1f} to {_hi[0]:.1f}): {_decide(_lo[0], _hi[0], threshold.value)}"), kind="neutral"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. With k = 12 and rain mean 0.8, the red β₂ sits well below the true ATE. Same data, same DAG, different *question*. Set k = 0 and the two agree.
    2. Set the rain mean to **−0.4** (true ATE ≈ 30 %) and n to **100**. The interval straddles 30 %, so the answer is **ambiguous**. Raise n and it stays ambiguous, because the truth sits right at the threshold. The video argues we should get comfortable saying exactly that.
    3. Change the threshold: the DAG cannot tell you whether 30 % matters. *"That's the job of scientific expertise."*
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Step 8: replicate, and replicate the DAG too

    A single study is provisional. The video adds a causal twist: every estimate so far **assumed the DAG is correct**, and that assumption
    *"is not something the DAG itself can prove, nor … the statistical analysis"*. Replications test the effect **and** the DAG.

    Below, 30 sites each run the study. At a fraction of sites the DAG is **wrong**: an unmeasured confounder (say, a local irrigation habit) opens a
    back door nobody adjusted for. Press **▶ play** and watch the running meta-analytic estimate (inverse-variance weighted) settle, or fail to.
    """)
    return


@app.cell
def replication_class(anywidget, traitlets):
    class ReplicationSim(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 680, H = 470, L = 70, R = 20, T = 20, NS = "http://www.w3.org/2000/svg";
      const ctl = document.createElement("div"); ctl.style.cssText = "display:flex;gap:6px;align-items:center;font:13px system-ui,sans-serif;margin-bottom:4px";
      el.appendChild(ctl);
      const svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", font: "12px system-ui,sans-serif" }); el.appendChild(svg);
      const mk = (t, a) => { const e = document.createElementNS(NS, t); for (const k in a) e.setAttribute(k, a[k]); svg.appendChild(e); return e; };
      const btn = (label, fn) => { const b = document.createElement("button"); b.textContent = label; b.onclick = fn;
        b.style.cssText = "padding:3px 10px;border-radius:6px;border:1px solid #999;background:transparent;color:inherit;cursor:pointer"; ctl.appendChild(b); return b; };
      let timer = null;
      const sync = () => model.save_changes();
      const step = () => { const n = model.get("studies").length, s = model.get("shown");
        if (s >= n) { pause(); return; } model.set("shown", s + 1); if ((s + 1) % 5 === 0 || s + 1 === n) sync(); draw(); };
      const pause = () => { if (timer) { clearInterval(timer); timer = null; sync(); } play.textContent = "▶ play"; };
      const play = btn("▶ play", () => { if (timer) pause(); else { timer = setInterval(step, 350); play.textContent = "⏸ pause"; } });
      btn("step", () => { step(); sync(); });
      btn("↺ reset", () => { pause(); model.set("shown", 0); sync(); draw(); });
      const lab = document.createElement("label"); const cb = document.createElement("input"); cb.type = "checkbox";
      cb.checked = model.get("reveal"); cb.onchange = () => { model.set("reveal", cb.checked); sync(); draw(); };
      lab.append(cb, " reveal sites where the DAG was wrong"); ctl.appendChild(lab);
      function draw() {
        svg.innerHTML = "";
        const S = model.get("studies"), k = model.get("shown"), thr = model.get("threshold"), truth = model.get("truth");
        const all = S.flatMap(s => [s.est - 2 * s.se, s.est + 2 * s.se]).concat([thr, truth]);
        const lo = Math.min(...all) - 2, hi = Math.max(...all) + 2;
        const sx = v => L + (v - lo) / (hi - lo) * (W - L - R), rowH = (H - T - 70) / S.length;
        for (let v = Math.ceil(lo / 10) * 10; v <= hi; v += 10) { mk("line", { x1: sx(v), x2: sx(v), y1: T, y2: H - 40, stroke: "rgba(128,128,128,.2)" });
          mk("text", { x: sx(v), y: H - 25, "text-anchor": "middle", fill: "currentColor" }).textContent = v; }
        mk("text", { x: (L + W - R) / 2, y: H - 8, "text-anchor": "middle", fill: "currentColor" }).textContent = "estimated effect of fertilizer (%)";
        mk("line", { x1: sx(thr), x2: sx(thr), y1: T, y2: H - 40, stroke: "currentColor", "stroke-dasharray": "6 4", "stroke-width": 1.5 });
        mk("line", { x1: sx(truth), x2: sx(truth), y1: T, y2: H - 40, stroke: "#2a9d8f", "stroke-dasharray": "3 3", "stroke-width": 2 });
        let sw = 0, swx = 0;
        S.slice(0, k).forEach((s, i) => { const y = T + i * rowH + rowH / 2, w = 1 / (s.se * s.se); sw += w; swx += w * s.est;
          const col = model.get("reveal") && s.wrong ? "#e4572e" : "#3b82f6";
          mk("line", { x1: sx(s.est - 1.96 * s.se), x2: sx(s.est + 1.96 * s.se), y1: y, y2: y, stroke: col, "stroke-width": 1.5 });
          mk("rect", { x: sx(s.est) - 3.5, y: y - 3.5, width: 7, height: 7, fill: col });
          mk("text", { x: L - 6, y: y + 4, "text-anchor": "end", fill: "currentColor", opacity: .7 }).textContent = "site " + (i + 1); });
        if (k > 0) { const m = swx / sw, se = Math.sqrt(1 / sw), y = H - 52;
          mk("polygon", { points: `${sx(m - 1.96 * se)},${y} ${sx(m)},${y - 8} ${sx(m + 1.96 * se)},${y} ${sx(m)},${y + 8}`, fill: "#3b82f6", opacity: .85 });
          mk("text", { x: L - 6, y: y + 4, "text-anchor": "end", fill: "currentColor", "font-weight": 700 }).textContent = `meta (${k})`; }
      }
      for (const t of ["studies", "shown", "reveal", "threshold"]) model.on("change:" + t, draw);
      draw();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        studies = traitlets.List(traitlets.Dict()).tag(sync=True)
        shown = traitlets.Int(0).tag(sync=True)
        reveal = traitlets.Bool(False).tag(sync=True)
        threshold = traitlets.Float(30.0).tag(sync=True)
        truth = traitlets.Float(35.0).tag(sync=True)

    return (ReplicationSim,)


@app.cell
def replication_controls(mo):
    wrong_share = mo.ui.slider(0, 1, step=0.1, value=0.3, label="share of sites where the DAG is wrong", show_value=True)
    hidden_bias = mo.ui.slider(-30, 30, step=1, value=-15, label="bias from the hidden back door (%)", show_value=True)
    site_n = mo.ui.slider(30, 1000, step=10, value=150, label="n per site", show_value=True)
    reroll_sites = mo.ui.button(label="🎲 new sites", value=0, on_click=lambda v: v + 1)
    mo.hstack([mo.vstack([wrong_share, hidden_bias, site_n]), reroll_sites], justify="start", gap=2)
    return hidden_bias, reroll_sites, site_n, wrong_share


@app.cell
def replication(
    ReplicationSim,
    hidden_bias,
    mo,
    np,
    reroll_sites,
    site_n,
    threshold,
    wrong_share,
):
    def make_studies(n_sites=30, share_wrong=0.3, bias=-15.0, n=150, seed=0):
        _rng = np.random.default_rng(seed)
        _se = 2 * 25 / np.sqrt(n)                       # SE of a mean difference, outcome SD ≈ 25, two equal arms
        _wrong = _rng.random(n_sites) < share_wrong
        _est = 35 + _wrong * bias * _rng.uniform(0.6, 1.4, n_sites) + _rng.normal(0, _se, n_sites)
        return [dict(est=float(e), se=float(_se), wrong=bool(w)) for e, w in zip(_est, _wrong)]

    replication = mo.ui.anywidget(ReplicationSim(
        studies=make_studies(share_wrong=wrong_share.value, bias=hidden_bias.value, n=site_n.value, seed=reroll_sites.value),
        threshold=float(threshold.value), truth=35.0))
    replication
    return (replication,)


@app.cell
def replication_readout(mo, np, pd, replication):
    _S = pd.DataFrame(replication.value["studies"][: replication.value["shown"]])
    if len(_S) < 2:
        _out = mo.md("_Press **▶ play** or **step** to add at least two sites._")
    else:
        _w = 1 / _S.se**2
        _m = float((_w * _S.est).sum() / _w.sum())
        _se_m = float(np.sqrt(1 / _w.sum()))
        _Q = float((_w * (_S.est - _m) ** 2).sum())
        _I2 = max(0.0, (_Q - (len(_S) - 1)) / _Q) if _Q > 0 else 0.0
        from scipy import stats as _stats
        _p = float(_stats.chi2.sf(_Q, len(_S) - 1))
        _out = mo.vstack([
            mo.hstack([
                mo.stat(f"{_m:.1f} ± {1.96*_se_m:.1f}", label=f"meta estimate after {len(_S)} sites"),
                mo.stat(f"{_I2:.0%}", label="I² (heterogeneity)"),
                mo.stat(f"{_p:.3f}", label="Cochran's Q p-value"),
                mo.stat(f"{_S.wrong.mean():.0%}", label="sites with wrong DAG (hidden)"),
            ], justify="start"),
            mo.callout(mo.md(
                "High heterogeneity is a **clue that the DAG differs across sites**. The next study should probe *that* part of the DAG, not just re-run the same model."
                if _I2 > 0.4 else
                "Sites agree. That is consistent with (not proof of) the assumed DAG holding everywhere."), kind="warn" if _I2 > 0.4 else "success"),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Set the wrong-DAG share to **0**: the diamond homes in on the green truth, and I² stays low.
    2. Set it to **0.3** and bias to **−15**: the meta estimate is pulled toward the threshold, and I² jumps. Tick *reveal* to see which sites misled you.
    3. Set the share to **1.0**: every site is biased the *same* way. I² is low again and the meta-analysis is confidently wrong. **Replication cannot rescue a DAG error that every study shares**, so probe the DAG directly.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🧪 Sandbox: draw any DAG, and the notebook simulates it

    Build your own causal story (or load a preset). The notebook treats your drawing as a **linear structural model**: every arrow gets a fixed random
    coefficient, every node gets standard-normal noise. It then simulates 5,000 rows and fits $Y \sim X + Z$ for:
    the naive model, **your** adjusted set, every **minimal back-door set**, and "adjust for everything" (the kitchen-sink approach the video warns about).
    The **true total effect** is the sum over directed paths of the product of the arrow coefficients.
    """)
    return


@app.cell
def sandbox(DAG_PRESETS, DagEditor, mo):
    sandbox = mo.ui.anywidget(DagEditor(**DAG_PRESETS["M-bias"], presets=DAG_PRESETS))
    sandbox
    return (sandbox,)


@app.cell
def sandbox_readout(
    adjustment_sets,
    alt,
    classify_set,
    mo,
    np,
    ols,
    pd,
    sandbox,
):
    def simulate_linear_dag(nodes, edges, n=5000, seed=42):
        """Linear-Gaussian SEM on an arbitrary DAG; coefficients are deterministic per edge name."""
        ids = [d["id"] for d in nodes]
        E = [(a, b) for a, b in edges if a in ids and b in ids]
        coef = {}
        for a, b in E:
            import zlib
            _r = np.random.default_rng(zlib.crc32((a + ">" + b).encode()))
            coef[(a, b)] = float(_r.choice([-1, 1]) * _r.uniform(0.5, 1.2))
        order, done = [], set()
        while len(order) < len(ids):
            progressed = False
            for v in ids:
                if v not in done and all(a in done for a, b in E if b == v):
                    order.append(v); done.add(v); progressed = True
            if not progressed:
                raise ValueError("graph has a cycle")
        _rng = np.random.default_rng(seed)
        data = {}
        for v in order:
            data[v] = sum(coef[(a, b)] * data[a] for a, b in E if b == v) + _rng.normal(size=n)
        return pd.DataFrame(data), coef

    def true_total_effect(edges, coef, X, Y):
        _ch = {}
        for a, b in coef:
            _ch.setdefault(a, []).append(b)
        def walk(u, prod):
            if u == Y:
                return prod
            return sum(walk(v, prod * coef[(u, v)]) for v in _ch.get(u, []))
        return walk(X, 1.0)

    _v = sandbox.value
    _X, _Y = _v["exposure"], _v["outcome"]
    _ids = [d["id"] for d in _v["nodes"]]
    if _X not in _ids or _Y not in _ids or _X == _Y:
        _out = mo.callout(mo.md("Choose an exposure **X** and an outcome **Y** in the sandbox."), kind="warn")
    else:
        _df, _coef = simulate_linear_dag(_v["nodes"], _v["edges"])
        _truth = true_total_effect(_v["edges"], _coef, _X, _Y)
        _cands = {"naive ∅": []}
        _cands[f"your set {{{', '.join(_v['adjusted'])}}}"] = [z for z in _v["adjusted"] if z not in (_X, _Y)]
        for _s in adjustment_sets(_v["nodes"], _v["edges"], _X, _Y)[:5]:
            _cands[f"minimal {{{', '.join(_s)}}}"] = _s
        _cands["kitchen sink (all others)"] = [i for i in _ids if i not in (_X, _Y)]
        _rows = []
        for _lab, _Zs in _cands.items():
            _c, _se = ols(_df[_Y].to_numpy(), {c: _df[c] for c in [_X] + list(_Zs)})
            _cls = classify_set(_v["nodes"], _v["edges"], _X, _Y, _Zs)
            _rows.append(dict(model=_lab, estimate=_c[_X], lo=_c[_X] - 1.96 * _se[_X], hi=_c[_X] + 1.96 * _se[_X],
                              identifies={"total": "total effect", "direct": "direct effect"}.get(_cls, "biased / partial")))
        _r = pd.DataFrame(_rows)
        _sc = alt.Scale(domain=["total effect", "direct effect", "biased / partial"], range=["#2a9d8f", "#7b61ff", "#e4572e"])
        _b = alt.Chart(_r).encode(y=alt.Y("model:N", sort=None, title=None))
        _ch = (_b.mark_rule(size=3).encode(x=alt.X("lo:Q", title=f"estimated effect of {_X} on {_Y}"), x2="hi:Q", color=alt.Color("identifies:N", scale=_sc))
               + _b.mark_point(size=150, filled=True).encode(x="estimate:Q", color=alt.Color("identifies:N", scale=_sc),
                   tooltip=["model", alt.Tooltip("estimate:Q", format=".3f"), "identifies"])
               + alt.Chart(pd.DataFrame(dict(t=[_truth]))).mark_rule(color="#2a9d8f", strokeDash=[6, 4], size=2).encode(x="t:Q")
               ).properties(width="container", height=40 * len(_r) + 40)
        _out = mo.vstack([
            mo.md(f"**True total effect** of {_X} on {_Y} = **{_truth:.3f}** (green dashed line). "
                  f"Arrow coefficients: " + ", ".join(f"{a}→{b}: {c:+.2f}" for (a, b), c in _coef.items())),
            _ch,
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Sandbox challenges**

    1. **M-bias:** ▢ Adjust for *M* only. *Your set* turns red (biased) while ∅ is fine. The kitchen sink is fine again only because it also includes U1 and U2, which are usually *unobserved*. Then add an arrow *U1 → Growth*: what is the minimal set now?
    2. **Build the "butterfly":** make *M* a confounder (M → X, M → Y) *and* a collider (U1 → M ← U2, U1 → X, U2 → Y). The tool should find that you need {M, U1} or {M, U2}.
    3. **Kitchen sink vs. collider:** load *Collider* and compare ∅, the minimal set and the kitchen sink. Only adjusting for Nitrogen *without* Soil hurts.
    4. **Mediator trap:** in the *Mediator* preset, adjust for *Root* and confirm the estimate falls onto the *direct* effect.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 📝 Quiz: where does the DAG help?

    Answer all questions, then flip the reveal switch. All questions use the video's fertilizer DAG unless stated otherwise.
    """)
    return


@app.cell
def quiz(mo):
    quiz_items = {
        "Q1. In the fertilizer DAG, which variable must you adjust for to get the total effect?": (["Rain", "Root", "Nitrogen", "Soil"], "Rain",
            "Rain is the only confounder; {Rain} is the minimal back-door set."),
        "Q2. Soil nitrogen (Fertilizer → Nitrogen ← Soil) is a…": (["confounder", "mediator", "collider", "instrument"], "collider",
            "Two arrows point into it, so conditioning on it opens Fertilizer → Nitrogen ← Soil → Growth."),
        "Q3. Adjusting for Rain AND Root growth estimates the…": (["total effect", "direct effect", "a biased mess", "indirect effect"], "direct effect",
            "Blocking the mediator leaves only Fertilizer → Growth: the direct effect (15 % in §2)."),
        "Q4. Your growth scale also weighs the soil water. Mostly a problem of…": (["bias", "precision", "identification", "nothing"], "precision",
            "Noise in the outcome opens no back door: unbiased but less precise (§3)."),
        "Q5. Rainfall is only known as a regional historical average. Adjusting for it…": (["fully closes the back door", "leaves residual confounding", "creates collider bias", "is unnecessary"],
            "leaves residual confounding", "A noisy proxy of a confounder only partly blocks the path (§3)."),
        "Q6. Which step does the video say DAGs help with LEAST?": (["1 hypothesis", "2 measurement", "5 residuals", "8 replication"], "5 residuals",
            "Residuals evaluate the model as a whole; they have nothing to do with identification."),
        "Q7. Who decides that 30 % growth is the threshold worth acting on?": (["the DAG", "p < .05", "scientific / substantive expertise", "the residuals"],
            "scientific / substantive expertise", "DAGs tell you which quantity to estimate, not how large is large enough."),
        "Q8. With a Rain × Fertilizer interaction, the main-effects β₂ equals the ATE.": (["always", "not in general", "only if n is large"], "not in general",
            "β₂ is a variance-weighted average; g-computation targets the ATE (§5)."),
    }
    quiz = mo.ui.dictionary({_q: mo.ui.dropdown(_opts, label="") for _q, (_opts, _a, _e) in quiz_items.items()})
    reveal_quiz = mo.ui.switch(label="Reveal answers")
    mo.vstack([mo.vstack([mo.hstack([mo.md(_q), quiz[_q]], justify="space-between") for _q in quiz_items]), reveal_quiz])
    return quiz, quiz_items, reveal_quiz


@app.cell
def quiz_score(mo, quiz, quiz_items, reveal_quiz):
    _answered = sum(v is not None for v in quiz.value.values())
    _score = sum(quiz.value[_q] == _a for _q, (_o, _a, _e) in quiz_items.items())
    if not reveal_quiz.value:
        _out = mo.md(f"_Answered {_answered}/{len(quiz_items)}. Flip **Reveal answers** when ready._")
    else:
        _rows = "\n".join(
            f"| {'✅' if quiz.value[_q] == _a else '❌'} | {_q.split('.')[0]} | {quiz.value[_q] or '—'} | **{_a}** | {_e} |"
            for _q, (_o, _a, _e) in quiz_items.items())
        _out = mo.vstack([
            mo.stat(f"{_score} / {len(quiz_items)}", label="score"),
            mo.md("| | Q | your answer | correct | why |\n|---|---|---|---|---|\n" + _rows),
        ])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap: the eight steps, revised with DAGs

    | step | original purpose (8-steps paper) | what the DAG adds | where to play |
    |---|---|---|---|
    | 1 Hypothesis | name the parameter of interest (β₂), predict its size (≥ 30 %) | **construct the DAG**: exposure, outcome, full causal structure | §1 editor |
    | 2 Measurement | are measures reliable and valid? | **identify the adjustment set**; see how *bad measurement* breaks identification (noisy confounder ≠ noisy outcome) | §1, §2, §3 |
    | 3 Univariate plots | quality control; anticipate modelling problems | *context*: you know each variable's role (confounder, mediator, collider) | §2 |
    | 4 Plot the model | check functional form, outliers, interactions | **little to say**, and a risk of myopia: visualise confounders and their interactions too | §4 |
    | 5 Residuals | check assumptions the plots hide | **nothing**: residuals are not about identification | §4 ③ |
    | 6 Estimates | interpret effect sizes before p-values | choose the **causal estimand** (ATE, direct, indirect), which may ≠ the coefficient | §5 |
    | 7 Decision (optional) | decide against substantive thresholds, embrace ambiguity | which quantity to decide on, **not** how big is big enough | §5 |
    | 8 Replicate | single studies are provisional | the **DAG is provisional too**; replicate it and probe its weak arrows | §6 |

    > **Central message:** good causal reasoning doesn't guarantee a good statistical model, and a good statistical model doesn't guarantee good causal reasoning. **We need both.**
    """)
    return


if __name__ == "__main__":
    app.run()

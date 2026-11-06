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
    import heapq
    import math
    import random

    return alt, anywidget, heapq, math, mo, np, pd, random, traitlets


@app.cell(hide_code=True)
def title(mo):
    mo.vstack([
        mo.md(r"""
    # Google Maps is unreasonably fast: shortest paths, from Dijkstra to contraction hierarchies
    *An interactive companion to Veritasium's **"Google Maps is unreasonably fast. Let me explain"**.*

    > New York to San Francisco: more than 64 million intersections and roughly $10^{220}$ possible routes.
    > Your phone finds the best one in about four seconds. **How?**
    """),
        mo.Html('<iframe width="640" height="360" src="https://www.youtube.com/embed/kS-CGkiPetQ" title="Google Maps is unreasonably fast" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'),
    ])
    return


@app.cell(hide_code=True)
def colors(mo):
    mo.md(r"""
    **How to read the pictures.** Every widget in this notebook uses the same colors:
    <span style="color:#2ca02c">**green ring = source**</span>,
    <span style="color:#d62728">**red ring = target**</span>,
    <span style="color:#f58518">**orange = frontier**</span> (we know *a* cost, but it might still drop),
    <span style="color:#4c78a8">**blue = explored from the source**</span> (cost is final),
    <span style="color:#b279a2">**purple = explored from the target**</span> (backward search),
    <span style="color:#e8b100">**gold = the shortest path**</span>, and
    <span style="color:#8c564b">**dashed brown = shortcut**</span> (an edge that isn't a real road).

    The video's order is the notebook's order: **brute force → BFS → Dijkstra → A\* → travel time → bidirectional → hierarchies → contraction hierarchies**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0 · Why you can't just try every route

    The simplest "algorithm" is to list every route and keep the shortest. Picture a city of $n\times n$ blocks and only allow routes that
    always head **east or north** (never backtrack). Each route is a sequence of $n$ "east" moves and $n$ "north" moves, so there are

    $$\#\text{routes} = \binom{2n}{n} \approx \frac{4^n}{\sqrt{\pi n}}.$$

    That number explodes. The video's estimate for the real North American network, with backtracking-free routes from
    New York to San Francisco, is about $10^{220}$. At a billion routes per second, that's more than $10^{200}$ years.
    """)
    return


@app.cell
def brute_n(mo):
    brute_n = mo.ui.slider(1, 400, value=10, step=1, label="City size n (blocks per side)", full_width=True)
    brute_n
    return (brute_n,)


@app.cell
def brute_readout(alt, brute_n, math, mo, np, pd):
    _n = brute_n.value
    _log10_routes = (math.lgamma(2 * _n + 1) - 2 * math.lgamma(_n + 1)) / math.log(10)
    _log10_years = _log10_routes - 9 - math.log10(3.156e7)
    _ns = np.arange(1, 401)
    _df = pd.DataFrame({
        "n": _ns,
        "log10 routes": [(math.lgamma(2 * k + 1) - 2 * math.lgamma(k + 1)) / math.log(10) for k in _ns],
    })
    _chart = (
        alt.Chart(_df).mark_line(color="#4c78a8").encode(
            x=alt.X("n:Q", title="city size n"),
            y=alt.Y("log10 routes:Q", title="log10(number of routes)"))
        + alt.Chart(pd.DataFrame({"y": [220]})).mark_rule(color="#d62728", strokeDash=[5, 4]).encode(y="y:Q")
        + alt.Chart(pd.DataFrame({"n": [_n], "log10 routes": [_log10_routes]})).mark_point(color="#f58518", size=120, filled=True).encode(x="n:Q", y="log10 routes:Q")
    ).properties(width="container", height=220, title="Dashed red line: the video's 10^220 estimate for NYC → SF")
    mo.vstack([
        mo.hstack([
            mo.stat(f"10^{_log10_routes:.1f}", label="routes (east/north only)"),
            mo.stat(f"{_n * _n + 2 * _n + 1:,}", label="intersections"),
            mo.stat(("< 1 second" if _log10_years < -7.5 else f"10^{_log10_years:.1f} years"), label="time at 10^9 routes/s"),
        ], justify="start"),
        _chart,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this.** Slide to n = 10: that's about 185,000 routes, which is instant. Then try n = 50, which needs about $10^{29}$ routes
    and about $10^{12}$ years. At what n do you cross the red line? A whole continent's intersections only fit in a tiny city of a few
    hundred blocks per side. **Enumerating routes is hopeless. We need an algorithm that never looks at most of them.**
    """), kind="info")
    return


@app.cell
def graph_algos(heapq, math):
    def search_states(nodes, edges, s, t, unit=False):
        """Dijkstra (or BFS when unit=True) on a small labelled graph, recording one state per explored node.
        nodes: {id: (label, x, y)} with x, y in [0, 1]; edges: [(u, v, w)] undirected."""
        adj = {k: [] for k in nodes}
        for u, v, w in edges:
            ww = 1 if unit else w
            adj[u].append((v, ww))
            adj[v].append((u, ww))
        cost = {k: math.inf for k in nodes}
        cost[s] = 0
        parent, explored, states = {}, set(), []

        def snap(caption, current=None, relaxed=(), path=()):
            _pe = set(zip(path, path[1:]))
            ns = []
            for k, (lab, x, y) in nodes.items():
                if k in path:
                    st = "path"
                elif k == current:
                    st = "current"
                elif k in explored:
                    st = "explored"
                elif cost[k] < math.inf:
                    st = "frontier"
                else:
                    st = "unvisited"
                ns.append(dict(id=k, label=lab, x=x, y=y, status=st,
                               inner="∞" if cost[k] == math.inf else f"{cost[k]:g}",
                               role="source" if k == s else ("target" if k == t else "")))
            es = []
            for u, v, w in edges:
                if (u, v) in _pe or (v, u) in _pe:
                    st = "path"
                elif (u, v) in relaxed or (v, u) in relaxed:
                    st = "relaxed"
                elif parent.get(v) == u or parent.get(u) == v:
                    st = "tree"
                else:
                    st = "normal"
                es.append(dict(u=u, v=v, w=(1 if unit else w), status=st))
            states.append(dict(nodes=ns, edges=es, caption=caption))

        name = lambda k: nodes[k][0]
        snap(f"Start: cost({name(s)}) = 0, every other node has cost ∞.")
        while True:
            cand = [k for k in nodes if k not in explored and cost[k] < math.inf]
            if not cand:
                break
            u = min(cand, key=lambda k: (cost[k], name(k)))
            if u == t:
                explored.add(u)
                path = [t]
                while path[-1] != s:
                    path.append(parent[path[-1]])
                path = path[::-1]
                snap(f"<b>{name(t)}</b> now has the lowest cost of all unexplored nodes ({cost[t]:g}). Every cheaper path has already been explored, "
                     f"so this is the shortest path: {' → '.join(name(k) for k in path)} (cost <b>{cost[t]:g}</b>).", path=path)
                break
            notes, relaxed = [], []
            for v, w in adj[u]:
                if v in explored:
                    continue
                nd = cost[u] + w
                if nd < cost[v]:
                    old = "∞" if cost[v] == math.inf else f"{cost[v]:g}"
                    notes.append(f"{name(v)}: {old} → <b>{nd:g}</b> ✔")
                    cost[v] = nd
                    parent[v] = u
                    relaxed.append((u, v))
                else:
                    notes.append(f"{name(v)}: {cost[u]:g}+{w:g}={nd:g} ≥ {cost[v]:g}, kept ✘")
            explored.add(u)
            extra = ""
            if cost[t] < math.inf and t not in explored:
                nxt = min((cost[k] for k in nodes if k not in explored and cost[k] < math.inf), default=math.inf)
                if nxt < cost[t]:
                    extra = f"<br><i>{name(t)} has been reached with cost {cost[t]:g}, but an unexplored node only costs {nxt:g}. A shorter path might be hiding there, so keep going.</i>"
            snap(f"Explore <b>{name(u)}</b> (cost {cost[u]:g}). " + ("; ".join(notes) if notes else "No unexplored neighbours.") + extra,
                 current=u, relaxed=relaxed)
        return states


    def dijkstra_dist(adj, s):
        """Plain Dijkstra on adjacency {u: [(v, w)]}; returns dist dict."""
        dist, pq, done = {s: 0}, [(0, s)], set()
        while pq:
            d, u = heapq.heappop(pq)
            if u in done:
                continue
            done.add(u)
            for v, w in adj[u]:
                if d + w < dist.get(v, math.inf):
                    dist[v] = d + w
                    heapq.heappush(pq, (d + w, v))
        return dist

    return dijkstra_dist, search_states


@app.cell(hide_code=True)
def nl_graph(math):
    _cities = {
        "Rotterdam": (4.48, 51.92), "Den Haag": (4.30, 52.07), "Leiden": (4.49, 52.16), "Haarlem": (4.64, 52.38),
        "Amsterdam": (4.90, 52.37), "Alkmaar": (4.75, 52.63), "Utrecht": (5.12, 52.09), "Gouda": (4.71, 52.01),
        "Dordrecht": (4.69, 51.81), "Breda": (4.78, 51.59), "Den Bosch": (5.30, 51.69), "Eindhoven": (5.48, 51.44),
        "Nijmegen": (5.85, 51.84), "Arnhem": (5.90, 51.98), "Amersfoort": (5.39, 52.16), "Apeldoorn": (5.97, 52.21),
        "Zwolle": (6.09, 52.52), "Lelystad": (5.47, 52.52), "Leeuwarden": (5.80, 53.20), "Groningen": (6.57, 53.22),
        "Assen": (6.56, 52.99), "Enschede": (6.89, 52.22),
    }
    _roads = [
        ("Rotterdam", "Den Haag"), ("Rotterdam", "Gouda"), ("Rotterdam", "Dordrecht"), ("Den Haag", "Leiden"),
        ("Leiden", "Haarlem"), ("Leiden", "Amsterdam"), ("Leiden", "Gouda"), ("Haarlem", "Amsterdam"), ("Haarlem", "Alkmaar"),
        ("Amsterdam", "Alkmaar"), ("Amsterdam", "Utrecht"), ("Amsterdam", "Lelystad"), ("Gouda", "Utrecht"),
        ("Dordrecht", "Breda"), ("Dordrecht", "Den Bosch"), ("Breda", "Eindhoven"), ("Breda", "Den Bosch"),
        ("Den Bosch", "Eindhoven"), ("Den Bosch", "Utrecht"), ("Den Bosch", "Nijmegen"), ("Nijmegen", "Arnhem"),
        ("Utrecht", "Amersfoort"), ("Utrecht", "Arnhem"), ("Amersfoort", "Apeldoorn"), ("Amersfoort", "Lelystad"),
        ("Arnhem", "Apeldoorn"), ("Apeldoorn", "Zwolle"), ("Apeldoorn", "Enschede"), ("Lelystad", "Zwolle"),
        ("Zwolle", "Assen"), ("Zwolle", "Leeuwarden"), ("Zwolle", "Enschede"), ("Assen", "Groningen"),
        ("Leeuwarden", "Groningen"), ("Alkmaar", "Leeuwarden"), ("Eindhoven", "Nijmegen"),
    ]
    _xs = [lon * 68 for lon, lat in _cities.values()]
    _ys = [lat * 111 for lon, lat in _cities.values()]
    _x0, _x1, _y0, _y1 = min(_xs), max(_xs), min(_ys), max(_ys)
    nl_nodes = {c: (c, (lon * 68 - _x0) / (_x1 - _x0), 1 - (lat * 111 - _y0) / (_y1 - _y0)) for c, (lon, lat) in _cities.items()}
    # edge weight = straight-line km / 10, rounded, at least 1 (like the video's small integer weights)
    nl_edges = [
        (a, b, max(1, round(math.hypot((_cities[a][0] - _cities[b][0]) * 68, (_cities[a][1] - _cities[b][1]) * 111) / 10)))
        for a, b in _roads
    ]
    return nl_edges, nl_nodes


@app.cell
def GraphStepper(anywidget, traitlets):
    class GraphStepper(anywidget.AnyWidget):
        """Renders precomputed graph states [{nodes, edges, caption}] with prev / next / play controls."""
        _esm = r"""
    function render({ model, el }) {
      const NS = "http://www.w3.org/2000/svg", W = model.get("width"), H = model.get("height"), PAD = 34;
      const ctl = document.createElement("div");
      ctl.style.cssText = "display:flex;gap:6px;align-items:center;font:13px sans-serif;margin-bottom:4px;flex-wrap:wrap";
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none" });
      const cap = document.createElement("div");
      cap.style.cssText = "font:14px sans-serif;min-height:3em;margin-top:4px;line-height:1.4";
      el.append(ctl, svg, cap);
      let timer = null;
      const n = () => model.get("states").length;
      const setI = v => { model.set("i", Math.max(0, Math.min(n() - 1, v))); model.save_changes(); };
      const B = (t, f) => { const b = document.createElement("button"); b.textContent = t; b.onclick = f;
        b.style.cssText = "padding:2px 10px;border-radius:6px;border:1px solid #999;background:transparent;color:inherit;cursor:pointer";
        ctl.appendChild(b); return b; };
      const stop = () => { if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; } };
      B("⏮", () => { stop(); setI(0); });
      B("◀", () => { stop(); setI(model.get("i") - 1); });
      const play = B("▶ play", () => {
        if (timer) { stop(); return; }
        if (model.get("i") >= n() - 1) setI(0);
        play.textContent = "⏸ pause";
        timer = setInterval(() => { if (model.get("i") >= n() - 1) { stop(); return; } setI(model.get("i") + 1); }, model.get("interval"));
      });
      B("▶", () => { stop(); setI(model.get("i") + 1); });
      B("⏭", () => { stop(); setI(n() - 1); });
      const lab = document.createElement("span"); lab.style.marginLeft = "8px"; ctl.appendChild(lab);
      const mk = (tag, a, p = svg) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
      const sx = x => PAD + x * (W - 2 * PAD), sy = y => PAD + y * (H - 2 * PAD);
      const NODE = { unvisited: "#c7c7c7", frontier: "#f58518", explored: "#4c78a8", explored_b: "#b279a2", both: "#54a24b", current: "#f58518", path: "#e8b100", focus: "#e45756", contracted: "#8a8a8a" };
      const EDGE = { normal: "#999", tree: "#4c78a8", tree_b: "#b279a2", relaxed: "#f58518", path: "#e8b100", shortcut: "#8c564b", focus: "#e45756", faded: "#999" };
      function draw() {
        const st = model.get("states"); if (!st.length) { svg.innerHTML = ""; return; }
        const i = Math.min(model.get("i"), st.length - 1), s = st[i];
        svg.innerHTML = "";
        const pos = {}; s.nodes.forEach(nd => pos[nd.id] = nd);
        s.edges.forEach(e => {
          const a = pos[e.u], b = pos[e.v];
          const st2 = e.status || "normal";
          const ln = mk("line", { x1: sx(a.x), y1: sy(a.y), x2: sx(b.x), y2: sy(b.y), stroke: EDGE[st2] || "#999",
            "stroke-width": st2 === "path" ? 6 : (st2 === "normal" || st2 === "faded" ? 1.5 : 3.5),
            "stroke-opacity": st2 === "faded" ? 0.25 : (st2 === "normal" ? 0.6 : 0.95) });
          if (e.dashed) ln.setAttribute("stroke-dasharray", "7 5");
          if (e.w !== null && e.w !== undefined) {
            const mx = (sx(a.x) + sx(b.x)) / 2, my = (sy(a.y) + sy(b.y)) / 2, txt = String(e.w), bw = 7 * txt.length + 8;
            mk("rect", { x: mx - bw / 2, y: my - 8, width: bw, height: 15, rx: 4, fill: "rgba(255,255,255,0.88)", stroke: e.dashed ? "#8c564b" : "none" });
            const t = mk("text", { x: mx, y: my + 4, "text-anchor": "middle", "font-size": 11, fill: "#333", "font-family": "sans-serif" }); t.textContent = txt;
          }
        });
        s.nodes.forEach(nd => {
          const fill = NODE[nd.status] || "#c7c7c7";
          const ring = nd.role === "source" ? "#2ca02c" : nd.role === "target" ? "#d62728" : (nd.status === "current" ? "#222" : "rgba(255,255,255,0.9)");
          mk("circle", { cx: sx(nd.x), cy: sy(nd.y), r: 15, fill, stroke: ring, "stroke-width": nd.role || nd.status === "current" ? 4.5 : 1.5 });
          const t = mk("text", { x: sx(nd.x), y: sy(nd.y) + 4, "text-anchor": "middle", "font-size": 11, "font-weight": "bold",
            fill: (nd.status === "unvisited" || nd.status === "path") ? "#222" : "#fff", "font-family": "sans-serif" });
          t.textContent = nd.inner ?? "";
          if (nd.label) { const l = mk("text", { x: sx(nd.x), y: sy(nd.y) + 29, "text-anchor": "middle", "font-size": 11, fill: "currentColor", "font-family": "sans-serif" }); l.textContent = nd.label; }
        });
        cap.innerHTML = s.caption;
        lab.textContent = `step ${i} / ${st.length - 1}`;
      }
      model.on("change:i", draw); model.on("change:states", draw); draw();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        states = traitlets.List(traitlets.Dict()).tag(sync=True)
        i = traitlets.Int(0).tag(sync=True)
        width = traitlets.Int(680).tag(sync=True)
        height = traitlets.Int(460).tag(sync=True)
        interval = traitlets.Int(900).tag(sync=True)

    return (GraphStepper,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Rotterdam → Groningen: breadth-first search, then Dijkstra's 20-minute idea

    In 1956 Dijkstra needed a demo for the ARMAC computer that non-mathematicians could understand: *what's the shortest way to travel
    from Rotterdam to Groningen?* Below is a simplified graph of the Netherlands. The nodes are cities, the edges are roads, and the numbers are distances in units of about 10 km.

    **Breadth-first search (BFS)** explores every node 1 step away, then 2 steps away, and so on. It finds the path with the fewest
    *edges*, because it treats every road as the same length.

    **Dijkstra's algorithm** keeps a **cost** for every node: the shortest distance found so far from the source. It starts with
    $\text{cost}(s)=0$ and $\infty$ everywhere else. Then it repeats:

    1. pick the **unexplored node $u$ with the lowest cost** and mark it explored;
    2. for every neighbour $v$, **relax** the edge: $\;\text{cost}(v) \leftarrow \min\big(\text{cost}(v),\ \text{cost}(u) + w(u,v)\big)$.

    You can stop when the target is the cheapest unexplored node. By then every path cheaper than it has been explored.
    """)
    return


@app.cell
def nl_controls(mo, nl_nodes):
    nl_algo = mo.ui.radio(["BFS (every road = 1)", "Dijkstra (real distances)"], value="Dijkstra (real distances)", label="Algorithm", inline=True)
    nl_src = mo.ui.dropdown(sorted(nl_nodes), value="Rotterdam", label="From")
    nl_tgt = mo.ui.dropdown(sorted(nl_nodes), value="Groningen", label="To")
    mo.hstack([nl_algo, nl_src, nl_tgt], justify="start", gap=2)
    return nl_algo, nl_src, nl_tgt


@app.cell
def nl_stepper(
    GraphStepper,
    mo,
    nl_algo,
    nl_edges,
    nl_nodes,
    nl_src,
    nl_tgt,
    search_states,
):
    nl_states = search_states(nl_nodes, nl_edges, nl_src.value, nl_tgt.value, unit=nl_algo.value.startswith("BFS"))
    nl_stepper = mo.ui.anywidget(GraphStepper(states=nl_states, height=520))
    nl_stepper
    return nl_states, nl_stepper


@app.cell
def nl_readout(
    dijkstra_dist,
    mo,
    nl_edges,
    nl_nodes,
    nl_src,
    nl_states,
    nl_stepper,
    nl_tgt,
    pd,
    search_states,
):
    _i = min(nl_stepper.value["i"], len(nl_states) - 1)
    _st = nl_states[_i]
    _rows = sorted(
        [{"city": n["label"], "cost": n["inner"], "status": n["status"]} for n in _st["nodes"] if n["status"] != "unvisited"],
        key=lambda r: (r["status"] not in ("explored", "path", "current"), float("inf") if r["cost"] == "∞" else float(r["cost"])),
    )
    # Compare both algorithms' final answers in true distance
    _adj = {k: [] for k in nl_nodes}
    for _u, _v, _w in nl_edges:
        _adj[_u].append((_v, _w)); _adj[_v].append((_u, _w))
    _true = dijkstra_dist(_adj, nl_src.value)[nl_tgt.value]
    _bfs_final = search_states(nl_nodes, nl_edges, nl_src.value, nl_tgt.value, unit=True)[-1]
    _bfs_path = [n["id"] for n in _bfs_final["nodes"] if n["status"] == "path"]
    _edgew = {frozenset((a, b)): w for a, b, w in nl_edges}
    _bfs_nodes_on_path = set(_bfs_path)
    _bfs_len = sum(w for a, b, w in nl_edges if a in _bfs_nodes_on_path and b in _bfs_nodes_on_path and any(e["status"] == "path" and {e["u"], e["v"]} == {a, b} for e in _bfs_final["edges"]))
    mo.hstack([
        mo.vstack([
            mo.hstack([
                mo.stat(sum(r["status"] in ("explored", "path", "current") for r in _rows), label="nodes explored so far"),
                mo.stat(f"{_true:g}", label="true shortest distance (Dijkstra)"),
                mo.stat(f"{_bfs_len:g}", label="distance of BFS's fewest-hops route", caption="≥ the true distance"),
            ], justify="start"),
        ]),
        mo.ui.table(pd.DataFrame(_rows), selection=None, page_size=8, label="Nodes touched (explored first, by cost)"),
    ], widths=[1, 1], align="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this.**
    1. Press **▶ play** with Dijkstra and watch for the moment the caption says *Groningen has been reached, but an unexplored node is cheaper,
       so keep going*. This is the video's "cost 19 vs 16" moment. Reaching the target is **not** the stopping rule. *Exploring* it is.
    2. Switch to **BFS**. It finishes in fewer steps, but compare *distance of BFS's route* with the true distance. On some pairs
       (try Den Haag → Enschede) the fewest-hops route is longer.
    3. Look for a ✘ in the caption. It means a node was reached again, but by a worse path, so its cost wasn't updated.
    """), kind="info")
    return


@app.cell
def grid_algos(heapq, math):
    GRID_ROWS, GRID_COLS = 22, 40


    def grid_search(rows, cols, walls, highways, s, t, algo="dijkstra", w=1.0, metric="distance"):
        """Dijkstra / A* / bidirectional Dijkstra on a 4-connected grid.
        Edge cost = mean of the two cells' costs; a cell costs 1, or 0.25 for a highway when metric == 'time'.
        A* uses h = Manhattan distance x (min cost per step) x w. Ties in A* are broken toward smaller h. Mirrors the GridLab widget's JS exactly."""
        wall, hw = set(walls), set(highways)
        cc = lambda i: 0.25 if (metric == "time" and i in hw) else 1.0
        ec = lambda i, j: (cc(i) + cc(j)) / 2
        hs = 0.25 if metric == "time" else 1.0
        tr, tc = divmod(t, cols)

        def h(i):
            r, c = divmod(i, cols)
            return (abs(r - tr) + abs(c - tc)) * hs

        def nbrs(i):
            r, c = divmod(i, cols)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, c2 = r + dr, c + dc
                if 0 <= rr < rows and 0 <= c2 < cols and rr * cols + c2 not in wall:
                    yield rr * cols + c2

        seq = 0
        if algo != "bidi":
            ww = w if algo == "astar" else 0.0
            dist, par, done, order = {s: 0.0}, {}, set(), []
            pq = [(ww * h(s), 0.0, seq, s)]
            while pq:
                _, _, _, u = heapq.heappop(pq)
                if u in done:
                    continue
                done.add(u)
                order.append((u, 0))
                if u == t:
                    break
                for v in nbrs(u):
                    nd = dist[u] + ec(u, v)
                    if nd < dist.get(v, math.inf):
                        dist[v], par[v] = nd, u
                        seq += 1
                        heapq.heappush(pq, (nd + ww * h(v), h(v) if ww else 0.0, seq, v))
            if t not in done:
                return dict(cost=math.inf, explored=len(order), path=[], order=order)
            path = [t]
            while path[-1] != s:
                path.append(par[path[-1]])
            return dict(cost=dist[t], explored=len(order), path=path[::-1], order=order)

        D, P, done = [{s: 0.0}, {t: 0.0}], [{}, {}], [set(), set()]
        pq = [[(0.0, 0, s)], [(0.0, 0, t)]]
        mu, meet, order = math.inf, None, []
        while True:
            for k in (0, 1):
                while pq[k] and pq[k][0][2] in done[k]:
                    heapq.heappop(pq[k])
            if not pq[0] or not pq[1]:
                break
            if pq[0][0][0] + pq[1][0][0] >= mu:
                break
            k = 0 if pq[0][0][0] <= pq[1][0][0] else 1
            _, _, u = heapq.heappop(pq[k])
            done[k].add(u)
            order.append((u, k))
            for v in nbrs(u):
                nd = D[k][u] + ec(u, v)
                if nd < D[k].get(v, math.inf):
                    D[k][v], P[k][v] = nd, u
                    seq += 1
                    heapq.heappush(pq[k], (nd, seq, v))
                if v in D[1 - k] and D[k][u] + ec(u, v) + D[1 - k][v] < mu:
                    mu = D[k][u] + ec(u, v) + D[1 - k][v]
                    meet = (u, v) if k == 0 else (v, u)
        if meet is None:
            return dict(cost=math.inf, explored=len(order), path=[], order=order)
        a, b = meet
        left = [a]
        while left[-1] != s:
            left.append(P[0][left[-1]])
        right = [b]
        while right[-1] != t:
            right.append(P[1][right[-1]])
        return dict(cost=mu, explored=len(order), path=left[::-1] + right, order=order)


    def run_grid(v, algo=None, w=None, metric=None):
        """Run grid_search using a GridLab widget's .value dict, optionally overriding settings."""
        return grid_search(v["rows"], v["cols"], v["walls"], v["highways"], v["start"], v["goal"],
                           algo or v["algo"], v["w"] if w is None else w, metric or v["metric"])


    def highway_cells(rows, cols, hrows=(5, 16), hcols=(7, 32)):
        return sorted({r * cols + c for r in hrows for c in range(cols)} | {r * cols + c for c in hcols for r in range(rows)})

    return GRID_COLS, GRID_ROWS, grid_search, highway_cells, run_grid


@app.cell
def GridLab(anywidget, traitlets):
    class GridLab(anywidget.AnyWidget):
        """Paint walls / highways, drag S and T; the search runs (and animates) in JS. Python re-checks it."""
        _esm = r"""
    function render({ model, el }) {
      const R = model.get("rows"), C = model.get("cols"), S = 17, W = C * S, H = R * S;
      const show = model.get("show");
      const bar = document.createElement("div");
      bar.style.cssText = "display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-bottom:6px;font:13px sans-serif";
      const bar2 = bar.cloneNode(); 
      const canvas = document.createElement("canvas"); canvas.width = W; canvas.height = H;
      Object.assign(canvas.style, { width: "100%", maxWidth: W + "px", touchAction: "none", cursor: "crosshair", borderRadius: "6px" });
      const info = document.createElement("div"); info.style.cssText = "font:13px sans-serif;margin-top:4px;min-height:1.4em";
      el.append(bar, bar2, canvas, info);
      const ctx = canvas.getContext("2d");
      const btnCss = "padding:2px 8px;border-radius:6px;border:1px solid #999;background:transparent;color:inherit;cursor:pointer";
      const sel = (parent, label, opts, trait) => {
        const l = document.createElement("label"); l.textContent = label + " ";
        const s = document.createElement("select"); s.style.cssText = "color:inherit;background:transparent;border-radius:4px";
        opts.forEach(([v, t]) => { const o = document.createElement("option"); o.value = v; o.textContent = t; o.style.color = "#222"; s.appendChild(o); });
        s.value = model.get(trait);
        s.onchange = () => { model.set(trait, s.value); model.save_changes(); run(true); };
        model.on("change:" + trait, () => { s.value = model.get(trait); });
        l.appendChild(s); parent.appendChild(l); return l;
      };
      if (show.includes("algo")) sel(bar, "Algorithm", [["dijkstra", "Dijkstra"], ["astar", "A*"], ["bidi", "Bidirectional Dijkstra"]], "algo");
      if (show.includes("metric")) sel(bar, "Optimise", [["distance", "distance (highway = road)"], ["time", "travel time (highway 4× faster)"]], "metric");
      if (show.includes("w")) {
        const l = document.createElement("label"); l.textContent = "heuristic weight w ";
        const r = document.createElement("input"); r.type = "range"; r.min = 0; r.max = 5; r.step = 0.25; r.value = model.get("w");
        const o = document.createElement("b"); o.textContent = (+model.get("w")).toFixed(2);
        r.oninput = () => { o.textContent = (+r.value).toFixed(2); model.set("w", +r.value); run(false); };
        r.onchange = () => { model.save_changes(); run(true); };
        l.append(r, o); bar.appendChild(l);
      }
      let mode = "wall";
      const modeL = document.createElement("label"); modeL.textContent = "Paint ";
      const modeS = document.createElement("select"); modeS.style.cssText = "color:inherit;background:transparent;border-radius:4px";
      [["wall", "🧱 wall"], ["highway", "🛣 highway"], ["erase", "erase"]].forEach(([v, t]) => { const o = document.createElement("option"); o.value = v; o.textContent = t; o.style.color = "#222"; modeS.appendChild(o); });
      modeS.onchange = () => { mode = modeS.value; };
      modeL.appendChild(modeS); bar2.appendChild(modeL);
      const idx = (r, c) => r * C + c;
      const presets = {
        "Clear": () => [[], model.get("highways")],
        "Clear all": () => [[], []],
        "Random walls": () => { const w = []; for (let i = 0; i < R * C; i++) if (Math.random() < 0.28) w.push(i); return [w, model.get("highways")]; },
        "Wall with a gap": () => { const w = []; const c = Math.floor(C / 2); for (let r = 0; r < R; r++) if (r < R - 3) w.push(idx(r, c)); 
                                   for (let cc = c - 8; cc <= c; cc++) w.push(idx(R - 4, cc) ); return [w, model.get("highways")]; },
        "U-trap": () => { const w = []; const c0 = Math.floor(C * 0.6), r0 = 4, r1 = R - 5; for (let r = r0; r <= r1; r++) w.push(idx(r, c0));
                          for (let c = c0 - 10; c <= c0; c++) { w.push(idx(r0, c)); w.push(idx(r1, c)); } return [w, model.get("highways")]; },
        "Highways": () => { const h = []; for (let c = 0; c < C; c++) { h.push(idx(5, c)); h.push(idx(R - 6, c)); } for (let r = 0; r < R; r++) { h.push(idx(r, 7)); h.push(idx(r, C - 8)); } return [model.get("walls"), [...new Set(h)]]; },
      };
      for (const name of model.get("presets")) {
        const b = document.createElement("button"); b.textContent = name; b.style.cssText = btnCss;
        b.onclick = () => { let [w, h] = presets[name](); const st = new Set([model.get("start"), model.get("goal")]);
          model.set("walls", w.filter(i => !st.has(i))); model.set("highways", h); model.save_changes(); run(true); };
        bar2.appendChild(b);
      }
      // ---------- search (mirrors Python grid_search)
      class Heap { constructor() { this.a = []; }
        lt(x, y) { for (let k = 0; k < x.length - 1; k++) { if (x[k] < y[k]) return true; if (x[k] > y[k]) return false; } return false; }
        push(x) { const a = this.a; a.push(x); let i = a.length - 1; while (i > 0) { const p = (i - 1) >> 1; if (this.lt(a[i], a[p])) { [a[i], a[p]] = [a[p], a[i]]; i = p; } else break; } }
        pop() { const a = this.a, top = a[0], last = a.pop(); if (a.length) { a[0] = last; let i = 0; for (;;) { const l = 2 * i + 1, r = l + 1; let m = i;
            if (l < a.length && this.lt(a[l], a[m])) m = l; if (r < a.length && this.lt(a[r], a[m])) m = r; if (m === i) break; [a[i], a[m]] = [a[m], a[i]]; i = m; } } return top; }
        get size() { return this.a.length; } top() { return this.a[0]; } }
      function search() {
        const walls = new Set(model.get("walls")), hw = new Set(model.get("highways"));
        const metric = model.get("metric"), algo = model.get("algo"), s = model.get("start"), t = model.get("goal");
        const cc = i => (metric === "time" && hw.has(i)) ? 0.25 : 1.0, ec = (i, j) => (cc(i) + cc(j)) / 2, hs = metric === "time" ? 0.25 : 1.0;
        const tr = Math.floor(t / C), tc = t % C;
        const h = i => (Math.abs(Math.floor(i / C) - tr) + Math.abs(i % C - tc)) * hs;
        const nbrs = i => { const r = Math.floor(i / C), c = i % C, out = [];
          for (const [dr, dc] of [[-1, 0], [1, 0], [0, -1], [0, 1]]) { const rr = r + dr, c2 = c + dc;
            if (rr >= 0 && rr < R && c2 >= 0 && c2 < C && !walls.has(rr * C + c2)) out.push(rr * C + c2); } return out; };
        let seq = 0; const order = [];
        if (algo !== "bidi") {
          const ww = algo === "astar" ? model.get("w") : 0;
          const dist = new Map([[s, 0]]), par = new Map(), done = new Set(), pq = new Heap(); pq.push([ww * h(s), 0, seq, s]);
          while (pq.size) { const [, , , u] = pq.pop(); if (done.has(u)) continue; done.add(u); order.push([u, 0]); if (u === t) break;
            for (const v of nbrs(u)) { const nd = dist.get(u) + ec(u, v); if (nd < (dist.has(v) ? dist.get(v) : Infinity)) { dist.set(v, nd); par.set(v, u); seq++; pq.push([nd + ww * h(v), ww ? h(v) : 0, seq, v]); } } }
          if (!done.has(t)) return { cost: Infinity, order, path: [] };
          const path = [t]; while (path[path.length - 1] !== s) path.push(par.get(path[path.length - 1]));
          return { cost: dist.get(t), order, path };
        }
        const D = [new Map([[s, 0]]), new Map([[t, 0]])], P = [new Map(), new Map()], done = [new Set(), new Set()], pq = [new Heap(), new Heap()];
        pq[0].push([0, 0, s]); pq[1].push([0, 0, t]);
        let mu = Infinity, meet = null;
        for (;;) {
          for (const k of [0, 1]) while (pq[k].size && done[k].has(pq[k].top()[2])) pq[k].pop();
          if (!pq[0].size || !pq[1].size) break;
          if (pq[0].top()[0] + pq[1].top()[0] >= mu) break;
          const k = pq[0].top()[0] <= pq[1].top()[0] ? 0 : 1;
          const [, , u] = pq[k].pop(); done[k].add(u); order.push([u, k]);
          for (const v of nbrs(u)) {
            const nd = D[k].get(u) + ec(u, v);
            if (nd < (D[k].has(v) ? D[k].get(v) : Infinity)) { D[k].set(v, nd); P[k].set(v, u); seq++; pq[k].push([nd, seq, v]); }
            if (D[1 - k].has(v) && D[k].get(u) + ec(u, v) + D[1 - k].get(v) < mu) { mu = D[k].get(u) + ec(u, v) + D[1 - k].get(v); meet = k === 0 ? [u, v] : [v, u]; }
          }
        }
        if (!meet) return { cost: Infinity, order, path: [] };
        const left = [meet[0]]; while (left[left.length - 1] !== s) left.push(P[0].get(left[left.length - 1]));
        const right = [meet[1]]; while (right[right.length - 1] !== t) right.push(P[1].get(right[right.length - 1]));
        return { cost: mu, order, path: left.reverse().concat(right) };
      }
      // ---------- drawing
      let res = { order: [], path: [], cost: Infinity }, shown = 0, raf = null;
      function draw() {
        const walls = new Set(model.get("walls")), hw = new Set(model.get("highways"));
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = "rgba(128,128,128,0.08)"; ctx.fillRect(0, 0, W, H);
        for (const i of hw) { ctx.fillStyle = "rgba(200,170,90,0.45)"; ctx.fillRect((i % C) * S, Math.floor(i / C) * S, S, S); }
        for (let k = 0; k < shown && k < res.order.length; k++) { const [i, side] = res.order[k];
          ctx.fillStyle = side ? "rgba(178,121,162,0.6)" : "rgba(76,120,168,0.55)"; ctx.fillRect((i % C) * S + 1, Math.floor(i / C) * S + 1, S - 2, S - 2); }
        if (shown >= res.order.length) for (const i of res.path) { ctx.fillStyle = "#e8b100"; ctx.fillRect((i % C) * S + 3, Math.floor(i / C) * S + 3, S - 6, S - 6); }
        for (const i of walls) { ctx.fillStyle = "#3a3a3a"; ctx.fillRect((i % C) * S, Math.floor(i / C) * S, S, S); }
        ctx.strokeStyle = "rgba(128,128,128,0.18)"; ctx.lineWidth = 1;
        for (let c = 0; c <= C; c++) { ctx.beginPath(); ctx.moveTo(c * S, 0); ctx.lineTo(c * S, H); ctx.stroke(); }
        for (let r = 0; r <= R; r++) { ctx.beginPath(); ctx.moveTo(0, r * S); ctx.lineTo(W, r * S); ctx.stroke(); }
        for (const [i, col, lab] of [[model.get("start"), "#2ca02c", "S"], [model.get("goal"), "#d62728", "T"]]) {
          const x = (i % C) * S + S / 2, y = Math.floor(i / C) * S + S / 2;
          ctx.beginPath(); ctx.arc(x, y, S / 2 + 1, 0, 2 * Math.PI); ctx.fillStyle = col; ctx.fill();
          ctx.fillStyle = "#fff"; ctx.font = "bold 11px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle"; ctx.fillText(lab, x, y + 1); }
        const names = { dijkstra: "Dijkstra", astar: `A* (w = ${(+model.get("w")).toFixed(2)})`, bidi: "Bidirectional Dijkstra" };
        const k = Math.min(shown, res.order.length);
        info.innerHTML = `<b>${names[model.get("algo")]}</b> · explored <b>${k}</b> cells` +
          (k >= res.order.length ? (isFinite(res.cost) ? ` · path cost <b>${res.cost.toFixed(2)}</b> (${model.get("metric") === "time" ? "time units" : "steps"})` : " · <b>no path!</b>") : " …");
      }
      function run(animate) {
        if (raf) cancelAnimationFrame(raf); raf = null;
        res = search();
        if (!animate) { shown = res.order.length; draw(); return; }
        shown = 0; const per = Math.max(2, Math.ceil(res.order.length / 70));
        const tick = () => { shown += per; draw(); if (shown < res.order.length) raf = requestAnimationFrame(tick); else { shown = res.order.length; draw(); raf = null; } };
        tick();
      }
      // ---------- pointer interaction
      let drag = null;
      const cellAt = e => { const b = canvas.getBoundingClientRect(); const c = Math.floor((e.clientX - b.left) / b.width * C), r = Math.floor((e.clientY - b.top) / b.height * R);
        return (r >= 0 && r < R && c >= 0 && c < C) ? idx(r, c) : -1; };
      function paint(i) {
        if (i < 0 || i === model.get("start") || i === model.get("goal")) return;
        const walls = new Set(model.get("walls")), hw = new Set(model.get("highways"));
        if (mode === "wall") { walls.add(i); } else if (mode === "highway") { hw.add(i); walls.delete(i); } else { walls.delete(i); hw.delete(i); }
        model.set("walls", [...walls]); model.set("highways", [...hw]);
      }
      canvas.addEventListener("pointerdown", e => { const i = cellAt(e); if (i < 0) return; canvas.setPointerCapture(e.pointerId);
        if (i === model.get("start")) drag = "start"; else if (i === model.get("goal")) drag = "goal"; else { drag = "paint"; paint(i); } run(false); });
      canvas.addEventListener("pointermove", e => { if (!drag) return; const i = cellAt(e); if (i < 0) return;
        if (drag === "paint") paint(i);
        else if (!model.get("walls").includes(i) && i !== model.get(drag === "start" ? "goal" : "start")) model.set(drag, i);
        run(false); });
      const end = () => { if (!drag) return; drag = null; model.save_changes(); run(true); };
      canvas.addEventListener("pointerup", end); canvas.addEventListener("pointercancel", end);
      for (const tr of ["walls", "highways", "start", "goal", "algo", "metric", "w"]) model.on("change:" + tr, () => { if (!drag) run(false); });
      run(true);
      return () => { if (raf) cancelAnimationFrame(raf); };
    }
    export default { render };
    """
        rows = traitlets.Int(22).tag(sync=True)
        cols = traitlets.Int(40).tag(sync=True)
        walls = traitlets.List(traitlets.Int()).tag(sync=True)
        highways = traitlets.List(traitlets.Int()).tag(sync=True)
        start = traitlets.Int(11 * 40 + 4).tag(sync=True)
        goal = traitlets.Int(11 * 40 + 35).tag(sync=True)
        algo = traitlets.Unicode("astar").tag(sync=True)
        metric = traitlets.Unicode("distance").tag(sync=True)
        w = traitlets.Float(1.0).tag(sync=True)
        show = traitlets.List(traitlets.Unicode(), default_value=["algo", "w"]).tag(sync=True)
        presets = traitlets.List(traitlets.Unicode(), default_value=["Clear", "Random walls", "Wall with a gap", "U-trap"]).tag(sync=True)

    return (GridLab,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · A\*: punish the illogical directions

    On a big map, Dijkstra is fast but **wasteful**. Going from Newark Airport to the Central Park Zoo, it explores all 10 km journeys, then all 20 km journeys, and so on,
    which includes Staten Island and large parts of New Jersey: about 65,000 nodes. It spreads out in *every* direction, even directly away from the target.

    **A\*** is Dijkstra with one change. Nodes are ordered by their cost **plus a guess of the remaining distance**:

    $$f(v) = \underbrace{\text{cost}(v)}_{\text{known, from source}} + \; w\cdot\underbrace{h(v)}_{\text{straight-line distance to target}}$$

    Polylog's picture from the video: lift every node to a **height equal to its straight-line distance from the target**. Moving uphill
    (away from the target) now costs extra, so the search rolls downhill toward the target. With $w = 0$ you get plain Dijkstra. With $w = 1$
    the guess never *over*estimates the true distance, so A\* still returns the **shortest** path. With $w > 1$ the slope is steeper and the search is more
    aggressive, but it may return a path that is *not* the shortest.

    On a grid where you can only step up, down, left, or right, the straight-line guess is the **Manhattan distance** $|\Delta r| + |\Delta c|$.
    """)
    return


@app.cell
def grid_astar(GridLab, mo):
    grid_astar = mo.ui.anywidget(GridLab(algo="astar", w=1.0, show=["algo", "w"]))
    grid_astar
    return (grid_astar,)


@app.cell
def astar_readout(alt, grid_astar, mo, np, pd, run_grid):
    _v = grid_astar.value
    _rows = []
    for _label, _algo, _w in [("Dijkstra", "dijkstra", 0), ("A* (w = 1)", "astar", 1.0), (f"A* (w = {_v['w']:g}, yours)", "astar", _v["w"]), ("Bidirectional Dijkstra", "bidi", 0)]:
        _r = run_grid(_v, algo=_algo, w=_w)
        _rows.append({"algorithm": _label, "cells explored": _r["explored"], "path cost": _r["cost"]})
    _df = pd.DataFrame(_rows)
    _best = _df["path cost"].min()
    _df["shortest?"] = np.where(np.isclose(_df["path cost"], _best), "✔ shortest", "✘ longer")
    _chart = alt.Chart(_df).mark_bar().encode(
        y=alt.Y("algorithm:N", sort=None, title=None),
        x=alt.X("cells explored:Q"),
        color=alt.Color("shortest?:N", scale=alt.Scale(domain=["✔ shortest", "✘ longer"], range=["#4c78a8", "#d62728"]), legend=alt.Legend(orient="bottom")),
        tooltip=list(_df.columns),
    ).properties(width="container", height=150, title="Same maze, same S and T. Python re-runs every algorithm.")
    _d = _df.iloc[0]["cells explored"] / max(_df.iloc[1]["cells explored"], 1)
    mo.vstack([
        mo.hstack([
            mo.stat(f"{_d:.1f}×", label="Dijkstra explores this many times more than A* (w = 1)"),
            mo.stat(f"{_best:g}", label="shortest path cost"),
        ], justify="start"),
        _chart,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this.**
    1. On the empty grid, switch between **Dijkstra** and **A\***. Dijkstra floods a diamond in every direction, while A\* heads straight for T.
       The video saw the same thing in New York: about 65,000 nodes for Dijkstra and about 7,000 for A\*.
    2. Press **U-trap**. A\* runs into the cup, fills it, and only then goes around. *At worst, A\* checks as many nodes as Dijkstra.*
    3. Push **w** to 3 or more on **Random walls**. The search gets very narrow, but the bar turns red: the path is no longer the shortest.
       **w = 1 is the most you can use and still be guaranteed the shortest path.**
    4. Drag **S** or **T** while you watch the search repaint live.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Shortest doesn't mean fastest

    You don't care about distance. You care about **time**. So the edge weights become travel times, and the heuristic has to become a
    *time* that never overestimates: the straight-line distance divided by the **maximum speed allowed anywhere on the graph**,

    $$h_{\text{time}}(v) = \frac{\text{straight-line distance}(v, t)}{v_{\max}}.$$

    This guess is honest, but it's a **huge underestimate**. It assumes you could drive at highway speed along the straight line, even
    through a residential maze. A weak heuristic means a flat "hill", and A\* falls back to behaving like Dijkstra, while it still pays for computing $h$ (and in the real world,
    a square root) at every node. Peter Sanders, in the video: *for travel time, A\* loses against a well-tuned Dijkstra.*

    In the grid below, **sandy cells are highways** that are 4× faster when you optimise travel time, so $h$ is divided by 4.
    """)
    return


@app.cell
def grid_time(GRID_COLS, GRID_ROWS, GridLab, highway_cells, mo):
    grid_time = mo.ui.anywidget(GridLab(
        algo="astar", metric="time", w=1.0, show=["algo", "metric"],
        highways=highway_cells(GRID_ROWS, GRID_COLS),
        start=3 * GRID_COLS + 2, goal=19 * GRID_COLS + 37,
        presets=["Clear", "Clear all", "Highways", "Random walls"],
    ))
    grid_time
    return (grid_time,)


@app.cell
def time_readout(alt, grid_time, math, mo, pd, run_grid):
    _v = grid_time.value
    _rows = []
    for _metric in ["distance", "time"]:
        _d = run_grid(_v, algo="dijkstra", w=0, metric=_metric)
        _a = run_grid(_v, algo="astar", w=1.0, metric=_metric)
        _rows += [
            {"optimise": _metric, "algorithm": "Dijkstra", "cells explored": _d["explored"]},
            {"optimise": _metric, "algorithm": "A*", "cells explored": _a["explored"]},
        ]
        assert math.isclose(_d["cost"], _a["cost"])  # admissible heuristic → same optimum
    _df = pd.DataFrame(_rows)
    _get = lambda m, a: int(_df.query("optimise == @m and algorithm == @a")["cells explored"].iloc[0])
    _chart = alt.Chart(_df).mark_bar().encode(
        x=alt.X("algorithm:N", title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("cells explored:Q"),
        color=alt.Color("algorithm:N", scale=alt.Scale(domain=["Dijkstra", "A*"], range=["#4c78a8", "#f58518"]), legend=None),
        column=alt.Column("optimise:N", title="what we optimise"),
        tooltip=list(_df.columns),
    ).properties(width=220, height=200)
    mo.vstack([
        mo.hstack([
            mo.stat(f"{_get('distance', 'Dijkstra') / max(_get('distance', 'A*'), 1):.1f}×", label="A* advantage when optimising distance"),
            mo.stat(f"{_get('time', 'Dijkstra') / max(_get('time', 'A*'), 1):.1f}×", label="A* advantage when optimising travel time"),
            mo.stat(f"{_get('time', 'A*') / max(_get('distance', 'A*'), 1):.1f}×", label="A* work: time vs distance"),
        ], justify="start"),
        _chart,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this.** Flip **Optimise** between *distance* and *travel time* while A\* is selected. Watch the blue blob swell: the heuristic
    just got 4× flatter. In the video's New York test, A\* explored 9.5× fewer nodes than Dijkstra for distance, but switching to travel
    time **tripled** A\*'s work, while Dijkstra's grew only 10%. Then press **Clear all** (no highways). With no fast roads, why does travel time stop
    mattering? Hint: $v_{\max}$ in the formula only exists because highways exist.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Search from both ends

    Run one Dijkstra **forward from the source** and one **backward from the target**, always growing the side whose next node is cheaper.
    If the two points are $R$ apart, a single search covers a disc of area about $\pi R^2$. Two searches each reach about $R/2$:

    $$2\cdot \pi\left(\tfrac{R}{2}\right)^2 = \tfrac12\,\pi R^2 .$$

    That's **half the work**, and sometimes better. The stopping rule is subtle. Keep $\mu$ = the best complete path seen where the two searches touch,
    and stop once $\text{top}_{\text{fwd}} + \text{top}_{\text{bwd}} \ge \mu$. Stopping the moment the two frontiers meet is a classic bug,
    because the first meeting point is not always on the shortest path.

    In the video, Carnegie Hall → Wall Street went from about 7,200 explored nodes to about 2,600, close to 3× fewer.
    """)
    return


@app.cell
def grid_bidi(GridLab, mo):
    grid_bidi = mo.ui.anywidget(GridLab(algo="bidi", show=["algo"], presets=["Clear", "Random walls", "Wall with a gap"]))
    grid_bidi
    return (grid_bidi,)


@app.cell
def bidi_readout(alt, grid_bidi, grid_search, math, mo, pd, run_grid):
    _v = grid_bidi.value
    _uni = run_grid(_v, algo="dijkstra")
    _bi = run_grid(_v, algo="bidi")
    # Sweep on an open grid: explored cells vs distance R, uni vs bi
    _N = 61
    _c = (_N // 2) * _N + _N // 2
    _sweep = []
    for _R in range(2, 29, 2):
        _t = _c + _R  # target R cells to the right
        _sweep += [
            {"R": _R, "search": "one-way Dijkstra", "explored": grid_search(_N, _N, [], [], _c, _t, "dijkstra")["explored"]},
            {"R": _R, "search": "bidirectional", "explored": grid_search(_N, _N, [], [], _c, _t, "bidi")["explored"]},
        ]
    _sw = pd.DataFrame(_sweep)
    _chart = alt.Chart(_sw).mark_line(point=True).encode(
        x=alt.X("R:Q", title="distance R between S and T (open grid)"),
        y=alt.Y("explored:Q", title="cells explored"),
        color=alt.Color("search:N", scale=alt.Scale(domain=["one-way Dijkstra", "bidirectional"], range=["#4c78a8", "#b279a2"]), legend=alt.Legend(orient="top-left")),
    ).properties(width="container", height=230, title="Both curves grow like R², and the purple one has about half the constant")
    _ratio = _sw.pivot(index="R", columns="search", values="explored")
    _r = (_ratio["bidirectional"] / _ratio["one-way Dijkstra"]).iloc[-1]
    mo.vstack([
        mo.hstack([
            mo.stat(_uni["explored"], label="one-way Dijkstra (your grid)"),
            mo.stat(_bi["explored"], label="bidirectional (your grid)"),
            mo.stat(f"{_uni['explored'] / max(_bi['explored'], 1):.2f}×", label="speed-up"),
            mo.stat("✔" if math.isclose(_uni["cost"], _bi["cost"]) or _uni["cost"] == _bi["cost"] else "✘", label="same path cost?"),
            mo.stat(f"{_r:.2f}", label="bi / uni at R = 28 (theory ≈ 0.5)"),
        ], justify="start"),
        _chart,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this.** Move S and T far apart on the empty grid and compare the speed-up with the theoretical 2×. Then press **Wall with a gap**.
    Now the two blobs grow on opposite sides of the wall, so how much do they save? Bidirectional search is a big win, but it's still
    **hierarchy-blind**. As Ben Strasser puts it, it will still check whether driving halfway into the In-N-Out drive-through gives a shorter path.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Road hierarchy and the pre-processing trade-off

    Humans don't route like Dijkstra. You **meander out on local roads, hop on a highway, and hop off near the destination**. Early in-car GPS
    units (1990s) hard-coded this. Surveyors labelled every road (*express highway / major local road / narrow road*). The device ran a
    bidirectional Dijkstra on narrow roads inside a small **candidate area**, then moved up a level, and then up again to highways.

    The flaw: **how big should the candidate area be?** If it's too small, you get a highway route when a chain of local roads was really shorter. If it's too large,
    you're back to Dijkstra. Nothing *proves* the answer is the shortest.

    The deeper idea is a **trade-off between pre-processing time and query time**. At one extreme, precompute every shortest path into a giant lookup table.
    At the other, precompute nothing and run Dijkstra. Google needs queries **under a millisecond** with **reasonable** pre-processing.
    """)
    return


@app.cell
def table_N(mo):
    table_N = mo.ui.slider(steps=[1e3, 1e4, 1e5, 1e6, 1e7, 6.4e7, 1e8], value=6.4e7, label="Intersections N in the network", show_value=True, full_width=True)
    table_N
    return (table_N,)


@app.cell
def tradeoff_chart(alt, mo, pd, table_N):
    _N = table_N.value
    _bytes = _N ** 2 * 2  # one 2-byte "next hop" entry per ordered pair
    _dijkstra_query = 7.0 * _N / 6.4e7  # video: ~7 s per query on 64M nodes, scale linearly
    _table_build = _N * _dijkstra_query  # one full Dijkstra per source
    _fmt_bytes = lambda b: next(f"{b / 10 ** e:.1f} {u}" for e, u in [(15, "PB"), (12, "TB"), (9, "GB"), (6, "MB"), (3, "kB"), (0, "B")] if b >= 10 ** e or e == 0)
    _fmt_t = lambda s: (f"{s / 3.156e7:.1f} years" if s > 3.156e7 else f"{s / 86400:.1f} days" if s > 86400 else f"{s / 3600:.1f} h" if s > 3600 else f"{s:.2g} s")
    _pts = pd.DataFrame([
        {"method": "Dijkstra (no pre-processing)", "preprocessing [s]": 0.1, "query [s]": _dijkstra_query},
        {"method": "Lookup table of all paths", "preprocessing [s]": _table_build, "query [s]": 1e-6},
        {"method": "Customizable contraction hierarchy", "preprocessing [s]": 6000 * _N / 6.4e7 + 1, "query [s]": 2e-4},
    ])
    _zone = pd.DataFrame([{"x": 0.1, "x2": 3.6e4, "y": 1e-7, "y2": 1e-3}])
    _x = alt.X("preprocessing [s]:Q", scale=alt.Scale(type="log", domain=[0.05, 1e10]), title="pre-processing time (s, log; 0.1 = none)")
    _y = alt.Y("query [s]:Q", scale=alt.Scale(type="log", domain=[1e-7, 1e2]), title="query time per route (s, log)")
    _chart = (
        alt.Chart(_zone).mark_rect(color="#54a24b", opacity=0.15).encode(x=alt.X("x:Q", scale=alt.Scale(type="log")), x2="x2", y=alt.Y("y:Q", scale=alt.Scale(type="log")), y2="y2")
        + alt.Chart(_pts).mark_point(size=220, filled=True).encode(x=_x, y=_y, color=alt.Color("method:N", scale=alt.Scale(range=["#4c78a8", "#d62728", "#e8b100"]), legend=alt.Legend(orient="bottom", columns=1)), tooltip=list(_pts.columns))
        + alt.Chart(_pts).mark_text(dy=-16, fontSize=11).encode(x=_x, y=_y, text="method:N")
    ).properties(width="container", height=300, title="Green zone: under 1 ms per query, with at most ~10 h of pre-processing")
    mo.vstack([
        mo.hstack([
            mo.stat(_fmt_bytes(_bytes), label="lookup table size (2 bytes × N²)"),
            mo.stat(_fmt_t(_table_build), label="building it: one Dijkstra per node, single core"),
            mo.stat(_fmt_t(_dijkstra_query), label="one Dijkstra query"),
        ], justify="start"),
        _chart,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this.** Set N to 64 million (North America). The table needs about **8 PB**, which the video compares to *all of Wikipedia 16 times over*.
    Building it takes **more than a decade** of single-core compute, and any road closure invalidates parts of it. Now slide N down to 10,000:
    a city-sized table is just 200 MB. **Quadratic storage is what kills it**, and that's why we need a smarter middle ground: the green zone.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Contraction hierarchies: rank the nodes, add shortcuts, search only upward

    Instead of humans labelling roads, the computer **ranks every node by importance**:

    * **Nested dissection.** Find a *small set of nodes that cuts the graph roughly in half*: the bridge between two towns, or the 102 Mississippi crossings
      that split North America. Give them the **highest ranks**. Then recurse into each half (the Ohio River, the Rockies, …) until every node is ranked.
    * **Contract from the bottom.** "Pull down" the nodes in rank order. When you contract node $v$, look at its neighbours with **higher** rank. Every pair $a, b$
      forms a **lower triangle** $a - v - b$. Add a **shortcut** $a - b$ with weight $w(a,v)+w(v,b)$. If an edge $a - b$ already exists, keep the minimum.
    * **Query.** Run a **bidirectional Dijkstra where each side may only move *up* the hierarchy**. The two searches meet near the top.
      Answer $=\min_x \big(d_\uparrow(s,x)+d_\uparrow(t,x)\big)$.

    Without shortcuts, the upward-only search misses paths that dip through lower-ranked nodes (the video's "cost 15, but there is a path that costs 3").
    Shortcuts are exactly those dips, precomputed.

    In the stepper, the **number inside each node is its rank** (higher = more important), and the vertical position follows the rank too.
    """)
    return


@app.cell
def ch_algos(heapq, math):
    def ch_contract(nodes, edges, rank):
        """Contract nodes in increasing rank; for each lower triangle a-v-b add/lower shortcut a-b. Returns (adj, shortcuts, steps)."""
        adj = {k: {} for k in nodes}
        for u, v, w in edges:
            adj[u][v] = min(w, adj[u].get(v, math.inf))
            adj[v][u] = adj[u][v]
        original = {frozenset((u, v)) for u, v, _ in edges}
        shortcuts, steps = {}, []
        for v in sorted(nodes, key=rank.get):
            up = sorted([u for u in adj[v] if rank[u] > rank[v]], key=rank.get)
            changes = []
            for _a in range(len(up)):
                for _b in range(_a + 1, len(up)):
                    a, b = up[_a], up[_b]
                    c = adj[v][a] + adj[v][b]
                    if b not in adj[a]:
                        adj[a][b] = adj[b][a] = c
                        shortcuts[frozenset((a, b))] = c
                        changes.append((a, b, c, "new shortcut"))
                    elif c < adj[a][b]:
                        adj[a][b] = adj[b][a] = c
                        shortcuts[frozenset((a, b))] = c
                        changes.append((a, b, c, "lowered"))
                    else:
                        changes.append((a, b, adj[a][b], "kept"))
            steps.append((v, up, changes))
        return adj, shortcuts, steps, original


    def ch_upward(adj, rank, src):
        """Dijkstra that only relaxes edges to higher-ranked nodes. Returns (dist, parent, settled order)."""
        dist, par, done, order, pq = {src: 0}, {}, set(), [], [(0, src)]
        while pq:
            d, u = heapq.heappop(pq)
            if u in done:
                continue
            done.add(u)
            order.append(u)
            for v, w in adj[u].items():
                if rank[v] > rank[u] and d + w < dist.get(v, math.inf):
                    dist[v], par[v] = d + w, u
                    heapq.heappush(pq, (d + w, v))
        return dist, par, order


    def ch_query(adj, rank, s, t):
        dF, pF, oF = ch_upward(adj, rank, s)
        dB, pB, oB = ch_upward(adj, rank, t)
        best, meet = math.inf, None
        for x in dF:
            if x in dB and dF[x] + dB[x] < best:
                best, meet = dF[x] + dB[x], x
        path = []
        if meet is not None:
            a = [meet]
            while a[-1] != s:
                a.append(pF[a[-1]])
            b = [meet]
            while b[-1] != t:
                b.append(pB[b[-1]])
            path = a[::-1] + b[1:]
        return dict(cost=best, meet=meet, path=path, fwd=oF, bwd=oB, dF=dF, dB=dB, space=len(oF) + len(oB))


    def plain_adj(nodes, edges):
        adj = {k: [] for k in nodes}
        for u, v, w in edges:
            adj[u].append((v, w)); adj[v].append((u, w))
        return adj

    return ch_contract, ch_query, plain_adj


@app.cell(hide_code=True)
def ch_graph():
    # Two towns joined by two crossings (E, F): a tiny version of "the bridge" and "the Mississippi" in the video.
    ch_pos = {"A": (0.03, 0.5), "B": (0.17, 0.12), "C": (0.17, 0.88), "D": (0.32, 0.5), "E": (0.5, 0.2), "F": (0.5, 0.8),
              "G": (0.68, 0.5), "H": (0.83, 0.12), "I": (0.83, 0.88), "J": (0.97, 0.5)}
    ch_edges = [("A", "B", 4), ("A", "C", 1), ("B", "D", 3), ("C", "D", 1), ("D", "E", 5), ("D", "F", 4), ("B", "E", 7), ("C", "F", 3),
                ("E", "F", 6), ("E", "G", 2), ("F", "G", 5), ("E", "H", 6), ("F", "I", 4), ("G", "H", 1), ("G", "I", 3),
                ("H", "J", 2), ("I", "J", 1), ("G", "J", 6)]
    _orders = {  # listed from LOWEST to HIGHEST rank
        "Nested dissection (cut {E,F}, then {B,C} and {G,J})": "IHDAJGCBFE",
        "Same cuts, ranks shuffled within each cut": "AHIDGJBCEF",
        "Bad: most important nodes contracted first": "EFBCGJADHI",
        "Left to right": "ABCDEFGHIJ",
    }
    ch_rankings = {k: {n: r + 1 for r, n in enumerate(o)} for k, o in _orders.items()}
    return ch_edges, ch_pos, ch_rankings


@app.cell
def ch_controls(ch_pos, ch_rankings, mo):
    ch_order = mo.ui.dropdown(list(ch_rankings), value="Nested dissection (cut {E,F}, then {B,C} and {G,J})", label="Ranking")
    ch_src = mo.ui.dropdown(sorted(ch_pos), value="A", label="Query from")
    ch_tgt = mo.ui.dropdown(sorted(ch_pos), value="J", label="to")
    ch_use_shortcuts = mo.ui.switch(value=True, label="use shortcuts")
    mo.hstack([ch_order, ch_src, ch_tgt, ch_use_shortcuts], justify="start", gap=2)
    return ch_order, ch_src, ch_tgt, ch_use_shortcuts


@app.cell
def ch_states_cell(
    GraphStepper,
    ch_contract,
    ch_edges,
    ch_order,
    ch_pos,
    ch_query,
    ch_rankings,
    ch_src,
    ch_tgt,
    ch_use_shortcuts,
    dijkstra_dist,
    math,
    mo,
    plain_adj,
):
    def _ch_states(rank, s, t, use_shortcuts):
        _adj, _sc, _steps, _orig = ch_contract(ch_pos, ch_edges, rank)
        if not use_shortcuts:  # upward search on the original graph only
            _adj = {k: {} for k in ch_pos}
            for u, v, w in ch_edges:
                _adj[u][v] = _adj[v][u] = w
        hx = lambda n: ch_pos[n][0]
        hy = lambda n: 1 - (rank[n] - 1) / (len(rank) - 1)
        states, contracted, shown_sc = [], set(), {}

        def edges_now(focus=(), path=(), up_f=(), up_b=()):
            pe = {frozenset(p) for p in zip(path, path[1:])}
            es = []
            for u, v, w in ch_edges:
                k = frozenset((u, v))
                ww = shown_sc.get(k, w) if k in shown_sc else w
                st = "path" if k in pe else "focus" if k in focus else "tree" if k in up_f else "tree_b" if k in up_b else "normal"
                es.append(dict(u=u, v=v, w=ww, status=st))
            for k, w in shown_sc.items():
                if k in {frozenset((u, v)) for u, v, _ in ch_edges}:
                    continue
                u, v = sorted(k)
                st = "path" if k in pe else "focus" if k in focus else "tree" if k in up_f else "tree_b" if k in up_b else "shortcut"
                es.append(dict(u=u, v=v, w=w, status=st, dashed=True))
            return es

        def nodes_now(status=lambda n: "unvisited", inner=lambda n: str(rank[n])):
            return [dict(id=n, label=n, x=hx(n), y=hy(n), status=status(n), inner=inner(n),
                         role="source" if n == s else "target" if n == t else "") for n in ch_pos]

        states.append(dict(nodes=nodes_now(), edges=edges_now(),
                           caption="<b>Phase 1, pre-processing.</b> Nodes are pulled down by rank (rank 1 at the bottom). We contract them from the bottom up."))
        if use_shortcuts:
            for v, up, changes in _steps:
                tri = set()
                for a, b, c, kind in changes:
                    tri |= {frozenset((v, a)), frozenset((v, b))}
                    if kind != "kept":
                        shown_sc[frozenset((a, b))] = c
                    tri.add(frozenset((a, b)))
                msg = ("; ".join(f"{a}–{v}–{b}: {kind} {a}–{b} = {c}" for a, b, c, kind in changes)
                       if changes else (f"only one higher neighbour ({up[0]}), so no triangle" if up else "no higher neighbours"))
                states.append(dict(nodes=nodes_now(lambda n: "current" if n == v else "contracted" if n in contracted else "unvisited"),
                                   edges=edges_now(focus=tri),
                                   caption=f"Contract <b>{v}</b> (rank {rank[v]}). Higher neighbours: {', '.join(up) or 'none'}. {msg}"))
                contracted.add(v)
        q = ch_query(_adj, rank, s, t)
        true = dijkstra_dist(plain_adj(ch_pos, ch_edges), s)[t]
        upf = {frozenset((u, v)) for u in q["fwd"] for v in q["fwd"] if u != v}
        states.append(dict(nodes=nodes_now(lambda n: "explored" if n in q["fwd"] else "unvisited",
                                           lambda n: f"{q['dF'][n]:g}" if n in q["dF"] else "·"),
                           edges=edges_now(),
                           caption=f"<b>Phase 3, query.</b> Search <i>upward only</i> from {s}: it settles {len(q['fwd'])} nodes ({', '.join(q['fwd'])}). The numbers are upward distances."))
        states.append(dict(nodes=nodes_now(lambda n: ("both" if n in q["fwd"] and n in q["bwd"] else "explored" if n in q["fwd"] else "explored_b" if n in q["bwd"] else "unvisited"),
                                           lambda n: f"{q['dB'][n]:g}" if n in q["dB"] else (f"{q['dF'][n]:g}" if n in q["dF"] else "·")),
                           edges=edges_now(),
                           caption=f"Search upward from {t} too ({len(q['bwd'])} nodes, purple). The green nodes were reached from both sides. Candidates: "
                                   + ", ".join(f"{x}: {q['dF'][x]:g}+{q['dB'][x]:g}" for x in q["dF"] if x in q["dB"])))
        ok = math.isclose(q["cost"], true)
        states.append(dict(nodes=nodes_now(lambda n: "path" if n in q["path"] else ("explored" if n in q["fwd"] else "explored_b" if n in q["bwd"] else "unvisited")),
                           edges=edges_now(path=q["path"]),
                           caption=(f"Best meeting node <b>{q['meet']}</b>: cost <b>{q['cost']:g}</b>. Plain Dijkstra says {true:g}. "
                                    + ("✔ Correct, and only " + str(q["space"]) + " nodes were settled." if ok else
                                       "✘ <b>Wrong!</b> The upward-only search missed a path through lower-ranked nodes. Turn the shortcuts back on.")) if q["meet"] else
                                   "✘ The upward searches never met."))
        return states, q, true, len(_sc)


    ch_states, ch_q, ch_true, ch_n_shortcuts = _ch_states(ch_rankings[ch_order.value], ch_src.value, ch_tgt.value, ch_use_shortcuts.value)
    ch_stepper = mo.ui.anywidget(GraphStepper(states=ch_states, height=470, interval=1300))
    ch_stepper
    return ch_n_shortcuts, ch_q, ch_true


@app.cell
def ch_readout(
    ch_contract,
    ch_edges,
    ch_n_shortcuts,
    ch_order,
    ch_pos,
    ch_q,
    ch_query,
    ch_rankings,
    ch_src,
    ch_tgt,
    ch_true,
    dijkstra_dist,
    math,
    mo,
    plain_adj,
):
    _rank = ch_rankings[ch_order.value]
    _adj_sc, _sc, _, _ = ch_contract(ch_pos, ch_edges, _rank)
    _adj_plain = {k: {} for k in ch_pos}
    for _u, _v, _w in ch_edges:
        _adj_plain[_u][_v] = _adj_plain[_v][_u] = _w
    _pa = plain_adj(ch_pos, ch_edges)
    _pairs = [(a, b) for a in ch_pos for b in ch_pos if a < b]
    _ok_sc = _ok_plain = _space = 0
    for _a, _b in _pairs:
        _t = dijkstra_dist(_pa, _a)[_b]
        _q1 = ch_query(_adj_sc, _rank, _a, _b)
        _q0 = ch_query(_adj_plain, _rank, _a, _b)
        _ok_sc += math.isclose(_q1["cost"], _t)
        _ok_plain += _q0["meet"] is not None and math.isclose(_q0["cost"], _t)
        _space += _q1["space"]
    mo.hstack([
        mo.stat(f"{ch_q['cost']:g}" if ch_q["meet"] else "—", label=f"CH answer {ch_src.value}→{ch_tgt.value}", caption=f"Dijkstra: {ch_true:g}"),
        mo.stat(ch_n_shortcuts, label="shortcuts added by this ranking"),
        mo.stat(f"{_ok_sc}/{len(_pairs)}", label="pairs correct WITH shortcuts"),
        mo.stat(f"{_ok_plain}/{len(_pairs)}", label="pairs correct WITHOUT shortcuts"),
        mo.stat(f"{_space / len(_pairs):.1f}", label="avg nodes settled per query"),
    ], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this.**
    1. Step through phase 1 with **nested dissection**. Watch D, C and G create dashed shortcuts as they're pulled down.
    2. Turn **use shortcuts** off and query **A → J**. The upward-only search returns a wrong (too long) answer, just like the video's "15 vs 3".
       The readout counts how many of the 45 pairs break.
    3. Switch the ranking. **Every ranking gets 45/45 correct with shortcuts.** That's why the video says *any* ranking is valid, and ranks within a cut can be
       shuffled freely. Bad rankings pay in a different currency: more shortcuts and bigger searches.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · A bad ranking is slow, a good one is logarithmic

    The video's cleanest example is **a road that is just a line of nodes**.

    * Contract **left to right**: no shortcuts are needed at all, but a query between the two ends must climb through **every** node. That's no better than Dijkstra.
    * **Nested dissection**: the middle node gets the top rank, then the middles of each half, and so on. Every node's upward search is a short climb of about $\log_2 n$ steps.
      The price is a few extra shortcuts.

    *"You want this trade-off between the number of shortcuts that you introduce and how much faster the query gets."*
    """)
    return


@app.cell
def line_n(mo):
    line_n = mo.ui.slider(steps=[7, 15, 31, 63, 127, 255], value=31, label="nodes on the line", show_value=True, full_width=True)
    line_n
    return (line_n,)


@app.cell
def line_readout(alt, ch_contract, ch_query, line_n, mo, np, pd, random):
    def _nd_rank(lo, hi, top, rank):
        """Nested dissection on a path lo..hi: middle node gets the highest remaining rank, recurse."""
        if lo > hi:
            return top
        mid = (lo + hi) // 2
        rank[mid] = top
        top = _nd_rank(lo, mid - 1, top - 1, rank)
        return _nd_rank(mid + 1, hi, top, rank)


    _n = line_n.value
    _nodes = list(range(_n))
    _edges = [(k, k + 1, 1) for k in range(_n - 1)]
    _ranks = {"left to right": {k: k + 1 for k in _nodes}}
    _r = {}
    _nd_rank(0, _n - 1, _n, _r)
    _ranks["nested dissection"] = _r
    _rng = random.Random(0)
    _perm = _nodes[:]
    _rng.shuffle(_perm)
    _ranks["random"] = {k: i + 1 for i, k in enumerate(_perm)}
    _pairs = [(_rng.randrange(_n), _rng.randrange(_n)) for _ in range(300)] + [(0, _n - 1)]
    _rows = []
    for _name, _rk in _ranks.items():
        _adj, _sc, _, _ = ch_contract(_nodes, _edges, _rk)
        _sp = [ch_query(_adj, _rk, a, b)["space"] for a, b in _pairs]
        _rows.append({"ranking": _name, "shortcuts": len(_sc), "avg nodes settled": float(np.mean(_sp[:-1])), "end-to-end settled": _sp[-1]})
    line_df = pd.DataFrame(_rows)
    _long = line_df.melt(id_vars="ranking", value_vars=["shortcuts", "avg nodes settled", "end-to-end settled"])
    _chart = alt.Chart(_long).mark_bar().encode(
        x=alt.X("value:Q", title=None),
        y=alt.Y("ranking:N", title=None),
        color=alt.Color("ranking:N", scale=alt.Scale(domain=["left to right", "random", "nested dissection"], range=["#d62728", "#999", "#e8b100"]), legend=None),
        row=alt.Row("variable:N", title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
        tooltip=["ranking", "variable", "value"],
    ).properties(width=520, height=70)
    mo.vstack([mo.ui.table(line_df.round(1), selection=None), _chart])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The real thing: a customizable contraction hierarchy (CCH) on North America

    Ben Strasser ran it for the video. The three phases have very different costs, and they rerun at very different times:
    """)
    return


@app.cell
def cch_numbers(mo):
    mo.vstack([
        mo.hstack([
            mo.stat("≈ 1 h 40 min", label="Phase 1: order nodes + add shortcuts", caption="reruns only if the road graph changes"),
            mo.stat("≈ 1 s", label="Phase 2: compute shortcut weights", caption="7–8 s single-core; reruns on every traffic update"),
            mo.stat("≈ 200 µs", label="Phase 3: one query", caption="vs ≈ 7 s for Dijkstra"),
        ], justify="start"),
        mo.hstack([
            mo.stat(f"{7 / 200e-6:,.0f}×", label="faster than Dijkstra"),
            mo.stat("1,450", label="nodes settled on average", caption="vs most of 64 M for Dijkstra"),
            mo.stat(f"{64e6 / 1450:,.0f}×", label="smaller search space", caption="the video quotes ≈ 44,000×"),
        ], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this.** Double the line length a few times. With *left to right*, the end-to-end search grows **linearly** (all $n$ nodes). With *nested dissection*, it grows by
    only about **two nodes per doubling** (one climb from each end, about $\log_2 n$ steps each). That logarithmic climb is why a San Francisco → Montreal query touches
    only about 1,236 nodes: a cluster near each end, plus the big cuts like the Mississippi at the top of the hierarchy.
    """), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Sandbox: build your own city

    Everything is switched on: paint walls and highways, drag S and T, and pick the algorithm, objective, and heuristic weight. The table re-runs **all** algorithms on your map.
    Some challenges:

    * Build a map where **A\* explores more than half of what Dijkstra explores**, even with distance as the objective.
    * Build a map where **bidirectional** search is *worse* than one-way Dijkstra.
    * Use a highway grid and *time*. Find the w at which A\*'s path stops being the fastest.
    """)
    return


@app.cell
def grid_sandbox(GRID_COLS, GRID_ROWS, GridLab, highway_cells, mo):
    grid_sandbox = mo.ui.anywidget(GridLab(
        algo="astar", metric="time", show=["algo", "metric", "w"], highways=highway_cells(GRID_ROWS, GRID_COLS),
        presets=["Clear", "Clear all", "Random walls", "Wall with a gap", "U-trap", "Highways"],
    ))
    grid_sandbox
    return (grid_sandbox,)


@app.cell
def sandbox_readout(grid_sandbox, math, mo, pd, run_grid):
    _v = grid_sandbox.value
    _rows = []
    for _label, _algo, _w in [("Dijkstra", "dijkstra", 0), ("A* (w = 1)", "astar", 1.0), (f"A* (w = {_v['w']:g})", "astar", _v["w"]), ("Bidirectional Dijkstra", "bidi", 0)]:
        _r = run_grid(_v, algo=_algo, w=_w)
        _rows.append({"algorithm": _label, "cells explored": _r["explored"], f"path cost ({_v['metric']})": _r["cost"]})
    _df = pd.DataFrame(_rows)
    _best = _df.iloc[0, 2]
    _df["optimal?"] = ["✔" if (math.isclose(c, _best) or c == _best) else "✘" for c in _df.iloc[:, 2]]
    _df["vs Dijkstra"] = [f"{e / max(_df.iloc[0, 1], 1):.0%}" for e in _df["cells explored"]]
    mo.ui.table(_df, selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 9 · Quiz: check your intuition
    """)
    return


@app.cell
def quiz(mo):
    quiz_questions = {
        "q1": ("Dijkstra has reached Groningen with cost 19, but another unexplored node has cost 16. What now?",
               ["Stop: we found Groningen", "Keep exploring: a path through the cost-16 node might be shorter", "Restart with BFS"], 1),
        "q2": ("Why is BFS not enough for real roads?",
               ["It is too slow", "It treats every road as the same length", "It can't handle cycles"], 1),
        "q3": ("A* with w = 1 and an h that never overestimates returns…",
               ["the shortest path, usually exploring fewer nodes", "a fast but possibly longer path", "the same search as BFS"], 0),
        "q4": ("For travel time, h = straight-line distance / max speed. Why does A* barely help?",
               ["The heuristic overestimates", "It is a huge underestimate, so the 'hill' is almost flat", "Travel times can be negative"], 1),
        "q5": ("Two searches from both ends cover roughly what fraction of one search's area?",
               ["1/4", "1/2", "the same"], 1),
        "q6": ("Nested dissection gives the highest ranks to…",
               ["the nodes with the most edges", "small sets of nodes that split the graph roughly in half", "nodes near the source"], 1),
        "q7": ("Why does a contraction hierarchy need shortcuts?",
               ["To make the map look nicer", "So that the upward-only search doesn't miss paths through lower-ranked nodes", "To store traffic data"], 1),
        "q8": ("A traffic jam changes some travel times. Which CCH phase must rerun?",
               ["Phase 1 (ordering + shortcuts, ~1h40)", "Phase 2 (shortcut weights, ~1 s)", "Nothing"], 1),
    }
    quiz = mo.ui.dictionary({k: mo.ui.radio(opts, label=f"**{i + 1}. {q}**") for i, (k, (q, opts, _)) in enumerate(quiz_questions.items())})
    quiz_reveal = mo.ui.switch(label="Reveal answers")
    mo.vstack([quiz, quiz_reveal])
    return quiz, quiz_questions, quiz_reveal


@app.cell(hide_code=True)
def quiz_score(mo, quiz, quiz_questions, quiz_reveal):
    _answered = {k: v for k, v in quiz.value.items() if v is not None}
    _correct = {k for k, v in _answered.items() if v == quiz_questions[k][1][quiz_questions[k][2]]}
    _lines = [f"### Score: {len(_correct)} / {len(quiz_questions)}" + (" 🎯" if len(_correct) == len(quiz_questions) else "")]
    if quiz_reveal.value:
        for _i, (_k, (_q, _opts, _a)) in enumerate(quiz_questions.items()):
            _mark = "✅" if _k in _correct else ("❌" if _k in _answered else "▫️")
            _lines.append(f"{_mark} **{_i + 1}.** {_opts[_a]}")
    mo.callout(mo.md("\n\n".join(_lines)), kind="success" if len(_correct) == len(quiz_questions) else "neutral")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap

    | Idea | Rule | What it buys | Weak spot |
    |---|---|---|---|
    | Brute force | try every route | trivially correct | $\sim10^{220}$ routes |
    | BFS | explore 1 step, 2 steps, … away | fewest edges | ignores road lengths |
    | **Dijkstra** (1956) | explore the cheapest unexplored node; relax $\min(\text{cost}(v), \text{cost}(u)+w)$ | shortest path, no pre-processing | floods in all directions (≈ 7 s on North America) |
    | **A\*** | order by $\text{cost} + w\,h$ | ≈ 10× fewer nodes for *distance* | weak $h$ for *travel time* |
    | **Bidirectional** | grow from both ends, stop when $\text{top}_f+\text{top}_b \ge \mu$ | ≈ ½ the area | still hierarchy-blind |
    | Manual road classes (1990s GPS) | search local → major → highway in candidate areas | fast | no guarantee of the shortest path |
    | **Contraction hierarchy (CCH)** | nested-dissection ranks + bottom-up shortcuts + upward bidirectional Dijkstra | ≈ 200 µs queries, 35,000× faster, exact | hours of pre-processing; phase 2 reruns for traffic |

    At its core, every one of these is still Dijkstra's 20-minute invention. *"Simplicity is prerequisite for reliability."* (Edsger W. Dijkstra)
    """)
    return


if __name__ == "__main__":
    app.run()

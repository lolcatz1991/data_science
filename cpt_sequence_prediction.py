# /// script
# dependencies = [
#     "altair==6.3.0",
#     "anywidget==0.11.0",
#     "marimo",
#     "marimo-lens==0.2.4",
#     "numpy==2.5.4",
#     "pandas==3.1.1",
#     "traitlets==5.16.1",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
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
    # Sequence prediction with CPT and CPT+, explained by playing with it
    *An interactive companion to Philippe Fournier-Viger's video **"The CPT+ Algorithm for Sequence Prediction"**.*

    > Someone went to the **gym (A)**, then a **restaurant (B)**, then the **movie theater (C)**. Where will they go next?
    > You typed *"hello how are"*. What is the next word?

    **Sequence prediction**: given a set of training sequences of *symbols*, predict the **next symbol** of a new sequence.
    It is used for web-page prefetching, product recommendation, next-location and next-word prediction.

    Every method in this lesson follows the same two phases:

    1. **Training**: training sequences → build a *sequence prediction model*.
    2. **Prediction**: model + a new sequence (e.g. `A B C`) → the predicted next symbol (e.g. `D`).

    We first build the classical models (**DG, PPM, AKOM**), see where they break, then build **CPT**
    (Compact Prediction Tree) by hand, query it, make it noise tolerant, compress it (**CPT+**) and finally race all of them on data.
    """),
        mo.callout(mo.md(r"""
    **Color language used throughout**

    - <span style="color:#8c6bb1">**purple**</span> DG · <span style="color:#3b6fb6">**blue**</span> PPM order 1 · <span style="color:#56a0d3">**light blue**</span> PPM order 2 · <span style="color:#f18f01">**orange**</span> AKOM · <span style="color:#2a9d8f">**teal**</span> CPT · <span style="color:#e4572e">**red**</span> CPT+
    - In the widgets: <span style="color:#f18f01">**orange**</span> = *just added / new*, <span style="color:#2a9d8f">**teal**</span> = *matches the query*, <span style="color:#e4572e">**red**</span> = *the prediction*.
    """), kind="info"),
    ])
    return


@app.cell
def cpt_lib(alt):
    # Core algorithms used by every section (pure Python, readable on purpose)
    from collections import Counter, defaultdict
    from itertools import combinations


    def parse_seqs(text):
        return [l.split() if " " in l.strip() else list(l.strip()) for l in text.strip().splitlines() if l.strip()]


    # ---------- Dependency Graph ----------
    def train_dg(seqs, window=2):
        occ, edges = Counter(), Counter()
        for s in seqs:
            for i, a in enumerate(s):
                occ[a] += 1
                for b in set(s[i + 1 : i + 1 + window]):
                    edges[(a, b)] += 1
        return {"occ": occ, "edges": edges, "window": window}


    def predict_dg(m, ctx):
        if not ctx or m["occ"][ctx[-1]] == 0:
            return {}
        a = ctx[-1]
        return {b: c / m["occ"][a] for (x, b), c in m["edges"].items() if x == a}


    def size_dg(m):
        return len(m["occ"]) + len(m["edges"])


    # ---------- PPM order k / AKOM ----------
    def train_ppm(seqs, k):
        t, occ = defaultdict(Counter), Counter()
        for s in seqs:
            for i in range(len(s) - k + 1):
                occ[tuple(s[i : i + k])] += 1
                if i + k < len(s):
                    t[tuple(s[i : i + k])][s[i + k]] += 1
        t = dict(t)
        t["_occ"] = occ
        return t


    def predict_ppm(t, ctx, k):
        if len(ctx) < k:
            return {}
        c = t.get(tuple(ctx[-k:]))
        if not c:
            return {}
        n = t["_occ"][tuple(ctx[-k:])]
        return {b: v / n for b, v in c.items()}


    def size_ppm(t):
        return sum(1 + len(c) for key, c in t.items() if key != "_occ")


    def train_akom(seqs, K):
        return {k: train_ppm(seqs, k) for k in range(1, K + 1)}


    def predict_akom(ms, ctx):
        for k in sorted(ms, reverse=True):
            d = predict_ppm(ms[k], ctx, k)
            if d:
                return d
        return {}


    def size_akom(ms):
        return sum(size_ppm(t) for t in ms.values())


    def best(d):
        return max(sorted(d), key=lambda b: d[b]) if d else None


    # ---------- CPT ----------
    class CPT:
        def __init__(self):
            self.nodes = [{"id": 0, "sym": "root", "parent": None, "children": {}}]
            self.lookup = {}
            self.ii = defaultdict(set)
            self.seqs = {}
            self.freq = Counter()

        def insert(self, sid, seq):
            cur, new = 0, []
            for x in seq:
                ch = self.nodes[cur]["children"]
                if x not in ch:
                    nid = len(self.nodes)
                    self.nodes.append({"id": nid, "sym": x, "parent": cur, "children": {}})
                    ch[x] = nid
                    new.append(nid)
                cur = ch[x]
                self.ii[x].add(sid)
            self.lookup[sid] = cur
            self.seqs[sid] = list(seq)
            self.freq.update(set(seq))
            return new

        def recover(self, sid):
            out, n = [], self.lookup[sid]
            while n:
                out.append(self.nodes[n]["sym"])
                n = self.nodes[n]["parent"]
            return out[::-1]

        def n_nodes(self):
            return len(self.nodes) - 1

        def matching(self, q):
            q = set(q)
            if not q or any(x not in self.ii for x in q):
                return []
            return sorted(set.intersection(*(self.ii[x] for x in q)))

        def consequent(self, sid, q):
            s, need = self.recover(sid), set(q)
            for i, x in enumerate(s):
                need.discard(x)
                if not need:
                    return s[i + 1 :]
            return []

        def count_table(self, q):
            ct = Counter()
            for sid in self.matching(q):
                for x in set(self.consequent(sid, q)) - set(q):
                    ct[x] += 1
            return ct

        def predict(self, q, prefix=None, max_level=0, plus=False):
            """Returns (prediction or None, count table, list of tried subqueries)."""
            q = list(q)[-prefix:] if prefix else list(q)
            tried = [tuple(q)]
            ct = self.count_table(q)
            if ct:
                return best(ct), ct, tried
            if plus:  # CPT+: drop least-frequent symbols first, one at a time
                cur = list(q)
                for _ in range(max_level):
                    if len(cur) <= 1:
                        break
                    drop = min(cur, key=lambda x: (self.freq[x], x))
                    cur.remove(drop)
                    tried.append(tuple(cur))
                    ct = self.count_table(cur)
                    if ct:
                        return best(ct), ct, tried
                return None, Counter(), tried
            for lvl in range(1, max_level + 1):  # CPT recursive divider
                if len(q) - lvl < 1:
                    break
                agg = Counter()
                for sub in combinations(q, len(q) - lvl):
                    tried.append(sub)
                    agg.update(self.count_table(sub))
                if agg:
                    return best(agg), agg, tried
            return None, Counter(), tried


    def train_cpt(seqs, split=None):
        m = CPT()
        for i, s in enumerate(seqs):
            m.insert(f"S{i+1}", s[-split:] if split else s)
        return m


    # ---------- tree helpers (for drawing & compression) ----------
    def cpt_tree(m):
        """generic tree: list of {id,label,parent}; pointers sid->node"""
        return [{"id": n["id"], "label": n["sym"], "parent": n["parent"]} for n in m.nodes], dict(m.lookup)


    def frequent_pairs(m, minsup=0.3, top=5):
        """Adjacent pairs (as in the tree) appearing in >= minsup of the training sequences."""
        c = Counter()
        for sid in m.lookup:
            s = m.recover(sid)
            c.update(set(zip(s, s[1:])))
        n = max(len(m.lookup), 1)
        return [p for p, v in c.most_common(top) if v / n >= minsup]


    def cplus_size(m, minsup=0.3):
        """Node count after CPT+ compression: FSC (frequent pairs) then SBC."""
        for p in frequent_pairs(m, minsup):
            m = fsc(m, list(p))
        t, ptr = sbc(*cpt_tree(m))
        return len(t) - 1


    def fsc(m, pattern):
        """Frequent subsequence compression: replace chains matching `pattern` by a node X=pattern."""
        sym = "".join(pattern)
        out, lookup = CPT(), {}
        for sid in m.lookup:
            s, r, i = m.recover(sid), [], 0
            while i < len(s):
                if s[i : i + len(pattern)] == list(pattern):
                    r.append(sym); i += len(pattern)
                else:
                    r.append(s[i]); i += 1
            out.insert(sid, r)
        return out


    def sbc(tree, pointers):
        """Simple branch compression: merge a node with its only child when the node is not a sequence end."""
        nodes = {n["id"]: dict(n) for n in tree}
        kids = defaultdict(list)
        for n in tree:
            if n["parent"] is not None:
                kids[n["parent"]].append(n["id"])
        ends = set(pointers.values())
        changed = True
        while changed:
            changed = False
            for nid in list(nodes):
                if nid == 0 or nid not in nodes:
                    continue
                if len(kids[nid]) == 1 and nid not in ends:
                    c = kids[nid][0]
                    nodes[nid]["label"] += nodes[c]["label"]
                    kids[nid] = kids.pop(c, [])
                    for g in kids[nid]:
                        nodes[g]["parent"] = nid
                    if c in ends:
                        ends.discard(c); ends.add(nid)
                        pointers = {k: (nid if v == c else v) for k, v in pointers.items()}
                    del nodes[c]
                    changed = True
        return list(nodes.values()), pointers


    def layout(tree, W=520, H=260, top=24):
        kids = defaultdict(list)
        for n in tree:
            if n["parent"] is not None:
                kids[n["parent"]].append(n["id"])
        pos, leaf = {}, [0]
        depth = {}

        def walk(n, d):
            depth[n] = d
            if not kids[n]:
                x = leaf[0]; leaf[0] += 1
            else:
                xs = [walk(c, d + 1) for c in kids[n]]
                x = sum(xs) / len(xs)
            pos[n] = x
            return x

        walk(0, 0)
        nl, md = max(leaf[0], 1), max(depth.values()) or 1
        return {n: (40 + (W - 80) * (pos[n] + 0.5) / nl, top + (H - top - 20) * depth[n] / md) for n in pos}


    # ---------- synthetic data ----------
    def gen_data(n, n_routes=12, dep=3, noise=0.1, seed=0, data_seed=None):
        """Route model: start segment (identifies the route) -> shared corridor of dep-1 hubs -> destination.
        The destination is only predictable by looking back >= dep symbols. Noise inserts / swaps symbols."""
        import numpy as np
        world = np.random.default_rng(seed)
        starts = [f"p{i}" for i in range(40)]
        hubs = [f"h{i}" for i in range(6)]
        dests = [f"d{i}" for i in range(6)]
        rare = [f"x{i}" for i in range(80)]
        routes = []
        for r in range(n_routes):
            pre = [str(x) for x in world.choice(starts, int(world.integers(3, 6)), replace=False)]
            cor = [str(x) for x in world.choice(hubs, max(dep - 1, 0))]
            routes.append((pre, cor, dests[int(world.integers(0, len(dests)))]))
        rng = np.random.default_rng(seed + 1000 if data_seed is None else data_seed)
        out = []
        for _ in range(n):
            pre, cor, d = routes[int(rng.integers(0, n_routes))]
            body = list(pre) + list(cor)
            noisy = []
            for x in body:
                if rng.random() < noise:
                    # noise: half rare "typo" symbols, half confusable common symbols
                    pool = rare if rng.random() < 0.5 else starts + hubs
                    noisy.append(str(rng.choice(pool)))
                noisy.append(x)
            for k in range(len(noisy) - 1):
                if rng.random() < noise / 2:
                    noisy[k], noisy[k + 1] = noisy[k + 1], noisy[k]
            out.append(noisy + [d])
        return out


    def evaluate(train, test, prefix=4, K=3, window=2, max_level=2):
        import time
        dg, p1, ak = train_dg(train, window), train_ppm(train, 1), train_akom(train, K)
        cpt = train_cpt(train)
        models = {
            "DG": (lambda q: best(predict_dg(dg, q)), size_dg(dg)),
            "PPM1": (lambda q: best(predict_ppm(p1, q, 1)), size_ppm(p1)),
            f"AKOM": (lambda q: best(predict_akom(ak, q)), size_akom(ak)),
            "CPT": (lambda q: cpt.predict(q, prefix, 0)[0], cpt.n_nodes()),
            "CPT+": (lambda q: cpt.predict(q, prefix, max_level, plus=True)[0], cplus_size(cpt)),
        }
        rows = []
        for name, (f, size) in models.items():
            t0, hit, cov = time.perf_counter(), 0, 0
            for s in test:
                p = f(s[:-1])
                cov += p is not None
                hit += p == s[-1]
            rows.append({"model": name, "accuracy": hit / len(test), "coverage": cov / len(test),
                         "size": size, "ms": 1000 * (time.perf_counter() - t0)})
        return rows


    MODEL_COLORS = {"DG": "#8c6bb1", "PPM1": "#3b6fb6", "PPM2": "#56a0d3", "AKOM": "#f18f01", "CPT": "#2a9d8f", "CPT+": "#e4572e"}
    color_scale = alt.Scale(domain=list(MODEL_COLORS), range=list(MODEL_COLORS.values()))
    return (
        CPT,
        best,
        color_scale,
        cplus_size,
        cpt_tree,
        evaluate,
        frequent_pairs,
        fsc,
        gen_data,
        layout,
        parse_seqs,
        predict_akom,
        predict_dg,
        predict_ppm,
        sbc,
        size_akom,
        size_ppm,
        train_akom,
        train_cpt,
        train_dg,
        train_ppm,
    )


@app.cell(hide_code=True)
def s2_intro(mo):
    mo.md(r"""
    ## 1 · Classical baselines: DG, PPM and AKOM

    The video trains every baseline on the same two sequences:

    $$S_1 = \langle A, B, C, A, C, D, D\rangle \qquad S_2 = \langle C, C, A, B, C, B, C, A\rangle$$

    | model | what it stores | how it predicts after context … x |
    |---|---|---|
    | **DG** (dependency graph), window $w$ | arc $a\to b$ weighted by how often $b$ appears **within the next $w$ symbols** after $a$ | $P(b\mid a)=\dfrac{\#(a\to b)}{\#a}$, using only the last symbol |
    | **PPM order 1** | how often $b$ comes **immediately** after $a$ | $P(b\mid a)=\dfrac{\#(ab)}{\#a}$ |
    | **PPM order 2** | how often $c$ comes immediately after the pair $ab$ | $P(c\mid ab)=\dfrac{\#(abc)}{\#ab}$ |
    | **AKOM** (All-K-Order Markov) | PPM of every order $1..K$ | try order $K$; if the context was never seen, fall back to $K-1$, …, 1 |

    The video's worked numbers: DG gives $P(C\mid A)=3/4$, PPM1 gives $P(B\mid A)=2/4$ and $P(C\mid A)=1/4$,
    and PPM2 gives $P(C\mid AB)=2/2$.
    """)
    return


@app.cell
def video_seqs(mo):
    S1 = list("ABCACDD")
    S2 = list("CCABCBCA")
    video_train = [S1, S2]
    ctx_pick = mo.ui.dropdown(["A", "B", "C", "D", "AB", "AC", "BC", "CA", "BA", "BD", "AD", "ABC"], value="A", label="context (what happened so far)")
    dg_window = mo.ui.slider(1, 4, value=2, label="DG lookahead window w", show_value=True)
    akom_k = mo.ui.slider(1, 5, value=2, label="AKOM order K", show_value=True)
    mo.hstack([ctx_pick, dg_window, akom_k], justify="start", gap=2)
    return akom_k, ctx_pick, dg_window, video_train


@app.cell
def baseline_chart(
    akom_k,
    alt,
    best,
    color_scale,
    ctx_pick,
    dg_window,
    pd,
    predict_akom,
    predict_dg,
    predict_ppm,
    train_akom,
    train_dg,
    train_ppm,
    video_train,
):
    _ctx = list(ctx_pick.value)
    _dg = train_dg(video_train, dg_window.value)
    _models = {
        "DG": predict_dg(_dg, _ctx),
        "PPM1": predict_ppm(train_ppm(video_train, 1), _ctx, 1),
        "PPM2": predict_ppm(train_ppm(video_train, 2), _ctx, 2),
        "AKOM": predict_akom(train_akom(video_train, akom_k.value), _ctx),
    }
    _rows = []
    for _m, _d in _models.items():
        for _s in "ABCD":
            _rows.append({"model": _m, "next": _s, "P": _d.get(_s, 0.0),
                          "label": "★" if _d and _s == best(_d) else "",
                          "reason": "" if _d else (f"✗ needs {_m[-1]} symbols of context" if _m[:3] == "PPM" and len(_ctx) < int(_m[-1]) else "✗ context never seen")})
    _df = pd.DataFrame(_rows)
    _base = alt.Chart(_df).encode(x=alt.X("next:N", title=None), y=alt.Y("P:Q", scale=alt.Scale(domain=[0, 1.05]), title="P(next | context)"))
    _bars = _base.mark_bar().encode(color=alt.Color("model:N", scale=color_scale, legend=None), tooltip=["model", "next", alt.Tooltip("P:Q", format=".2f")])
    _txt = _base.mark_text(dy=-8, fontSize=14).encode(text="label:N")
    _none = alt.Chart(_df).transform_filter((alt.datum.reason != "") & (alt.datum.next == "A")).mark_text(color="#999", fontSize=11).encode(x=alt.value(75), y=alt.value(90), text="reason:N")
    baseline_view = (_bars + _txt + _none).properties(width=150, height=180).facet(column=alt.Column("model:N", sort=list(_models), title=f"next-symbol distribution after context {' '.join(_ctx)}"))
    baseline_view
    return


@app.cell
def dg_readout(alt, ctx_pick, dg_window, mo, pd, train_dg, video_train):
    _dg = train_dg(video_train, dg_window.value)
    _a = ctx_pick.value[-1]
    _arcs = pd.DataFrame([{"from": a, "to": b, "count": c} for (a, b), c in _dg["edges"].items()])
    _heat = alt.Chart(_arcs).mark_rect(color="#8c6bb1").encode(
        x=alt.X("to:N"), y=alt.Y("from:N"), opacity=alt.Opacity("count:Q", legend=None), tooltip=["from", "to", "count"]
    ) + alt.Chart(_arcs).mark_text(fontSize=14).encode(x="to:N", y="from:N", text="count:Q")
    mo.hstack([
        _heat.properties(width=200, height=200, title=f"DG arcs, window w={dg_window.value}"),
        mo.md(f"""
    **DG, by hand** for the last symbol **{_a}** (it occurs **{_dg['occ'][_a]}** times):

    """ + "\n".join(f"- P({b} | {_a}) = #({_a}→{b}) / #{_a} = {c} / {_dg['occ'][_a]} = **{c/_dg['occ'][_a]:.2f}**"
                      for (x, b), c in sorted(_dg["edges"].items()) if x == _a) + """

    _Counting precisely with w = 2, A→B is found only **2** times (the B right after each AB); the slide in the video says 3.
    A→C = 3 matches the video exactly. Slide w to see arcs appear: a bigger window means more (and fuzzier) dependencies._
    """),
    ], justify="start", gap=2)
    return


@app.cell
def akom_size(
    akom_k,
    alt,
    gen_data,
    mo,
    pd,
    size_akom,
    size_ppm,
    train_akom,
    train_ppm,
    video_train,
):
    _big = gen_data(400, noise=0.1, data_seed=7)
    _rows = []
    for _k in range(1, 6):
        _rows.append({"K": _k, "data": "video (2 sequences)", "nodes": size_akom(train_akom(video_train, _k))})
        _rows.append({"K": _k, "data": "400 noisy route sequences", "nodes": size_akom(train_akom(_big, _k))})
    _df = pd.DataFrame(_rows)
    _line = alt.Chart(_df).mark_line(point=True, color="#f18f01").encode(
        x="K:O", y=alt.Y("nodes:Q", title="model size (contexts + arcs)"), strokeDash="data:N", tooltip=["K", "data", "nodes"])
    _rule = alt.Chart(pd.DataFrame({"K": [akom_k.value]})).mark_rule(color="#e4572e").encode(x="K:O")
    mo.vstack([
        mo.stat(size_akom(train_akom(video_train, akom_k.value)), label=f"AKOM nodes at K={akom_k.value} (video data)",
                caption=f"PPM1 alone: {size_ppm(train_ppm(video_train, 1))}"),
        (_line + _rule).properties(width="container", height=220, title="AKOM grows fast with K: there are up to |Σ|^K contexts"),
    ])
    return


@app.cell(hide_code=True)
def s2_try(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Context **A**: PPM1 says B (2/4) while DG with w = 2 ties B and C. Move **w** to 1: DG turns into PPM1.
    2. Pick **BD** ("break it"): BD never occurs in the training data, so **PPM2 cannot predict**. AKOM falls back to order 1 and uses just D.
    3. Compare **AB** with **BA**. A shopper who buys A then B probably wants the same thing as one who buys B then A, but every
       order-sensitive model treats them as unrelated contexts. That is the *order noise* problem CPT is designed to handle.
    4. Raise **K** and watch the size counter: on real data AKOM explodes, as the video warns.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def s3_intro(mo):
    mo.md(r"""
    ## 2 · Training CPT: a tree, a lookup table and an inverted index

    Every model above **loses information**: from a PPM or DG model you cannot recover the training sequences.
    CPT (Compact Prediction Tree, 2013) is **lossless**: it keeps *all* training data, stored compactly, in three structures:

    1. **Prediction tree**: each sequence is inserted as a branch from the root. Sequences that **share a prefix share nodes** (AB is stored once for ABC, AB and ABDC).
    2. **Lookup table**: one pointer per sequence, $S_x \to$ the node where it ends. Walking **backwards** to the root gives the sequence back.
    3. **Inverted index**: one bit-vector per symbol; bit $(s, S_x)=1$ if symbol $s$ appears in $S_x$.

    The video's training sequences are ABC, AB, ABDC, BC and BD. Step through the insertion below.
    """)
    return


@app.cell
def tree_widget_cls(anywidget, traitlets):
    class TreeStepper(anywidget.AnyWidget):
        """Renders a list of CPT states: tree + lookup-table pointers + inverted index. Python computes the states."""
        _esm = r"""
    const NS = "http://www.w3.org/2000/svg";
    function render({ model, el }) {
      const W = 560, H = 330;
      const ctl = document.createElement("div"); ctl.style.margin = "4px 0";
      const row = document.createElement("div"); Object.assign(row.style, { display: "flex", gap: "18px", flexWrap: "wrap", alignItems: "flex-start" });
      const svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none" });
      const side = document.createElement("div"); side.style.fontFamily = "monospace";
      const cap = document.createElement("div"); cap.style.margin = "6px 0"; cap.style.minHeight = "2.4em";
      row.append(svg, side); el.append(ctl, row, cap);
      const mk = (tag, a, p = svg) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
      const go = d => { const n = model.get("states").length; model.set("i", Math.max(0, Math.min(n - 1, model.get("i") + d))); model.save_changes(); };
      for (const [t, d] of [["⏮ start", -1e9], ["◀ back", -1], ["next ▶", 1], ["end ⏭", 1e9]]) {
        const b = document.createElement("button"); b.textContent = t; b.style.marginRight = "6px"; b.onclick = () => go(d); ctl.appendChild(b); }
      function draw() {
        const states = model.get("states"); if (!states.length) return;
        const s = states[Math.min(model.get("i"), states.length - 1)];
        svg.innerHTML = "";
        const style = mk("style", {}); style.textContent = "@keyframes flash{0%{fill:#f18f01;r:22}100%{}} .new{animation:flash .9s ease-out}";
        const byId = {}; s.nodes.forEach(n => byId[n.id] = n);
        s.nodes.forEach(n => { if (n.parent !== null && byId[n.parent]) { const p = byId[n.parent];
          mk("line", { x1: p.x, y1: p.y, x2: n.x, y2: n.y, stroke: n.hl ? "#2a9d8f" : "#999", "stroke-width": n.hl ? 4 : 2 }); } });
        // lookup-table pointers
        const ptrs = s.pointers || [], PY = H - 18;
        ptrs.forEach((p, k) => { const n = byId[p.node]; if (!n) return;
          const x = 40 + (W - 80) * (k + 0.5) / Math.max(ptrs.length, 1);
          const col = p.new ? "#f18f01" : (p.hl ? "#2a9d8f" : "#888");
          mk("path", { d: `M${x},${PY - 10} C${x},${PY - 50} ${n.x},${n.y + 50} ${n.x},${n.y + 16}`, fill: "none", stroke: col, "stroke-dasharray": "4 3", "stroke-width": p.new ? 2.5 : 1.5, "marker-end": "" });
          mk("circle", { cx: n.x, cy: n.y + 16, r: 3, fill: col });
          mk("rect", { x: x - 18, y: PY - 10, width: 36, height: 20, rx: 4, fill: col, opacity: 0.9 });
          const t = mk("text", { x, y: PY + 5, "text-anchor": "middle", "font-size": 12, fill: "#fff", "font-weight": "bold" }); t.textContent = p.sid; });
        s.nodes.forEach(n => {
          const isRoot = n.parent === null;
          const w = Math.max(30, 11 * n.label.length + 14);
          const fill = isRoot ? "#ddd" : n.new ? "#f18f01" : n.hl ? "#2a9d8f" : "#e9e4f0";
          const r = mk("rect", { x: n.x - w / 2, y: n.y - 14, width: w, height: 28, rx: 14, fill, stroke: "#666", "stroke-width": 1 });
          if (n.new) r.setAttribute("class", "new");
          const t = mk("text", { x: n.x, y: n.y + 5, "text-anchor": "middle", "font-size": isRoot ? 11 : 14, "font-weight": "bold", fill: (n.new || n.hl) ? "#fff" : "#333" });
          t.textContent = n.label;
        });
        // inverted index
        const tb = s.table; let h = "";
        if (tb && tb.rows.length) {
          h += "<div style='font-weight:bold;margin-bottom:4px'>inverted index</div><table style='border-collapse:collapse'>";
          h += "<tr><td></td>" + tb.cols.map(c => `<td style='padding:3px 7px;color:#888'>${c}</td>`).join("") + "</tr>";
          tb.rows.forEach(r => { h += `<tr><td style='padding:3px 7px;font-weight:bold'>${r}</td>` + tb.cols.map(c => {
            const v = tb.bits[r][c] || 0, nw = (tb.new || []).includes(r + "|" + c);
            return `<td style='padding:3px 7px;text-align:center;border:1px solid #ccc;background:${nw ? "#f18f01" : v ? "rgba(42,157,143,.18)" : "transparent"};color:${nw ? "#fff" : "inherit"}'>${v}</td>`; }).join("") + "</tr>"; });
          h += "</table>";
        }
        if (s.stats) h += `<div style='margin-top:10px;font-family:sans-serif'>${s.stats}</div>`;
        side.innerHTML = h;
        cap.innerHTML = `<b>step ${model.get("i")}/${states.length - 1}</b>: ${s.caption}`;
      }
      model.on("change:i", draw); model.on("change:states", draw); draw();
    }
    export default { render };
    """
        states = traitlets.List(traitlets.Dict()).tag(sync=True)
        i = traitlets.Int(0).tag(sync=True)

    return (TreeStepper,)


@app.cell
def tree_states(CPT, cpt_tree, layout):
    def tree_state(tree, pointers, caption="", new_nodes=(), new_ptr=None, hl_nodes=(), hl_ptr=(), table=None, stats="", W=560, H=290):
        """Turn a generic tree (list of {id,label,parent}) into a drawable state for TreeStepper."""
        _pos = layout(tree, W=W, H=H)
        return {
            "nodes": [{"id": n["id"], "label": n["label"], "parent": n["parent"], "x": _pos[n["id"]][0], "y": _pos[n["id"]][1],
                       "new": n["id"] in new_nodes, "hl": n["id"] in hl_nodes} for n in tree],
            "pointers": [{"sid": s, "node": v, "new": s == new_ptr, "hl": s in hl_ptr} for s, v in pointers.items()],
            "table": table, "caption": caption, "stats": stats,
        }


    def cpt_build_states(seqs):
        """Insert sequences one by one into a CPT, recording a drawable state after each insertion."""
        _m, _states = CPT(), []
        _cols = [f"S{k+1}" for k in range(len(seqs))]
        _rows = sorted({x for s in seqs for x in s})
        _states.append(tree_state(*cpt_tree(_m), caption="Empty model: a root, an empty lookup table, an empty inverted index.",
                                  table={"rows": _rows, "cols": _cols, "bits": {r: {} for r in _rows}, "new": []},
                                  stats="nodes: 0 · symbols inserted: 0"))
        _total = 0
        for _k, _s in enumerate(seqs):
            _sid = _cols[_k]
            _new = _m.insert(_sid, _s)
            _total += len(_s)
            _bits = {r: {c: int(c in _m.ii.get(r, set())) for c in _cols} for r in _rows}
            _reused = len(_s) - len(_new)
            _cap = (f"Insert <b>{_sid} = {' '.join(_s)}</b>: " +
                    (f"{_reused} node(s) reused (shared prefix), " if _reused else "") +
                    f"{len(_new)} new node(s); pointer {_sid} → its last node; column {_sid} of the inverted index filled.")
            _states.append(tree_state(*cpt_tree(_m), caption=_cap, new_nodes=set(_new), new_ptr=_sid,
                                      hl_nodes=set(), table={"rows": _rows, "cols": _cols, "bits": _bits,
                                      "new": [f"{r}|{_sid}" for r in set(_s)]},
                                      stats=f"tree nodes: <b>{_m.n_nodes()}</b> · symbols inserted: <b>{_total}</b>"))
        return _states, _m


    cpt_train_seqs = [list(s) for s in ["ABC", "AB", "ABDC", "BC", "BD"]]
    builder_states, video_cpt = cpt_build_states(cpt_train_seqs)
    return builder_states, cpt_train_seqs, tree_state, video_cpt


@app.cell
def builder(TreeStepper, builder_states, mo):
    builder = mo.ui.anywidget(TreeStepper(states=builder_states, i=0))
    builder
    return (builder,)


@app.cell
def builder_readout(CPT, builder, cpt_train_seqs, mo):
    _i = builder.value["i"]
    _m = CPT()
    for _k, _s in enumerate(cpt_train_seqs[:_i]):
        _m.insert(f"S{_k+1}", _s)
    _sym = sum(len(s) for s in cpt_train_seqs[:_i])
    mo.vstack([
        mo.hstack([
            mo.stat(_m.n_nodes(), label="prediction-tree nodes", bordered=True),
            mo.stat(_sym, label="symbols in the training DB so far", bordered=True),
            mo.stat(f"{(_m.n_nodes() / _sym if _sym else 1):.0%}", label="tree size / DB size", caption="lower = more prefix sharing", bordered=True),
        ], justify="start"),
        mo.md("**Lossless check**: rebuild every sequence by walking each pointer back to the root: " +
              ", ".join(f"{sid} = `{''.join(_m.recover(sid))}`" for sid in _m.lookup) if _m.lookup else "_(nothing inserted yet)_"),
    ])
    return


@app.cell(hide_code=True)
def s3_try(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Step to S2 = AB: **no new node** is created, only a pointer. Prefix sharing is what makes the tree *compact*.
    2. At S4 = BC, the tree has to start a second branch from the root, because CPT only shares *prefixes*.
    3. At the end, 8 nodes hold 13 symbols, and the lossless check still reproduces every sequence exactly.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def s4_intro(mo):
    mo.md(r"""
    ## 3 · Predicting with CPT: intersect, look up, count

    To predict what comes after a query such as **AB**:

    1. **Intersect**: AND the inverted-index bit-vectors of the query symbols: $A \wedge B = 11100$, so the sequences containing *both* A and B are $S_1, S_2, S_3$.
    2. **Look up**: follow each matching sequence's pointer and read it back from the tree.
    3. **Consequent**: take what comes **after** the point where all query symbols have appeared. The order inside the query does not matter, which makes CPT tolerant to order noise.
    4. **Count table**: count each symbol in the consequents. The prediction is the most frequent one.

    The video's result: after AB the count table is **C: 2, D: 1**, so CPT **predicts C**. Build queries below by clicking symbols.
    """)
    return


@app.cell
def query_widget_cls(anywidget, traitlets):
    class QueryBuilder(anywidget.AnyWidget):
        """Click symbols to build a query; shows the bit-vector AND, matching sequences and consequents (cheap math in JS)."""
        _esm = r"""
    function render({ model, el }) {
      const bar = document.createElement("div"), view = document.createElement("div");
      view.style.fontFamily = "monospace"; el.append(bar, view);
      const btn = (t, f, bg) => { const b = document.createElement("button"); b.textContent = t; b.onclick = f;
        Object.assign(b.style, { marginRight: "6px", padding: "4px 12px", fontWeight: "bold", background: bg || "" }); bar.appendChild(b); };
      const set = q => { model.set("query", q); model.save_changes(); };
      bar.append("add to query: ");
      model.get("symbols").forEach(s => btn(s, () => { const q = model.get("query"); if (!q.includes(s)) set([...q, s]); }));
      btn("⌫", () => set(model.get("query").slice(0, -1)));
      btn("clear", () => set([]));
      const cell = (v, bg, fg) => `<td style="padding:3px 8px;border:1px solid #ccc;text-align:center;background:${bg};color:${fg || "inherit"}">${v}</td>`;
      function draw() {
        const q = model.get("query"), seqs = model.get("seqs"), bits = model.get("bits"), sids = Object.keys(seqs);
        let h = `<div style="font-size:18px;margin:8px 0">query: <b style="color:#2a9d8f">⟨ ${q.join(" ") || "…"} ⟩</b> → ?</div>`;
        if (!q.length) { view.innerHTML = h + "<i>click symbols above</i>"; return; }
        const and = sids.map(s => q.every(x => (bits[x] || {})[s]) ? 1 : 0);
        h += "<table style='border-collapse:collapse'><tr><td></td>" + sids.map(s => `<td style='padding:3px 8px;color:#888'>${s}</td>`).join("") + "</tr>";
        q.forEach(x => { h += `<tr><td style='padding:3px 8px'><b>${x}</b></td>` + sids.map(s => cell((bits[x] || {})[s] || 0, (bits[x] || {})[s] ? "rgba(42,157,143,.15)" : "transparent")).join("") + "</tr>"; });
        h += `<tr><td style='padding:3px 8px'><b>AND</b></td>` + and.map(v => cell(v, v ? "#2a9d8f" : "transparent", v ? "#fff" : "#999")).join("") + "</tr></table>";
        const ct = {};
        h += "<div style='margin-top:10px'>";
        sids.forEach((s, k) => {
          const seq = seqs[s];
          if (!and[k]) { h += `<div style="color:#aaa">${s}: ${seq.join(" ")} <i>(no match)</i></div>`; return; }
          const need = new Set(q); let cut = -1;
          seq.forEach((x, j) => { if (cut < 0) { need.delete(x); if (!need.size) cut = j; } });
          const seen = new Set();
          h += `<div>${s}: ` + seq.map((x, j) => {
            if (j <= cut) return `<span style="color:${q.includes(x) ? "#2a9d8f" : "#999"};font-weight:${q.includes(x) ? "bold" : "normal"}">${x}</span>`;
            if (!q.includes(x) && !seen.has(x)) { seen.add(x); ct[x] = (ct[x] || 0) + 1; }
            return `<span style="background:#f18f01;color:#fff;padding:0 4px;border-radius:3px">${x}</span>`;
          }).join(" ") + (cut === seq.length - 1 ? " <i style='color:#999'>(empty consequent)</i>" : "") + "</div>";
        });
        h += "</div>";
        const ks = Object.keys(ct).sort((a, b) => ct[b] - ct[a] || a.localeCompare(b));
        h += "<div style='margin-top:10px'>count table: " + (ks.length ? ks.map((x, j) => `<b style="color:${j ? "inherit" : "#e4572e"}">${x}: ${ct[x]}</b>`).join(" · ") : "<i>empty: no prediction</i>") + "</div>";
        view.innerHTML = h;
      }
      model.on("change:query", draw); draw();
    }
    export default { render };
    """
        symbols = traitlets.List(traitlets.Unicode()).tag(sync=True)
        seqs = traitlets.Dict().tag(sync=True)
        bits = traitlets.Dict().tag(sync=True)
        query = traitlets.List(traitlets.Unicode()).tag(sync=True)

    return (QueryBuilder,)


@app.cell
def query_builder(QueryBuilder, mo, video_cpt):
    def cpt_widget_data(m):
        _sids = list(m.lookup)
        return dict(symbols=sorted(m.ii), seqs={s: m.recover(s) for s in _sids},
                    bits={x: {s: int(s in m.ii[x]) for s in _sids} for x in m.ii})

    query_builder = mo.ui.anywidget(QueryBuilder(**cpt_widget_data(video_cpt), query=["A", "B"]))
    query_builder
    return (query_builder,)


@app.cell
def query_readout(alt, mo, np, pd, query_builder, video_cpt):
    _q = query_builder.value["query"]
    _pred, _ct, _ = video_cpt.predict(_q)
    _df = pd.DataFrame({"symbol": list(_ct), "count": list(_ct.values())})
    _df["role"] = np.where(_df["symbol"] == _pred, "prediction", "other")
    _chart = alt.Chart(_df).mark_bar().encode(
        x=alt.X("count:Q", axis=alt.Axis(tickMinStep=1)), y=alt.Y("symbol:N", sort="-x"),
        color=alt.Color("role:N", scale=alt.Scale(domain=["prediction", "other"], range=["#e4572e", "#2a9d8f"]), legend=None),
    ) if len(_df) else None
    mo.hstack([
        mo.vstack([
            mo.md(f"**Python's CPT** for query `{' '.join(_q) or '∅'}`"),
            mo.md(f"matching sequences: **{', '.join(video_cpt.matching(_q)) or 'none'}**"),
            mo.stat(_pred or "—", label="prediction", caption="no prediction: the count table is empty" if not _pred else f"count table {dict(_ct)}", bordered=True),
        ]),
        _chart.properties(width=300, height=120, title="count table") if _chart is not None else mo.md("_empty count table_"),
    ], justify="start", gap=3)
    return


@app.cell(hide_code=True)
def s4_try(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. **A B** reproduces the video: S1, S2 and S3 match, and the consequents are C, (nothing), D C, so C: 2, D: 1 → **C**.
    2. Build **B A**. The result is the same as for AB: CPT ignores order *within* the query. PPM2 treated AB and BA as different contexts.
    3. Build **B D**. PPM2 could not predict this one (section 1), while CPT finds S3 and S5 and predicts C.
    4. Build **A B C**. Every match ends right there, so the count table is empty and CPT cannot predict. The next section fixes that.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def s5_intro(mo):
    mo.md(r"""
    ## 4 · Noise reduction (recursive divider) and sequence splitting

    A query can contain a symbol that never appears in the training data, or so many symbols that no training sequence contains all of them.
    Then the intersection is empty and CPT **cannot predict**. Removing the **noise** helps.

    - **CPT, recursive divider**: if the query fails, try every subquery with **1** symbol removed, then every subquery with **2** removed, and so on up to *max level*.
      For ABCD that means BCD, ACD, ABD and ABC, then CD, BD, BC, AD, AC and AB, …
    - **CPT+, improved noise reduction**: remove the **least frequent** symbols first. Rare symbols are the most likely to be noise, and far fewer subqueries are tried.

    Coverage is the share of queries for which the model produces *some* prediction. Raising the max level pushes coverage towards 100%.
    """)
    return


@app.cell
def noise_controls(mo):
    noise_query = mo.ui.text(value="A E B C", label="query (space-separated; E never appears in training)")
    max_level = mo.ui.slider(0, 3, value=2, label="max level (symbols removed)", show_value=True)
    nr_method = mo.ui.radio(["CPT (all subsets)", "CPT+ (least frequent first)"], value="CPT (all subsets)", inline=True)
    mo.hstack([noise_query, max_level, nr_method], justify="start", gap=2)
    return max_level, noise_query, nr_method


@app.cell
def noise_trace(max_level, mo, noise_query, nr_method, pd, video_cpt):
    _q = noise_query.value.split()
    _plus = nr_method.value.startswith("CPT+")
    _pred, _ct, _tried = video_cpt.predict(_q, max_level=max_level.value, plus=_plus)
    _rows = []
    for _sub in _tried:
        _c = video_cpt.count_table(_sub)
        _rows.append({"level": len(_q) - len(_sub), "subquery": " ".join(_sub),
                      "matching": ", ".join(video_cpt.matching(_sub)) or "—",
                      "count table": ", ".join(f"{k}:{v}" for k, v in sorted(_c.items())) or "∅"})
    mo.vstack([
        mo.md(f"**{len(_tried)} subqueries tried.** Symbol frequencies (number of sequences containing it): " +
              ", ".join(f"{x}={video_cpt.freq[x]}" for x in _q)),
        mo.ui.table(pd.DataFrame(_rows), selection=None, pagination=False),
        mo.stat(_pred or "—", label="prediction", bordered=True,
                caption=(f"aggregated count table {dict(_ct)}" if _pred else "still no prediction: raise the max level")),
    ])
    return


@app.cell
def coverage_chart(alt, color_scale, gen_data, pd, train_cpt):
    _train = gen_data(300, noise=0.25, data_seed=11)
    _test = gen_data(150, noise=0.25, data_seed=12)
    _m = train_cpt(_train)
    _rows = []
    for _lvl in range(0, 4):
        for _name, _plus in [("CPT", False), ("CPT+", True)]:
            _hit = _cov = 0
            for _s in _test:
                _p = _m.predict(_s[:-1], prefix=6, max_level=_lvl, plus=_plus)[0]
                _cov += _p is not None
                _hit += _p == _s[-1]
            _rows += [{"max level": _lvl, "model": _name, "metric": "coverage", "value": _cov / len(_test)},
                      {"max level": _lvl, "model": _name, "metric": "accuracy", "value": _hit / len(_test)}]
    alt.Chart(pd.DataFrame(_rows)).mark_line(point=True).encode(
        x="max level:O", y=alt.Y("value:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
        color=alt.Color("model:N", scale=color_scale), strokeDash="metric:N", tooltip=["model", "metric", "max level", alt.Tooltip("value:Q", format=".1%")],
    ).properties(width="container", height=240, title="Noisy routes (25% noise), prefix 6: removing symbols raises coverage (and accuracy)")
    return


@app.cell
def split_controls(mo):
    split_len = mo.ui.slider(2, 14, value=14, label="split length (keep only the last N symbols of each training sequence)", show_value=True)
    split_len
    return (split_len,)


@app.cell
def split_chart(alt, gen_data, mo, np, pd, split_len, train_cpt):
    _train = gen_data(300, noise=0.15, data_seed=21)
    _test = gen_data(150, noise=0.15, data_seed=22)
    _rows = []
    for _L in range(2, 15):
        _m = train_cpt(_train, split=_L)
        _acc = np.mean([_m.predict(_s[:-1], prefix=4, max_level=2, plus=True)[0] == _s[-1] for _s in _test])
        _rows.append({"split length": _L, "tree nodes": _m.n_nodes(), "accuracy": _acc})
    _df = pd.DataFrame(_rows)
    _cur = _df[_df["split length"] == split_len.value].iloc[0]
    _b = alt.Chart(_df).encode(x="split length:O")
    _nodes = _b.mark_bar(color="#2a9d8f", opacity=0.6).encode(y=alt.Y("tree nodes:Q"))
    _acc = _b.mark_line(point=True, color="#e4572e").encode(y=alt.Y("accuracy:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])))
    _sel = alt.Chart(pd.DataFrame({"split length": [split_len.value]})).mark_rule(color="#333", strokeDash=[4, 3]).encode(x="split length:O")
    mo.vstack([
        mo.hstack([mo.stat(int(_cur["tree nodes"]), label="tree nodes", bordered=True),
                   mo.stat(f"{_cur['accuracy']:.0%}", label="CPT+ accuracy", bordered=True)], justify="start"),
        alt.layer(_nodes + _sel, _acc).resolve_scale(y="independent").properties(width="container", height=240,
            title="Sequence splitting: a much smaller tree for about the same accuracy"),
    ])
    return


@app.cell(hide_code=True)
def s5_try(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. With the query **A E B C** and max level 0 there is no prediction, because E was never seen. At level 1 the subquery A B C still fails, and level 2 recovers the AB → C pattern.
    2. Switch to **CPT+**. E has frequency 0, so it is removed first and far fewer subqueries are tried. On the route data in the chart,
       CPT+ explores a single "least frequent first" path, so its coverage rises a little more slowly than exhaustive CPT, but it tries only a handful of subqueries instead of $\binom{6}{k}$.
    3. Drag the **split length** down. The tree shrinks a lot, while accuracy only collapses once you cut away the symbols that carry the dependency (the video saw ~4× fewer nodes at 5 vs 40).
    """), kind="warn")
    return


@app.cell(hide_code=True)
def s6_intro(mo):
    mo.md(r"""
    ## 5 · CPT+ compression: frequent subsequences and simple branches

    CPT+ (PAKDD 2015) makes the tree smaller **without changing its predictions**:

    - **Frequent Subsequence Compression (FSC)**: a subsequence that appears in many training sequences (here **AB**) is replaced by a single node $X = AB$.
    - **Simple Branch Compression (SBC)**: a *simple branch* is a chain in which every node has exactly one child (e.g. B→C, or E→X→A in the video). The chain becomes one node (**BC**, **EXA**).
      Here a chain is only merged when the node above is not the end of a training sequence, so every lookup pointer still lands on a node.

    Toggle the two compressions and compare the trees.
    """)
    return


@app.cell
def compress_controls(mo):
    use_fsc = mo.ui.switch(value=True, label="Frequent Subsequence Compression")
    use_sbc = mo.ui.switch(value=True, label="Simple Branch Compression")
    fsc_minsup = mo.ui.slider(0.2, 1.0, step=0.1, value=0.6, label="FSC min support (share of sequences)", show_value=True)
    mo.hstack([use_fsc, use_sbc, fsc_minsup], justify="start", gap=2)
    return fsc_minsup, use_fsc, use_sbc


@app.cell
def compress_view(
    TreeStepper,
    cpt_tree,
    frequent_pairs,
    fsc,
    fsc_minsup,
    mo,
    sbc,
    tree_state,
    use_fsc,
    use_sbc,
    video_cpt,
):
    def compress(m, fsc_on, sbc_on, minsup):
        _pats = frequent_pairs(m, minsup) if fsc_on else []
        for _p in _pats:
            m = fsc(m, list(_p))
        _tree, _ptr = cpt_tree(m)
        if sbc_on:
            _tree, _ptr = sbc(_tree, _ptr)
        return _tree, _ptr, _pats

    _raw = cpt_tree(video_cpt)
    _tree, _ptr, _pats = compress(video_cpt, use_fsc.value, use_sbc.value, fsc_minsup.value)
    _merged = {n["id"] for n in _tree if len(n["label"]) > 1 and n["parent"] is not None}
    _states = [
        tree_state(*_raw, caption=f"Plain CPT tree: <b>{len(_raw[0]) - 1}</b> nodes.", stats=f"nodes: <b>{len(_raw[0]) - 1}</b>"),
        tree_state(_tree, _ptr, new_nodes=_merged,
                   caption=f"CPT+ tree: <b>{len(_tree) - 1}</b> nodes. Orange = compressed nodes. "
                           + (f"FSC patterns: {', '.join(''.join(p) for p in _pats)}. " if _pats else "No FSC pattern. ")
                           + ("SBC on." if use_sbc.value else "SBC off."),
                   stats=f"nodes: <b>{len(_tree) - 1}</b> (was {len(_raw[0]) - 1})"),
    ]
    compress_widget = mo.ui.anywidget(TreeStepper(states=_states, i=1))
    compress_widget
    return (compress,)


@app.cell
def compress_sizes(alt, compress, fsc_minsup, gen_data, pd, train_cpt):
    _rows = []
    for _noise in [0.0, 0.1, 0.2, 0.3]:
        _m = train_cpt(gen_data(300, noise=_noise, data_seed=31))
        for _lab, _f, _s in [("CPT", False, False), ("+ FSC", True, False), ("+ SBC", False, True), ("CPT+ (FSC+SBC)", True, True)]:
            _rows.append({"noise": _noise, "variant": _lab, "nodes": len(compress(_m, _f, _s, fsc_minsup.value)[0]) - 1})
    alt.Chart(pd.DataFrame(_rows)).mark_bar().encode(
        x=alt.X("variant:N", sort=None, title=None), y="nodes:Q",
        color=alt.Color("variant:N", sort=None, scale=alt.Scale(range=["#2a9d8f", "#9fd3cb", "#f4b183", "#e4572e"]), legend=None),
        column=alt.Column("noise:O", title="noise level of the route data"), tooltip=["variant", "nodes"],
    ).properties(width=130, height=200, title="Tree size on 300 route sequences")
    return


@app.cell(hide_code=True)
def s6_try(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Toggle **FSC** only: A→B (shared by ABC, AB and ABDC) becomes one node **AB**, matching the video's X node.
    2. Toggle **SBC** only: A has a single child, so A→B is merged into **AB** anyway, and the D→C tail becomes **DC**. The B under the root keeps its two children C and D, so it is not a simple branch.
    3. Prediction is unchanged: recovering a sequence still walks the same symbols, just packed into fewer nodes.
       The chart shows the savings are largest on clean data, where long chains are shared.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def s7_intro(mo):
    mo.md(r"""
    ## 6 · Experiment: race all models on synthetic route data

    The paper races the models on real datasets (BMS, FIFA, Kosarak, SIGN, Bible). Here we use a seeded **route model**:
    each trip has a *start segment* (which identifies the route), then a shared **corridor** of $d-1$ hub stops, then a **destination**.
    The destination depends on what happened **$d$ steps back**, so a model has to look past the corridor to predict it. Noise inserts stray symbols (rare "typos" or common stops) and swaps neighbouring symbols.

    The two measures from the video, for each test sequence (predict its last symbol from the rest):

    $$\text{accuracy}=\frac{\#\text{correct predictions}}{\#\text{test sequences}}\qquad \text{coverage}=\frac{\#\text{sequences with a prediction}}{\#\text{test sequences}}$$
    """)
    return


@app.cell
def exp_controls(mo):
    exp_noise = mo.ui.slider(0.0, 0.4, step=0.05, value=0.15, label="noise", show_value=True)
    exp_dep = mo.ui.slider(1, 5, value=3, label="dependency length d", show_value=True)
    exp_n = mo.ui.slider(50, 800, step=50, value=400, label="# training sequences", show_value=True)
    exp_prefix = mo.ui.slider(1, 10, value=4, label="CPT prefix size", show_value=True)
    exp_k = mo.ui.slider(1, 5, value=2, label="AKOM K", show_value=True)
    exp_level = mo.ui.slider(0, 3, value=2, label="CPT+ max level", show_value=True)
    mo.vstack([mo.hstack([exp_noise, exp_dep, exp_n], justify="start"), mo.hstack([exp_prefix, exp_k, exp_level], justify="start")])
    return exp_dep, exp_k, exp_level, exp_n, exp_noise, exp_prefix


@app.cell
def experiment(
    alt,
    color_scale,
    evaluate,
    exp_dep,
    exp_k,
    exp_level,
    exp_n,
    exp_noise,
    exp_prefix,
    gen_data,
    mo,
    pd,
):
    exp_train = gen_data(exp_n.value, dep=exp_dep.value, noise=exp_noise.value, seed=3, data_seed=100)
    exp_test = gen_data(200, dep=exp_dep.value, noise=exp_noise.value, seed=3, data_seed=200)
    exp_results = pd.DataFrame(evaluate(exp_train, exp_test, prefix=exp_prefix.value, K=exp_k.value, window=2, max_level=exp_level.value))
    _long = exp_results.melt(id_vars="model", value_vars=["accuracy", "coverage"], var_name="metric")
    _perf = alt.Chart(_long).mark_bar().encode(
        x=alt.X("model:N", sort=list(exp_results.model), title=None), y=alt.Y("value:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("model:N", scale=color_scale, legend=None), opacity=alt.Opacity("metric:N", scale=alt.Scale(range=[1, 0.4])),
        xOffset="metric:N", tooltip=["model", "metric", alt.Tooltip("value:Q", format=".1%")],
    ).properties(width=330, height=230, title="accuracy (solid) and coverage (faded)")
    _size = alt.Chart(exp_results).mark_bar().encode(
        x=alt.X("model:N", sort=list(exp_results.model), title=None), y=alt.Y("size:Q", title="model size (nodes)"),
        color=alt.Color("model:N", scale=color_scale, legend=None), tooltip=["model", "size", alt.Tooltip("ms:Q", format=".1f", title="test time (ms)")],
    ).properties(width=250, height=230, title="model size")
    mo.vstack([
        mo.md(f"Example training sequence: `{' '.join(exp_train[0])}`"),
        alt.hconcat(_perf, _size),
        mo.ui.table(exp_results.round(3), selection=None, pagination=False),
    ])
    return exp_test, exp_train


@app.cell
def prefix_sweep(
    alt,
    color_scale,
    exp_level,
    exp_prefix,
    exp_test,
    exp_train,
    np,
    pd,
    train_cpt,
):
    _m = train_cpt(exp_train)
    _rows = []
    for _p in range(1, 11):
        for _name, _lvl, _plus in [("CPT", 0, False), ("CPT+", exp_level.value, True)]:
            _acc = np.mean([_m.predict(_s[:-1], prefix=_p, max_level=_lvl, plus=_plus)[0] == _s[-1] for _s in exp_test])
            _rows.append({"prefix size": _p, "model": _name, "accuracy": _acc})
    _line = alt.Chart(pd.DataFrame(_rows)).mark_line(point=True).encode(
        x="prefix size:O", y=alt.Y("accuracy:Q", axis=alt.Axis(format="%")), color=alt.Color("model:N", scale=color_scale),
        tooltip=["model", "prefix size", alt.Tooltip("accuracy:Q", format=".1%")])
    _rule = alt.Chart(pd.DataFrame({"prefix size": [exp_prefix.value]})).mark_rule(strokeDash=[4, 3], color="#333").encode(x="prefix size:O")
    (_line + _rule).properties(width="container", height=240, title="How many past symbols should CPT use? (cf. the video's FIFA chart)")
    return


@app.cell(hide_code=True)
def s7_try(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. With noise 0 and d = 3: AKOM with K ≥ 3 and CPT are both near-perfect, while DG and PPM1 only see the corridor and guess. Then set **K = 2**: AKOM falls apart.
    2. Raise the **noise**. Strict-order AKOM decays, while CPT+ (order-free matching plus noise reduction) holds up better, at the cost of slower prediction.
    3. The **prefix sweep** has a peak: too few symbols miss the dependency, and too many add noise that empties the intersection (plain CPT). As in the video's FIFA chart, there is a best prefix size.
    4. Look at **model size**: AKOM grows quickly with K. Raw CPT is big too, but CPT+ compression shrinks it.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def s8_intro(mo):
    mo.md(r"""
    ## 7 · Sandbox: bring your own sequences

    One training sequence per line. Symbols are separated by spaces (`gym restaurant cinema`), or a line without spaces is read as single characters (`ABDC`).
    """)
    return


@app.cell
def sandbox_controls(mo):
    sb_train = mo.ui.text_area(value="ABC\nAB\nABDC\nBC\nBD", label="training sequences", rows=7, full_width=True)
    sb_query = mo.ui.text(value="A B", label="query (space-separated)")
    mo.hstack([sb_train, sb_query], widths=[2, 1])
    return sb_query, sb_train


@app.cell
def sandbox(
    best,
    cplus_size,
    mo,
    parse_seqs,
    pd,
    predict_akom,
    predict_dg,
    predict_ppm,
    sb_query,
    sb_train,
    train_akom,
    train_cpt,
    train_dg,
    train_ppm,
):
    _seqs = parse_seqs(sb_train.value)
    _q = sb_query.value.split()
    _cpt = train_cpt(_seqs)
    _ak = train_akom(_seqs, 3)
    _out = {
        "DG (w=2)": predict_dg(train_dg(_seqs, 2), _q),
        "PPM1": predict_ppm(train_ppm(_seqs, 1), _q, 1),
        "PPM2": predict_ppm(train_ppm(_seqs, 2), _q, 2),
        "AKOM (K=3)": predict_akom(_ak, _q),
    }
    _rows = [{"model": k, "prediction": best(v) or "—", "scores": ", ".join(f"{s}:{p:.2f}" for s, p in sorted(v.items(), key=lambda t: -t[1]))} for k, v in _out.items()]
    for _name, _lvl, _plus in [("CPT", 0, False), ("CPT (+recursive divider, level 2)", 2, False), ("CPT+ noise reduction (level 2)", 2, True)]:
        _p, _ct, _ = _cpt.predict(_q, max_level=_lvl, plus=_plus)
        _rows.append({"model": _name, "prediction": _p or "—", "scores": ", ".join(f"{k}:{v}" for k, v in _ct.most_common())})
    mo.vstack([
        mo.md(f"{len(_seqs)} sequences, {sum(map(len, _seqs))} symbols → CPT tree with **{_cpt.n_nodes()}** nodes (CPT+ compressed: **{cplus_size(_cpt, 0.3)}**)"),
        mo.ui.table(pd.DataFrame(_rows), selection=None, pagination=False),
    ])
    return


@app.cell(hide_code=True)
def quiz_intro(mo):
    mo.md(r"""
    ## 8 · Quiz: predict, then reveal

    Work each answer out on paper first. Q1 uses the video's $S_1, S_2$; Q2–Q3 use the CPT training set ABC, AB, ABDC, BC, BD.
    """)
    return


@app.cell
def quiz(mo):
    quiz = mo.ui.dictionary({
        "q1": mo.ui.dropdown(["1/4", "2/4", "2/3", "3/4"], label="Q1. PPM order 1: P(B | A) = ?"),
        "q2": mo.ui.dropdown(["A", "B", "C", "D", "no prediction"], label="Q2. CPT prediction after query ⟨B⟩?"),
        "q3": mo.ui.dropdown(["A", "B", "C", "D", "no prediction"], label="Q3. CPT (no noise reduction) after query ⟨C A⟩?"),
        "q4": mo.ui.dropdown(["DG", "PPM1", "PPM2", "AKOM"], label="Q4. Which model cannot predict after context BD (video data)?"),
        "q5": mo.ui.dropdown(["it loses rare patterns", "it is lossless", "it is order-sensitive", "it needs a lookahead window"], label="Q5. CPT's key property vs PPM/DG?"),
    })
    reveal = mo.ui.switch(label="reveal answers")
    mo.vstack([quiz, reveal])
    return quiz, reveal


@app.cell
def quiz_check(
    mo,
    predict_akom,
    predict_dg,
    predict_ppm,
    quiz,
    reveal,
    train_akom,
    train_dg,
    train_ppm,
    video_cpt,
    video_train,
):
    _frac = {0.25: "1/4", 0.5: "2/4", 2 / 3: "2/3", 0.75: "3/4"}
    _p = predict_ppm(train_ppm(video_train, 1), ["A"], 1)["B"]
    _truth = {
        "q1": min(_frac.items(), key=lambda kv: abs(kv[0] - _p))[1],
        "q2": video_cpt.predict(["B"])[0] or "no prediction",
        "q3": video_cpt.predict(["C", "A"])[0] or "no prediction",
        "q4": [m for m, d in {"DG": predict_dg(train_dg(video_train, 2), ["B", "D"]), "PPM1": predict_ppm(train_ppm(video_train, 1), ["B", "D"], 1),
                              "PPM2": predict_ppm(train_ppm(video_train, 2), ["B", "D"], 2), "AKOM": predict_akom(train_akom(video_train, 2), ["B", "D"])}.items() if not d][0],
        "q5": "it is lossless",
    }
    _why = {
        "q1": "#(AB) = 2 and #A = 4 (the final A of S2 still counts as an occurrence), so 2/4.",
        "q2": "All 5 sequences contain B. Consequents: C, ∅, D C, C, D → C:3, D:2 → C.",
        "q3": "S1 and S3 contain A and C, but C is the last symbol in both, so the consequents are empty → no prediction. A noise-reduction level would rescue it.",
        "q4": "BD never appears contiguously, so PPM2 has no such context. AKOM falls back to order 1.",
        "q5": "Every training sequence can be recovered from the tree via the lookup table.",
    }
    _score = sum(quiz.value[k] == v for k, v in _truth.items())
    mo.stop(not reveal.value, mo.md(f"_Answered {sum(v is not None for v in quiz.value.values())}/5. Flip **reveal answers** when ready._"))
    mo.vstack([
        mo.md("\n".join(f"- **{k.upper()}** {'✅' if quiz.value[k] == v else '❌'} answer: **{v}**: {_why[k]}" for k, v in _truth.items())),
        mo.stat(f"{_score}/5", label="score", bordered=True),
    ])
    return


@app.cell(hide_code=True)
def recap(mo):
    mo.md(r"""
    ## Recap

    | model | lossless? | noise tolerant? | size | prediction speed | uses how much context |
    |---|---|---|---|---|---|
    | **DG** | no | a little (window) | small | very fast | last symbol |
    | **PPM order k** | no | no (strict order) | small–medium | very fast | exactly k symbols, fails on unseen contexts |
    | **AKOM** | no | no | **exponential** in K | fast | up to K, with fallback |
    | **CPT** | **yes** | yes (order-free matching, recursive divider) | large but prefix-shared | slower (bit-vector AND + counting) | any prefix size |
    | **CPT+** | yes | yes (removes least-frequent symbols first) | **much smaller** (FSC + SBC, splitting) | slower than PPM | any prefix size |

    **Take-aways**: CPT keeps *all* the training data (tree + lookup table + inverted index) and predicts by intersecting bit-vectors and counting consequents.
    CPT+ adds compression and smarter noise reduction. The trade-off is accuracy against speed and size, and the best model depends on the data:
    when the order really matters (e.g. words in the Bible), AKOM can still win.
    """)
    return


@app.cell(hide_code=True)
def done_toast(mo):
    mo.status.toast("CPT / CPT+ lesson ready", "Start with the CPT builder in section 2.")
    return


@app.cell(hide_code=True)
def _():
    # marimo-lens: agent-managed Lens cell
    import marimo as _mo
    from marimo_lens import Lens as _Lens

    _lens = _Lens()
    _mo.output.append(_lens)
    return


if __name__ == "__main__":
    app.run()

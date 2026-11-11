# /// script
# dependencies = [
#     "altair==6.3.0",
#     "anywidget==0.11.0",
#     "marimo>=0.25.2",
#     "numpy==2.5.4",
#     "pandas==3.1.1",
#     "traitlets==5.16.1",
# ]
# ///

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
    # PrefixSpan: mining sequential patterns by growing prefixes
    *An interactive companion to Philippe Fournier-Viger's video **"The PrefixSpan algorithm"**.*

    > Customers buy things over time: first *apple and bread*, then *cake*, then *apple* again.
    > **Which sub-sequences of purchases show up for many customers?**

    PrefixSpan answers that by **growing patterns one item at a time, depth-first**, and by
    only ever looking at a shrinking **projected database**, so it never wastes time on patterns
    that don't exist in the data. Every section below has something to step through, click or slide.
    """),
        mo.callout(mo.md(r"""
    **Colour language used throughout**

    - <span style="color:#2a9d55">**green**</span>: frequent (support ≥ minsup), a pattern we output
    - <span style="color:#999">**grey**</span> / <span style="color:#d1495b">**red ✕**</span>: infrequent, pruned
    - <span style="color:#f18f01">**orange**</span>: the current prefix / the step we are looking at
    - <span style="color:#3b6fb6">**blue**</span>: a pointer into the original database (pseudo-projection)
    - <span style="opacity:.45;text-decoration:line-through">struck-out grey text</span>: the part of a sequence removed by projection
    """), kind="neutral"),
    ])
    return


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    ## 1 · Sequences, itemsets, support

    A **sequence database** is a list of sequences. Each sequence is an *ordered* list of
    **itemsets**; items inside one itemset happen *at the same time* (bought together), and
    itemsets later in the sequence happen *after*.

    The video writes a sequence like ⟨(a,b),(c),(a)⟩: *apple and bread together, then cake, then apple*.
    We write it compactly as **(a,b)(c)(a)**.

    Two ways a pattern can grow, and the video stresses not to confuse them:

    | notation | meaning | name |
    |---|---|---|
    | **(a,b)** | a and b *together* in the same itemset | itemset-extension (the video's "_b") |
    | **(a)(b)** | a, and *later* b | sequence-extension |

    A pattern is *contained* in a sequence if its itemsets can be matched, in order, to itemsets
    of the sequence that are supersets of them. The **support** of a pattern is the number of
    sequences that contain it. With **minsup = 3**, we want every pattern contained in ≥ 3 sequences.

    **Why PrefixSpan?** Earlier algorithms such as **GSP** generate candidate patterns by *joining*
    smaller patterns without looking at the data, so they may waste time counting patterns that
    appear nowhere. PrefixSpan only extends a pattern with items it actually *sees* in the
    (projected) database.
    """)
    return


@app.cell
def example_db(mo, pd):
    # The video's example database (minsup = 3)
    DB = [
        [("a", "b"), ("c",), ("a",)],
        [("a", "b"), ("b",), ("c",)],
        [("b",), ("c",), ("d",)],
        [("b",), ("a", "b"), ("c",)],
    ]
    MINSUP = 3


    def fmt_itemset(itemset):
        return "(" + ",".join(itemset) + ")"


    def fmt_seq(seq):
        return "".join(fmt_itemset(_s) for _s in seq) if seq else "⟨⟩"


    mo.ui.table(
        pd.DataFrame({"SID": [f"S{_i + 1}" for _i in range(len(DB))],
                      "sequence": [fmt_seq(_s) for _s in DB],
                      "shopping story": ["apple+bread → cake → apple", "apple+bread → bread → cake",
                                         "bread → cake → dates", "bread → apple+bread → cake"]}),
        selection=None, label="Sequence database (minsup = 3)",
    )
    return DB, MINSUP, fmt_itemset, fmt_seq


@app.cell(hide_code=True)
def algo_md(mo):
    mo.md(r"""
    ## 2 · The algorithm

    The video's simple version, as recursive pseudocode:

    ```
    PrefixSpan(prefix s, projected DB D|s, minsup):
        scan D|s once → support of every item that can extend s
        for each item x with support ≥ minsup:
            output s' = s extended by x
            build projected DB D|s'      # keep only what comes *after* the first match
            PrefixSpan(s', D|s', minsup)
    ```

    **Projection.** For a prefix, each sequence keeps only the part **after the first occurrence**
    of the prefix; sequences that don't contain the prefix are dropped. If the match ended in the
    middle of an itemset, the rest of that itemset stays as a *partial itemset*, written **(_b)**.

    **Counting extensions** of a prefix whose last itemset is L:

    - **itemset-extension** x (x sorts after L's items): x appears in the partial itemset `(_x)`,
      *or* in a later itemset that contains all of L plus x.
    - **sequence-extension** x: x appears in any later, complete itemset.

    The implementation below works with **pointers** `(sequence, itemset, item)` into the original
    database (the pseudo-projection of Section 4), and records a full **trace** of the depth-first search.
    """)
    return


@app.cell
def prefixspan_impl(DB, MINSUP, fmt_itemset, mo):
    def project_pointers(db, pattern):
        """Return {sid: (k, j)} = position of the LAST item of `pattern` at its first (earliest) match."""
        if not pattern:
            return {_sid: (-1, -1) for _sid in range(len(db))}
        _ptrs = {}
        for _sid, _seq in enumerate(db):
            _k = -1
            _ok = True
            for _itemset in pattern:
                _k += 1
                while _k < len(_seq) and not set(_itemset) <= set(_seq[_k]):
                    _k += 1
                if _k >= len(_seq):
                    _ok = False
                    break
            if _ok:
                _ptrs[_sid] = (_k, _seq[_k].index(pattern[-1][-1]))
        return _ptrs


    def count_extensions(db, pattern, ptrs):
        """Scan the projected DB once. Returns two dicts item -> set(sid): itemset- and sequence-extensions."""
        _iext, _sext = {}, {}
        _last = set(pattern[-1]) if pattern else set()
        _maxlast = max(pattern[-1]) if pattern else None
        for _sid, (_k, _j) in ptrs.items():
            _seq = db[_sid]
            if pattern:
                for _x in _seq[_k][_j + 1:]:                     # partial itemset (_x)
                    _iext.setdefault(_x, set()).add(_sid)
                for _m in range(_k + 1, len(_seq)):              # later itemsets containing L ∪ {x}
                    if _last <= set(_seq[_m]):
                        for _x in _seq[_m]:
                            if _x > _maxlast:
                                _iext.setdefault(_x, set()).add(_sid)
            for _m in range(_k + 1, len(_seq)):
                for _x in _seq[_m]:
                    _sext.setdefault(_x, set()).add(_sid)
        return _iext, _sext


    def extend(pattern, item, kind):
        if kind == "i":
            return pattern[:-1] + [tuple(sorted(pattern[-1] + (item,)))]
        return pattern + [(item,)]


    def prefixspan(db, minsup, max_len=None, trace=False, max_patterns=None):
        """PrefixSpan with pseudo-projection. Returns (patterns, stats, trace_nodes).
        patterns: list of (pattern, support). stats: counters for the performance section."""
        _out, _nodes = [], []
        _stats = {"projections": 0, "extensions_counted": 0, "projected_items": 0, "blind_candidates": 0}
        _alphabet = sorted({_x for _s in db for _I in _s for _x in _I})

        def _rec(_pattern, _ptrs, _parent):
            if max_patterns and len(_out) >= max_patterns:
                return
            _iext, _sext = count_extensions(db, _pattern, _ptrs)
            _stats["extensions_counted"] += len(_iext) + len(_sext)
            # a GSP-style generator that never looks at the data would try every item both ways
            _stats["blind_candidates"] += len(_alphabet) * (2 if _pattern else 1)
            _cands = sorted([(_x, "i", len(_v)) for _x, _v in _iext.items()]) + \
                     sorted([(_x, "s", len(_v)) for _x, _v in _sext.items()])
            _me = None
            if trace:
                _me = len(_nodes)
                _nodes.append({"id": _me, "parent": _parent, "pattern": [list(_I) for _I in _pattern],
                               "ptrs": {int(_k): list(_v) for _k, _v in _ptrs.items()},
                               "counts": [{"item": _x, "kind": _t, "sup": _n, "freq": _n >= minsup}
                                          for _x, _t, _n in _cands],
                               "children": []})
            if max_len is not None and sum(len(_I) for _I in _pattern) >= max_len:
                return
            for _x, _t, _n in _cands:
                if _n < minsup:
                    continue
                _new = extend(_pattern, _x, _t)
                _out.append((_new, _n))
                _p2 = project_pointers(db, _new)
                _stats["projections"] += 1
                _stats["projected_items"] += sum(
                    len(db[_s][_k]) - _j - 1 + sum(len(_I) for _I in db[_s][_k + 1:]) for _s, (_k, _j) in _p2.items())
                if trace:
                    _nodes[_me]["children"].append(len(_nodes))
                _rec(_new, _p2, _me)

        _rec([], project_pointers(db, []), None)
        return _out, _stats, _nodes


    def pattern_str(pattern):
        return "".join(fmt_itemset(_I) for _I in pattern)


    patterns, _stats, trace_nodes = prefixspan(DB, MINSUP, trace=True)
    _found = {pattern_str(_p): _s for _p, _s in patterns}
    assert {_c["item"]: _c["sup"] for _c in trace_nodes[0]["counts"]} == {"a": 3, "b": 4, "c": 4, "d": 1}
    assert _found == {"(a)": 3, "(b)": 4, "(c)": 4, "(a)(c)": 3, "(a,b)": 3, "(a,b)(c)": 3, "(b)(c)": 4}, _found
    mo.callout(mo.md(
        "✅ Reproduces the video: item supports **a:3 b:4 c:4 d:1**, and the **7** sequential patterns "
        + ", ".join(f"**{_k}**:{_v}" for _k, _v in _found.items())), kind="success")
    return extend, pattern_str, prefixspan, project_pointers, trace_nodes


@app.cell(hide_code=True)
def stepper_md(mo):
    mo.md(r"""
    ## 3 · Watch the depth-first search

    Step through PrefixSpan on the video's database. **Left:** the search tree, exactly like the
    video's picture: the empty sequence at the root, each child is the prefix extended by one item.
    Green nodes are frequent and get extended further; grey ✕ nodes are counted once in the
    projected database and then **pruned**: nothing that starts with them can be frequent.
    The **orange** node is where the algorithm is right now.

    **Right:** that node's projected database (struck-out grey = removed by projection,
    `_` marks a partial itemset) and the count table from a single scan, split into
    **itemset-extensions** (together, `(…,x)`) and **sequence-extensions** (later, `(…)(x)`).
    """)
    return


@app.cell
def stepper_states(DB, MINSUP, extend, pattern_str, trace_nodes):
    def stepper_payload(db, nodes, minsup):
        """Turn the trace into (tree, states) for the stepper widget."""
        _tree, _rows_by_node = [], {}
        _leaf = [0]

        def _label(_pat):
            return pattern_str([tuple(_I) for _I in _pat]) if _pat else "⟨⟩"

        def _place(_nid, _depth):
            _n = nodes[_nid]
            _ys, _kids = [], []
            _child_iter = iter(_n["children"])
            for _c in _n["counts"]:
                if _c["freq"] and (_n["children"] and True):
                    _cid = next(_child_iter, None)
                    if _cid is None:     # max_len stop
                        continue
                    _ys.append(_place(_cid, _depth + 1))
                else:
                    _pat = extend([tuple(_I) for _I in _n["pattern"]], _c["item"], _c["kind"]) if _n["pattern"] or _c["kind"] == "s" else None
                    _y = _leaf[0]; _leaf[0] += 1
                    _tree.append({"key": f"p{_nid}-{_c['item']}{_c['kind']}", "label": pattern_str(_pat),
                                  "depth": _depth + 1, "y": _y, "parent": _nid, "status": "pruned",
                                  "sup": _c["sup"], "appear": _nid, "node": None})
                    _ys.append(_y)
            if not _ys:
                _y = _leaf[0]; _leaf[0] += 1
            else:
                _y = (min(_ys) + max(_ys)) / 2
            _tree.append({"key": f"n{_nid}", "label": _label(_n["pattern"]), "depth": _depth, "y": _y,
                          "parent": _n["parent"], "status": "root" if _nid == 0 else "freq",
                          "sup": None, "appear": _n["parent"] if _n["parent"] is not None else 0, "node": _nid})
            return _y

        _place(0, 0)
        for _t in _tree:
            if _t["node"] is not None and _t["node"] > 0:
                _par = nodes[_t["node"]]["parent"]
                _ppat = [tuple(_I) for _I in nodes[_par]["pattern"]]
                _t["sup"] = len(nodes[_t["node"]]["ptrs"])
        _total_items = sum(len(_I) for _s in db for _I in _s)
        _states, _found = [], []
        for _n in nodes:
            _rows = []
            _remaining = 0
            for _sid, _seq in enumerate(db):
                _p = _n["ptrs"].get(_sid)
                _its = []
                for _k, _I in enumerate(_seq):
                    _its.append([{"x": _x, "used": _p is not None and (_k < _p[0] or (_k == _p[0] and _j <= _p[1]))}
                                 for _j, _x in enumerate(_I)])
                if _p is not None:
                    _remaining += sum(1 for _I in _its for _c in _I if not _c["used"])
                _rows.append({"sid": f"S{_sid + 1}", "alive": _p is not None, "itemsets": _its})
            if _n["id"] > 0:
                _found.append(_label(_n["pattern"]))
            _states.append({"node": _n["id"], "label": _label(_n["pattern"]), "rows": _rows,
                            "counts": _n["counts"], "found": list(_found),
                            "remaining": _remaining, "total": _total_items,
                            "nseq": sum(_r["alive"] for _r in _rows)})
        return _tree, _states


    tree_layout, stepper_states = stepper_payload(DB, trace_nodes, MINSUP)
    len(stepper_states), len(tree_layout)
    return stepper_payload, stepper_states, tree_layout


@app.cell
def stepper_widget_class(anywidget, traitlets):
    class PrefixSpanStepper(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const NS = "http://www.w3.org/2000/svg";
      const C = { freq: "#2a9d55", pruned: "#999", red: "#d1495b", cur: "#f18f01", root: "#555" };
      el.style.fontFamily = "system-ui, sans-serif";
      const ctl = document.createElement("div"); ctl.style.marginBottom = "6px";
      const wrap = document.createElement("div");
      Object.assign(wrap.style, { display: "flex", gap: "18px", flexWrap: "wrap", alignItems: "flex-start" });
      const left = document.createElement("div"); left.style.flex = "1 1 420px";
      const right = document.createElement("div"); right.style.flex = "1 1 300px";
      const cap = document.createElement("div"); cap.style.margin = "6px 0";
      wrap.append(left, right); el.append(ctl, cap, wrap);
      let timer = null;
      const n = () => model.get("states").length;
      const go = d => { model.set("i", Math.max(0, Math.min(n() - 1, model.get("i") + d))); model.save_changes(); };
      const btn = (t, f) => { const b = document.createElement("button"); b.textContent = t; b.onclick = f;
        Object.assign(b.style, { marginRight: "4px", padding: "3px 10px", cursor: "pointer" }); ctl.appendChild(b); return b; };
      btn("⏮", () => go(-1e9)); btn("◀ prev", () => go(-1));
      const play = btn("▶ play", () => {
        if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; return; }
        if (model.get("i") >= n() - 1) { model.set("i", 0); model.save_changes(); }
        play.textContent = "⏸ pause";
        timer = setInterval(() => { if (model.get("i") >= n() - 1) { clearInterval(timer); timer = null; play.textContent = "▶ play"; return; } go(1); }, 1400);
      });
      btn("next ▶", () => go(1)); btn("⏭", () => go(1e9));

      function drawTree(s, i) {
        const tree = model.get("tree");
        const rows = Math.max(...tree.map(t => t.y)) + 1, depth = Math.max(...tree.map(t => t.depth));
        const W = 560, RH = 24, H = rows * RH + 20, COL = (W - 110) / Math.max(depth, 1);
        const X = d => 30 + d * COL, Y = y => 16 + y * RH;
        const svg = document.createElementNS(NS, "svg");
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
        Object.assign(svg.style, { width: "100%", maxWidth: W + "px" });
        const mk = (tag, a) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); svg.appendChild(e); return e; };
        const byNode = {}; tree.forEach(t => { if (t.node !== null) byNode[t.node] = t; });
        const visible = t => t.status === "root" || (t.status === "freq" ? t.node <= i : t.appear <= i);
        tree.forEach(t => { if (t.parent === null || !visible(t)) return; const p = byNode[t.parent];
          mk("path", { d: `M${X(p.depth)},${Y(p.y)} C${X(p.depth) + COL / 2},${Y(p.y)} ${X(t.depth) - COL / 2},${Y(t.y)} ${X(t.depth)},${Y(t.y)}`,
            fill: "none", stroke: t.status === "pruned" ? "#ccc" : "#8fcfa6", "stroke-width": 1.5, "stroke-dasharray": t.status === "pruned" ? "3 3" : "" }); });
        tree.forEach(t => {
          if (!visible(t)) return;
          const cur = t.node === s.node;
          const col = cur ? C.cur : t.status === "pruned" ? C.pruned : t.status === "root" ? C.root : C.freq;
          mk("circle", { cx: X(t.depth), cy: Y(t.y), r: cur ? 8 : 6, fill: col, stroke: cur ? "#7a4500" : "none", "stroke-width": 2 });
          if (t.status === "pruned") {
            const x = X(t.depth), y = Y(t.y);
            mk("path", { d: `M${x - 4},${y - 4} L${x + 4},${y + 4} M${x + 4},${y - 4} L${x - 4},${y + 4}`, stroke: C.red, "stroke-width": 2 });
          }
          const tx = mk("text", { x: X(t.depth) + 11, y: Y(t.y) + 4, "font-size": 12, fill: col,
            "font-weight": cur ? 700 : 400, "font-family": "ui-monospace, monospace" });
          tx.textContent = t.label + (t.sup !== null && t.status !== "root" ? `  ·${t.sup}` : "");
          if (t.status === "pruned") tx.setAttribute("text-decoration", "line-through");
        });
        left.innerHTML = ""; left.appendChild(svg);
      }

      function drawRight(s) {
        const cell = c => `<span style="${c.used ? "opacity:.4;text-decoration:line-through;color:#888" : "font-weight:600"}">${c.x}</span>`;
        const seq = r => r.itemsets.map(I => {
          const partial = I.some(c => c.used) && I.some(c => !c.used);
          return "(" + (partial ? '<span style="color:#f18f01">_</span>' : "") + I.map(cell).join(",") + ")"; }).join("");
        let h = `<div style="font-weight:600;margin-bottom:4px">Projected DB of <span style="color:#f18f01;font-family:monospace">${s.label}</span></div>`;
        h += `<table style="border-collapse:collapse;font-family:ui-monospace,monospace;font-size:14px">`;
        s.rows.forEach(r => { h += `<tr style="opacity:${r.alive ? 1 : .25}"><td style="padding:2px 8px;color:#888">${r.sid}</td>
          <td style="padding:2px 8px">${r.alive ? seq(r) : '<span style="text-decoration:line-through">' + seq(r) + "</span> (no match)"}</td></tr>`; });
        h += `</table>`;
        const tbl = (kind, title) => {
          const cs = s.counts.filter(c => c.kind === kind);
          if (!cs.length) return `<div style="color:#999;font-size:13px;margin-top:6px">${title}: none</div>`;
          const pre = s.label === "⟨⟩" ? "" : s.label;
          return `<div style="margin-top:8px;font-size:13px;font-weight:600">${title}</div><div>` + cs.map(c => {
            const pat = kind === "i" ? `${pre.slice(0, -1)},${c.item})` : `${pre}(${c.item})`;
            return `<span style="display:inline-block;margin:2px;padding:3px 7px;border-radius:6px;font-family:monospace;
              background:${c.freq ? "rgba(42,157,85,.18)" : "rgba(150,150,150,.15)"};color:${c.freq ? "#1d7a41" : "#888"};
              ${c.freq ? "" : "text-decoration:line-through"}">${pat} : ${c.sup}</span>`; }).join("") + "</div>";
        };
        if (s.label !== "⟨⟩") h += tbl("i", "itemset-extensions (together, “_x”)");
        h += tbl("s", s.label === "⟨⟩" ? "single items" : "sequence-extensions (followed by)");
        h += `<div style="margin-top:8px;font-size:12px;color:#888">minsup = ${model.get("minsup")} · green kept, grey pruned</div>`;
        right.innerHTML = h;
      }
      function draw() {
        const i = model.get("i"), s = model.get("states")[i];
        cap.innerHTML = `<b>step ${i + 1}/${n()}</b>: scanning the projected DB of <b style="color:#f18f01;font-family:monospace">${s.label}</b>
          · patterns output so far: ${s.found.length ? s.found.map(f => `<span style="color:#2a9d55;font-family:monospace">${f}</span>`).join(", ") : "none"}`;
        drawTree(s, i); drawRight(s);
      }
      model.on("change:i", draw); model.on("change:states", draw); draw();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        states = traitlets.List(traitlets.Dict()).tag(sync=True)
        tree = traitlets.List(traitlets.Dict()).tag(sync=True)
        minsup = traitlets.Int(3).tag(sync=True)
        i = traitlets.Int(0).tag(sync=True)

    return (PrefixSpanStepper,)


@app.cell
def stepper_view(MINSUP, PrefixSpanStepper, mo, stepper_states, tree_layout):
    stepper = mo.ui.anywidget(PrefixSpanStepper(states=stepper_states, tree=tree_layout, minsup=MINSUP))
    stepper
    return (stepper,)


@app.cell(hide_code=True)
def stepper_readout(DB, mo, stepper, stepper_states):
    _s = stepper_states[stepper.value["i"]]
    mo.vstack([
        mo.hstack([
            mo.stat(len(_s["found"]), label="patterns output so far", caption="of 7 in total"),
            mo.stat(f"{_s['remaining']} / {_s['total']}", label="items left in projected DB",
                    caption=f"{_s['remaining'] / _s['total']:.0%} of the original"),
            mo.stat(f"{_s['nseq']} / {len(DB)}", label="sequences still alive"),
            mo.stat(sum(_c["freq"] for _c in _s["counts"]), label="extensions kept here",
                    caption=f"{sum(not _c['freq'] for _c in _s['counts'])} pruned"),
        ], justify="start", gap=1),
        mo.callout(mo.md(r"""
    **Try this**

    1. Step to **(a)**. Find the two different counts for b: **(a,b)** = 3 (together) vs **(a)(b)** = 1 (later). Same letters, different patterns!
    2. Watch *items left in projected DB* as the prefix gets longer: the database keeps **shrinking**, which is why deep levels are cheap.
    3. Note that **d** is pruned at the root and never appears again, and that nobody ever counts a pattern like (d)(a): PrefixSpan only counts what it sees.
    """), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def pointer_md(mo):
    mo.md(r"""
    ## 4 · Optimization 1: pseudo-projection (pointers instead of copies)

    Copying the database for *every* prefix costs a lot of time and memory: there may be thousands
    of prefixes. The video's fix: **don't copy, point**. A projected database is just one
    **pointer per sequence** into the *original* database, marking where the prefix's first match ended.

    Build a prefix yourself by clicking. Buttons show the **support** each extension would have
    (green if ≥ minsup). Watch the <span style="color:#3b6fb6">**blue pointers**</span> slide right through the
    original sequences, consumed items turn grey, and sequences without a match fade out.

    The toggle applies the video's second optimization: **remove infrequent items** (like d) from
    the database once, after the first scan, so every later scan is shorter.
    """)
    return


@app.cell
def pointer_widget_class(anywidget, traitlets):
    class PointerBuilder(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      el.style.fontFamily = "system-ui, sans-serif";
      const top = document.createElement("div"), view = document.createElement("div"), btns = document.createElement("div");
      btns.style.margin = "8px 0"; el.append(top, view, btns);
      const sub = (A, B) => A.every(x => B.includes(x));
      function supports(db) { const s = {}; db.forEach(q => { new Set(q.flat()).forEach(x => s[x] = (s[x] || 0) + 1); }); return s; }
      function workingDb() {
        const db = model.get("db");
        if (!model.get("remove_infrequent")) return db;
        const s = supports(db), ms = model.get("minsup");
        return db.map(q => q.map(I => I.filter(x => s[x] >= ms)));   // keep positions: empty itemsets just get skipped
      }
      function toPattern(prefix) {
        const P = [];
        prefix.forEach(([x, k]) => { if (k === "i" && P.length) P[P.length - 1] = [...P[P.length - 1], x].sort(); else P.push([x]); });
        return P;
      }
      function project(db, P) {   // sid -> [k, j] position of last prefix item, first match
        const out = {};
        db.forEach((q, sid) => {
          if (!P.length) { out[sid] = [-1, -1]; return; }
          let k = -1;
          for (const I of P) { k++; while (k < q.length && !sub(I, q[k])) k++; if (k >= q.length) return; }
          out[sid] = [k, model.get("db")[sid][k].indexOf(P[P.length - 1][P[P.length - 1].length - 1])];
        });
        return out;
      }
      const fmt = P => P.length ? P.map(I => "(" + I.join(",") + ")").join("") : "⟨⟩";
      function draw() {
        const db = workingDb(), orig = model.get("db"), prefix = model.get("prefix"), P = toPattern(prefix);
        const ptr = project(db, P), ms = model.get("minsup"), sup = supports(orig);
        const nsup = Object.keys(ptr).length;
        top.innerHTML = `<div style="font-size:15px">prefix <b style="color:#f18f01;font-family:monospace;font-size:17px">${fmt(P)}</b>
          &nbsp; support <b style="color:${nsup >= ms ? "#2a9d55" : "#d1495b"}">${P.length ? nsup : "–"}</b>
          <span style="color:#888;font-size:13px">(minsup ${ms})</span></div>`;
        let h = `<table style="border-collapse:collapse;font-family:ui-monospace,monospace;font-size:15px;margin-top:6px">`;
        orig.forEach((q, sid) => {
          const p = ptr[sid];
          h += `<tr style="opacity:${p ? 1 : .25}"><td style="color:#888;padding:4px 10px 4px 0">S${sid + 1}</td><td style="padding:4px 0">`;
          q.forEach((I, k) => {
            h += `<span style="display:inline-block;border:1px solid rgba(128,128,128,.4);border-radius:6px;padding:2px 4px;margin-right:4px">`;
            I.forEach((x, j) => {
              const removed = model.get("remove_infrequent") && sup[x] < ms;
              const used = p && (k < p[0] || (k === p[0] && j <= p[1]));
              const here = p && p[0] === k && p[1] === j && P.length;
              let st = "display:inline-block;width:20px;text-align:center;position:relative;";
              if (removed) st += "color:#d1495b;opacity:.5;text-decoration:line-through;";
              else if (used) st += "color:#999;opacity:.45;text-decoration:line-through;";
              else st += "font-weight:600;";
              h += `<span style="${st}">${x}${here ? '<span style="position:absolute;left:14px;top:-9px;color:#3b6fb6;font-size:16px;text-decoration:none;opacity:1">▼</span>' : ""}</span>`;
            });
            h += `</span>`;
          });
          h += p ? (P.length ? `<span style="color:#3b6fb6;font-size:12px;margin-left:6px">ptr → (itemset ${p[0] + 1}, item ${p[1] + 1})</span>` : "")
                 : `<span style="color:#d1495b;font-size:12px;margin-left:6px">no match, dropped</span>`;
          h += `</td></tr>`;
        });
        view.innerHTML = h + "</table>";
        btns.innerHTML = "";
        const alphabet = [...new Set(orig.flat(2))].sort();
        const mkb = (label, s, fn, title) => { const b = document.createElement("button");
          b.innerHTML = `<span style="font-family:monospace">${label}</span> <small>${s}</small>`; b.title = title; b.onclick = fn;
          Object.assign(b.style, { margin: "2px", padding: "3px 8px", cursor: "pointer", borderRadius: "6px",
            border: `1px solid ${s >= ms ? "#2a9d55" : "#bbb"}`, background: s >= ms ? "rgba(42,157,85,.15)" : "transparent",
            color: s >= ms ? "#1d7a41" : "#888" }); return b; };
        const row = (t) => { const d = document.createElement("div"); d.innerHTML = `<small style="color:#888;display:inline-block;width:170px">${t}</small>`; btns.appendChild(d); return d; };
        const push = (x, k) => { model.set("prefix", [...prefix, [x, k]]); model.save_changes(); };
        const rs = row(P.length ? "followed by … (sequence-ext.)" : "start with item");
        alphabet.forEach(x => { const s = Object.keys(project(db, [...P, [x]])).length;
          rs.appendChild(mkb(`${fmt(P) === "⟨⟩" ? "" : fmt(P)}(${x})`, s, () => push(x, "s"), "sequence-extension")); });
        if (P.length) {
          const last = P[P.length - 1], ri = row("together (itemset-ext., _x)");
          alphabet.filter(x => x > last[last.length - 1]).forEach(x => {
            const Q = [...P.slice(0, -1), [...last, x]]; const s = Object.keys(project(db, Q)).length;
            ri.appendChild(mkb(fmt(Q), s, () => push(x, "i"), "itemset-extension")); });
        }
        const ctl = row("");
        const undo = document.createElement("button"); undo.textContent = "↶ undo"; undo.onclick = () => { model.set("prefix", prefix.slice(0, -1)); model.save_changes(); };
        const reset = document.createElement("button"); reset.textContent = "↺ reset"; reset.onclick = () => { model.set("prefix", []); model.save_changes(); };
        const lab = document.createElement("label"); lab.style.marginLeft = "12px";
        const cb = document.createElement("input"); cb.type = "checkbox"; cb.checked = model.get("remove_infrequent");
        cb.onchange = () => { model.set("remove_infrequent", cb.checked); model.save_changes(); };
        lab.append(cb, " remove infrequent items first (optimization 2)");
        [undo, reset].forEach(b => { b.style.margin = "2px"; b.style.cursor = "pointer"; ctl.appendChild(b); }); ctl.appendChild(lab);
      }
      ["change:prefix", "change:remove_infrequent", "change:db", "change:minsup"].forEach(e => model.on(e, draw));
      draw();
    }
    export default { render };
    """
        db = traitlets.List().tag(sync=True)
        minsup = traitlets.Int(3).tag(sync=True)
        prefix = traitlets.List().tag(sync=True)
        remove_infrequent = traitlets.Bool(False).tag(sync=True)

    return (PointerBuilder,)


@app.cell
def pointer_view(DB, MINSUP, PointerBuilder, mo):
    pointer_builder = mo.ui.anywidget(PointerBuilder(db=[[list(_I) for _I in _s] for _s in DB], minsup=MINSUP,
                                                      prefix=[["a", "s"]]))
    pointer_builder
    return (pointer_builder,)


@app.cell(hide_code=True)
def pointer_readout(
    DB,
    MINSUP,
    extend,
    mo,
    pattern_str,
    pointer_builder,
    prefixspan,
    project_pointers,
):
    def prefix_to_pattern(prefix):
        _P = []
        for _x, _k in prefix:
            _P = extend(_P, _x, _k) if _P else [(_x,)]
        return _P


    def drop_infrequent(db, minsup):
        _sup = {}
        for _s in db:
            for _x in {_x for _I in _s for _x in _I}:
                _sup[_x] = _sup.get(_x, 0) + 1
        return [[tuple(_x for _x in _I if _sup[_x] >= minsup) for _I in _s] for _s in db]


    def postfix_items(db, ptrs):
        return sum(len(db[_s][_k]) - _j - 1 + sum(len(_I) for _I in db[_s][_k + 1:]) for _s, (_k, _j) in ptrs.items())


    _v = pointer_builder.value
    _db = drop_infrequent(DB, MINSUP) if _v["remove_infrequent"] else DB
    _P = prefix_to_pattern(_v["prefix"])
    _ptrs = project_pointers(_db, _P)
    _copied = postfix_items(_db, _ptrs) if _P else sum(len(_I) for _s in _db for _I in _s)
    # whole run: real copies vs pointers, for every projected DB PrefixSpan builds
    _all, _st, _ = prefixspan(_db, MINSUP)
    _ptr_total = sum(_sup for _, _sup in _all)
    mo.vstack([
        mo.hstack([
            mo.stat(len(_ptrs) if _P else "–", label=f"support of {pattern_str(_P) or '⟨⟩'}",
                    caption="frequent ✓" if _P and len(_ptrs) >= MINSUP else ("infrequent ✕" if _P else "pick an item")),
            mo.stat(_copied, label="items a real copy would store", caption="this projected DB"),
            mo.stat(len(_ptrs), label="pointers needed instead", caption="one (itemset, item) pair per sequence"),
            mo.stat(f"{_st['projected_items']} vs {_ptr_total}", label="whole run: copied items vs pointers",
                    caption=f"over all {_st['projections']} projections"),
        ], justify="start", gap=1),
        mo.callout(mo.md(r"""
    **Try this**

    1. Click **(a)** then **(a,b)** then **(a,b)(c)**: the pointer moves right each time, and S3 is dropped right away.
    2. Compare **(a)(b)** (support 1) with **(a,b)** (support 3). The pointer for (a,b) stays *inside* the first itemset.
    3. Tick *remove infrequent items*: **d** disappears, and the copy count for the whole run drops while the patterns stay the same.
       On a tiny DB the savings look small; on real data with millions of items the video quotes ~10× time and memory.
    """), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def perf_md(mo):
    mo.md(r"""
    ## 5 · What makes PrefixSpan slow? Lower the minimum support

    The video lists what drives the cost: **minsup** first (lower it and the time can grow
    *exponentially*), then the database: number of sequences, their length, how similar they are,
    and the number of distinct items.

    Generate a random sequence database below (seeded, so it's reproducible). Item popularity follows a
    skewed (Zipf-like) law so some items are common, like real shopping data. The chart sweeps minsup
    from 60 % of the sequences down to your chosen floor. Runs are capped at 20 000 patterns so the
    notebook stays responsive.
    """)
    return


@app.cell
def perf_controls(mo):
    perf_n = mo.ui.slider(20, 400, step=10, value=150, label="# sequences")
    perf_items = mo.ui.slider(4, 40, step=1, value=12, label="# distinct items")
    perf_len = mo.ui.slider(2, 10, step=1, value=5, label="avg. # itemsets per sequence")
    perf_floor = mo.ui.slider(0.02, 0.4, step=0.01, value=0.08, label="lowest minsup (fraction)")
    perf_seed = mo.ui.number(0, 999, value=7, label="seed")
    mo.vstack([mo.hstack([perf_n, perf_items, perf_len], justify="start"), mo.hstack([perf_floor, perf_seed], justify="start")])
    return perf_floor, perf_items, perf_len, perf_n, perf_seed


@app.cell
def perf_data(
    fmt_seq,
    mo,
    np,
    pd,
    perf_floor,
    perf_items,
    perf_len,
    perf_n,
    perf_seed,
    prefixspan,
):
    def random_db(n_seq, n_items, avg_len, seed):
        _rng = np.random.default_rng(seed)
        _alpha = [chr(97 + _i) if _i < 26 else f"i{_i}" for _i in range(n_items)]
        _p = 1 / np.arange(1, n_items + 1) ** 1.1
        _p /= _p.sum()
        _db = []
        for _ in range(n_seq):
            _L = max(1, _rng.poisson(avg_len))
            _seq = []
            for _ in range(_L):
                _k = 1 + _rng.binomial(2, 0.25)
                _seq.append(tuple(sorted(set(_rng.choice(_alpha, size=_k, p=_p)))))
            _db.append(_seq)
        return _db


    def perf_sweep(db, floor, cap=20000):
        import time as _time
        _rows = []
        for _frac in np.unique(np.round(np.geomspace(0.6, floor, 10), 3))[::-1]:
            _ms = max(1, int(round(_frac * len(db))))
            _t0 = _time.perf_counter()
            _pats, _st, _ = prefixspan(db, _ms, max_patterns=cap)
            _rows.append({"minsup_frac": float(_frac), "minsup": _ms, "patterns": len(_pats),
                          "runtime_ms": 1000 * (_time.perf_counter() - _t0),
                          "projections": _st["projections"],
                          "extensions counted (PrefixSpan)": _st["extensions_counted"],
                          "blind candidates (GSP-style)": _st["blind_candidates"],
                          "capped": len(_pats) >= cap})
        return pd.DataFrame(_rows)


    perf_db = random_db(perf_n.value, perf_items.value, perf_len.value, int(perf_seed.value))
    perf_df = perf_sweep(perf_db, perf_floor.value)
    mo.md("**Sample of the generated database:** " + " · ".join(f"`{fmt_seq(_s)}`" for _s in perf_db[:5]) + " …")
    return (perf_df,)


@app.cell
def perf_chart(alt, perf_df):
    _base = alt.Chart(perf_df).encode(
        x=alt.X("minsup_frac:Q", title="minsup (fraction of sequences)", scale=alt.Scale(reverse=True)))
    _pat = _base.mark_line(point=True, color="#2a9d55").encode(
        y=alt.Y("patterns:Q", scale=alt.Scale(type="symlog"), title="# frequent patterns"),
        tooltip=["minsup", "patterns", "capped"])
    _rt = _base.mark_line(point=True, color="#f18f01").encode(
        y=alt.Y("runtime_ms:Q", scale=alt.Scale(type="symlog"), title="runtime (ms)"),
        tooltip=["minsup", alt.Tooltip("runtime_ms:Q", format=".1f")])
    _long = perf_df.melt(id_vars=["minsup_frac"], value_vars=["extensions counted (PrefixSpan)", "blind candidates (GSP-style)"])
    _cmp = alt.Chart(_long).mark_line(point=True).encode(
        x=alt.X("minsup_frac:Q", title="minsup (fraction of sequences)", scale=alt.Scale(reverse=True)),
        y=alt.Y("value:Q", scale=alt.Scale(type="symlog"), title="patterns looked at"),
        color=alt.Color("variable:N", scale=alt.Scale(range=["#999", "#2a9d55"]), title=None,
                        legend=alt.Legend(orient="bottom")),
        strokeDash=alt.StrokeDash("variable:N", legend=None))
    perf_chart = alt.hconcat(
        alt.layer(_pat, _rt).resolve_scale(y="independent").properties(width=330, height=240,
            title="patterns (green) and runtime (orange) explode as minsup drops"),
        _cmp.properties(width=330, height=240, title="what gets counted"),
    )
    perf_chart
    return


@app.cell(hide_code=True)
def perf_readout(mo, perf_df):
    _last = perf_df.iloc[-1]
    _first = perf_df.iloc[0]
    mo.vstack([
        mo.hstack([
            mo.stat(int(_last["patterns"]), label=f"patterns at minsup={_last['minsup']}",
                    caption=f"vs {int(_first['patterns'])} at minsup={_first['minsup']}" + (" (capped!)" if _last["capped"] else "")),
            mo.stat(f"{_last['runtime_ms']:.0f} ms", label="runtime at lowest minsup",
                    caption=f"{_last['runtime_ms'] / max(_first['runtime_ms'], 1e-3):.0f}× the highest minsup"),
            mo.stat(f"{_last['blind candidates (GSP-style)'] / max(_last['extensions counted (PrefixSpan)'], 1):.1f}×",
                    label="blind candidates / counted extensions"),
        ], justify="start", gap=1),
        mo.callout(mo.md(r"""
    *Blind candidates* = what a generator that never looks at the data would have to test: every
    item as an itemset- and as a sequence-extension of every frequent pattern. PrefixSpan only counts
    the items that actually occur in each projected database (green line). (Real GSP's join is smarter
    than "every item", but still creates candidates without checking the data.)

    **Try this**

    1. Drag *lowest minsup* down towards 0.02 and watch the curves bend upward: that's the exponential blow-up.
    2. Raise *# distinct items* with everything else fixed: more items → sparser data → fewer patterns, but more blind candidates.
    3. Raise *avg. # itemsets*: longer sequences → more patterns and longer scans.
    """), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def sandbox_md(mo):
    mo.md(r"""
    ## 6 · Sandbox: mine your own sequences

    One sequence per line. Write itemsets in parentheses, `(ab)(c)(a)` or `(a,b)(c)(a)`, or
    separate itemsets with spaces, `ab c a`. Single letters are items; use commas for longer item
    names, e.g. `(milk,bread)(eggs)`.

    The video notes PrefixSpan is **easy to extend with constraints**; here you can cap the
    pattern length (number of items), a one-line check in the recursion.
    """)
    return


@app.cell
def sandbox_controls(mo):
    sandbox_text = mo.ui.text_area(value="(ab)(c)(a)\n(ab)(b)(c)\n(b)(c)(d)\n(b)(ab)(c)\n(a)(bc)(d)",
                                   label="Sequence database", rows=7, full_width=True)
    sandbox_minsup = mo.ui.slider(1, 10, value=2, label="minsup (absolute)", show_value=True)
    sandbox_maxlen = mo.ui.slider(1, 8, value=8, label="max pattern length (items)", show_value=True)
    mo.vstack([sandbox_text, mo.hstack([sandbox_minsup, sandbox_maxlen], justify="start")])
    return sandbox_maxlen, sandbox_minsup, sandbox_text


@app.cell
def sandbox_mine(
    fmt_seq,
    mo,
    pattern_str,
    pd,
    prefixspan,
    sandbox_maxlen,
    sandbox_minsup,
    sandbox_text,
):
    import re as _re


    def parse_db(text):
        def _itemset(_tok):
            _tok = _tok.strip()
            _xs = [_x.strip() for _x in _tok.split(",")] if "," in _tok else list(_tok.replace(" ", ""))
            return tuple(sorted({_x for _x in _xs if _x}))

        _db = []
        for _line in text.strip().splitlines():
            _line = _line.strip()
            if not _line:
                continue
            _toks = _re.findall(r"\(([^)]*)\)", _line) if "(" in _line else _line.split()
            _seq = [_I for _I in (_itemset(_t) for _t in _toks) if _I]
            if _seq:
                _db.append(_seq)
        return _db


    sandbox_db = parse_db(sandbox_text.value)
    sandbox_patterns, _, sandbox_trace = prefixspan(sandbox_db, sandbox_minsup.value, max_len=sandbox_maxlen.value,
                                                    trace=True, max_patterns=5000)
    _df = pd.DataFrame([{"pattern": pattern_str(_p), "support": _s, "length": sum(len(_I) for _I in _p),
                         "# itemsets": len(_p)} for _p, _s in sandbox_patterns])
    mo.vstack([
        mo.md(f"Parsed **{len(sandbox_db)}** sequences: " + " · ".join(f"`{fmt_seq(_s)}`" for _s in sandbox_db[:8])
              + (" …" if len(sandbox_db) > 8 else "")),
        mo.hstack([mo.stat(len(sandbox_patterns), label="frequent patterns"),
                   mo.stat(len(sandbox_trace), label="prefixes explored (DFS nodes)")], justify="start", gap=1),
        mo.ui.table(_df.sort_values(["length", "pattern"]) if len(_df) else _df, selection=None,
                    label="Frequent sequential patterns") if len(_df) else mo.callout("No frequent patterns. Lower minsup.", kind="warn"),
    ])
    return parse_db, sandbox_db, sandbox_trace


@app.cell
def sandbox_stepper(
    PrefixSpanStepper,
    mo,
    sandbox_db,
    sandbox_minsup,
    sandbox_trace,
    stepper_payload,
):
    mo.stop(len(sandbox_trace) > 150 or not sandbox_db,
            mo.md("_Search tree too large to animate (> 150 nodes); raise minsup to step through it._"))
    _tree, _states = stepper_payload(sandbox_db, sandbox_trace, sandbox_minsup.value)
    mo.vstack([mo.md("**Step through your own database:**"),
               mo.ui.anywidget(PrefixSpanStepper(states=_states, tree=_tree, minsup=sandbox_minsup.value))])
    return


@app.cell(hide_code=True)
def quiz_md(mo):
    mo.md(r"""
    ## 7 · Check yourself

    Commit to your answers **first**, then flip the reveal switch. All answers are about the
    video's database (S1 (a,b)(c)(a), S2 (a,b)(b)(c), S3 (b)(c)(d), S4 (b)(a,b)(c)) unless stated.
    """)
    return


@app.cell
def quiz_form(mo):
    _sup = ["0", "1", "2", "3", "4"]
    quiz = mo.ui.dictionary({
        "q1": mo.ui.dropdown(_sup, label="1. Support of (a)(c)?"),
        "q2": mo.ui.dropdown(_sup, label="2. Support of (a)(b), a *followed by* b?"),
        "q3": mo.ui.dropdown(_sup, label="3. Support of (a,b), a and b *together*?"),
        "q4": mo.ui.dropdown(_sup, label="4. Support of (b)(a)?"),
        "q5": mo.ui.dropdown(["S1, S2, S4", "S1, S2", "S1, S2, S3, S4", "S2, S4"],
                             label="5. Which sequences survive in the projected DB of (a,b)?"),
        "q6": mo.ui.number(0, 20, value=0, label="6. Mini DB (a)(b)(c) / (a,c)(b) / (b)(c), minsup 2: how many frequent patterns?"),
        "q7": mo.ui.dropdown(["yes", "no"], label="7. Can PrefixSpan ever count a pattern that occurs in 0 sequences?"),
    })
    reveal = mo.ui.switch(label="reveal answers")
    mo.vstack([quiz, reveal])
    return quiz, reveal


@app.cell(hide_code=True)
def quiz_grade(
    DB,
    mo,
    parse_db,
    pattern_str,
    prefixspan,
    project_pointers,
    quiz,
    reveal,
):
    def support(db, pattern):
        return len(project_pointers(db, pattern))


    _mini = parse_db("(a)(b)(c)\n(a,c)(b)\n(b)(c)")
    _truth = {
        "q1": str(support(DB, [("a",), ("c",)])),
        "q2": str(support(DB, [("a",), ("b",)])),
        "q3": str(support(DB, [("a", "b")])),
        "q4": str(support(DB, [("b",), ("a",)])),
        "q5": ", ".join(f"S{_s + 1}" for _s in project_pointers(DB, [("a", "b")])),
        "q6": len(prefixspan(_mini, 2)[0]),
        "q7": "no",
    }
    _why = {
        "q1": "S1, S2, S4 have an a and a later c.",
        "q2": "Only S2 has a b in a *later* itemset than an a. Don't confuse with (a,b)!",
        "q3": "S1, S2 and S4 all contain the itemset (a,b).",
        "q4": "S1 (b … a) and S4 (b)(a,b). Order matters: (a)(b) had support 1.",
        "q5": "S3 has no itemset containing both a and b, so it is dropped.",
        "q6": f"{', '.join(pattern_str(_p) for _p, _ in prefixspan(_mini, 2)[0])}.",
        "q7": "No. Extensions are only counted when the item is seen in the projected DB, so support ≥ 1 always.",
    }
    _ans = quiz.value
    _ok = {_k: _ans.get(_k) not in (None, "") and str(_ans[_k]) == str(_truth[_k]) for _k in _truth}
    mo.stop(not reveal.value, mo.md("_Answer all questions, then flip **reveal answers**._"))
    mo.vstack([
        mo.md(f"### Score: {sum(_ok.values())} / {len(_ok)}"),
        mo.md("\n".join(f"- **{_k[1:]}.** {'✅' if _ok[_k] else '❌'} answer **{_truth[_k]}**. {_why[_k]}" for _k in _truth)),
    ])
    return


@app.cell(hide_code=True)
def recap(mo):
    mo.vstack([mo.md("## Recap"), mo.md(r"""
    | idea | what to remember | where you played with it |
    |---|---|---|
    | sequence DB & support | ordered itemsets; support = # sequences containing the pattern | §1 table |
    | itemset- vs sequence-extension | **(a,b)** together ≠ **(a)(b)** later; counted separately ("_b") | §3 count tables, §4 buttons |
    | database projection | keep only what follows the first match of the prefix; drop non-matching sequences | §3 right panel |
    | depth-first search | extend one frequent prefix all the way down, then backtrack | §3 search tree |
    | pruning (anti-monotonicity) | an infrequent prefix can't have frequent extensions (d, (a)(b), …) | §3 grey ✕ nodes |
    | only real patterns | candidates come from the data, never from blind joins (vs GSP) | §5 right chart |
    | pseudo-projection | one pointer per sequence instead of a copy, ~10× less memory | §4 |
    | remove infrequent items | drop d once after the first scan; every later scan is shorter | §4 toggle |
    | performance | minsup ↓ ⇒ patterns & time ↑ (exponentially); DB size, length, #items matter | §5 sliders |
    | constraints | easy to add, e.g. max length, max gap | §6 |
    """), mo.md("*Source: Philippe Fournier-Viger, “The PrefixSpan algorithm” (video). A fast Java implementation is in the open-source SPMF library.*")])
    return


@app.cell(hide_code=True)
def done_toast(mo):
    mo.status.toast("PrefixSpan lesson ready", "Start with the stepper in §3 and press ▶ play.")
    return


if __name__ == "__main__":
    app.run()

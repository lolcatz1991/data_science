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
    # Entropy, explained by playing with it
    *An interactive companion to DataMListic's video **"Entropy - Explained"**.*

    > Someone picks a number from one to eight. You may ask yes/no questions. **How many do you need?**

    That guessing game leads, in six steps, to one of the central ideas of information theory:
    **surprise → entropy → optimal codes → cross-entropy & KL divergence → maximum entropy.**
    Every section below has something to click, drag or paint.
    """),
    mo.callout(mo.md(r"""
    **Color language used throughout**

    - <span style="color:#3b6fb6">**blue**</span>: the *true* distribution $P$ (what actually happens)
    - <span style="color:#f18f01">**orange**</span>: a *model* / codebook distribution $Q$ (what you *assume*)
    - <span style="color:#e4572e">**red**</span>: the key quantity, an entropy or a bit count
    - <span style="color:#999">**grey**</span>: eliminated, impossible, or a reference/ghost value

    All logs are base 2, so every quantity is measured in **bits**.
    """), kind="info"),
        mo.Html('<iframe width="560" height="315" src="https://www.youtube.com/embed/ZBp-uFwYlQg" title="Entropy - Explained" frameborder="0" allowfullscreen></iframe>'),
    ])
    return


@app.cell
def helpers(np):
    BLUE, ORANGE, RED, GREY = "#3b6fb6", "#f18f01", "#e4572e", "#999999"

    def entropy(p):
        """H(P) = -sum p log2 p, in bits (0 log 0 = 0)."""
        p = np.asarray(p, float); p = p / p.sum()
        nz = p > 0
        return float(-(p[nz] * np.log2(p[nz])).sum())

    def cross_entropy(p, q):
        """H(P,Q) = -sum p log2 q, in bits. Infinite if q=0 where p>0."""
        p = np.asarray(p, float); p = p / p.sum()
        q = np.asarray(q, float); q = q / q.sum()
        nz = p > 0
        if np.any(q[nz] == 0):
            return float("inf")
        return float(-(p[nz] * np.log2(q[nz])).sum())

    def kl(p, q):
        """D_KL(P||Q) = H(P,Q) - H(P)."""
        return cross_entropy(p, q) - entropy(p)

    def huffman(p):
        """Return Huffman code words (list of '0'/'1' strings) for probabilities p."""
        import heapq
        p = list(np.asarray(p, float) / np.sum(p))
        if len(p) == 1:
            return ["0"]
        heap = [(w, i, [i]) for i, w in enumerate(p)]
        heapq.heapify(heap)
        codes = [""] * len(p)
        counter = len(p)
        while len(heap) > 1:
            w1, _, s1 = heapq.heappop(heap)
            w2, _, s2 = heapq.heappop(heap)
            for k in s1: codes[k] = "0" + codes[k]
            for k in s2: codes[k] = "1" + codes[k]
            heapq.heappush(heap, (w1 + w2, counter, s1 + s2)); counter += 1
        return codes

    return BLUE, GREY, ORANGE, RED, cross_entropy, entropy, huffman, kl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The guessing game

    If all 8 numbers are equally likely, the best strategy is **binary search**:
    *"Is it in the left half?" → "Is it in the first two?" → "Is it the first one?"*
    Each answer cuts the candidates in half, so you always need exactly

    $$\log_2 8 = 3 \text{ questions.}$$

    But what if some numbers are **more likely** than others? Then a smart questioner splits the
    **probability mass** in half, not the *count* of candidates, and common numbers get found faster.

    **How to play:** the widget secretly draws a number from the bar heights. Click cells to select a set, then
    press **"Is it in my selection?"**. Eliminated cells turn grey. Or press **🤖 auto** to watch the
    mass-halving strategy play a round. Each finished round is logged to Python.
    """)
    return


@app.cell
def GuessGame_cls(anywidget, traitlets):
    class GuessGame(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const N = 8;
      let secret = null, alive = [], sel = new Set(), asked = 0, done = false, timer = null;
      const top = document.createElement("div"); top.style.margin = "4px 0";
      const board = document.createElement("div"); board.style.display = "flex"; board.style.gap = "6px";
      const msg = document.createElement("div"); msg.style.margin = "8px 0"; msg.style.minHeight = "1.4em";
      el.append(top, board, msg);
      const btn = (t, f) => { const b = document.createElement("button"); b.textContent = t; b.onclick = f;
        b.style.marginRight = "6px"; top.appendChild(b); return b; };
      function sample() { const p = model.get("probs"); const s = p.reduce((a,b)=>a+b,0);
        let r = Math.random() * s; for (let i = 0; i < p.length; i++) { r -= p[i]; if (r <= 0) return i; } return p.length - 1; }
      function newRound() { if (timer) { clearInterval(timer); timer = null; }
        secret = sample(); alive = Array.from({length: N}, (_, i) => model.get("probs")[i] > 0);
        sel = new Set(); asked = 0; done = false; msg.innerHTML = "New secret drawn. Select cells and ask!"; draw(); }
      function ask() {
        if (done || sel.size === 0) return;
        asked++;
        const yes = sel.has(secret);
        for (let i = 0; i < N; i++) if (alive[i] && (sel.has(i) !== yes)) alive[i] = false;
        msg.innerHTML = `Q${asked}: is it in {${[...sel].map(i=>i+1).join(", ")}}? → <b>${yes ? "YES" : "NO"}</b>`;
        sel = new Set();
        if (alive.filter(a => a).length === 1) {
          done = true;
          msg.innerHTML += ` &nbsp; 🎉 It's <b>${secret + 1}</b>, found in <b style="color:#e4572e">${asked}</b> question${asked>1?"s":""}.`;
          const h = [...model.get("history"), asked]; model.set("history", h);
          const s = [...model.get("secrets"), secret + 1]; model.set("secrets", s); model.save_changes();
        }
        draw();
      }
      function autoQ() {   // split remaining probability mass as evenly as possible (greedy, sorted)
        const p = model.get("probs");
        const idx = [...Array(N).keys()].filter(i => alive[i]).sort((a, b) => p[b] - p[a]);
        const tot = idx.reduce((a, i) => a + p[i], 0);
        let best = null, bestGap = 1e9, acc = 0;
        for (let k = 1; k < idx.length; k++) { acc += p[idx[k-1]];
          const g = Math.abs(acc - tot / 2); if (g < bestGap) { bestGap = g; best = k; } }
        sel = new Set(idx.slice(0, best)); draw();
      }
      function auto() { if (done) newRound(); if (timer) return;
        timer = setInterval(() => { if (done) { clearInterval(timer); timer = null; return; }
          if (sel.size === 0) autoQ(); else ask(); }, 550); }
      function draw() {
        board.innerHTML = "";
        const p = model.get("probs"), mx = Math.max(...p, 1e-9);
        for (let i = 0; i < N; i++) {
          const c = document.createElement("div");
          const isSel = sel.has(i);
          Object.assign(c.style, { width: "60px", height: "110px", border: isSel ? "3px solid #f18f01" : "1px solid #8884",
            borderRadius: "8px", cursor: alive[i] && !done ? "pointer" : "default", position: "relative",
            display: "flex", flexDirection: "column", justifyContent: "flex-end", alignItems: "center",
            opacity: alive[i] ? 1 : 0.3, overflow: "hidden" });
          const bar = document.createElement("div");
          Object.assign(bar.style, { width: "100%", height: (70 * p[i] / mx) + "px",
            background: alive[i] ? "#3b6fb6aa" : "#999" });
          const lab = document.createElement("div"); lab.style.fontWeight = "bold";
          lab.innerHTML = (done && i === secret) ? `<span style="color:#e4572e">${i+1}★</span>` : (i + 1);
          const pr = document.createElement("div"); pr.style.fontSize = "11px"; pr.textContent = p[i].toFixed(3);
          c.append(lab, pr, bar);
          c.onclick = () => { if (!alive[i] || done) return; sel.has(i) ? sel.delete(i) : sel.add(i); draw(); };
          board.appendChild(c);
        }
      }
      btn("🎲 new round", newRound);
      btn("❓ Is it in my selection?", ask);
      btn("💡 suggest split", () => { if (!done) autoQ(); });
      btn("🤖 auto", auto);
      btn("🤖×20 fast", () => { for (let r = 0; r < 20; r++) { newRound();
          while (!done) { autoQ(); ask(); } } });
      btn("🧹 clear log", () => { model.set("history", []); model.set("secrets", []); model.save_changes(); });
      model.on("change:probs", newRound);
      newRound();
      return () => { if (timer) clearInterval(timer); };
    }
    export default { render };
    """
        probs = traitlets.List(traitlets.Float()).tag(sync=True)
        history = traitlets.List(traitlets.Int()).tag(sync=True)
        secrets = traitlets.List(traitlets.Int()).tag(sync=True)

    return (GuessGame,)


@app.cell
def game_dist(mo):
    GAME_PRESETS = {
        "Uniform (all 1/8)": [1/8] * 8,
        "Skewed (1/2, 1/4, 1/8, ...)": [1/2, 1/4, 1/8, 1/16, 1/32, 1/64, 1/128, 1/128],
        "Almost certain (0.93 on '1')": [0.93] + [0.01] * 7,
    }
    game_dist = mo.ui.dropdown(list(GAME_PRESETS), value="Uniform (all 1/8)", label="Secret-number distribution")
    game_dist
    return GAME_PRESETS, game_dist


@app.cell
def game(GAME_PRESETS, GuessGame, game_dist, mo):
    game = mo.ui.anywidget(GuessGame(probs=GAME_PRESETS[game_dist.value]))
    game
    return (game,)


@app.cell
def _(GAME_PRESETS, entropy, game, game_dist, mo, np):
    _h = game.value["history"]
    _p = np.array(GAME_PRESETS[game_dist.value])
    _H = entropy(_p)
    _avg = np.mean(_h) if _h else float("nan")
    mo.vstack([
        mo.hstack([
            mo.stat(len(_h), label="rounds played"),
            mo.stat(f"{_avg:.2f}" if _h else "–", label="your avg. # questions"),
            mo.stat(f"{np.log2(8):.2f}", label="binary search on count: log₂8"),
            mo.stat(f"{_H:.2f} bits", label="entropy H(P) of this distribution"),
        ]),
        mo.md("_Play a few rounds (or press 🤖×20) and compare your average with the entropy._" if not _h else
              f"Last rounds: {_h[-15:]}"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. With **Uniform**, press 🤖×20 a few times: every round takes exactly **3** questions = $\log_2 8$.
    2. Switch to **Skewed**. Now the auto-player often finishes in **1** question (it asks "is it 1?" first). The average falls well below 3, close to the entropy shown on the right.
    3. **Almost certain**: you barely need to ask. There's very little *uncertainty* left to remove. Entropy measures exactly that.
    """), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · The measure of surprise

    Key insight from the video: **rare events carry more information than common ones.**
    A fair coin landing heads? Not very surprising: **1 bit**. An event with probability 1/8? **3 bits**.

    $$I(x) = -\log_2 P(x)$$

    Why the logarithm? If two events are **independent**, their surprises should **add**.
    Probabilities multiply, $P(A \cap B) = P(A)\,P(B)$, and the log turns that multiplication into addition:

    $$-\log_2\big(P(A)P(B)\big) = -\log_2 P(A) \;+\; -\log_2 P(B).$$
    """)
    return


@app.cell
def surprise_sliders(mo):
    surprise_pa = mo.ui.slider(0.01, 1.0, step=0.01, value=0.5, label="P(A)", show_value=True)
    surprise_pb = mo.ui.slider(0.01, 1.0, step=0.01, value=0.125, label="P(B)", show_value=True)
    mo.hstack([surprise_pa, surprise_pb])
    return surprise_pa, surprise_pb


@app.cell
def surprise_chart(
    BLUE,
    GREY,
    ORANGE,
    RED,
    alt,
    mo,
    np,
    pd,
    surprise_pa,
    surprise_pb,
):
    _pa, _pb = surprise_pa.value, surprise_pb.value
    _grid = pd.DataFrame({"p": np.linspace(0.005, 1, 300)})
    _grid["surprise"] = -np.log2(_grid["p"])
    _pts = pd.DataFrame({"p": [_pa, _pb, _pa * _pb], "surprise": -np.log2([_pa, _pb, _pa * _pb]),
                         "event": ["A", "B", "A and B"]})
    _curve = alt.Chart(_grid).mark_line(color=GREY).encode(
        x=alt.X("p:Q", title="probability p"), y=alt.Y("surprise:Q", title="surprise −log₂ p  (bits)", scale=alt.Scale(domain=[0, 14])))
    _dots = alt.Chart(_pts).mark_circle(size=160).encode(x="p", y="surprise",
        color=alt.Color("event:N", scale=alt.Scale(domain=["A", "B", "A and B"], range=[BLUE, ORANGE, RED])),
        tooltip=["event", alt.Tooltip("p", format=".4f"), alt.Tooltip("surprise", format=".3f")])
    _rules = alt.Chart(_pts).mark_rule(strokeDash=[4, 3]).encode(y="surprise", color=alt.Color("event:N", legend=None))
    _sa, _sb = -np.log2(_pa), -np.log2(_pb)
    mo.vstack([
        (_curve + _rules + _dots).properties(width="container", height=280, title="Surprise of an outcome"),
        mo.md(f"**{_sa:.3f}** bits (A) + **{_sb:.3f}** bits (B) = "
              f"<span style='color:{RED}'>**{_sa + _sb:.3f}**</span> bits = −log₂({_pa:.2f}·{_pb:.3f}) = −log₂({_pa*_pb:.4f})"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Set P(A) = 0.5: exactly **1 bit**. Set P(B) = 0.125: exactly **3 bits**, the video's two examples.
    2. Push P(A) to 1.0: a certain event has **zero** surprise. You learned nothing.
    3. Watch the red dot: the joint event's surprise is always the sum of the other two. That's the property that *forces* the log.
    """), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Expected surprise: the entropy formula

    If surprise tells you how informative **one** outcome is, entropy is the **average surprise across all outcomes**,
    each weighted by how often it actually happens:

    $$H(X) = -\sum_x P(x)\,\log_2 P(x) = \mathbb{E}\big[-\log_2 P(X)\big].$$

    **Paint a distribution** over 8 outcomes below (drag across the bars, or use a preset). The widget shows $H$ live;
    the chart underneath decomposes it into per-outcome contributions $P(x)\cdot(-\log_2 P(x))$.
    """)
    return


@app.cell
def DistPainter_cls(anywidget, np, traitlets):
    class DistPainter(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 640, H = 230, PADL = 40, TOP = 20, plotH = 170;
      const NS = "http://www.w3.org/2000/svg";
      const toolbar = document.createElement("div"); toolbar.style.marginBottom = "4px"; el.appendChild(toolbar);
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none", cursor: "crosshair" });
      el.appendChild(svg);
      const mk = (tag, a, parent = svg) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); parent.appendChild(e); return e; };
      const toSvg = evt => { const p = svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const presets = {
        "Uniform": (k, n) => 1,
        "Video: ½ ¼ ⅛ ⅛": (k, n) => [8, 4, 2, 2][k] || 0,
        "Peaked": (k, n) => Math.exp(-0.5 * ((k - n/2) / 0.9) ** 2),
        "Geometric": (k, n) => Math.pow(0.5, k),
        "Certain": (k, n) => k === 0 ? 1 : 0,
        "Two coins": (k, n) => k < 2 ? 1 : 0,
      };
      for (const name in presets) {
        const b = document.createElement("button"); b.textContent = name; b.style.marginRight = "4px";
        b.onclick = () => { const n = model.get("weights").length;
          const w = Array.from({ length: n }, (_, k) => presets[name](k, n));
          const m = Math.max(...w, 1e-9); model.set("weights", w.map(v => v / m)); model.save_changes(); };
        toolbar.appendChild(b);
      }
      const color = model.get("color");
      let painting = false;
      function draw() {
        svg.innerHTML = "";
        const w = model.get("weights"), n = w.length, bw = (W - PADL) / n;
        const s = w.reduce((a, b) => a + b, 0);
        const p = w.map(v => s > 0 ? v / s : 0);
        const Hn = -p.reduce((a, v) => a + (v > 0 ? v * Math.log2(v) : 0), 0);
        mk("line", { x1: PADL, x2: W, y1: TOP + plotH, y2: TOP + plotH, stroke: "#888" });
        p.forEach((v, k) => {
          const h = w[k] * plotH;
          mk("rect", { x: PADL + k * bw + 3, y: TOP + plotH - h, width: bw - 6, height: h, fill: color, opacity: 0.75, rx: 3 });
          const t = mk("text", { x: PADL + (k + .5) * bw, y: TOP + plotH + 16, "text-anchor": "middle", "font-size": 12, fill: "currentColor" });
          t.textContent = `${model.get("labels")[k] ?? k + 1}`;
          const t2 = mk("text", { x: PADL + (k + .5) * bw, y: TOP + plotH - h - 4, "text-anchor": "middle", "font-size": 11, fill: "currentColor" });
          t2.textContent = s > 0 ? v.toFixed(3) : "";
        });
        const tt = mk("text", { x: W - 6, y: 14, "text-anchor": "end", "font-size": 15, fill: "#e4572e", "font-weight": "bold" });
        tt.textContent = s > 0 ? `H = ${Hn.toFixed(3)} bits  (max log₂${n} = ${Math.log2(n).toFixed(2)})` : "paint something!";
      }
      function paint(evt) {
        const pt = toSvg(evt), n = model.get("weights").length, bw = (W - PADL) / n;
        const k = Math.floor((pt.x - PADL) / bw); if (k < 0 || k >= n) return;
        const w = [...model.get("weights")];
        w[k] = Math.max(0, Math.min(1, (TOP + plotH - pt.y) / plotH));
        model.set("weights", w); draw();
      }
      svg.addEventListener("pointerdown", e => { painting = true; svg.setPointerCapture(e.pointerId); paint(e); });
      svg.addEventListener("pointermove", e => painting && paint(e));
      const stop = () => { if (painting) { painting = false; model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointercancel", stop);
      model.on("change:weights", draw); draw();
    }
    export default { render };
    """
        weights = traitlets.List(traitlets.Float()).tag(sync=True)
        labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
        color = traitlets.Unicode("#3b6fb6").tag(sync=True)

    def painter_probs(w):
        """Normalize a painter's weights to a probability vector (uniform if empty)."""
        _w = np.asarray(w, float)
        return _w / _w.sum() if _w.sum() > 0 else np.full(len(_w), 1 / len(_w))

    return DistPainter, painter_probs


@app.cell
def painter(BLUE, DistPainter, mo):
    painter = mo.ui.anywidget(DistPainter(weights=[1, .5, .25, .25, 0, 0, 0, 0],
                                            labels=[str(i) for i in range(1, 9)], color=BLUE))
    painter
    return (painter,)


@app.cell
def entropy_readout(
    GREY,
    RED,
    alt,
    entropy,
    mo,
    np,
    painter,
    painter_probs,
    pd,
):
    _p = painter_probs(painter.value["weights"])
    _df = pd.DataFrame({"x": [str(i) for i in range(1, 9)], "P(x)": _p})
    _df["surprise"] = np.where(_p > 0, -np.log2(np.where(_p > 0, _p, 1)), np.nan)
    _df["contribution"] = np.where(_p > 0, _df["P(x)"] * _df["surprise"], 0.0)
    _H = entropy(_p)
    _bars = alt.Chart(_df).mark_bar(color=RED, opacity=0.8).encode(
        x=alt.X("x:N", title="outcome x", sort=None), y=alt.Y("contribution:Q", title="P(x)·(−log₂P(x))  bits"),
        tooltip=["x", alt.Tooltip("P(x)", format=".3f"), alt.Tooltip("surprise", format=".2f"), alt.Tooltip("contribution", format=".3f")])
    _surp = alt.Chart(_df).mark_tick(color=GREY, thickness=2, size=30).encode(x=alt.X("x:N", sort=None), y="surprise:Q")
    mo.vstack([
        mo.hstack([mo.stat(f"{_H:.3f} bits", label="H(X) = Σ contributions"),
                   mo.stat(f"{np.log2((_p > 0).sum()):.3f}", label="log₂(# possible outcomes)"),
                   mo.stat(f"{2**_H:.2f}", label="2^H: 'effective # of equally-likely outcomes'")]),
        (_bars + _surp).properties(width="container", height=220,
            title="Red bars = each outcome's share of H · grey ticks = its raw surprise"),
    ])
    return


@app.cell
def coin_p(mo):
    coin_p = mo.ui.slider(0.0, 1.0, step=0.01, value=0.5, label="P(heads) = p", show_value=True)
    mo.vstack([mo.md(r"""
    ### The binary entropy function

    For a coin with $P(\text{heads}) = p$, the formula simplifies to

    $$H_b(p) = -p\log_2 p - (1-p)\log_2(1-p).$$

    "Look at this curve," says the video: at $p=\tfrac12$ (maximum uncertainty) it peaks at **exactly 1 bit**;
    at the extremes, when you already know what will happen, it drops to **zero**.
    """), coin_p])
    return (coin_p,)


@app.cell
def binary_entropy(BLUE, RED, alt, coin_p, entropy, np, pd):
    _g = pd.DataFrame({"p": np.linspace(0, 1, 201)})
    _g["H"] = [entropy([q, 1 - q]) for q in _g["p"]]
    _pt = pd.DataFrame({"p": [coin_p.value], "H": [entropy([coin_p.value, 1 - coin_p.value])]})
    _c = alt.Chart(_g).mark_line(color=RED, strokeWidth=3).encode(x=alt.X("p:Q", title="p = P(heads)"), y=alt.Y("H:Q", title="H_b(p)  bits"))
    _d = alt.Chart(_pt).mark_circle(size=200, color=BLUE).encode(x="p", y="H")
    _t = alt.Chart(_pt).mark_text(dy=-16, fontWeight="bold").encode(x="p", y="H", text=alt.Text("H:Q", format=".3f"))
    (_c + _d + _t).properties(width="container", height=240, title="Binary entropy: the whole story in one curve")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Press **Uniform**: $H = 3$ bits, the same 3 questions as the guessing game. Uniform is the *maximum* for 8 outcomes.
    2. Press **Video: ½ ¼ ⅛ ⅛**: $H = 1.75$ bits. Remember that number; it returns in Section 4.
    3. Press **Certain**: $H = 0$. Then paint a tiny bar on another outcome. Note how its grey *surprise* tick is huge, but its red *contribution* stays small. Rare events are surprising, but they are also rare.
    4. Slide the coin to $p = 0.11$: the entropy is already only half a bit.
    """), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · The coding connection

    Here's what makes entropy more than a formula: **it is the minimum average number of bits needed to encode a message.**

    - 4 equally likely symbols → 2 bits each (`00 01 10 11`), and $H = 2$. ✔
    - Probabilities $\tfrac12, \tfrac14, \tfrac18, \tfrac18$ → give the common symbol a **short** code (`0`) and the rare ones
      **longer** codes (`10`, `110`, `111`). The average drops to **1.75 bits/symbol**, *exactly* the entropy.

    You can't do better: **entropy is the theoretical floor** (Shannon's source coding theorem).
    The ideal code length for symbol $x$ is its surprise, $\ell(x) \approx -\log_2 P(x)$.
    Below, the codes are built with **Huffman's algorithm**, which finds the best prefix code for any distribution.
    """)
    return


@app.cell
def code_dist(mo):
    CODE_PRESETS = {
        "Video: ½ ¼ ⅛ ⅛": [1/2, 1/4, 1/8, 1/8],
        "4 equally likely": [1/4] * 4,
        "Skewed: 0.7 0.1 0.1 0.1": [0.7, 0.1, 0.1, 0.1],
        "Painted distribution (Section 3)": None,
    }
    code_dist = mo.ui.dropdown(list(CODE_PRESETS), value="Video: ½ ¼ ⅛ ⅛", label="Source distribution")
    code_dist
    return CODE_PRESETS, code_dist


@app.cell
def code_table(
    CODE_PRESETS,
    code_dist,
    entropy,
    huffman,
    mo,
    np,
    painter,
    painter_probs,
    pd,
):
    _raw = CODE_PRESETS[code_dist.value]
    code_p = np.array(_raw) if _raw is not None else painter_probs(painter.value["weights"])
    code_labels = list("ABCDEFGH")[: len(code_p)]
    code_huff = huffman(code_p)
    _nfix = int(np.ceil(np.log2(len(code_p))))
    code_fixed = [format(i, f"0{_nfix}b") for i in range(len(code_p))]
    _tbl = pd.DataFrame({
        "symbol": code_labels, "P(x)": np.round(code_p, 4),
        "surprise −log₂P": np.round(np.where(code_p > 0, -np.log2(np.where(code_p > 0, code_p, 1)), np.inf), 3),
        "fixed code": code_fixed, "Huffman code": code_huff, "Huffman length": [len(c) for c in code_huff],
    })
    _avg_h = float(np.sum(code_p * np.array([len(c) for c in code_huff])))
    mo.vstack([
        mo.hstack([mo.stat(f"{entropy(code_p):.3f}", label="entropy H (the floor)"),
                   mo.stat(f"{_avg_h:.3f}", label="Huffman avg bits/symbol"),
                   mo.stat(f"{_nfix}", label="fixed-length bits/symbol")]),
        mo.ui.table(_tbl, selection=None, pagination=False),
    ])
    return code_fixed, code_huff, code_labels, code_p


@app.cell
def EncoderSim_cls(anywidget, traitlets):
    class EncoderSim(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      let timer = null, st = null;
      const ctl = document.createElement("div"); ctl.style.marginBottom = "4px";
      const stream = document.createElement("div");
      Object.assign(stream.style, { fontFamily: "monospace", fontSize: "13px", margin: "6px 0", lineHeight: "1.6" });
      const canvas = document.createElement("canvas"); canvas.width = 640; canvas.height = 220;
      canvas.style.width = "100%"; canvas.style.maxWidth = "640px";
      el.append(ctl, stream, canvas);
      const cx = canvas.getContext("2d");
      const btn = (t, f) => { const b = document.createElement("button"); b.textContent = t; b.onclick = f; b.style.marginRight = "6px"; ctl.appendChild(b); return b; };
      function reset() { const books = model.get("codebooks");
        st = { n: 0, bits: books.map(() => 0), paths: books.map(() => []), recent: [] }; draw(); }
      function sample() { const p = model.get("probs"); let r = Math.random(); for (let i = 0; i < p.length; i++) { r -= p[i]; if (r <= 0) return i; } return p.length - 1; }
      function step(k = 1) {
        const books = model.get("codebooks");
        for (let j = 0; j < k; j++) {
          const s = sample(); st.n++;
          books.forEach((b, i) => { st.bits[i] += b.codes[s].length; st.paths[i].push(st.bits[i] / st.n); });
          st.recent.push(s); if (st.recent.length > 12) st.recent.shift();
        }
        draw();
        if (st.n % 50 < k) { model.set("summary", { n: st.n, bits: [...st.bits] }); model.save_changes(); }
      }
      function draw() {
        const books = model.get("codebooks"), labels = model.get("labels");
        stream.innerHTML = `<div>last symbols: <b>${st.recent.map(s => labels[s]).join(" ")}</b></div>` +
          books.map((b, i) => `<div><span style="color:${b.color};font-weight:bold">${b.name}</span>: ` +
            st.recent.map(s => b.codes[s]).join("·") +
            ` &nbsp; → <b>${st.n ? (st.bits[i] / st.n).toFixed(3) : "–"}</b> bits/symbol</div>`).join("");
        cx.clearRect(0, 0, 640, 220);
        const ymax = Math.max(...books.map(b => Math.max(...b.codes.map(c => c.length))), model.get("H") + 0.5);
        const Y = v => 205 - v / ymax * 190;
        cx.font = "11px sans-serif"; cx.fillStyle = "#888";
        for (let v = 0; v <= ymax; v += 0.5) { cx.fillText(v.toFixed(1), 2, Y(v) + 3); cx.strokeStyle = "#8882"; cx.beginPath(); cx.moveTo(28, Y(v)); cx.lineTo(640, Y(v)); cx.stroke(); }
        cx.strokeStyle = "#e4572e"; cx.lineWidth = 2; cx.setLineDash([6, 4]); cx.beginPath(); cx.moveTo(28, Y(model.get("H"))); cx.lineTo(640, Y(model.get("H"))); cx.stroke();
        cx.setLineDash([]); cx.fillStyle = "#e4572e"; cx.fillText("entropy H = " + model.get("H").toFixed(3), 540, Y(model.get("H")) - 5);
        books.forEach((b, i) => { const path = st.paths[i]; if (!path.length) return;
          cx.strokeStyle = b.color; cx.lineWidth = 2; cx.beginPath();
          path.forEach((v, t) => { const x = 28 + t / Math.max(path.length - 1, 1) * 610; t ? cx.lineTo(x, Y(v)) : cx.moveTo(x, Y(v)); });
          cx.stroke(); });
        cx.fillStyle = "#888"; cx.fillText(`symbols sent: ${st.n}`, 34, 14);
      }
      const play = btn("▶ play", () => { if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; }
        else { timer = setInterval(() => step(1), 60); play.textContent = "⏸ pause"; } });
      btn("step", () => step(1));
      btn("⏩ +500", () => step(500));
      btn("↺ reset", () => { reset(); model.set("summary", {}); model.save_changes(); });
      model.on("change:codebooks", reset); model.on("change:probs", reset); model.on("change:H", draw);
      reset();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        probs = traitlets.List(traitlets.Float()).tag(sync=True)
        labels = traitlets.List(traitlets.Unicode()).tag(sync=True)
        codebooks = traitlets.List(traitlets.Dict()).tag(sync=True)
        H = traitlets.Float(0.0).tag(sync=True)
        summary = traitlets.Dict().tag(sync=True)

    return (EncoderSim,)


@app.cell
def encoder(
    BLUE,
    EncoderSim,
    GREY,
    code_fixed,
    code_huff,
    code_labels,
    code_p,
    entropy,
    mo,
):
    encoder = mo.ui.anywidget(EncoderSim(
        probs=[float(v) for v in code_p], labels=code_labels, H=entropy(code_p),
        codebooks=[{"name": "fixed-length", "codes": code_fixed, "color": GREY},
                   {"name": "Huffman (built for P)", "codes": code_huff, "color": BLUE}]))
    encoder
    return (encoder,)


@app.cell
def _(code_p, encoder, entropy, mo):
    _s = encoder.value["summary"]
    mo.md("_Press ▶ play or ⏩ +500. The running averages are synced to Python every 50 symbols._") if not _s else mo.hstack([
        mo.stat(_s["n"], label="symbols encoded (synced)"),
        mo.stat(f"{_s['bits'][0] / _s['n']:.3f}", label="fixed-length bits/symbol"),
        mo.stat(f"{_s['bits'][1] / _s['n']:.3f}", label="Huffman bits/symbol"),
        mo.stat(f"{entropy(code_p):.3f}", label="entropy floor"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. With **Video: ½ ¼ ⅛ ⅛**, press ⏩ a few times. The blue line settles on **1.75**, right on the red entropy line, while the fixed code is stuck at 2.
    2. Switch to **4 equally likely**: Huffman can't beat 2 bits, because there's no imbalance to exploit.
    3. Try **Skewed: 0.7 0.1 0.1 0.1**: Huffman gets 1.5 bits but $H≈1.36$. Whole-bit codes can't hit the floor exactly unless the probabilities are powers of ½ (block coding or arithmetic coding gets closer).
    4. Pick **Painted distribution**, then go back up and repaint. The codebook rebuilds itself.
    """), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · The wrong code: cross-entropy and KL divergence

    What if you design your codebook for a distribution <span style="color:#f18f01">**Q**</span>, but the data actually comes from
    <span style="color:#3b6fb6">**P**</span>? Symbol $x$ gets a code of length $-\log_2 Q(x)$, but it shows up with frequency $P(x)$:

    $$\underbrace{H(P,Q) = -\sum_x P(x)\log_2 Q(x)}_{\text{cross-entropy: avg bits under the wrong model}}
    \qquad
    \underbrace{D_{KL}(P\,\|\,Q) = H(P,Q) - H(P)}_{\text{wasted bits}} \;\ge 0$$

    KL is **zero only when Q = P**. That's why cross-entropy is the loss function everywhere in ML:
    *minimizing cross-entropy pushes your model Q toward the true distribution P.*

    Below, P is the video's ½ ¼ ⅛ ⅛. Set your model **Q** with the sliders (they get normalized).
    """)
    return


@app.cell
def kl_q(mo):
    kl_q = mo.ui.array([mo.ui.slider(0.01, 1.0, step=0.01, value=0.25, label=f"Q weight for {_s}", show_value=True)
                        for _s in "ABCD"])
    mo.hstack([kl_q, mo.md("Q is rescaled to sum to 1.")], justify="start", gap=2)
    return (kl_q,)


@app.cell
def kl_readout(
    BLUE,
    ORANGE,
    alt,
    cross_entropy,
    entropy,
    kl,
    kl_q,
    mo,
    np,
    pd,
):
    KL_P = np.array([1/2, 1/4, 1/8, 1/8])
    kl_qv = np.array(kl_q.value) / np.sum(kl_q.value)
    _df = pd.DataFrame({"symbol": list("ABCD") * 2, "prob": np.r_[KL_P, kl_qv],
                        "dist": ["P (true)"] * 4 + ["Q (model)"] * 4})
    _bars = alt.Chart(_df).mark_bar().encode(
        x=alt.X("symbol:N"), xOffset="dist:N", y=alt.Y("prob:Q", title="probability"),
        color=alt.Color("dist:N", scale=alt.Scale(range=[BLUE, ORANGE]), title=None),
        tooltip=["dist", "symbol", alt.Tooltip("prob", format=".3f")])
    _len = pd.DataFrame({"symbol": list("ABCD"),
                         "ideal length under P": -np.log2(KL_P), "length you pay under Q": -np.log2(kl_qv),
                         "P(x)": KL_P})
    _len["extra bits × P(x)"] = _len["P(x)"] * (_len["length you pay under Q"] - _len["ideal length under P"])
    _H, _HPQ = entropy(KL_P), cross_entropy(KL_P, kl_qv)
    mo.vstack([
        mo.hstack([mo.stat(f"{_H:.3f}", label="H(P): ideal bits"),
                   mo.stat(f"{_HPQ:.3f}", label="H(P,Q): bits you actually use"),
                   mo.stat(f"{_HPQ - _H:.3f}", label="D_KL(P‖Q): wasted bits"),
                   mo.stat(f"{kl(kl_qv, KL_P):.3f}", label="D_KL(Q‖P) (note: not symmetric!)")]),
        _bars.properties(width=560, height=220, title="True P vs your model Q"),
        mo.ui.table(_len.round(3), selection=None, pagination=False),
    ])
    return KL_P, kl_qv


@app.cell
def ce_loss_curve(
    BLUE,
    ORANGE,
    RED,
    alt,
    coin_p,
    cross_entropy,
    entropy,
    mo,
    np,
    pd,
):
    _qs = np.linspace(0.01, 0.99, 197)
    _curve = pd.DataFrame({"Q(heads)": _qs})
    _curve["H(P,Q)"] = [cross_entropy([coin_p.value, 1 - coin_p.value], [_q, 1 - _q]) for _q in _qs]
    _curve["H(P)"] = entropy([coin_p.value, 1 - coin_p.value])
    _long = _curve.melt("Q(heads)", var_name="quantity", value_name="bits")
    _ch = alt.Chart(_long).mark_line(strokeWidth=3).encode(
        x="Q(heads):Q", y=alt.Y("bits:Q", scale=alt.Scale(domain=[0, 4]), title="bits"),
        color=alt.Color("quantity:N", scale=alt.Scale(domain=["H(P,Q)", "H(P)"], range=[ORANGE, RED])),
        strokeDash=alt.StrokeDash("quantity:N", legend=None))
    _m = alt.Chart(pd.DataFrame({"x": [coin_p.value]})).mark_rule(color=BLUE, strokeDash=[3, 3]).encode(x="x:Q")
    mo.vstack([
        mo.md(f"**Cross-entropy as a loss.** True coin: P(heads) = **{coin_p.value:.2f}** (the slider from Section 3). "
              "Sweep the model's guess Q(heads): the orange curve is the loss, the gap above the red line is KL, "
              "and the minimum sits exactly at Q = P (blue rule)."),
        (_ch + _m).properties(width="container", height=240),
    ])
    return


@app.cell
def wrong_encoder(BLUE, EncoderSim, KL_P, ORANGE, entropy, huffman, kl_qv, mo):
    _book_q = huffman(kl_qv)
    wrong_encoder = mo.ui.anywidget(EncoderSim(
        probs=[float(v) for v in KL_P], labels=list("ABCD"), H=entropy(KL_P),
        codebooks=[{"name": "Huffman built for P (right code)", "codes": huffman(KL_P), "color": BLUE},
                   {"name": "Huffman built for Q (wrong code)", "codes": _book_q, "color": ORANGE}]))
    mo.vstack([mo.md("**Send data from P through both codebooks.** The orange code was designed for your Q:"), wrong_encoder])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Set all Q sliders equal (uniform model): $H(P,Q) = 2$, so you waste **0.25 bits** per symbol.
    2. Set Q to 0.5, 0.25, 0.125, 0.125 (proportions 4:2:1:1). KL drops to **0**: the right model costs nothing extra.
    3. **Break it:** drag Q(A) down to 0.01. The common symbol now gets a ~7-bit code, and KL explodes. Being confidently wrong about frequent events is very expensive.
    4. Compare the last two stats: $D_{KL}(P\|Q) \ne D_{KL}(Q\|P)$. KL is a divergence, **not** a distance.
    """), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Maximum entropy: the distribution that assumes the least

    *Given what you know, which distribution should you choose?* The one with **maximum entropy**: it encodes the least
    extra assumptions beyond your constraints.

    - **No constraints** (just "one of $n$ outcomes") → the max-entropy distribution is **uniform**.
    - **Fixed mean** on a die → an exponential tilt $P(k) \propto e^{\lambda k}$ (Jaynes' famous dice problem).
    - **Fixed mean and variance** on the real line → the **Gaussian**. *"The bell curve isn't just convenient; it's the distribution that assumes the least."*

    First the die: you only know its **average roll**. Which distribution over 1–6 should you believe?
    """)
    return


@app.cell
def die_mean(mo):
    die_mean = mo.ui.slider(1.2, 5.8, step=0.05, value=3.5, label="known average roll", show_value=True)
    die_mean
    return (die_mean,)


@app.cell
def maxent_die(GREY, RED, alt, die_mean, entropy, np, pd):
    from scipy.optimize import brentq as _brentq
    _k = np.arange(1, 7)
    def maxent_die(m):
        """Max-entropy distribution on 1..6 with mean m: p_k ∝ exp(lam*k)."""
        _f = lambda lam: (np.exp(lam * _k) / np.exp(lam * _k).sum() * _k).sum() - m
        lam = 0.0 if abs(m - 3.5) < 1e-9 else _brentq(_f, -30, 30)
        w = np.exp(lam * _k)
        return w / w.sum()
    _pm = maxent_die(die_mean.value)
    # a competitor with the same mean: mass split between the two faces around the mean
    _lo = int(np.clip(np.floor(die_mean.value), 1, 5)); _t = die_mean.value - _lo
    _alt = np.zeros(6); _alt[_lo - 1] = 1 - _t; _alt[_lo] = _t
    _df = pd.DataFrame({"face": np.r_[_k, _k], "prob": np.r_[_pm, _alt],
                        "dist": [f"max-entropy  H={entropy(_pm):.3f}"] * 6 + [f"two-face guess  H={entropy(_alt):.3f}"] * 6})
    alt.Chart(_df).mark_bar().encode(
        x=alt.X("face:O"), xOffset="dist:N", y=alt.Y("prob:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("dist:N", scale=alt.Scale(range=[RED, GREY]), title=None, legend=alt.Legend(orient="top")),
    ).properties(width="container", height=240,
                 title=f"Both have mean {die_mean.value:.2f}; only one avoids inventing extra structure")
    return


@app.cell
def cont_sigma(mo):
    cont_sigma = mo.ui.slider(0.5, 2.0, step=0.1, value=1.0, label="σ (shared standard deviation)", show_value=True)
    mo.vstack([mo.md(r"""
    ### Fix the mean and variance → the Gaussian

    For continuous variables we use *differential entropy* $h(X) = -\int f(x)\log_2 f(x)\,dx$.
    Every distribution below has **mean 0 and variance $\sigma^2$**. Which one has the most entropy?
    """), cont_sigma])
    return (cont_sigma,)


@app.cell
def maxent_cont(BLUE, GREY, ORANGE, RED, alt, cont_sigma, mo, np, pd):
    from scipy import stats as _st
    _s = cont_sigma.value
    _dists = {
        "Gaussian": _st.norm(0, _s),
        "Laplace": _st.laplace(0, _s / np.sqrt(2)),
        "Logistic": _st.logistic(0, _s * np.sqrt(3) / np.pi),
        "Triangular": _st.triang(0.5, loc=-_s * np.sqrt(6), scale=2 * _s * np.sqrt(6)),
        "Uniform": _st.uniform(-_s * np.sqrt(3), 2 * _s * np.sqrt(3)),
    }
    cont_h = {n: float(d.entropy() / np.log(2)) for n, d in _dists.items()}   # nats → bits
    _x = np.linspace(-4.5 * _s, 4.5 * _s, 400)
    _df = pd.concat([pd.DataFrame({"x": _x, "density": d.pdf(_x), "dist": f"{n}  h={cont_h[n]:.3f} bits"})
                     for n, d in _dists.items()])
    _gname = f"Gaussian  h={cont_h['Gaussian']:.3f} bits"
    _names = sorted(_df["dist"].unique(), key=lambda s: -float(s.split("h=")[1].split()[0]))
    _ch = alt.Chart(_df).mark_line().encode(
        x="x:Q", y="density:Q",
        color=alt.Color("dist:N", sort=_names, title="ranked by entropy",
                        scale=alt.Scale(domain=_names, range=[RED, BLUE, ORANGE, "#6c5b7b", GREY])),
        strokeWidth=alt.condition(alt.datum.dist == _gname, alt.value(4), alt.value(1.8)))
    mo.vstack([
        _ch.properties(width="container", height=260, title=f"Same mean (0) and variance ({_s**2:.2f}), different entropy"),
        mo.md(f"Gaussian closed form: $h = \\tfrac12\\log_2(2\\pi e\\sigma^2)$ = **{0.5*np.log2(2*np.pi*np.e*_s**2):.3f}** bits ✔ (matches scipy: {cont_h['Gaussian']:.3f})"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Set the die's average to **3.5**: the max-entropy answer is exactly **uniform** ($H = \log_2 6 ≈ 2.585$). No constraint beyond "fair-looking", so no extra assumptions.
    2. Slide it to **4.5** (Jaynes' example): the answer tilts smoothly toward 6. The two-face guess matches the same mean but invents a strong claim ("only 4s and 5s happen"), and its entropy is far lower.
    3. In the continuous chart, change σ. The *ranking* never changes: the **Gaussian is always on top**. Every curve's entropy shifts by the same $\log_2\sigma$.
    """), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Sandbox: paint P and Q

    Paint a **true** distribution <span style="color:#3b6fb6">**P**</span> and a **model** <span style="color:#f18f01">**Q**</span>
    over 8 outcomes. Everything in this lesson updates live: entropy, the Huffman code for P, the cross-entropy and KL.
    """)
    return


@app.cell
def sandbox(BLUE, DistPainter, ORANGE, mo):
    sand_p = mo.ui.anywidget(DistPainter(weights=[1, .8, .6, .4, .3, .2, .1, .05], labels=list("ABCDEFGH"), color=BLUE))
    sand_q = mo.ui.anywidget(DistPainter(weights=[1] * 8, labels=list("ABCDEFGH"), color=ORANGE))
    mo.hstack([mo.vstack([mo.md("**P (true)**"), sand_p]), mo.vstack([mo.md("**Q (model)**"), sand_q])], widths="equal")
    return sand_p, sand_q


@app.cell
def sandbox_readout(
    cross_entropy,
    entropy,
    huffman,
    mo,
    np,
    painter_probs,
    sand_p,
    sand_q,
):
    _p = painter_probs(sand_p.value["weights"]); _q = painter_probs(sand_q.value["weights"])
    _hp = huffman(_p)
    _avg = float(np.sum(_p * np.array([len(c) for c in _hp])))
    _ce = cross_entropy(_p, _q)
    mo.vstack([
        mo.hstack([mo.stat(f"{entropy(_p):.3f}", label="H(P)"),
                   mo.stat(f"{_avg:.3f}", label="Huffman avg for P"),
                   mo.stat(f"{entropy(_q):.3f}", label="H(Q)"),
                   mo.stat("∞" if np.isinf(_ce) else f"{_ce:.3f}", label="H(P,Q)"),
                   mo.stat("∞" if np.isinf(_ce) else f"{_ce - entropy(_p):.3f}", label="D_KL(P‖Q)")]),
        mo.md("Huffman code for P: " + " · ".join(f"`{s}`={c}" for s, c in zip("ABCDEFGH", _hp))),
        mo.callout(mo.md("**Challenges:** (1) make $H(P)$ exactly 2 bits using 4 bars. (2) get KL below 0.01 with P and Q looking different to the eye. "
                         "(3) make KL = ∞. Hint: Q must say 'impossible' where P says 'possible'. "
                         "(4) find a P where Huffman exactly equals H(P)."), kind="info"),
    ])
    return


@app.cell
def quiz(mo):
    QUIZ = [
        ("How many yes/no questions to find a uniformly random number from 1 to 32?", ["4", "5", "16", "32"], "5"),
        ("An event has probability 1/16. Its surprise is…", ["1/16 bit", "4 bits", "16 bits", "0.25 bits"], "4 bits"),
        ("Which coin has the highest entropy?", ["P(heads)=0.9", "P(heads)=0.5", "P(heads)=0.1", "P(heads)=1.0"], "P(heads)=0.5"),
        ("Symbols with P = ½, ¼, ⅛, ⅛. The best possible average code length is…", ["2 bits", "1.5 bits", "1.75 bits", "1 bit"], "1.75 bits"),
        ("D_KL(P‖Q) = 0 means…", ["P and Q are independent", "Q = P exactly", "H(P) = 0", "Q is uniform"], "Q = P exactly"),
        ("You know only the mean and variance of a real-valued quantity. Max entropy picks…", ["Uniform", "Laplace", "Gaussian", "a point mass"], "Gaussian"),
    ]
    quiz = mo.ui.dictionary({str(i): mo.ui.radio(opts, label=f"**Q{i+1}.** {q}") for i, (q, opts, _a) in enumerate(QUIZ)})
    quiz_reveal = mo.ui.switch(label="Reveal answers")
    mo.vstack([mo.md("## 8 · Quiz: check your intuition"), quiz, quiz_reveal])
    return QUIZ, quiz, quiz_reveal


@app.cell
def quiz_score(QUIZ, entropy, mo, np, quiz, quiz_reveal):
    # sanity: the "truths" come from the notebook's own functions
    assert abs(np.log2(32) - 5) < 1e-12 and abs(-np.log2(1/16) - 4) < 1e-12
    assert abs(entropy([.5, .25, .125, .125]) - 1.75) < 1e-12
    _ans = quiz.value
    _score = sum(_ans.get(str(i)) == a for i, (_q, _o, a) in enumerate(QUIZ))
    _rows = [f"- Q{i+1}: {'✅' if _ans.get(str(i)) == a else '❌'} correct answer: **{a}**" for i, (_q, _o, a) in enumerate(QUIZ)]
    mo.vstack([
        mo.stat(f"{_score} / {len(QUIZ)}", label="score"),
        mo.md("\n".join(_rows)) if quiz_reveal.value else mo.md("_Flip the switch to see the answers._"),
    ])
    return


@app.cell(hide_code=True)
def recap(mo):
    mo.md(r"""
    ## Recap

    | Idea | Formula | One-line intuition | Where you played with it |
    |---|---|---|---|
    | Guessing game | $\log_2 n$ questions | Each yes/no answer halves the uncertainty | §1 guessing-game widget |
    | Surprise | $I(x) = -\log_2 P(x)$ | Rare = informative; log makes independent surprises add | §2 sliders |
    | Entropy | $H(X) = -\sum P(x)\log_2 P(x)$ | Average surprise; max $\log_2 n$ (uniform), min 0 (certain) | §3 painter, binary entropy curve |
    | Coding | $H \le \bar\ell < H + 1$ | Entropy is the floor on average bits/symbol | §4 Huffman table + encoder simulator |
    | Cross-entropy | $H(P,Q) = -\sum P\log_2 Q$ | Bits you pay using a code built for the wrong model | §5 Q sliders, loss curve |
    | KL divergence | $D_{KL}(P\|Q) = H(P,Q) - H(P) \ge 0$ | Wasted bits; 0 iff Q = P; not symmetric | §5, sandbox |
    | Max entropy | uniform / exp-tilt / Gaussian | Choose the distribution that assumes the least | §6 die + continuous comparison |

    *"Entropy measures what you don't know, and that turns out to be one of the most useful things you can measure."*
    """)
    return


if __name__ == "__main__":
    app.run()

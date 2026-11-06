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
    from scipy import stats

    return alt, anywidget, mo, np, pd, stats, traitlets


@app.cell(hide_code=True)
def title(mo):
    mo.vstack([
        mo.md(r"""
    # The Optimizer's Curse
    *An interactive companion to marimo's video **"The Optimizer's Curse"**.*

    > You win an auction. You feel great. Then someone tells you **100 people** were bidding. Suddenly you feel a lot worse.

    Whenever you **pick the maximum of noisy estimates**, the winner is, on average, the most *over-optimistic* estimate.
    In auctions that is the **winner's curse**; in machine learning (hyperparameter search, model selection, evals) it is the
    **optimizer's curse**. This notebook replays the video's simulations so you can *feel* why it happens and how to guard against it.
    """),
        mo.Html('<iframe width="100%" height="400" src="https://www.youtube.com/embed/Q-KsTooC9WE" title="The Optimizer\'s Curse" frameborder="0" allow="accelerometer; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>'),
    ])
    return


@app.cell(hide_code=True)
def colors(mo):
    mo.callout(mo.md(r"""
    **Color language used throughout**

    <span style="color:#2a9d8f">■ green</span> = the **true value** (what the thing is really worth) &nbsp;·&nbsp;
    <span style="color:#8d86a8">■ grey-purple</span> = **noisy estimates / bids / rolls** &nbsp;·&nbsp;
    <span style="color:#e4572e">■ red</span> = the **winner / maximum** &nbsp;·&nbsp;
    <span style="color:#1d70b8">■ blue</span> = a **corrected** estimate (our remedy)
    """), kind="neutral")
    return


@app.cell
def intro_guess(mo):
    intro_guess = mo.ui.radio(
        options=["I probably got a bargain", "About right, the market is efficient", "I probably overpaid"],
        label="**Before scrolling:** you just won an auction against 99 other bidders who all estimated the value as well as you did. What's your gut feeling?",
    )
    intro_guess
    return (intro_guess,)


@app.cell(hide_code=True)
def _(intro_guess, mo):
    mo.md(
        "_Pick an answer above; section 1 will tell you who's right._" if intro_guess.value is None
        else ("✅ **Right.** With 100 equally-informed bidders, the winner is the one whose estimate was the *most* too high. Let's see by how much."
              if "overpaid" in intro_guess.value
              else "🤔 **Hold that thought.** Run the auction simulator below and see where the winning bid lands relative to the green line.")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · The winner's curse

    Everyone at the auction forms a **value belief**: the true value $v$ plus some estimation noise,

    $$b_i = v + \varepsilon_i, \qquad \varepsilon_i \sim \mathcal N(0, \sigma^2), \qquad i = 1,\dots,N.$$

    Nobody is biased: each single estimate is right *on average*. But the auctioneer doesn't take an average, he takes the **maximum**:

    $$\text{winning bid} = \max_i b_i, \qquad E[\max_i b_i] = v + \sigma\, e_N, \qquad e_N = E[\max \text{ of } N \text{ standard normals}] > 0.$$

    $e_N$ grows with $N$ (≈1.87 for 20 bidders, ≈2.51 for 100, ≈3.24 for 1000). The more people bid, the more the winner overpays.

    Press **▶ play**: each frame is one auction. The dots are the $N$ bids, the red one wins, and the histogram below collects winning bids.
    """)
    return


@app.cell
def theory(np, stats):
    def _make():
        _z = np.linspace(-6, 8, 4001)
        _phi, _Phi = stats.norm.pdf(_z), stats.norm.cdf(_z)
        def expected_max_z(n):
            """E[max of n iid standard normals], via numerical integration of z * n*phi*Phi^(n-1)."""
            n = max(int(n), 1)
            return float(np.trapezoid(_z * n * _phi * _Phi ** (n - 1), _z))
        return expected_max_z
    expected_max_z = _make()
    {n: round(expected_max_z(n), 3) for n in [1, 2, 20, 100, 1000]}
    return (expected_max_z,)


@app.cell
def auction_class(anywidget, traitlets):
    class AuctionSim(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 680, H = 330, PAD = 34, XMIN = 60, XMAX = 160, NB = 50;
      const NS = "http://www.w3.org/2000/svg";
      const ctl = document.createElement("div");
      Object.assign(ctl.style, { display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap", marginBottom: "6px", fontFamily: "sans-serif", fontSize: "13px" });
      el.appendChild(ctl);
      const btn = (t, f) => { const b = document.createElement("button"); b.textContent = t; b.onclick = f; b.style.padding = "3px 10px"; ctl.appendChild(b); return b; };
      const lab = (t, min, max, step, key) => {
        const w = document.createElement("label"); w.textContent = t + " ";
        const i = document.createElement("input"); i.type = "range"; i.min = min; i.max = max; i.step = step; i.value = model.get(key);
        const s = document.createElement("b"); s.textContent = i.value;
        i.oninput = () => { s.textContent = i.value; };
        i.onchange = () => { model.set(key, +i.value); reset(); };
        w.append(i, s); ctl.appendChild(w);
      };
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none" });
      el.appendChild(svg);
      const info = document.createElement("div"); Object.assign(info.style, { fontFamily: "sans-serif", fontSize: "13px" }); el.appendChild(info);
      const mk = (tag, a) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); svg.appendChild(e); return e; };
      const sx = v => PAD + (v - XMIN) / (XMAX - XMIN) * (W - 2 * PAD);
      const gauss = () => { let u = 0, v = 0; while (!u) u = Math.random(); v = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
      let timer = null, winners = [], bids = [], counts = new Array(NB).fill(0);
      function step() {
        const n = model.get("n_bidders"), s = model.get("sigma"), v = model.get("value");
        bids = Array.from({ length: n }, () => v + s * gauss());
        const w = Math.max(...bids); winners.push(w);
        const k = Math.floor((w - XMIN) / (XMAX - XMIN) * NB); if (k >= 0 && k < NB) counts[k]++;
        draw();
        if (winners.length % 25 === 0) sync();
      }
      function sync() { model.set("winners", winners.slice(-5000)); model.save_changes(); }
      function draw() {
        svg.innerHTML = "";
        const v = model.get("value");
        mk("text", { x: PAD, y: 16, "font-size": 12, fill: "currentColor" }).textContent = "one auction: every bid (grey) and the winner (red)";
        mk("line", { x1: PAD, x2: W - PAD, y1: 50, y2: 50, stroke: "#999" });
        bids.forEach(b => mk("circle", { cx: sx(b), cy: 50 + (Math.random() - .5) * 22, r: 3.5, fill: "#8d86a8", opacity: .6 }));
        if (bids.length) { const w = Math.max(...bids); mk("circle", { cx: sx(w), cy: 50, r: 7, fill: "#e4572e", stroke: "white", "stroke-width": 1.5 }); }
        const B = 300, T = 100, mx = Math.max(...counts, 1), bw = (W - 2 * PAD) / NB;
        mk("text", { x: PAD, y: T - 6, "font-size": 12, fill: "currentColor" }).textContent = `histogram of winning bids (${winners.length} auctions)`;
        counts.forEach((c, i) => mk("rect", { x: PAD + i * bw + 1, y: B - c / mx * (B - T), width: bw - 2, height: c / mx * (B - T), fill: "#e4572e", opacity: .75 }));
        mk("line", { x1: PAD, x2: W - PAD, y1: B, y2: B, stroke: "#999" });
        for (let t = XMIN; t <= XMAX; t += 10) { mk("text", { x: sx(t), y: B + 16, "font-size": 11, "text-anchor": "middle", fill: "currentColor" }).textContent = t; }
        mk("line", { x1: sx(v), x2: sx(v), y1: 30, y2: B, stroke: "#2a9d8f", "stroke-width": 3, "stroke-dasharray": "6 4" });
        mk("text", { x: sx(v) + 4, y: 30, "font-size": 12, fill: "#2a9d8f" }).textContent = "true value";
        if (winners.length) {
          const m = winners.reduce((a, b) => a + b, 0) / winners.length;
          mk("line", { x1: sx(m), x2: sx(m), y1: T, y2: B, stroke: "#e4572e", "stroke-width": 2.5 });
          info.innerHTML = `average winning bid <b style="color:#e4572e">${m.toFixed(2)}</b> vs true value <b style="color:#2a9d8f">${v}</b>: the winner overpays <b>${(m - v).toFixed(2)}</b> on average`;
        } else info.innerHTML = "press ▶ play";
      }
      function reset() { winners = []; bids = []; counts = new Array(NB).fill(0); draw(); sync(); }
      const play = btn("▶ play", () => {
        if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; sync(); }
        else { timer = setInterval(step, 40); play.textContent = "⏸ pause"; }
      });
      btn("⏩ +500", () => { for (let i = 0; i < 499; i++) { const n = model.get("n_bidders"), s = model.get("sigma"), v = model.get("value");
          let w = -Infinity; for (let j = 0; j < n; j++) w = Math.max(w, v + s * gauss()); winners.push(w);
          const k = Math.floor((w - XMIN) / (XMAX - XMIN) * NB); if (k >= 0 && k < NB) counts[k]++; } step(); sync(); });
      btn("↺ reset", reset);
      lab("bidders N", 1, 1000, 1, "n_bidders");
      lab("noise σ", 1, 15, 1, "sigma");
      draw();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        n_bidders = traitlets.Int(100).tag(sync=True)
        sigma = traitlets.Float(10.0).tag(sync=True)
        value = traitlets.Float(100.0).tag(sync=True)
        winners = traitlets.List(traitlets.Float()).tag(sync=True)

    return (AuctionSim,)


@app.cell
def auction(AuctionSim, mo):
    auction = mo.ui.anywidget(AuctionSim(n_bidders=100, sigma=10.0, value=100.0))
    auction
    return (auction,)


@app.cell
def auction_readout(auction, expected_max_z, mo, np):
    _w = np.array(auction.value["winners"])
    _n, _s, _v = auction.value["n_bidders"], auction.value["sigma"], auction.value["value"]
    _theory = _v + _s * expected_max_z(_n)
    mo.hstack([
        mo.stat(f"{len(_w)}", label="auctions synced to Python"),
        mo.stat(f"{_w.mean():.2f}" if len(_w) else "–", label="simulated mean winning bid"),
        mo.stat(f"{_theory:.2f}", label="theory: v + σ·e_N", caption=f"e_N = {expected_max_z(_n):.3f} for N = {_n}"),
        mo.stat(f"{_theory - _v:+.2f}", label="expected overpayment"),
    ], justify="space-around")
    return


@app.cell
def max_dists(alt, np, pd, stats):
    _z = np.linspace(-3, 6, 600)
    _rows = []
    for _N in [1, 20, 100, 1000]:
        _pdf = _N * stats.norm.pdf(_z) * stats.norm.cdf(_z) ** (_N - 1)
        _rows.append(pd.DataFrame({"winning bid": 100 + 10 * _z, "density": _pdf / 10, "bidders": f"N = {_N}"}))
    _df = pd.concat(_rows)
    _base = alt.Chart(_df).mark_area(opacity=0.35, line=True).encode(
        x=alt.X("winning bid:Q", scale=alt.Scale(domain=[70, 160])),
        y=alt.Y("density:Q", stack=None),
        color=alt.Color("bidders:N", sort=["N = 1", "N = 20", "N = 100", "N = 1000"],
                        scale=alt.Scale(range=["#8d86a8", "#f4a261", "#e4572e", "#9b2226"])),
    )
    _rule = alt.Chart(pd.DataFrame({"x": [100]})).mark_rule(color="#2a9d8f", strokeWidth=3, strokeDash=[6, 4]).encode(x="x:Q")
    (_base + _rule).properties(width="container", height=260, title="Distribution of the winning bid (v = 100, σ = 10): exact density N·φ·Φ^(N−1)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Set **N = 1**: the "winner" is just one honest estimate, and the red line sits on the green one. No curse.
    2. Go to **N = 20**, then **100**, then **1000**. The histogram slides right *and* gets narrower, just like the video's animation (and the exact curves above).
    3. Set **σ = 1**. The overpayment shrinks 10×: the curse is *noise × selection*. No noise, no curse.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · The remedy: shade your bid

    The video's notebook has sliders for the **actual value**, the bidders' belief **μ, σ**, and the **number of candidates N**.
    If (big if) you can quantify your own doubt $\hat\sigma$ and guess how many people are bidding, the theory hands you the expected
    overpayment, so you can subtract it up front:

    $$\text{shaded bid} = b - \hat\sigma \, e_N .$$

    It is a function from *number of candidates → overpayment*, which you then invert to bring the overpayment **back to zero**.
    You'll be more conservative and lose some auctions, but the auctions you *win* will no longer be at a loss on average.
    """)
    return


@app.cell
def remedy_controls(mo):
    true_value = mo.ui.slider(50, 150, value=100, label="actual value of the item", show_value=True)
    belief_mu = mo.ui.slider(50, 150, value=100, label="μ of everyone's belief", show_value=True)
    belief_sigma = mo.ui.slider(1, 30, value=10, label="σ of everyone's belief", show_value=True)
    n_candidates = mo.ui.slider(1, 300, value=50, label="number of candidates N", show_value=True)
    believed_noise = mo.ui.slider(0, 30, step=0.5, value=0, label="your believed noise σ̂ (for shading)", show_value=True)
    reroll = mo.ui.button(label="🎲 new auction", value=0, on_click=lambda v: v + 1)
    mo.hstack([mo.vstack([true_value, belief_mu, belief_sigma]), mo.vstack([n_candidates, believed_noise, reroll])])
    return (
        belief_mu,
        belief_sigma,
        believed_noise,
        n_candidates,
        reroll,
        true_value,
    )


@app.cell
def one_auction_view(
    alt,
    belief_mu,
    belief_sigma,
    believed_noise,
    expected_max_z,
    mo,
    n_candidates,
    np,
    pd,
    reroll,
    true_value,
):
    _rng = np.random.default_rng(reroll.value)
    _bids = _rng.normal(belief_mu.value, belief_sigma.value, n_candidates.value)
    _e = expected_max_z(n_candidates.value)
    _w = _bids.max()
    _df = pd.DataFrame({"bid": _bids, "who": "bidder", "y": _rng.uniform(-1, 1, len(_bids))})
    _df.loc[_df.bid.idxmax(), "who"] = "winner"
    _pts = alt.Chart(_df).mark_circle(size=70).encode(
        x=alt.X("bid:Q", scale=alt.Scale(domain=[min(30, _bids.min() - 5), max(190, _w + 5)])),
        y=alt.Y("y:Q", axis=None, scale=alt.Scale(domain=[-1.5, 1.5])),
        color=alt.Color("who:N", scale=alt.Scale(domain=["bidder", "winner"], range=["#8d86a8", "#e4572e"]), legend=None),
        size=alt.condition("datum.who == 'winner'", alt.value(220), alt.value(60)),
        tooltip=["bid:Q", "who:N"],
    )
    _marks = pd.DataFrame({"x": [true_value.value, _w - believed_noise.value * _e],
                           "what": ["actual value", "winner's shaded bid"]})
    _rules = alt.Chart(_marks).mark_rule(strokeWidth=3, strokeDash=[6, 4]).encode(
        x="x:Q", color=alt.Color("what:N", scale=alt.Scale(domain=["actual value", "winner's shaded bid"], range=["#2a9d8f", "#1d70b8"])))
    one_auction = (_pts + _rules).properties(width="container", height=170, title="One auction: each dot is a bid")
    mo.vstack([one_auction, mo.hstack([
        mo.stat(f"{_w:.1f}", label="winning bid"),
        mo.stat(f"{_w - true_value.value:+.1f}", label="winner's overpayment"),
        mo.stat(f"{belief_sigma.value * _e:.1f}", label="expected overpayment σ·e_N", caption="if μ = actual value"),
        mo.stat(f"{_w - believed_noise.value * _e - true_value.value:+.1f}", label="overpayment after shading"),
    ], justify="space-around")])
    return


@app.cell
def overpay_curve(
    alt,
    belief_mu,
    belief_sigma,
    believed_noise,
    expected_max_z,
    np,
    pd,
    true_value,
):
    _rng = np.random.default_rng(1)
    _Ns = np.unique(np.round(np.geomspace(1, 300, 25)).astype(int))
    _rows = []
    for _N in _Ns:
        _wins = _rng.normal(belief_mu.value, belief_sigma.value, (2000, _N)).max(axis=1)
        _rows.append({"N": _N, "naive": _wins.mean() - true_value.value,
                      "shaded": (_wins - believed_noise.value * expected_max_z(_N)).mean() - true_value.value})
    _df = pd.DataFrame(_rows).melt("N", var_name="strategy", value_name="avg overpayment")
    _lines = alt.Chart(_df).mark_line(point=True).encode(
        x=alt.X("N:Q", scale=alt.Scale(type="log"), title="number of candidates N (log)"),
        y="avg overpayment:Q",
        color=alt.Color("strategy:N", scale=alt.Scale(domain=["naive", "shaded"], range=["#e4572e", "#1d70b8"])),
    )
    _zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#2a9d8f", strokeWidth=3, strokeDash=[6, 4]).encode(y="y:Q")
    (_lines + _zero).properties(width="container", height=260, title="Average overpayment of the winner (2000 simulated auctions per N)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Drag **N** up and press 🎲 a few times: the red dot lives in the right tail, above the green line, nearly every time.
    2. Now set **σ̂ = σ** (10 by default). The blue *shaded* curve collapses onto the zero line for every N. That's the
       "numerically verify" moment from the video.
    3. Set σ̂ larger than σ: you now *under*bid. Shading buys safety with conservatism.
    4. The practical remedy: **make N small.** The video's house was bought over Christmas, when few others were bidding. Drag N to 1–3 and watch the overpayment vanish.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · The optimizer's curse: a dice game

    Player **A** and player **B** each roll a d20 to see who is "better at rolling". Over many rounds they're equally good.
    Then B starts to cheat: he rolls **k** dice and only reports the **maximum**. The maximum of $k$ d20 rolls has

    $$P(M \le m) = \left(\tfrac{m}{20}\right)^k \quad\Rightarrow\quad P(M = m) = \left(\tfrac{m}{20}\right)^k - \left(\tfrac{m-1}{20}\right)^k .$$

    For $k=1$ the distribution is flat; for $k=2$ it's a **staircase** that walks up to 20; for $k=100$ it's "practically impossible not to roll a 20".
    Nothing about B's dice changed. Only the **reporting rule** did.
    """)
    return


@app.cell
def dice_class(anywidget, traitlets):
    class DiceDuel(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 680, H = 280, PAD = 30;
      const NS = "http://www.w3.org/2000/svg";
      const ctl = document.createElement("div");
      Object.assign(ctl.style, { display: "flex", gap: "8px", alignItems: "center", flexWrap: "wrap", marginBottom: "6px", fontFamily: "sans-serif", fontSize: "13px" });
      el.appendChild(ctl);
      const btn = (t, f) => { const b = document.createElement("button"); b.textContent = t; b.onclick = f; b.style.padding = "3px 10px"; ctl.appendChild(b); return b; };
      const kl = document.createElement("label"); kl.textContent = "B rolls k dice, reports max: ";
      const ks = document.createElement("select");
      [1, 2, 3, 5, 10, 100].forEach(k => { const o = document.createElement("option"); o.value = k; o.textContent = k; ks.appendChild(o); });
      ks.value = model.get("k"); ks.onchange = () => { model.set("k", +ks.value); reset(); };
      kl.appendChild(ks);
      const dice = document.createElement("div");
      Object.assign(dice.style, { fontFamily: "monospace", fontSize: "15px", minHeight: "22px", margin: "4px 0" });
      const svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none" });
      const info = document.createElement("div"); Object.assign(info.style, { fontFamily: "sans-serif", fontSize: "13px" });
      el.append(dice, svg, info);
      const mk = (tag, a) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); svg.appendChild(e); return e; };
      const d20 = () => 1 + Math.floor(Math.random() * 20);
      let timer = null, A = new Array(20).fill(0), B = new Array(20).fill(0), wins = 0, ties = 0, n = 0;
      function roll(show) {
        const k = model.get("k"), a = d20(), bs = Array.from({ length: k }, d20), b = Math.max(...bs);
        A[a - 1]++; B[b - 1]++; n++; if (b > a) wins++; else if (b === a) ties++;
        if (show) dice.innerHTML = `A: <b style="color:#8d86a8">${a}</b> &nbsp; B: ${k <= 10 ? bs.map(x => x === b ? `<b style="color:#e4572e">${x}</b>` : `<span style="opacity:.5">${x}</span>`).join(" ") : `(${k} dice) max = <b style="color:#e4572e">${b}</b>`}`;
      }
      function sync() { model.set("counts_a", A); model.set("counts_b", B); model.save_changes(); }
      function draw() {
        svg.innerHTML = "";
        const half = (W - 3 * PAD) / 2, T = 30, Bt = H - 30;
        [[A, "Player A (1 die)", "#8d86a8", PAD], [B, `Player B (max of ${model.get("k")})`, "#e4572e", 2 * PAD + half]].forEach(([c, title, col, x0]) => {
          const mx = Math.max(...c, 1), bw = half / 20;
          mk("text", { x: x0, y: 16, "font-size": 12, fill: "currentColor" }).textContent = title;
          c.forEach((v, i) => {
            mk("rect", { x: x0 + i * bw + 1, y: Bt - v / mx * (Bt - T), width: bw - 2, height: v / mx * (Bt - T), fill: col, opacity: .8 });
            if (i % 4 === 0 || i === 19) mk("text", { x: x0 + (i + .5) * bw, y: Bt + 14, "font-size": 10, "text-anchor": "middle", fill: "currentColor" }).textContent = i + 1;
          });
          const mean = c.reduce((s, v, i) => s + v * (i + 1), 0) / Math.max(n, 1);
          if (n) { const xm = x0 + (mean - .5) * bw; mk("line", { x1: xm, x2: xm, y1: T - 4, y2: Bt, stroke: "currentColor", "stroke-width": 2, "stroke-dasharray": "4 3" });
            mk("text", { x: xm + 3, y: T + 6, "font-size": 11, fill: "currentColor" }).textContent = "mean " + mean.toFixed(2); }
        });
        info.innerHTML = n ? `${n} rounds · B "wins" <b style="color:#e4572e">${(100 * wins / n).toFixed(1)}%</b>, ties ${(100 * ties / n).toFixed(1)}%, A wins ${(100 * (n - wins - ties) / n).toFixed(1)}%` : "press ▶ play";
      }
      function reset() { A = new Array(20).fill(0); B = new Array(20).fill(0); wins = ties = n = 0; dice.innerHTML = ""; draw(); sync(); }
      const play = btn("▶ play", () => {
        if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; sync(); }
        else { timer = setInterval(() => { roll(true); draw(); if (n % 20 === 0) sync(); }, 60); play.textContent = "⏸ pause"; }
      });
      btn("⏩ +1000", () => { for (let i = 0; i < 1000; i++) roll(i === 999); draw(); sync(); });
      btn("↺ reset", reset);
      ctl.appendChild(kl);
      draw();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        k = traitlets.Int(2).tag(sync=True)
        counts_a = traitlets.List(traitlets.Int(), default_value=[0] * 20).tag(sync=True)
        counts_b = traitlets.List(traitlets.Int(), default_value=[0] * 20).tag(sync=True)

    return (DiceDuel,)


@app.cell
def dice(DiceDuel, mo):
    dice = mo.ui.anywidget(DiceDuel(k=2))
    dice
    return (dice,)


@app.cell
def dice_readout(alt, dice, mo, np, pd):
    def max_d20_pmf(k):
        """Exact P(max of k d20 = m) for m = 1..20."""
        _m = np.arange(1, 21)
        return (_m / 20) ** k - ((_m - 1) / 20) ** k

    _k = dice.value["k"]
    _cb = np.array(dice.value["counts_b"])
    _df = pd.DataFrame({"face": np.arange(1, 21), "exact": max_d20_pmf(_k),
                        "observed": _cb / max(_cb.sum(), 1)})
    _bars = alt.Chart(_df).mark_bar(color="#e4572e", opacity=0.6).encode(x="face:O", y=alt.Y("observed:Q", title="probability"))
    _tick = alt.Chart(_df).mark_tick(color="currentColor", thickness=3, size=22).encode(x="face:O", y="exact:Q", tooltip=["face", alt.Tooltip("exact:Q", format=".3f")])
    _exp = float((np.arange(1, 21) * max_d20_pmf(_k)).sum())
    mo.vstack([
        (_bars + _tick).properties(width="container", height=200, title=f"Player B, k = {_k}: observed (bars) vs exact pmf (ticks)"),
        mo.hstack([mo.stat("10.50", label="E[A's roll]"), mo.stat(f"{_exp:.2f}", label=f"E[max of {_k} d20]"),
                   mo.stat(f"{max_d20_pmf(_k)[-1]:.1%}", label="P(B reports a 20)"),
                   mo.stat(f"{int(_cb.sum())}", label="rounds synced")], justify="space-around"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Start with **k = 1** and ⏩ a few thousand rounds: two flat histograms, and B wins ~47.5% (ties take the rest). Fair.
    2. Switch to **k = 2**: the staircase appears and B's mean jumps from 10.5 to ~13.8.
    3. Try **k = 100**: P(B reports a 20) ≈ 99.4%. B isn't a better roller. B is just *choosing the max of noise*.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Is this machine learning? Yes: hyperparameter search

    A randomized search tries `n_iter` settings and reports the **best cross-validation score**. Every CV score is
    "true quality + noise" (which folds, which random seed, which shuffle). Picking the max is exactly player B rolling more dice:

    $$\text{reported score} = \max_{j \le n_\text{iter}} \big(q_j + \varepsilon_j\big) \quad\text{tends to exceed}\quad q_{j^*} .$$

    So with more iterations you may not be optimizing the model; you may be **optimizing the random number generator**.

    First a pure version: imagine **every configuration is equally good** (true accuracy 0.80) and CV adds a bit of noise.
    """)
    return


@app.cell
def pure_controls(mo):
    n_configs = mo.ui.slider(1, 1000, value=50, label="settings tried (n_iter)", show_value=True)
    cv_noise = mo.ui.slider(0.0, 0.05, step=0.0025, value=0.015, label="CV noise (sd of a score)", show_value=True)
    mo.hstack([n_configs, cv_noise])
    return cv_noise, n_configs


@app.cell
def pure_search(alt, cv_noise, expected_max_z, mo, n_configs, np, pd):
    _rng = np.random.default_rng(7)
    _reps = 400
    _cv = 0.80 + cv_noise.value * _rng.standard_normal((_reps, n_configs.value))
    _best_cv = _cv.max(axis=1)
    _test = 0.80 + cv_noise.value * _rng.standard_normal(_reps)  # fresh holdout of the chosen config
    _df = pd.concat([pd.DataFrame({"score": _best_cv, "what": "reported best CV score"}),
                     pd.DataFrame({"score": _test, "what": "same model on fresh test data"})])
    _h = alt.Chart(_df).mark_bar(opacity=0.6, binSpacing=0).encode(
        x=alt.X("score:Q", bin=alt.Bin(maxbins=50), scale=alt.Scale(domain=[0.7, 0.95])),
        y=alt.Y("count():Q", stack=None),
        color=alt.Color("what:N", scale=alt.Scale(domain=["reported best CV score", "same model on fresh test data"], range=["#e4572e", "#1d70b8"])),
    )
    _r = alt.Chart(pd.DataFrame({"x": [0.80]})).mark_rule(color="#2a9d8f", strokeWidth=3, strokeDash=[6, 4]).encode(x="x:Q")
    mo.vstack([
        (_h + _r).properties(width="container", height=220, title=f"{_reps} repeated searches, all configs truly 0.80"),
        mo.hstack([mo.stat("0.800", label="true accuracy of every config"),
                   mo.stat(f"{_best_cv.mean():.3f}", label="avg reported best CV"),
                   mo.stat(f"{0.80 + cv_noise.value * expected_max_z(n_configs.value):.3f}", label="theory 0.80 + sd·e_n"),
                   mo.stat(f"{_test.mean():.3f}", label="avg on fresh data")], justify="space-around"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. With **n_iter = 1** the red and blue histograms coincide: no selection, no curse.
    2. Push **n_iter → 1000**: the "best" score climbs to ~0.85 although *nothing* is better than 0.80. The blue holdout doesn't move.
    3. Shrink the CV noise to 0 (lots of data, many folds): the curse disappears. Noise × selection again.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A real search, with the video's fix: add `random_state` as a hyperparameter

    Below is a genuine scikit-learn `RandomizedSearchCV` on an **imbalanced** dataset with a logistic-regression model
    (`SGDClassifier(loss="log_loss")`). Its `random_state` controls the SGD shuffle, so it *should be pure noise*.
    That makes it a **control**: any hyperparameter whose pattern in the top scores looks just like `random_state` is
    probably noise too.

    The parallel-coordinates chart puts each setting on its own vertical axis (rescaled to 0–1) and the CV score on the right.
    Use the range slider to highlight only the high scores and look for axes where the red lines **bunch up**.
    """)
    return


@app.cell
def n_iter(mo):
    n_iter = mo.ui.slider(10, 400, step=10, value=120, label="n_iter of RandomizedSearchCV", show_value=True)
    n_iter
    return (n_iter,)


@app.cell
def search(mo, n_iter, np, pd):
    from sklearn.datasets import make_classification
    from sklearn.linear_model import SGDClassifier
    from sklearn.model_selection import RandomizedSearchCV, train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import loguniform, randint
    import warnings

    _X, _y = make_classification(n_samples=800, n_features=12, n_informative=4, weights=[0.85, 0.15], flip_y=0.03, random_state=0)
    _Xtr, _Xte, _ytr, _yte = train_test_split(_X, _y, test_size=0.4, stratify=_y, random_state=0)
    _space = {
        "sgdclassifier__alpha": loguniform(1e-5, 1e1),
        "sgdclassifier__class_weight": [None, "balanced"],
        "sgdclassifier__max_iter": [5, 10, 20, 50, 100],
        "sgdclassifier__penalty": ["l2", "l1"],
        "sgdclassifier__random_state": randint(0, 10_000),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search = RandomizedSearchCV(
            make_pipeline(StandardScaler(), SGDClassifier(loss="log_loss", tol=None)),
            _space, n_iter=n_iter.value, cv=5, scoring="accuracy", random_state=42, n_jobs=1,
        ).fit(_Xtr, _ytr)
    search_test_score = search.score(_Xte, _yte)
    search_results = pd.DataFrame({
        "log10 alpha": np.log10(search.cv_results_["param_sgdclassifier__alpha"].astype(float)),
        "class_weight": [str(v) for v in search.cv_results_["param_sgdclassifier__class_weight"]],
        "max_iter": search.cv_results_["param_sgdclassifier__max_iter"].astype(int),
        "penalty": search.cv_results_["param_sgdclassifier__penalty"].astype(str),
        "random_state": search.cv_results_["param_sgdclassifier__random_state"].astype(int),
        "cv score": search.cv_results_["mean_test_score"],
    })
    mo.hstack([mo.stat(f"{search.best_score_:.3f}", label="best CV score (what gets reported)"),
               mo.stat(f"{search_test_score:.3f}", label="that model on held-out test set"),
               mo.stat(f"{search.best_score_ - search_test_score:+.3f}", label="optimism"),
               mo.stat(f"{(search_results['cv score'] >= search_results['cv score'].max() - 0.01).sum()}", label="configs within 0.01 of best")],
              justify="space-around")
    return (search_results,)


@app.cell
def score_floor(mo, np, search_results):
    score_floor = mo.ui.range_slider(
        start=float(np.floor(search_results["cv score"].min() * 100) / 100), stop=float(np.ceil(search_results["cv score"].max() * 100) / 100),
        step=0.005, value=[float(search_results["cv score"].quantile(0.8)), float(np.ceil(search_results["cv score"].max() * 100) / 100)],
        label="highlight configs with CV score in", show_value=True, full_width=True)
    score_floor
    return (score_floor,)


@app.cell
def parallel_coords(alt, mo, pd, score_floor, search_results):
    _df = search_results.copy()
    _df["id"] = range(len(_df))
    _axes = ["log10 alpha", "class_weight", "max_iter", "penalty", "random_state", "cv score"]
    _num = pd.DataFrame({"id": _df["id"]})
    for _a in _axes:
        _col = _df[_a]
        if not pd.api.types.is_numeric_dtype(_col):
            _codes = {v: i for i, v in enumerate(sorted(_col.unique()))}
            _col = _col.map(_codes).astype(float)
        _num[_a] = (_col - _col.min()) / ((_col.max() - _col.min()) or 1)
    _num["selected"] = _df["cv score"].between(*score_floor.value)
    _long = _num.melt(["id", "selected"], var_name="axis", value_name="scaled")
    _long = _long.merge(_df.melt("id", value_vars=_axes, var_name="axis", value_name="raw").astype({"raw": str}), on=["id", "axis"])
    _pc = alt.Chart(_long).mark_line(strokeWidth=1.3).encode(
        x=alt.X("axis:N", sort=_axes, title=None, axis=alt.Axis(labelAngle=0, labelFontSize=12)),
        y=alt.Y("scaled:Q", title="rescaled (0 = min, 1 = max)"),
        detail="id:N",
        color=alt.condition("datum.selected", alt.value("#e4572e"), alt.value("#8d86a855")),
        opacity=alt.condition("datum.selected", alt.value(0.9), alt.value(0.25)),
        order=alt.Order("selected:Q"),
        tooltip=["axis:N", "raw:N"],
    )
    _ticks = alt.Chart(pd.DataFrame({"axis": _axes})).mark_rule(color="#999").encode(x=alt.X("axis:N", sort=_axes))
    mo.vstack([
        (_ticks + _pc).properties(width="container", height=330, title="Parallel coordinates of the search (red = highlighted CV range)"),
        mo.md(f"**{int(_num['selected'].sum())}** of {len(_num)} configs highlighted. "
              f"Among them: class_weight=None in **{(_df.loc[_num['selected'], 'class_weight'] == 'None').mean():.0%}**, "
              f"median log10 alpha **{_df.loc[_num['selected'], 'log10 alpha'].median():.2f}** (all: {_df['log10 alpha'].median():.2f}), "
              f"median random_state **{_df.loc[_num['selected'], 'random_state'].median():.0f}** (all: {_df['random_state'].median():.0f})."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Highlight just the top band of CV scores. Does `class_weight` bunch up? (On imbalanced data with *accuracy* as the metric, `None` tends to win, as in the video.)
    2. Compare `log10 alpha` with `random_state`. Alpha shows a clear region; random_state is spread "all over the place". Any axis that looks like random_state, mentally drop it.
    3. Raise **n_iter** and watch the *optimism* stat: the reported best tends to creep up while the test score does not. Report the **distribution** of good-region scores, not the single max.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Sandbox: options that really differ, and a Bayesian remedy

    So far everyone was equally good. In real life the options **do** differ: true values $q_j \sim \mathcal N(\mu_0, \tau^2)$, and we see
    noisy estimates $x_j = q_j + \varepsilon_j$, $\varepsilon_j \sim \mathcal N(0, \sigma_j^2)$. We pick the largest estimate and
    expect to get $x_{j^*}$. Smith & Winkler (2006), who named the *optimizer's curse*, show the fix is to **shrink** every
    estimate toward the prior before choosing:

    $$\hat q_j = \mu_0 + \frac{\tau^2}{\tau^2 + \sigma_j^2}\,(x_j - \mu_0).$$

    Noisier estimates get shrunk harder, so with *unequal* noise, shrinkage can even change **which** option you pick.
    Each dot is one option in one simulated decision; red = the naive pick, blue = the shrunk pick.
    """)
    return


@app.cell
def sandbox_controls(mo):
    sb_n = mo.ui.slider(2, 200, value=20, label="options to choose from", show_value=True)
    sb_tau = mo.ui.slider(0.0, 3.0, step=0.1, value=1.0, label="τ: real spread of true values", show_value=True)
    sb_sigma = mo.ui.slider(0.1, 3.0, step=0.1, value=1.0, label="σ: typical estimation noise", show_value=True)
    sb_unequal = mo.ui.switch(value=True, label="unequal noise per option (σ_j between 0.2σ and 2σ)")
    mo.vstack([mo.hstack([sb_n, sb_tau]), mo.hstack([sb_sigma, sb_unequal])])
    return sb_n, sb_sigma, sb_tau, sb_unequal


@app.cell
def sandbox(alt, mo, np, pd, sb_n, sb_sigma, sb_tau, sb_unequal):
    def simulate_decisions(n, tau, sigma, unequal, reps=1000, seed=3):
        """Returns per-decision arrays: naive pick (estimate, true) and shrunk pick (estimate, true)."""
        _rng = np.random.default_rng(seed)
        _q = tau * _rng.standard_normal((reps, n))
        _s = sigma * (_rng.uniform(0.2, 2.0, (reps, n)) if unequal else np.ones((reps, n)))
        _x = _q + _s * _rng.standard_normal((reps, n))
        _shr = (tau ** 2 / (tau ** 2 + _s ** 2)) * _x  # mu0 = 0
        _r = np.arange(reps)
        _a, _b = _x.argmax(1), _shr.argmax(1)
        return dict(q=_q, x=_x, naive_est=_x[_r, _a], naive_true=_q[_r, _a], shr_est=_shr[_r, _b], shr_true=_q[_r, _b])

    sim = simulate_decisions(sb_n.value, sb_tau.value, sb_sigma.value, sb_unequal.value)
    _k = 60  # plot the first 60 decisions only
    _pts = pd.DataFrame({"estimate": sim["x"][:_k].ravel(), "true value": sim["q"][:_k].ravel(), "kind": "option"})
    _pts = pd.concat([_pts,
        pd.DataFrame({"estimate": sim["naive_est"][:_k], "true value": sim["naive_true"][:_k], "kind": "naive pick (estimate x)"}),
        pd.DataFrame({"estimate": sim["shr_est"][:_k], "true value": sim["shr_true"][:_k], "kind": "shrunk pick (estimate q̂)"})])
    _lim = float(np.abs(_pts[["estimate", "true value"]]).max().max()) + 0.5
    _sc = alt.Chart(_pts).mark_circle().encode(
        x=alt.X("estimate:Q", scale=alt.Scale(domain=[-_lim, _lim])),
        y=alt.Y("true value:Q", scale=alt.Scale(domain=[-_lim, _lim])),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["option", "naive pick (estimate x)", "shrunk pick (estimate q̂)"], range=["#8d86a8", "#e4572e", "#1d70b8"])),
        size=alt.condition("datum.kind == 'option'", alt.value(12), alt.value(70)),
        opacity=alt.condition("datum.kind == 'option'", alt.value(0.25), alt.value(0.9)),
    )
    _diag = alt.Chart(pd.DataFrame({"a": [-_lim, _lim]})).mark_line(color="#2a9d8f", strokeDash=[6, 4], strokeWidth=2).encode(x="a:Q", y="a:Q")
    mo.vstack([
        (_sc + _diag).properties(width="container", height=360, title="Estimate vs truth (green diagonal = honest estimate). Points right of it are optimistic."),
        mo.hstack([
            mo.stat(f"{(sim['naive_est'] - sim['naive_true']).mean():+.2f}", label="naive disappointment", caption="avg (expected − realized)"),
            mo.stat(f"{(sim['shr_est'] - sim['shr_true']).mean():+.2f}", label="shrunk disappointment", caption="≈ 0 is the goal"),
            mo.stat(f"{sim['naive_true'].mean():.2f}", label="true value of naive pick"),
            mo.stat(f"{sim['shr_true'].mean():.2f}", label="true value of shrunk pick"),
        ], justify="space-around"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Challenges**

    1. Set **τ = 0** (all options identical). The naive pick looks great but the truth is 0. Shrinkage estimates exactly 0: it knows there's nothing to find.
    2. Set **σ ≪ τ** (σ = 0.1, τ = 2). Little noise, so the curse is tiny and shrinkage barely does anything.
    3. With **unequal noise on**, compare the two *true value* stats: shrinkage doesn't just fix the forecast, it picks **better options**, because the naive rule keeps chasing the noisiest estimate.
    4. Crank **options to 200**: the naive disappointment grows like σ·e_N; the blue one stays ≈ 0.
    """), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Quiz
    Answer all six, then flip the switch to score yourself.
    """)
    return


@app.cell
def quiz(mo):
    quiz_items = {
        "1. 100 honest bidders, each unbiased. The winning bid is, on average…": (["below the true value", "equal to the true value", "above the true value"], "above the true value"),
        "2. Going from 20 to 1000 bidders (same σ), the winning-bid distribution…": (["moves right and gets narrower", "moves right and gets wider", "doesn't change"], "moves right and gets narrower"),
        "3. If everyone's estimates had zero noise (σ = 0), the winner's curse would…": (["get worse", "disappear", "stay the same"], "disappear"),
        "4. Player B reports the max of 2 d20 rolls. Their reported-roll distribution is…": (["flat", "a staircase going up to 20", "bell-shaped around 10.5"], "a staircase going up to 20"),
        "5. In a hyperparameter search, a parameter whose top-score lines look just like random_state's is probably…": (["the most important one", "noise for model choice", "a bug"], "noise for model choice"),
        "6. Which is NOT a remedy discussed here?": (["shade your bid by σ̂·e_N", "keep the number of candidates small", "try more hyperparameter settings and report the max", "shrink estimates toward a prior"], "try more hyperparameter settings and report the max"),
    }
    quiz = mo.ui.dictionary({_q: mo.ui.radio(_opts) for _q, (_opts, _a) in quiz_items.items()})
    reveal = mo.ui.switch(label="reveal answers")
    mo.vstack([quiz, reveal])
    return quiz, quiz_items, reveal


@app.cell
def quiz_score(mo, quiz, quiz_items, reveal):
    _rows = []
    for _q, (_opts, _a) in quiz_items.items():
        _g = quiz.value[_q]
        _rows.append((_q, _g, _a, _g == _a))
    _score = sum(r[3] for r in _rows)
    mo.stop(not reveal.value, mo.md(f"_Answered {sum(r[1] is not None for r in _rows)}/{len(_rows)}. Flip **reveal answers** when ready._"))
    mo.vstack([
        mo.md(f"### Score: {_score}/{len(_rows)} {'🏆' if _score == len(_rows) else ''}"),
        mo.md("\n".join(f"- {'✅' if _ok else '❌'} **{_q}** your answer: *{_g}*; correct: **{_a}**" for _q, _g, _a, _ok in _rows)),
    ])
    return


@app.cell(hide_code=True)
def recap(mo):
    mo.md(r"""
    ## Recap

    | Idea | What happens | Formula / rule | Remedy |
    |---|---|---|---|
    | **Winner's curse** | The highest of N unbiased bids is biased upward | $E[\max b_i] = v + \sigma e_N$ | Shade: bid $b - \hat\sigma e_N$ |
    | **More candidates** | Winning bid moves right (and narrows) | $e_{20}≈1.87,\ e_{100}≈2.51,\ e_{1000}≈3.24$ | Bid when few others do (Christmas!) |
    | **Dice game** | Max of k rolls is a staircase toward 20 | $P(M=m)=(m/20)^k-((m-1)/20)^k$ | Don't let the reporter pick the max |
    | **Optimizer's curse (ML)** | Best CV score of n_iter configs overstates holdout | max of (quality + noise) | Add `random_state` as a control, read parallel coords, report a distribution |
    | **Shrinkage** | Noisy estimates pulled toward the prior | $\hat q = \mu_0 + \frac{\tau^2}{\tau^2+\sigma^2}(x-\mu_0)$ | Removes expected disappointment, picks better options |

    **One sentence:** whenever you *select the maximum of noisy numbers*, expect to be disappointed, by about **noise × e_N**.
    """)
    return


if __name__ == "__main__":
    app.run()

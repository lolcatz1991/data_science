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
    # p-values: what they are and how to interpret them
    *An interactive companion to StatQuest's **"p-values: What they are and how to interpret them"** (Josh Starmer).*

    > Drug A cured 37% of people, drug B cured 31%. Is drug A actually better, or did a few weird random things
    > (a rare allergy here, a strong placebo effect there) just land unevenly? **The p-value is the number that helps decide.**
    """),
        mo.Html('<iframe width="560" height="315" src="https://www.youtube.com/embed/vemZtEM63GY" title="p-values" frameborder="0" allowfullscreen></iframe>'),
        mo.md(r"""
    **Color language used throughout:** <span style="color:#f18f01">**orange = drug A**</span>,
    <span style="color:#2e86ab">**blue = drug B**</span>, <span style="color:#e4572e">**red = p < threshold ("significant")**</span>,
    <span style="color:#999">**grey = not significant**</span>.

    The notebook follows the video: (1) why one patient isn't enough, (2) the p-value under "no difference" and the 0.05 threshold,
    (3) choosing other thresholds, (4) what p-values look like when there *is* a difference, (5) p-values don't measure effect size,
    plus a bonus on misconceptions and **p-hacking**, a sandbox, a quiz and a recap.
    """),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Why one person per drug tells us nothing

    The video starts with **one** person on drug A (cured, hooray!) and **one** on drug B (not cured, bummer).
    Maybe drug B's patient had an allergy, missed a dose, or drug A's patient just had a placebo effect.
    **Weird random things happen**, so we need many people. But even with many people, how big must the gap be before we stop
    blaming chance?

    To answer that, we compute a **p-value** for the 2×2 table (here with Fisher's exact test):

    | | cured | not cured |
    |---|---|---|
    | drug A | $a$ | $b$ |
    | drug B | $c$ | $d$ |

    Pick one of the video's experiments, or set your own true cure rates and run a fresh trial.
    """)
    return


@app.cell
def trial_controls(mo):
    trial_preset = mo.ui.dropdown(
        options=["Video: 1 person each", "Video: 2 people each", "Video: 1043/3 vs 2/1432", "Video: 37% vs 31%", "Simulate my own trial"],
        value="Video: 37% vs 31%", label="Experiment",
    )
    trial_n = mo.ui.slider(1, 2000, value=100, label="people per drug", show_value=True)
    trial_pa = mo.ui.slider(0, 100, value=40, label="true cure rate A (%)", show_value=True)
    trial_pb = mo.ui.slider(0, 100, value=30, label="true cure rate B (%)", show_value=True)
    trial_rerun = mo.ui.run_button(label="🎲 run a fresh trial")
    mo.vstack([trial_preset, mo.hstack([trial_n, trial_pa, trial_pb]), trial_rerun])
    return trial_n, trial_pa, trial_pb, trial_preset, trial_rerun


@app.cell
def trial_calc(
    np,
    stats,
    trial_n,
    trial_pa,
    trial_pb,
    trial_preset,
    trial_rerun,
):
    def fisher_p(a, b, c, d):
        """Two-sided Fisher exact p-value for the table [[a, b], [c, d]] (A cured/not, B cured/not)."""
        return float(stats.fisher_exact([[a, b], [c, d]])[1])

    trial_rerun
    _presets = {
        "Video: 1 person each": (1, 0, 0, 1),
        "Video: 2 people each": (2, 0, 1, 1),
        "Video: 1043/3 vs 2/1432": (1043, 3, 2, 1432),
        "Video: 37% vs 31%": (73, 125, 59, 131),
    }
    if trial_preset.value in _presets:
        trial_table = _presets[trial_preset.value]
    else:
        _rng = np.random.default_rng()
        _a = int(_rng.binomial(trial_n.value, trial_pa.value / 100))
        _c = int(_rng.binomial(trial_n.value, trial_pb.value / 100))
        trial_table = (_a, trial_n.value - _a, _c, trial_n.value - _c)
    trial_p = fisher_p(*trial_table)
    return fisher_p, trial_p, trial_table


@app.cell
def trial_view(alt, mo, pd, trial_p, trial_table):
    _a, _b, _c, _d = trial_table
    _df = pd.DataFrame({
        "drug": ["A", "A", "B", "B"],
        "outcome": ["cured", "not cured", "cured", "not cured"],
        "people": [_a, _b, _c, _d],
    })
    _chart = alt.Chart(_df).mark_bar().encode(
        y=alt.Y("drug:N", title=None),
        x=alt.X("people:Q", stack="normalize", title="share of people", axis=alt.Axis(format="%")),
        color=alt.Color("outcome:N", scale=alt.Scale(domain=["cured", "not cured"], range=["#76b041", "#ddd"]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["drug", "outcome", "people"],
    ).properties(width="container", height=110)
    _ra, _rb = _a / max(_a + _b, 1), _c / max(_c + _d, 1)
    _sig = trial_p < 0.05
    mo.vstack([
        mo.hstack([
            mo.stat(f"{_ra:.1%}", label="cured on drug A", caption=f"{_a} of {_a+_b}"),
            mo.stat(f"{_rb:.1%}", label="cured on drug B", caption=f"{_c} of {_c+_d}"),
            mo.stat(f"{trial_p:.3g}", label="p-value", caption="< 0.05: confident they differ" if _sig else "≥ 0.05: can't rule out chance",
                    direction="increase" if _sig else None),
        ], justify="start", gap=2),
        _chart,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. *1 person each*: a 100% vs 0% cure rate, yet p = 1. One person can't distinguish a drug from bad luck.
    2. *1043/3 vs 2/1432*: p is essentially 0. Nobody believes this is chance.
    3. *37% vs 31%*: p ≈ 0.24, so **not** confident. Then choose *Simulate my own trial* with A=40%, B=30% and press 🎲 a few times
       at n = 30 and at n = 1000. Same true difference, very different p-values.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · The p-value, defined by giving the *same drug twice*

    The video's key move: give **drug A to both groups**. Now *every* difference is 100% weird random stuff.
    Most runs give a large p-value (like 0.9), but every once in a while all the allergic people land in one group and
    all the strong placebo reactors in the other, and we get p = 0.01. Calling that "different" is a **false positive**.

    > **A 0.05 threshold means:** if there is no difference and we repeat the experiment many times,
    > about **5%** of the experiments give $p < 0.05$, i.e. a false positive.

    Formally, the p-value is the probability, *assuming the null hypothesis (no difference) is true*, of seeing a difference
    **at least as extreme** as the one observed:

    $$p = P\big(\,|\text{difference}| \ge |\text{observed difference}| \;\big|\; H_0\,\big)$$

    Below, each falling dot is one complete experiment (two groups, same drug, two-proportion z-test).
    It lands in the bin of its p-value. <span style="color:#e4572e">**Red**</span> dots are false positives.
    """)
    return


@app.cell
def SameDrugSim_class(anywidget, traitlets):
    class SameDrugSim(anywidget.AnyWidget):
        """Repeats a two-group experiment in JS and drops each p-value into a histogram."""
        _esm = r"""
    function erf(x){const s=Math.sign(x);x=Math.abs(x);const t=1/(1+0.3275911*x);
      const y=1-(((((1.061405429*t-1.453152027)*t)+1.421413741)*t-0.284496736)*t+0.254829592)*t*Math.exp(-x*x);return s*y;}
    const pnorm = z => 0.5*(1+erf(z/Math.SQRT2));
    function binom(n,p){let k=0;for(let i=0;i<n;i++) if(Math.random()<p) k++;return k;}
    function render({ model, el }) {
      const W=640,H=260,PADL=36,PADB=34,TOP=12,BINS=20;
      const NS="http://www.w3.org/2000/svg";
      const ctl=document.createElement("div"); ctl.style.cssText="display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-bottom:6px";
      el.appendChild(ctl);
      const svg=document.createElementNS(NS,"svg"); svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
      Object.assign(svg.style,{width:"100%",maxWidth:W+"px",userSelect:"none"}); el.appendChild(svg);
      const info=document.createElement("div"); info.style.cssText="font-family:monospace;margin-top:4px"; el.appendChild(info);
      const mk=(tag,a,p=svg)=>{const e=document.createElementNS(NS,tag);for(const k in a)e.setAttribute(k,a[k]);p.appendChild(e);return e;};
      const btn=(t,f)=>{const b=document.createElement("button");b.textContent=t;b.onclick=f;ctl.appendChild(b);return b;};
      let pv=[...model.get("pvals")], timer=null, last=null;
      function experiment(){
        const n=model.get("n"), r=model.get("rate"), e=model.get("effect");
        const a=binom(n,Math.min(1,r+e)), c=binom(n,r);
        const pa=a/n, pb=c/n, pool=(a+c)/(2*n), se=Math.sqrt(pool*(1-pool)*2/n);
        const p = se>0 ? 2*(1-pnorm(Math.abs(pa-pb)/se)) : 1;
        pv.push(p); last={a,c,n,p};
        if(pv.length%20===0){model.set("pvals",pv.slice());model.save_changes();}
        draw();
      }
      function draw(){
        svg.innerHTML="";
        const alpha=model.get("alpha"), plotW=W-PADL-10, plotH=H-TOP-PADB, bw=plotW/BINS;
        const counts=new Array(BINS).fill(0); pv.forEach(p=>counts[Math.min(BINS-1,Math.floor(p*BINS))]++);
        const mx=Math.max(10,...counts);
        // alpha region
        mk("rect",{x:PADL,y:TOP,width:Math.max(1,alpha*plotW),height:plotH,fill:"#e4572e",opacity:0.12});
        counts.forEach((k,i)=>{
          const h=k/mx*plotH, lo=i/BINS, hi=(i+1)/BINS;
          const redFrac = hi<=alpha?1:(lo>=alpha?0:(alpha-lo)/(hi-lo));
          mk("rect",{x:PADL+i*bw+1,y:TOP+plotH-h,width:bw-2,height:h,fill:"#bbb"});
          mk("rect",{x:PADL+i*bw+1,y:TOP+plotH-h,width:(bw-2)*redFrac,height:h,fill:"#e4572e"});
        });
        // expected uniform level
        const exp=pv.length/BINS/mx*plotH;
        if(pv.length && model.get("effect")===0) mk("line",{x1:PADL,x2:PADL+plotW,y1:TOP+plotH-exp,y2:TOP+plotH-exp,stroke:"currentColor","stroke-dasharray":"5 4",opacity:0.6});
        mk("line",{x1:PADL,x2:PADL+plotW,y1:TOP+plotH,y2:TOP+plotH,stroke:"currentColor"});
        for(let t=0;t<=10;t+=2){const x=PADL+t/10*plotW; const tx=mk("text",{x,y:H-PADB+16,"text-anchor":"middle","font-size":11,fill:"currentColor"}); tx.textContent=(t/10).toFixed(1);}
        const lab=mk("text",{x:PADL+plotW/2,y:H-4,"text-anchor":"middle","font-size":12,fill:"currentColor"}); lab.textContent="p-value of each experiment";
        if(last){ const x=PADL+last.p*plotW; mk("circle",{cx:x,cy:TOP+8,r:7,fill:last.p<alpha?"#e4572e":"#999"}); }
        const fp=pv.filter(p=>p<alpha).length;
        info.innerHTML=`experiments: <b>${pv.length}</b> &nbsp; p &lt; ${alpha}: <b style="color:#e4572e">${fp}</b> (${pv.length?(100*fp/pv.length).toFixed(1):"0.0"}%)`
          + (last?` &nbsp; last: A cured ${last.a}/${last.n}, B cured ${last.c}/${last.n}, p = ${last.p.toFixed(3)}`:"");
      }
      const play=btn("▶ play",()=>{ if(timer){clearInterval(timer);timer=null;play.textContent="▶ play";model.set("pvals",pv.slice());model.save_changes();}
        else{timer=setInterval(experiment,60);play.textContent="⏸ pause";}});
      btn("+1 experiment",experiment);
      btn("⏩ +1000",()=>{for(let i=0;i<1000;i++){experiment();} model.set("pvals",pv.slice());model.save_changes();});
      btn("↺ reset",()=>{pv=[];last=null;model.set("pvals",[]);model.save_changes();draw();});
      const sp=document.createElement("span"); sp.textContent=" threshold α:"; ctl.appendChild(sp);
      for(const a of [0.00001,0.01,0.05,0.2]){ btn(String(a),()=>{model.set("alpha",a);model.save_changes();}); }
      model.on("change:alpha",draw); model.on("change:pvals",()=>{ if(!timer && model.get("pvals").length===0){pv=[];} draw(); });
      draw();
      return ()=>clearInterval(timer);
    }
    export default { render };
    """
        n = traitlets.Int(100).tag(sync=True)
        rate = traitlets.Float(0.35).tag(sync=True)
        effect = traitlets.Float(0.0).tag(sync=True)
        alpha = traitlets.Float(0.05).tag(sync=True)
        pvals = traitlets.List(traitlets.Float()).tag(sync=True)

    return (SameDrugSim,)


@app.cell
def same_drug_widget(SameDrugSim, mo):
    same_drug = mo.ui.anywidget(SameDrugSim(n=100, rate=0.35, effect=0.0, alpha=0.05))
    same_drug
    return (same_drug,)


@app.cell
def same_drug_readout(mo, np, same_drug):
    _p = np.array(same_drug.value["pvals"])
    _alpha = same_drug.value["alpha"]
    if len(_p) == 0:
        _out = mo.md("_Press **▶ play** or **⏩ +1000** above to start repeating the experiment._")
    else:
        _out = mo.hstack([
            mo.stat(f"{len(_p)}", label="experiments (same drug twice)"),
            mo.stat(f"{(_p < _alpha).mean():.1%}", label=f"false positives (p < {_alpha})", caption=f"theory: {_alpha:.3%}"),
            mo.stat(f"{np.median(_p):.2f}", label="median p-value", caption="theory: 0.50 (uniform)"),
        ], justify="start", gap=2)
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Press **⏩ +1000** a few times. The histogram flattens: under "no difference", p-values are **uniformly** spread between 0 and 1.
       That's *why* "p < 0.05" happens 5% of the time: it's just the leftmost 5% of a flat distribution.
    2. Watch single experiments with **+1 experiment**. Red dots happen even though both groups got *the same drug*.
    3. A p-value of 0.9 isn't "evidence the drugs are the same"; under $H_0$, 0.9 is exactly as likely as 0.1.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · Picking the threshold: 0.00001, 0.05 or 0.2?

    The threshold (often called **α**) is the false-positive rate you're willing to live with:

    | situation in the video | threshold | false positives |
    |---|---|---|
    | it's *extremely* important to be right | 0.00001 | 1 in 100,000 experiments |
    | the usual default | 0.05 | 1 in 20 |
    | "will the ice-cream truck be on time?" | 0.2 | 2 in 10 |

    0.05 is popular because pushing false positives below 5% "often costs more than it's worth" (more people, more money).
    Click the **α buttons** in the simulator above; the table below re-scores the *same* simulated experiments at every threshold.
    """)
    return


@app.cell
def threshold_table(mo, np, pd, same_drug):
    _p = np.array(same_drug.value["pvals"])
    _rows = []
    for _a in [0.00001, 0.01, 0.05, 0.2]:
        _rows.append({
            "threshold α": _a,
            "expected false-positive rate": f"{_a:.3%}",
            "observed in your simulation": f"{(_p < _a).mean():.3%}" if len(_p) else "—",
            "false positives / experiments": f"{int((_p < _a).sum())} / {len(_p)}",
            "selected": "◀" if abs(_a - same_drug.value["alpha"]) < 1e-12 else "",
        })
    mo.ui.table(pd.DataFrame(_rows), selection=None)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. With ~3000 experiments, α = 0.2 gives ≈ 20% false positives and α = 0.00001 almost surely gives **zero**: you'd need ~100,000 experiments to see one.
    2. The threshold is chosen **before** looking at the data. Moving it after you see p = 0.07 ("let's use 0.1!") is a form of p-hacking (section 6).
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · When there *is* a difference: p-values pile up near 0

    Section 2 gave the same drug twice ($H_0$ true) and p-values came out **flat**. What if drug A really is better ($H_1$ true)?
    Then small p-values become common, and the share of experiments with $p < \alpha$ is the test's **power**: the chance
    of correctly detecting a real difference.

    $$\text{under } H_0:\; p \sim \text{Uniform}(0,1) \qquad\qquad \text{under } H_1:\; p \text{ piles up near } 0 \;(\text{more so with bigger effects and more people})$$

    Overlay: <span style="color:#999">**grey = same drug ($H_0$)**</span>, <span style="color:#f18f01">**orange = drug A truly better ($H_1$)**</span>.
    """)
    return


@app.cell
def ztest_helper(np, stats):
    def ztest_p(a, c, n_a, n_b=None):
        """Vectorised two-sided two-proportion z-test p-value (A: a/n_a cured, B: c/n_b cured)."""
        n_b = n_a if n_b is None else n_b
        a, c = np.asarray(a, float), np.asarray(c, float)
        pool = (a + c) / (n_a + n_b)
        se = np.sqrt(pool * (1 - pool) * (1 / n_a + 1 / n_b))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = np.where(se > 0, np.abs(a / n_a - c / n_b) / se, 0.0)
        return 2 * stats.norm.sf(z)

    return (ztest_p,)


@app.cell
def h1_controls(mo):
    h1_n = mo.ui.slider(10, 1000, step=10, value=200, label="people per drug", show_value=True)
    h1_effect = mo.ui.slider(0, 20, step=1, value=6, label="true difference (percentage points, A − B)", show_value=True)
    h1_alpha = mo.ui.dropdown(options={"0.01": 0.01, "0.05": 0.05, "0.2": 0.2}, value="0.05", label="α")
    mo.hstack([h1_n, h1_effect, h1_alpha], justify="start")
    return h1_alpha, h1_effect, h1_n


@app.cell
def h0_vs_h1(alt, h1_alpha, h1_effect, h1_n, mo, np, pd, ztest_p):
    _rng = np.random.default_rng(1)
    _reps, _base, _n = 20000, 0.31, h1_n.value
    _pA = _base + h1_effect.value / 100
    _p0 = ztest_p(_rng.binomial(_n, _base, _reps), _rng.binomial(_n, _base, _reps), _n)
    _p1 = ztest_p(_rng.binomial(_n, _pA, _reps), _rng.binomial(_n, _base, _reps), _n)
    _edges = np.linspace(0, 1, 41)
    _rows = []
    for _lab, _arr in [("same drug (H0)", _p0), ("A truly better (H1)", _p1)]:
        _h, _ = np.histogram(_arr, _edges)
        _rows += [{"world": _lab, "lo": _edges[i], "hi": _edges[i + 1], "density": _h[i] / _reps / (_edges[1] - _edges[0])} for i in range(len(_h))]
    _df = pd.DataFrame(_rows)
    _bars = alt.Chart(_df).mark_rect(opacity=0.55).encode(
        x=alt.X("lo:Q", title="p-value", scale=alt.Scale(domain=[0, 1])), x2="hi:Q",
        y=alt.Y("density:Q", title="density", stack=None),
        color=alt.Color("world:N", scale=alt.Scale(domain=["same drug (H0)", "A truly better (H1)"], range=["#999", "#f18f01"]), legend=alt.Legend(orient="top", title=None)),
    )
    _rule = alt.Chart(pd.DataFrame({"a": [h1_alpha.value]})).mark_rule(color="#e4572e", strokeWidth=2).encode(x="a:Q")
    power = float((_p1 < h1_alpha.value).mean())
    mo.vstack([
        (_bars + _rule).properties(width="container", height=260),
        mo.hstack([
            mo.stat(f"{(_p0 < h1_alpha.value).mean():.1%}", label="H0: false-positive rate", caption=f"≈ α = {h1_alpha.value}"),
            mo.stat(f"{power:.1%}", label="H1: power (p < α)", caption="chance a real difference is detected"),
            mo.stat(f"{1 - power:.1%}", label="H1: missed (false negative)"),
        ], justify="start", gap=2),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Set the true difference to **0**: both histograms become flat. You can't tell the worlds apart.
    2. Use the video's **6-point** difference with ~200 people per drug: power is only ≈ 25%, so p ≈ 0.24 (the video's result) is very plausible
       **even though drug A really is better**. "Not significant" ≠ "no difference".
    3. Keep 6 points and push people to 1000: the orange pile slams against 0.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · p-values do **not** measure effect size

    The video's final (and most interesting) point: a small p-value tells us drug A is *probably different* from drug B,
    **not how different**. The p-value mixes two things: the size of the difference and the amount of data.

    * 6-point difference, few people → p = **0.24** (not significant)
    * 1-point difference, *lots* of people → p = **0.04** (significant!)

    Drag the <span style="color:#f18f01">**A**</span> and <span style="color:#2e86ab">**B**</span> markers to set the observed cure rates.
    The difference updates instantly; release to recompute p-values across sample sizes.
    """)
    return


@app.cell
def RateDragger_class(anywidget, traitlets):
    class RateDragger(anywidget.AnyWidget):
        """Two draggable cure-rate markers (A, B) on a 0-100% axis."""
        _esm = r"""
    function render({ model, el }) {
      const W=640,H=130,PAD=30,Y0=80, NS="http://www.w3.org/2000/svg";
      const svg=document.createElementNS(NS,"svg"); svg.setAttribute("viewBox",`0 0 ${W} ${H}`);
      Object.assign(svg.style,{width:"100%",maxWidth:W+"px",userSelect:"none",touchAction:"none"}); el.appendChild(svg);
      const mk=(t,a)=>{const e=document.createElementNS(NS,t);for(const k in a)e.setAttribute(k,a[k]);svg.appendChild(e);return e;};
      const toSvg=e=>{const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;return p.matrixTransform(svg.getScreenCTM().inverse());};
      const sx=v=>PAD+v/100*(W-2*PAD), inv=px=>(px-PAD)/(W-2*PAD)*100;
      let drag=null;
      function draw(){
        svg.innerHTML="";
        const a=model.get("rate_a"), b=model.get("rate_b");
        mk("line",{x1:sx(0),x2:sx(100),y1:Y0,y2:Y0,stroke:"currentColor","stroke-width":2});
        for(let t=0;t<=100;t+=10){mk("line",{x1:sx(t),x2:sx(t),y1:Y0,y2:Y0+6,stroke:"currentColor"});
          const tx=mk("text",{x:sx(t),y:Y0+20,"text-anchor":"middle","font-size":11,fill:"currentColor"}); tx.textContent=t+"%";}
        mk("rect",{x:Math.min(sx(a),sx(b)),y:Y0-6,width:Math.abs(sx(a)-sx(b)),height:12,fill:"#e4572e",opacity:0.3});
        const d=mk("text",{x:(sx(a)+sx(b))/2,y:Y0+40,"text-anchor":"middle","font-size":13,fill:"#e4572e","font-weight":"bold"});
        d.textContent=`difference A − B = ${(a-b).toFixed(1)} points`;
        [["rate_a",a,"#f18f01","A",Y0-40],["rate_b",b,"#2e86ab","B",Y0-16]].forEach(([k,v,col,lab,y])=>{
          mk("line",{x1:sx(v),x2:sx(v),y1:y,y2:Y0,stroke:col,"stroke-width":2});
          const c=mk("circle",{cx:sx(v),cy:y,r:12,fill:col,stroke:"#fff","stroke-width":2}); c.style.cursor="grab";
          const t=mk("text",{x:sx(v),y:y+4,"text-anchor":"middle","font-size":11,fill:"#fff","pointer-events":"none"}); t.textContent=lab;
          c.addEventListener("pointerdown",e=>{drag=k;svg.setPointerCapture(e.pointerId);});
        });
      }
      svg.addEventListener("pointermove",e=>{ if(!drag) return;
        model.set(drag,Math.max(0,Math.min(100,Math.round(inv(toSvg(e).x)*2)/2))); draw(); });
      const stop=()=>{ if(drag){drag=null;model.save_changes();} };
      svg.addEventListener("pointerup",stop); svg.addEventListener("pointerleave",stop);
      model.on("change:rate_a",draw); model.on("change:rate_b",draw); draw();
    }
    export default { render };
    """
        rate_a = traitlets.Float(37.0).tag(sync=True)
        rate_b = traitlets.Float(31.0).tag(sync=True)

    return (RateDragger,)


@app.cell
def rates_widget(RateDragger, mo):
    rates = mo.ui.anywidget(RateDragger(rate_a=37.0, rate_b=31.0))
    effect_n = mo.ui.slider(steps=[10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000], value=200, label="people per drug in *your* study", show_value=True)
    mo.vstack([rates, effect_n])
    return effect_n, rates


@app.cell
def effect_vs_n(alt, effect_n, mo, np, pd, rates, ztest_p):
    _ra, _rb = rates.value["rate_a"] / 100, rates.value["rate_b"] / 100
    _ns = np.unique(np.round(np.logspace(1, 5, 120)).astype(int))
    _curve = pd.DataFrame({"people per drug": _ns, "p": ztest_p(_ra * _ns, _rb * _ns, _ns)})
    _curve["significant"] = np.where(_curve["p"] < 0.05, "p < 0.05", "p ≥ 0.05")
    _n = effect_n.value
    effect_p = float(ztest_p(_ra * _n, _rb * _n, _n))
    _line = alt.Chart(_curve).mark_line(color="#999").encode(x=alt.X("people per drug:Q", scale=alt.Scale(type="log")), y=alt.Y("p:Q", title="p-value", scale=alt.Scale(domain=[0, 1])))
    _pts = alt.Chart(_curve).mark_circle(size=25).encode(x="people per drug:Q", y="p:Q",
        color=alt.Color("significant:N", scale=alt.Scale(domain=["p < 0.05", "p ≥ 0.05"], range=["#e4572e", "#999"]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["people per drug", alt.Tooltip("p:Q", format=".4f")])
    _rule = alt.Chart(pd.DataFrame({"y": [0.05]})).mark_rule(color="#e4572e", strokeDash=[5, 4]).encode(y="y:Q")
    _you = alt.Chart(pd.DataFrame({"x": [_n], "y": [effect_p]})).mark_point(shape="diamond", size=200, filled=True, color="black").encode(x="x:Q", y="y:Q")
    _diff = (_ra - _rb) * 100
    mo.vstack([
        mo.hstack([
            mo.stat(f"{_diff:+.1f} pts", label="effect size (A − B)", caption="doesn't change with n"),
            mo.stat(f"{effect_p:.3g}", label=f"p-value at n = {_n:,}", caption="significant" if effect_p < 0.05 else "not significant"),
        ], justify="start", gap=2),
        (_line + _pts + _rule + _you).properties(width="container", height=260, title="Same observed difference, different amounts of data"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Leave A = 37%, B = 31% (the video's 6-point gap) at n ≈ 200: p ≈ 0.2. Slide n up to 1000: now p is tiny. **The effect didn't change; only the data did.**
    2. Drag B to 36% (a 1-point gap) and set n = 20,000: p ≈ 0.04, "significant" for a difference nobody may care about.
    3. So always report the **effect size** (and ideally a confidence interval) next to the p-value.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Bonus: misconceptions and p-hacking

    *The video doesn't cover this section, but it follows from section 2: if 5% of no-difference experiments are "significant",
    then running **many** of them is a reliable way to manufacture a finding.*

    **What a p-value is NOT**

    | Misreading | Why it's wrong |
    |---|---|
    | "p = 0.03 means a 3% chance the drugs are the same" | p is computed *assuming* they're the same. It's P(data this extreme \| $H_0$), not P($H_0$ \| data). |
    | "p = 0.24 proves the drugs are the same" | Section 4: with low power, real differences often give large p. *Fail to reject* ≠ accept. |
    | "p = 0.0001 means a huge effect" | Section 5: tiny effects + lots of data give tiny p. |
    | "p < 0.05 means the result will replicate" | With 5% false positives and low power, many "findings" don't. |

    **p-hacking** = trying analyses until something gives p < 0.05. Two classic ways:

    1. **Many outcomes** (the "jelly bean" problem): test 20 things that don't work; at α = 0.05,
       $P(\text{at least one } p<0.05) = 1 - 0.95^{k}$, which is 64% for $k = 20$.
    2. **Optional stopping**: check the p-value after every new batch of patients and stop as soon as p < 0.05.
    """)
    return


@app.cell
def hack_controls(mo):
    hack_k = mo.ui.slider(1, 60, value=20, label="useless outcomes tested per study (k)", show_value=True)
    hack_reroll = mo.ui.run_button(label="🎲 run another study")
    mo.hstack([hack_k, hack_reroll], justify="start")
    return hack_k, hack_reroll


@app.cell
def jelly_beans(alt, hack_k, hack_reroll, mo, np, pd, ztest_p):
    hack_reroll
    _rng = np.random.default_rng()
    _k, _n = hack_k.value, 100
    # one study: k outcomes, both groups identical (H0 true for every outcome)
    _p_one = ztest_p(_rng.binomial(_n, 0.3, _k), _rng.binomial(_n, 0.3, _k), _n)
    _df = pd.DataFrame({"outcome": [f"#{i+1}" for i in range(_k)], "p": _p_one, "result": np.where(_p_one < 0.05, "p < 0.05 🎉", "p ≥ 0.05")})
    _bars = alt.Chart(_df).mark_bar().encode(
        x=alt.X("outcome:N", sort=None, title="outcome tested (none truly differ)"),
        y=alt.Y("p:Q", scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("result:N", scale=alt.Scale(domain=["p < 0.05 🎉", "p ≥ 0.05"], range=["#e4572e", "#bbb"]), legend=alt.Legend(orient="top", title=None)),
        tooltip=["outcome", alt.Tooltip("p:Q", format=".3f")],
    )
    _rule = alt.Chart(pd.DataFrame({"y": [0.05]})).mark_rule(color="#e4572e", strokeDash=[5, 4]).encode(y="y:Q")
    # many studies to estimate the family-wise false positive rate
    _P = ztest_p(_rng.binomial(_n, 0.3, (4000, _k)), _rng.binomial(_n, 0.3, (4000, _k)), _n)
    hack_fwer = float((_P < 0.05).any(axis=1).mean())
    mo.vstack([
        (_bars + _rule).properties(width="container", height=220, title="One study, k outcomes, no real effects"),
        mo.hstack([
            mo.stat(f"{int((_p_one < 0.05).sum())}", label="'discoveries' in this study"),
            mo.stat(f"{hack_fwer:.1%}", label="studies with ≥ 1 p < 0.05 (4000 sims)", caption=f"theory 1 − 0.95^{_k} = {1 - 0.95**_k:.1%}"),
            mo.stat(f"{0.05 / _k:.4f}", label="Bonferroni threshold 0.05 / k", caption="one simple fix"),
        ], justify="start", gap=2),
    ])
    return


@app.cell
def peek_controls(mo):
    peek_batch = mo.ui.slider(5, 100, step=5, value=10, label="people added per drug between peeks", show_value=True)
    peek_max = mo.ui.slider(1, 50, value=20, label="maximum number of peeks", show_value=True)
    mo.hstack([peek_batch, peek_max], justify="start")
    return peek_batch, peek_max


@app.cell
def optional_stopping(alt, mo, np, pd, peek_batch, peek_max, ztest_p):
    _rng = np.random.default_rng(7)
    _sims, _b, _L = 4000, peek_batch.value, peek_max.value
    _a = _rng.binomial(_b, 0.3, (_sims, _L)).cumsum(axis=1)
    _c = _rng.binomial(_b, 0.3, (_sims, _L)).cumsum(axis=1)
    _n = _b * np.arange(1, _L + 1)
    _P = ztest_p(_a, _c, _n)
    _ever = np.maximum.accumulate(_P < 0.05, axis=1)
    peek_rate = float(_ever[:, -1].mean())
    _df = pd.DataFrame({"peek": np.arange(1, _L + 1), "share of studies 'significant' so far": _ever.mean(axis=0)})
    _line = alt.Chart(_df).mark_line(point=True, color="#e4572e").encode(
        x=alt.X("peek:Q", title="number of peeks"), y=alt.Y("share of studies 'significant' so far:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, max(0.3, peek_rate + 0.05)])))
    _rule = alt.Chart(pd.DataFrame({"y": [0.05]})).mark_rule(strokeDash=[5, 4], color="#999").encode(y="y:Q")
    mo.vstack([
        (_line + _rule).properties(width="container", height=220, title="Optional stopping with the same drug in both groups"),
        mo.stat(f"{peek_rate:.1%}", label=f"false-positive rate after {_L} peeks", caption="the promised rate was 5%"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Set k = 1: back to 5%. Set k = 20 and re-roll: most studies "discover" something. Every red bar is a false positive.
    2. In optional stopping, set peeks to 1 (≈ 5%), then 20 (≈ 20%+). Each peek is another lottery ticket.
    3. The cure is boring: **decide the outcome, sample size and α in advance**, and correct for multiple tests.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Sandbox: analyse your own 2×2 table

    Type in any experiment (drugs, A/B tests, ad clicks). You get the p-value **and** the effect size with a 95% confidence interval,
    so you never report one without the other.
    """)
    return


@app.cell
def sandbox_controls(mo):
    sb = mo.ui.dictionary({
        "a": mo.ui.number(0, 1_000_000, value=73, label="A cured"),
        "b": mo.ui.number(0, 1_000_000, value=125, label="A not cured"),
        "c": mo.ui.number(0, 1_000_000, value=59, label="B cured"),
        "d": mo.ui.number(0, 1_000_000, value=131, label="B not cured"),
        "alpha": mo.ui.dropdown(options=["0.00001", "0.01", "0.05", "0.2"], value="0.05", label="threshold α"),
    })
    mo.hstack(list(sb.elements.values()), justify="start", wrap=True)
    return (sb,)


@app.cell
def sandbox_out(fisher_p, mo, np, sb, ztest_p):
    _a, _b, _c, _d = (int(sb.value[k] or 0) for k in "abcd")
    _alpha = float(sb.value["alpha"])
    mo.stop(min(_a + _b, _c + _d) == 0, mo.md("_Each drug needs at least one person._"))
    _pf = fisher_p(_a, _b, _c, _d)
    _pz = float(ztest_p(_a, _c, _a + _b, _c + _d))
    _ra, _rb = _a / (_a + _b), _c / (_c + _d)
    _se = np.sqrt(_ra * (1 - _ra) / (_a + _b) + _rb * (1 - _rb) / (_c + _d))
    _lo, _hi = (_ra - _rb) - 1.96 * _se, (_ra - _rb) + 1.96 * _se
    _sig = _pf < _alpha
    mo.vstack([
        mo.hstack([
            mo.stat(f"{_pf:.3g}", label="p-value (Fisher exact)", caption=f"z-test: {_pz:.3g}"),
            mo.stat(f"{(_ra - _rb) * 100:+.1f} pts", label="effect: A − B cure rate", caption=f"95% CI [{_lo*100:+.1f}, {_hi*100:+.1f}]"),
        ], justify="start", gap=2),
        mo.callout(mo.md(
            f"**Reject the null hypothesis** at α = {_alpha}: we're confident drug A differs from drug B. Check the effect size to decide whether the difference matters."
            if _sig else
            f"**Fail to reject** at α = {_alpha}. That's *not* proof the drugs are the same; the CI shows which differences are still plausible."
        ), kind="danger" if _sig else "neutral"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 · Quiz: check your intuition
    Answer all five, then flip the reveal switch.
    """)
    return


@app.cell
def quiz(mo):
    quiz = mo.ui.dictionary({
        "q1": mo.ui.radio(["about 0%", "about 5%", "about 50%", "it depends on how good drug A is"], label="**1.** You give drug A to both groups and repeat 1000 times. What share of experiments has p < 0.05?"),
        "q2": mo.ui.radio(["The drugs are the same", "We can't conclude they differ", "Drug B is better"], label="**2.** The video's 37% vs 31% study gave p = 0.24. What do you conclude at α = 0.05?"),
        "q3": mo.ui.radio(["The 6-point difference with p = 0.24", "The 1-point difference with p = 0.04", "Can't tell from p alone"], label="**3.** Which study shows the *larger* effect?"),
        "q4": mo.ui.radio(["0.00001", "0.05", "0.2"], label="**4.** Deciding whether the ice-cream truck arrives on time: which threshold did the video suggest?"),
        "q5": mo.ui.radio(["5%", "about 64%", "100%"], label="**5.** You test 20 useless supplements at α = 0.05. Chance of at least one 'significant' result?"),
    })
    quiz_reveal = mo.ui.switch(label="reveal answers")
    mo.vstack([*quiz.elements.values(), quiz_reveal])
    return quiz, quiz_reveal


@app.cell(hide_code=True)
def quiz_score(mo, quiz, quiz_reveal):
    _key = {
        "q1": ("about 5%", "Under H0 p-values are uniform, so p < 0.05 happens 5% of the time, whatever drug A is."),
        "q2": ("We can't conclude they differ", "A large p means 'failed to see a difference', not 'proved they're the same'."),
        "q3": ("The 6-point difference with p = 0.24", "p-values don't measure effect size; the 1-point study just had many more people."),
        "q4": ("0.2", "Low stakes → tolerate 2 false positives in 10."),
        "q5": ("about 64%", "1 − 0.95²⁰ ≈ 0.64: the jelly-bean problem."),
    }
    _answered = sum(v is not None for v in quiz.value.values())
    _score = sum(quiz.value[k] == v[0] for k, v in _key.items())
    if not quiz_reveal.value:
        _out = mo.md(f"_{_answered}/5 answered. Flip **reveal answers** when ready._")
    else:
        _lines = [f"- {'✅' if quiz.value[k] == a else '❌'} **Q{k[1]}**: {a}. {why}" for k, (a, why) in _key.items()]
        _out = mo.callout(mo.md(f"### Score: {_score}/5\n" + "\n".join(_lines)), kind="success" if _score >= 4 else "warn")
    _out
    return


@app.cell(hide_code=True)
def recap(mo):
    mo.md(r"""
    ## Recap

    | Idea | One-liner | Where you saw it |
    |---|---|---|
    | Why we need many people | Weird random things (allergies, placebo, missed doses) can make any single comparison misleading | §1 trial explorer |
    | p-value | P(a difference at least this extreme \| no real difference) | §2 same-drug simulator |
    | Under $H_0$ | p-values are uniform on [0, 1], so p < α happens α of the time (false positives) | §2 histogram |
    | Threshold α | Chosen in advance: 0.00001 (critical), 0.05 (default), 0.2 (ice-cream truck) | §3 threshold table |
    | Under $H_1$ | p-values pile up near 0; share below α = power | §4 overlay |
    | Hypothesis testing | Null hypothesis = the drugs are the same; small p → reject it | §2–§4 |
    | Effect size | Small p ≠ big difference; p shrinks with n even for tiny effects | §5 drag A/B |
    | p-hacking | Many tests or repeated peeking inflate false positives far beyond α | §6 |

    **Double bam!** Always report the p-value *together with* the effect size and a confidence interval, and fix α, n and the outcome before you look.
    """)
    return


if __name__ == "__main__":
    app.run()

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
    mo.md(r"""
    # Statistical paradoxes are everywhere: the inspection paradox
    *An interactive companion to marimo's video **"Statistical Paradoxes Are Everywhere"**.*

    > A teacher and every student in the same room write down **the same number**: how many students are in this class.
    > Repeat this for every class in the school. Then average the teachers' notes and average the students' notes.
    > **The two averages disagree, by a lot.**

    This is the **inspection paradox**. When the *way you sample* depends on the *thing you measure*,
    your average quietly changes. In this notebook you'll replay three riddles from the video in simulation, which is the
    **mental replay** technique the video recommends, and then try a bonus riddle about your bus being "always late".
    """)
    return


@app.cell
def intro_guess(mo):
    intro_guess = mo.ui.radio(
        options=["Teachers' average is higher", "They're about the same", "Students' average is higher"],
        label="**Before you scroll: whose average class size is bigger?**",
    )
    intro_guess
    return (intro_guess,)


@app.cell(hide_code=True)
def _(intro_guess, mo):
    mo.md(
        "_Lock in an answer above. Section 1 will show you who was right._"
        if intro_guess.value is None
        else (
            "✅ **Right!** The students see bigger classes. Now let's find out *why*."
            if intro_guess.value.startswith("Students")
            else "🤔 **Interesting guess.** Keep it in mind and watch the simulation below."
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Riddle 1 · The teacher and the students

    Each class contributes **one** teacher's note, but **n** students' notes, and every one of those notes says *n*.
    A class of 100 gets 100 votes from its students and still only one from its teacher.
    A class of 1 gets one vote either way.

    Averaging over teachers gives the plain mean. Averaging over students gives a **size-biased** mean:

    $$\bar n_{\text{teacher}} = E[n] \qquad\qquad \bar n_{\text{student}} = \frac{\sum n\cdot n}{\sum n} = \frac{E[n^2]}{E[n]} = E[n] + \frac{\operatorname{Var}(n)}{E[n]}$$

    The gap is exactly $\operatorname{Var}(n)/E[n]$. If every class had the same size, there would be **no paradox**.

    Press **▶ play** to watch classes form one by one. Then press **⏩** to run it fast.
    """)
    return


@app.cell
def class_controls(mo):
    class_mean = mo.ui.slider(4, 40, value=10, label="Average class size (teacher's view)", show_value=True)
    class_spread = mo.ui.slider(0.0, 1.5, step=0.05, value=0.8, label="How unequal are class sizes? (coefficient of variation)", show_value=True)
    mo.vstack([class_mean, class_spread])
    return class_mean, class_spread


@app.cell
def class_sizes(class_mean, class_spread, np):
    def sample_class_sizes(mean, cv, n, seed=0):
        """Class sizes >= 1 with a given mean and coefficient of variation (gamma distributed)."""
        rng = np.random.default_rng(seed)
        if cv < 0.01:
            return np.full(n, max(1, round(mean)), dtype=int)
        shape = 1 / cv**2
        return np.maximum(1, np.round(rng.gamma(shape, mean / shape, n))).astype(int)

    class_sizes = sample_class_sizes(class_mean.value, class_spread.value, 20_000)
    return (class_sizes,)


@app.cell
def classroom_widget(anywidget, traitlets):
    class ClassroomSim(anywidget.AnyWidget):
        """Animated replay: each step creates a classroom; the teacher adds 1 note, the students add n notes."""

        _esm = r"""
    function render({ model, el }) {
      const W = 700, H = 300;
      const ctl = document.createElement("div");
      Object.assign(ctl.style, { display: "flex", gap: "6px", alignItems: "center", font: "13px system-ui, sans-serif", marginBottom: "6px" });
      el.appendChild(ctl);
      const canvas = document.createElement("canvas");
      canvas.width = W * 2; canvas.height = H * 2;
      Object.assign(canvas.style, { width: "100%", maxWidth: W + "px", borderRadius: "8px", background: "rgba(127,127,127,0.06)" });
      el.appendChild(canvas);
      const ctx = canvas.getContext("2d"); ctx.scale(2, 2);
      const btn = (t, fn) => { const b = document.createElement("button"); b.textContent = t; b.onclick = fn;
        Object.assign(b.style, { padding: "3px 10px", borderRadius: "12px", border: "1px solid #bbb", cursor: "pointer" }); ctl.appendChild(b); return b; };
      const label = document.createElement("span"); label.style.marginLeft = "8px";

      let S, timer = null, fast = false;
      const reset = () => { S = { idx: 0, n: 0, t: {}, s: {}, tSum: 0, tCnt: 0, sSum: 0, sCnt: 0 }; draw(); };
      function step() {
        const sizes = model.get("sizes"); if (!sizes.length) return;
        const n = sizes[S.idx % sizes.length]; S.idx++; S.n = n;
        S.t[n] = (S.t[n] || 0) + 1; S.s[n] = (S.s[n] || 0) + n;
        S.tSum += n; S.tCnt += 1; S.sSum += n * n; S.sCnt += n;
      }
      function tick() { const k = fast ? 40 : 1; for (let i = 0; i < k; i++) step(); draw(); }
      function setTimer() { clearInterval(timer); timer = setInterval(tick, fast ? 30 : 700); }

      function draw() {
        ctx.clearRect(0, 0, W, H);
        ctx.font = "12px system-ui, sans-serif"; ctx.fillStyle = "#888";
        // room
        ctx.strokeStyle = "#aaa"; ctx.strokeRect(10, 10, 230, 280);
        ctx.fillText(`classroom #${S.idx}`, 18, 28);
        if (S.n) {
          ctx.fillStyle = "#e4572e"; ctx.fillRect(110, 40, 20, 20);
          ctx.fillStyle = "#e4572e"; ctx.fillText("teacher: 1 note", 82, 76);
          const per = 12, r = S.n > 60 ? 3.5 : 6, gap = S.n > 60 ? 9 : 17;
          for (let i = 0; i < S.n; i++) {
            const x = 30 + (i % per) * gap, y = 100 + Math.floor(i / per) * gap;
            if (y > 270) break;
            ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fillStyle = "#2e86ab"; ctx.fill();
          }
          ctx.fillStyle = "#2e86ab"; ctx.fillText(`${S.n} students: ${S.n} notes, each saying "${S.n}"`, 18, 284);
        }
        // histograms (share of notes)
        const X0 = 270, X1 = 690, maxN = Math.max(20, ...Object.keys(S.t).map(Number));
        const bw = (X1 - X0) / (maxN + 1);
        const panel = (obj, cnt, y0, h, col, title, avg) => {
          ctx.fillStyle = "#888"; ctx.fillText(title, X0, y0 - 6);
          ctx.strokeStyle = "#ccc"; ctx.beginPath(); ctx.moveTo(X0, y0 + h); ctx.lineTo(X1, y0 + h); ctx.stroke();
          if (!cnt) return;
          const shares = Object.entries(obj).map(([k, v]) => [Number(k), v / cnt]);
          const m = Math.max(...shares.map(d => d[1]));
          ctx.fillStyle = col;
          for (const [k, p] of shares) { const bh = p / m * (h - 4); ctx.fillRect(X0 + k * bw, y0 + h - bh, Math.max(bw - 1, 1), bh); }
          const ax = X0 + avg * bw + bw / 2;
          ctx.strokeStyle = "#222"; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(ax, y0); ctx.lineTo(ax, y0 + h); ctx.stroke(); ctx.setLineDash([]);
          ctx.fillStyle = "#222"; ctx.fillText(`avg ${avg.toFixed(1)}`, ax + 4, y0 + 12);
        };
        const tAvg = S.tCnt ? S.tSum / S.tCnt : 0, sAvg = S.sCnt ? S.sSum / S.sCnt : 0;
        panel(S.t, S.tCnt, 30, 110, "#e4572e", `teachers' notes (${S.tCnt})`, tAvg);
        panel(S.s, S.sCnt, 170, 110, "#2e86ab", `students' notes (${S.sCnt})`, sAvg);
        ctx.fillStyle = "#888"; ctx.fillText("class size →", X1 - 70, 296);
      }
      const play = btn("▶ play", () => { if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; } else { setTimer(); play.textContent = "⏸ pause"; } });
      btn("step", () => { step(); draw(); });
      const ff = btn("⏩ fast", () => { fast = !fast; ff.textContent = fast ? "🐢 slow" : "⏩ fast"; if (timer) setTimer(); });
      btn("↺ reset", reset);
      ctl.appendChild(label);
      model.on("change:sizes", reset);
      reset();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        sizes = traitlets.List(traitlets.Int()).tag(sync=True)

    return (ClassroomSim,)


@app.cell
def classroom(ClassroomSim, class_sizes, mo):
    classroom = mo.ui.anywidget(ClassroomSim(sizes=class_sizes[:2000].tolist()))
    classroom
    return


@app.cell
def class_chart(alt, class_sizes, mo, np, pd):
    _n = class_sizes
    _teacher = _n.mean()
    _student = (_n**2).sum() / _n.sum()
    _cap = min(int(np.percentile(_n, 99.5)), 150)
    _k = np.arange(1, _cap + 1)
    _cnt = np.bincount(np.minimum(_n, _cap), minlength=_cap + 1)[1:]
    _df = pd.concat([
        pd.DataFrame({"class size": _k, "share of notes": _cnt / _cnt.sum(), "who is asked": "teachers"}),
        pd.DataFrame({"class size": _k, "share of notes": _cnt * _k / (_cnt * _k).sum(), "who is asked": "students"}),
    ])
    _colors = alt.Scale(domain=["teachers", "students"], range=["#e4572e", "#2e86ab"])
    _bars = alt.Chart(_df).mark_area(opacity=0.45, interpolate="step").encode(
        x=alt.X("class size:Q"), y=alt.Y("share of notes:Q", stack=None),
        color=alt.Color("who is asked:N", scale=_colors, legend=alt.Legend(orient="top", title=None)),
        tooltip=["who is asked", "class size", alt.Tooltip("share of notes:Q", format=".2%")],
    )
    _rules = alt.Chart(pd.DataFrame({"who is asked": ["teachers", "students"], "avg": [_teacher, _student]})).mark_rule(strokeWidth=2.5, strokeDash=[6, 3]).encode(
        x="avg:Q", color=alt.Color("who is asked:N", scale=_colors))
    mo.vstack([
        mo.hstack([
            mo.stat(f"{_teacher:.1f}", label="Teachers' average", caption="the true mean class size", bordered=True),
            mo.stat(f"{_student:.1f}", label="Students' average", caption=f"{(_student / _teacher - 1):+.0%} vs teachers", bordered=True),
            mo.stat(f"{_n.var() / _teacher:.1f}", label="Var(n) / E[n]", caption="theory predicts this gap", bordered=True),
        ], widths="equal"),
        (_bars + _rules).properties(width="container", height=230, title="20,000 classrooms: the same rooms, two different averages"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this:**
    1. Set the inequality slider to **0**. Every class is the same size and the paradox vanishes.
    2. Push the inequality up. The students' average runs away, even though the teachers' average stays put.
    3. In the animation, find a *big* class and count how many blue notes it adds to the bottom histogram at once.

    **Tool from the video: do a mental replay.** Imagine the process that generated each row of your data. Who got to write a note, and how many notes did they get?
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Riddle 2 · "How many siblings do you have?"

    Now the teacher keeps their own number to themselves and only asks the students. That seems fair: one student, one answer.

    But it still doesn't tell you how many children families have. Two things go wrong:

    * A family with *k* children sends **k** representatives into the room. This is the size bias from Riddle 1 again.
    * A family with **0 children sends no one**. That family is *invisible* to anyone surveying the room.

    Riddle 1 was distorted by the heavy end. This one is also distorted by an invisible **zero** end.
    """)
    return


@app.cell
def kids_controls(mo):
    kids_mean = mo.ui.slider(0.3, 5.0, step=0.1, value=1.8, label="Average kids per family", show_value=True)
    kids_model = mo.ui.dropdown(
        options=["Poisson (kids arrive at random)", "Geometric (many small, a few huge families)"],
        value="Poisson (kids arrive at random)", label="Family model")
    mo.hstack([kids_mean, kids_model], justify="start", gap=2)
    return kids_mean, kids_model


@app.cell
def kids_dist(kids_mean, kids_model, np):
    def family_distribution(mean, model, kmax=15):
        """P(family has k children), k = 0..kmax."""
        k = np.arange(kmax + 1)
        if model.startswith("Poisson"):
            from math import factorial
            p = np.array([np.exp(-mean) * mean**i / factorial(i) for i in k])
        else:
            q = mean / (1 + mean)  # geometric on {0,1,...} with the given mean
            p = (1 - q) * q**k
        return k, p / p.sum()

    kids_k, p_family = family_distribution(kids_mean.value, kids_model.value)
    # The room: each child reports (family size - 1) siblings; families are weighted by number of kids
    p_room_kids = kids_k * p_family / (kids_k * p_family).sum()
    return kids_k, p_family, p_room_kids


@app.cell
def kids_chart(alt, kids_k, mo, p_family, p_room_kids, pd):
    _rows = []
    for _k, _pf, _pr in zip(kids_k, p_family, p_room_kids):
        _rows.append({"kids in family": int(_k), "probability": _pf, "survey": "ask the parents (truth)"})
        _rows.append({"kids in family": int(_k), "probability": _pr, "survey": "ask the room (students)"})
    _df = pd.DataFrame(_rows)
    _colors = alt.Scale(domain=["ask the parents (truth)", "ask the room (students)"], range=["#2e86ab", "#e4572e"])
    _chart = alt.Chart(_df).mark_bar().encode(
        x=alt.X("kids in family:O"), xOffset="survey:N",
        y=alt.Y("probability:Q", axis=alt.Axis(format="%")),
        color=alt.Color("survey:N", scale=_colors, legend=alt.Legend(orient="top", title=None)),
        tooltip=["survey", "kids in family", alt.Tooltip("probability:Q", format=".1%")],
    ).properties(width="container", height=260, title="Same population, two surveys")
    _true = (kids_k * p_family).sum()
    _room = (kids_k * p_room_kids).sum()
    mo.vstack([
        mo.hstack([
            mo.stat(f"{_true:.2f}", label="True kids per family", bordered=True),
            mo.stat(f"{_room:.2f}", label="Kids per family, room survey", caption=f"= {_room - 1:.2f} siblings + yourself", bordered=True),
            mo.stat(f"{p_family[0]:.0%}", label="Childless families", caption="invisible to the room survey", bordered=True),
        ], widths="equal"),
        _chart,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🧩 Challenge from the video: can you reverse it?

    You only get to survey the room (the red bars). Can you reconstruct the parents' distribution (blue)?
    Each family with $k$ kids shows up $k$ times, so dividing by $k$ should undo the weighting:

    $$P_{\text{family}}(k) \propto \frac{P_{\text{room}}(k)}{k} \quad (k \ge 1)$$

    Toggle the switch to try it. Then look closely at the **k = 0** bar.
    """)
    return


@app.cell
def reverse_switch(mo):
    reverse_switch = mo.ui.switch(label="Reconstruct the parents' view from the room survey")
    reverse_switch
    return (reverse_switch,)


@app.cell
def reverse_chart(
    alt,
    kids_k,
    mo,
    np,
    p_family,
    p_room_kids,
    pd,
    reverse_switch,
):
    mo.stop(not reverse_switch.value, mo.md("_Switch on to reconstruct._"))
    _k1 = kids_k[1:]
    _recon = p_room_kids[1:] / _k1
    _recon = _recon / _recon.sum()
    _true_cond = p_family[1:] / p_family[1:].sum()
    _df = pd.concat([
        pd.DataFrame({"kids in family": kids_k, "probability": p_family, "which": "truth (incl. childless)"}),
        pd.DataFrame({"kids in family": np.r_[0, _k1], "probability": np.r_[np.nan, _recon], "which": "reconstructed from room"}),
    ])
    _c = alt.Chart(_df.dropna()).mark_bar().encode(
        x="kids in family:O", xOffset="which:N", y=alt.Y("probability:Q", axis=alt.Axis(format="%")),
        color=alt.Color("which:N", scale=alt.Scale(range=["#2e86ab", "#76b041"]), legend=alt.Legend(orient="top", title=None)),
    ).properties(width="container", height=220)
    mo.vstack([_c, mo.callout(mo.md(
        f"The reconstruction matches the truth **among families with kids** (max error {np.abs(_recon - _true_cond).max():.1e}). "
        f"But the **{p_family[0]:.0%} of childless families can never be recovered** from the room. No student represents them. "
        "Fixing the weighting is easy. Recovering people who never entered your sample is impossible without outside information."), kind="warn")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Riddle 3 · One day of players vs. everyone so far

    The analytics trap. Imagine an online game (the video uses real *World of Warcraft* logs). You want to know
    **how many hours per day a typical player plays**. You could look at:

    * 🔵 **everyone who played during a period** (for example, "all players so far"), or
    * 🔴 **everyone who played on a single survey day**.

    Same game, same players, same metric. Which group plays more hours per day? Make a guess first.

    Our simulated players come in two types, as in the video:
    **tourists** try the game once for a short session and never return, and **regulars** come back day after day.
    """)
    return


@app.cell
def r3_guess(mo):
    r3_guess = mo.ui.radio(options=["🔵 the period group", "about equal", "🔴 the single-day group"], label="Who plays more hours per day?", inline=True)
    r3_guess
    return (r3_guess,)


@app.cell
def player_controls(mo):
    tourist_share = mo.ui.slider(0.0, 0.95, step=0.05, value=0.6, label="Share of new players who are tourists", show_value=True)
    regular_life = mo.ui.slider(5, 300, step=5, value=90, label="Regulars' average lifetime (days)", show_value=True)
    mo.vstack([tourist_share, regular_life])
    return regular_life, tourist_share


@app.cell
def players(np, pd, regular_life, tourist_share):
    def simulate_players(tourist_share, regular_life, n=40_000, days=730, seed=1):
        """Each player: join day, last day (inclusive), hours played per active day."""
        rng = np.random.default_rng(seed)
        # popularity rises then fades, like a real game's population curve
        _w = np.sin(np.linspace(0.15, 2.6, days)) + 0.25
        join = rng.choice(days, size=n, p=_w / _w.sum())
        tourist = rng.random(n) < tourist_share
        life = np.where(tourist, 1, rng.geometric(1 / regular_life, n))
        hours = np.where(tourist, rng.gamma(2, 0.35 / 2, n), rng.gamma(3, 3.0 / 3, n))
        return pd.DataFrame({"join": join, "last": np.minimum(join + life - 1, days - 1), "tourist": tourist, "hours": hours})

    players = simulate_players(tourist_share.value, regular_life.value)
    _days = 730
    _diff = np.zeros(_days + 1)
    np.add.at(_diff, players["join"].to_numpy(), 1)
    np.add.at(_diff, players["last"].to_numpy() + 1, -1)
    daily_active = np.cumsum(_diff)[:_days]
    return daily_active, players


@app.cell
def timeline_widget(anywidget, traitlets):
    class SurveyTimeline(anywidget.AnyWidget):
        """Population curve with a draggable survey window: blue = [start, day], red = the survey day."""

        _esm = r"""
    function render({ model, el }) {
      const W = 700, H = 200, PL = 46, PR = 12, PT = 12, PB = 34;
      const NS = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none", font: "11px system-ui, sans-serif" });
      el.appendChild(svg);
      const mk = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); svg.appendChild(e); return e; };
      const toSvg = (evt) => { const p = svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      let drag = null;

      function draw() {
        const c = model.get("counts"), n = c.length, start = model.get("start"), day = model.get("day");
        const mx = Math.max(...c, 1);
        const sx = d => PL + d / (n - 1) * (W - PL - PR), sy = v => PT + (1 - v / mx) * (H - PT - PB);
        svg.innerHTML = "";
        mk("rect", { x: sx(start), y: PT, width: Math.max(sx(day) - sx(start), 2), height: H - PT - PB, fill: "#2e86ab", opacity: 0.18 });
        let d = `M${sx(0)},${sy(0)} ` + c.map((v, i) => `L${sx(i)},${sy(v)}`).join(" ") + ` L${sx(n - 1)},${sy(0)} Z`;
        mk("path", { d, fill: "#6c5b7b", opacity: 0.35, stroke: "#6c5b7b" });
        mk("line", { x1: sx(day), x2: sx(day), y1: PT, y2: H - PB, stroke: "#e4572e", "stroke-width": 2.5 });
        mk("line", { x1: PL, x2: W - PR, y1: H - PB, y2: H - PB, stroke: "#999" });
        for (let t = 0; t < n; t += 90) { const tx = mk("text", { x: sx(t), y: H - PB + 14, "text-anchor": "middle", fill: "#888" }); tx.textContent = "day " + t; }
        const yl = mk("text", { x: 4, y: PT + 10, fill: "#888" }); yl.textContent = "players/day";
        const ml = mk("text", { x: 4, y: PT + 24, fill: "#888" }); ml.textContent = Math.round(mx);
        const handle = (x, col, which, lab) => {
          const g = mk("circle", { cx: x, cy: H - PB, r: 9, fill: col, stroke: "white", "stroke-width": 2 });
          g.style.cursor = "ew-resize";
          g.addEventListener("pointerdown", e => { drag = which; svg.setPointerCapture(e.pointerId); e.stopPropagation(); });
          const t = mk("text", { x, y: H - 4, "text-anchor": "middle", fill: col, "font-weight": "bold" }); t.textContent = lab;
        };
        handle(sx(start), "#2e86ab", "start", `period start: ${start}`);
        handle(sx(day), "#e4572e", "day", `survey day: ${day}`);
      }
      svg.addEventListener("pointermove", e => {
        if (!drag) return;
        const n = model.get("counts").length;
        let v = Math.round((toSvg(e).x - PL) / (W - PL - PR) * (n - 1));
        v = Math.max(0, Math.min(n - 1, v));
        if (drag === "start") model.set("start", Math.min(v, model.get("day")));
        else { model.set("day", v); if (model.get("start") > v) model.set("start", v); }
      });
      const stop = () => { if (drag) { drag = null; model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointerleave", stop);
      for (const t of ["counts", "start", "day"]) model.on("change:" + t, draw);
      draw();
    }
    export default { render };
    """
        counts = traitlets.List(traitlets.Float()).tag(sync=True)
        start = traitlets.Int(0).tag(sync=True)
        day = traitlets.Int(365).tag(sync=True)

    return (SurveyTimeline,)


@app.cell
def timeline(SurveyTimeline, daily_active, mo):
    timeline = mo.ui.anywidget(SurveyTimeline(counts=daily_active.round(1).tolist(), start=0, day=365))
    mo.vstack([mo.md("**Drag the blue handle** (start of the period) and **the red handle** (survey day)."), timeline])
    return (timeline,)


@app.cell
def player_chart(alt, mo, np, pd, players, timeline):
    _start, _day = timeline.value["start"], timeline.value["day"]
    _p = players
    _blue = _p[(_p["join"] <= _day) & (_p["last"] >= _start)]
    _red = _p[(_p["join"] <= _day) & (_p["last"] >= _day)]
    _bins = np.linspace(0, 10, 41)
    def _hist(df, label):
        c, e = np.histogram(df["hours"].clip(upper=9.99), bins=_bins)
        return pd.DataFrame({"lo": e[:-1], "hi": e[1:], "share": c / max(c.sum(), 1), "group": label})
    _lab_b = f"🔵 played during day {_start}–{_day}"
    _lab_r = f"🔴 played on day {_day}"
    _df = pd.concat([_hist(_blue, _lab_b), _hist(_red, _lab_r)])
    _col = alt.Scale(domain=[_lab_b, _lab_r], range=["#2e86ab", "#e4572e"])
    _ch = alt.Chart(_df).mark_bar(opacity=0.5).encode(
        x=alt.X("lo:Q", bin="binned", title="hours played per active day"), x2="hi:Q",
        y=alt.Y("share:Q", stack=None, axis=alt.Axis(format="%"), title="share of players"),
        color=alt.Color("group:N", scale=_col, legend=alt.Legend(orient="top", title=None)),
    ).properties(width="container", height=240)
    _mb, _mr = _blue["hours"].mean(), _red["hours"].mean()
    mo.vstack([
        mo.hstack([
            mo.stat(f"{_mb:.2f} h", label="🔵 avg hours/day, period group", caption=f"{len(_blue):,} players · {_blue['tourist'].mean():.0%} tourists", bordered=True),
            mo.stat(f"{_mr:.2f} h", label="🔴 avg hours/day, single day", caption=f"{len(_red):,} players · {_red['tourist'].mean():.0%} tourists", bordered=True),
            mo.stat(f"{_mr / _mb:.1f}×" if _mb > 0 else "–", label="ratio red / blue", bordered=True),
        ], widths="equal"),
        _ch,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Why?** Recreate the video's experiment: drag the **blue handle onto the red one**. Both groups are now identical (purple). Then widen the period one day at a time.

    * A **tourist** is in your data for exactly one day. Every extra day in the period brings in a fresh batch of them.
    * A **regular** is in the data almost every day. Widening the period adds hardly any new regulars, because you had already counted them.

    A long period therefore piles up light users, while a single day is dominated by the heavy users who keep coming back.
    **It's Riddle 1 again:** sampling "who is here today" weights each player by how many days they play.

    Set tourists to **0%** and most of the gap disappears. Now make regulars' lives very short and watch what happens.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo, r3_guess):
    mo.md(
        "" if r3_guess.value is None else
        ("✅ **You called it.** The single-day crowd is dominated by heavy users." if r3_guess.value.startswith("🔴")
         else "🤔 **Most people guess that.** Look at the share of tourists in each group above.")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Bonus · Why is your bus always late?

    This is a classic from Allen Downey's *Probably Overthinking It*, which the video recommends.
    Buses arrive on average every **10 minutes**. You show up at a random time. How long do you wait?
    "Half the gap, so 5 minutes" is only right if the buses are **perfectly regular**.

    You are far more likely to arrive during a *long* gap, because long gaps cover more of the clock.
    That's the size bias again, this time in time instead of in people:

    $$E[\text{wait}] = \frac{E[g^2]}{2\,E[g]} = \frac{E[g]}{2}\left(1 + \mathrm{CV}^2\right)$$

    **Click on the timeline to arrive at the stop**, or drop 200 random passengers.
    """)
    return


@app.cell
def bus_cv(mo):
    bus_cv = mo.ui.slider(0.0, 2.0, step=0.1, value=1.0, label="Bus irregularity (CV of gaps: 0 = perfect schedule, 1 = random, 2 = bunching)", show_value=True, full_width=True)
    bus_cv
    return (bus_cv,)


@app.cell
def bus_gaps(bus_cv, np):
    _rng = np.random.default_rng(3)
    if bus_cv.value < 0.01:
        bus_gaps = np.full(40, 10.0)
    else:
        _shape = 1 / bus_cv.value**2
        bus_gaps = _rng.gamma(_shape, 10 / _shape, 40)
    bus_gaps = bus_gaps * (400 / bus_gaps.sum())  # 400 minutes shown, mean gap 10
    return (bus_gaps,)


@app.cell
def bus_widget(anywidget, traitlets):
    class BusStop(anywidget.AnyWidget):
        """Timeline of buses; click to arrive, see your wait. Arrival times sync back to Python."""

        _esm = r"""
    function render({ model, el }) {
      const W = 700, H = 150, PL = 10, PR = 10, Y = 70;
      const NS = "http://www.w3.org/2000/svg";
      const bar = document.createElement("div");
      Object.assign(bar.style, { display: "flex", gap: "6px", marginBottom: "4px", font: "13px system-ui, sans-serif", alignItems: "center" });
      el.appendChild(bar);
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", cursor: "crosshair", font: "11px system-ui, sans-serif" });
      el.appendChild(svg);
      const mk = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); svg.appendChild(e); return e; };
      const toSvg = (evt) => { const p = svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const btn = (t, fn) => { const b = document.createElement("button"); b.textContent = t; b.onclick = fn;
        Object.assign(b.style, { padding: "3px 10px", borderRadius: "12px", border: "1px solid #bbb", cursor: "pointer" }); bar.appendChild(b); };
      const info = document.createElement("span"); info.style.marginLeft = "8px";

      const times = () => { const g = model.get("gaps"); let t = 0; return g.map(x => (t += x)); };
      const total = () => model.get("gaps").reduce((a, b) => a + b, 0);
      const sx = t => PL + t / total() * (W - PL - PR);
      const waitFor = (t) => { const bt = times(); for (const b of bt) if (b >= t) return [b - t, b]; return [null, null]; };

      function draw() {
        svg.innerHTML = "";
        const bt = times(), arr = model.get("arrivals");
        mk("line", { x1: PL, x2: W - PR, y1: Y, y2: Y, stroke: "#999", "stroke-width": 2 });
        let prev = 0;
        bt.forEach((b) => {
          const g = b - prev;
          mk("rect", { x: sx(prev), y: Y - 3 - Math.min(g, 40), width: sx(b) - sx(prev), height: Math.min(g, 40), fill: g > 15 ? "#f18f01" : "#6c5b7b", opacity: 0.15 });
          const r = mk("text", { x: sx(b), y: Y + 4, "text-anchor": "middle", "font-size": 14 }); r.textContent = "🚌";
          prev = b;
        });
        for (let t = 0; t <= total(); t += 60) { const tx = mk("text", { x: sx(t), y: H - 6, "text-anchor": "middle", fill: "#888" }); tx.textContent = t + " min"; }
        let sum = 0, cnt = 0;
        arr.forEach((t, i) => {
          const [w, b] = waitFor(t); if (w === null) return;
          sum += w; cnt++;
          const yy = Y + 18 + (i % 6) * 7;
          mk("line", { x1: sx(t), x2: sx(b), y1: yy, y2: yy, stroke: "#e4572e", "stroke-width": 1.5, opacity: 0.6 });
          mk("circle", { cx: sx(t), cy: yy, r: 2.5, fill: "#2e86ab" });
        });
        const last = arr[arr.length - 1];
        if (last !== undefined) { const [w] = waitFor(last); if (w !== null && arr.length < 5) {
          const tx = mk("text", { x: sx(last), y: 14, "text-anchor": "middle", fill: "#e4572e", "font-weight": "bold" }); tx.textContent = `you wait ${w.toFixed(1)} min`; } }
        info.innerHTML = cnt ? `passengers: <b>${cnt}</b> · their average wait: <b>${(sum / cnt).toFixed(1)} min</b>` : "click the timeline to arrive at the stop";
      }
      svg.addEventListener("click", e => {
        const t = (toSvg(e).x - PL) / (W - PL - PR) * total();
        if (t < 0 || t > total()) return;
        model.set("arrivals", [...model.get("arrivals"), t]); model.save_changes();
      });
      btn("👥 drop 200 random passengers", () => {
        const T = total(); model.set("arrivals", [...model.get("arrivals"), ...Array.from({ length: 200 }, () => Math.random() * T)]); model.save_changes();
      });
      btn("clear", () => { model.set("arrivals", []); model.save_changes(); });
      bar.appendChild(info);
      model.on("change:gaps", () => { model.set("arrivals", []); draw(); });
      model.on("change:arrivals", draw);
      draw();
    }
    export default { render };
    """
        gaps = traitlets.List(traitlets.Float()).tag(sync=True)
        arrivals = traitlets.List(traitlets.Float()).tag(sync=True)

    return (BusStop,)


@app.cell
def bus(BusStop, bus_gaps, mo):
    bus = mo.ui.anywidget(BusStop(gaps=bus_gaps.round(3).tolist(), arrivals=[]))
    bus
    return (bus,)


@app.cell
def bus_stats(bus, bus_gaps, mo, np):
    _g = bus_gaps
    _bt = np.cumsum(_g)
    _arr = np.array(bus.value["arrivals"], dtype=float)
    _arr = _arr[_arr <= _bt[-1]]
    _waits = _bt[np.searchsorted(_bt, _arr)] - _arr if len(_arr) else np.array([])
    _theory = (_g**2).mean() / (2 * _g.mean())
    mo.hstack([
        mo.stat(f"{_g.mean() / 2:.1f} min", label="Naive guess: half the average gap", bordered=True),
        mo.stat(f"{_theory:.1f} min", label="Theory: E[g²] / 2E[g]", caption="for this exact schedule", bordered=True),
        mo.stat(f"{_waits.mean():.1f} min" if len(_waits) else "–", label="Your passengers' average wait", caption=f"{len(_waits)} passengers", bordered=True),
        mo.stat(f"{(np.repeat(_g, np.maximum(1, np.round(_g * 10).astype(int)))).mean():.1f} min", label="Avg gap experienced by a passenger", caption=f"vs {_g.mean():.0f} min schedule average", bordered=True),
    ], widths="equal")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spot the paradox

    The video's real promise: once you know it, **you'll notice it everywhere**. For each scenario, does the inspection paradox bias the number?
    """)
    return


@app.cell
def quiz(mo):
    quiz_items = {
        "Your friends have more friends than you do, on average.": ("yes", "The friendship paradox. People with many friends appear in many friend lists, so they are sampled in proportion to their popularity."),
        "Surveying current prisoners gives a longer average sentence than the court records.": ("yes", "Long sentences stay in the prison longer, so on any given day they are over-represented."),
        "On the highway, most cars seem to overtake you or be overtaken; few drive your speed.": ("yes", "The cars you encounter are weighted by their speed difference from you. Same-speed cars are never 'inspected'."),
        "The average of 1,000 fair die rolls is about 3.5.": ("no", "Each roll is sampled exactly once and the sampling doesn't depend on the value. There's no paradox here."),
        "Gym members you meet at the gym work out more often than the average member.": ("yes", "Frequent visitors are there more often, which is the same as the single-day players in Riddle 3."),
        "Patients in a hospital today have longer stays than the average admission.": ("yes", "Long stays occupy beds on more days, so a one-day snapshot oversamples them."),
    }
    quiz = mo.ui.dictionary({_q: mo.ui.radio(["yes", "no"], inline=True) for _q in quiz_items})
    quiz_reveal = mo.ui.switch(label="Reveal answers")
    mo.vstack([mo.vstack([mo.hstack([quiz[_q], mo.md(_q)], justify="start") for _q in quiz_items]), quiz_reveal])
    return quiz, quiz_items, quiz_reveal


@app.cell
def quiz_result(mo, quiz, quiz_items, quiz_reveal):
    mo.stop(not quiz_reveal.value, mo.md("_Answer, then flip the switch._"))
    _rows, _score = [], 0
    for _q, (_ans, _why) in quiz_items.items():
        _ok = quiz.value[_q] == _ans
        _score += _ok
        _rows.append(f"| {'✅' if _ok else '❌'} | {_q} | **{_ans}** | {_why} |")
    mo.md(f"### Score: {_score}/{len(quiz_items)} {'🏆' if _score == len(quiz_items) else ''}\n\n| | scenario | paradox? | why |\n|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap: how to defend yourself

    | Riddle | Who gets over-sampled | Who goes missing |
    |---|---|---|
    | Teacher vs students | big classes (n notes each) | – |
    | Siblings | big families | childless families (zero kids, zero answers) |
    | Players per day | heavy, returning users | one-off tourists pile up in long periods |
    | Bus stop | long gaps | short gaps |

    1. **Mental replay**: simulate the process that created each row. Who got to write a note, and how many?
    2. **Check the extremes**: a class of 1 vs. a class of 100. A family of 0. A player who quits after 10 minutes.
    3. **Name your unit of averaging**: per class, per student, per player, per player-day? Each one answers a *different* question.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Danger, or totally fine?** As the video ends, a metric is a *proxy*. A biased number can still be useful if it moves together with what you care about.
    The real danger is **comparing numbers built with different sampling**, like a per-day dashboard against a per-period report, and concluding that behaviour changed.

    📚 **Further reading**: *Probably Overthinking It* by Allen B. Downey, which has a whole chapter on this paradox.
    """), kind="success")
    return


if __name__ == "__main__":
    app.run()

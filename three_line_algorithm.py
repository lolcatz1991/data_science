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
    # AI lost to a 3-line algorithm: PID vs. neural networks
    *An interactive companion to Graduate ML's video **"AI Lost to A 3 Line Algorithm"**.*

    > You probably think you need AI to keep a self-balancing robot upright. A controller so simple it fits in **three lines**
    > (proportional, integral, derivative) does the job. So when do you actually *need* the neural network?
    """),
        mo.Html('<iframe width="560" height="315" src="https://www.youtube.com/embed/CqE2yToKvmE" title="AI Lost to A 3 Line Algorithm" frameborder="0" allowfullscreen></iframe>'),
    ])
    return


@app.cell(hide_code=True)
def plan(mo):
    mo.md(r"""
    Google's first rule of ML ("don't be afraid to launch without machine learning") is the thread through this lesson. We follow the video's order:

    | # | Section | What you'll touch |
    |---|---|---|
    | 1 | Calculus in 60 seconds | rectangles under a curve, a secant line that shrinks into a tangent |
    | 2 | PID: the three-part story | a **live balancing robot** with Kp / Ki / Kd knobs |
    | 3 | Neurons, and why they need a bend | stack linear layers until they collapse, then **drag a response curve** that ReLUs must fit |
    | 4 | Choosing between AI and PID | add a load, climb a hill, face unknown worlds |
    | 5 | Sandbox, quiz, recap | free play and a self-check |

    **Color language (used everywhere):** <span style="color:#e4572e">**red = error / lean angle θ**</span>,
    <span style="color:#f18f01">**orange = P term**</span>, <span style="color:#6c5b7b">**purple = I term**</span>,
    <span style="color:#2a9d8f">**teal = D term**</span>, <span style="color:#3b82f6">**blue = neural network / ReLU**</span>, grey = the target or a reference.
    """)
    return


@app.cell(hide_code=True)
def calc_intro(mo):
    mo.md(r"""
    ## 1 · Calculus in under 60 seconds

    The robot's controller uses exactly two ideas from calculus, so we'll start there, as the video does.

    **Integral: stack rectangles.** Put rectangles under the curve and add up their areas. Make them thinner, and thinner, *infinitely* thin,
    and the sum becomes the area under the curve:
    $$\int_0^T e(t)\,dt \;=\; \lim_{n\to\infty}\sum_{k=1}^{n} e(t_k)\,\Delta t .$$

    **Derivative: squeeze a secant.** Take two points, find the slope between them, then push them infinitely close together:
    $$\frac{de}{dt}(t_0) \;=\; \lim_{h\to 0}\frac{e(t_0+h)-e(t_0)}{h}.$$

    The curve below is an **error signal** $e(t)$ (red), how far a robot is from upright over time. Area *below* zero counts as negative.
    """)
    return


@app.cell
def calc_controls(mo):
    n_rects = mo.ui.slider(2, 200, value=6, label="rectangles n", show_value=True)
    t0_slider = mo.ui.slider(0.2, 4.0, step=0.05, value=1.2, label="point t₀", show_value=True)
    h_slider = mo.ui.slider(0.01, 1.5, step=0.01, value=0.8, label="gap h", show_value=True)
    mo.hstack([n_rects, t0_slider, h_slider], justify="start")
    return h_slider, n_rects, t0_slider


@app.cell
def calc_charts(alt, h_slider, mo, n_rects, np, pd, t0_slider):
    def err_curve(t):
        return 0.8 * np.exp(-t / 1.6) * np.cos(2.6 * t)

    def err_slope(t):
        return 0.8 * np.exp(-t / 1.6) * (-np.cos(2.6 * t) / 1.6 - 2.6 * np.sin(2.6 * t))

    _T = 5.0
    _tt = np.linspace(0, _T, 600)
    _curve = pd.DataFrame({"t": _tt, "e": err_curve(_tt)})
    _n = n_rects.value
    _edges = np.linspace(0, _T, _n + 1)
    _rect = pd.DataFrame({"t0": _edges[:-1], "t1": _edges[1:], "e": err_curve(_edges[:-1])})
    _riemann = float((_rect.e * (_T / _n)).sum())
    _exact = float(np.trapezoid(err_curve(np.linspace(0, _T, 20001)), np.linspace(0, _T, 20001)))

    _base = alt.Chart(_curve).mark_line(color="#e4572e", strokeWidth=2.5).encode(
        x=alt.X("t:Q", title="time t (s)"), y=alt.Y("e:Q", title="error e(t)"))
    _bars = alt.Chart(_rect).mark_rect(opacity=0.35, stroke="#6c5b7b", color="#6c5b7b").encode(
        x="t0:Q", x2="t1:Q", y=alt.datum(0), y2="e:Q")
    _integral_chart = (_bars + _base).properties(width=330, height=230, title=f"Σ of {_n} rectangles = {_riemann:.4f}")

    _t0, _h = t0_slider.value, h_slider.value
    _secant = (err_curve(_t0 + _h) - err_curve(_t0)) / _h
    _true = float(err_slope(_t0))
    _tl = np.linspace(max(0, _t0 - 0.8), min(_T, _t0 + _h + 0.8), 50)
    _lines = pd.concat([
        pd.DataFrame({"t": _tl, "e": err_curve(_t0) + _secant * (_tl - _t0), "line": "secant"}),
        pd.DataFrame({"t": _tl, "e": err_curve(_t0) + _true * (_tl - _t0), "line": "true tangent"}),
    ])
    _pts = pd.DataFrame({"t": [_t0, _t0 + _h], "e": [err_curve(_t0), err_curve(_t0 + _h)]})
    _deriv_chart = (
        _base
        + alt.Chart(_lines).mark_line(strokeWidth=2).encode(
            x="t:Q", y="e:Q",
            color=alt.Color("line:N", scale=alt.Scale(domain=["secant", "true tangent"], range=["#2a9d8f", "#999"]), title=None),
            strokeDash=alt.condition(alt.datum.line == "true tangent", alt.value([5, 4]), alt.value([1, 0])))
        + alt.Chart(_pts).mark_circle(size=90, color="#2a9d8f").encode(x="t:Q", y="e:Q")
    ).properties(width=330, height=230, title=f"secant slope {_secant:.3f} vs tangent {_true:.3f}")

    mo.vstack([
        mo.hstack([_integral_chart, _deriv_chart]),
        mo.hstack([
            mo.stat(f"{_riemann:.4f}", label="rectangle sum", caption=f"true area {_exact:.4f}"),
            mo.stat(f"{abs(_riemann - _exact):.4f}", label="area error"),
            mo.stat(f"{_secant:.3f}", label="secant slope", caption=f"true slope {_true:.3f}"),
            mo.stat(f"{abs(_secant - _true):.3f}", label="slope error"),
        ], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def calc_try(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Drag **n** from 2 to 200. Watch the area error shrink roughly like $1/n$. That limit is the integral.
    2. Shrink **h** toward 0.01. The teal secant snaps onto the dashed tangent. That limit is the derivative.
    3. Put $t_0$ on a peak of the wave. The true slope is ≈ 0 there: the error has stopped changing *for an instant*.
       The PID's D term will read exactly this number.
    """), kind="info")
    return


@app.cell(hide_code=True)
def pid_intro(mo):
    mo.md(r"""
    ## 2 · PID: a big equation that is a simple three-part story

    $$
    u(t) \;=\; \underbrace{\color{#f18f01}{K_p\, e(t)}}_{\text{now}}
    \;+\; \underbrace{\color{#6c5b7b}{K_i \int_0^t e(\tau)\,d\tau}}_{\text{memory}}
    \;+\; \underbrace{\color{#2a9d8f}{K_d\, \frac{de}{dt}}}_{\text{trend}}
    $$

    * **P** reacts to the **current** error: lean more, push harder.
    * **I** reacts to the error **accumulated over time** (the area under the error curve from section 1). It gives the robot *memory*.
    * **D** reacts to **how quickly the error is changing** (the slope from section 1). It makes the robot *less jerky*.
    * Each **K** is a **gain** you tune by hand, using physics or brute force. $u(t)$ is how hard the motor drives.

    In code, that is three lines:
    ```python
    integral += e * dt
    derivative = (e - e_prev) / dt
    u = Kp * e + Ki * integral + Kd * derivative
    ```

    Our robot is an inverted pendulum on a wheel. The lean angle $\theta$ obeys
    $\ddot\theta = \tfrac{g}{L}\sin\theta + \tfrac{u}{m}\cos\theta + d$, where $m$ is the load, $d$ is a disturbance (a hill), and the error is $e=-\theta$ (target: upright).
    The simulator below runs this physics live in your browser.
    """)
    return


@app.cell
def simulate_fn(np, pd):
    def simulate(Kp, Ki, Kd, mass=1.0, slope_deg=0.0, theta0=0.2, T=6.0, dt=0.005,
                 umax=80.0, noise=0.0, seed=0, g_over_L=19.6):
        """Inverted-pendulum robot under PID control. Returns (DataFrame, fallen)."""
        _rng = np.random.default_rng(seed)
        th, om, integ = theta0, 0.0, 0.0
        e_prev = -theta0
        dist = g_over_L * np.sin(np.radians(slope_deg))
        rows, fallen = [], False
        for k in range(int(T / dt)):
            e = -(th + noise * _rng.standard_normal())       # error = target(0) - measured lean
            integ += e * dt                                   # line 1: integral
            deriv = (e - e_prev) / dt                         # line 2: derivative
            e_prev = e
            P, I, D = Kp * e, Ki * integ, Kd * deriv
            u = float(np.clip(P + I + D, -umax, umax))         # line 3: PID (+ motor limit)
            alpha = g_over_L * np.sin(th) + (u / mass) * np.cos(th) + dist
            om += alpha * dt
            th += om * dt
            rows.append((k * dt, th, P, I, D, u))
            if abs(th) > np.pi / 2:
                fallen = True
                break
        df = pd.DataFrame(rows, columns=["t", "theta", "P", "I", "D", "u"])
        return df, fallen

    EFFORT = 1e-3   # λ: price of motor effort

    def cost(df, fallen, T=6.0):
        """∫(θ² + λ·u²)dt: lean as little as possible without burning the motor. Falling over is a huge penalty."""
        return 100.0 + (T - df.t.iloc[-1]) * 10 if fallen else float(((df.theta ** 2) + EFFORT * df.u ** 2).sum() * (df.t.iloc[1] - df.t.iloc[0]))

    _df, _f = simulate(40, 20, 6)
    f"sanity: default gains fall over? {_f}; final lean {_df.theta.iloc[-1]:.4f} rad"
    return EFFORT, cost, simulate


@app.cell
def PidRobot(anywidget, traitlets):
    class PidRobot(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 640, H = 300, A = 19.6, DT = 0.005, UMAX = 80;
      const wrap = document.createElement("div");
      wrap.style.fontFamily = "system-ui, sans-serif";
      el.appendChild(wrap);
      const ctl = document.createElement("div");
      ctl.style.cssText = "display:flex;flex-wrap:wrap;gap:14px;align-items:center;margin-bottom:6px";
      wrap.appendChild(ctl);
      const sliders = {};
      const mkSlider = (key, label, max, step, color) => {
        const box = document.createElement("label");
        box.style.cssText = `color:${color};font-weight:600;font-size:13px`;
        const s = document.createElement("input");
        s.type = "range"; s.min = 0; s.max = max; s.step = step; s.value = model.get(key);
        s.style.verticalAlign = "middle";
        const v = document.createElement("span"); v.textContent = ` ${(+s.value).toFixed(1)}`;
        s.oninput = () => { v.textContent = ` ${(+s.value).toFixed(1)}`; model.set(key, +s.value); };
        s.onchange = () => model.save_changes();          // sync to Python on release only
        box.append(label + " ", s, v); ctl.appendChild(box); sliders[key] = [s, v];
      };
      mkSlider("kp", "Kp", 200, 1, "#f18f01");
      mkSlider("ki", "Ki", 100, 1, "#6c5b7b");
      mkSlider("kd", "Kd", 30, 0.1, "#2a9d8f");
      const btns = document.createElement("div"); btns.style.cssText = "display:flex;gap:6px;margin-bottom:6px"; wrap.appendChild(btns);
      const btn = (t, fn) => { const b = document.createElement("button"); b.textContent = t; b.onclick = fn; b.style.cssText = "padding:3px 10px;border-radius:6px;cursor:pointer"; btns.appendChild(b); return b; };
      const canvas = document.createElement("canvas"); canvas.width = W; canvas.height = H;
      canvas.style.cssText = "width:100%;max-width:640px;border-radius:10px;background:rgba(127,127,127,0.08)";
      wrap.appendChild(canvas);
      const c = canvas.getContext("2d");
      let st, timer = null;
      const reset = () => { st = { th: 0.2, om: 0, integ: 0, ep: -0.2, t: 0, fallen: false, P: 0, I: 0, D: 0, u: 0, trail: [] }; };
      reset();
      function step() {
        if (st.fallen) return;
        const kp = model.get("kp"), ki = model.get("ki"), kd = model.get("kd"), m = model.get("mass");
        const e = -st.th; st.integ += e * DT; const d = (e - st.ep) / DT; st.ep = e;
        st.P = kp * e; st.I = ki * st.integ; st.D = kd * d;
        st.u = Math.max(-UMAX, Math.min(UMAX, st.P + st.I + st.D));
        const al = A * Math.sin(st.th) + (st.u / m) * Math.cos(st.th);
        st.om += al * DT; st.th += st.om * DT; st.t += DT;
        if (Math.abs(st.th) > Math.PI / 2) { st.fallen = true; st.th = Math.sign(st.th) * Math.PI / 2; }
      }
      function draw() {
        c.clearRect(0, 0, W, H);
        const gx = 170, gy = 210;
        c.strokeStyle = "#888"; c.lineWidth = 2; c.beginPath(); c.moveTo(20, gy + 22); c.lineTo(320, gy + 22); c.stroke();
        c.setLineDash([4, 4]); c.beginPath(); c.moveTo(gx, gy); c.lineTo(gx, gy - 150); c.stroke(); c.setLineDash([]);
        c.save(); c.translate(gx, gy); c.rotate(st.th);
        c.fillStyle = st.fallen ? "#e4572e" : "#6c757d"; c.fillRect(-16, -140, 32, 140);
        c.fillStyle = "#e4572e"; c.beginPath(); c.arc(0, -140, 9, 0, 7); c.fill();
        c.restore();
        c.fillStyle = "#444"; c.beginPath(); c.arc(gx, gy, 22, 0, 7); c.fill();
        c.fillStyle = "#e4572e"; c.font = "bold 14px system-ui";
        c.fillText(`θ = ${(st.th * 180 / Math.PI).toFixed(1)}°`, 30, 30);
        c.fillStyle = "#888"; c.font = "13px system-ui"; c.fillText(`t = ${st.t.toFixed(1)} s   load m = ${model.get("mass")}`, 30, 50);
        if (st.fallen) { c.fillStyle = "#e4572e"; c.font = "bold 26px system-ui"; c.fillText("FELL OVER", 60, 100); }
        // P/I/D bars
        const bx = 360, by = 40, bw = 250;
        [["P", st.P, "#f18f01"], ["I", st.I, "#6c5b7b"], ["D", st.D, "#2a9d8f"], ["u", st.u, "#444"]].forEach(([n, v, col], i) => {
          const y = by + i * 34, mid = bx + bw / 2, len = Math.max(-1, Math.min(1, v / UMAX)) * bw / 2;
          c.fillStyle = "rgba(127,127,127,0.18)"; c.fillRect(bx, y, bw, 22);
          c.fillStyle = col; c.fillRect(Math.min(mid, mid + len), y, Math.abs(len), 22);
          c.fillStyle = "#888"; c.fillText(`${n} ${v.toFixed(1)}`, bx - 0, y - 3 + 0);
        });
        // trail of theta
        st.trail.push(st.th); if (st.trail.length > 250) st.trail.shift();
        const ty = 250, tx = 360;
        c.strokeStyle = "#999"; c.lineWidth = 1; c.beginPath(); c.moveTo(tx, ty); c.lineTo(tx + 250, ty); c.stroke();
        c.strokeStyle = "#e4572e"; c.lineWidth = 2; c.beginPath();
        st.trail.forEach((v, i) => { const x = tx + i, y = ty - v * 80; i ? c.lineTo(x, y) : c.moveTo(x, y); });
        c.stroke(); c.fillStyle = "#888"; c.fillText("lean θ (recent)", tx, ty + 40);
      }
      function frame() { for (let k = 0; k < 3; k++) step(); draw(); }
      const play = btn("⏸ pause", () => {
        if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; }
        else { timer = setInterval(frame, 16); play.textContent = "⏸ pause"; } });
      btn("👉 push", () => { st.om += 1.5; });
      btn("👈 push", () => { st.om -= 1.5; });
      btn("↺ reset", () => { reset(); draw(); });
      model.on("change:mass", () => { reset(); draw(); });
      ["kp", "ki", "kd"].forEach(k => model.on("change:" + k, () => { const [s, v] = sliders[k]; s.value = model.get(k); v.textContent = ` ${(+model.get(k)).toFixed(1)}`; }));
      timer = setInterval(frame, 16);
      return () => clearInterval(timer);
    }
    export default { render };
    """
        kp = traitlets.Float(40.0).tag(sync=True)
        ki = traitlets.Float(20.0).tag(sync=True)
        kd = traitlets.Float(6.0).tag(sync=True)
        mass = traitlets.Float(1.0).tag(sync=True)

    return (PidRobot,)


@app.cell
def pid_robot(PidRobot, mo):
    pid_robot = mo.ui.anywidget(PidRobot())
    pid_robot
    return (pid_robot,)


@app.cell
def pid_readout(alt, cost, mo, pid_robot, simulate):
    _g = pid_robot.value
    gains = (float(_g["kp"]), float(_g["ki"]), float(_g["kd"]))
    _df, _fell = simulate(*gains)
    _long = _df.melt(id_vars="t", value_vars=["P", "I", "D"], var_name="term", value_name="value")
    _theta = alt.Chart(_df).mark_line(color="#e4572e", strokeWidth=2.5).encode(
        x=alt.X("t:Q", title="time (s)"), y=alt.Y("theta:Q", title="lean θ (rad)")).properties(width=640, height=170,
        title="Lean angle after starting tilted at θ = 0.2 rad")
    _terms = alt.Chart(_long).mark_line(strokeWidth=2).encode(
        x=alt.X("t:Q", title="time (s)"), y=alt.Y("value:Q", title="motor contribution"),
        color=alt.Color("term:N", scale=alt.Scale(domain=["P", "I", "D"], range=["#f18f01", "#6c5b7b", "#2a9d8f"]))
    ).properties(width=640, height=170, title="Who is doing the work? Each term's share of u(t)")
    _settle = _df.t[(_df.theta.abs() > 0.02)].max() if not _fell else float("nan")
    mo.vstack([
        mo.md(f"**Python replay of the same gains** (Kp={gains[0]:.1f}, Ki={gains[1]:.1f}, Kd={gains[2]:.1f}). Updates when you release a slider."),
        mo.hstack([
            mo.stat("FELL" if _fell else "upright", label="outcome"),
            mo.stat(f"{_df.theta.abs().max():.3f}", label="max |θ| (rad)"),
            mo.stat("—" if _fell else f"{_settle:.2f} s", label="settles within ±0.02 rad"),
            mo.stat(f"{cost(_df, _fell):.4f}", label="cost ∫(θ² + λu²)dt"),
        ], justify="start"),
        alt.vconcat(_theta, _terms),
    ])
    return (gains,)


@app.cell(hide_code=True)
def pid_try(mo):
    mo.callout(mo.md(r"""
    **Try this** (on the live robot, then read the Python replay)

    1. **Kp below ≈ 20** → it falls. Gravity's pull ($g/L ≈ 19.6$) beats a weak proportional push. P must out-muscle physics.
    2. **Kd → 0** with Kp = 40: it wobbles forever (or wobbles itself over). D brakes the swing, which is the "less jerky" term.
    3. **Ki → 0** here barely matters: on flat ground nothing pushes steadily. Remember this for the *hill* in section 4.
    4. Hit **👉 push** a few times. A well-tuned PID recovers in about a second, and nobody trained it.
    """), kind="info")
    return


@app.cell(hide_code=True)
def nn_intro(mo):
    mo.md(r"""
    ## 3 · Neural networks, and why they need to get "curve-ier"

    A neuron takes inputs, multiplies them by **weights**, adds a **bias**: $y = w\cdot x + b$, which is just $y = mx + b$.
    Stack many neurons into layers and... you'd think you get something powerful.

    **The catch:** a weighted sum of weighted sums is still a weighted sum.
    $$W_3\big(W_2(W_1x+b_1)+b_2\big)+b_3 \;=\; \underbrace{(W_3W_2W_1)}_{W}\,x + \underbrace{(\dots)}_{b}$$
    Without an activation function, **a deep network is no different from a single layer**. Check it yourself below.
    """)
    return


@app.cell
def collapse_controls(mo):
    depth = mo.ui.slider(1, 12, value=4, label="number of linear layers", show_value=True)
    reroll = mo.ui.button(label="🎲 new random weights")
    mo.hstack([depth, reroll], justify="start")
    return depth, reroll


@app.cell
def collapse_demo(alt, depth, mo, np, pd, reroll):
    reroll
    _rng = np.random.default_rng()
    _x = np.linspace(-3, 3, 61)[:, None]          # 1 input (lean angle) -> 1 output (motor)
    _h = _x.copy()
    _W, _b = np.eye(1), np.zeros(1)
    for _k in range(depth.value):
        _wk = _rng.normal(0, 1, (_h.shape[1], 1 if _k == depth.value - 1 else 8))
        _bk = _rng.normal(0, 0.5, _wk.shape[1])
        _h = _h @ _wk + _bk                        # a layer of neurons, no activation
        _W, _b = _W @ _wk, _b @ _wk + _bk          # the equivalent single neuron
    _single = _x @ _W + _b
    _df = pd.DataFrame({"x": _x[:, 0], f"{depth.value}-layer network": _h[:, 0], "single neuron y = mx + b": _single[:, 0]})
    _long = _df.melt("x", var_name="model", value_name="y")
    _chart = alt.Chart(_long).mark_line().encode(
        x=alt.X("x:Q", title="input (lean)"), y=alt.Y("y:Q", title="output"),
        color=alt.Color("model:N", scale=alt.Scale(range=["#3b82f6", "#999"])),
        strokeDash=alt.StrokeDash("model:N", scale=alt.Scale(range=[[1, 0], [6, 4]])),
        strokeWidth=alt.value(3),
    ).properties(width="container", height=220)
    mo.vstack([
        _chart,
        mo.hstack([
            mo.stat(f"{float(_W[0, 0]):.3f}", label="collapsed slope m"),
            mo.stat(f"{float(_b[0]):.3f}", label="collapsed bias b"),
            mo.stat(f"{np.abs(_h - _single).max():.1e}", label="max difference", caption="just floating-point noise"),
        ], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def relu_intro(mo):
    mo.md(r"""
    ### ReLU: add one bend

    The fix is an **activation function** that bends each neuron's output. The simplest effective one is **ReLU**, the rectified linear unit:
    $$\operatorname{ReLU}(z)=\max(0,z)\qquad\text{negative} \to 0,\quad \text{positive} \to \text{unchanged}.$$

    One bend is enough. Adding up many shifted, scaled ReLUs, $\;\hat u(\theta)=b+\sum_j w_j\,\operatorname{ReLU}(\theta-c_j)\;$, can trace **any** curve, piece by piece.

    That matters for the robot. A linear controller that sees **twice the lean doubles the motor**, always.
    A real robot may want something else: a gentle response near upright, a hard shove at large angles, saturation at the motor's limit.
    **Drag the blue dots** to design the response you'd like ("motor command vs. lean"). Then see who can follow it: a single neuron $y=mx+b$, or a ReLU network.
    """)
    return


@app.cell
def ResponseDesigner(anywidget, traitlets):
    class ResponseDesigner(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 640, H = 260, PAD = 36, XMIN = -30, XMAX = 30, YMIN = -1.2, YMAX = 1.2;
      const NS = "http://www.w3.org/2000/svg";
      const bar = document.createElement("div"); bar.style.cssText = "display:flex;gap:6px;margin-bottom:4px"; el.appendChild(bar);
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none" });
      el.appendChild(svg);
      const mk = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); svg.appendChild(e); return e; };
      const toSvg = (evt) => { const p = svg.createSVGPoint(); p.x = evt.clientX; p.y = evt.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const sx = v => PAD + (v - XMIN) / (XMAX - XMIN) * (W - 2 * PAD);
      const sy = v => H - PAD - (v - YMIN) / (YMAX - YMIN) * (H - 2 * PAD);
      const iy = py => YMIN + (H - PAD - py) / (H - 2 * PAD) * (YMAX - YMIN);
      const presets = {
        "Linear": x => -x / 30,
        "Deadzone + shove": x => Math.abs(x) < 6 ? -x / 60 : -Math.sign(x) * Math.min(1, 0.1 + (Math.abs(x) - 6) / 12),
        "Saturating": x => -Math.tanh(x / 8),
        "Asymmetric": x => x > 0 ? -Math.min(1, x / 10) : -x / 40,
      };
      for (const name in presets) {
        const b = document.createElement("button"); b.textContent = name; b.style.cssText = "padding:2px 9px;border-radius:6px;cursor:pointer";
        b.onclick = () => { const xs = model.get("xs"); model.set("ys", xs.map(x => Math.round(presets[name](x) * 100) / 100)); model.save_changes(); };
        bar.appendChild(b);
      }
      let drag = -1;
      function draw() {
        const xs = model.get("xs"), ys = model.get("ys");
        svg.innerHTML = "";
        mk("line", { x1: sx(XMIN), x2: sx(XMAX), y1: sy(0), y2: sy(0), stroke: "#999" });
        mk("line", { x1: sx(0), x2: sx(0), y1: sy(YMIN), y2: sy(YMAX), stroke: "#999" });
        const t1 = mk("text", { x: W - PAD, y: sy(0) - 6, "text-anchor": "end", fill: "#888", "font-size": 12 }); t1.textContent = "lean θ (°) →";
        const t2 = mk("text", { x: sx(0) + 6, y: PAD - 10, fill: "#888", "font-size": 12 }); t2.textContent = "motor command u";
        mk("polyline", { points: xs.map((x, i) => `${sx(x)},${sy(ys[i])}`).join(" "), fill: "none", stroke: "#999", "stroke-width": 2, "stroke-dasharray": "5 4" });
        xs.forEach((x, i) => {
          const c = mk("circle", { cx: sx(x), cy: sy(ys[i]), r: 9, fill: "#3b82f6", stroke: "white", "stroke-width": 2 });
          c.style.cursor = "ns-resize";
          c.addEventListener("pointerdown", e => { drag = i; svg.setPointerCapture(e.pointerId); });
        });
      }
      svg.addEventListener("pointermove", e => {
        if (drag < 0) return;
        const ys = [...model.get("ys")];
        ys[drag] = Math.max(-1.1, Math.min(1.1, Math.round(iy(toSvg(e).y) * 100) / 100));
        model.set("ys", ys); draw();
      });
      const stop = () => { if (drag >= 0) { drag = -1; model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointerleave", stop);
      model.on("change:ys", draw); draw();
    }
    export default { render };
    """
        xs = traitlets.List(traitlets.Float()).tag(sync=True)
        ys = traitlets.List(traitlets.Float()).tag(sync=True)

    return (ResponseDesigner,)


@app.cell
def designer(ResponseDesigner, mo, np):
    _xs = [float(v) for v in np.linspace(-30, 30, 13)]
    _ys = [(-np.sign(x) * min(1.0, 0.1 + (abs(x) - 6) / 12) if abs(x) >= 6 else -x / 60) for x in _xs]
    designer = mo.ui.anywidget(ResponseDesigner(xs=_xs, ys=[round(float(y), 2) for y in _ys]))
    n_relu = mo.ui.slider(1, 24, value=4, label="ReLU neurons in the hidden layer", show_value=True)
    mo.vstack([designer, n_relu])
    return designer, n_relu


@app.cell
def relu_fit(alt, designer, mo, n_relu, np, pd):
    def relu(z):
        return np.maximum(0.0, z)

    def fit_relu_net(x, y, n_units):
        """One hidden layer of ReLUs with hinges spread over the input range; output weights by least squares."""
        hinges = np.linspace(x.min(), x.max(), n_units + 2)[1:-1]
        feats = np.column_stack([np.ones_like(x), x] + [relu(x - c) for c in hinges])
        w, *_ = np.linalg.lstsq(feats, y, rcond=None)
        return (lambda q: np.column_stack([np.ones_like(q), q] + [relu(q - c) for c in hinges]) @ w), hinges

    _x = np.array(designer.value["xs"]); _y = np.array(designer.value["ys"])
    _xd = np.linspace(-30, 30, 241); _yd = np.interp(_xd, _x, _y)        # the curve you drew, densely sampled
    _m, _b = np.polyfit(_xd, _yd, 1)
    _net, _hinges = fit_relu_net(_xd, _yd, n_relu.value)
    _lin_mse = float(np.mean((_m * _xd + _b - _yd) ** 2))
    _net_mse = float(np.mean((_net(_xd) - _yd) ** 2))
    _df = pd.DataFrame({"θ": _xd, "your target": _yd, "single neuron (y = mx + b)": _m * _xd + _b, f"ReLU net ({n_relu.value} neurons)": _net(_xd)})
    _long = _df.melt("θ", var_name="model", value_name="u")
    _chart = alt.Chart(_long).mark_line(strokeWidth=2.5).encode(
        x=alt.X("θ:Q", title="lean θ (°)"), y=alt.Y("u:Q", title="motor command u"),
        color=alt.Color("model:N", scale=alt.Scale(domain=list(_df.columns[1:]), range=["#999", "#f18f01", "#3b82f6"])),
        strokeDash=alt.condition(alt.datum.model == "your target", alt.value([5, 4]), alt.value([1, 0])),
    )
    _rules = alt.Chart(pd.DataFrame({"hinge": _hinges})).mark_rule(color="#3b82f6", opacity=0.25).encode(x="hinge:Q")
    mo.vstack([
        (_rules + _chart).properties(width="container", height=260, title="Faint blue lines = where each ReLU bends"),
        mo.hstack([
            mo.stat(f"{_lin_mse:.4f}", label="single-neuron error (MSE)"),
            mo.stat(f"{_net_mse:.4f}", label="ReLU-net error (MSE)"),
            mo.stat(f"{_lin_mse / max(_net_mse, 1e-9):.0f}×", label="ReLU net is better by"),
        ], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def relu_try(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Click **Linear**. Both models are perfect: if the world is linear, you don't need the bend.
    2. Click **Deadzone + shove** and slide neurons from 1 to 24. Every extra ReLU adds one more corner, and the error collapses.
    3. Click **Asymmetric**: lean forward → push hard, lean back → push gently. The orange line can't do it. That's the video's point
       about how "a 10° lean while accelerating backwards doesn't need the same response as a 10° lean standing still": a **non-linear** response.
    4. Note the **PID** is a *sum of linear terms* in the error. Its shape is fixed; only the K's slide it around.
    """), kind="info")
    return


@app.cell(hide_code=True)
def choose_intro(mo):
    mo.md(r"""
    ## 4 · Choosing between AI and PID

    Both robots can balance. So why ever use the one with AI? The video gives three tests, and most people forget the last one.

    1. **Add a load.** The K's were tuned for the robot's physics. Change the physics (heavier load = sluggish motor response, $u/m$) and the old gains may no longer work.
    2. **Climb a hill.** A constant push $d$ from the slope. Now the I term earns its keep, but the best gains shift again.
    3. **Unknown worlds.** If you don't know the challenge in advance, you'd have to tune a PID for *every imaginable environment*. You'd never finish.

    Below, the **"your PID"** line uses the gains you set on the live robot in section 2. The **"re-tuned PID"** is what you'd get by brute-force
    searching gains *for this exact world*, which is the job a learning controller does automatically.
    """)
    return


@app.cell
def batch_fn(EFFORT, np):
    def batch_cost(Kp, Ki, Kd, mass=1.0, slope_deg=0.0, theta0=0.2, T=4.0, dt=0.01, umax=80.0, g_over_L=19.6):
        """Vectorized version of `simulate`: one robot per gain triple, returns cost per robot (same cost as section 2)."""
        Kp, Ki, Kd = (np.asarray(v, float) for v in np.broadcast_arrays(Kp, Ki, Kd))
        th = np.full(Kp.shape, theta0); om = np.zeros_like(th); integ = np.zeros_like(th); ep = -th.copy()
        dist = g_over_L * np.sin(np.radians(slope_deg))
        acc = np.zeros_like(th); alive = np.ones(th.shape, bool); fell_at = np.full(th.shape, T)
        for k in range(int(T / dt)):
            e = -th; integ += e * dt; d = (e - ep) / dt; ep = e
            u = np.clip(Kp * e + Ki * integ + Kd * d, -umax, umax)
            om += (g_over_L * np.sin(th) + u / mass * np.cos(th) + dist) * dt
            th = np.where(alive, th + om * dt, th)
            acc += np.where(alive, (th ** 2 + EFFORT * u ** 2) * dt, 0.0)
            newly = alive & (np.abs(th) > np.pi / 2)
            fell_at[newly] = k * dt; alive &= ~newly
        return np.where(alive, acc, 100.0 + (T - fell_at) * 10)

    KP_GRID = np.arange(10, 201, 10.0)
    KI_GRID = np.array([0.0, 10.0, 20.0, 40.0, 80.0])
    KD_GRID = np.arange(1, 31, 2.0)
    _P, _I, _D = np.meshgrid(KP_GRID, KI_GRID, KD_GRID, indexing="ij")
    GAIN_TABLE = np.column_stack([_P.ravel(), _I.ravel(), _D.ravel()])

    def best_gains(mass=1.0, slope_deg=0.0):
        _c = batch_cost(GAIN_TABLE[:, 0], GAIN_TABLE[:, 1], GAIN_TABLE[:, 2], mass=mass, slope_deg=slope_deg)
        return GAIN_TABLE[np.argmin(_c)], _c

    _g, _c = best_gains()
    f"{len(GAIN_TABLE)} gain combos searched; best on flat ground, no load: Kp={_g[0]:.0f}, Ki={_g[1]:.0f}, Kd={_g[2]:.0f}"
    return GAIN_TABLE, batch_cost, best_gains


@app.cell
def env_controls(mo):
    load = mo.ui.slider(0.5, 4.0, step=0.1, value=1.0, label="load m (×)", show_value=True)
    hill = mo.ui.slider(-15, 15, step=1, value=0, label="hill slope (°)", show_value=True)
    mo.hstack([load, hill], justify="start")
    return hill, load


@app.cell
def env_compare(
    GAIN_TABLE,
    alt,
    best_gains,
    cost,
    gains,
    hill,
    load,
    mo,
    np,
    pd,
    simulate,
):
    _best, _costs = best_gains(load.value, hill.value)
    _mine, _mine_fell = simulate(*gains, mass=load.value, slope_deg=hill.value)
    _ret, _ret_fell = simulate(*_best, mass=load.value, slope_deg=hill.value)
    _df = pd.concat([
        _mine.assign(controller="your PID (tuned for flat, m=1)"),
        _ret.assign(controller=f"re-tuned PID ({_best[0]:.0f}, {_best[1]:.0f}, {_best[2]:.0f})"),
    ])
    _lines = alt.Chart(_df).mark_line(strokeWidth=2.5).encode(
        x=alt.X("t:Q", title="time (s)"), y=alt.Y("theta:Q", title="lean θ (rad)", scale=alt.Scale(domain=[-1.6, 1.6], clamp=True)),
        color=alt.Color("controller:N", scale=alt.Scale(range=["#e4572e", "#3b82f6"]), legend=alt.Legend(orient="bottom")),
    ).properties(width=330, height=250, title="Same world, two sets of gains")
    # cost landscape over (Kp, Kd), at the re-tuned Ki
    _mask = GAIN_TABLE[:, 1] == _best[1]
    _heat = pd.DataFrame({"Kp": GAIN_TABLE[_mask, 0], "Kd": GAIN_TABLE[_mask, 2], "cost": np.log10(_costs[_mask])})
    _hm = alt.Chart(_heat).mark_rect().encode(
        x=alt.X("Kp:O", axis=alt.Axis(format=".0f")), y=alt.Y("Kd:O", sort="descending", axis=alt.Axis(format=".0f")),
        color=alt.Color("cost:Q", scale=alt.Scale(scheme="viridis", reverse=True), title="log₁₀ cost"),
        tooltip=["Kp", "Kd", "cost"])
    _star = alt.Chart(pd.DataFrame({"Kp": [_best[0]], "Kd": [_best[2]]})).mark_point(shape="cross", size=200, color="#3b82f6", filled=True).encode(x="Kp:O", y="Kd:O")
    _heatmap = (_hm + _star).properties(width=330, height=250, title=f"Cost landscape (Ki={_best[1]:.0f}); bright = good")
    _c_mine = cost(_mine, _mine_fell); _c_ret = cost(_ret, _ret_fell)
    mo.vstack([
        mo.hstack([_lines, _heatmap]),
        mo.hstack([
            mo.stat("FELL" if _mine_fell else f"{_c_mine:.4f}", label="your PID cost"),
            mo.stat("FELL" if _ret_fell else f"{_c_ret:.4f}", label="re-tuned PID cost"),
            mo.stat(f"{_mine.theta.iloc[-1]:+.3f} rad", label="your PID final lean", caption="≠ 0 → steady-state error"),
        ], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def env_try(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Raise the **load** to 2.5×. With the default gains the robot falls: $K_p/m$ dropped below gravity's 19.6. Watch the blue cross on the
       heatmap slide right, toward a bigger Kp and Kd (≈ 60→110 and 11→25). *Same algorithm, new physics, new gains.*
    2. Back at m = 1, set a **15° hill**. Go to section 2, set **Ki = 0**, and return: the red line now settles *tilted* (a steady-state error).
       Only the integral's memory removes it.
    3. Combine a heavy load with a steep hill. The good region on the heatmap shrinks. That's the "when the forces change, the gains need to change" story.
    """), kind="info")
    return


@app.cell(hide_code=True)
def worlds_intro(mo):
    mo.md(r"""
    ### Test 3: many unknown worlds

    Now we don't know the world in advance. We draw random worlds (load 0.5–4×, slope ±15°) and compare three strategies:

    * **your PID**: one fixed set of gains, deployed everywhere.
    * **best single PID**: the one gain triple (from our search grid) that does best *on average* over all worlds. This is careful robust tuning, done by brute force.
    * **adapt per world**: re-tune for each world. This is an upper bound on what an adaptive/learning controller could achieve.
    """)
    return


@app.cell
def worlds_controls(mo):
    n_worlds = mo.ui.slider(10, 80, step=10, value=40, label="number of random worlds", show_value=True)
    new_worlds = mo.ui.button(label="🎲 new worlds")
    mo.hstack([n_worlds, new_worlds], justify="start")
    return n_worlds, new_worlds


@app.cell
def worlds_result(
    GAIN_TABLE,
    alt,
    batch_cost,
    gains,
    mo,
    n_worlds,
    new_worlds,
    np,
    pd,
):
    new_worlds
    _rng = np.random.default_rng()
    _masses = _rng.uniform(0.5, 4.0, n_worlds.value)
    _slopes = _rng.uniform(-15, 15, n_worlds.value)
    _C = np.stack([batch_cost(GAIN_TABLE[:, 0], GAIN_TABLE[:, 1], GAIN_TABLE[:, 2], mass=_m, slope_deg=_s) for _m, _s in zip(_masses, _slopes)])
    _mine = np.array([batch_cost(*gains, mass=_m, slope_deg=_s) for _m, _s in zip(_masses, _slopes)]).ravel()
    _robust = _C[:, np.argmin(_C.mean(axis=0))]
    _adapt = _C.min(axis=1)
    _rg = GAIN_TABLE[np.argmin(_C.mean(axis=0))]
    _summary = pd.DataFrame({
        "strategy": ["your PID", f"best single PID ({_rg[0]:.0f},{_rg[1]:.0f},{_rg[2]:.0f})", "adapt per world"],
        "falls (%)": [100 * np.mean(c >= 100) for c in (_mine, _robust, _adapt)],
        "median cost": [np.median(c) for c in (_mine, _robust, _adapt)],
    })
    _bars = alt.Chart(_summary).mark_bar().encode(
        y=alt.Y("strategy:N", sort=None, title=None), x=alt.X("falls (%):Q", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("strategy:N", scale=alt.Scale(range=["#e4572e", "#f18f01", "#3b82f6"]), legend=None),
    ).properties(width="container", height=120, title="Share of worlds where the robot fell over")
    mo.vstack([_bars, mo.ui.table(_summary.round(4), selection=None)])
    return


@app.cell(hide_code=True)
def worlds_try(mo):
    mo.callout(mo.md(r"""
    **What to notice**

    * The fixed PID tuned for one world falls in many heavy-load worlds.
    * Robust tuning gets you a long way: one cleverly chosen PID often survives almost everything here. That is the *law of least power* in action.
    * But "adapt per world" always has lower cost, because the best gains really do differ between worlds. The more varied and unknown the worlds,
      the bigger that gap, and the stronger the case for a controller that **learns**.
    """), kind="success")
    return


@app.cell(hide_code=True)
def tradeoffs(mo):
    mo.md(r"""
    ### The verdict

    | | **PID** (3 lines) | **Neural network** |
    |---|---|---|
    | Shape of response | sum of linear terms in the error | any non-linear function (ReLU bends) |
    | Setup | tune 3 gains by hand | collect data, train (time!) |
    | Deploy & debug | trivial, interpretable | complex, harder to trust |
    | Physics change (load, hill) | must be **re-tuned** | can **adapt** |
    | Best for | simple, known, stable problems | varied, unknown, non-linear problems |

    Start with the dumb algorithm. Reach for AI when the world refuses to stay simple.
    """)
    return


@app.cell(hide_code=True)
def sandbox_intro(mo):
    mo.md(r"""
    ## 5 · Sandbox: break the robot

    Every knob in one place, including two the video hints at: **sensor noise** (the D term differentiates noise, so it amplifies jitter)
    and the **motor limit** (a real motor can't push infinitely hard, which is itself a non-linearity a PID doesn't know about).
    """)
    return


@app.cell
def sandbox_controls(mo):
    sandbox = mo.ui.dictionary({
        "Kp": mo.ui.slider(0, 200, step=1, value=60, label="Kp", show_value=True),
        "Ki": mo.ui.slider(0, 100, step=1, value=10, label="Ki", show_value=True),
        "Kd": mo.ui.slider(0, 30, step=0.5, value=11, label="Kd", show_value=True),
        "mass": mo.ui.slider(0.5, 4, step=0.1, value=1.0, label="load m", show_value=True),
        "slope": mo.ui.slider(-15, 15, step=1, value=0, label="hill (°)", show_value=True),
        "noise": mo.ui.slider(0, 0.03, step=0.001, value=0.0, label="sensor noise (rad)", show_value=True),
        "umax": mo.ui.slider(10, 200, step=5, value=80, label="motor limit", show_value=True),
        "theta0": mo.ui.slider(0.05, 1.2, step=0.05, value=0.2, label="initial lean (rad)", show_value=True),
    })
    mo.hstack([mo.vstack(list(sandbox.elements.values())[:4]), mo.vstack(list(sandbox.elements.values())[4:])])
    return (sandbox,)


@app.cell
def sandbox_plot(alt, cost, mo, np, sandbox, simulate):
    _v = sandbox.value
    _df, _fell = simulate(_v["Kp"], _v["Ki"], _v["Kd"], mass=_v["mass"], slope_deg=_v["slope"],
                          theta0=_v["theta0"], umax=_v["umax"], noise=_v["noise"], seed=1)
    _th = alt.Chart(_df).mark_line(color="#e4572e", strokeWidth=2).encode(
        x=alt.X("t:Q", title="time (s)", scale=alt.Scale(domain=[0, 6])), y=alt.Y("theta:Q", title="lean θ (rad)")).properties(width=640, height=160)
    _long = _df.melt("t", value_vars=["P", "I", "D", "u"], var_name="term")
    _u = alt.Chart(_long).mark_line(strokeWidth=1.5).encode(
        x=alt.X("t:Q", title="time (s)", scale=alt.Scale(domain=[0, 6])), y=alt.Y("value:Q", title="motor"),
        color=alt.Color("term:N", scale=alt.Scale(domain=["P", "I", "D", "u"], range=["#f18f01", "#6c5b7b", "#2a9d8f", "#444"])),
    ).properties(width=640, height=160)
    _sat = float(np.mean(np.abs(_df.u) >= _v["umax"] - 1e-9))
    mo.vstack([
        mo.hstack([
            mo.stat("FELL" if _fell else "upright", label="outcome"),
            mo.stat(f"{cost(_df, _fell):.4f}", label="cost"),
            mo.stat(f"{_df.theta.iloc[-1]:+.3f}", label="final lean (rad)"),
            mo.stat(f"{100 * _sat:.0f}%", label="time motor maxed out"),
        ], justify="start"),
        alt.vconcat(_th, _u),
    ])
    return


@app.cell(hide_code=True)
def sandbox_try(mo):
    mo.callout(mo.md(r"""
    **Challenges**

    1. **Noise vs. D:** set noise to 0.01 and Kd to 25. Look at the teal D trace: it turns into static. Find the smallest Kd that still balances.
    2. **Motor limit:** set the initial lean to 0.8 rad and the motor limit to 10. No gains can save it: gravity's pull ($19.6\sin 0.8 ≈ 14$) is stronger than the saturated motor. Raise the limit until some gains work.
    3. **Hill hero:** with m = 3 and a 12° hill, find gains with cost < 1.0 (hint: you'll need all three terms, and big ones).
    """), kind="warn")
    return


@app.cell(hide_code=True)
def quiz_intro(mo):
    mo.md(r"""
    ## Quiz: check yourself
    """)
    return


@app.cell
def quiz(mo):
    _Q = {
        "q1": ("Which PID term gives the robot *memory* of past error?", ["P", "I", "D"]),
        "q2": ("Which term mainly makes the robot *less jerky*?", ["P", "I", "D"]),
        "q3": ("A 5-layer network with **no** activation functions is equivalent to…", ["a 5× bigger network", "a single linear neuron", "a ReLU network"]),
        "q4": ("ReLU(−3) = ?", ["−3", "0", "3"]),
        "q5": ("You add a heavy load to a PID robot. What has to happen?", ["nothing, PID adapts", "re-tune the K gains", "add more ReLUs"]),
        "q6": ("On a constant hill, P + D alone leaves the robot settled…", ["upright", "slightly tilted", "falling"]),
        "q7": ("Google's law of least power says…", ["always use the biggest model", "don't be afraid to launch without ML", "never use ML"]),
    }
    quiz = mo.ui.dictionary({k: mo.ui.radio(options=opts, label=f"**{i + 1}.** {q}") for i, (k, (q, opts)) in enumerate(_Q.items())})
    reveal = mo.ui.switch(label="reveal answers")
    mo.vstack([quiz, reveal])
    return quiz, reveal


@app.cell
def quiz_score(mo, quiz, reveal):
    _answers = {"q1": "I", "q2": "D", "q3": "a single linear neuron", "q4": "0",
                "q5": "re-tune the K gains", "q6": "slightly tilted", "q7": "don't be afraid to launch without ML"}
    _why = {"q1": "the integral adds up the area under the error curve",
            "q2": "it reacts to how quickly the error is changing and brakes the swing",
            "q3": "products of weight matrices collapse into one W and one b",
            "q4": "negative inputs become zero",
            "q5": "the gains were tuned to the old physics (section 4: Kp needs to beat g/L·m)",
            "q6": "only the integral builds up enough push to cancel a constant force",
            "q7": "start simple; use AI when the problem demands it"}
    _score = sum(quiz.value[k] == v for k, v in _answers.items())
    _rows = [f"| {k[1:]} | {quiz.value[k] or '—'} | {'✅' if quiz.value[k] == v else '❌'} | {v}: {_why[k]} |" for k, v in _answers.items()]
    mo.vstack([
        mo.stat(f"{_score} / {len(_answers)}", label="score"),
        mo.md("| # | you | | answer |\n|---|---|---|---|\n" + "\n".join(_rows)) if reveal.value
        else mo.md("_Flip **reveal answers** when you're done._"),
    ])
    return


@app.cell(hide_code=True)
def recap(mo):
    mo.md(r"""
    ## Recap

    | Idea | One-liner | Where you touched it |
    |---|---|---|
    | Integral | add infinitely thin rectangles → area under the curve | §1 rectangle slider |
    | Derivative | squeeze a secant → slope at a point | §1 gap h slider |
    | **P** term | react to the error **now** | §2 live robot, Kp |
    | **I** term | react to **accumulated** error (memory); kills steady tilt on hills | §2 Ki, §4 hill |
    | **D** term | react to how **fast** the error changes; less jerky, but amplifies noise | §2 Kd, §5 noise |
    | Gains K | tuned by physics or brute force; depend on the system | §4 heatmap |
    | Linear collapse | stacked linear layers = one neuron $y=mx+b$ | §3 depth slider |
    | ReLU | $\max(0,z)$; sums of bends fit any curve → non-linear control | §3 response designer |
    | PID vs. AI | PID: simple, reliable, doesn't adapt. NN: adapts, costs training and complexity | §4 unknown worlds |

    **Rule of thumb:** launch the 3-line algorithm first, and bring in learning when the world gets too varied to hand-tune.
    """)
    return


if __name__ == "__main__":
    app.run()

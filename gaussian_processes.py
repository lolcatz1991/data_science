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
    # Gaussian Processes: distributions over *functions*
    *An interactive companion to DataMListic's video **"Gaussian Processes"**.*

    A Gaussian process (GP) doesn't hand you **one** function that fits your data. It hands you a whole **distribution of
    functions**, and that distribution tells you where you can trust the fit and where you can't. In this notebook you'll
    build a GP from scratch in numpy, click data points into existing, and watch the uncertainty collapse around them.
    """),
        mo.Html('<iframe width="560" height="315" src="https://www.youtube.com/embed/BTEbaPoZ08Y" title="Gaussian Processes" frameborder="0" allowfullscreen></iframe>'),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Color language (same everywhere in this notebook)

    | color | meaning |
    |---|---|
    | <span style="color:#d62728">**red**</span> | observed data points |
    | <span style="color:#2ca02c">**green**</span> | the GP mean $m(x)$ (prior or posterior) |
    | <span style="color:#1f77b4">**blue band**</span> | $\pm 2\sigma$ uncertainty (~95% region) |
    | thin multi-colored lines | individual **sample functions** drawn from the GP |

    **Roadmap** (follows the video): ① what a GP is → ② the prior → ③ the posterior (click to add data) →
    ④ kernels → ⑤ combining kernels → ⑥ a practical example → sandbox, quiz, recap.
    """)
    return


@app.cell
def gp_helpers(alt, np, pd):
    def rbf(x1, x2, sigma=1.0, ell=1.0):
        """Squared exponential / RBF: sigma^2 exp(-|x-x'|^2 / (2 ell^2))."""
        d = np.subtract.outer(np.asarray(x1, float), np.asarray(x2, float))
        return sigma**2 * np.exp(-(d**2) / (2 * ell**2))


    def periodic(x1, x2, sigma=1.0, ell=1.0, p=2.0):
        """Periodic: sigma^2 exp(-2 sin^2(pi |x-x'| / p) / ell^2)."""
        d = np.abs(np.subtract.outer(np.asarray(x1, float), np.asarray(x2, float)))
        return sigma**2 * np.exp(-2 * np.sin(np.pi * d / p) ** 2 / ell**2)


    def linear(x1, x2, sigma=1.0, sigma_b=0.5, c=0.0):
        """Linear: sigma_b^2 + sigma^2 (x-c)(x'-c)."""
        return sigma_b**2 + sigma**2 * np.multiply.outer(np.asarray(x1, float) - c, np.asarray(x2, float) - c)


    def sample_mvn(mean, cov, n, rng, jitter=1e-8):
        """Draw n samples from N(mean, cov) via Cholesky: f = mean + L z."""
        _L = np.linalg.cholesky(cov + jitter * np.max(np.diag(cov)) * np.eye(len(cov)) + 1e-12 * np.eye(len(cov)))
        return mean[None, :] + rng.standard_normal((n, len(cov))) @ _L.T


    def gp_posterior(kernel, X, y, Xs, noise=1e-3):
        """Exact GP regression. Returns posterior mean, covariance at Xs and the log marginal likelihood."""
        X, y, Xs = np.asarray(X, float), np.asarray(y, float), np.asarray(Xs, float)
        Kss = kernel(Xs, Xs)
        if len(X) == 0:
            return np.zeros(len(Xs)), Kss, 0.0
        K = kernel(X, X) + (noise**2 + 1e-8) * np.eye(len(X))
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        Ks = kernel(X, Xs)
        mu = Ks.T @ alpha                              # m* = K*^T K^-1 y
        V = np.linalg.solve(L, Ks)
        cov = Kss - V.T @ V                            # S* = K** - K*^T K^-1 K*
        lml = -0.5 * y @ alpha - np.log(np.diag(L)).sum() - 0.5 * len(X) * np.log(2 * np.pi)
        return mu, cov, lml


    def band_df(xs, mu, cov):
        _sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        return pd.DataFrame({"x": xs, "mean": mu, "lo": mu - 2 * _sd, "hi": mu + 2 * _sd})


    def samples_df(xs, F):
        return pd.DataFrame({"x": np.tile(xs, len(F)), "f": F.ravel(), "sample": np.repeat(np.arange(len(F)), len(xs)).astype(str)})


    def gp_chart(xs, mu, cov, F=None, X=None, y=None, title="", width="container", height=260, ydom=None):
        """Blue band + green mean + sample lines + red data."""
        _b = band_df(xs, mu, cov)
        _ys = alt.Scale(domain=ydom) if ydom else alt.Scale(zero=False)
        _layers = [
            alt.Chart(_b).mark_area(opacity=0.18, color="#1f77b4").encode(x="x:Q", y=alt.Y("lo:Q", title="f(x)", scale=_ys), y2="hi:Q"),
        ]
        if F is not None and len(F):
            _layers.append(alt.Chart(samples_df(xs, F)).mark_line(strokeWidth=1.2, opacity=0.75, clip=True).encode(
                x="x:Q", y=alt.Y("f:Q", scale=_ys), color=alt.Color("sample:N", legend=None, scale=alt.Scale(scheme="tableau10"))))
        _layers.append(alt.Chart(_b).mark_line(color="#2ca02c", strokeWidth=3).encode(x="x:Q", y=alt.Y("mean:Q", scale=_ys)))
        if X is not None and len(X):
            _layers.append(alt.Chart(pd.DataFrame({"x": X, "y": y})).mark_circle(color="#d62728", size=70, opacity=1).encode(x="x:Q", y="y:Q"))
        return alt.layer(*_layers).properties(width=width, height=height, title=title)

    return (
        gp_chart,
        gp_posterior,
        linear,
        periodic,
        rbf,
        sample_mvn,
        samples_df,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · A GP is a distribution over functions

    > *"A Gaussian process is a collection of random variables where any finite subset follows a multivariate Gaussian distribution."*

    We write

    $$f(x) \sim \mathcal{GP}\big(m(x),\; k(x, x')\big)$$

    * $m(x)$ is the **mean function**: the expected value of $f$ at $x$ (usually $0$).
    * $k(x,x')$ is the **covariance function** or **kernel**: how strongly $f(x)$ and $f(x')$ move together.

    The video's trick: pick just **five points** $x_1,\dots,x_5$. A GP says their function values are jointly Gaussian,

    $$\begin{bmatrix} f(x_1)\\ \vdots \\ f(x_5)\end{bmatrix} \sim \mathcal N\!\left(\mathbf 0,\; K\right),\qquad K_{ij} = k(x_i, x_j).$$

    Sample that vector, connect the dots, and you have a (rough) function. Add more points and the dots become a smooth
    curve. It's like **rolling a bunch of correlated dice together**: neighbours tend to land on similar numbers.
    """)
    return


@app.cell
def s1_controls(mo):
    s1_n = mo.ui.slider(3, 60, value=5, step=1, label="number of points $n$", show_value=True)
    s1_ell = mo.ui.slider(0.2, 3.0, value=1.0, step=0.1, label="length-scale $\\ell$", show_value=True)
    s1_roll = mo.ui.button(label="🎲 roll the correlated dice", value=0, on_click=lambda v: v + 1)
    mo.hstack([s1_n, s1_ell, s1_roll], justify="start", gap=2)
    return s1_ell, s1_n, s1_roll


@app.cell
def s1_chart(alt, np, pd, rbf, s1_ell, s1_n, s1_roll, sample_mvn, samples_df):
    _x = np.linspace(-5, 5, s1_n.value)
    s1_K = rbf(_x, _x, 1.0, s1_ell.value)
    _rng = np.random.default_rng(100 + s1_roll.value)
    s1_F = sample_mvn(np.zeros(len(_x)), s1_K, 3, _rng)

    _i, _j = np.meshgrid(np.arange(len(_x)), np.arange(len(_x)), indexing="ij")
    _heat = alt.Chart(pd.DataFrame({"i": _i.ravel() + 1, "j": _j.ravel() + 1, "k": s1_K.ravel()})).mark_rect().encode(
        x=alt.X("j:O", title="j", axis=alt.Axis(labels=len(_x) <= 15)),
        y=alt.Y("i:O", title="i", axis=alt.Axis(labels=len(_x) <= 15)),
        color=alt.Color("k:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]), title="K_ij"),
        tooltip=["i", "j", alt.Tooltip("k:Q", format=".3f")],
    ).properties(width=260, height=260, title="Covariance matrix K = k(x_i, x_j)")

    _df = samples_df(_x, s1_F)
    _lines = alt.Chart(_df).mark_line(strokeWidth=1.5).encode(x=alt.X("x:Q", scale=alt.Scale(domain=[-5, 5])), y=alt.Y("f:Q", title="f(x)", scale=alt.Scale(domain=[-3.2, 3.2])), color=alt.Color("sample:N", legend=None, scale=alt.Scale(scheme="tableau10")))
    _dots = _lines.mark_circle(size=60 if len(_x) <= 15 else 15, opacity=1)
    _rule = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#2ca02c", strokeWidth=2).encode(y="y:Q")
    alt.hconcat(_heat, (_rule + _lines + _dots).properties(width=380, height=260, title=f"3 samples of (f(x_1)...f(x_{len(_x)}))")).resolve_scale(color="independent")
    return s1_F, s1_K


@app.cell
def _(mo, np, s1_F, s1_K, s1_n):
    _x = np.linspace(-5, 5, s1_n.value)
    _dx = _x[1] - _x[0]
    _jump = np.abs(np.diff(s1_F, axis=1)).mean()
    mo.hstack([
        mo.stat(f"{_dx:.2f}", label="spacing between neighbours"),
        mo.stat(f"{s1_K[0, 1]:.3f}", label="corr(f(x₁), f(x₂))", caption="k(x₁,x₂) / σ²"),
        mo.stat(f"{s1_K[0, -1]:.1e}", label="corr(first, last)", caption="far apart → ~independent"),
        mo.stat(f"{_jump:.2f}", label="avg |jump| between neighbours"),
    ], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. With $n=5$, roll the dice a few times: the five values look nearly independent (pale off-diagonal). Now slide $n$ up to 60: the *same* distribution, seen at more places, becomes a smooth curve.
    2. Drag $\ell$ down to 0.2: the bright band in $K$ gets thin and the curves get wiggly. At $\ell=3$ the whole matrix glows and samples barely bend.
    3. Notice that average |jump| shrinks as $n$ grows: neighbours become more correlated, which is exactly what makes the function *continuous*.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · The prior: functions *before* you see data

    Before observing anything, we choose a **prior**. For simplicity the mean function is set to zero, $m(x)=0$
    (the <span style="color:#2ca02c">**green**</span> line; red in the video). The kernel gives every point a variance
    $k(x,x)=\sigma^2$, so the <span style="color:#1f77b4">**blue band**</span> $0 \pm 2\sigma$ is the 95% region where the
    functions are expected to live.

    Every sample below is a full function drawn from $\mathcal{GP}(0, k_{\text{RBF}})$. The band is the same width
    everywhere: the prior has no reason to be more confident anywhere.
    """)
    return


@app.cell
def s2_controls(mo):
    s2_sigma = mo.ui.slider(0.2, 2.0, value=1.0, step=0.1, label="signal scale $\\sigma$", show_value=True)
    s2_ell = mo.ui.slider(0.2, 3.0, value=1.0, step=0.1, label="length-scale $\\ell$", show_value=True)
    s2_n = mo.ui.slider(1, 10, value=5, step=1, label="# samples", show_value=True)
    s2_roll = mo.ui.button(label="🎲 new samples", value=0, on_click=lambda v: v + 1)
    mo.hstack([s2_sigma, s2_ell, s2_n, s2_roll], justify="start", gap=2)
    return s2_ell, s2_n, s2_roll, s2_sigma


@app.cell
def s2_chart(
    gp_chart,
    gp_posterior,
    np,
    rbf,
    s2_ell,
    s2_n,
    s2_roll,
    s2_sigma,
    sample_mvn,
):
    xs_grid = np.linspace(-5, 5, 150)
    _k = lambda a, b: rbf(a, b, s2_sigma.value, s2_ell.value)
    _mu, _cov, _ = gp_posterior(_k, [], [], xs_grid)
    s2_F = sample_mvn(_mu, _cov, s2_n.value, np.random.default_rng(7 + s2_roll.value))
    gp_chart(xs_grid, _mu, _cov, s2_F, title="Prior GP(0, k_RBF): mean, ±2σ band and sample functions", ydom=[-5, 5])
    return (xs_grid,)


@app.cell
def _(mo, np, rbf, s2_ell, s2_sigma, sample_mvn, xs_grid):
    _k = lambda a, b: rbf(a, b, s2_sigma.value, s2_ell.value)
    _F = sample_mvn(np.zeros(len(xs_grid)), _k(xs_grid, xs_grid), 2000, np.random.default_rng(1))
    _inside = np.mean(np.abs(_F) <= 2 * s2_sigma.value)
    mo.hstack([
        mo.stat(f"{s2_sigma.value:.2f}", label="prior σ (from the kernel)"),
        mo.stat(f"{_F.std(axis=0).mean():.2f}", label="empirical σ over 2000 samples"),
        mo.stat(f"{_inside:.1%}", label="share of sample values inside ±2σ", caption="theory: 95.4%"),
    ], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Raise $\sigma$: the band and the samples stretch vertically, but their *wiggliness* doesn't change. $\sigma$ is "how far", $\ell$ is "how fast".
    2. Shrink $\ell$ to 0.3 and compare: same band, very different functions. The band alone can't tell you the kernel; the samples can.
    3. Check the readout: ~95% of all sampled values land inside the blue band, whatever you choose.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · The posterior: click data into existence

    When we observe data, the distribution **updates**. Conditioning a Gaussian on some of its coordinates gives another
    Gaussian, so the posterior at test points $X_*$ given data $(X, \mathbf y)$ with noise $\sigma_n$ is

    $$
    \begin{aligned}
    \mathbf m_* &= K_*^\top\,(K + \sigma_n^2 I)^{-1}\,\mathbf y \\
    \Sigma_* &= K_{**} - K_*^\top\,(K + \sigma_n^2 I)^{-1}\,K_*
    \end{aligned}
    $$

    with $K = k(X,X)$, $K_* = k(X,X_*)$, $K_{**} = k(X_*,X_*)$. The first line pulls the
    <span style="color:#2ca02c">**green mean**</span> through the <span style="color:#d62728">**red points**</span>;
    the second line *subtracts* uncertainty wherever the kernel says "this test point is correlated with an observation".

    **Click anywhere in the plot to add an observation.** Drag points to move them; double-click one to delete it.
    The widget solves the GP (Cholesky, in JavaScript) on every frame, so you see the update instantly.
    """)
    return


@app.cell
def gp_canvas_class(anywidget, linear, periodic, rbf, traitlets):
    class GPCanvas(anywidget.AnyWidget):
        """Click to add data, drag to move, double-click a point to delete. The full GP posterior is computed live in JS."""
        _esm = r"""
    const NS = "http://www.w3.org/2000/svg";
    const KERNELS = ["rbf", "periodic", "linear", "periodic+linear", "rbf+linear"];
    function kfun(name, s, l, p) {
      const s2 = s * s;
      const rbf = (a, b) => s2 * Math.exp(-((a - b) ** 2) / (2 * l * l));
      const per = (a, b) => s2 * Math.exp(-2 * Math.sin(Math.PI * Math.abs(a - b) / p) ** 2 / (l * l));
      const lin = (a, b) => 0.25 + (s2 / 9) * a * b;   // sigma_b = 0.5, slope sd = sigma/3, c = 0
      return { rbf, periodic: per, linear: lin,
               "periodic+linear": (a, b) => per(a, b) + lin(a, b),
               "rbf+linear": (a, b) => rbf(a, b) + lin(a, b) }[name];
    }
    function chol(A, n) {
      const L = new Float64Array(n * n);
      for (let i = 0; i < n; i++) for (let j = 0; j <= i; j++) {
        let s = A[i * n + j];
        for (let k = 0; k < j; k++) s -= L[i * n + k] * L[j * n + k];
        L[i * n + j] = i === j ? Math.sqrt(Math.max(s, 1e-10)) : s / L[j * n + j];
      }
      return L;
    }
    function fwd(L, b, n) { const x = new Float64Array(n); for (let i = 0; i < n; i++) { let s = b[i]; for (let k = 0; k < i; k++) s -= L[i * n + k] * x[k]; x[i] = s / L[i * n + i]; } return x; }
    function bwd(L, b, n) { const x = new Float64Array(n); for (let i = n - 1; i >= 0; i--) { let s = b[i]; for (let k = i + 1; k < n; k++) s -= L[k * n + i] * x[k]; x[i] = s / L[i * n + i]; } return x; }
    function rng(seed) { let a = seed >>> 0; const u = () => { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; };
      return () => Math.sqrt(-2 * Math.log(u() + 1e-12)) * Math.cos(2 * Math.PI * u()); }

    function render({ model, el }) {
      const W = 680, H = 340, PAD = 34, XMIN = -5, XMAX = 5, YMIN = -3.5, YMAX = 3.5, M = 100;
      const COLORS = ["#4e79a7", "#f28e2b", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"];
      const sx = v => PAD + (v - XMIN) / (XMAX - XMIN) * (W - 2 * PAD);
      const sy = v => H - PAD - (v - YMIN) / (YMAX - YMIN) * (H - 2 * PAD);
      const ix = px => XMIN + (px - PAD) / (W - 2 * PAD) * (XMAX - XMIN);
      const iy = py => YMIN + (H - PAD - py) / (H - 2 * PAD) * (YMAX - YMIN);
      const grid = Array.from({ length: M }, (_, i) => XMIN + i * (XMAX - XMIN) / (M - 1));

      // --- toolbar
      const bar = document.createElement("div");
      Object.assign(bar.style, { display: "flex", flexWrap: "wrap", gap: "10px", alignItems: "center", fontSize: "13px", marginBottom: "4px" });
      el.appendChild(bar);
      const sel = document.createElement("select");
      KERNELS.forEach(k => { const o = document.createElement("option"); o.value = k; o.textContent = k; sel.appendChild(o); });
      sel.onchange = () => { model.set("kernel", sel.value); model.save_changes(); draw(); };
      const lab = t => { const s = document.createElement("span"); s.textContent = t; return s; };
      bar.append(lab("kernel"), sel);
      const sliders = {};
      for (const [key, label, lo, hi, st] of [["ell", "ℓ", 0.1, 3, 0.05], ["sigma", "σ", 0.2, 2.5, 0.05], ["period", "p", 0.5, 6, 0.1], ["noise", "noise σₙ", 0.0, 1, 0.01], ["n_samples", "samples", 0, 8, 1]]) {
        const wrap = document.createElement("label"); wrap.style.whiteSpace = "nowrap";
        const r = document.createElement("input"); Object.assign(r, { type: "range", min: lo, max: hi, step: st }); r.style.width = "80px"; r.style.verticalAlign = "middle";
        const v = document.createElement("span"); v.style.display = "inline-block"; v.style.width = "32px";
        r.oninput = () => { model.set(key, +r.value); v.textContent = r.value; draw(); };
        r.onchange = () => model.save_changes();
        wrap.append(label + " ", r, v); bar.appendChild(wrap); sliders[key] = [r, v];
      }
      const btn = (t, f) => { const b = document.createElement("button"); b.textContent = t; b.onclick = f; bar.appendChild(b); };
      btn("🎲 resample", () => { model.set("seed", model.get("seed") + 1); model.save_changes(); draw(); });
      btn("↶ undo", () => { const a = [...model.get("points")]; a.pop(); model.set("points", a); model.save_changes(); });
      btn("✕ clear", () => { model.set("points", []); model.save_changes(); });

      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none", cursor: "crosshair", border: "1px solid rgba(128,128,128,.3)", borderRadius: "6px" });
      el.appendChild(svg);
      const info = document.createElement("div"); info.style.fontSize = "12px"; info.style.opacity = 0.8; el.appendChild(info);
      const mk = (tag, attrs, parent = svg) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
      const toSvg = e => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const clampY = v => Math.max(YMIN + 0.1, Math.min(YMAX - 0.1, v));

      function posterior() {
        const pts = model.get("points"), n = pts.length, k = kfun(model.get("kernel"), model.get("sigma"), model.get("ell"), model.get("period"));
        const nz = model.get("noise") ** 2 + 1e-6;
        const X = pts.map(p => p[0]), y = pts.map(p => p[1]);
        const Kgg = new Float64Array(M * M); for (let i = 0; i < M; i++) for (let j = 0; j < M; j++) Kgg[i * M + j] = k(grid[i], grid[j]);
        let mean = new Float64Array(M), cov = Kgg;
        if (n) {
          const K = new Float64Array(n * n); for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) K[i * n + j] = k(X[i], X[j]) + (i === j ? nz : 0);
          const L = chol(K, n), alpha = bwd(L, fwd(L, y, n), n);
          const V = [];   // V[g] = L^-1 k(X, g)
          for (let g = 0; g < M; g++) { const ks = X.map(xi => k(xi, grid[g])); mean[g] = ks.reduce((s, v, i) => s + v * alpha[i], 0); V.push(fwd(L, ks, n)); }
          cov = new Float64Array(M * M);
          for (let i = 0; i < M; i++) for (let j = 0; j <= i; j++) { let s = Kgg[i * M + j]; for (let q = 0; q < n; q++) s -= V[i][q] * V[j][q]; cov[i * M + j] = cov[j * M + i] = s; }
        }
        return { mean, cov };
      }
      const path = ys => grid.map((g, i) => `${i ? "L" : "M"}${sx(g).toFixed(1)},${sy(Math.max(YMIN - 2, Math.min(YMAX + 2, ys[i]))).toFixed(1)}`).join("");

      let drag = -1, moved = false;
      function draw() {
        sel.value = model.get("kernel");
        for (const key in sliders) { sliders[key][0].value = model.get(key); sliders[key][1].textContent = (+model.get(key)).toFixed(key === "n_samples" ? 0 : 2); }
        sliders.period[0].disabled = !model.get("kernel").includes("periodic");
        const { mean, cov } = posterior();
        const sd = grid.map((_, i) => Math.sqrt(Math.max(cov[i * M + i], 0)));
        svg.innerHTML = "";
        const clip = mk("clipPath", { id: "gpclip" }); mk("rect", { x: PAD, y: PAD, width: W - 2 * PAD, height: H - 2 * PAD }, clip);
        mk("rect", { x: PAD, y: PAD, width: W - 2 * PAD, height: H - 2 * PAD, fill: "rgba(128,128,128,0.04)" });
        for (let t = -3; t <= 3; t++) { mk("line", { x1: PAD, x2: W - PAD, y1: sy(t), y2: sy(t), stroke: "rgba(128,128,128,.18)" }); mk("text", { x: PAD - 6, y: sy(t) + 4, "text-anchor": "end", "font-size": 10, fill: "currentColor" }).textContent = t; }
        for (let t = -5; t <= 5; t++) mk("text", { x: sx(t), y: H - PAD + 14, "text-anchor": "middle", "font-size": 10, fill: "currentColor" }).textContent = t;
        const g = mk("g", { "clip-path": "url(#gpclip)" });
        const up = grid.map((x, i) => `${sx(x).toFixed(1)},${sy(mean[i] + 2 * sd[i]).toFixed(1)}`), dn = grid.map((x, i) => `${sx(x).toFixed(1)},${sy(mean[i] - 2 * sd[i]).toFixed(1)}`).reverse();
        mk("polygon", { points: up.concat(dn).join(" "), fill: "#1f77b4", "fill-opacity": 0.18, stroke: "none" }, g);
        const ns = model.get("n_samples");
        if (ns > 0) {
          const Lc = chol(cov.map((v, i) => v + (i % (M + 1) === 0 ? 1e-6 : 0)), M), z = rng(model.get("seed") * 7919 + 1);
          for (let s = 0; s < ns; s++) {
            const e = Array.from({ length: M }, z), f = mean.map((m, i) => { let acc = m; for (let q = 0; q <= i; q++) acc += Lc[i * M + q] * e[q]; return acc; });
            mk("path", { d: path(f), fill: "none", stroke: COLORS[s % COLORS.length], "stroke-width": 1.3, opacity: 0.85 }, g);
          }
        }
        mk("path", { d: path(mean), fill: "none", stroke: "#2ca02c", "stroke-width": 3 }, g);
        model.get("points").forEach((p, i) => {
          const c = mk("circle", { cx: sx(p[0]), cy: sy(p[1]), r: 7, fill: "#d62728", stroke: "white", "stroke-width": 2 });
          c.style.cursor = "grab";
          c.addEventListener("pointerdown", e => { e.stopPropagation(); drag = i; moved = false; svg.setPointerCapture(e.pointerId); });
          c.addEventListener("dblclick", e => { e.stopPropagation(); const a = [...model.get("points")]; a.splice(i, 1); model.set("points", a); model.save_changes(); });
        });
        const avg = sd.reduce((a, b) => a + b, 0) / M;
        info.textContent = `${model.get("points").length} points · mean posterior σ over the plot = ${avg.toFixed(3)} · click empty space to add, drag to move, double-click a point to delete`;
      }
      svg.addEventListener("pointerdown", e => {
        if (drag >= 0) return;
        const p = toSvg(e); if (p.x < PAD || p.x > W - PAD) return;
        model.set("points", [...model.get("points"), [Math.round(ix(p.x) * 100) / 100, Math.round(clampY(iy(p.y)) * 100) / 100]]);
        model.save_changes();
      });
      svg.addEventListener("pointermove", e => {
        if (drag < 0) return; moved = true;
        const p = toSvg(e), a = model.get("points").map(q => [...q]);
        a[drag] = [Math.round(Math.max(XMIN, Math.min(XMAX, ix(p.x))) * 100) / 100, Math.round(clampY(iy(p.y)) * 100) / 100];
        model.set("points", a); draw();          // local only while dragging
      });
      const stop = () => { if (drag >= 0) { drag = -1; if (moved) model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointerleave", stop);
      for (const t of ["points", "kernel", "ell", "sigma", "period", "noise", "n_samples", "seed"]) model.on("change:" + t, draw);
      draw();
    }
    export default { render };
    """
        points = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)
        kernel = traitlets.Unicode("rbf").tag(sync=True)
        ell = traitlets.Float(1.0).tag(sync=True)
        sigma = traitlets.Float(1.0).tag(sync=True)
        period = traitlets.Float(2.0).tag(sync=True)
        noise = traitlets.Float(0.05).tag(sync=True)
        n_samples = traitlets.Int(3).tag(sync=True)
        seed = traitlets.Int(0).tag(sync=True)


    def canvas_kernel(v):
        """The same kernels the GPCanvas widget uses in JS, built from its .value dict."""
        _s, _l, _p = v["sigma"], v["ell"], v["period"]
        _lin = lambda a, b: linear(a, b, _s / 3, 0.5, 0.0)
        _parts = {
            "rbf": lambda a, b: rbf(a, b, _s, _l),
            "periodic": lambda a, b: periodic(a, b, _s, _l, _p),
            "linear": _lin,
        }
        _parts["periodic+linear"] = lambda a, b: _parts["periodic"](a, b) + _lin(a, b)
        _parts["rbf+linear"] = lambda a, b: _parts["rbf"](a, b) + _lin(a, b)
        return _parts[v["kernel"]]

    return GPCanvas, canvas_kernel


@app.cell
def gp_canvas(GPCanvas, mo):
    gp_canvas = mo.ui.anywidget(GPCanvas(points=[[-3.0, -1.0], [-1.0, 1.2], [2.5, 0.6]]))
    gp_canvas
    return (gp_canvas,)


@app.cell
def s3_readout(
    canvas_kernel,
    gp_canvas,
    gp_chart,
    gp_posterior,
    mo,
    np,
    sample_mvn,
    xs_grid,
):
    _v = gp_canvas.value
    _pts = np.array(_v["points"], dtype=float).reshape(-1, 2)
    _X, _y = _pts[:, 0], _pts[:, 1]
    _mu, _cov, _lml = gp_posterior(canvas_kernel(_v), _X, _y, xs_grid, noise=_v["noise"])
    _sd = np.sqrt(np.clip(np.diag(_cov), 0, None))
    _prior_sd = np.sqrt(np.diag(canvas_kernel(_v)(xs_grid, xs_grid)))
    _F = sample_mvn(_mu, _cov, 5, np.random.default_rng(_v["seed"]))
    if len(_X):
        _near = np.abs(xs_grid[:, None] - _X[None, :]).min(axis=1)
        _sd_near, _sd_far = _sd[_near < 0.15].mean(), _sd[_near == _near.max()].mean()
    else:
        _sd_near = _sd_far = _sd.mean()
    mo.vstack([
        mo.hstack([
            mo.stat(len(_X), label="observations"),
            mo.stat(f"{_sd_near:.3f}", label="posterior σ next to data"),
            mo.stat(f"{_sd_far:.3f}", label="posterior σ farthest from data", caption=f"prior σ ≈ {_prior_sd.mean():.2f}"),
            mo.stat(f"{(1 - (_sd**2).mean() / (_prior_sd**2).mean()):.0%}", label="variance explained away"),
            mo.stat(f"{_lml:.2f}", label="log marginal likelihood", caption="log p(y | X, kernel)"),
        ], justify="start"),
        gp_chart(xs_grid, _mu, _cov, _F, _X, _y, title="Same posterior, recomputed in numpy (gp_posterior) with 5 fresh posterior samples", height=220, ydom=[-4, 4]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Press **✕ clear**: you're back to the prior (flat green line, uniform band). Add **one** point: the mean jumps through it and the band pinches to ~σₙ there, but only within about $\ell$ of it.
    2. Add points one by one from left to right, like the video: "uncertainty is smallest near our data points and grows larger as we move away". Watch *variance explained away* climb.
    3. Set noise σₙ to 0, then put two points at almost the same $x$ but very different $y$. The mean goes crazy (the model must pass through both!). Raise the noise: the GP now averages them. That's why we model noise.
    4. Shrink $\ell$ to 0.2: between data points the GP falls back to the prior almost immediately. Grow it to 3: one point informs the whole axis.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Kernels decide what functions you can represent

    > *"Basically the kernel determines what kinds of functions we can represent."*

    The video walks through three kernels:

    | kernel | formula | samples look like |
    |---|---|---|
    | **RBF** (squared exponential) | $k(x,x') = \sigma^2 \exp\!\left(-\dfrac{\lvert x-x'\rvert^2}{2\ell^2}\right)$ | smooth, gradually varying |
    | **Periodic** | $k(x,x') = \sigma^2 \exp\!\left(-\dfrac{2\sin^2(\pi \lvert x-x'\rvert / p)}{\ell^2}\right)$ | repeating with period $p$ |
    | **Linear** | $k(x,x') = \sigma_b^2 + \sigma^2 (x-c)(x'-c)$ | straight lines, random slope & intercept |

    Left chart: one **row** of the kernel, $k(0, x')$: how much $f(0)$ is correlated with $f(x')$. Middle: the full
    covariance matrix. Right: prior samples. Read them together: the shape of the kernel row *is* the personality of the functions.
    """)
    return


@app.cell
def s4_controls(mo):
    s4_kernel = mo.ui.dropdown(["RBF", "Periodic", "Linear"], value="RBF", label="kernel")
    s4_sigma = mo.ui.slider(0.2, 2.0, value=1.0, step=0.1, label="$\\sigma$", show_value=True)
    s4_ell = mo.ui.slider(0.2, 3.0, value=1.0, step=0.1, label="$\\ell$ (RBF, periodic)", show_value=True)
    s4_p = mo.ui.slider(0.5, 6.0, value=2.5, step=0.1, label="period $p$", show_value=True)
    s4_sb = mo.ui.slider(0.1, 2.0, value=0.5, step=0.1, label="$\\sigma_b$ (linear)", show_value=True)
    s4_c = mo.ui.slider(-4.0, 4.0, value=0.0, step=0.5, label="$c$ (linear)", show_value=True)
    s4_roll = mo.ui.button(label="🎲 new samples", value=0, on_click=lambda v: v + 1)
    mo.vstack([mo.hstack([s4_kernel, s4_sigma, s4_ell, s4_p], justify="start", gap=2), mo.hstack([s4_sb, s4_c, s4_roll], justify="start", gap=2)])
    return s4_c, s4_ell, s4_kernel, s4_p, s4_roll, s4_sb, s4_sigma


@app.cell
def s4_chart(
    alt,
    gp_chart,
    gp_posterior,
    linear,
    np,
    pd,
    periodic,
    rbf,
    s4_c,
    s4_ell,
    s4_kernel,
    s4_p,
    s4_roll,
    s4_sb,
    s4_sigma,
    sample_mvn,
    xs_grid,
):
    s4_k = {
        "RBF": lambda a, b: rbf(a, b, s4_sigma.value, s4_ell.value),
        "Periodic": lambda a, b: periodic(a, b, s4_sigma.value, s4_ell.value, s4_p.value),
        "Linear": lambda a, b: linear(a, b, s4_sigma.value / 3, s4_sb.value, s4_c.value),
    }[s4_kernel.value]
    _xs = np.linspace(-5, 5, 60)
    _row = alt.Chart(pd.DataFrame({"x'": xs_grid, "k": s4_k(np.array([0.0]), xs_grid)[0]})).mark_area(
        line={"color": "#1f77b4"}, color="#1f77b4", opacity=0.3).encode(x="x':Q", y=alt.Y("k:Q", title="k(0, x')")).properties(width=200, height=220, title="Kernel row k(0, x')")
    _K = s4_k(_xs, _xs)
    _i, _j = np.meshgrid(np.arange(60), np.arange(60), indexing="ij")
    _heat = alt.Chart(pd.DataFrame({"i": _i.ravel(), "j": _j.ravel(), "k": _K.ravel()})).mark_rect().encode(
        x=alt.X("j:O", axis=None), y=alt.Y("i:O", axis=None), color=alt.Color("k:Q", scale=alt.Scale(scheme="blues"), legend=None)
    ).properties(width=200, height=200, title="Covariance matrix")
    _mu, _cov, _ = gp_posterior(s4_k, [], [], xs_grid)
    _F = sample_mvn(_mu, _cov, 5, np.random.default_rng(3 + s4_roll.value))
    alt.hconcat(_row, _heat, gp_chart(xs_grid, _mu, _cov, _F, title=f"Prior samples · {s4_kernel.value}", width=280, height=220)).resolve_scale(color="independent")
    return (s4_k,)


@app.cell
def _(mo, np, s4_k, s4_kernel):
    _d = np.array([0.5, 1.0, 2.0, 4.0])
    _r = s4_k(np.array([0.0]), _d)[0] / s4_k(np.array([0.0]), np.array([0.0]))[0, 0]
    mo.hstack([mo.stat(f"{v:.2f}", label=f"k(0,{d:g}) / k(0,0)") for d, v in zip(_d, _r)]
              + [mo.stat("yes" if s4_kernel.value != "Linear" else "no (grows with x)", label="stationary? (depends only on x−x')")], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. **Periodic**, $p=2.5$: the kernel row comes back to its peak at $x'=2.5, 5$. The covariance matrix shows stripes, and the readout $k(0,p)/k(0,0)$ would be exactly 1: points one period apart are *perfectly* correlated.
    2. **Linear**: the row is a straight line and the band is a bow-tie (narrowest at $x=c$). Move $c$ and the pinch point follows. This kernel is *not stationary*.
    3. **RBF**: shrink $\ell$ and watch the row narrow, the matrix's bright diagonal thin out and the samples get busier. Same story as Section 1.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Combining kernels: add and multiply

    > *"The really interesting thing about kernels is that we can combine them to create complex patterns."*

    Sums and products of valid kernels are valid kernels:

    $$k_{+}(x,x') = k_{\text{per}}(x,x') + k_{\text{lin}}(x,x') \qquad k_{\times}(x,x') = k_{\text{per}}(x,x')\cdot k_{\text{lin}}(x,x')$$

    * **Sum** = independent functions added: a *linear trend with periodic fluctuations riding on top*
      ("a steadily growing economy with seasonal fluctuations").
    * **Product** = one modulates the other: the *amplitude of the oscillations grows as we move away from the center*.

    The animation below morphs continuously through samples of the chosen GP. Every frame is an exact draw
    (the trick: $f(t) = L\,(z_1\cos t + z_2\sin t)$ with $z_1,z_2\sim\mathcal N(0,I)$ is again $\mathcal N(0, LL^\top)$),
    so watch the *family* of shapes, not one curve.
    """)
    return


@app.cell
def sample_animator_class(anywidget, traitlets):
    class SampleAnimator(anywidget.AnyWidget):
        """Animates GP prior samples: f(t) = L (z1 cos t + z2 sin t) stays exactly N(0, L L^T) for every t."""
        _esm = r"""
    function render({ model, el }) {
      const W = 680, H = 300, PAD = 30;
      const COLORS = ["#4e79a7", "#f28e2b", "#76b7b2", "#edc948", "#b07aa1", "#ff9da7"];
      const ctl = document.createElement("div"); ctl.style.marginBottom = "4px"; el.appendChild(ctl);
      const cv = document.createElement("canvas"); cv.width = W; cv.height = H;
      Object.assign(cv.style, { width: "100%", maxWidth: W + "px", border: "1px solid rgba(128,128,128,.3)", borderRadius: "6px" });
      el.appendChild(cv);
      const ctx = cv.getContext("2d");
      let timer = null, t = 0, Z = [];
      const randn = () => Math.sqrt(-2 * Math.log(Math.random() + 1e-12)) * Math.cos(2 * Math.PI * Math.random());
      function reroll() { const m = model.get("xs").length; Z = Array.from({ length: model.get("n_curves") }, () => [Array.from({ length: m }, randn), Array.from({ length: m }, randn)]); }
      function draw() {
        const xs = model.get("xs"), m = xs.length, L = model.get("L");
        if (Z.length !== model.get("n_curves") || (Z[0] && Z[0][0].length !== m)) reroll();
        const sd = xs.map((_, i) => { let s = 0; for (let q = 0; q <= i; q++) s += L[i * m + q] ** 2; return Math.sqrt(s); });
        const ymax = Math.max(1, 2.6 * Math.max(...sd));
        const X = i => PAD + i / (m - 1) * (W - 2 * PAD), Y = v => H / 2 - v / ymax * (H / 2 - PAD);
        ctx.clearRect(0, 0, W, H);
        ctx.fillStyle = "rgba(31,119,180,0.18)"; ctx.beginPath();
        xs.forEach((_, i) => i ? ctx.lineTo(X(i), Y(2 * sd[i])) : ctx.moveTo(X(i), Y(2 * sd[i])));
        for (let i = m - 1; i >= 0; i--) ctx.lineTo(X(i), Y(-2 * sd[i]));
        ctx.fill();
        ctx.strokeStyle = "#2ca02c"; ctx.lineWidth = 3; ctx.beginPath(); ctx.moveTo(X(0), Y(0)); ctx.lineTo(X(m - 1), Y(0)); ctx.stroke();
        Z.forEach(([z1, z2], c) => {
          const ct = Math.cos(t), st = Math.sin(t);
          ctx.strokeStyle = COLORS[c % COLORS.length]; ctx.lineWidth = 1.6; ctx.beginPath();
          for (let i = 0; i < m; i++) { let f = 0; for (let q = 0; q <= i; q++) f += L[i * m + q] * (z1[q] * ct + z2[q] * st); i ? ctx.lineTo(X(i), Y(f)) : ctx.moveTo(X(i), Y(f)); }
          ctx.stroke();
        });
        ctx.fillStyle = "gray"; ctx.font = "11px sans-serif";
        ctx.fillText(`x from ${xs[0]} to ${xs[m - 1]} · y-range ±${ymax.toFixed(1)} · ${model.get("label")}`, PAD, H - 8);
      }
      const btn = (txt, f) => { const b = document.createElement("button"); b.textContent = txt; b.onclick = f; b.style.marginRight = "6px"; ctl.appendChild(b); return b; };
      const play = btn("⏸ pause", () => { if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; } else start(); });
      btn("🎲 new curves", () => { reroll(); draw(); });
      function start() { timer = setInterval(() => { t += 0.03 * model.get("speed"); draw(); }, 33); play.textContent = "⏸ pause"; }
      for (const k of ["L", "xs", "n_curves", "label"]) model.on("change:" + k, () => { reroll(); draw(); });
      reroll(); draw(); start();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        L = traitlets.List(traitlets.Float()).tag(sync=True)
        xs = traitlets.List(traitlets.Float()).tag(sync=True)
        n_curves = traitlets.Int(4).tag(sync=True)
        speed = traitlets.Float(1.0).tag(sync=True)
        label = traitlets.Unicode("").tag(sync=True)

    return (SampleAnimator,)


@app.cell
def s5_combo(mo):
    s5_combo = mo.ui.radio(["Periodic", "Linear", "Periodic + Linear", "Periodic × Linear", "RBF + Periodic", "RBF × Periodic"], value="Periodic + Linear", label="kernel", inline=True)
    s5_combo
    return (s5_combo,)


@app.cell
def s5_anim(SampleAnimator, linear, mo, np, periodic, rbf, s5_combo):
    def combo_kernel(name):
        _per = lambda a, b: periodic(a, b, 1.0, 1.0, 2.0)
        _lin = lambda a, b: linear(a, b, 0.35, 0.3, 0.0)
        _rbf = lambda a, b: rbf(a, b, 1.0, 2.5)
        return {
            "Periodic": _per,
            "Linear": _lin,
            "Periodic + Linear": lambda a, b: _per(a, b) + _lin(a, b),
            "Periodic × Linear": lambda a, b: _per(a, b) * _lin(a, b),
            "RBF + Periodic": lambda a, b: _rbf(a, b) + 0.3 * _per(a, b),
            "RBF × Periodic": lambda a, b: _rbf(a, b) * _per(a, b),
        }[name]


    _xs = np.linspace(-5, 5, 90)
    _K = combo_kernel(s5_combo.value)(_xs, _xs)
    _L = np.linalg.cholesky(_K + 1e-6 * np.eye(len(_xs)))
    s5_anim = mo.ui.anywidget(SampleAnimator(L=_L.ravel().tolist(), xs=_xs.round(3).tolist(), n_curves=4, label=s5_combo.value))
    s5_anim
    return (combo_kernel,)


@app.cell
def _(alt, combo_kernel, np, pd, s5_combo, xs_grid):
    _rows = []
    for _name in ["Periodic", "Linear", "Periodic + Linear", "Periodic × Linear"]:
        _v = np.diag(combo_kernel(_name)(xs_grid, xs_grid))
        _rows.append(pd.DataFrame({"x": xs_grid, "prior σ(x)": np.sqrt(_v), "kernel": _name}))
    _df = pd.concat(_rows)
    alt.Chart(_df).mark_line(strokeWidth=2.5).encode(
        x="x:Q", y="prior σ(x):Q",
        color=alt.Color("kernel:N", scale=alt.Scale(scheme="dark2")),
        opacity=alt.condition(alt.datum.kernel == s5_combo.value, alt.value(1.0), alt.value(0.25)),
    ).properties(width="container", height=200, title="Linked readout: how wide is the prior band at each x? (selected kernel highlighted)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Start with **Periodic**, then **Linear**, then **Periodic + Linear**: you should see waves that drift up or down as a whole (trend + season).
    2. Switch to **Periodic × Linear**: the waves are tiny near $x=0$ and grow toward the edges, and they flip sign across zero. The σ(x) chart shows the bow-tie.
    3. **RBF × Periodic** is the "locally periodic" kernel: a repeating pattern that slowly changes its shape. Compare it with RBF + Periodic, where the pattern is fixed and a smooth wander is added.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Practical example: which kernel fits?

    The video's finale: noisy data from an unknown pattern (the dashed **true function**, which in real life we would not
    know) and three GPs fit to the same <span style="color:#d62728">**red points**</span>:

    * **RBF**: "not bad, but it misses the main pattern, which is the oscillations".
    * **Periodic**: "captures the periodicity but doesn't capture the linear pattern".
    * **Periodic + Linear**: "captures the underlying pattern beautifully, with reasonable uncertainty estimates everywhere".

    Here the true function is $f(x) = 0.3x + \sin(2\pi x / 2.5)$. Data are observed only on $[-5, 2]$, so the region
    $x>2$ is a genuine **extrapolation** test. We score each kernel with the log marginal likelihood
    $\log p(\mathbf y\mid X)$ (how well the kernel *explains* the data, the quantity you'd maximise to tune hyperparameters)
    and with the RMSE against the truth on held-out $x$.
    """)
    return


@app.cell
def s6_controls(mo):
    s6_n = mo.ui.slider(5, 60, value=25, step=1, label="# observations", show_value=True)
    s6_noise = mo.ui.slider(0.05, 1.0, value=0.25, step=0.05, label="noise σₙ", show_value=True)
    s6_roll = mo.ui.button(label="🎲 new dataset", value=0, on_click=lambda v: v + 1)
    mo.hstack([s6_n, s6_noise, s6_roll], justify="start", gap=2)
    return s6_n, s6_noise, s6_roll


@app.cell
def s6_chart(
    alt,
    gp_chart,
    gp_posterior,
    linear,
    np,
    pd,
    periodic,
    rbf,
    s6_n,
    s6_noise,
    s6_roll,
):
    def true_f(x):
        return 0.3 * x + np.sin(2 * np.pi * x / 2.5)


    _rng = np.random.default_rng(42 + s6_roll.value)
    s6_X = np.sort(_rng.uniform(-5, 2, s6_n.value))
    s6_y = true_f(s6_X) + s6_noise.value * _rng.standard_normal(s6_n.value)
    s6_kernels = {
        "RBF": lambda a, b: rbf(a, b, 1.5, 1.0),
        "Periodic": lambda a, b: periodic(a, b, 1.2, 1.0, 2.5),
        "Periodic + Linear": lambda a, b: periodic(a, b, 1.0, 1.0, 2.5) + linear(a, b, 0.4, 0.5, 0.0),
    }
    _charts, _rows = [], []
    _xs_test = np.linspace(-5, 5, 150)
    for _name, _k in s6_kernels.items():
        _mu, _cov, _lml = gp_posterior(_k, s6_X, s6_y, _xs_test, noise=s6_noise.value)
        _out = _xs_test > 2
        _rows.append({
            "kernel": _name,
            "log marginal likelihood": round(_lml, 2),
            "RMSE inside data [-5,2]": round(float(np.sqrt(np.mean((_mu[~_out] - true_f(_xs_test[~_out])) ** 2))), 3),
            "RMSE extrapolating (2,5]": round(float(np.sqrt(np.mean((_mu[_out] - true_f(_xs_test[_out])) ** 2))), 3),
            "truth inside ±2σ (%)": round(100 * float(np.mean(np.abs(_mu - true_f(_xs_test)) <= 2 * np.sqrt(np.clip(np.diag(_cov), 0, None)))), 1),
        })
        _truth = alt.Chart(pd.DataFrame({"x": _xs_test, "y": true_f(_xs_test)})).mark_line(color="#bcbd22", strokeDash=[5, 3], strokeWidth=2).encode(x="x:Q", y="y:Q")
        _edge = alt.Chart(pd.DataFrame({"x": [2.0]})).mark_rule(color="gray", strokeDash=[2, 2]).encode(x="x:Q")
        _charts.append(alt.layer(gp_chart(_xs_test, _mu, _cov, None, s6_X, s6_y, width=210, height=220, ydom=[-5, 5]), _truth, _edge).properties(title=_name))
    s6_scores = pd.DataFrame(_rows)
    alt.hconcat(*_charts)
    return (s6_scores,)


@app.cell
def _(mo, s6_scores):
    _best = s6_scores.loc[s6_scores["log marginal likelihood"].idxmax(), "kernel"]
    mo.vstack([
        mo.ui.table(s6_scores, selection=None),
        mo.md(f"Highest log marginal likelihood: **{_best}**. Dashed vertical line = edge of the data; dashed olive = the true function."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Look right of the dashed line: RBF's mean drifts back to 0 and its band balloons, Periodic keeps waving but around the wrong level, Periodic + Linear keeps climbing with the trend.
    2. Drop to 5 observations: every model is uncertain, and the likelihood ranking can flip. With little data, the prior (the kernel) is doing most of the work.
    3. Crank the noise to 1.0: the GP means get smoother and the bands wider. The *structure* the kernel assumes matters even more when the data are noisy.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 🧪 Sandbox

    A second canvas with everything unlocked: pick any kernel (including **periodic+linear** and **rbf+linear**),
    set its hyperparameters, and click your own data. Below it, numpy re-fits the same data with every kernel and
    ranks them by log marginal likelihood.

    **Challenges**
    * Draw a rising zig-zag and find a kernel + settings that extrapolates it convincingly to the right.
    * Make the RBF's log marginal likelihood beat periodic+linear on data *you* drew.
    * Find settings where 2 posterior samples look completely different between the data points but identical at them.
    """)
    return


@app.cell
def sandbox(GPCanvas, mo):
    sandbox = mo.ui.anywidget(GPCanvas(
        points=[[-4.5, -2.0], [-3.8, -0.6], [-3.2, -1.7], [-2.2, -0.3], [-1.5, -1.3], [-0.5, 0.2], [0.2, -0.7], [1.0, 0.8]],
        kernel="periodic+linear", ell=1.0, sigma=0.8, period=1.4, noise=0.15, n_samples=4,
    ))
    sandbox
    return (sandbox,)


@app.cell
def _(canvas_kernel, gp_posterior, mo, np, pd, sandbox, xs_grid):
    _v = sandbox.value
    _pts = np.array(_v["points"], dtype=float).reshape(-1, 2)
    _rows = []
    for _name in ["rbf", "periodic", "linear", "periodic+linear", "rbf+linear"]:
        _k = canvas_kernel({**_v, "kernel": _name})
        _, _, _lml = gp_posterior(_k, _pts[:, 0], _pts[:, 1], xs_grid[:2], noise=_v["noise"])
        _rows.append({"kernel": _name, "log marginal likelihood": round(_lml, 2), "selected": "◀" if _name == _v["kernel"] else ""})
    _df = pd.DataFrame(_rows).sort_values("log marginal likelihood", ascending=False)
    mo.hstack([
        mo.vstack([mo.stat(len(_pts), label="your points"), mo.stat(_df.iloc[0]["kernel"], label="best kernel (same hyperparameters)")]),
        mo.ui.table(_df, selection=None),
    ], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 📝 Quiz

    Answer all five, then flip the switch to see your score.
    """)
    return


@app.cell
def quiz(mo):
    quiz = mo.ui.dictionary({
        "q1": mo.ui.radio(["A single best-fit function", "A distribution over functions", "A distribution over data points only"], label="**1.** What does a Gaussian process give you?"),
        "q2": mo.ui.radio(["The mean function", "The kernel / covariance function", "The noise level"], label="**2.** Which ingredient decides how smooth or periodic the sampled functions are?"),
        "q3": mo.ui.radio(["Shrinks everywhere equally", "Shrinks near the data and stays large far away", "Grows near the data"], label="**3.** After observing data, the posterior uncertainty band..."),
        "q4": mo.ui.radio(["Periodic × Linear", "Periodic + Linear", "RBF alone"], label="**4.** Which kernel models 'a steadily growing economy with seasonal fluctuations'?"),
        "q5": mo.ui.radio(["0", "≈ 0.37", "1"], label="**5.** For a periodic kernel with period p, what is k(0, p) / k(0, 0)?"),
    })
    quiz
    return (quiz,)


@app.cell
def _(mo):
    reveal = mo.ui.switch(label="reveal answers")
    reveal
    return (reveal,)


@app.cell
def quiz_score(mo, np, periodic, quiz, reveal):
    _answers = {"q1": "A distribution over functions", "q2": "The kernel / covariance function",
                "q3": "Shrinks near the data and stays large far away", "q4": "Periodic + Linear", "q5": "1"}
    # q5's truth is computed with the notebook's own kernel instead of being hard-coded
    _r = periodic(np.array([0.0]), np.array([2.5]), 1.0, 1.0, 2.5)[0, 0] / periodic(np.array([0.0]), np.array([0.0]), 1.0, 1.0, 2.5)[0, 0]
    assert abs(_r - 1) < 1e-9
    _score = sum(quiz.value.get(_q) == _a for _q, _a in _answers.items())
    _lines = [f"- **{_q}**: {'✅' if quiz.value.get(_q) == _a else '❌'} correct answer: *{_a}*" for _q, _a in _answers.items()]
    mo.md(f"### Score: {_score} / 5\n" + "\n".join(_lines)) if reveal.value else mo.md("_Flip the switch when you're done._")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap

    | concept | the one-liner | formula | where you touched it |
    |---|---|---|---|
    | Gaussian process | a distribution over functions; any finite set of values is jointly Gaussian | $f \sim \mathcal{GP}(m, k)$ | §1 correlated dice |
    | Prior | before data: mean 0, constant ±2σ band | $f(X)\sim\mathcal N(0, K)$ | §2 prior samples |
    | Posterior | condition on data: mean passes through points, uncertainty shrinks near them | $m_*=K_*^\top(K+\sigma_n^2I)^{-1}y$, $\Sigma_*=K_{**}-K_*^\top(K+\sigma_n^2I)^{-1}K_*$ | §3 click-to-add canvas |
    | Kernels | the kernel *is* the assumption: smooth (RBF), repeating (periodic), trend (linear) | see §4 table | §4 kernel explorer |
    | Combining | sums add behaviours, products modulate them | $k_{per}+k_{lin}$, $k_{per}\cdot k_{lin}$ | §5 animator |
    | Model choice | the right structure extrapolates; the log marginal likelihood helps pick it | $\log p(y\mid X)$ | §6 + sandbox |

    **When to reach for a GP:** regression with limited data, or whenever you need explicit uncertainty estimates
    (e.g. Bayesian optimization, the video's suggested next topic). Exact GPs cost $O(n^3)$, so beyond a few thousand points
    you need approximations.
    """)
    return


if __name__ == "__main__":
    app.run()

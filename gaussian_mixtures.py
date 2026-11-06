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
    # Gaussian Mixture Models, by hand: watch EM think
    *An interactive companion to Atif Adib's video **"Gaussian Mixture Models Explained | Basics of ML"**.*

    k-means draws hard borders and assumes round clusters. A **Gaussian mixture model (GMM)** instead says:
    *every point was produced by one of K bell curves; I just don't know which one.* We'll rebuild the video's
    path: the 1D bell curve, the 2D bell curve, then the **Expectation-Maximization (EM)** loop, written in plain numpy,
    which you can step through one E-step and M-step at a time, starting from means you drag into place yourself.
    """),
        mo.Html('<iframe width="560" height="315" src="https://www.youtube.com/embed/hJLaHWaLsyg" title="Gaussian Mixture Models Explained" frameborder="0" allowfullscreen></iframe>'),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Color language used throughout.** Gaussian **1 = orange**, **2 = blue**, **3 = green**, **4 = amber**, **5 = purple**.
    A point's color is the Gaussian it most likely came from; it fades toward **gray** the more *uncertain* that is.
    The **red** curve or marker is always the quantity being measured (a likelihood, a mean you can drag).
    """), kind="info")
    return


@app.cell
def em_core(np, pd):
    # ---------- EM for a Gaussian mixture, from scratch in numpy ----------
    COLORS = ["#e4572e", "#2e86ab", "#76b041", "#f1a208", "#8e44ad"]
    GRAY = "#a0a0a0"

    def gauss_pdf_1d(x, mu, sigma):
        """N(x | mu, sigma^2): the bell curve from the video."""
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))

    def mvn_logpdf(X, mu, cov):
        """log N(x | mu, Sigma) for every row of X (Cholesky instead of an explicit inverse)."""
        d = X.shape[1]
        L = np.linalg.cholesky(cov)
        z = np.linalg.solve(L, (X - mu).T)
        return -0.5 * np.sum(z**2, axis=0) - np.log(np.diag(L)).sum() - 0.5 * d * np.log(2 * np.pi)

    def e_step(X, phi, mus, covs):
        """Expectation: weights w_ik = phi_k N(x_i|mu_k,Sigma_k) / sum_j (...). Also returns the log-likelihood."""
        logp = np.stack([np.log(phi[k]) + mvn_logpdf(X, mus[k], covs[k]) for k in range(len(phi))], axis=1)
        m = logp.max(axis=1, keepdims=True)
        lse = m[:, 0] + np.log(np.exp(logp - m).sum(axis=1))   # log-sum-exp, stable
        return np.exp(logp - lse[:, None]), float(lse.sum())

    def m_step(X, W, reg=1e-6):
        """Maximization: re-estimate phi, mu, Sigma as weighted averages."""
        Nk = W.sum(axis=0) + 1e-12
        phi = Nk / len(X)
        mus = (W.T @ X) / Nk[:, None]
        covs = []
        for k in range(W.shape[1]):
            D = X - mus[k]
            covs.append((W[:, k, None] * D).T @ D / Nk[k] + reg * np.eye(X.shape[1]))
        return phi, mus, np.array(covs)

    def run_em(X, init_means, n_iter=50, tol=1e-6, init_var=1.0):
        """Full EM. Returns a history list: init, then alternating E / M snapshots."""
        K = len(init_means)
        phi = np.full(K, 1.0 / K)
        mus = np.array(init_means, dtype=float)
        covs = np.array([init_var * np.eye(X.shape[1]) for _ in range(K)])
        hist = [dict(phase="init", it=0, phi=phi, mus=mus, covs=covs, W=None, ll=None)]
        prev = -np.inf
        for t in range(1, n_iter + 1):
            W, ll = e_step(X, phi, mus, covs)
            hist.append(dict(phase="E", it=t, phi=phi, mus=mus, covs=covs, W=W, ll=ll))
            phi, mus, covs = m_step(X, W)
            hist.append(dict(phase="M", it=t, phi=phi, mus=mus, covs=covs, W=W, ll=ll))
            if ll - prev < tol:
                break
            prev = ll
        W, ll = e_step(X, phi, mus, covs)
        hist.append(dict(phase="E", it=t + 1, phi=phi, mus=mus, covs=covs, W=W, ll=ll))
        return hist

    def kmeans(X, init_means, n_iter=100):
        """Plain Lloyd k-means from the same starting means, for comparison."""
        C = np.array(init_means, dtype=float)
        for _ in range(n_iter):
            lab = np.argmin(((X[:, None, :] - C[None]) ** 2).sum(-1), axis=1)
            newC = np.array([X[lab == k].mean(0) if np.any(lab == k) else C[k] for k in range(len(C))])
            if np.allclose(newC, C):
                break
            C = newC
        return lab, C

    def certainty_colors(W):
        """Color of the most likely Gaussian, faded toward gray by uncertainty."""
        K = W.shape[1]
        k = W.argmax(1)
        c = np.clip((W.max(1) - 1 / K) / (1 - 1 / K + 1e-12), 0, 1) if K > 1 else np.ones(len(W))
        g = np.array([160, 160, 160])
        out = []
        for ki, ci in zip(k, c):
            rgb = np.array([int(COLORS[ki][j:j + 2], 16) for j in (1, 3, 5)])
            mix = (ci * rgb + (1 - ci) * g).astype(int)
            out.append("#%02x%02x%02x" % tuple(mix))
        return out

    def ellipse_params(cov, nsig=2.0):
        vals, vecs = np.linalg.eigh(cov)
        v = vecs[:, 1]
        return float(np.degrees(np.arctan2(v[1], v[0]))), float(nsig * np.sqrt(vals[1])), float(nsig * np.sqrt(vals[0]))

    def ellipse_df(mus, covs, nsig=2.0, n=60):
        rows = []
        _t = np.linspace(0, 2 * np.pi, n)
        for k, (mu, cov) in enumerate(zip(mus, covs)):
            vals, vecs = np.linalg.eigh(cov)
            pts = (vecs @ np.diag(nsig * np.sqrt(vals)) @ np.vstack([np.cos(_t), np.sin(_t)])).T + mu
            for j, (a, b) in enumerate(pts):
                rows.append(dict(x=a, y=b, comp=f"G{k+1}", order=j))
        return pd.DataFrame(rows)

    def make_dataset(name, n=300, seed=0):
        """Generated toy datasets, all inside the [-6, 6] box."""
        r = np.random.default_rng(seed)
        if name == "two blobs (video)":
            A = r.multivariate_normal([-2, -1], [[1.0, 0.3], [0.3, 0.8]], n // 2)
            B = r.multivariate_normal([2, 1.5], [[0.8, -0.2], [-0.2, 1.2]], n - n // 2)
            X, y = np.vstack([A, B]), np.r_[np.zeros(n // 2), np.ones(n - n // 2)]
        elif name == "elongated (k-means fails)":
            A = r.multivariate_normal([0, -1.3], [[7.0, 0.0], [0.0, 0.12]], n // 2)
            B = r.multivariate_normal([0, 1.3], [[7.0, 0.0], [0.0, 0.12]], n - n // 2)
            X, y = np.vstack([A, B]), np.r_[np.zeros(n // 2), np.ones(n - n // 2)]
        elif name == "overlapping":
            A = r.multivariate_normal([-1.2, 0], [[1.2, 0.4], [0.4, 1.0]], n // 2)
            B = r.multivariate_normal([1.2, 0.5], [[1.0, -0.3], [-0.3, 1.3]], n - n // 2)
            X, y = np.vstack([A, B]), np.r_[np.zeros(n // 2), np.ones(n - n // 2)]
        else:  # "big + small"
            A = r.multivariate_normal([-1, 0], [[3.0, 0.0], [0.0, 3.0]], int(n * 0.8))
            B = r.multivariate_normal([3, 3], [[0.15, 0.0], [0.0, 0.15]], n - int(n * 0.8))
            X, y = np.vstack([A, B]), np.r_[np.zeros(int(n * 0.8)), np.ones(n - int(n * 0.8))]
        return np.clip(X, -5.9, 5.9), y.astype(int)

    DATASETS = ["two blobs (video)", "elongated (k-means fails)", "overlapping", "big + small"]
    return (
        COLORS,
        DATASETS,
        GRAY,
        certainty_colors,
        e_step,
        ellipse_df,
        ellipse_params,
        gauss_pdf_1d,
        kmeans,
        make_dataset,
        mvn_logpdf,
        run_em,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 · Why not just k-means?

    The video opens with the complaint: k-means *"cannot handle noise or overlapping clusters and only works really well
    when the clusters are globular."* k-means splits space with straight borders halfway between centroids,
    so every point gets a **hard** label and every cluster is implicitly a **circle of the same size**.

    A GMM replaces each centroid with a full bell curve (its own center **μ**, shape **Σ** and share **φ**) and replaces hard labels
    with **probabilities**. Pick a dataset and compare both, started from *identical* initial means.
    """)
    return


@app.cell
def s1_controls(DATASETS, mo):
    s1_data = mo.ui.dropdown(DATASETS, value="elongated (k-means fails)", label="Dataset")
    s1_seed = mo.ui.slider(0, 20, value=3, label="Initial-means seed")
    mo.hstack([s1_data, s1_seed], justify="start")
    return s1_data, s1_seed


@app.cell
def s1_chart(
    COLORS,
    alt,
    certainty_colors,
    ellipse_df,
    kmeans,
    make_dataset,
    mo,
    np,
    pd,
    run_em,
    s1_data,
    s1_seed,
):
    _X, _y = make_dataset(s1_data.value, 300, seed=1)
    _r = np.random.default_rng(s1_seed.value)
    _init = _X[_r.choice(len(_X), 2, replace=False)]
    _lab, _C = kmeans(_X, _init)
    _h = run_em(_X, _init, n_iter=200)
    _W = _h[-1]["W"]
    _df = pd.DataFrame(dict(x=_X[:, 0], y=_X[:, 1],
                            km=[COLORS[k] for k in _lab], gmm=certainty_colors(_W)))
    _dom = alt.Scale(domain=[-6, 6])
    def _panel(col, title):
        return alt.Chart(_df, title=title).mark_circle(size=28, opacity=0.85).encode(
            x=alt.X("x:Q", scale=_dom), y=alt.Y("y:Q", scale=_dom), color=alt.Color(f"{col}:N", scale=None)
        ).properties(width=300, height=300)
    _cent = alt.Chart(pd.DataFrame(dict(x=_C[:, 0], y=_C[:, 1]))).mark_point(shape="cross", size=250, color="black", strokeWidth=3).encode(x="x:Q", y="y:Q")
    _ell = alt.Chart(ellipse_df(_h[-1]["mus"], _h[-1]["covs"])).mark_line(color="black", strokeDash=[4, 3]).encode(
        x="x:Q", y="y:Q", detail="comp:N", order="order:Q")
    _acc_km = max(np.mean(_lab == _y), np.mean(_lab != _y))
    _acc_g = max(np.mean(_W.argmax(1) == _y), np.mean(_W.argmax(1) != _y))
    mo.vstack([
        alt.hconcat(_panel("km", "k-means (hard labels)") + _cent, _panel("gmm", "GMM via EM (soft, 2σ ellipses)") + _ell),
        mo.hstack([mo.stat(f"{_acc_km:.0%}", label="k-means agreement with truth"),
                   mo.stat(f"{_acc_g:.0%}", label="GMM agreement with truth"),
                   mo.stat(f"{int(np.sum(_W.max(1) < 0.8))}", label="GMM points with < 80% certainty")], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - On **elongated**, k-means cuts the two long stripes *vertically* (round-cluster assumption). The GMM learns flat, wide ellipses and separates them. Try several seeds.
    - Switch to **overlapping**: k-means paints a crisp border, the GMM grays out the contested middle, which is the *"gray out the points it is uncertain about"* moment from the end of the video.
    - **big + small**: k-means steals points from the big cloud for the small one. Why? (Hint: the border sits halfway between centroids, regardless of spread.)
    """), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 · The bell curve in 1D

    *"To describe a bell curve you need two things: its **mean** μ, around which it is centered, and its **variance** σ²,
    which describes how sharp or flat it is."* Given μ and σ, this equation gives the **likelihood** of any point x:

    $$\mathcal{N}(x \mid \mu, \sigma^2) = \frac{1}{\sigma\sqrt{2\pi}}\; \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

    Drag the **red triangle** (μ), the **red circle** on the shoulder (σ, sits at μ+σ) and the **purple diamond** (the point x).
    """)
    return


@app.cell
def Bell1D_class(anywidget, traitlets):
    class Bell1D(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const W = 640, H = 270, PAD = 34, XMIN = -6, XMAX = 6, Y0 = 215, TOP = 22;
      const NS = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none", touchAction: "none" });
      el.appendChild(svg);
      const mk = (tag, a, p = svg) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
      const toSvg = (e) => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const sx = v => PAD + (v - XMIN) / (XMAX - XMIN) * (W - 2 * PAD);
      const inv = px => XMIN + (px - PAD) / (W - 2 * PAD) * (XMAX - XMIN);
      const pdf = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2) / (s * Math.sqrt(2 * Math.PI));
      let drag = null;
      function handle(node, which) {
        node.style.cursor = "grab";
        node.addEventListener("pointerdown", e => { drag = which; svg.setPointerCapture(e.pointerId); e.preventDefault(); });
      }
      function draw() {
        const m = model.get("mu"), s = model.get("sigma"), x = model.get("x");
        const ymax = Math.max(0.45, pdf(m, m, s) * 1.15);
        const sy = v => Y0 - v / ymax * (Y0 - TOP);
        svg.innerHTML = "";
        let band = `M ${sx(m - s)} ${Y0} `, curve = "";
        for (let i = 0; i <= 300; i++) {
          const xv = XMIN + i / 300 * (XMAX - XMIN), y = sy(pdf(xv, m, s));
          curve += (i ? "L" : "M") + sx(xv) + " " + y + " ";
          if (xv >= m - s && xv <= m + s) band += `L ${sx(xv)} ${y} `;
        }
        band += `L ${sx(m + s)} ${Y0} Z`;
        mk("path", { d: band, fill: "#e4572e", opacity: 0.15 });
        mk("path", { d: curve, fill: "none", stroke: "#e4572e", "stroke-width": 3 });
        mk("line", { x1: PAD, x2: W - PAD, y1: Y0, y2: Y0, stroke: "currentColor", opacity: 0.6 });
        for (let t = XMIN; t <= XMAX; t++) {
          mk("line", { x1: sx(t), x2: sx(t), y1: Y0, y2: Y0 + 5, stroke: "currentColor", opacity: 0.6 });
          const tx = mk("text", { x: sx(t), y: Y0 + 18, "text-anchor": "middle", "font-size": 11, fill: "currentColor" }); tx.textContent = t;
        }
        mk("line", { x1: sx(m), x2: sx(m), y1: Y0, y2: sy(pdf(m, m, s)), stroke: "#e4572e", "stroke-dasharray": "4 3" });
        // sample point
        const px = pdf(x, m, s);
        mk("line", { x1: sx(x), x2: sx(x), y1: Y0, y2: sy(px), stroke: "#6c5b7b", "stroke-width": 3 });
        mk("circle", { cx: sx(x), cy: sy(px), r: 4, fill: "#6c5b7b" });
        const lab = mk("text", { x: sx(x) + 8, y: Math.max(TOP + 10, sy(px) - 8), "font-size": 13, fill: "#6c5b7b", "font-weight": "bold" });
        lab.textContent = `N(x) = ${px.toFixed(3)}   (${((x - m) / s).toFixed(2)} σ from μ)`;
        // handles
        handle(mk("polygon", { points: `${sx(m)},${Y0 + 2} ${sx(m) - 11},${Y0 + 24} ${sx(m) + 11},${Y0 + 24}`, fill: "#e4572e" }), "mu");
        handle(mk("circle", { cx: sx(m + s), cy: sy(pdf(m + s, m, s)), r: 9, fill: "#e4572e", stroke: "white", "stroke-width": 2 }), "sigma");
        handle(mk("polygon", { points: `${sx(x)},${Y0 + 28} ${sx(x) - 9},${Y0 + 39} ${sx(x)},${Y0 + 50} ${sx(x) + 9},${Y0 + 39}`, fill: "#6c5b7b" }), "x");
        const cap = mk("text", { x: PAD, y: 14, "font-size": 12, fill: "currentColor" });
        cap.textContent = `μ = ${m.toFixed(2)}   σ = ${s.toFixed(2)}   peak height 1/(σ√2π) = ${pdf(m, m, s).toFixed(3)}`;
      }
      svg.addEventListener("pointermove", e => {
        if (!drag) return;
        const v = Math.max(XMIN, Math.min(XMAX, Math.round(inv(toSvg(e).x) * 20) / 20));
        if (drag === "mu") model.set("mu", v);
        else if (drag === "x") model.set("x", v);
        else model.set("sigma", Math.max(0.25, Math.min(4, Math.abs(v - model.get("mu")))));
      });
      const stop = () => { if (drag) { drag = null; model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointerleave", stop);
      model.on("change:mu", draw); model.on("change:sigma", draw); model.on("change:x", draw);
      draw();
    }
    export default { render };
    """
        mu = traitlets.Float(0.0).tag(sync=True)
        sigma = traitlets.Float(1.0).tag(sync=True)
        x = traitlets.Float(1.5).tag(sync=True)

    return (Bell1D,)


@app.cell
def bell(Bell1D, mo):
    bell = mo.ui.anywidget(Bell1D(mu=0.0, sigma=1.0, x=1.5))
    bell
    return (bell,)


@app.cell
def bell_readout(bell, gauss_pdf_1d, mo, np):
    _m, _s, _x = bell.value["mu"], bell.value["sigma"], bell.value["x"]
    _lik = gauss_pdf_1d(_x, _m, _s)
    _z = (_x - _m) / _s
    mo.vstack([
        mo.hstack([
            mo.stat(f"{_lik:.4f}", label="likelihood N(x | μ, σ²)"),
            mo.stat(f"{_z:+.2f}", label="z = (x − μ)/σ"),
            mo.stat(f"{np.exp(-0.5 * _z**2):.1%}", label="height relative to the peak"),
            mo.stat(f"{gauss_pdf_1d(_m, _m, _s):.3f}", label="peak height 1/(σ√2π)"),
        ], justify="start"),
        mo.md(f"Python agrees with the widget: the point sits **{abs(_z):.2f} σ** from μ, so its likelihood is "
              f"e^(−z²/2) = **{np.exp(-0.5*_z**2):.3f}** times the peak."),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - Drag x onto μ: the likelihood is the maximum, *"the closer you sample from the mean the higher the likelihood."*
    - Put x at exactly μ+σ, then μ+2σ: relative height is always **60.7 %** and **13.5 %**, whatever σ is.
    - Halve σ with the shoulder handle: the peak **doubles** (the area must stay 1). A likelihood is a density, so it can exceed 1. Try σ = 0.25.
    """), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 · The bell curve in 2D

    *"The mean would be represented by a **vector**, with one mean for each dimension, and sigma by a **matrix** containing
    pairwise variances. The equation remains the same, except simple multiplications are replaced by matrix multiplications."*

    $$\mathcal{N}(\mathbf{x}\mid\boldsymbol\mu,\Sigma) = \frac{1}{2\pi\,|\Sigma|^{1/2}}\exp\!\Big(-\tfrac12(\mathbf{x}-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)\Big),
    \qquad \Sigma = \begin{pmatrix}\sigma_x^2 & \rho\,\sigma_x\sigma_y\\ \rho\,\sigma_x\sigma_y & \sigma_y^2\end{pmatrix}$$

    Compare to 1D: $(x-\mu)^2/\sigma^2$ became $(\mathbf{x}-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf{x}-\boldsymbol\mu)$, the **squared Mahalanobis distance**.
    The off-diagonal term ρ is what lets a GMM learn *tilted* ellipses, which k-means can't.
    """)
    return


@app.cell
def s3_controls(mo):
    s3 = mo.ui.dictionary({
        "μx": mo.ui.slider(-3, 3, step=0.1, value=0.0, label="μx"),
        "μy": mo.ui.slider(-3, 3, step=0.1, value=0.0, label="μy"),
        "σx": mo.ui.slider(0.3, 3, step=0.1, value=1.5, label="σx"),
        "σy": mo.ui.slider(0.3, 3, step=0.1, value=0.8, label="σy"),
        "ρ": mo.ui.slider(-0.95, 0.95, step=0.05, value=0.6, label="ρ (correlation)"),
        "px": mo.ui.slider(-4, 4, step=0.1, value=1.5, label="point x"),
        "py": mo.ui.slider(-4, 4, step=0.1, value=-1.0, label="point y"),
    })
    mo.hstack([mo.vstack([s3["μx"], s3["μy"]]), mo.vstack([s3["σx"], s3["σy"], s3["ρ"]]), mo.vstack([s3["px"], s3["py"]])], justify="start")
    return (s3,)


@app.cell
def s3_chart(alt, ellipse_df, mo, mvn_logpdf, np, pd, s3):
    _v = s3.value
    _mu = np.array([_v["μx"], _v["μy"]])
    _cov = np.array([[_v["σx"]**2, _v["ρ"]*_v["σx"]*_v["σy"]], [_v["ρ"]*_v["σx"]*_v["σy"], _v["σy"]**2]])
    _g = np.linspace(-4, 4, 46)
    _gx, _gy = np.meshgrid(_g, _g)
    _G = np.c_[_gx.ravel(), _gy.ravel()]
    _dens = np.exp(mvn_logpdf(_G, _mu, _cov))
    _step = _g[1] - _g[0]
    _heat = alt.Chart(pd.DataFrame(dict(x=_G[:, 0], y=_G[:, 1], x2=_G[:, 0] + _step, y2=_G[:, 1] + _step, density=_dens))).mark_rect().encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[-4, 4.2])), x2="x2:Q", y=alt.Y("y:Q", scale=alt.Scale(domain=[-4, 4.2])), y2="y2:Q",
        color=alt.Color("density:Q", scale=alt.Scale(scheme="oranges")), tooltip=["x:Q", "y:Q", "density:Q"])
    _ells = pd.concat([ellipse_df([_mu], [_cov], nsig=s).assign(level=f"{s}σ") for s in (1, 2)])
    _ell = alt.Chart(_ells).mark_line(color="black", strokeDash=[4, 3]).encode(x="x:Q", y="y:Q", detail="level:N", order="order:Q")
    _p = np.array([_v["px"], _v["py"]])
    _pt = alt.Chart(pd.DataFrame(dict(x=[_p[0]], y=[_p[1]]))).mark_point(shape="diamond", size=220, filled=True, color="#6c5b7b").encode(x="x:Q", y="y:Q")
    _mu_pt = alt.Chart(pd.DataFrame(dict(x=[_mu[0]], y=[_mu[1]]))).mark_point(shape="cross", size=200, color="#c0392b", strokeWidth=3).encode(x="x:Q", y="y:Q")
    _d = _p - _mu
    _maha2 = float(_d @ np.linalg.solve(_cov, _d))
    _lik2 = float(np.exp(mvn_logpdf(_p[None], _mu, _cov))[0])
    _eu = float(np.linalg.norm(_d))
    mo.hstack([
        (_heat + _ell + _mu_pt + _pt).properties(width=380, height=360, title="density N(x | μ, Σ) with 1σ and 2σ ellipses"),
        mo.vstack([
            mo.md(f"**Σ** = [[{_cov[0,0]:.2f}, {_cov[0,1]:.2f}], [{_cov[1,0]:.2f}, {_cov[1,1]:.2f}]]   |Σ| = {np.linalg.det(_cov):.3f}"),
            mo.stat(f"{_lik2:.4f}", label="likelihood of the ◆ point"),
            mo.stat(f"{_eu:.2f}", label="Euclidean distance to μ (what k-means uses)"),
            mo.stat(f"{np.sqrt(_maha2):.2f}", label="Mahalanobis distance to μ (what a GMM uses)"),
        ]),
    ], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - Set ρ = 0: the ellipse aligns with the axes and Σ is diagonal. Set σx = σy too: a circle, the only shape k-means "believes in".
    - Put the ◆ at (1.5, 1.5) and then at (1.5, −1.5) with ρ = 0.8. Same Euclidean distance, very different likelihood: the Mahalanobis distance follows the ellipse.
    - Push ρ toward ±0.95 and watch |Σ| shrink toward 0: the bell collapses onto a line and its peak shoots up.
    """), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 · Expectation-Maximization, one step at a time

    *"Just like any other clustering algorithm, GMM is iterative and has two steps."* With K Gaussians, mixing weights φ_k (the video's **Phi**),
    means μ_k and covariances Σ_k:

    **E-step (assignment / "expectation").** For every point, compute its likelihood under each Gaussian, then *"convert these likelihoods to
    weights by dividing each likelihood vector by its sum"*:

    $$w_{ik} = \frac{\varphi_k\,\mathcal{N}(\mathbf{x}_i\mid\boldsymbol\mu_k,\Sigma_k)}{\sum_{j=1}^{K}\varphi_j\,\mathcal{N}(\mathbf{x}_i\mid\boldsymbol\mu_j,\Sigma_j)}$$

    **M-step (update / "maximization").** Re-fit each Gaussian to *all* points, weighted by $w_{ik}$:

    $$N_k=\sum_i w_{ik},\quad \varphi_k=\frac{N_k}{n},\quad \boldsymbol\mu_k=\frac{1}{N_k}\sum_i w_{ik}\mathbf{x}_i,\quad \Sigma_k=\frac{1}{N_k}\sum_i w_{ik}(\mathbf{x}_i-\boldsymbol\mu_k)(\mathbf{x}_i-\boldsymbol\mu_k)^\top$$

    Repeat. The log-likelihood $\log L=\sum_i\log\sum_k\varphi_k\mathcal{N}(\mathbf{x}_i\mid\boldsymbol\mu_k,\Sigma_k)$ **never goes down**.
    The `e_step` / `m_step` / `run_em` functions in the first code cell are exactly these formulas in numpy (the video uses
    `scipy.stats.multivariate_normal`; here the density is written out with a Cholesky factor).

    **Step 1:** like k-means, *"we randomly initialize our Gaussians"*. Here **you** are the random initializer: drag the colored
    means (each starts as a circle with Σ = I and φ = 1/K) anywhere, even somewhere silly.
    """)
    return


@app.cell
def em_controls(DATASETS, mo):
    em_data = mo.ui.dropdown(DATASETS, value="two blobs (video)", label="Dataset")
    em_K = mo.ui.slider(1, 5, value=2, label="K (number of Gaussians)", show_value=True)
    em_iters = mo.ui.slider(1, 100, value=40, label="max iterations", show_value=True)
    mo.hstack([em_data, em_K, em_iters], justify="start")
    return em_K, em_data, em_iters


@app.cell
def InitDragger_class(anywidget, traitlets):
    class InitDragger(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const S = 440, PAD = 20, LO = -6, HI = 6;
      const NS = "http://www.w3.org/2000/svg";
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${S} ${S}`);
      Object.assign(svg.style, { width: "100%", maxWidth: S + "px", userSelect: "none", touchAction: "none", border: "1px solid #8884", borderRadius: "8px" });
      el.appendChild(svg);
      const mk = (tag, a, p = svg) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
      const toSvg = (e) => { const p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY; return p.matrixTransform(svg.getScreenCTM().inverse()); };
      const sx = v => PAD + (v - LO) / (HI - LO) * (S - 2 * PAD);
      const sy = v => S - PAD - (v - LO) / (HI - LO) * (S - 2 * PAD);
      const ix = p => LO + (p - PAD) / (S - 2 * PAD) * (HI - LO);
      const iy = p => LO + (S - PAD - p) / (S - 2 * PAD) * (HI - LO);
      const unit = (S - 2 * PAD) / (HI - LO);
      let drag = -1;
      function draw() {
        svg.innerHTML = "";
        const cols = model.get("colors");
        mk("line", { x1: sx(LO), x2: sx(HI), y1: sy(0), y2: sy(0), stroke: "currentColor", opacity: 0.15 });
        mk("line", { x1: sx(0), x2: sx(0), y1: sy(LO), y2: sy(HI), stroke: "currentColor", opacity: 0.15 });
        for (const [x, y] of model.get("points")) mk("circle", { cx: sx(x), cy: sy(y), r: 2.6, fill: "#a0a0a0", opacity: 0.8 });
        model.get("means").forEach(([x, y], k) => {
          mk("circle", { cx: sx(x), cy: sy(y), r: 2 * unit, fill: cols[k], opacity: 0.08, stroke: cols[k], "stroke-dasharray": "4 3" });
          const c = mk("circle", { cx: sx(x), cy: sy(y), r: 11, fill: cols[k], stroke: "white", "stroke-width": 2.5 });
          c.style.cursor = "grab";
          c.addEventListener("pointerdown", e => { drag = k; svg.setPointerCapture(e.pointerId); e.preventDefault(); });
          const t = mk("text", { x: sx(x), y: sy(y) + 4, "text-anchor": "middle", "font-size": 11, fill: "white", "font-weight": "bold" });
          t.textContent = "μ" + (k + 1); t.style.pointerEvents = "none";
        });
        const cap = mk("text", { x: 8, y: 16, "font-size": 12, fill: "currentColor" });
        cap.textContent = "drag the initial means (dashed circle = initial 2σ, Σ = I)";
      }
      svg.addEventListener("pointermove", e => {
        if (drag < 0) return;
        const p = toSvg(e), m = model.get("means").map(a => [...a]);
        m[drag] = [Math.max(LO, Math.min(HI, Math.round(ix(p.x) * 10) / 10)), Math.max(LO, Math.min(HI, Math.round(iy(p.y) * 10) / 10))];
        model.set("means", m);
      });
      const stop = () => { if (drag >= 0) { drag = -1; model.save_changes(); } };
      svg.addEventListener("pointerup", stop); svg.addEventListener("pointerleave", stop);
      model.on("change:means", draw); model.on("change:points", draw); draw();
    }
    export default { render };
    """
        points = traitlets.List().tag(sync=True)
        means = traitlets.List().tag(sync=True)
        colors = traitlets.List().tag(sync=True)

    return (InitDragger,)


@app.cell
def init_w(COLORS, InitDragger, em_K, em_data, make_dataset, mo, np):
    em_X, _ = make_dataset(em_data.value, 300, seed=1)
    _start = [[-4.5, 4.5], [4.5, 4.5], [0.0, -4.5], [-4.5, -4.5], [4.5, -1.0]]
    init_w = mo.ui.anywidget(InitDragger(points=np.round(em_X, 3).tolist(), means=_start[: em_K.value], colors=COLORS[: em_K.value]))
    init_w
    return em_X, init_w


@app.cell
def EMStepper_class(anywidget, traitlets):
    class EMStepper(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const S = 440, PAD = 20, LO = -6, HI = 6, SW = 440, SH = 90;
      const NS = "http://www.w3.org/2000/svg";
      const ctl = document.createElement("div");
      Object.assign(ctl.style, { display: "flex", gap: "6px", alignItems: "center", marginBottom: "6px", flexWrap: "wrap" });
      const cap = document.createElement("div");
      Object.assign(cap.style, { minHeight: "3em", fontSize: "14px", margin: "6px 0", maxWidth: S + "px" });
      const mkSvg = (w, h) => { const s = document.createElementNS(NS, "svg"); s.setAttribute("viewBox", `0 0 ${w} ${h}`);
        Object.assign(s.style, { width: "100%", maxWidth: w + "px", display: "block" }); return s; };
      const svg = mkSvg(S, S), spark = mkSvg(SW, SH);
      svg.style.border = "1px solid #8884"; svg.style.borderRadius = "8px";
      el.append(ctl, svg, cap, spark);
      const mk = (tag, a, p) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); p.appendChild(e); return e; };
      const sx = v => PAD + (v - LO) / (HI - LO) * (S - 2 * PAD);
      const sy = v => S - PAD - (v - LO) / (HI - LO) * (S - 2 * PAD);
      const unit = (S - 2 * PAD) / (HI - LO);
      let timer = null;
      const n = () => model.get("states").length;
      const go = d => { model.set("i", Math.max(0, Math.min(n() - 1, model.get("i") + d))); model.save_changes(); };
      const btn = (t, f, title) => { const b = document.createElement("button"); b.textContent = t; b.title = title || ""; b.onclick = f;
        Object.assign(b.style, { padding: "3px 10px", cursor: "pointer" }); ctl.appendChild(b); return b; };
      btn("⏮", () => go(-1e9), "back to init"); btn("◀", () => go(-1), "half-step back");
      btn("▶", () => go(1), "next half-step (E or M)");
      btn("E+M ⏭", () => go(2), "one full iteration"); btn("⏭", () => go(1e9), "converged");
      const play = btn("▶ play", () => {
        if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; return; }
        if (model.get("i") >= n() - 1) { model.set("i", 0); model.save_changes(); }
        play.textContent = "⏸ pause";
        timer = setInterval(() => { if (model.get("i") >= n() - 1) { clearInterval(timer); timer = null; play.textContent = "▶ play"; return; } go(1); }, 450);
      });
      function draw() {
        const st = model.get("states"), i = Math.min(model.get("i"), st.length - 1), s = st[i];
        svg.innerHTML = ""; spark.innerHTML = "";
        if (!s) return;
        const pts = model.get("points"), cols = model.get("kcolors");
        pts.forEach(([x, y], j) => mk("circle", { cx: sx(x), cy: sy(y), r: 3, fill: s.colors[j], opacity: 0.9 }, svg));
        s.ells.forEach(([ang, rx, ry], k) => {
          const [mx, my] = s.means[k];
          mk("ellipse", { cx: sx(mx), cy: sy(my), rx: Math.max(rx * unit, 1), ry: Math.max(ry * unit, 1), fill: cols[k], "fill-opacity": 0.07,
            stroke: cols[k], "stroke-width": s.phase === "M" ? 3 : 2, transform: `rotate(${-ang} ${sx(mx)} ${sy(my)})` }, svg);
          if (s.prev) { const [px, py] = s.prev[k];
            mk("line", { x1: sx(px), y1: sy(py), x2: sx(mx), y2: sy(my), stroke: cols[k], "stroke-width": 2, "stroke-dasharray": "3 3" }, svg);
            mk("circle", { cx: sx(px), cy: sy(py), r: 5, fill: "none", stroke: cols[k], opacity: 0.6 }, svg); }
          mk("circle", { cx: sx(mx), cy: sy(my), r: 8, fill: cols[k], stroke: "white", "stroke-width": 2 }, svg);
        });
        const tag = mk("text", { x: 8, y: 18, "font-size": 13, "font-weight": "bold", fill: s.phase === "E" ? "#c0392b" : (s.phase === "M" ? "#2e86ab" : "currentColor") }, svg);
        tag.textContent = `${s.phase === "init" ? "initialization" : (s.phase + "-step")} · iteration ${s.it} · state ${i}/${st.length - 1}`;
        cap.innerHTML = s.caption;
        // log-likelihood sparkline
        const ll = st.map(t => t.ll).filter(v => v !== null);
        if (ll.length > 1) {
          const lo = Math.min(...ll), hi = Math.max(...ll), L = 36, R = SW - 8, T = 10, B = SH - 18;
          const X = j => L + j / (ll.length - 1) * (R - L), Y = v => B - (v - lo) / (hi - lo || 1) * (B - T);
          mk("path", { d: ll.map((v, j) => (j ? "L" : "M") + X(j) + " " + Y(v)).join(" "), fill: "none", stroke: "#c0392b", "stroke-width": 2 }, spark);
          const cur = Math.max(0, Math.min(ll.length - 1, s.ll === null ? 0 : s.it - 1));
          if (s.ll !== null) mk("circle", { cx: X(cur), cy: Y(s.ll), r: 5, fill: "#c0392b" }, spark);
          const t = mk("text", { x: L, y: SH - 3, "font-size": 11, fill: "currentColor" }, spark);
          t.textContent = `log-likelihood per iteration: ${lo.toFixed(1)} → ${hi.toFixed(1)}`;
        }
      }
      model.on("change:i", draw); model.on("change:states", draw); draw();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        points = traitlets.List().tag(sync=True)
        states = traitlets.List().tag(sync=True)
        kcolors = traitlets.List().tag(sync=True)
        i = traitlets.Int(0).tag(sync=True)

    return (EMStepper,)


@app.cell
def em_step(
    COLORS,
    EMStepper,
    GRAY,
    certainty_colors,
    ellipse_params,
    em_X,
    em_iters,
    init_w,
    mo,
    np,
    run_em,
):
    def to_widget_states(hist, n):
        _out, _prev = [], None
        for _h in hist:
            _cols = certainty_colors(_h["W"]) if _h["W"] is not None else [GRAY] * n
            _phi = ", ".join(f"{p:.2f}" for p in _h["phi"])
            if _h["phase"] == "init":
                _c = "Initialization: K Gaussians at your dragged means, Σ = I, φ = 1/K. No weights yet, so every point is gray. Press ▶."
            elif _h["phase"] == "E":
                _c = (f"<b style='color:#c0392b'>E-step</b>: each point computes its likelihood under every Gaussian and normalizes them into weights "
                      f"w<sub>ik</sub> (color = most likely Gaussian, gray = undecided). Gaussians did not move. log L = <b>{_h['ll']:.2f}</b>")
            else:
                _c = (f"<b style='color:#2e86ab'>M-step</b>: each Gaussian jumps to the weighted mean of the points (dashed arrow) and re-fits Σ "
                      f"to their weighted spread. Colors unchanged. φ = [{_phi}]")
            _out.append(dict(phase=_h["phase"], it=_h["it"], colors=_cols,
                             means=np.round(_h["mus"], 4).tolist(),
                             prev=(np.round(_prev, 4).tolist() if (_h["phase"] == "M" and _prev is not None) else None),
                             ells=[list(ellipse_params(c)) for c in _h["covs"]],
                             ll=_h["ll"], caption=_c))
            _prev = _h["mus"]
        return _out

    em_hist = run_em(em_X, init_w.value["means"], n_iter=em_iters.value)
    em_step = mo.ui.anywidget(EMStepper(points=np.round(em_X, 3).tolist(), states=to_widget_states(em_hist, len(em_X)),
                                       kcolors=COLORS[: len(init_w.value["means"])], i=0))
    em_step
    return em_hist, em_step


@app.cell
def em_readout(alt, em_hist, em_step, mo, np, pd):
    _i = min(em_step.value["i"], len(em_hist) - 1)
    _h = em_hist[_i]
    _rows = []
    for _k in range(len(_h["phi"])):
        _c = _h["covs"][_k]
        _rows.append({"Gaussian": f"G{_k+1}", "φ_k": round(float(_h["phi"][_k]), 3),
                      "μ_k": f"({_h['mus'][_k][0]:.2f}, {_h['mus'][_k][1]:.2f})",
                      "Σ_k": f"[[{_c[0,0]:.2f}, {_c[0,1]:.2f}], [{_c[1,0]:.2f}, {_c[1,1]:.2f}]]",
                      "N_k = Σ_i w_ik": (round(float(_h["W"][:, _k].sum()), 1) if _h["W"] is not None else "–")})
    _lls = pd.DataFrame([dict(iteration=h["it"], logL=h["ll"]) for h in em_hist if h["phase"] == "E"])
    _line = alt.Chart(_lls).mark_line(point=True, color="#c0392b").encode(
        x=alt.X("iteration:Q"), y=alt.Y("logL:Q", scale=alt.Scale(zero=False), title="log-likelihood"))
    _rule = alt.Chart(pd.DataFrame(dict(iteration=[_h["it"]]))).mark_rule(strokeDash=[4, 3]).encode(x="iteration:Q")
    _mono = bool(np.all(np.diff(_lls["logL"].to_numpy()) >= -1e-8))
    mo.vstack([
        mo.md(f"**State {_i}: {_h['phase']} (iteration {_h['it']})**. Parameters right now:"),
        mo.ui.table(pd.DataFrame(_rows), selection=None),
        mo.hstack([mo.stat(f"{len(_lls)}", label="E-steps until convergence"),
                   mo.stat(f"{_lls['logL'].iloc[-1]:.2f}", label="final log-likelihood"),
                   mo.stat("yes" if _mono else "NO", label="log L never decreased?")], justify="start"),
        (_line + _rule).properties(width="container", height=200),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - Press **▶** slowly and alternate your attention: on **E** only colors change, on **M** only the Gaussians move. That's the whole algorithm.
    - Drag both means into the **same corner** (far from all data). EM still finds the two blobs, but needs more iterations. Check the counter.
    - Drag μ1 and μ2 to the **exact same spot**: the two Gaussians stay identical forever (symmetry is never broken). This is why real implementations initialize randomly.
    - Set K = 3 on *two blobs*: one blob gets split, and φ shows the split. Set K = 1: EM converges in one step (it's just the sample mean and covariance).
    """), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 · Soft assignments: graying out the doubt

    The video's punchline: the model *"is not only able to cluster the points but also gray out the points it is uncertain about,
    as they are equally likely to have come from either of the two Gaussians."* That uncertainty is just the final weight row
    $w_{i\cdot}$, the video's `predict_proba`: likelihoods times **Phi**, divided by their sum.

    Below, using the converged model from section 4, every point whose top weight is below the threshold is grayed out.
    """)
    return


@app.cell
def s5_thr(mo):
    s5_thr = mo.ui.slider(0.5, 0.99, step=0.01, value=0.8, label="certainty threshold  max_k w_ik ≥", show_value=True)
    s5_thr
    return (s5_thr,)


@app.cell
def s5_chart(COLORS, GRAY, alt, em_X, em_hist, mo, np, pd, s5_thr):
    _W = em_hist[-1]["W"]
    _K = _W.shape[1]
    _top = _W.max(1)
    _df = pd.DataFrame(dict(x=em_X[:, 0], y=em_X[:, 1], top=_top,
                            color=[COLORS[k] if t >= s5_thr.value else GRAY for k, t in zip(_W.argmax(1), _top)],
                            **{f"w{k+1}": np.round(_W[:, k], 3) for k in range(_K)}))
    _sc = alt.Chart(_df).mark_circle(size=32).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[-6, 6])), y=alt.Y("y:Q", scale=alt.Scale(domain=[-6, 6])),
        color=alt.Color("color:N", scale=None), tooltip=[f"w{k+1}:Q" for k in range(_K)]).properties(width=330, height=330, title="hard colors only where certain")
    _hist = alt.Chart(_df).mark_bar(color="#6c5b7b").encode(
        x=alt.X("top:Q", bin=alt.Bin(extent=[1 / _K if _K > 1 else 0, 1], maxbins=25), title="max_k w_ik"), y="count()").properties(width=300, height=150, title="how sure is the model?")
    _thr = alt.Chart(pd.DataFrame(dict(t=[s5_thr.value]))).mark_rule(color="#c0392b", strokeWidth=2).encode(x="t:Q")
    _unc = _df.sort_values("top").head(5)[["x", "y"] + [f"w{k+1}" for k in range(_K)]].round(2)
    _ent = float(-(np.clip(_W, 1e-12, 1) * np.log(np.clip(_W, 1e-12, 1))).sum(1).mean())
    mo.hstack([_sc, mo.vstack([_hist + _thr,
        mo.hstack([mo.stat(f"{int((_top < s5_thr.value).sum())} / {len(_top)}", label="grayed-out points"),
                   mo.stat(f"{_ent:.3f} nats", label="mean entropy of w_i")], justify="start"),
        mo.md("**The five most doubtful points** (rows sum to 1):"), mo.ui.table(_unc, selection=None)])], justify="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    - Switch section 4 to **overlapping**: the gray band between the clusters grows and the histogram gets a fat left tail.
    - On **two blobs**, almost every point is > 0.99 sure: a GMM with well-separated clusters behaves almost like k-means.
    - Raise the threshold to 0.99. The gray points form a *curved* band (a quadratic boundary), not k-means' straight line, because the two Σ differ.
    """), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 · Sandbox: generate a mixture, fit a mixture

    Build a hidden mixture of `true K` random Gaussians, then fit a GMM with your own `fit K`. The bar chart on the right fits every
    K from 1 to 6 and scores it with the **BIC** $=-2\log L + p\log n$ with $p = 6K-1$ free parameters in 2D. Lowest BIC wins, which is a
    standard way to pick K when you don't know it.
    """)
    return


@app.cell
def sb_controls(mo):
    sb = mo.ui.dictionary({
        "true_K": mo.ui.slider(1, 5, value=3, label="true K", show_value=True),
        "fit_K": mo.ui.slider(1, 6, value=3, label="fit K", show_value=True),
        "spread": mo.ui.slider(0.2, 2.0, step=0.1, value=0.7, label="cluster spread", show_value=True),
        "n": mo.ui.slider(60, 600, step=20, value=300, label="points n", show_value=True),
        "seed": mo.ui.slider(0, 50, value=7, label="data seed", show_value=True),
        "init_seed": mo.ui.slider(0, 50, value=0, label="init seed", show_value=True),
    })
    mo.hstack([mo.vstack([sb["true_K"], sb["fit_K"], sb["spread"]]), mo.vstack([sb["n"], sb["seed"], sb["init_seed"]])], justify="start")
    return (sb,)


@app.cell
def sb_fit(np, pd, run_em, sb):
    def sample_mixture(K, spread, n, seed):
        _r = np.random.default_rng(seed)
        _C = _r.uniform(-3.8, 3.8, size=(K, 2))
        _phi = _r.dirichlet(np.full(K, 4.0))
        _z = _r.choice(K, size=n, p=_phi)
        _X = np.empty((n, 2))
        for _k in range(K):
            _A = _r.normal(size=(2, 2)) * spread
            _cov = _A @ _A.T + 0.05 * np.eye(2)
            _X[_z == _k] = _r.multivariate_normal(_C[_k], _cov, int((_z == _k).sum()))
        return np.clip(_X, -5.9, 5.9), _z

    def fit_gmm(X, K, seed, n_iter=200):
        _r = np.random.default_rng(seed)
        return run_em(X, X[_r.choice(len(X), K, replace=False)], n_iter=n_iter)

    sb_X, sb_z = sample_mixture(sb.value["true_K"], sb.value["spread"], sb.value["n"], sb.value["seed"])
    sb_hist = fit_gmm(sb_X, sb.value["fit_K"], sb.value["init_seed"])
    sb_bic = pd.DataFrame([dict(K=_k, BIC=-2 * fit_gmm(sb_X, _k, sb.value["init_seed"])[-1]["ll"] + (6 * _k - 1) * np.log(len(sb_X)))
                           for _k in range(1, 7)])
    return sb_X, sb_bic, sb_hist


@app.cell
def sb_chart(
    alt,
    certainty_colors,
    ellipse_df,
    mo,
    np,
    pd,
    sb,
    sb_X,
    sb_bic,
    sb_hist,
):
    _f = sb_hist[-1]
    _df = pd.DataFrame(dict(x=sb_X[:, 0], y=sb_X[:, 1], color=certainty_colors(_f["W"])))
    _sc = alt.Chart(_df).mark_circle(size=26).encode(
        x=alt.X("x:Q", scale=alt.Scale(domain=[-6, 6])), y=alt.Y("y:Q", scale=alt.Scale(domain=[-6, 6])), color=alt.Color("color:N", scale=None))
    _el = alt.Chart(ellipse_df(_f["mus"], _f["covs"])).mark_line(color="black", strokeDash=[4, 3]).encode(x="x:Q", y="y:Q", detail="comp:N", order="order:Q")
    _best = int(sb_bic.loc[sb_bic.BIC.idxmin(), "K"])
    _bar = alt.Chart(sb_bic.assign(best=lambda d: np.where(d.K == _best, "best", "other"))).mark_bar().encode(
        x="K:O", y=alt.Y("BIC:Q", scale=alt.Scale(zero=False)),
        color=alt.Color("best:N", scale=alt.Scale(domain=["best", "other"], range=["#c0392b", "#bbb"]), legend=None)).properties(width=220, height=300, title="BIC by K (lower = better)")
    mo.vstack([
        alt.hconcat((_sc + _el).properties(width=340, height=340, title=f"fit K = {sb.value['fit_K']} (true K = {sb.value['true_K']})"), _bar),
        mo.hstack([mo.stat(f"{_f['ll']:.1f}", label="final log-likelihood"),
                   mo.stat(f"{_f['it']}", label="EM iterations"),
                   mo.stat(f"{_best}", label="K chosen by BIC"),
                   mo.stat("✅" if _best == sb.value["true_K"] else "❌", label="BIC found the true K?")], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Challenges**
    1. Find a `data seed` where BIC picks the wrong K at spread 0.7. What do the clusters look like? (Usually two overlap almost completely.)
    2. Set `fit K` = true K and change only `init seed`. Does the final log-likelihood change? EM only finds a **local** maximum, so production code runs several restarts.
    3. Push `fit K` to 6 with n = 60. Watch for a tiny, needle-thin ellipse: a Gaussian collapsing onto a few points sends the likelihood toward +∞ (the reason for the `reg` term in `m_step`).
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 · Quiz
    Answer all six, then flip the switch.
    """)
    return


@app.cell
def quiz(mo):
    quiz = mo.ui.dictionary({
        "q1": mo.ui.dropdown(["moves the means", "computes each point's weights w_ik", "picks K", "computes the BIC"], label="1. The E-step…"),
        "q2": mo.ui.dropdown(["0", "0.25", "0.5", "1"], label="2. Two identical Gaussians with equal φ; a point exactly halfway between them has w_i1 = ?"),
        "q3": mo.ui.dropdown(["it halves", "it stays the same", "it doubles", "it quadruples"], label="3. In 1D, you double σ. The peak height of the bell…"),
        "q4": mo.ui.dropdown(["a circle", "an ellipse tilted up-right", "an ellipse tilted down-right", "a vertical ellipse"], label="4. Σ = [[1, 0.8], [0.8, 1]] looks like…"),
        "q5": mo.ui.dropdown(["it can go up or down", "it never decreases", "it always decreases", "it is constant"], label="5. Across EM iterations, the log-likelihood…"),
        "q6": mo.ui.dropdown(["elongated or differently sized clusters", "tiny datasets only", "1D data only", "never"], label="6. A GMM beats k-means most clearly on…"),
    })
    quiz_reveal = mo.ui.switch(label="Reveal answers")
    mo.vstack([quiz, quiz_reveal])
    return quiz, quiz_reveal


@app.cell
def quiz_feedback(e_step, gauss_pdf_1d, mo, np, quiz, quiz_reveal):
    _answers = {"q1": "computes each point's weights w_ik", "q2": "0.5", "q3": "it halves",
                "q4": "an ellipse tilted up-right", "q5": "it never decreases", "q6": "elongated or differently sized clusters"}
    _why = {"q1": "E = assignment: likelihoods normalized to weights. The M-step moves the means.",
            "q2": f"Equal likelihoods normalize to 1/2. Check with the notebook: w = {e_step(np.array([[0.0, 0.0]]), np.array([.5, .5]), np.array([[-1., 0], [1., 0]]), np.array([np.eye(2)] * 2))[0][0, 0]:.2f}",
            "q3": f"Peak = 1/(σ√2π): {gauss_pdf_1d(0, 0, 1):.3f} → {gauss_pdf_1d(0, 0, 2):.3f}.",
            "q4": "Positive covariance: x and y grow together.",
            "q5": "EM is guaranteed to be monotone (see the 'never decreased' stat in section 4).",
            "q6": "k-means assumes round, equal-sized clusters; Σ_k and φ_k remove that assumption."}
    _score = sum(quiz.value[k] == v for k, v in _answers.items())
    if not quiz_reveal.value:
        _out = mo.md(f"_Answered {sum(v is not None for v in quiz.value.values())}/6. Flip the switch when ready._")
    else:
        _out = mo.vstack([mo.stat(f"{_score} / 6", label="score")] + [
            mo.md(f"{'✅' if quiz.value[k] == v else '❌'} **{k[1]}.** {v}. {_why[k]}") for k, v in _answers.items()])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap

    | Idea | Formula | Where you touched it |
    |---|---|---|
    | k-means limits | hard labels, round equal clusters | §1 side-by-side |
    | 1D bell curve | $\mathcal N(x\mid\mu,\sigma^2)=\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2}$ | §1D draggable μ, σ, x |
    | 2D bell curve | μ vector, Σ matrix, Mahalanobis distance | §3 sliders + heatmap |
    | E-step | $w_{ik}\propto\varphi_k\,\mathcal N(\mathbf x_i\mid\boldsymbol\mu_k,\Sigma_k)$, rows sum to 1 | §4 stepper, colors change |
    | M-step | weighted φ, μ, Σ | §4 stepper, Gaussians move |
    | Convergence | log L never decreases; local optimum only | §4 log-L curve, §6 init seed |
    | Soft clustering | gray = $\max_k w_{ik}$ small | §5 threshold |
    | Choosing K | BIC $=-2\log L+p\log n$ | §6 BIC bars |

    *Next steps:* try `sklearn.mixture.GaussianMixture` (same algorithm, k-means++ init, `n_init` restarts, `covariance_type`), or fit a GMM to your own data, as the video suggests.
    """)
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def lesson_imports():
    import marimo as mo
    import numpy as np
    import altair as alt
    import anywidget
    import traitlets
    from scipy.stats import norm

    return alt, anywidget, mo, norm, np, traitlets


@app.cell(hide_code=True)
def lesson_welcome(mo):
    mo.md(r"""
    # Bayesian regression: same model, a new way to express uncertainty
    ### An interactive companion to *A Better Way to Learn Bayesian Statistics*

    **Start here:** drag a point in the first experiment. Then change your prior and watch
    what happens to the *same regression slope*. No coin flips required.

    Based on [Simplistics (QuantPsych)'s video](https://www.youtube.com/watch?v=ME-8sm7pZbM).
    The video introduces a **course philosophy**, not a worked numerical analysis. Its
    backbone is our backbone: familiar linear models → familiar visualizations → new
    Bayesian interpretations. The experiments below are an original Python companion.

    **You need:** basic regression intuition. **Allow:** about 20–30 minutes.

    1. Move data and fit a familiar line.
    2. Add a prior and see a posterior.
    3. Interpret credible intervals and predictions.
    4. Check fit, explore real data, and test your intuition.

    **Color key:** gray = classical/data-only; blue = prior; orange = posterior;
    purple = uncertainty for a new observation.

    > **Model contract.** We use a normal linear model with a user-specified, *known*
    > residual standard deviation σ. This allows exact Bayesian answers and instant
    > interaction. Ordinary `lm()` / `brm()` also estimate σ; their intervals need not
    > match ours. Our posterior draws are independent normal draws, **not MCMC**.
    """)
    return


@app.cell
def regression_math(alt, norm, np):
    def fit_regression(points, sigma, slope_mean=0.0, slope_sd=8.0, intercept_mean=50.0, intercept_sd=30.0):
        """Normal likelihood + independent normal coefficient priors; sigma is known.

        points: rows [predictor, response]; predictor is already in modeling units.
        Returns the exact joint posterior and the classical least-squares fit.
        """
        data = np.asarray(points, dtype=float)
        X = np.column_stack([np.ones(len(data)), data[:, 0]])
        y = data[:, 1]
        prior_mean = np.array([intercept_mean, slope_mean], dtype=float)
        prior_precision = np.diag([intercept_sd ** -2, slope_sd ** -2])
        precision = prior_precision + X.T @ X / sigma**2
        covariance = np.linalg.solve(precision, np.eye(2))
        mean = np.linalg.solve(precision, prior_precision @ prior_mean + X.T @ y / sigma**2)
        ols = np.linalg.lstsq(X, y, rcond=None)[0]
        ols_covariance = np.linalg.solve(X.T @ X / sigma**2, np.eye(2))
        return {"mean": mean, "cov": covariance, "ols": ols, "ols_cov": ols_covariance,
                "X": X, "y": y, "sigma": float(sigma)}

    def prediction_summary(fit, x, level=0.95):
        design = np.column_stack([np.ones(np.size(x)), np.atleast_1d(x)])
        mean = design @ fit["mean"]
        variance = np.einsum("ij,jk,ik->i", design, fit["cov"], design)
        z = norm.ppf((1 + level) / 2)
        return mean, z * np.sqrt(variance), z * np.sqrt(variance + fit["sigma"]**2)

    def records_chart(records):
        return alt.Chart(alt.InlineData(values=records))

    def check_regression_math():
        points = [[-1, 46], [0, 50], [1, 54]]
        fit = fit_regression(points, 2, slope_mean=0, slope_sd=2)
        # Slope precision: 1/4 + (1+0+1)/4 = 3/4; posterior slope = 8/3.
        assert np.isclose(fit["mean"][1], 8 / 3)
        assert np.isclose(fit["cov"][1, 1], 4 / 3)
        assert np.allclose(fit["ols"], [50, 4])
        mean, mean_half, new_half = prediction_summary(fit, [-1, 0, 1])
        assert np.all(new_half > mean_half)
        assert np.all(np.linalg.eigvalsh(fit["cov"]) > 0)
        broad = fit_regression(points, 2, slope_sd=1e6, intercept_sd=1e6)
        assert np.allclose(broad["mean"], broad["ols"])
        return "Exact-posterior checks passed."

    check_regression_math()
    return fit_regression, prediction_summary, records_chart


@app.cell(hide_code=True)
def regression_hook(mo):
    mo.md(r"""
    ## 1 · Keep the regression you already know
    *Video connection: incremental learning (5:55) and the Avengers example (8:23, 11:01).*

    The video's example predicts **PTSD from agility**. Here we use **fictional teaching
    data**, not clinical measurements or the video's original dataset. No causal claim is implied.

    \[
    \mathrm{PTSD}_i = \alpha + \beta x_i + \epsilon_i,
    \qquad x_i=(\mathrm{agility}_i-50)/10,
    \qquad \epsilon_i\sim\mathcal N(0,\sigma^2).
    \]

    So **α** is the expected score at agility 50; **β** is the change in expected score
    for **10 more agility points**. The model stays the same when we change the estimation framework.

    **Try this:** (1) drag one point upward; (2) compare 12 with 60 observations;
    (3) choose “Curved” and ask whether a straight line is adequate.
    An OLS line minimizes squared vertical errors—it does not prove the model is appropriate.
    """)
    return


@app.cell
def point_lab_class(anywidget, traitlets):
    class RegressionPlayground(anywidget.AnyWidget):
        points = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)
        scenarios = traitlets.Dict().tag(sync=True)
        _esm = r"""
        function render({model, el}) {
          el.innerHTML = `<div style="font:14px system-ui; padding:14px; border:1px solid #8885; border-radius:12px">
            <div class="toolbar" style="display:flex;gap:8px;flex-wrap:wrap"></div>
            <svg viewBox="0 0 760 330" style="width:100%;touch-action:none" role="img" aria-label="Drag response values to explore regression"></svg>
            <label>Keyboard alternative: point <select aria-label="Choose observation"></select></label>
            <label> score <input type="range" min="0" max="100" step="0.5" aria-label="Observation score"></label>
            <span class="readout" aria-live="polite"></span>
          </div>`;
          const svg=el.querySelector('svg'), select=el.querySelector('select'), input=el.querySelector('input');
          const toolbar=el.querySelector('.toolbar'), readout=el.querySelector('.readout');
          const ns='http://www.w3.org/2000/svg';
          const sx=x=>50+(x-10)/80*680, sy=y=>285-y/100*260;
          function mk(tag, attrs, text) { const e=document.createElementNS(ns,tag);
            for(const [k,v] of Object.entries(attrs)) e.setAttribute(k,v);
            if(text!==undefined)e.textContent=text; svg.appendChild(e); return e; }
          for (const [label, pts] of Object.entries(model.get('scenarios'))) {
            const b=document.createElement('button'); b.textContent=label;
            b.onclick=()=>{model.set('points',pts.map(p=>[...p]));model.save_changes();}; toolbar.appendChild(b);
          }
          let dragging=-1;
          function draw() {
            const p=model.get('points'); svg.replaceChildren();
            for(let y=0;y<=100;y+=25){mk('line',{x1:50,x2:730,y1:sy(y),y2:sy(y),stroke:'#8883'});
              mk('text',{x:40,y:sy(y)+4,'text-anchor':'end',fill:'currentColor','font-size':12},y);}
            for(let x=10;x<=90;x+=20)mk('text',{x:sx(x),y:305,'text-anchor':'middle',fill:'currentColor','font-size':12},x);
            mk('text',{x:390,y:326,'text-anchor':'middle',fill:'currentColor'},'Agility');
            mk('text',{x:50,y:16,fill:'currentColor','font-size':12},'Fictional PTSD score');
            const mx=p.reduce((a,v)=>a+v[0],0)/p.length, my=p.reduce((a,v)=>a+v[1],0)/p.length;
            const b=p.reduce((a,v)=>a+(v[0]-mx)*(v[1]-my),0)/p.reduce((a,v)=>a+(v[0]-mx)**2,0);
            mk('line',{x1:sx(10),x2:sx(90),y1:sy(my+b*(10-mx)),y2:sy(my+b*(90-mx)),stroke:'#888','stroke-width':3});
            p.forEach((v,i)=>{const c=mk('circle',{cx:sx(v[0]),cy:sy(v[1]),r:6,fill:'#168aad',stroke:'currentColor','stroke-width':0.5});
              c.style.cursor='ns-resize'; c.addEventListener('pointerdown',e=>{dragging=i;select.value=String(i);svg.setPointerCapture(e.pointerId);e.preventDefault();});});
            if(select.options.length!==p.length){select.replaceChildren();p.forEach((_,i)=>select.add(new Option(String(i+1),String(i))));}
            input.value=p[Number(select.value)||0][1];
            readout.textContent=` n = ${p.length}; OLS slope per 10 agility = ${(10*b).toFixed(2)}`;
          }
          function update(i,y){const p=model.get('points').map(v=>[...v]);p[i][1]=Math.max(0,Math.min(100,y));model.set('points',p);}
          svg.addEventListener('pointermove',e=>{if(dragging<0)return;const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;
            update(dragging,(285-p.matrixTransform(svg.getScreenCTM().inverse()).y)/260*100);});
          const stop=()=>{if(dragging>=0){dragging=-1;model.save_changes();}};
          svg.addEventListener('pointerup',stop);svg.addEventListener('pointercancel',stop);
          select.onchange=draw;
          input.oninput=()=>update(Number(select.value),Number(input.value));input.onchange=()=>model.save_changes();
          model.on('change:points',draw);draw();
          return()=>model.off('change:points',draw);
        }
        export default {render};
        """

    return (RegressionPlayground,)


@app.cell
def point_lab(RegressionPlayground, mo, np):
    _rng = np.random.default_rng(42)
    _a = np.linspace(15, 85, 12)
    _b = np.linspace(15, 85, 60)
    _noise = _rng.normal(0, 8, 12)
    _base = np.column_stack([_a, 50 - 4 * (_a - 50) / 10 + _noise]).tolist()
    _scenarios = {
        "Original · 12": _base,
        "More data · 60": np.column_stack([_b, 50 - 4 * (_b - 50) / 10 + _rng.normal(0, 8, 60)]).tolist(),
        "Flat": np.column_stack([_a, 50 + _noise]).tolist(),
        "Positive": np.column_stack([_a, 50 + 4 * (_a - 50) / 10 + _noise]).tolist(),
        "Curved": np.column_stack([_a, 25 + 3.5 * ((_a - 50) / 10)**2 + _noise / 2]).tolist(),
    }
    data_lab = mo.ui.anywidget(RegressionPlayground(points=_base, scenarios=_scenarios))
    data_lab
    return (data_lab,)


@app.cell
def observations(data_lab, fit_regression, mo, np):
    observations = np.asarray(data_lab.value["points"], dtype=float)
    model_points = np.column_stack([(observations[:, 0] - 50) / 10, observations[:, 1]])
    _fit = fit_regression(model_points, 8)
    mo.hstack([
        mo.stat(label="Observations", value=str(len(observations))),
        mo.stat(label="Classical intercept at agility 50", value=f"{_fit['ols'][0]:.2f}"),
        mo.stat(label="Classical slope per 10 agility", value=f"{_fit['ols'][1]:+.2f}"),
    ], justify="space-around")
    return model_points, observations


@app.cell(hide_code=True)
def prior_hook(mo):
    mo.md(r"""
    ## 2 · Same slope, now with a prior
    *Video connection: changing frameworks and adding prior syntax (11:01–14:00).*

    A **prior** expresses plausible coefficient values *before these observations*.
    The **likelihood** scores how well coefficients explain the observations under the model.
    The **posterior** combines both:

    \[
    p(\alpha,\beta\mid y,X,\sigma)\ \propto\
    p(y\mid X,\alpha,\beta,\sigma)\,p(\alpha)\,p(\beta).
    \]

    Here \(\alpha\sim\mathcal N(50,30^2)\) and \(\beta\sim\mathcal N(m,s^2)\), independently.
    Change **m** (where the prior is centered) and **s** (how strongly it concentrates).
    Smaller s means a stronger prior. σ is residual scatter around a line, not uncertainty about its slope.

    **Predict first:** if the data suggest a negative slope but the prior insists on +6,
    where will the posterior go?

    **Try this:** (1) set m = +6 and s = 0.5; (2) widen s to 15;
    (3) switch from 12 to 60 observations above. A strong prior matters most when data are weak.
    Priors should have a defensible origin; tuning them until an answer is attractive is not validation.
    """)
    return


@app.cell
def prior_controls(mo):
    prior_mean = mo.ui.slider(-10, 10, step=0.5, value=0, label="Prior slope center m", show_value=True)
    prior_sd = mo.ui.slider(0.25, 20, step=0.25, value=8, label="Prior slope SD s", show_value=True)
    noise_sd = mo.ui.slider(2, 25, step=0.5, value=8, label="Assumed known residual SD σ", show_value=True)
    interval_level = mo.ui.slider(50, 99, step=1, value=95, label="Interval probability (%)", show_value=True)
    mo.vstack([mo.hstack([prior_mean, prior_sd], widths="equal"), mo.hstack([noise_sd, interval_level], widths="equal")])
    return interval_level, noise_sd, prior_mean, prior_sd


@app.cell
def posterior_fit(
    fit_regression,
    interval_level,
    mo,
    model_points,
    noise_sd,
    norm,
    np,
    prior_mean,
    prior_sd,
):
    bayes_fit = fit_regression(model_points, noise_sd.value, prior_mean.value, prior_sd.value)
    slope_mean = float(bayes_fit["mean"][1])
    slope_sd = float(np.sqrt(bayes_fit["cov"][1, 1]))
    _z = norm.ppf((1 + interval_level.value / 100) / 2)
    slope_interval = (slope_mean - _z * slope_sd, slope_mean + _z * slope_sd)
    negative_probability = float(norm.cdf(0, loc=slope_mean, scale=slope_sd))
    mo.hstack([
        mo.stat(label="Posterior mean slope", value=f"{slope_mean:+.2f}"),
        mo.stat(label=f"{interval_level.value}% credible interval", value=f"[{slope_interval[0]:.2f}, {slope_interval[1]:.2f}]"),
        mo.stat(label="P(slope < 0 | data, model)", value=f"{negative_probability:.1%}"),
    ], justify="space-around")
    return bayes_fit, negative_probability, slope_mean, slope_sd


@app.cell
def prior_posterior_chart(
    alt,
    bayes_fit,
    mo,
    norm,
    np,
    prior_mean,
    prior_sd,
    records_chart,
    slope_mean,
    slope_sd,
):
    _classical_sd = np.sqrt(bayes_fit["ols_cov"][1, 1])
    _centers = [prior_mean.value, float(bayes_fit["ols"][1]), slope_mean]
    _sds = [prior_sd.value, float(_classical_sd), slope_sd]
    _labels = ["Prior", "Data-only slope curve", "Posterior"]
    _grid = np.unique(np.concatenate([np.linspace(m - 4*s, m + 4*s, 180) for m, s in zip(_centers, _sds)]))
    _rows = [{"slope": float(x), "density": float(norm.pdf(x, m, s)), "distribution": label}
             for m, s, label in zip(_centers, _sds, _labels) for x in _grid]
    _plot = records_chart(_rows).mark_line(strokeWidth=3).encode(
        x=alt.X("slope:Q", title="Slope β · score change per 10 agility"),
        y=alt.Y("density:Q", title="Probability density"),
        color=alt.Color("distribution:N", scale=alt.Scale(domain=_labels, range=["#168aad", "#888888", "#e67722"]), legend=alt.Legend(orient="bottom")),
        tooltip=["distribution:N", alt.Tooltip("slope:Q", format=".2f"), alt.Tooltip("density:Q", format=".3f")],
    ).properties(width="container", height=260)
    mo.vstack([_plot, mo.md("The gray curve is the normalized **profile likelihood** for the slope (known σ), centered on OLS. It is not a frequentist probability distribution for the fixed parameter. Curve height is density; probability comes from **area**.")])
    return


@app.cell
def same_fit_chart(
    alt,
    bayes_fit,
    interval_level,
    mo,
    np,
    observations,
    prediction_summary,
    records_chart,
):
    _xs = np.linspace(10, 90, 100)
    _grid_x = (_xs - 50) / 10
    _pred, _half, _ = prediction_summary(bayes_fit, _grid_x, interval_level.value / 100)
    _rows = [{"agility": float(x), "mean": float(y), "lower": float(y-h), "upper": float(y+h),
              "classical": float(bayes_fit["ols"][0] + bayes_fit["ols"][1] * (x-50)/10)}
             for x, y, h in zip(_xs, _pred, _half)]
    _base = records_chart(_rows).encode(x=alt.X("agility:Q", title="Agility"))
    _band = _base.mark_area(color="#e67722", opacity=0.18).encode(y=alt.Y("lower:Q", title="Fictional PTSD score"), y2="upper:Q")
    _posterior = _base.mark_line(color="#e67722", strokeWidth=3).encode(y="mean:Q")
    _classic = _base.mark_line(color="#888888", strokeDash=[6, 4], strokeWidth=2).encode(y="classical:Q")
    _points = records_chart([{"agility": float(a), "score": float(y)} for a, y in observations]).mark_circle(color="#168aad", size=65).encode(x="agility:Q", y="score:Q", tooltip=["agility:Q", "score:Q"])
    mo.vstack([(_band + _classic + _posterior + _points).properties(width="container", height=280),
               mo.md(f"**Same observations, same linear model:** dashed gray = OLS; orange = posterior mean. Shading = {interval_level.value}% **pointwise credible intervals for the mean**, not a band containing that fraction of individual observations or entire lines.")])
    return


@app.cell(hide_code=True)
def uncertainty_hook(mo):
    mo.md(r"""
    ## 3 · A posterior is a collection of plausible lines

    A coefficient table can hide the central idea: we have **uncertainty about α and β**,
    so many lines remain plausible. Each orange line below is one joint draw from the
    posterior. The faint lines show other draws. These are possible parameter values,
    not fresh datasets.

    A **95% credible interval** contains 95% of posterior probability for the specified
    quantity, conditional on the data, prior, and model. A **95% confidence procedure**
    covers the fixed true parameter in 95% of repeated datasets under its assumptions;
    that coverage is not a posterior probability statement about this particular interval.

    **Try this:** (1) press Play and watch plausible lines; (2) move the probe toward an
    edge; (3) increase prior strength above and watch the fan narrow. A narrower answer
    can reflect a stronger assumption, not more evidence.
    """)
    return


@app.cell
def posterior_lens_class(anywidget, traitlets):
    class PosteriorLens(anywidget.AnyWidget):
        draws = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)
        points = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)
        index = traitlets.Int(0).tag(sync=True)
        agility = traitlets.Float(50.0).tag(sync=True)
        _esm = r"""
        function render({model,el}) {
          el.innerHTML=`<div style="font:14px system-ui;padding:14px;border:1px solid #8885;border-radius:12px">
            <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap"><button class="next">Next line</button><button class="play">▶ Play</button><button class="reset">Reset draw</button>
            <label>Probe agility <input type="range" min="10" max="90" step="1" aria-label="Probe agility"></label></div>
            <svg viewBox="0 0 760 330" style="width:100%" role="img" aria-label="Plausible posterior regression lines"></svg>
            <div class="caption" aria-live="polite"></div>
          </div>`;
          const svg=el.querySelector('svg'), slider=el.querySelector('input'), caption=el.querySelector('.caption'), play=el.querySelector('.play');
          let timer=null;
          const samples=model.get('draws'), pts=model.get('points');
          const all=[...pts.map(p=>p[1]),...samples.flatMap(p=>[p[0]-4*p[1],p[0]+4*p[1]])];
          const lo=Math.floor(Math.min(...all)/10)*10-5,hi=Math.ceil(Math.max(...all)/10)*10+5;
          const sx=x=>50+(x-10)/80*680,sy=y=>285-(y-lo)/(hi-lo)*260;
          function mk(tag,attrs,text){const e=document.createElementNS('http://www.w3.org/2000/svg',tag);
            for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);if(text!==undefined)e.textContent=text;svg.appendChild(e);return e;}
          function draw(){svg.replaceChildren();const index=model.get('index'),a=model.get('agility'),d=samples[index];
            for(let i=0;i<5;i++){const y=lo+(hi-lo)*i/4;mk('line',{x1:50,x2:730,y1:sy(y),y2:sy(y),stroke:'#8883'});
              mk('text',{x:42,y:sy(y)+4,'text-anchor':'end',fill:'currentColor','font-size':12},y.toFixed(0));}
            for(let x=10;x<=90;x+=20)mk('text',{x:sx(x),y:305,'text-anchor':'middle',fill:'currentColor','font-size':12},x);
            mk('text',{x:390,y:326,'text-anchor':'middle',fill:'currentColor'},'Agility');
            for(const p of samples.slice(0,40))mk('line',{x1:sx(10),x2:sx(90),y1:sy(p[0]-4*p[1]),y2:sy(p[0]+4*p[1]),stroke:'#e67722','stroke-opacity':0.12});
            for(const p of pts)mk('circle',{cx:sx(p[0]),cy:sy(p[1]),r:4,fill:'#168aad'});
            mk('line',{x1:sx(10),x2:sx(90),y1:sy(d[0]-4*d[1]),y2:sy(d[0]+4*d[1]),stroke:'#e67722','stroke-width':3});
            mk('line',{x1:sx(a),x2:sx(a),y1:20,y2:285,stroke:'currentColor','stroke-dasharray':'4 4'});
            const y=d[0]+d[1]*(a-50)/10;
            mk('circle',{cx:sx(a),cy:sy(y),r:6,fill:'#e67722'});
            slider.value=a;caption.textContent=`Draw ${index+1}/${samples.length}: α = ${d[0].toFixed(2)}, β = ${d[1].toFixed(2)}. This line predicts mean ${y.toFixed(1)} at agility ${a}.`;
          }
          const next=()=>{model.set('index',(model.get('index')+1)%samples.length);model.save_changes();};
          el.querySelector('.next').onclick=next;
          play.onclick=()=>{if(timer){clearInterval(timer);timer=null;play.textContent='▶ Play';}else{timer=setInterval(next,800);play.textContent='⏸ Pause';}};
          el.querySelector('.reset').onclick=()=>{clearInterval(timer);timer=null;play.textContent='▶ Play';model.set('index',0);model.save_changes();};
          slider.oninput=()=>model.set('agility',Number(slider.value));slider.onchange=()=>model.save_changes();
          model.on('change:index',draw);model.on('change:agility',draw);draw();
          return()=>{clearInterval(timer);model.off('change:index',draw);model.off('change:agility',draw);};
        }
        export default {render};
        """

    return (PosteriorLens,)


@app.cell
def posterior_lens(PosteriorLens, bayes_fit, mo, np, observations):
    posterior_draws = np.random.default_rng(2026).multivariate_normal(bayes_fit["mean"], bayes_fit["cov"], size=200)
    posterior_lens = mo.ui.anywidget(PosteriorLens(draws=posterior_draws.tolist(), points=observations.tolist()))
    posterior_lens
    return posterior_draws, posterior_lens


@app.cell(hide_code=True)
def prediction_hook(mo):
    mo.md(r"""
    ## 4 · Predicting an average is easier than predicting a person

    Move the **probe agility** in the widget above. The orange interval below expresses
    uncertainty about the population **mean** at that agility. The purple interval predicts
    **one new observation**, so it also includes residual variation:

    \[
    \operatorname{Var}(y_{new}\mid x_{new},D)
    =\operatorname{Var}(\alpha+\beta x_{new}\mid D)+\sigma^2.
    \]

    **Try this:** (1) probe agility 50, then 90; (2) increase σ;
    (3) use the 60-observation preset. More data can locate the mean accurately without
    making individual outcomes predictable. All intervals are conditional on this model.
    """)
    return


@app.cell
def prediction_readout(
    alt,
    bayes_fit,
    interval_level,
    mo,
    posterior_draws,
    posterior_lens,
    prediction_summary,
    records_chart,
):
    probe_agility = float(posterior_lens.value["agility"])
    _mean, _mean_half, _new_half = prediction_summary(bayes_fit, [(probe_agility - 50) / 10], interval_level.value / 100)
    _m, _h, _p = float(_mean[0]), float(_mean_half[0]), float(_new_half[0])
    _rows = [
        {"quantity": "Expected mean score", "mean": _m, "lower": _m-_h, "upper": _m+_h},
        {"quantity": "One new observation", "mean": _m, "lower": _m-_p, "upper": _m+_p},
    ]
    _base = records_chart(_rows).encode(y=alt.Y("quantity:N", title=None), color=alt.Color("quantity:N", scale=alt.Scale(domain=["Expected mean score", "One new observation"], range=["#e67722", "#9467bd"]), legend=None))
    _intervals = _base.mark_rule(strokeWidth=8).encode(x=alt.X("lower:Q", title="Fictional PTSD score"), x2="upper:Q", tooltip=["quantity:N", alt.Tooltip("lower:Q", format=".2f"), alt.Tooltip("upper:Q", format=".2f")])
    _centers = _base.mark_point(filled=True, size=130, color="white", stroke="black").encode(x="mean:Q")
    _draw = posterior_draws[int(posterior_lens.value["index"])]
    mo.vstack([
        mo.md(f"### At agility {probe_agility:g}: expected score **{_m:.1f}**"),
        (_intervals + _centers).properties(width="container", height=100),
        mo.md(f"**{interval_level.value}% mean credible interval:** [{_m-_h:.1f}, {_m+_h:.1f}]  \n**{interval_level.value}% posterior predictive interval:** [{_m-_p:.1f}, {_m+_p:.1f}]  \nSelected draw slope: **{_draw[1]:+.2f}**. The orange dot in the widget shows one plausible mean, not a random person's outcome."),
        mo.md("The Gaussian teaching model can predict scores outside 0–100. That is a model limitation to check, especially near boundaries; Bayesian inference does not fix a mismatched likelihood."),
    ])
    return


@app.cell(hide_code=True)
def model_check_hook(mo):
    mo.md(r"""
    ## 5 · A posterior can be precise and still fit badly
    *Video connection: visualize fit/misfit, compare fits, and evaluate prior choices.*

    The video's repeated workflow is **estimate → visualize → compare**. Start by
    asking whether your model can generate data resembling what you observe.

    Below, residuals are observed scores minus the posterior mean line. We also simulate
    1,000 replicated datasets from the posterior predictive distribution, at the same
    agilities, and compare their residual RMS with the observed one.

    **Try this:** select **Curved** in section 1. Can you see structure in the residuals?
    Then increase σ: a larger noise assumption can hide the RMS mismatch without removing
    the curve. A single diagnostic cannot certify a model.
    """)
    return


@app.cell
def predictive_check(
    alt,
    bayes_fit,
    mo,
    noise_sd,
    np,
    observations,
    records_chart,
):
    _residual = bayes_fit["y"] - bayes_fit["X"] @ bayes_fit["mean"]
    _rng = np.random.default_rng(1701)
    _coef = _rng.multivariate_normal(bayes_fit["mean"], bayes_fit["cov"], size=1000)
    _replicas = _coef @ bayes_fit["X"].T + _rng.normal(0, noise_sd.value, size=(1000, len(observations)))
    _rep_rms = np.sqrt(np.mean((_replicas - bayes_fit["X"] @ bayes_fit["mean"])**2, axis=1))
    _obs_rms = float(np.sqrt(np.mean(_residual**2)))
    _check_low, _check_high = np.quantile(_rep_rms, [0.025, 0.975])
    _residual_plot = records_chart([{"agility": float(x), "residual": float(r)} for x, r in zip(observations[:, 0], _residual)]).mark_circle(size=75, color="#168aad").encode(x=alt.X("agility:Q", title="Agility"), y=alt.Y("residual:Q", title="Residual around posterior mean line"), tooltip=["agility:Q", alt.Tooltip("residual:Q", format=".2f")])
    _zero = records_chart([{"zero": 0}]).mark_rule(color="#888888").encode(y="zero:Q")
    _hist = records_chart([{"rms": float(v)} for v in _rep_rms]).mark_bar(color="#9467bd", opacity=0.65).encode(x=alt.X("rms:Q", bin=alt.Bin(maxbins=35), title="Replicated-data RMS around posterior mean line"), y=alt.Y("count():Q", title="Replications"))
    _obs = records_chart([{"observed": _obs_rms}]).mark_rule(color="#e67722", strokeWidth=3).encode(x="observed:Q")
    mo.vstack([
        (_residual_plot + _zero).properties(width="container", height=190),
        (_hist + _obs).properties(width="container", height=170),
        mo.md(f"**Observed RMS = {_obs_rms:.2f}** (orange line). Central 95% of replicated RMS values: **[{_check_low:.2f}, {_check_high:.2f}]**. This is a descriptive posterior predictive check, not a frequentist p-value or a model-selection score. Look for both unusual size **and systematic residual patterns**."),
    ])
    return


@app.cell(hide_code=True)
def computation_bridge(mo):
    mo.md(r"""
    ## 6 · What changes when software uses MCMC?
    *Video connection: output, chains, warm-up, and R-hat (about 12:20–14:00).*

    We solved a special model exactly. More general models often need **Markov chain
    Monte Carlo (MCMC)** to approximate the posterior. MCMC produces dependent draws;
    more iterations are not more observations.

    | Term | Practical interpretation |
    |---|---|
    | Chain | One run of the sampler, usually with its own starting point |
    | Warm-up | Initial iterations used to adapt the algorithm and settle; excluded from reported posterior summaries |
    | R-hat | Compares within- and between-chain variation; modern rank-normalized split R-hat near 1 is reassuring, not proof |
    | Effective sample size (ESS) | How much information dependent draws contain relative to independent draws; inspect bulk and tail ESS |
    | Divergences | For Hamiltonian Monte Carlo, warning signs that parts of the posterior may not be explored correctly |

    In practice, investigate **R-hat above roughly 1.01**, low ESS, divergences, and poor
    trace plots. Good sampler diagnostics establish neither good model fit nor good priors.

    ### A visual preview—not an MCMC implementation
    Switch among these **illustrative traces**. “Sticky” follows a stationary AR(1)
    process; “Separated” deliberately offsets chains. We do not report an imitation R-hat.
    """)
    return


@app.cell
def trace_control(mo):
    trace_mode = mo.ui.dropdown(["Well mixed", "Sticky", "Separated"], value="Well mixed", label="Illustrative trace behavior")
    trace_mode
    return (trace_mode,)


@app.cell
def trace_chart(alt, mo, np, records_chart, slope_mean, slope_sd, trace_mode):
    _rng = np.random.default_rng(2027)
    _rho = 0.97 if trace_mode.value == "Sticky" else 0.0
    _rows = []
    for _chain in range(4):
        _z = float(_rng.normal())
        for _i in range(200):
            _z = _rho * _z + np.sqrt(1 - _rho**2) * float(_rng.normal())
            _offset = (_chain - 1.5) * 2.5 if trace_mode.value == "Separated" else 0
            _rows.append({"iteration": _i + 1, "slope": slope_mean + slope_sd * (_z + _offset), "chain": str(_chain + 1)})
    _message = {
        "Well mixed": "The traces overlap and move rapidly. That looks reassuring, but still inspect numerical diagnostics and the model itself.",
        "Sticky": "Adjacent draws are similar. Two hundred saved iterations contain much less information than 200 independent draws.",
        "Separated": "The chains occupy different regions. Pooling them does not magically create a trustworthy posterior.",
    }[trace_mode.value]
    mo.vstack([records_chart(_rows).mark_line(opacity=0.8).encode(x="iteration:Q", y=alt.Y("slope:Q", scale=alt.Scale(zero=False)), color=alt.Color("chain:N", legend=alt.Legend(orient="bottom"))).properties(width="container", height=220), mo.md(_message)])
    return


@app.cell(hide_code=True)
def real_sandbox_hook(mo):
    mo.md(r"""
    ## 7 · Sandbox: take the same workflow to real observations

    Now use R's classic **cars** dataset: 50 measurements of speed (mph) and stopping
    distance (feet), recorded in the 1920s. These are historical observations, not modern
    braking guidance. The values are embedded below so this experiment works offline.

    [Dataset documentation](https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/cars.html)
    · [CSV mirror](https://vincentarelbundock.github.io/Rdatasets/csv/datasets/cars.csv)
    · Source: Ezekiel (1930), *Methods of Correlation Analysis*.

    We model \(\mathrm{distance}=\alpha+\beta(\mathrm{speed}-15)+\epsilon\).
    Now **β is feet per additional mph**, and α is mean stopping distance at 15 mph.
    We use \(\alpha\sim\mathcal N(40,40^2)\) and your slope prior below; σ is again assumed known.

    **Challenges:**
    1. Make a negative prior conflict with the upward data trend. How strong must it be to matter?
    2. Compare 10 with 50 observations, keeping the prior and σ fixed.
    3. Predict at 25 mph, then at 35 mph. A model's tidy extrapolation is not new evidence.
    4. Inspect the residual plot. Would constant noise and a straight line deserve closer checking?

    The subset order is fixed and randomized, so increasing n retains earlier observations.
    """)
    return


@app.cell
def cars_data(np):
    # Exact values from R datasets::cars; speed in mph, stopping distance in ft.
    cars_data = np.array([
        [4,2],[4,10],[7,4],[7,22],[8,16],[9,10],[10,18],[10,26],[10,34],
        [11,17],[11,28],[12,14],[12,20],[12,24],[12,28],[13,26],[13,34],
        [13,34],[13,46],[14,26],[14,36],[14,60],[14,80],[15,20],[15,26],
        [15,54],[16,32],[16,40],[17,32],[17,40],[17,50],[18,42],[18,56],
        [18,76],[18,84],[19,36],[19,46],[19,68],[20,32],[20,48],[20,52],
        [20,56],[20,64],[22,66],[23,54],[24,70],[24,92],[24,93],[24,120],[25,85],
    ], dtype=float)
    cars_order = np.random.default_rng(73).permutation(len(cars_data))
    return cars_data, cars_order


@app.cell
def sandbox_controls(mo):
    cars_n = mo.ui.slider(10, 50, step=5, value=50, label="Observations n", show_value=True)
    cars_prior = mo.ui.slider(-8, 8, step=0.25, value=0, label="Slope prior center (ft/mph)", show_value=True)
    cars_prior_sd = mo.ui.slider(0.1, 10, step=0.1, value=5, label="Slope prior SD", show_value=True)
    cars_sigma = mo.ui.slider(5, 35, step=1, value=15, label="Assumed known residual SD (ft)", show_value=True)
    cars_speed = mo.ui.slider(4, 35, step=1, value=20, label="Predict at speed (mph)", show_value=True)
    mo.vstack([mo.hstack([cars_n, cars_prior, cars_prior_sd], widths="equal"), mo.hstack([cars_sigma, cars_speed], widths="equal")])
    return cars_n, cars_prior, cars_prior_sd, cars_sigma, cars_speed


@app.cell
def sandbox_output(
    alt,
    cars_data,
    cars_n,
    cars_order,
    cars_prior,
    cars_prior_sd,
    cars_sigma,
    cars_speed,
    fit_regression,
    mo,
    norm,
    np,
    prediction_summary,
    records_chart,
):
    _selected = cars_data[cars_order[:cars_n.value]]
    cars_fit = fit_regression(np.column_stack([_selected[:, 0] - 15, _selected[:, 1]]), cars_sigma.value, cars_prior.value, cars_prior_sd.value, intercept_mean=40, intercept_sd=40)
    _xs = np.linspace(4, 35, 100)
    _m, _h, _p = prediction_summary(cars_fit, _xs - 15)
    _rows = [{"speed": float(x), "mean": float(m), "low": float(m-h), "high": float(m+h), "plow": float(m-p), "phigh": float(m+p)} for x, m, h, p in zip(_xs, _m, _h, _p)]
    _base = records_chart(_rows).encode(x=alt.X("speed:Q", title="Speed (mph)"))
    _predictive = _base.mark_area(color="#9467bd", opacity=0.12).encode(y=alt.Y("plow:Q", title="Stopping distance (ft)"), y2="phigh:Q")
    _credible = _base.mark_area(color="#e67722", opacity=0.3).encode(y="low:Q", y2="high:Q")
    _line = _base.mark_line(color="#e67722", strokeWidth=3).encode(y="mean:Q")
    _points = records_chart([{"speed": float(x), "distance": float(y)} for x, y in _selected]).mark_circle(color="#168aad", size=65).encode(x="speed:Q", y="distance:Q", tooltip=["speed:Q", "distance:Q"])
    _probe = records_chart([{"speed": cars_speed.value}]).mark_rule(strokeDash=[4, 4], color="#888888").encode(x="speed:Q")
    _pm, _ph, _pp = prediction_summary(cars_fit, [cars_speed.value - 15])
    _slope_sd = np.sqrt(cars_fit["cov"][1, 1])
    _res = cars_fit["y"] - cars_fit["X"] @ cars_fit["mean"]
    _rp = records_chart([{"speed": float(x), "residual": float(r)} for x, r in zip(_selected[:, 0], _res)]).mark_circle(color="#168aad", size=60).encode(x="speed:Q", y=alt.Y("residual:Q", title="Residual (ft)"))
    _zero = records_chart([{"zero": 0}]).mark_rule(color="#888888").encode(y="zero:Q")
    _extra = "**Extrapolation:** this speed lies outside the selected observations." if cars_speed.value < _selected[:, 0].min() or cars_speed.value > _selected[:, 0].max() else "This speed lies within the selected observations' range."
    mo.vstack([
        mo.hstack([mo.stat(label="OLS slope (ft/mph)", value=f"{cars_fit['ols'][1]:.2f}"), mo.stat(label="Posterior slope (ft/mph)", value=f"{cars_fit['mean'][1]:.2f}"), mo.stat(label="P(slope > 0 | data, model)", value=f"{norm.sf(0, cars_fit['mean'][1], _slope_sd):.1%}")]),
        (_predictive + _credible + _line + _points + _probe).properties(width="container", height=290),
        mo.md(f"At **{cars_speed.value} mph**, expected distance = **{_pm[0]:.1f} ft**.  \n95% mean credible interval: **[{_pm[0]-_ph[0]:.1f}, {_pm[0]+_ph[0]:.1f}] ft** (orange).  \n95% new-observation interval: **[{_pm[0]-_pp[0]:.1f}, {_pm[0]+_pp[0]:.1f}] ft** (purple).  \n{_extra}"),
        (_rp + _zero).properties(width="container", height=160),
        mo.md("Negative predicted distances and changing residual spread are useful clues that this simple Gaussian model may need improvement. A confidence or credible interval does not include uncertainty from every possible model misspecification."),
    ])
    return


@app.cell(hide_code=True)
def quiz_hook(mo):
    mo.md("""
    ## 8 · Check your intuition

    Make your predictions, then reveal feedback. The numerical question is tied to your
    current Avengers experiment: **what is the posterior probability that its slope is negative?**
    Try reasoning from the slope-density plot before looking back at its readout.
    """)
    return


@app.cell
def quiz_controls(mo):
    quiz_answers = mo.ui.dictionary({
        "interval": mo.ui.dropdown(["Choose…", "95% posterior probability lies in this interval, conditional on model and data", "95% of individual observations lie in the slope interval", "The model is correct with 95% probability"], value="Choose…", label="What does a 95% slope credible interval mean?"),
        "prior": mo.ui.dropdown(["Choose…", "Usually becomes less influential as informative data accumulate", "Disappears exactly after 30 observations", "Must always dominate the likelihood"], value="Choose…", label="With a fixed prior, what usually happens as informative observations accumulate?"),
        "prediction": mo.ui.dropdown(["Choose…", "One new observation", "The expected mean", "Always the same width"], value="Choose…", label="Under our model, which interval is wider at the same x and level?"),
        "mcmc": mo.ui.dropdown(["Choose…", "Reduce Monte Carlo error if sampling is working", "Provide more observed subjects", "Guarantee the model fits the data"], value="Choose…", label="What can more MCMC iterations do?"),
        "probability": mo.ui.slider(0, 100, step=1, value=50, label="Your guess: P(β < 0 | data, model), %", show_value=True),
    })
    quiz_reveal = mo.ui.switch(value=False, label="Reveal answers and explanations")
    mo.vstack([quiz_answers, quiz_reveal])
    return quiz_answers, quiz_reveal


@app.cell
def quiz_feedback(mo, negative_probability, quiz_answers, quiz_reveal):
    if not quiz_reveal.value:
        _feedback = mo.md("_Make your choices, then turn on Reveal._")
    else:
        _keys = {
            "interval": ("95% posterior probability lies in this interval, conditional on model and data", "A credible interval is a posterior statement about the parameter, conditional on assumptions."),
            "prior": ("Usually becomes less influential as informative data accumulate", "Information, not a magic sample size, determines the relative influence of likelihood and prior."),
            "prediction": ("One new observation", "Predictive variance adds σ² to uncertainty about the mean."),
            "mcmc": ("Reduce Monte Carlo error if sampling is working", "Computation can improve approximation; it cannot create new observations or fix a wrong likelihood."),
        }
        _lines, _score = [], 0
        for _key, (_correct, _why) in _keys.items():
            _ok = quiz_answers.value[_key] == _correct
            _score += int(_ok)
            _lines.append(f"- {'✅' if _ok else '↩️'} **{_key.title()}:** {_why}")
        _truth = 100 * negative_probability
        _error = abs(quiz_answers.value["probability"] - _truth)
        _score += int(_error <= 5)
        _lines.append(f"- {'✅' if _error <= 5 else '↩️'} **Probability:** {_truth:.1f}%. Your guess is {_error:.1f} percentage points away (within 5 earns the point).")
        _feedback = mo.md(f"### {_score}/5\n\n" + "\n".join(_lines) + "\n\nChange the experiments and try again. Understanding the explanation matters more than the score.")
    _feedback
    return


@app.cell(hide_code=True)
def recap_and_source(mo):
    mo.md(r"""
    ## Your new mental model

    | Familiar idea | Bayesian addition | Remember |
    |---|---|---|
    | Regression line | Distribution over possible lines | Same α, β, predictors, and response |
    | Coefficient estimate | Posterior mean, interval, and directional probability | Interpret conditional on prior, likelihood, and observed data |
    | Modeling assumptions | Explicit priors as well as a likelihood | Strong assumptions can make a wrong answer precise |
    | Prediction | Posterior predictive distribution | Include parameter uncertainty **and** residual variation |
    | Residual checks | Posterior predictive checks | Good computation does not guarantee a good model |
    | More complicated models | Different likelihoods, links, or group effects | Keep the estimate → visualize → compare workflow |

    ### Connecting this back to the video's R syntax

    ```r
    # Schematic examples; use your own avengers data frame.
    classical <- lm(PTSD ~ agility, data = avengers)
    bayesian <- brms::brm(
      PTSD ~ agility, data = avengers,
      prior = brms::prior(normal(0, 1), class = "b", coef = "agility")
    )
    ```

    The R slope above is **per one agility point**. Our teaching slope is per **ten**:
    divide our slope prior mean and SD by 10 when translating to that raw predictor.
    The example R prior is illustrative, not a recommendation. `brm()` also estimates
    residual σ and has other priors; it is **not the identical model** used here.

    The video describes consistent `flexplot` functions such as `estimates`, `visualize`,
    `compare.fits`, and `model.comparison`, and says its Bayesian integration was not
    published at recording time. We follow that workflow without assuming those integrations
    are available in your installed R packages.

    ### Where to go next

    1. Estimate σ rather than treating it as known, using a general Bayesian modeling package.
    2. Check prior predictive simulations before fitting real data.
    3. Compare predictive performance using held-out data or suitable cross-validation;
       visual similarity alone is not model selection. We did **not** compute Bayes factors here.
    4. Add a predictor, a group-level effect, or a likelihood/link for a binary or count response.
    5. When using MCMC, inspect convergence, ESS, and sampler warnings before interpreting results.

    **Source map:** [0:00 course motivation](https://www.youtube.com/watch?v=ME-8sm7pZbM&t=0s)
    · [2:44 applied-first teaching](https://www.youtube.com/watch?v=ME-8sm7pZbM&t=164s)
    · [5:55 build incrementally](https://www.youtube.com/watch?v=ME-8sm7pZbM&t=355s)
    · [8:23 consistent tools](https://www.youtube.com/watch?v=ME-8sm7pZbM&t=503s)
    · [11:01 classical vs Bayesian](https://www.youtube.com/watch?v=ME-8sm7pZbM&t=661s)
    · [14:00 next steps](https://www.youtube.com/watch?v=ME-8sm7pZbM&t=840s).

    **Provenance:** the teaching philosophy, Avengers terminology, and R-tool discussion
    come from the video. Numerical data for the Avengers experiments, interactive widgets,
    exact-posterior model, diagnostics demonstrations, and quizzes are companion material;
    the cars observations come from the separately linked dataset.
    """)
    return


if __name__ == "__main__":
    app.run()

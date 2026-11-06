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
    # Rapid deduplication & fuzzy matching with the Fellegi-Sunter model
    *An interactive companion to **Robin Linacre's** PyData talk on [Splink](https://github.com/moj-analytical-services/splink).*

    Three messy records all say *Lucas Smith*, but there's no person ID that ties them together. Below you'll build, from scratch in numpy and pandas, the same pipeline Splink uses:
    **fuzzy matching → term frequencies → m/u probabilities → blocking → Fellegi-Sunter scoring (waterfall) → clustering**.
    """),
        mo.Html('<iframe width="560" height="315" src="https://www.youtube.com/embed/eQtFkI8f02U" title="Robin Linacre - Splink" frameborder="0" allowfullscreen></iframe>'),
        mo.md(r"""
    **Colour language used throughout**

    <span style="color:#2a9d8f">■ teal = evidence *for* a match</span> (positive match weight) ·
    <span style="color:#e4572e">■ coral = evidence *against* a match</span> (negative weight) ·
    <span style="color:#6c5b7b">■ purple = the prior / final score</span> ·
    <span style="color:#f18f01">■ orange = the thing you're currently looking at</span>
    """),
    ])
    return


@app.cell(hide_code=True)
def s1_intro(mo):
    mo.md(r"""
    ## 1 · The problem: no unique ID

    Records about the **same entity** get collected without a unique identifier: within one dataset (*deduplication*) or across several (*linkage*). Robin's point is that both are the same problem: stack the tables and dedupe.

    The plan, at a very high level:

    1. **Pairwise comparisons**: compare record *i* with record *j* and predict a **match probability**.
    2. These predictions form a **graph**: records are nodes, probable matches are edges.
    3. **Connected components** of that graph are **clusters**, and each one gets a new `person_id`.

    Below is a synthetic dataset built for this notebook, in the spirit of Splink's "historical people" demo: about 1,200 true people, each with 1–5 records corrupted by typos, nicknames, missing values, a 7↔1 date slip, and a data-quality landmine (`Sir Baronet`). The hidden `true_id` column is the ground truth; the model never sees it.
    """)
    return


@app.cell
def string_functions():
    def levenshtein(a, b):
        """Edit distance: min # of single-character inserts, deletes and substitutions."""
        if a is None or b is None:
            return None
        _prev = list(range(len(b) + 1))
        for _i, _ca in enumerate(a, 1):
            _cur = [_i]
            for _j, _cb in enumerate(b, 1):
                _cur.append(min(_prev[_j] + 1, _cur[_j - 1] + 1, _prev[_j - 1] + (_ca != _cb)))
            _prev = _cur
        return _prev[-1]


    def jaro_winkler(a, b, p=0.1):
        """Jaro-Winkler similarity in [0, 1]; rewards a shared prefix (up to 4 chars)."""
        if a is None or b is None:
            return None
        if a == b:
            return 1.0
        _la, _lb = len(a), len(b)
        if _la == 0 or _lb == 0:
            return 0.0
        _win = max(max(_la, _lb) // 2 - 1, 0)
        _ma, _mb = [False] * _la, [False] * _lb
        _m = 0
        for _i in range(_la):
            for _j in range(max(0, _i - _win), min(_i + _win + 1, _lb)):
                if not _mb[_j] and a[_i] == b[_j]:
                    _ma[_i] = _mb[_j] = True
                    _m += 1
                    break
        if _m == 0:
            return 0.0
        _sa = [a[_i] for _i in range(_la) if _ma[_i]]
        _sb = [b[_j] for _j in range(_lb) if _mb[_j]]
        _t = sum(_x != _y for _x, _y in zip(_sa, _sb)) / 2
        _jaro = (_m / _la + _m / _lb + (_m - _t) / _m) / 3
        _pre = 0
        for _x, _y in zip(a[:4], b[:4]):
            if _x != _y:
                break
            _pre += 1
        return _jaro + _pre * p * (1 - _jaro)

    return jaro_winkler, levenshtein


@app.cell
def people_data(np, pd):
    def make_people(n_people=1200, seed=7):
        """Generate a messy, duplicated 'people' table with a hidden true_id."""
        _rng = np.random.default_rng(seed)
        _first = ["william", "john", "james", "george", "thomas", "charles", "henry", "robert", "edward", "richard",
                  "mary", "elizabeth", "anne", "margaret", "jane", "sarah", "catherine", "alice", "emily", "lucas",
                  "frederick", "arthur", "samuel", "joseph", "harriet", "florence", "isaac", "hugh", "rupert", "agnes",
                  "percival", "ottoline", "ignatius", "winifred", "barnaby", "cornelius", "evangeline", "lancelot",
                  "philippa", "ambrose", "tobias", "rosalind", "jasper", "octavia", "humphrey", "beatrix", "clement", "paul"]
        _sur = ["smith", "jones", "williams", "brown", "taylor", "davies", "wilson", "evans", "thomas", "johnson",
                "roberts", "walker", "wright", "robinson", "thompson", "white", "hughes", "edwards", "green", "hall",
                "wood", "harris", "lewis", "martin", "jackson", "clarke", "turner", "hill", "cooper", "ward",
                "linacre", "fairweather", "quilliam", "ravenscroft", "pemberton", "ashdown", "blackwood", "carrington",
                "delacroix", "everingham", "fitzwilliam", "gainsborough", "holloway", "inglethorpe", "kingsley",
                "loxley", "montague", "northcott", "chudley", "osbourne", "prendergast", "radcliffe", "stanhope"]
        _cities = ["london", "manchester", "birmingham", "leeds", "bristol", "liverpool", "york", "bath", "oxford",
                   "cambridge", "exeter", "norwich", "durham", "devon", "plymouth", "brighton", "leicester", "hull",
                   "carlisle", "lincoln", "chester", "derby", "truro", "ely", "wells"]
        _occ = ["politician", "writer", "soldier", "clergy", "artist", "sportsperson", "scientist", "musician",
                "lawyer", "actor", "engineer", "physician"]
        _nick = {"william": "bill", "robert": "bob", "richard": "dick", "elizabeth": "beth", "margaret": "maggie",
                 "edward": "ted", "thomas": "tom", "james": "jim", "catherine": "kate", "samuel": "sam",
                 "frederick": "fred", "joseph": "joe", "henry": "harry", "charles": "charlie"}
        _zipf = lambda n, s: (lambda w: w / w.sum())(1 / np.arange(1, n + 1) ** s)
        _pf, _ps, _pc, _po = _zipf(len(_first), 1.1), _zipf(len(_sur), 1.0), _zipf(len(_cities), 0.9), _zipf(len(_occ), 0.8)
        _letters = "abcdefghijklmnopqrstuvwxyz"

        def _typo(s):
            if len(s) < 3:
                return s
            _k, _op = _rng.integers(1, len(s) - 1), _rng.integers(4)
            if _op == 0:
                return s[:_k] + _rng.choice(list(_letters)) + s[_k + 1:]
            if _op == 1:
                return s[:_k] + s[_k + 1:]
            if _op == 2:
                return s[:_k] + _rng.choice(list(_letters)) + s[_k:]
            return s[:_k - 1] + s[_k] + s[_k - 1] + s[_k + 1:]

        _rows = []
        for _pid in range(n_people):
            _fn, _sn = _rng.choice(_first, p=_pf), _rng.choice(_sur, p=_ps)
            _y, _m, _d = _rng.integers(1700, 1950), _rng.integers(1, 13), _rng.integers(1, 29)
            _dob = f"{_y}-{_m:02d}-{_d:02d}"
            _city, _oc = _rng.choice(_cities, p=_pc), _rng.choice(_occ, p=_po)
            for _r in range(1 + min(_rng.poisson(1.2), 4)):
                _f, _s, _b, _c, _o = _fn, _sn, _dob, _city, _oc
                if _r > 0:  # corrupt the copies
                    _u = _rng.random(9)
                    if _u[0] < 0.15: _f = _typo(_f)
                    if _u[1] < 0.10 and _fn in _nick: _f = _nick[_fn]
                    if _u[2] < 0.15: _s = _typo(_s)
                    if _u[3] < 0.15:  # a handwritten 7<->1, or another single-digit slip
                        _b = _b.replace("7", "1", 1) if "7" in _b[5:] and _rng.random() < .5 else _b[:-1] + str(_rng.integers(10))
                    if _u[4] < 0.05: _b = None
                    if _u[5] < 0.15: _c = _rng.choice(_cities, p=_pc)
                    if _u[6] < 0.15: _c = None
                    if _u[7] < 0.10: _o = None
                    elif _u[7] < 0.16: _o = _rng.choice(_occ, p=_po)
                    if _u[8] < 0.03: _f = None
                _rows.append((_pid, _f, _s, _b, _c, _o))
        _df = pd.DataFrame(_rows, columns=["true_id", "first_name", "surname", "dob", "city", "occupation"])
        # data-quality landmine: some scraped titles ended up in the name fields
        _bad = _rng.random(len(_df)) < 0.05
        _df.loc[_bad, "first_name"], _df.loc[_bad, "surname"] = "sir", "baronet"
        _df = _df.sample(frac=1, random_state=seed).reset_index(drop=True)
        _df.insert(0, "unique_id", np.arange(len(_df)))
        return _df.astype(object).where(_df.notna(), None)


    people = make_people()
    COLS = ["first_name", "surname", "dob", "city", "occupation"]
    people
    return COLS, people


@app.cell
def data_summary(mo, people):
    _n = len(people)
    _sizes = people.groupby("true_id").size()
    n_true_pairs = int((_sizes * (_sizes - 1) // 2).sum())
    n_all_pairs = _n * (_n - 1) // 2
    mo.hstack([
        mo.stat(f"{_n:,}", label="records"),
        mo.stat(f"{people.true_id.nunique():,}", label="true people (hidden)"),
        mo.stat(f"{n_all_pairs:,}", label="all pairwise comparisons  n(n−1)/2"),
        mo.stat(f"{n_true_pairs:,}", label="true matching pairs"),
        mo.stat(f"1 in {n_all_pairs / n_true_pairs:,.0f}", label="prior: random pair is a match"),
    ], justify="space-around")
    return n_all_pairs, n_true_pairs


@app.cell(hide_code=True)
def s2_intro(mo):
    mo.md(r"""
    ## 2 · Information source #1: fuzzy matching

    Functions like **Levenshtein** and **Jaro-Winkler** look at *one column of one pair* and return a distance or a similarity.

    * **Levenshtein**: how many single-character edits turn `a` into `b`? `smith → smyth` is **1**.
      It's a dynamic program: $D[i,j] = \min\big(D[i-1,j]+1,\; D[i,j-1]+1,\; D[i-1,j-1] + [a_i \ne b_j]\big)$
    * **Jaro-Winkler**: a similarity on a 0–1 scale that counts matching characters within a window, penalises transpositions, and gives a bonus for a shared prefix (names are rarely mistyped at the start).

    **Step through the Levenshtein table.** Each cell is the cheapest way to turn a prefix of `a` into a prefix of `b`. The orange cell is being filled, and its three neighbours (delete ↓, insert →, substitute ↘) are outlined.
    """)
    return


@app.cell
def lev_widget_class(anywidget, traitlets):
    class LevenshteinStepper(anywidget.AnyWidget):
        """Step through the Levenshtein DP table cell by cell (JS computes the table; Python checks it)."""
        _esm = r"""
    function render({ model, el }) {
      const ctl = document.createElement("div"), view = document.createElement("div"), cap = document.createElement("div");
      ctl.style.cssText = "display:flex;gap:6px;margin-bottom:6px;align-items:center";
      cap.style.cssText = "margin-top:6px;font-family:sans-serif;font-size:14px;min-height:2.6em";
      el.append(ctl, view, cap);
      let timer = null;
      const table = () => {
        const a = model.get("a"), b = model.get("b");
        const D = Array.from({ length: a.length + 1 }, (_, i) => Array.from({ length: b.length + 1 }, (_, j) => i === 0 ? j : (j === 0 ? i : 0)));
        for (let i = 1; i <= a.length; i++) for (let j = 1; j <= b.length; j++)
          D[i][j] = Math.min(D[i-1][j] + 1, D[i][j-1] + 1, D[i-1][j-1] + (a[i-1] !== b[j-1] ? 1 : 0));
        return D;
      };
      const total = () => model.get("a").length * model.get("b").length;
      const go = d => { model.set("step", Math.max(0, Math.min(total(), model.get("step") + d))); model.save_changes(); };
      const btn = (t, f) => { const x = document.createElement("button"); x.textContent = t; x.onclick = f; ctl.appendChild(x); return x; };
      btn("⏮", () => go(-1e9)); btn("◀", () => go(-1)); btn("▶", () => go(1)); btn("⏭", () => go(1e9));
      const play = btn("▶ play", () => {
        if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; model.save_changes(); return; }
        if (model.get("step") >= total()) model.set("step", 0);
        play.textContent = "⏸ pause";
        timer = setInterval(() => {
          const s = model.get("step") + 1;
          if (s > total()) { clearInterval(timer); timer = null; play.textContent = "▶ play"; model.save_changes(); return; }
          model.set("step", s);  // local only while animating
        }, 220);
      });
      function draw() {
        const a = model.get("a"), b = model.get("b"), D = table(), s = Math.min(model.get("step"), total());
        const ci = s === 0 ? -1 : Math.floor((s - 1) / b.length) + 1, cj = s === 0 ? -1 : ((s - 1) % b.length) + 1;
        const filled = (i, j) => i === 0 || j === 0 || ((i - 1) * b.length + j) <= s;
        const cell = (txt, st) => `<td style="width:30px;height:30px;text-align:center;border-radius:5px;${st}">${txt}</td>`;
        let h = `<table style="border-collapse:separate;border-spacing:3px;font-family:monospace;font-size:15px"><tr>${cell("")}${cell("ε","opacity:.5")}`;
        for (const c of b) h += cell(`<b>${c}</b>`);
        h += "</tr>";
        for (let i = 0; i <= a.length; i++) {
          h += "<tr>" + cell(i === 0 ? '<span style="opacity:.5">ε</span>' : `<b>${a[i-1]}</b>`);
          for (let j = 0; j <= b.length; j++) {
            const cur = i === ci && j === cj;
            const nb = (i === ci - 1 && j === cj) || (i === ci && j === cj - 1) || (i === ci - 1 && j === cj - 1);
            const last = s === total() && i === a.length && j === b.length;
            let st = "background:rgba(108,91,123,.12);";
            if (cur) st = "background:#f18f01;color:white;font-weight:bold;";
            else if (last) st = "background:#6c5b7b;color:white;font-weight:bold;";
            if (nb) st += "outline:2px solid #f18f01;";
            h += cell(filled(i, j) ? D[i][j] : "", st);
          }
          h += "</tr>";
        }
        view.innerHTML = h + "</table>";
        if (s === 0) cap.innerHTML = "Row 0 / column 0 are the base cases: turning a prefix into the empty string ε costs its length. Press ▶.";
        else {
          const sub = D[ci-1][cj-1] + (a[ci-1] !== b[cj-1] ? 1 : 0), del = D[ci-1][cj] + 1, ins = D[ci][cj-1] + 1;
          const best = Math.min(sub, del, ins);
          const why = sub === best ? (a[ci-1] === b[cj-1] ? `'${a[ci-1]}' = '${b[cj-1]}' → copy diagonal for free` : `substitute '${a[ci-1]}'→'${b[cj-1]}' (+1)`) : (del === best ? `delete '${a[ci-1]}' (+1 from above)` : `insert '${b[cj-1]}' (+1 from left)`);
          cap.innerHTML = `<b>step ${s}/${total()}</b> · D[${ci},${cj}] = min(↘ ${sub}, ↓ ${del}, → ${ins}) = <b>${best}</b>: ${why}` +
            (s === total() ? `<br><b style="color:#6c5b7b">Levenshtein distance = ${D[a.length][b.length]}</b>` : "");
        }
      }
      model.on("change:step", draw); model.on("change:a", draw); model.on("change:b", draw); draw();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        a = traitlets.Unicode("smith").tag(sync=True)
        b = traitlets.Unicode("smyth").tag(sync=True)
        step = traitlets.Int(0).tag(sync=True)

    return (LevenshteinStepper,)


@app.cell
def lev_inputs(mo):
    lev_a = mo.ui.text(value="linacre", label="a")
    lev_b = mo.ui.text(value="linaker", label="b")
    mo.hstack([lev_a, lev_b], justify="start")
    return lev_a, lev_b


@app.cell
def lev_stepper_view(LevenshteinStepper, lev_a, lev_b, mo):
    lev_stepper = mo.ui.anywidget(LevenshteinStepper(a=lev_a.value.lower()[:12], b=lev_b.value.lower()[:12], step=0))
    lev_stepper
    return (lev_stepper,)


@app.cell
def lev_readout(jaro_winkler, lev_a, lev_b, lev_stepper, levenshtein, mo, pd):
    _a, _b = lev_a.value.lower()[:12], lev_b.value.lower()[:12]
    _examples = [("smith", "smyth"), ("linacre", "linaker"), ("william", "bill"), ("martha", "marhta"),
                 ("catherine", "katherine"), ("john", "jon"), ("1857-07-12", "1857-01-12"), (_a, _b)]
    _tbl = pd.DataFrame([{"a": _x, "b": _y, "levenshtein": levenshtein(_x, _y), "jaro_winkler": round(jaro_winkler(_x, _y), 3)}
                         for _x, _y in _examples])
    mo.vstack([
        mo.hstack([
            mo.stat(levenshtein(_a, _b), label=f"Levenshtein('{_a}', '{_b}')  (Python)"),
            mo.stat(f"{jaro_winkler(_a, _b):.3f}", label="Jaro-Winkler similarity"),
            mo.stat(f"{lev_stepper.value['step']}", label="stepper position"),
        ], justify="start"),
        mo.ui.table(_tbl, selection=None, label="Same functions, classic examples (last row = yours)"),
    ])
    return


@app.cell(hide_code=True)
def try2(mo):
    mo.callout(mo.md(r"""
    **Try this**

    * `william` vs `bill`: Levenshtein is 4 and Jaro-Winkler is low, yet it's the same person. Edit distance can't see **nicknames**, so you'd need a custom comparison level for that.
    * `martha` vs `marhta`: one transposition costs **2** edits but Jaro-Winkler still says 0.96.
    * The weakness Robin points out: these functions score a typo in **Smith** exactly like a typo in **Linacre**. They only look at the two values *in isolation*. Section 3 fixes that.
    """), kind="info")
    return


@app.cell(hide_code=True)
def s3_intro(mo):
    mo.md(r"""
    ## 3 · Information source #2: term frequencies (coincidences)

    A match on **Smith** is weaker evidence than a match on **Linacre**, because two *different* people share the surname Smith by coincidence far more often.

    Fellegi-Sunter (section 6) scores agreement with a **Bayes factor** $m/u$:

    * $m$ = P(agree | records are a **match**)
    * $u$ = P(agree | records are a **non-match**), i.e. the chance of a coincidence

    For an exact match on a specific value $v$, the coincidence rate is roughly the value's **term frequency** $f_v$, so the **term-frequency-adjusted** match weight is

    $$w_v = \log_2 \frac{m}{f_v} \qquad\text{vs. the average}\qquad \bar w = \log_2\frac{m}{\sum_v f_v^2}$$

    Pick a column and a value. The orange bar is your value, and the right-hand chart shows how much an exact match on it is worth.
    """)
    return


@app.cell
def tf_col_ui(mo):
    tf_col = mo.ui.dropdown(["surname", "first_name", "city", "occupation"], value="surname", label="column")
    tf_col
    return (tf_col,)


@app.cell
def tf_val_ui(mo, people, tf_col):
    _vals = people[tf_col.value].dropna().value_counts().index.tolist()
    tf_val = mo.ui.dropdown(_vals, value=("linacre" if "linacre" in _vals else _vals[-1]), label="value", searchable=True)
    tf_m = mo.ui.slider(0.5, 1.0, step=0.01, value=0.9, label="m (P agree | match)", show_value=True)
    mo.hstack([tf_val, tf_m], justify="start")
    return tf_m, tf_val


@app.cell
def tf_chart(alt, mo, np, pd, people, tf_col, tf_m, tf_val):
    _vc = people[tf_col.value].dropna().value_counts(normalize=True)
    _tf = pd.DataFrame({"value": _vc.index, "f": _vc.values})
    _tf["weight"] = np.log2(tf_m.value / _tf["f"])
    _avg_u = float((_tf["f"] ** 2).sum())
    _avg_w = np.log2(tf_m.value / _avg_u)
    _tf["selected"] = np.where(_tf["value"] == tf_val.value, "selected", "other")
    _show = pd.concat([_tf.head(25), _tf[_tf.value == tf_val.value]]).drop_duplicates("value")
    _col = alt.Color("selected:N", scale=alt.Scale(domain=["selected", "other"], range=["#f18f01", "#6c5b7b"]), legend=None)
    _freq = alt.Chart(_show).mark_bar().encode(
        x=alt.X("value:N", sort="-y", title=None), y=alt.Y("f:Q", title="term frequency f_v"), color=_col,
        tooltip=["value", alt.Tooltip("f:Q", format=".4f")]).properties(width=380, height=240, title=f"Skew in '{tf_col.value}' (top 25)")
    _wt = alt.Chart(_show).mark_bar().encode(
        x=alt.X("value:N", sort="y", title=None), y=alt.Y("weight:Q", title="match weight log2(m / f_v)"),
        color=_col, tooltip=["value", alt.Tooltip("weight:Q", format=".2f")]).properties(width=380, height=240, title="Exact-match weight, TF-adjusted")
    _rule = alt.Chart(pd.DataFrame({"y": [_avg_w]})).mark_rule(color="#2a9d8f", strokeDash=[6, 3], size=2).encode(y="y:Q")
    _f_sel = float(_tf.loc[_tf.value == tf_val.value, "f"].iloc[0])
    tf_weight_selected = float(np.log2(tf_m.value / _f_sel))
    mo.vstack([
        alt.hconcat(_freq, _wt + _rule),
        mo.hstack([
            mo.stat(f"{_f_sel:.4f}", label=f"f('{tf_val.value}')"),
            mo.stat(f"{tf_weight_selected:+.2f}", label="TF-adjusted weight (bits)"),
            mo.stat(f"{_avg_w:+.2f}", label="average weight, no TF (teal dashed line)"),
            mo.stat(f"{2 ** tf_weight_selected:,.0f}×", label="Bayes factor m/f_v"),
        ], justify="space-around"),
    ])
    return


@app.cell(hide_code=True)
def try3(mo):
    mo.callout(mo.md(r"""
    **Try this**

    * Compare `surname = smith` with `surname = linacre`. Linacre's exact match should be worth several more **bits** of evidence.
    * Switch to `first_name` and pick `sir`. That's the data-quality landmine: an exact match on "sir" is nearly worthless (and section 5 shows it can blow up your blocking).
    * `occupation = politician` is a common value, which is why Robin's waterfall showed a **negative TF adjustment** for it.
    """), kind="info")
    return


@app.cell(hide_code=True)
def s4_intro(mo):
    mo.md(r"""
    ## 4 · Information source #3: data quality, the *m* probabilities

    How often do values **differ among true matches**? Robin's example is two dates of birth for Lucas that differ by one character, a **7** vs a **1** (July vs January).

    * If the data was **typed from handwritten records**, that slip happens all the time. A 1-character difference is then *positive* evidence for a match.
    * If the dates came **from driving licences**, they're almost never wrong. The same difference is then *strong evidence against*.

    Each comparison level $\ell$ gets its own weight $w_\ell = \log_2 \dfrac{m_\ell}{u_\ell}$. A full disagreement gets $\log_2 \dfrac{1-m}{1-u}$, which is negative, and that's information that is often thrown away.

    First, the general relationship: drag $m$ and $u$.
    """)
    return


@app.cell
def mu_sliders(mo):
    mu_m = mo.ui.slider(0.01, 0.999, step=0.001, value=0.95, label="m = P(agree | match)", show_value=True)
    mu_u = mo.ui.slider(0.0005, 0.5, step=0.0005, value=0.01, label="u = P(agree | non-match)", show_value=True)
    mo.hstack([mu_m, mu_u], justify="start")
    return mu_m, mu_u


@app.cell
def mu_chart(alt, mo, mu_m, mu_u, np, pd):
    _m, _u = mu_m.value, mu_u.value
    _wa, _wd = np.log2(_m / _u), np.log2((1 - _m) / (1 - _u))
    _d = pd.DataFrame({"outcome": ["agree", "disagree"], "weight": [_wa, _wd]})
    _d["sign"] = np.where(_d.weight >= 0, "for", "against")
    _bars = alt.Chart(_d).mark_bar(size=60).encode(
        x=alt.X("outcome:N", title=None), y=alt.Y("weight:Q", title="match weight (bits)", scale=alt.Scale(domain=[-12, 12])),
        color=alt.Color("sign:N", scale=alt.Scale(domain=["for", "against"], range=["#2a9d8f", "#e4572e"]), legend=None),
        tooltip=[alt.Tooltip("weight:Q", format="+.2f")])
    _txt = _bars.mark_text(dy=-10, fontSize=14, fontWeight="bold").encode(text=alt.Text("weight:Q", format="+.2f"), color=alt.value("#444"))
    mo.vstack([
        (_bars + _txt + alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#888").encode(y="y:Q")).properties(width="container", height=240),
        mo.md(f"Agreeing multiplies the odds by **{_m/_u:,.1f}×**; disagreeing multiplies them by **{(1-_m)/(1-_u):.3g}×**."),
    ])
    return


@app.cell
def comparison_levels(COLS, jaro_winkler, levenshtein, np, pd, people):
    LEVELS = {
        "first_name": ["exact", "Jaro-Winkler ≥ 0.92", "Jaro-Winkler ≥ 0.80", "else"],
        "surname":    ["exact", "Jaro-Winkler ≥ 0.92", "Jaro-Winkler ≥ 0.80", "else"],
        "dob":        ["exact", "1 character different", "same year", "else"],
        "city":       ["exact", "else"],
        "occupation": ["exact", "else"],
    }


    def compare_values(col, a, b, _cache={}):
        """Comparison level index for one column of one pair (0 = most similar, -1 = null)."""
        if a is None or b is None:
            return -1
        if a == b:
            return 0
        if col in ("first_name", "surname"):
            _key = (a, b) if a < b else (b, a)
            if _key not in _cache:
                _cache[_key] = jaro_winkler(a, b)
            _s = _cache[_key]
            return 1 if _s >= 0.92 else (2 if _s >= 0.80 else 3)
        if col == "dob":
            if levenshtein(a, b) <= 1:
                return 1
            return 2 if a[:4] == b[:4] else 3
        return 1


    def compare_pairs(df, left, right):
        """Return an (n_pairs × n_cols) int matrix of comparison levels (the 'gamma' vectors)."""
        _recs = df[COLS].to_numpy()
        _g = np.empty((len(left), len(COLS)), dtype=int)
        for _k, (_i, _j) in enumerate(zip(left, right)):
            for _c, _col in enumerate(COLS):
                _g[_k, _c] = compare_values(_col, _recs[_i][_c], _recs[_j][_c])
        return _g


    # u probabilities: compare random pairs (overwhelmingly non-matches), as Splink's estimate_u_using_random_sampling does
    _rng = np.random.default_rng(1)
    _l, _r = _rng.integers(0, len(people), 60_000), _rng.integers(0, len(people), 60_000)
    _keep = _l != _r
    _g = compare_pairs(people, _l[_keep], _r[_keep])
    u_probs = {}
    for _c, _col in enumerate(COLS):
        _x = _g[:, _c][_g[:, _c] >= 0]
        _cnt = np.bincount(_x, minlength=len(LEVELS[_col])) + 0.5  # +0.5 smoothing so rare levels aren't log2(0)
        u_probs[_col] = _cnt / _cnt.sum()
    pd.DataFrame([{"column": _col, "level": _lv, "u": u_probs[_col][_k]} for _col in COLS for _k, _lv in enumerate(LEVELS[_col])])
    return LEVELS, compare_pairs, compare_values, u_probs


@app.cell
def dq_ui(mo):
    dq_scenario = mo.ui.radio(
        {"typed from handwritten records": "hand", "auto-filled from driving licences": "licence"},
        value="typed from handwritten records", label="Where did the dates of birth come from?")
    dq_scenario
    return (dq_scenario,)


@app.cell
def dq_chart(LEVELS, alt, dq_scenario, mo, np, pd, u_probs):
    _m = {"hand": [0.70, 0.22, 0.05, 0.03], "licence": [0.9996, 0.0001, 0.0001, 0.0002]}[dq_scenario.value]
    _u = u_probs["dob"]
    _d = pd.DataFrame({"level": LEVELS["dob"], "m": _m, "u": _u})
    _d["weight"] = np.log2(_d.m / _d.u)
    _d["sign"] = np.where(_d.weight >= 0, "for", "against")
    _d["highlight"] = np.where(_d.level == "1 character different", "Lucas: 1857-07-12 vs 1857-01-12", "")
    _c = alt.Chart(_d).mark_bar().encode(
        y=alt.Y("level:N", sort=LEVELS["dob"], title=None), x=alt.X("weight:Q", title="match weight log2(m/u)", scale=alt.Scale(domain=[-12, 18])),
        color=alt.Color("sign:N", scale=alt.Scale(domain=["for", "against"], range=["#2a9d8f", "#e4572e"]), legend=None),
        stroke=alt.condition(alt.datum.highlight != "", alt.value("#f18f01"), alt.value(None)), strokeWidth=alt.value(4),
        tooltip=["level", alt.Tooltip("m:Q", format=".3f"), alt.Tooltip("u:Q", format=".5f"), alt.Tooltip("weight:Q", format="+.2f")])
    _t = alt.Chart(_d).mark_text(align="left", dx=4, fontSize=12).encode(y=alt.Y("level:N", sort=LEVELS["dob"]), x="weight:Q", text=alt.Text("weight:Q", format="+.1f"))
    _w1 = float(_d.loc[1, "weight"])
    mo.vstack([
        (_c + _t).properties(width="container", height=170, title="Date-of-birth comparison levels (u estimated from random pairs)"),
        mo.callout(mo.md(f"A **1-character difference** in DOB is worth **{_w1:+.1f} bits**: "
                         + ("evidence **for** a match, because slips like 7↔1 are expected." if _w1 > 0 else "evidence **against** a match, because the source is reliable.")),
                   kind="success" if _w1 > 0 else "danger"),
        mo.ui.table(_d[["level", "m", "u", "weight"]].round(5), selection=None),
    ])
    return


@app.cell(hide_code=True)
def try4(mo):
    mo.callout(mo.md(r"""
    **Try this**

    * Top chart: set $m = u$. Both weights become 0, because a column that agrees just as often for matches and non-matches carries no information.
    * Push $m \to 0.999$. The **disagree** bar plunges, since a highly reliable column punishes mismatches hard.
    * Flip the DOB scenario. The *same* 1-character difference changes sign. You can't decide this from the two records alone; it's a property of the **whole dataset**.
    """), kind="info")
    return


@app.cell(hide_code=True)
def s5_intro(mo):
    mo.md(r"""
    ## 5 · Blocking: beating the $n^2$ problem

    Comparing every record with every other gives $n(n-1)/2$ pairs: **2.5 billion** for Robin's 50,000 rows. **Blocking** keeps only pairs that share some cheap key, e.g. `l.first_name = r.first_name`.

    Blocking is about **recall, not precision**. It should quickly throw away the *obvious* non-matches; the scoring model decides the rest. We use a **list of rules** (pairs found by *any* rule are kept), because a single rule like "first names equal" would silently drop every nickname and typo before it was ever scored.

    Tick rules on and off, and toggle the data cleaning. Splink-style, each bar counts only the **new** comparisons a rule adds.
    """)
    return


@app.cell
def blocking_rules(COLS, pd, people):
    BLOCKING_RULES = {
        "first_name & surname": ["first_name", "surname"],
        "dob": ["dob"],
        "surname & city": ["surname", "city"],
        "first_name & birth year": ["first_name", "dob_year"],
        "first_name": ["first_name"],
        "occupation": ["occupation"],
    }


    def with_keys(df):
        _d = df.copy()
        _d["dob_year"] = _d["dob"].map(lambda s: s[:4] if s else None)
        return _d


    def rule_count(df, keys):
        """Number of pairs a rule generates, without materialising them: sum over key groups of C(k, 2)."""
        _k = with_keys(df).dropna(subset=keys).groupby(keys).size()
        return int((_k * (_k - 1) // 2).sum())


    def block_pairs(df, rules):
        """Materialise the union of blocking rules; each pair is attributed to the first rule that finds it."""
        _d = with_keys(df)[["unique_id", "dob_year"] + COLS]
        _out = []
        for _name in rules:
            _keys = BLOCKING_RULES[_name]
            _x = _d.dropna(subset=_keys)[["unique_id"] + _keys]
            _p = _x.merge(_x, on=_keys, suffixes=("_l", "_r"))
            _p = _p[_p.unique_id_l < _p.unique_id_r][["unique_id_l", "unique_id_r"]]
            _p["rule"] = _name
            _out.append(_p)
        _all = pd.concat(_out, ignore_index=True).drop_duplicates(["unique_id_l", "unique_id_r"], keep="first")
        return _all.reset_index(drop=True)


    def clean_titles(df):
        _d = df.copy()
        _bad = (_d.first_name == "sir") & (_d.surname == "baronet")
        _d.loc[_bad, ["first_name", "surname"]] = None
        return _d.astype(object).where(_d.notna(), None)


    people_clean = clean_titles(people)
    return BLOCKING_RULES, block_pairs, people_clean, rule_count


@app.cell
def blocking_ui(BLOCKING_RULES, mo):
    block_choice = mo.ui.multiselect(list(BLOCKING_RULES), value=["first_name & surname", "dob", "surname & city", "first_name & birth year"], label="blocking rules (in order)")
    block_clean = mo.ui.switch(value=False, label="clean 'Sir Baronet' titles → null")
    mo.hstack([block_choice, block_clean], justify="start", gap=2)
    return block_choice, block_clean


@app.cell
def blocking_chart(
    BLOCKING_RULES,
    alt,
    block_choice,
    block_clean,
    block_pairs,
    mo,
    n_all_pairs,
    n_true_pairs,
    pd,
    people,
    people_clean,
    rule_count,
):
    _df = people_clean if block_clean.value else people
    _rules = list(block_choice.value)
    _raw = {r: rule_count(_df, BLOCKING_RULES[r]) for r in _rules}
    _upper = sum(_raw.values())
    _LIMIT = 400_000
    if not _rules:
        _out = mo.callout(mo.md("Pick at least one blocking rule."), kind="warn")
    elif _upper > _LIMIT:
        _bars = pd.DataFrame({"rule": _rules, "comparisons": [_raw[r] for r in _rules]})
        _out = mo.vstack([
            mo.callout(mo.md(f"**Stop!** These rules would generate up to **{_upper:,}** comparisons ({_upper / n_all_pairs:.0%} of all pairs). "
                             "On Robin's 50k rows this is the '50 trillion comparisons, blow up your computer' moment. Remove the broad rule."), kind="danger"),
            alt.Chart(_bars).mark_bar(color="#e4572e").encode(x=alt.X("comparisons:Q"), y=alt.Y("rule:N", sort=_rules, title=None)).properties(width="container", height=160),
        ])
    else:
        _p = block_pairs(_df, _rules)
        _tid = people.set_index("unique_id").true_id
        _p["is_match"] = _tid.loc[_p.unique_id_l].to_numpy() == _tid.loc[_p.unique_id_r].to_numpy()
        _by = _p.groupby("rule").agg(new_comparisons=("is_match", "size"), true_matches=("is_match", "sum")).reindex(_rules).fillna(0).reset_index()
        _by["cum"] = _by.new_comparisons.cumsum()
        _by["start"] = _by.cum - _by.new_comparisons
        _chart = alt.Chart(_by).mark_bar(color="#6c5b7b").encode(
            x=alt.X("start:Q", title="cumulative comparisons"), x2="cum:Q", y=alt.Y("rule:N", sort=_rules, title=None),
            tooltip=["rule", "new_comparisons", "true_matches"]).properties(width="container", height=170, title="Comparisons added by each rule")
        _recall = _p.is_match.sum() / n_true_pairs
        _out = mo.vstack([
            _chart,
            mo.hstack([
                mo.stat(f"{len(_p):,}", label="comparisons kept"),
                mo.stat(f"{len(_p) / n_all_pairs:.2%}", label="of all n(n−1)/2 pairs"),
                mo.stat(f"{_recall:.1%}", label="blocking recall (true pairs kept)"),
                mo.stat(f"{_p.is_match.mean():.1%}", label="precision (not the goal!)"),
            ], justify="space-around"),
        ])
    _out
    return


@app.cell
def blocking_offenders(block_clean, mo, people, people_clean):
    _df = people_clean if block_clean.value else people
    _g = (_df.dropna(subset=["first_name", "surname"]).groupby(["first_name", "surname"]).size()
          .rename("records").reset_index())
    _g["comparisons_created"] = _g.records * (_g.records - 1) // 2
    _g = _g.sort_values("comparisons_created", ascending=False).head(6)
    mo.vstack([mo.md("**Which key values generate the most comparisons for rule `first_name & surname`?**"),
               mo.ui.table(_g, selection=None)])
    return


@app.cell(hide_code=True)
def try5(mo):
    mo.callout(mo.md(r"""
    **Try this**

    * Look at the offenders table: `sir / baronet` dominates, just like in the talk. Flip **clean titles** on and watch it disappear.
    * Add the `occupation` rule and you hit the safety brake. Splink checks counts *before* materialising pairs, and so does this cell (it uses $\sum_k \binom{k}{2}$ per key).
    * Use only `first_name & surname`. Precision rises but **recall** drops, because every typo'd name is gone before scoring.
    """), kind="info")
    return


@app.cell(hide_code=True)
def s6_intro(mo):
    mo.md(r"""
    ## 6 · Fellegi-Sunter: adding up match weights, learned without labels

    We now have the three ingredients: similarity **levels** per column (source 1), **u** from coincidences (source 2), and **m** from data quality (source 3). Fellegi-Sunter combines them, assuming the columns are conditionally independent:

    $$\underbrace{\log_2\frac{\lambda}{1-\lambda}}_{\text{prior}} \;+\; \sum_{\text{columns } c} \underbrace{\log_2\frac{m_{c,\ell}}{u_{c,\ell}}}_{\text{match weight}} \;=\; \text{match score}\quad\Rightarrow\quad P(\text{match}) = \frac{2^{\text{score}}}{1+2^{\text{score}}}$$

    **No labels are needed.** Like Splink, we estimate **u** from random pairs (done in section 4) and then **m** with **Expectation-Maximisation** on blocked pairs. As in Splink, we run one EM session per blocking rule and learn only the columns that the rule did *not* force to agree (block on name → learn DOB/city/occupation; block on DOB → learn the names):

    * **E-step**: score every pair with the current $m$ and get $p_k = P(\text{match}\mid\gamma_k)$.
    * **M-step**: $m_{c,\ell} = \dfrac{\sum_k p_k\,[\gamma_{k,c}=\ell]}{\sum_k p_k}$, and $\lambda = \text{mean}(p_k)$.

    Scrub through the EM iterations. Bars are the learned weights; the black ticks are the *true* weights computed from the hidden labels, which EM never sees.
    """)
    return


@app.cell
def em_training(
    BLOCKING_RULES,
    COLS,
    LEVELS,
    block_pairs,
    compare_pairs,
    mo,
    n_all_pairs,
    n_true_pairs,
    np,
    people,
    people_clean,
    u_probs,
):
    TRAIN_RULES = ["first_name & surname", "dob", "surname & city", "first_name & birth year"]
    pairs = block_pairs(people_clean, TRAIN_RULES).astype({"unique_id_l": int, "unique_id_r": int})
    _tid = people.true_id.to_numpy().astype(int)
    pairs["is_match"] = _tid[pairs.unique_id_l.to_numpy()] == _tid[pairs.unique_id_r.to_numpy()]
    gamma = compare_pairs(people_clean, pairs.unique_id_l.to_numpy(), pairs.unique_id_r.to_numpy())


    def score_gamma(g, m, u, prior_logodds, cols=None):
        """Fellegi-Sunter match score (bits) for each row of a gamma matrix."""
        _s = np.full(len(g), prior_logodds, dtype=float)
        for _col in (cols or COLS):
            _c = COLS.index(_col)
            _w = np.log2(m[_col] / u[_col])
            _ok = g[:, _c] >= 0
            _s[_ok] += _w[g[_ok, _c]]
        return _s


    def em_session(rule, u, n_iter=20, lam=0.05):
        """One Splink-style EM training session: block on `rule`, learn m for the columns NOT in the rule (u stays fixed)."""
        _p = block_pairs(people_clean, [rule]).astype({"unique_id_l": int, "unique_id_r": int})
        _g = compare_pairs(people_clean, _p.unique_id_l.to_numpy(), _p.unique_id_r.to_numpy())
        _cols = [c for c in COLS if c not in BLOCKING_RULES[rule]]
        _m = {c: np.r_[0.9, np.full(len(LEVELS[c]) - 1, 0.1 / (len(LEVELS[c]) - 1))] for c in _cols}
        _hist = [{c: v.copy() for c, v in _m.items()}]
        for _it in range(n_iter):
            _pm = 1 / (1 + 2 ** -score_gamma(_g, _m, u, np.log2(lam / (1 - lam)), _cols))   # E-step
            for _col in _cols:                                                              # M-step
                _c = COLS.index(_col)
                _ok = _g[:, _c] >= 0
                _num = np.bincount(_g[_ok, _c], weights=_pm[_ok], minlength=len(LEVELS[_col])) + 0.01
                _m[_col] = _num / _num.sum()
            lam = float(_pm.mean())
            _hist.append({c: v.copy() for c, v in _m.items()})
        return _hist


    # two sessions, as in the Splink tutorial: names are learned from the DOB-blocked pairs and vice versa
    _sessions = [em_session("first_name & surname", u_probs), em_session("dob", u_probs)]
    em_history = []
    for _t in range(len(_sessions[0])):
        em_history.append({c: np.mean([s[_t][c] for s in _sessions if c in s[_t]], axis=0) for c in COLS})
    m_probs = em_history[-1]

    # prior λ, Splink's estimate_probability_two_random_records_match: count near-certain matches from a strict
    # deterministic rule, then divide by a *user-supplied guess* of that rule's recall
    DETERMINISTIC_RECALL = 0.4
    n_deterministic = int(((gamma[:, 0] == 0) & (gamma[:, 1] == 0) & (gamma[:, 2] == 0)).sum())
    prior_prob = n_deterministic / DETERMINISTIC_RECALL / n_all_pairs
    prior_logodds = float(np.log2(prior_prob / (1 - prior_prob)))

    # ground-truth m from the hidden labels (all truly matching pairs), for checking only
    _grp = people_clean.groupby(people.true_id).unique_id.apply(list)
    _tl, _tr = zip(*[(_a, _b) for _ids in _grp for _i, _a in enumerate(_ids) for _b in _ids[_i + 1:]])
    _gt = compare_pairs(people_clean, np.array(_tl, dtype=int), np.array(_tr, dtype=int))
    true_m = {}
    for _c, _col in enumerate(COLS):
        _x = _gt[:, _c][_gt[:, _c] >= 0]
        true_m[_col] = (np.bincount(_x, minlength=len(LEVELS[_col])) + 0.5) / (len(_x) + 0.5 * len(LEVELS[_col]))
    mo.hstack([
        mo.stat(f"{len(pairs):,}", label="blocked pairs (all 4 rules)"),
        mo.stat(f"{n_deterministic:,}", label="pairs exact on first name + surname + DOB"),
        mo.stat(f"{DETERMINISTIC_RECALL:.0%}", label="assumed recall of that rule"),
        mo.stat(f"1 in {1 / prior_prob:,.0f}", label=f"prior for random pair (truth 1 in {n_all_pairs / n_true_pairs:,.0f})"),
        mo.stat(f"{prior_logodds:+.1f}", label="prior match weight (bits)"),
    ], justify="space-around")
    return (
        em_history,
        gamma,
        m_probs,
        pairs,
        prior_logodds,
        score_gamma,
        true_m,
    )


@app.cell
def em_iter_ui(em_history, mo):
    em_iter = mo.ui.slider(0, len(em_history) - 1, value=0, step=1, label="EM iteration", show_value=True, full_width=True)
    em_iter
    return (em_iter,)


@app.cell
def match_weights_chart(
    COLS,
    LEVELS,
    alt,
    em_history,
    em_iter,
    mo,
    np,
    pd,
    true_m,
    u_probs,
):
    _h = em_history[em_iter.value]
    _rows = []
    for _col in COLS:
        for _k, _lv in enumerate(LEVELS[_col]):
            _w = float(np.log2(_h[_col][_k] / u_probs[_col][_k]))
            _rows.append({"column": _col, "level": f"{_col}: {_lv}", "weight": _w, "m": _h[_col][_k], "u": u_probs[_col][_k],
                          "true_weight": float(np.log2(true_m[_col][_k] / u_probs[_col][_k])), "sign": "for" if _w >= 0 else "against"})
    _d = pd.DataFrame(_rows)
    _order = _d.level.tolist()
    _y = alt.Y("level:N", sort=_order, title=None)
    _bars = alt.Chart(_d).mark_bar().encode(
        y=_y, x=alt.X("weight:Q", title="match weight log2(m/u)", scale=alt.Scale(domain=[-14, 14], clamp=True)),
        color=alt.Color("sign:N", scale=alt.Scale(domain=["for", "against"], range=["#2a9d8f", "#e4572e"]), legend=None),
        tooltip=["level", alt.Tooltip("m:Q", format=".4f"), alt.Tooltip("u:Q", format=".5f"), alt.Tooltip("weight:Q", format="+.2f"), alt.Tooltip("true_weight:Q", format="+.2f")])
    _ticks = alt.Chart(_d).mark_tick(color="black", thickness=2, size=16).encode(y=_y, x=alt.X("true_weight:Q", scale=alt.Scale(domain=[-14, 14], clamp=True)))
    _err = float(np.mean(np.abs(_d.weight - _d.true_weight)))
    mo.vstack([
        (_bars + _ticks).properties(width="container", height=330, title=f"Match weights chart, EM iteration {em_iter.value}"),
        mo.hstack([mo.stat(f"{_err:.2f} bits", label="mean |learned − true| weight")], justify="start"),
    ])
    return


@app.cell(hide_code=True)
def try6(mo):
    mo.callout(mo.md(r"""
    **Try this**

    * Start at iteration 0, where every column guesses $m_{\text{exact}} = 0.9$. Slide right and watch EM pull the bars onto the black ticks within a few iterations.
    * Look at the **dob: 1 character different** bar. EM learned, from unlabelled data, that DOB slips are common here. That's information source #3 working automatically.
    * **occupation: exact** is weak (broad categories), while **dob: exact** is huge. EM sets the relative importance, so no human has to weigh "first name vs. DOB" by hand.
    * Caveat: blocking on name equality inflates name $m$ a little (Splink trains one EM session per blocking rule to avoid that bias).
    """), kind="info")
    return


@app.cell(hide_code=True)
def s7_intro(mo):
    mo.md(r"""
    ## 7 · Reading one prediction: the waterfall chart

    Robin's favourite way to explain a prediction: *reach into a box of all records, pull out two in **sealed envelopes***. Before you look, the chance they match is the **prior** (about 1 in 1,500 here, so a very negative weight). Then you **open the envelopes** one column at a time, and each comparison adds its match weight. The running total is the match score; above **0 bits** the pair is more likely a match than not.

    **Load a real pair, then click the level chips** to ask *"what if DOB had matched exactly?"* (this is the match-weights chart and the waterfall in one).
    """)
    return


@app.cell
def waterfall_class(anywidget, traitlets):
    class Waterfall(anywidget.AnyWidget):
        """Clickable comparison levels → live waterfall of match weights (Fellegi-Sunter)."""
        _esm = r"""
    function render({ model, el }) {
      const NS = "http://www.w3.org/2000/svg", W = 700, H = 300, PADL = 46, PADR = 70, TOP = 18, BOT = 64;
      const chips = document.createElement("div");
      chips.style.cssText = "display:grid;grid-template-columns:110px 1fr;gap:4px 8px;font-family:sans-serif;font-size:13px;margin-bottom:8px";
      const svg = document.createElementNS(NS, "svg");
      svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
      Object.assign(svg.style, { width: "100%", maxWidth: W + "px", userSelect: "none" });
      const out = document.createElement("div");
      out.style.cssText = "font-family:sans-serif;font-size:15px;margin-top:4px";
      el.append(chips, svg, out);
      const mk = (tag, a) => { const e = document.createElementNS(NS, tag); for (const k in a) e.setAttribute(k, a[k]); svg.appendChild(e); return e; };
      const prob = s => 1 / (1 + Math.pow(2, -s));
      function drawChips() {
        chips.innerHTML = "";
        const cols = model.get("columns"), ch = model.get("choice");
        cols.forEach((c, i) => {
          const lab = document.createElement("div"); lab.innerHTML = `<b>${c.name}</b>`; chips.appendChild(lab);
          const row = document.createElement("div"); row.style.cssText = "display:flex;flex-wrap:wrap;gap:4px";
          [...c.labels.map((l, k) => [l, k, c.weights[k]]), ["null", -1, 0]].forEach(([l, k, w]) => {
            const b = document.createElement("button");
            const on = ch[i] === k;
            b.textContent = k < 0 ? "null (0)" : `${l} (${w >= 0 ? "+" : ""}${w.toFixed(1)})`;
            b.style.cssText = `border-radius:12px;padding:2px 9px;cursor:pointer;border:2px solid ${on ? "#f18f01" : "transparent"};` +
              `background:${k < 0 ? "rgba(128,128,128,.18)" : (w >= 0 ? "rgba(42,157,143,.22)" : "rgba(228,87,46,.22)")};font-weight:${on ? 700 : 400}`;
            b.onclick = () => { const a = [...model.get("choice")]; a[i] = k; model.set("choice", a); model.save_changes(); };
            row.appendChild(b);
          });
          chips.appendChild(row);
        });
      }
      function drawChart() {
        svg.innerHTML = "";
        const cols = model.get("columns"), ch = model.get("choice"), tf = model.get("tf");
        const steps = [{ name: "prior", w: model.get("prior"), kind: "prior" }];
        cols.forEach((c, i) => {
          steps.push({ name: c.name, w: ch[i] < 0 ? 0 : c.weights[ch[i]], kind: "col" });
          if (tf.length && tf[i]) steps.push({ name: "tf " + c.name, w: tf[i], kind: "tf" });
        });
        let cum = 0; const bars = steps.map(s => { const a = cum; cum += s.w; return { ...s, a, b: cum }; });
        bars.push({ name: "final", a: 0, b: cum, w: cum, kind: "final" });
        const vals = bars.flatMap(b => [b.a, b.b]).concat([0]);
        const lo = Math.min(...vals) - 1, hi = Math.max(...vals) + 1;
        const sy = v => TOP + (hi - v) / (hi - lo) * (H - TOP - BOT);
        const bw = (W - PADL - PADR) / bars.length;
        for (let t = Math.ceil(lo / 5) * 5; t <= hi; t += 5) {
          mk("line", { x1: PADL, x2: W - PADR, y1: sy(t), y2: sy(t), stroke: "rgba(128,128,128,.2)" });
          mk("text", { x: PADL - 6, y: sy(t) + 4, "text-anchor": "end", "font-size": 11, fill: "currentColor" }).textContent = t;
          mk("text", { x: W - PADR + 6, y: sy(t) + 4, "font-size": 10, fill: "#6c5b7b" }).textContent = (prob(t) * 100).toPrecision(2) + "%";
        }
        mk("line", { x1: PADL, x2: W - PADR, y1: sy(0), y2: sy(0), stroke: "currentColor", "stroke-width": 2 });
        mk("text", { x: W - PADR + 6, y: TOP - 4, "font-size": 10, fill: "#6c5b7b" }).textContent = "P(match)";
        bars.forEach((b, k) => {
          const x = PADL + k * bw + 4, y1 = sy(Math.max(b.a, b.b)), y2 = sy(Math.min(b.a, b.b));
          const fill = b.kind === "prior" || b.kind === "final" ? "#6c5b7b" : (b.w >= 0 ? "#2a9d8f" : "#e4572e");
          mk("rect", { x, y: y1, width: bw - 8, height: Math.max(y2 - y1, 1.5), fill, opacity: b.kind === "tf" ? 0.55 : 1, rx: 3 });
          if (k < bars.length - 1) mk("line", { x1: x + bw - 8, x2: x + bw + 4, y1: sy(b.b), y2: sy(b.b), stroke: "#999", "stroke-dasharray": "3 2" });
          mk("text", { x: x + (bw - 8) / 2, y: y1 - 4, "text-anchor": "middle", "font-size": 11, fill: "currentColor" }).textContent = (b.w >= 0 ? "+" : "") + b.w.toFixed(1);
          const t = mk("text", { x: x + (bw - 8) / 2, y: H - BOT + 14, "text-anchor": "end", "font-size": 11, fill: "currentColor",
            transform: `rotate(-35 ${x + (bw - 8) / 2} ${H - BOT + 14})` }); t.textContent = b.name;
        });
        const p = prob(cum);
        out.innerHTML = `Final match weight <b style="color:#6c5b7b">${cum.toFixed(2)}</b> bits → P(match) = <b style="color:${p >= .5 ? "#2a9d8f" : "#e4572e"}">${(p * 100).toFixed(p > .999 || p < .001 ? 4 : 1)}%</b>`;
      }
      const all = () => { drawChips(); drawChart(); };
      model.on("change:choice", all); model.on("change:columns", all); model.on("change:prior", all); model.on("change:tf", all);
      all();
    }
    export default { render };
    """
        columns = traitlets.List(traitlets.Dict()).tag(sync=True)
        choice = traitlets.List(traitlets.Int()).tag(sync=True)
        prior = traitlets.Float(-10.0).tag(sync=True)
        tf = traitlets.List(traitlets.Float()).tag(sync=True)

    return (Waterfall,)


@app.cell
def scored_pairs(
    COLS,
    LEVELS,
    gamma,
    m_probs,
    mo,
    np,
    pairs,
    prior_logodds,
    score_gamma,
    u_probs,
):
    match_weights = {c: np.log2(m_probs[c] / u_probs[c]) for c in COLS}
    wf_columns = [{"name": c, "labels": LEVELS[c], "weights": [float(w) for w in match_weights[c]]} for c in COLS]
    scored = pairs.assign(match_weight=score_gamma(gamma, m_probs, u_probs, prior_logodds))
    scored["match_probability"] = 1 / (1 + 2 ** -scored.match_weight)

    _g, _y, _s = gamma, scored.is_match.to_numpy(), scored.match_weight.to_numpy()
    def _first(mask):
        _ix = np.flatnonzero(mask)
        return int(_ix[0]) if len(_ix) else 0
    EXAMPLE_PAIRS = {
        "Same person, DOB slip (Robin's Lucas example)": _first(_y & (_g[:, 0] == 0) & (_g[:, 1] == 0) & (_g[:, 2] == 1) & (_g[:, 3] == 0)),
        "Same person, nickname + moved city": _first(_y & (_g[:, 0] == 3) & (_g[:, 3] == 1) & (_g[:, 2] == 0)),
        "Different people sharing a common name": _first(~_y & (_g[:, 0] == 0) & (_g[:, 1] == 0) & (_g[:, 2] == 3)),
        "Same person, surname typo": _first(_y & (_g[:, 1] == 1) & (_g[:, 0] == 0)),
        "A borderline pair (score near 0)": int(np.argmin(np.abs(_s))),
    }
    wf_pick = mo.ui.dropdown(list(EXAMPLE_PAIRS), value="Same person, DOB slip (Robin's Lucas example)", label="load a pair")
    wf_pick
    return EXAMPLE_PAIRS, match_weights, scored, wf_columns, wf_pick


@app.cell
def waterfall_view(
    EXAMPLE_PAIRS,
    Waterfall,
    gamma,
    mo,
    prior_logodds,
    wf_columns,
    wf_pick,
):
    _k = EXAMPLE_PAIRS[wf_pick.value]
    waterfall = mo.ui.anywidget(Waterfall(columns=wf_columns, choice=[int(v) for v in gamma[_k]], prior=prior_logodds, tf=[]))
    waterfall
    return (waterfall,)


@app.cell
def waterfall_readout(
    COLS,
    EXAMPLE_PAIRS,
    gamma,
    match_weights,
    mo,
    people_clean,
    prior_logodds,
    scored,
    waterfall,
    wf_pick,
):
    _k = EXAMPLE_PAIRS[wf_pick.value]
    _l, _r = int(scored.unique_id_l[_k]), int(scored.unique_id_r[_k])
    _recs = people_clean.loc[[_l, _r], ["unique_id"] + COLS].copy()
    _recs.insert(0, "record", ["left", "right"])
    _ch = waterfall.value["choice"]
    _score = prior_logodds + sum(match_weights[c][k] for c, k in zip(COLS, _ch) if k >= 0)
    _edited = list(_ch) != [int(v) for v in gamma[_k]]
    mo.vstack([
        mo.ui.table(_recs, selection=None, label="The two records"),
        mo.hstack([
            mo.stat(f"{_score:+.2f}", label="match weight (Python check)"),
            mo.stat(f"{1 / (1 + 2 ** -_score):.2%}", label="P(match)"),
            mo.stat("yes" if scored.is_match[_k] else "no", label="truly the same person? (hidden)"),
            mo.stat("what-if (edited)" if _edited else "as observed", label="levels"),
        ], justify="space-around"),
    ])
    return


@app.cell(hide_code=True)
def try7(mo):
    mo.callout(mo.md(r"""
    **Try this**

    * On the Lucas pair, click **dob: exact**. The DOB bar grows, which is exactly the "flick between the two arrows" moment on Robin's match-weights chart.
    * On *Different people sharing a common name*: name agreement pulls the score up, but the **dob: else** bar drags it back below 0.
    * Set every column to **null**. You're left with the prior, a reminder of how little a random pair is worth before you open the envelopes.
    """), kind="info")
    return


@app.cell(hide_code=True)
def s8_intro(mo):
    mo.md(r"""
    ## 8 · From pairwise predictions to a new person ID: clustering

    Pairwise predictions form a **graph**: records are nodes, and pairs above a probability threshold are edges. Each **connected component** becomes an **estimated cluster**, which is the `person_id` the data never had. This is transitive: if A–B and B–C are linked, A and C end up in the same cluster even if they were never compared directly.

    Slide the threshold and watch the precision/recall trade-off (labels are hidden from the model and used here only to score it). Below that, one neighbourhood is drawn as a graph.
    """)
    return


@app.cell
def cluster_threshold_ui(mo):
    cl_threshold = mo.ui.slider(0.01, 0.99, step=0.01, value=0.9, label="accept a match if P(match) ≥", show_value=True, full_width=True)
    cl_threshold
    return (cl_threshold,)


@app.cell
def cluster_functions(n_true_pairs, np, pd, scored):
    def connected_components(n, left, right):
        """Union-find: cluster id for each of n nodes given an edge list."""
        _parent = np.arange(n)
        def _find(x):
            while _parent[x] != x:
                _parent[x] = _parent[_parent[x]]
                x = _parent[x]
            return x
        for _a, _b in zip(left, right):
            _ra, _rb = _find(_a), _find(_b)
            if _ra != _rb:
                _parent[max(_ra, _rb)] = min(_ra, _rb)
        return np.array([_find(_i) for _i in range(n)])


    def evaluate_threshold(t):
        _e = scored[scored.match_probability >= t]
        _tp = int(_e.is_match.sum())
        return {"threshold": t, "precision": _tp / max(len(_e), 1), "recall": _tp / n_true_pairs, "edges": len(_e)}


    pr_curve = pd.DataFrame([evaluate_threshold(t) for t in np.round(np.linspace(0.01, 0.99, 50), 3)])
    return connected_components, evaluate_threshold, pr_curve


@app.cell
def clusters_view(
    alt,
    cl_threshold,
    connected_components,
    evaluate_threshold,
    mo,
    pd,
    people,
    people_clean,
    pr_curve,
    scored,
):
    _edges = scored[scored.match_probability >= cl_threshold.value]
    clusters = people_clean.assign(
        estimated_cluster=connected_components(len(people_clean), _edges.unique_id_l.to_numpy(), _edges.unique_id_r.to_numpy()),
        true_id=people.true_id)
    _ev = evaluate_threshold(cl_threshold.value)
    _pure = clusters.groupby("estimated_cluster").true_id.nunique().eq(1).mean()
    _long = pr_curve.melt("threshold", ["precision", "recall"], var_name="metric")
    _lines = alt.Chart(_long).mark_line(strokeWidth=3).encode(
        x=alt.X("threshold:Q", title="probability threshold"), y=alt.Y("value:Q", scale=alt.Scale(domain=[0.5, 1.0]), title=None),
        color=alt.Color("metric:N", scale=alt.Scale(domain=["precision", "recall"], range=["#2a9d8f", "#6c5b7b"]), legend=alt.Legend(orient="bottom", title=None)))
    _rule = alt.Chart(pd.DataFrame({"t": [cl_threshold.value]})).mark_rule(color="#f18f01", size=3).encode(x="t:Q")
    mo.vstack([
        (_lines + _rule).properties(width="container", height=230, title="Pairwise precision & recall vs threshold"),
        mo.hstack([
            mo.stat(f"{clusters.estimated_cluster.nunique():,}", label=f"estimated people (truth {people.true_id.nunique():,})"),
            mo.stat(f"{_ev['precision']:.1%}", label="pairwise precision"),
            mo.stat(f"{_ev['recall']:.1%}", label="pairwise recall (incl. blocking losses)"),
            mo.stat(f"{_pure:.1%}", label="clusters containing only one true person"),
        ], justify="space-around"),
    ])
    return (clusters,)


@app.cell
def cluster_pick_ui(clusters, mo):
    _sizes = clusters.estimated_cluster.value_counts()
    _cand = clusters[clusters.estimated_cluster.isin(_sizes[_sizes >= 3].index)]
    _ch = _cand[_cand.surname == "chudley"]
    _opts = {}
    for _cid in list(dict.fromkeys(list(_ch.estimated_cluster) + list(_cand.estimated_cluster)))[:15]:
        _r = clusters[clusters.estimated_cluster == _cid].iloc[0]
        _opts[f"{_r.first_name} {_r.surname} (cluster {_cid}, {_sizes[_cid]} records)"] = int(_cid)
    cl_pick = mo.ui.dropdown(_opts, value=next(iter(_opts)), label="show cluster") if _opts else None
    cl_pick
    return (cl_pick,)


@app.cell
def cluster_graph(
    COLS,
    alt,
    cl_pick,
    cl_threshold,
    clusters,
    mo,
    np,
    pd,
    scored,
):
    mo.stop(cl_pick is None, mo.md("_No clusters with ≥3 records at this threshold._"))
    _members = clusters[clusters.estimated_cluster == cl_pick.value]
    _ids = set(_members.unique_id.astype(int))
    # include near-miss neighbours (pairs that touch the cluster but fell below the threshold)
    _touch = scored[scored.unique_id_l.isin(_ids) | scored.unique_id_r.isin(_ids)]
    _touch = _touch[_touch.match_probability >= 0.01]
    _nodes = sorted(set(_touch.unique_id_l) | set(_touch.unique_id_r) | _ids)[:24]
    _ang = np.linspace(0, 2 * np.pi, len(_nodes), endpoint=False)
    _pos = {n: (np.cos(a), np.sin(a)) for n, a in zip(_nodes, _ang)}
    _e = _touch[_touch.unique_id_l.isin(_pos) & _touch.unique_id_r.isin(_pos)]
    _edf = pd.DataFrame([{"edge": k, "x": _pos[a][0], "y": _pos[a][1], "p": p, "kept": "linked" if p >= cl_threshold.value else "below threshold"}
                         for k, (a, b, p) in enumerate(zip(_e.unique_id_l, _e.unique_id_r, _e.match_probability))] +
                        [{"edge": k, "x": _pos[b][0], "y": _pos[b][1], "p": p, "kept": "linked" if p >= cl_threshold.value else "below threshold"}
                         for k, (a, b, p) in enumerate(zip(_e.unique_id_l, _e.unique_id_r, _e.match_probability))])
    _ndf = clusters.set_index("unique_id").loc[_nodes].reset_index()
    _ndf["x"], _ndf["y"] = [_pos[n][0] for n in _nodes], [_pos[n][1] for n in _nodes]
    _ndf["label"] = _ndf.first_name.fillna("∅") + " " + _ndf.surname.fillna("∅") + " · " + _ndf.dob.fillna("∅")
    _ndf[["city", "occupation"]] = _ndf[["city", "occupation"]].fillna("∅")
    _ndf["in_cluster"] = np.where(_ndf.estimated_cluster == cl_pick.value, "this cluster", "other cluster")
    _ax = alt.Axis(labels=False, ticks=False, grid=False, domain=False, title=None)
    _lines = alt.Chart(_edf).mark_line().encode(
        x=alt.X("x:Q", axis=_ax, scale=alt.Scale(domain=[-1.6, 1.6])), y=alt.Y("y:Q", axis=_ax, scale=alt.Scale(domain=[-1.3, 1.3])), detail="edge:N",
        color=alt.Color("kept:N", scale=alt.Scale(domain=["linked", "below threshold"], range=["#2a9d8f", "#e4572e"]), legend=alt.Legend(orient="bottom", title=None)),
        strokeDash=alt.condition(alt.datum.kept == "linked", alt.value([1, 0]), alt.value([4, 3])),
        strokeWidth=alt.Size("p:Q", scale=alt.Scale(range=[1, 5]), legend=None), tooltip=[alt.Tooltip("p:Q", format=".3f")])
    _pts = alt.Chart(_ndf).mark_circle(size=260, opacity=1).encode(
        x="x:Q", y="y:Q", color=alt.Color("in_cluster:N", scale=alt.Scale(domain=["this cluster", "other cluster"], range=["#f18f01", "#999"]), legend=None),
        tooltip=["unique_id", "label", "city", "occupation", "estimated_cluster", "true_id"])
    _txt = _pts.mark_text(dy=-16, fontSize=11).encode(text="label:N", color=alt.value("currentColor"))
    mo.vstack([
        (_lines + _pts + _txt).properties(width="container", height=380, title="Pairwise links around the chosen cluster"),
        mo.ui.table(_members[["unique_id", "estimated_cluster", "true_id"] + COLS], selection=None, label="The resolved entity: one estimated_cluster ID for every variant"),
    ])
    return


@app.cell(hide_code=True)
def try8(mo):
    mo.callout(mo.md(r"""
    **Try this**

    * Drop the threshold to 0.05. Recall goes up, but clusters start **chaining** different people together (watch "clusters containing only one true person" fall). That's the price of transitivity.
    * Raise it to 0.99. Clusters fragment: one person splits into several IDs.
    * Recall can never reach 100%, because pairs dropped by **blocking** were never scored. Blocking recall is an upper bound.
    """), kind="info")
    return


@app.cell(hide_code=True)
def s9_intro(mo):
    mo.md(r"""
    ## 9 · Sandbox: score your own pair

    Type two records. The notebook runs the full pipeline on them (comparison levels → learned weights → optional **term-frequency adjustments** for exact name and occupation matches → waterfall). Empty fields count as null.
    """)
    return


@app.cell
def sandbox_ui(COLS, mo):
    _defaults_l = {"first_name": "lucas", "surname": "smith", "dob": "1857-07-12", "city": "devon", "occupation": "politician"}
    _defaults_r = {"first_name": "lucas", "surname": "smyth", "dob": "1857-01-12", "city": "", "occupation": "politician"}
    sb_left = mo.ui.dictionary({c: mo.ui.text(value=_defaults_l[c]) for c in COLS}, label="record A")
    sb_right = mo.ui.dictionary({c: mo.ui.text(value=_defaults_r[c]) for c in COLS}, label="record B")
    sb_tf = mo.ui.switch(value=True, label="term-frequency adjustments")
    mo.vstack([mo.hstack([sb_left, sb_right], justify="start", gap=3), sb_tf])
    return sb_left, sb_right, sb_tf


@app.cell
def sandbox_waterfall(
    COLS,
    LEVELS,
    Waterfall,
    compare_values,
    match_weights,
    mo,
    np,
    people_clean,
    prior_logodds,
    sb_left,
    sb_right,
    sb_tf,
    u_probs,
    wf_columns,
):
    _norm = lambda v: v.strip().lower() or None
    _a = {c: _norm(sb_left.value[c]) for c in COLS}
    _b = {c: _norm(sb_right.value[c]) for c in COLS}
    _choice = [compare_values(c, _a[c], _b[c]) for c in COLS]
    _tf = []
    for _c, _col in enumerate(COLS):
        _adj = 0.0
        if sb_tf.value and _choice[_c] == 0 and _col in ("first_name", "surname", "occupation"):
            _f = (people_clean[_col] == _a[_col]).mean()
            _f = max(_f, 1 / len(people_clean))           # unseen value → treat as seen once
            _adj = float(np.log2(u_probs[_col][0] / _f))   # Splink's TF adjustment: swap average u for this value's frequency
        _tf.append(_adj)
    sandbox_score = prior_logodds + sum(match_weights[c][k] for c, k in zip(COLS, _choice) if k >= 0) + sum(_tf)
    sandbox_wf = mo.ui.anywidget(Waterfall(columns=wf_columns, choice=_choice, prior=prior_logodds, tf=_tf))
    mo.vstack([
        sandbox_wf,
        mo.md(" · ".join(f"**{c}**: {LEVELS[c][k] if k >= 0 else 'null'}" for c, k in zip(COLS, _choice))),
        mo.stat(f"{1 / (1 + 2 ** -sandbox_score):.2%}", label=f"P(match) for the pair as typed ({sandbox_score:+.2f} bits)"),
    ])
    return


@app.cell(hide_code=True)
def sandbox_challenges(mo):
    mo.callout(mo.md(r"""
    **Challenges**

    1. Make two records with **identical** names, but different people, that score below 50%.
    2. Find the cheapest edit to record B that flips the Lucas pair below 50%.
    3. Set both surnames to `smith`, then to `linacre`, with TF adjustments on. How many bits does the rarer surname add?
    4. Use `bill` vs `william`. Jaro-Winkler puts it in *else*. What custom comparison level would you add? (Hint: Robin's Q&A answer about SQL `CASE WHEN` levels, e.g. name inversions.)
    """), kind="neutral")
    return


@app.cell(hide_code=True)
def s10_intro(mo):
    mo.md(r"""
    ## 10 · Quiz: check your intuition
    """)
    return


@app.cell
def quiz_ui(mo):
    quiz = mo.ui.dictionary({
        "q1": mo.ui.dropdown(["Smith", "Linacre", "They're equal: same edit distance"], label="1. A 1-typo fuzzy match on which surname is stronger evidence?"),
        "q2": mo.ui.dropdown(["positive", "negative", "zero"], label="2. If DOB is copied from driving licences, the weight for 'DOB differs by 1 char' is…"),
        "q3": mo.ui.dropdown(["precision", "recall", "speed only"], label="3. Blocking rules should mainly maximise…"),
        "q4": mo.ui.dropdown(["log2(m/u)", "m − u", "log2(u/m)", "m/(m+u)"], label="4. The match weight of a comparison level is…"),
        "q5": mo.ui.dropdown(["labelled training pairs", "EM on unlabelled pairs + random sampling for u", "a hand-tuned score card"], label="5. Without labels, Splink learns m and u via…"),
        "q6": mo.ui.dropdown(["fewer clusters, more chaining", "more clusters, fragmented people", "no change"], label="6. Lowering the clustering threshold tends to give…"),
    })
    quiz_reveal = mo.ui.switch(label="reveal answers")
    mo.vstack([quiz, quiz_reveal])
    return quiz, quiz_reveal


@app.cell
def quiz_score(mo, quiz, quiz_reveal):
    _truth = {"q1": "Linacre", "q2": "negative", "q3": "recall", "q4": "log2(m/u)",
              "q5": "EM on unlabelled pairs + random sampling for u", "q6": "fewer clusters, more chaining"}
    _why = {"q1": "Coincidences: random people share 'Smith' (or a near-Smith) far more often → larger u → smaller m/u.",
            "q2": "Reliable source → m for that level is tiny, so log2(m/u) < 0. Same records, different dataset, opposite sign.",
            "q3": "Anything blocking drops is never scored; the model handles precision afterwards.",
            "q4": "w = log2(m/u), summed with the prior log-odds; P = 2^w / (1 + 2^w).",
            "q5": "u from random pairs (nearly all non-matches), m from EM per blocking rule.",
            "q6": "More edges → connected components merge (transitivity)."}
    _score = sum(quiz.value[k] == v for k, v in _truth.items())
    _rows = [f"| {k[1]} | {'✅' if quiz.value[k] == v else '❌'} | {v} | {_why[k]} |" for k, v in _truth.items()]
    mo.vstack([
        mo.stat(f"{_score} / {len(_truth)}", label="score"),
        mo.md("| # | | answer | why |\n|---|---|---|---|\n" + "\n".join(_rows)) if quiz_reveal.value else mo.md("_Flip **reveal answers** when you've committed._"),
    ])
    return


@app.cell(hide_code=True)
def recap(mo):
    mo.md(r"""
    ## Recap

    | Step | Idea | Formula / rule | Where you touched it |
    |---|---|---|---|
    | Fuzzy matching | similarity of two values in isolation | Levenshtein DP, Jaro-Winkler | §2 DP stepper |
    | Term frequency | coincidences depend on the value | $w_v = \log_2(m / f_v)$ | §3 Smith vs Linacre |
    | Data quality | how often true matches disagree | $m_\ell$, and $\log_2\frac{1-m}{1-u}$ for disagreement | §4 handwritten vs licence |
    | Blocking | avoid $n^2$, keep **recall** | union of `l.x = r.x` rules, $\sum_k \binom{k}{2}$ pre-count | §5 rule picker |
    | Fellegi-Sunter | add up independent evidence | $\text{score} = \log_2\frac{\lambda}{1-\lambda} + \sum_c \log_2\frac{m_{c,\ell}}{u_{c,\ell}}$ | §6 EM scrubber |
    | Unsupervised training | no labels needed | u by random sampling, m by EM per blocking rule | §6 |
    | Explainability | one prediction = a sum of bars | waterfall / match-weights chart | §7 clickable waterfall |
    | Clustering | pairs → entities | connected components above a threshold | §8 graph + P/R curve |

    **What this notebook simplifies vs. real Splink:** SQL backends (DuckDB/Spark) for scale, many more comparison templates, TF adjustments on fuzzy levels, and per-rule λ estimation. The maths is the same.
    """)
    return


if __name__ == "__main__":
    app.run()

# /// script
# dependencies = [
#     "altair==6.3.0",
#     "anywidget==0.11.0",
#     "marimo",
#     "numpy==2.5.4",
#     "pandas==3.1.1",
#     "traitlets==5.16.1",
# ]
# requires-python = ">=3.12"
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
    # Sir, this is a classification task
    *An interactive companion to marimo's video **"Sir, This is a Classification Task"**, about GLiNER-style encoder models.*

    > Everyone is fine-tuning LLMs like Qwen into **decision models** (classifiers).
    > It works. But a model built to *generate text* may be an expensive way to pick one of three labels.
    > **Do we need the decoder at all?**

    The video's answer, rebuilt here with things you can click:
    **cost of the two architectures → contextualised embeddings → labels as input tokens →
    entities as spans → label descriptions → sandbox & quiz.**
    """),
        mo.callout(mo.md(r"""
    **This notebook uses a toy model.** There is no neural network download here. The "encoder" is a hand-made
    9-dimensional embedding table (`money`, `failure`, `pos`, `neg`, `person`, `place`, `time`, `shipping`, `action`)
    plus two layers of *parameter-free* self-attention (query = key = value = embedding). Every number is readable,
    so you can see *why* a label wins. Real GLiNER models learn 768-d embeddings and the weights of the
    classification head. For those, see the extension at the end (`pip install gliner`, Fastino's models).

    **Color language**
    <span style="background:#2a9d8f;color:white;padding:1px 6px;border-radius:4px">label 1</span>
    <span style="background:#f18f01;color:white;padding:1px 6px;border-radius:4px">label 2</span>
    <span style="background:#8e44ad;color:white;padding:1px 6px;border-radius:4px">label 3</span>
    (green / orange / purple, as in the video), then red, blue, olive, pink for more labels.
    <span style="background:#495057;color:white;padding:1px 6px;border-radius:4px">[BOS] [SEP] [LABEL]</span> special tokens,
    <span style="background:#ddd;padding:1px 6px;border-radius:4px">text</span> input-text tokens.
    In cost charts, <span style="color:#7a7a7a">**grey = decoder LLM**</span> and <span style="color:#3b6fb6">**blue = encoder**</span>.
    """), kind="info"),
    ])
    return


@app.cell(hide_code=True)
def palette():
    LABEL_COLORS = ["#2a9d8f", "#f18f01", "#8e44ad", "#e4572e", "#3b6fb6", "#6a994e", "#c9184a"]
    SPECIAL_COLOR, TEXT_COLOR = "#495057", "#dddddd"
    DECODER_COLOR, ENCODER_COLOR = "#7a7a7a", "#3b6fb6"
    return (
        DECODER_COLOR,
        ENCODER_COLOR,
        LABEL_COLORS,
        SPECIAL_COLOR,
        TEXT_COLOR,
    )


@app.cell(hide_code=True)
def intro(mo):
    mo.md(r"""
    ## 1. Two ways to build a decision model

    **The popular way (fine-tuned decoder).** Take a Qwen model, fine-tune it so that after reading the prompt it
    *samples a token* that happens to be a class name. The whole generative stack is still there:
    the transformer blocks, a language-model head that scores **every word in a ~150k vocabulary**, and a sampling
    loop that runs once per generated token.

    **The video's question.** The transformer blocks alone already produce *contextualised embeddings*, one per
    input token. If we could read a decision straight off those embeddings, we could drop the decoding machinery and,
    in the GLiNER family, use a much smaller **encoder** (BERT/DeBERTa-sized, a few hundred million parameters)
    that runs on a CPU.
    """)
    return


@app.cell(hide_code=True)
def architecture_diagram(DECODER_COLOR, ENCODER_COLOR, LABEL_COLORS, mo):
    def _box(x, y, w, h, fill, text, fg="white", fs=13):
        return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}"/>'
                f'<text x="{x + w / 2}" y="{y + h / 2 + 5}" text-anchor="middle" fill="{fg}" font-size="{fs}">{text}</text>')

    def _arrow(x1, x2, y):
        return (f'<line x1="{x1}" y1="{y}" x2="{x2 - 6}" y2="{y}" stroke="currentColor" stroke-width="2"/>'
                f'<polygon points="{x2},{y} {x2 - 8},{y - 5} {x2 - 8},{y + 5}" fill="currentColor"/>')

    _y1, _y2 = 40, 175
    _parts = [
        '<text x="10" y="22" font-size="14" font-weight="bold" fill="currentColor">A. Fine-tuned decoder LLM (e.g. Qwen 0.8B-4B)</text>',
        _box(10, _y1, 150, 50, "#bbb", "prompt + text + label list", "black", 12),
        _arrow(160, 190, _y1 + 25),
        _box(190, _y1, 150, 50, DECODER_COLOR, "transformer blocks"),
        _arrow(340, 370, _y1 + 25),
        _box(370, _y1, 150, 50, DECODER_COLOR, "LM head: ~150k logits"),
        _arrow(520, 550, _y1 + 25),
        _box(550, _y1, 110, 50, DECODER_COLOR, "sample token"),
        _arrow(660, 690, _y1 + 25),
        _box(690, _y1, 90, 50, LABEL_COLORS[0], '"refund"'),
        f'<path d="M 605 {_y1 + 50} C 605 {_y1 + 85}, 265 {_y1 + 85}, 265 {_y1 + 52}" fill="none" stroke="{DECODER_COLOR}" stroke-dasharray="5 4" stroke-width="2"/>',
        f'<text x="435" y="{_y1 + 95}" text-anchor="middle" font-size="11" fill="currentColor">loop again for every extra output token</text>',
        '<text x="10" y="157" font-size="14" font-weight="bold" fill="currentColor">B. Encoder with label tokens (GLiNER-style, ~0.1B-0.5B)</text>',
        _box(10, _y2, 150, 50, "#bbb", "[LABEL] refund ... [SEP] text", "black", 11),
        _arrow(160, 190, _y2 + 25),
        _box(190, _y2, 150, 50, ENCODER_COLOR, "encoder (attention)"),
        _arrow(340, 370, _y2 + 25),
        _box(370, _y2, 150, 50, ENCODER_COLOR, "contextual embeddings"),
        _arrow(520, 550, _y2 + 25),
        _box(550, _y2, 110, 50, ENCODER_COLOR, "head: label · text"),
    ]
    for _k, (_lab, _p) in enumerate([("refund", .95), ("outage", .06), ("praise", .02)]):
        _parts.append(_arrow(660, 690, _y2 + 25))
        _parts.append(f'<rect x="690" y="{_y2 - 5 + 21 * _k}" width="{90 * _p + 4}" height="17" rx="3" fill="{LABEL_COLORS[_k]}"/>'
                      f'<text x="{698 + 90 * _p}" y="{_y2 + 8 + 21 * _k}" font-size="11" fill="currentColor">{_lab}</text>')
    _parts.append(f'<text x="435" y="{_y2 + 75}" text-anchor="middle" font-size="11" fill="currentColor">one forward pass, one score per label, no vocabulary, no sampling loop</text>')
    architecture_svg = mo.Html(f'<svg viewBox="0 0 800 260" style="width:100%;max-width:800px">{"".join(_parts)}</svg>')
    architecture_svg
    return


@app.cell(hide_code=True)
def cost_md(mo):
    mo.md(r"""
    ### How much compute does each design use?

    A useful rule of thumb: a transformer forward pass costs about
    $$\text{FLOPs} \approx 2 \cdot N_\text{params} \cdot N_\text{tokens}.$$
    The decoder reads the prompt (instructions, the label list and the text) and then runs **one more pass per generated token**.
    The encoder reads text and label tokens once.
    These are rough assumptions for an order-of-magnitude comparison, not benchmarks. Accuracy is not part of this
    calculation. The video's own benchmark found the fine-tuned GLiNER-style models competitive, with the caveat that the dataset may have leaked into pre-training.
    """)
    return


@app.cell
def cost_controls(mo):
    dec_params = mo.ui.slider(0.8, 4.0, step=0.1, value=0.8, label="decoder params (B)", show_value=True)
    enc_params = mo.ui.slider(0.05, 0.5, step=0.01, value=0.2, label="encoder params (B)", show_value=True)
    n_text_tokens = mo.ui.slider(10, 2000, step=10, value=60, label="text tokens", show_value=True)
    n_labels_cost = mo.ui.slider(2, 50, value=3, label="number of labels", show_value=True)
    prompt_overhead = mo.ui.slider(0, 500, step=10, value=80, label="decoder instruction tokens", show_value=True)
    gen_tokens = mo.ui.slider(1, 20, value=3, label="decoder generated tokens", show_value=True)
    cpu_gflops = mo.ui.slider(20, 1000, step=10, value=200, label="assumed CPU GFLOP/s", show_value=True)
    mo.hstack([
        mo.vstack([dec_params, prompt_overhead, gen_tokens]),
        mo.vstack([enc_params, n_text_tokens, n_labels_cost, cpu_gflops]),
    ], justify="start", gap=2)
    return (
        cpu_gflops,
        dec_params,
        enc_params,
        gen_tokens,
        n_labels_cost,
        n_text_tokens,
        prompt_overhead,
    )


@app.cell
def cost_readout(
    DECODER_COLOR,
    ENCODER_COLOR,
    alt,
    cpu_gflops,
    dec_params,
    enc_params,
    gen_tokens,
    mo,
    n_labels_cost,
    n_text_tokens,
    pd,
    prompt_overhead,
):
    _label_toks = 2 * n_labels_cost.value  # "[LABEL] name" per label, used by both prompts
    _dec_tokens = prompt_overhead.value + _label_toks + n_text_tokens.value
    _dec_flops = 2 * dec_params.value * 1e9 * (_dec_tokens + gen_tokens.value)
    _enc_tokens = 2 + _label_toks + n_text_tokens.value  # [BOS] labels [SEP] text
    _enc_flops = 2 * enc_params.value * 1e9 * _enc_tokens
    _ratio = _dec_flops / _enc_flops
    _df = pd.DataFrame({
        "model": ["decoder LLM", "encoder + label tokens"],
        "GFLOPs": [_dec_flops / 1e9, _enc_flops / 1e9],
    })
    _chart = alt.Chart(_df).mark_bar().encode(
        x=alt.X("GFLOPs:Q", scale=alt.Scale(type="log"), title="GFLOPs per prediction (log scale)"),
        y=alt.Y("model:N", title=None),
        color=alt.Color("model:N", scale=alt.Scale(domain=list(_df.model), range=[DECODER_COLOR, ENCODER_COLOR]), legend=None),
        tooltip=["model", alt.Tooltip("GFLOPs:Q", format=".1f")],
    ).properties(width="container", height=110)
    mo.vstack([
        mo.hstack([
            mo.stat(f"{_dec_flops / 1e9:,.0f} GFLOPs", label="decoder", caption=f"{_dec_tokens} prompt + {gen_tokens.value} generated tokens"),
            mo.stat(f"{_enc_flops / 1e9:,.0f} GFLOPs", label="encoder", caption=f"{_enc_tokens} tokens, one pass"),
            mo.stat(f"{_ratio:,.1f}x", label="ratio", caption="decoder / encoder compute"),
            mo.stat(f"{_dec_flops / (cpu_gflops.value * 1e9) * 1000:,.0f} vs {_enc_flops / (cpu_gflops.value * 1e9) * 1000:,.0f} ms",
                    label="rough CPU time", caption="at the assumed sustained throughput"),
        ]),
        _chart,
        mo.callout(mo.md(f"""
    **Try this.** (1) Push the decoder to 4B: the gap widens to ~{2 * 4e9 * (_dec_tokens + gen_tokens.value) / _enc_flops:,.0f}x.
    (2) Set 50 labels: the label tokens cost *both* models context, which is the price of the label-token trick that the video mentions.
    (3) Make the text 2000 tokens long: the ratio settles towards *decoder params / encoder params*, because text dominates both.
    """), kind="neutral"),
    ])
    return


@app.cell(hide_code=True)
def toy_model_md(mo):
    mo.md(r"""
    ### The toy encoder used in the rest of this notebook

    The next three cells are the whole model. Read them: there is nothing hidden.

    * `embed(token)`: a lookup in the hand-made table below. Special tokens and stop words are zero vectors.
      Unknown words get a deterministic random direction, so the toy "doesn't know" them.
      Capitalised unknown words lean towards *person/place*, a crude stand-in for what a real model learns about names.
    * `encode(...)`: two layers of $H \leftarrow \operatorname{unit}\big(H + \operatorname{softmax}(HH^\top / T)\,H\big)$, attention with query = key = value = $H$.
      Label blocks may attend to themselves and to the text. The text attends only to itself (GLiNER uses full attention; we isolate the labels so each one is an independent decision).
    * the **head**: $\text{logit} = 12\,(\cos(\text{label}, \text{text}) - 0.75)$ followed by a sigmoid (independent labels) or a softmax (mutually exclusive labels).
      In a real model this head is learned. Here it is fixed.
    """)
    return


@app.cell
def toy_vocab(np):
    import hashlib as _hashlib
    import re as _re

    DIMS = ["money", "failure", "pos", "neg", "person", "place", "time", "shipping", "action"]

    # A hand-made 9-dimensional "semantic" embedding table. Every number is readable:
    # e.g. `charged` is mostly money with a hint of negativity.
    _SPEC = {
        # --- topic words ---------------------------------------------------------
        "card": {"money": .8}, "charged": {"money": 1, "neg": .3}, "charge": {"money": 1, "neg": .2},
        "twice": {"neg": .25, "time": .3}, "money": {"money": 1}, "paid": {"money": .9},
        "pay": {"money": .9}, "payment": {"money": 1}, "bill": {"money": .9}, "billing": {"money": 1},
        "invoice": {"money": .9}, "price": {"money": .8}, "refund": {"money": 1, "neg": .25},
        "refunded": {"money": 1}, "back": {"money": .2, "time": .1}, "wants": {"action": .6, "neg": .1},
        "want": {"action": .6, "neg": .1}, "return": {"money": .5, "shipping": .4},
        "outage": {"failure": 1, "neg": .4}, "down": {"failure": .8, "neg": .4},
        "crashed": {"failure": 1, "neg": .5}, "crash": {"failure": 1, "neg": .4},
        "error": {"failure": .9, "neg": .4}, "broken": {"failure": .8, "neg": .5},
        "app": {"failure": .4}, "website": {"failure": .4}, "server": {"failure": .6},
        "login": {"failure": .5}, "service": {"failure": .5}, "offline": {"failure": .9, "neg": .3},
        "praise": {"pos": 1}, "love": {"pos": 1}, "great": {"pos": .9}, "thanks": {"pos": .8},
        "thank": {"pos": .8}, "amazing": {"pos": 1}, "awesome": {"pos": 1}, "happy": {"pos": 1},
        "good": {"pos": .7}, "excellent": {"pos": 1}, "helpful": {"pos": .8}, "customer": {"person": .3},
        "angry": {"neg": 1}, "terrible": {"neg": 1}, "awful": {"neg": 1}, "hate": {"neg": 1},
        "furious": {"neg": 1}, "bad": {"neg": .8}, "annoyed": {"neg": .8}, "sad": {"neg": .7},
        "calm": {"pos": .3}, "again": {"neg": .2, "time": .2},
        "shipping": {"shipping": 1}, "package": {"shipping": 1}, "parcel": {"shipping": 1},
        "delivery": {"shipping": 1}, "delivered": {"shipping": 1}, "arrived": {"action": .4, "shipping": .8, "time": .2},
        "late": {"shipping": .4, "time": .5, "neg": .4}, "order": {"shipping": .6, "money": .3},
        "courier": {"shipping": 1}, "lost": {"shipping": .5, "neg": .5},
        # --- entity words ----------------------------------------------------------
        "person": {"person": 1}, "location": {"place": 1}, "place": {"place": 1}, "city": {"place": 1},
        "date": {"time": 1}, "time": {"time": 1}, "organization": {"person": .4, "place": .4},
        "ada": {"person": .9}, "lovelace": {"person": 1}, "grace": {"person": .9}, "hopper": {"person": .9},
        "alan": {"person": .9}, "turing": {"person": 1}, "marie": {"person": .9}, "curie": {"person": 1},
        "visited": {"action": 1, "place": .2}, "met": {"action": 1, "person": .2}, "new": {"place": .8, "time": .2},
        "york": {"place": 1}, "paris": {"place": 1}, "london": {"place": 1}, "berlin": {"place": 1},
        "tokyo": {"place": 1}, "copenhagen": {"place": 1}, "amsterdam": {"place": 1},
        "may": {"time": .9}, "june": {"time": 1}, "january": {"time": 1}, "monday": {"time": 1},
        "friday": {"time": 1}, "yesterday": {"time": 1}, "today": {"time": 1}, "week": {"time": .9},
        # --- description words -----------------------------------------------------
        "unhappy": {"neg": .9}, "compliment": {"pos": 1},
        "human": {"person": 1}, "name": {"person": .6}, "where": {"place": .8}, "when": {"time": .8},
        "calendar": {"time": 1}, "day": {"time": 1}, "works": {"action": .5, "failure": .3, "pos": .2},
        "not": {"neg": .3}, "someone": {"person": .8}, "moves": {"action": .5, "shipping": .6},
    }
    STOPWORDS = {"my", "got", "the", "a", "an", "in", "on", "at", "to", "of", "and", "i", "it",
                 "me", "for", "is", "was", "this", "that", "with", "your", "you", "has", "have"}
    SPECIALS = ["[BOS]", "[SEP]", "[LABEL]", "[ENT]", "[DESC]"]


    def _vec(spec):
        return np.array([spec.get(d, 0.0) for d in DIMS], dtype=float)


    VOCAB = {w: _vec(s) for w, s in _SPEC.items()}


    def embed(token):
        """Static (layer-0) embedding of one token. Specials & stopwords are zero vectors."""
        if token in SPECIALS:
            return np.zeros(len(DIMS))
        _low = token.lower()
        if _low in VOCAB:
            return VOCAB[_low].copy()
        if _low in STOPWORDS:
            return np.zeros(len(DIMS))
        if _re.fullmatch(r"\d+", _low):  # numbers look like dates in this toy
            return _vec({"time": .7})
        # unknown word: a small, deterministic pseudo-random vector (the toy "doesn't know" it)
        _seed = int(_hashlib.md5(_low.encode()).hexdigest()[:8], 16)
        _v = np.random.default_rng(_seed).normal(0, 1, len(DIMS))
        _v = _v / np.linalg.norm(_v)
        if token[:1].isupper():  # capitalised unknown word: probably a name (person or place)
            _v = 0.4 * _v + _vec({"person": .6, "place": .6})
        return _v


    def tokenize(text):
        """Toy tokenizer: one token per word (real models use sub-word tokens)."""
        return _re.findall(r"[A-Za-z]+|\d+", text)


    def parse_labels(text):
        return [_l.strip() for _l in text.split(",") if _l.strip()]


    return DIMS, STOPWORDS, embed, parse_labels, tokenize


@app.cell
def toy_encoder(STOPWORDS, embed, np, pd, tokenize):
    SCALE, BIAS = 12.0, 0.75  # fixed "classification head": logit = SCALE * (cos - BIAS)


    def build_sequence(text_tokens, labels, marker="[LABEL]", descs=None):
        """[BOS] ([LABEL] name [DESC] description)* [SEP] text  ->  tokens, kinds, groups.

        group = index of the label block a token belongs to (-1 for text / [BOS] / [SEP]).
        """
        _toks, _kinds, _groups = ["[BOS]"], ["special"], [-1]
        for _g, _lab in enumerate(labels):
            _toks.append(marker); _kinds.append("marker"); _groups.append(_g)
            for _w in tokenize(_lab) or [_lab]:
                _toks.append(_w); _kinds.append("label"); _groups.append(_g)
            if descs and descs[_g]:
                _toks.append("[DESC]"); _kinds.append("marker"); _groups.append(_g)
                for _w in tokenize(descs[_g]):
                    _toks.append(_w); _kinds.append("desc"); _groups.append(_g)
        _toks.append("[SEP]"); _kinds.append("special"); _groups.append(-1)
        for _w in text_tokens:
            _toks.append(_w); _kinds.append("text"); _groups.append(-1)
        return _toks, _kinds, np.array(_groups)


    def _unit(M):
        _n = np.linalg.norm(M, axis=-1, keepdims=True)
        return np.where(_n > 1e-9, M / np.maximum(_n, 1e-9), M)


    def encode(tokens, groups, temperature=0.5, n_layers=2, isolate_labels=True):
        """Toy encoder: n_layers of parameter-free self-attention (Q = K = V = H).

        H_{l+1} = unit( H_l + softmax(H_l H_l^T / T) H_l )
        isolate_labels: a label block may look at itself + the text, the text only at itself,
        so adding/removing a label never changes the other labels' scores.
        """
        H = np.stack([embed(_t) for _t in tokens])
        _g = np.asarray(groups)
        allowed = (_g[None, :] == -1) | (_g[None, :] == _g[:, None])
        if not isolate_labels:
            allowed = np.ones_like(allowed)
        layers, attns = [H], []
        for _ in range(n_layers):
            _S = H @ H.T / temperature
            _S = np.where(allowed, _S, -np.inf)
            _S = _S - _S.max(axis=1, keepdims=True)
            _A = np.exp(_S)
            _A = _A / _A.sum(axis=1, keepdims=True)
            H = _unit(H + _A @ H)
            layers.append(H); attns.append(_A)
        return {"tokens": list(tokens), "groups": _g, "layers": layers, "attn": attns, "out": H}


    def label_vectors(enc, kinds, n_labels, use_desc=True):
        """Label representation = mean of the contextualised outputs of its name (+ description) tokens."""
        _keep = {"label", "desc"} if use_desc else {"label"}
        _vs = []
        for _g in range(n_labels):
            _idx = [i for i, k in enumerate(kinds) if enc["groups"][i] == _g and k in _keep]
            _vs.append(_unit(enc["out"][_idx].mean(axis=0)))
        return np.array(_vs)


    def text_vector(enc, kinds):
        _idx = [i for i, k in enumerate(kinds) if k == "text"]
        return _unit(enc["out"][_idx].mean(axis=0))


    def sigmoid(z):
        return 1 / (1 + np.exp(-z))


    def classify(text, labels, mode="sigmoid", descs=None, temperature=0.5, n_layers=2,
                 isolate_labels=True):
        """Whole-text classification with label tokens. Returns a DataFrame (one row per label)."""
        _toks, _kinds, _groups = build_sequence(tokenize(text), labels, "[LABEL]", descs)
        _enc = encode(_toks, _groups, temperature, n_layers, isolate_labels)
        _L = label_vectors(_enc, _kinds, len(labels))
        _cos = _L @ text_vector(_enc, _kinds)
        _logit = SCALE * (_cos - BIAS)
        if mode == "softmax":
            _p = np.exp(_logit - _logit.max()); _p = _p / _p.sum()
        else:
            _p = sigmoid(_logit)
        return pd.DataFrame({"label": labels, "cosine": _cos, "logit": _logit, "prob": _p})


    def score_spans(text, entity_types, max_width=3, agg="mean", temperature=0.5, n_layers=2,
                    length_bonus=0.02):
        """Score every span (up to max_width tokens) against every [ENT] token."""
        _words = tokenize(text)
        _toks, _kinds, _groups = build_sequence(_words, entity_types, "[ENT]")
        _enc = encode(_toks, _groups, temperature, n_layers, True)
        _E = label_vectors(_enc, _kinds, len(entity_types))
        _off = _kinds.index("text")
        _T = _enc["out"][_off:]
        _rows = []
        for _s in range(len(_words)):
            for _w in range(1, max_width + 1):
                _e = _s + _w
                if _e > len(_words):
                    break
                if _words[_s].lower() in STOPWORDS or _words[_e - 1].lower() in STOPWORDS:
                    continue  # spans may not start/end on a function word
                if agg == "mean":
                    _v = _T[_s:_e].mean(axis=0)
                elif agg == "max":
                    _v = _T[_s:_e].max(axis=0)
                else:  # first+last
                    _v = _T[_s] + _T[_e - 1]
                _v = _unit(_v)
                for _k, _ent in enumerate(entity_types):
                    _c = float(_v @ _E[_k]) + length_bonus * (_w - 1)
                    _rows.append({"start": _s, "end": _e, "span": " ".join(_words[_s:_e]),
                                  "entity": _ent, "cosine": _c,
                                  "score": float(sigmoid(SCALE * (_c - BIAS)))})
        return _words, pd.DataFrame(_rows)


    def greedy_decode(span_df, threshold=0.5):
        """Keep the best-scoring spans above threshold that don't overlap an already-kept span."""
        _kept, _used = [], set()
        for _r in span_df[span_df.score >= threshold].sort_values("score", ascending=False).itertuples():
            _cells = set(range(_r.start, _r.end))
            if _cells & _used:
                continue
            _kept.append({"start": _r.start, "end": _r.end, "span": _r.span, "entity": _r.entity,
                          "score": _r.score})
            _used |= _cells
        return sorted(_kept, key=lambda d: d["start"])


    return (
        build_sequence,
        classify,
        encode,
        greedy_decode,
        label_vectors,
        score_spans,
        text_vector,
    )


@app.cell
def embedding_table(DIMS, alt, embed, pd):
    _words = ["card", "charged", "twice", "refund", "outage", "praise", "shipping", "angry", "happy",
              "ada", "lovelace", "visited", "new", "york", "may", "person", "location", "date"]
    _df = pd.DataFrame([{"word": _w, "dim": _d, "value": float(embed(_w)[_i])}
                        for _w in _words for _i, _d in enumerate(DIMS)])
    embedding_table_chart = alt.Chart(_df).mark_rect().encode(
        x=alt.X("dim:N", sort=DIMS, title=None, axis=alt.Axis(orient="top", labelAngle=0)),
        y=alt.Y("word:N", sort=_words, title=None),
        color=alt.Color("value:Q", scale=alt.Scale(scheme="blues", domain=[0, 1]), title="weight"),
        tooltip=["word", "dim", alt.Tooltip("value:Q", format=".2f")],
    ).properties(width="container", height=360, title="Layer-0 embeddings (a slice of the vocabulary)")
    embedding_table_chart
    return


@app.cell(hide_code=True)
def attention_md(mo):
    mo.md(r"""
    ## 2. Contextualised embeddings: every token looks at every other token

    The video's mental model of attention: each token's embedding is **compared with all the other embeddings**, and the
    token's new embedding is a weighted mix of the tokens it found similar. After a few layers, every output "knows" about
    its context.

    Here is the video's input sequence with the labels written *into* the sequence:

    `[BOS] [LABEL] refund [LABEL] outage [LABEL] praise [SEP] my card got charged twice`

    **Click a token** in the explorer (hover previews it). Arcs show where that token's attention goes in the chosen layer:
    thicker arc = more weight. With the default settings, the green `refund` token sends most of its attention to *card* and *charged*, while
    the purple `praise` finds nothing in the text it resembles.
    """)
    return


@app.cell
def attention_controls(mo):
    attn_temperature = mo.ui.slider(0.1, 2.0, step=0.05, value=0.5, label="attention temperature T", show_value=True)
    attn_layer = mo.ui.dropdown(["1", "2"], value="1", label="layer")
    attn_isolate = mo.ui.checkbox(value=True, label="isolate label blocks (labels can't see each other)")
    attn_text = mo.ui.text(value="my card got charged twice", label="text", full_width=True)
    mo.vstack([attn_text, mo.hstack([attn_temperature, attn_layer, attn_isolate], justify="start", gap=2)])
    return attn_isolate, attn_layer, attn_temperature, attn_text


@app.cell
def attention_sequence(
    LABEL_COLORS,
    SPECIAL_COLOR,
    TEXT_COLOR,
    attn_isolate,
    attn_temperature,
    attn_text,
    build_sequence,
    encode,
    tokenize,
):
    _labels = ["refund", "outage", "praise"]
    attn_tokens, attn_kinds, _groups = build_sequence(tokenize(attn_text.value), _labels, "[LABEL]")
    attn_enc = encode(attn_tokens, _groups, attn_temperature.value, 2, attn_isolate.value)
    attn_colors = [
        SPECIAL_COLOR if _k in ("special", "marker") else (TEXT_COLOR if _k == "text" else LABEL_COLORS[_g % len(LABEL_COLORS)])
        for _k, _g in zip(attn_kinds, _groups)
    ]
    return attn_colors, attn_enc, attn_tokens


@app.cell
def attention_widget_class(anywidget, traitlets):
    class AttentionExplorer(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const NS = "http://www.w3.org/2000/svg";
      const wrap = document.createElement("div");
      const cap = document.createElement("div");
      cap.style.cssText = "font:13px sans-serif;margin-top:4px;min-height:1.4em";
      const svg = document.createElementNS(NS, "svg");
      wrap.appendChild(svg); el.append(wrap, cap);
      const mk = (tag, attrs, parent = svg) => { const e = document.createElementNS(NS, tag);
        for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
      let hover = -1;
      function layout() {
        const toks = model.get("tokens"); let x = 10; const xs = [];
        toks.forEach(t => { const w = Math.max(34, 8.2 * t.length + 16); xs.push([x, w]); x += w + 6; });
        return { xs, W: x + 4 };
      }
      function draw() {
        const toks = model.get("tokens"), cols = model.get("colors"), A = model.get("attn");
        const sel = hover >= 0 ? hover : model.get("selected");
        const { xs, W } = layout(); const H = 230, Y = 180;
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
        Object.assign(svg.style, { width: "100%", minWidth: Math.min(W, 700) + "px", userSelect: "none" });
        svg.innerHTML = "";
        const cx = i => xs[i][0] + xs[i][1] / 2;
        if (sel >= 0 && sel < toks.length) {
          const row = A[sel];
          row.forEach((w, j) => {
            if (w < 0.004) return;
            const x1 = cx(sel), x2 = cx(j), h = j === sel ? 40 : Math.min(150, 25 + Math.abs(x2 - x1) * 0.35);
            const d = j === sel
              ? `M ${x1 - 8} ${Y - 2} C ${x1 - 30} ${Y - h}, ${x1 + 30} ${Y - h}, ${x1 + 8} ${Y - 2}`
              : `M ${x1} ${Y - 2} Q ${(x1 + x2) / 2} ${Y - 2 * h} ${x2} ${Y - 2}`;
            mk("path", { d, fill: "none", stroke: cols[sel] === "#dddddd" ? "#888" : cols[sel],
              "stroke-width": 1 + 14 * w, opacity: 0.25 + 0.75 * Math.min(1, w * 2.5), "stroke-linecap": "round" });
          });
        }
        toks.forEach((t, i) => {
          const [x, w] = xs[i];
          const g = mk("g", {}); g.style.cursor = "pointer";
          const isSel = i === sel;
          mk("rect", { x, y: Y, width: w, height: 30, rx: 7, fill: cols[i],
            stroke: isSel ? "currentColor" : "none", "stroke-width": 2.5 }, g);
          const tx = mk("text", { x: x + w / 2, y: Y + 20, "text-anchor": "middle", "font-size": 13,
            "font-family": "monospace", fill: cols[i] === "#dddddd" ? "#222" : "white" }, g);
          tx.textContent = t;
          if (sel >= 0) {
            const lab = mk("text", { x: x + w / 2, y: Y + 46, "text-anchor": "middle", "font-size": 11, fill: "currentColor" }, g);
            lab.textContent = (A[sel][i] * 100).toFixed(0) + "%";
          }
          g.addEventListener("pointerenter", () => { hover = i; draw(); });
          g.addEventListener("pointerleave", () => { hover = -1; draw(); });
          g.addEventListener("click", () => { model.set("selected", i); model.save_changes(); });
        });
        svg.setAttribute("height", H + 20);
        if (sel >= 0) {
          const top = A[sel].map((w, j) => [w, j]).filter(p => p[1] !== sel).sort((a, b) => b[0] - a[0]).slice(0, 3);
          cap.innerHTML = `<b>${toks[sel]}</b> ${hover >= 0 ? "(hover)" : "(selected)"} attends to itself ${(A[sel][sel] * 100).toFixed(0)}%, then ` +
            top.map(p => `<b>${toks[p[1]]}</b> ${(p[0] * 100).toFixed(0)}%`).join(", ");
        }
      }
      ["tokens", "colors", "attn", "selected"].forEach(k => model.on("change:" + k, draw));
      draw();
    }
    export default { render };
    """
        tokens = traitlets.List(traitlets.Unicode()).tag(sync=True)
        colors = traitlets.List(traitlets.Unicode()).tag(sync=True)
        attn = traitlets.List(traitlets.List(traitlets.Float())).tag(sync=True)
        selected = traitlets.Int(2).tag(sync=True)

    return (AttentionExplorer,)


@app.cell
def attention_widget(
    AttentionExplorer,
    attn_colors,
    attn_enc,
    attn_layer,
    attn_tokens,
    mo,
    np,
):
    attention_explorer = mo.ui.anywidget(AttentionExplorer(
        tokens=attn_tokens,
        colors=attn_colors,
        attn=np.round(attn_enc["attn"][int(attn_layer.value) - 1], 4).tolist(),
        selected=attn_tokens.index("refund"),
    ))
    attention_explorer
    return (attention_explorer,)


@app.cell
def attention_readout(
    DIMS,
    TEXT_COLOR,
    alt,
    attention_explorer,
    attn_colors,
    attn_enc,
    attn_layer,
    attn_tokens,
    mo,
    pd,
):
    _A = attn_enc["attn"][int(attn_layer.value) - 1]
    _sel = min(attention_explorer.value["selected"], len(attn_tokens) - 1)
    _names = [f"{_i:02d} {_t}" for _i, _t in enumerate(attn_tokens)]
    _hm = pd.DataFrame([{"query": _names[_i], "key": _names[_j], "weight": float(_A[_i, _j])}
                        for _i in range(len(_names)) for _j in range(len(_names))])
    _heat = alt.Chart(_hm).mark_rect().encode(
        x=alt.X("key:N", sort=_names, title="attends to (key)"),
        y=alt.Y("query:N", sort=_names, title="token (query)"),
        color=alt.Color("weight:Q", scale=alt.Scale(scheme="greens"), title="attention"),
        tooltip=["query", "key", alt.Tooltip("weight:Q", format=".2f")],
    ).properties(width=380, height=380, title=f"Attention matrix, layer {attn_layer.value}")
    _heat = _heat + alt.Chart(_hm[_hm["query"] == _names[_sel]]).mark_rect(fill=None, stroke="black", strokeWidth=1.5).encode(
        x=alt.X("key:N", sort=_names), y=alt.Y("query:N", sort=_names))
    _before = attn_enc["layers"][0][_sel]
    _after = attn_enc["layers"][-1][_sel]
    _bd = pd.DataFrame(
        [{"dim": _d, "stage": "layer 0 (static)", "value": float(_before[_i])} for _i, _d in enumerate(DIMS)]
        + [{"dim": _d, "stage": "after 2 layers", "value": float(_after[_i])} for _i, _d in enumerate(DIMS)]
    )
    _bars = alt.Chart(_bd).mark_bar().encode(
        x=alt.X("value:Q", title="value"),
        y=alt.Y("stage:N", title=None),
        row=alt.Row("dim:N", sort=DIMS, title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
        color=alt.Color("stage:N", scale=alt.Scale(range=["#bbbbbb", attn_colors[_sel] if attn_colors[_sel] != TEXT_COLOR else "#555"]), title=None),
    ).properties(width=220, height=22, title=f"Embedding of '{attn_tokens[_sel]}' before vs after attention")
    mo.hstack([_heat, _bars], justify="start", gap=2)
    return


@app.cell(hide_code=True)
def attention_try(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Click **`[LABEL]`** before *outage*. It is a zero vector, so in layer 1 it attends *uniformly* to everything it may see, and its output becomes a summary of the text.
       (Real GLiNER reads the label decision off this marker token, and it can do that because the marker's weights are learned. In our weight-free toy, we read the label's name tokens instead.)
    2. Drag the **temperature** to 0.1: attention becomes nearly one-hot (each token mostly looks at itself or its nearest neighbor). At 2.0 it smears out across all tokens.
    3. Untick **isolate label blocks**: now `refund` can also look at `outage` and `praise`. Then switch to layer 2 and watch the text tokens' rows in the heatmap change.
    4. Change the text to *"the app crashed and login is down"* and click `outage`.
    """), kind="neutral")
    return


@app.cell(hide_code=True)
def labels_md(mo):
    mo.md(r"""
    ## 3. The trick: labels are just more input tokens

    Once the labels sit in the input sequence, each label gets its own **contextualised output embedding**.
    A small **classification head** compares it with the text and returns one score per label:

    $$\text{logit}_\ell = 12\,\big(\cos(\mathbf h_\ell, \bar{\mathbf h}_\text{text}) - 0.75\big)$$

    * **independent labels** → $p_\ell = \sigma(\text{logit}_\ell)$. Each label is its own yes/no question, so several can be "on" (multi-label).
    * **mutually exclusive labels** → $p = \operatorname{softmax}(\text{logit})$. Exactly one class wins, like a scikit-learn classifier.

    Because the label is *input* rather than a fixed output neuron, you can **rename or add a label without retraining**
    (the video's *shipping* example), and you can ask **several questions at once**, e.g. a *topic* and a *mood*.
    """)
    return


@app.cell
def classify_controls(mo):
    cls_text = mo.ui.text(value="my card got charged twice", label="text", full_width=True)
    cls_labels = mo.ui.text(value="refund, outage, praise", label="topic labels (comma-separated)", full_width=True)
    cls_moods = mo.ui.text(value="angry, happy", label="mood labels (second, independent head)", full_width=True)
    cls_mode = mo.ui.radio(["sigmoid (independent)", "softmax (mutually exclusive)"], value="sigmoid (independent)", label="normalisation")
    cls_temperature = mo.ui.slider(0.1, 2.0, step=0.05, value=0.5, label="attention temperature", show_value=True)
    mo.vstack([cls_text, cls_labels, cls_moods, mo.hstack([cls_mode, cls_temperature], justify="start", gap=2)])
    return cls_labels, cls_mode, cls_moods, cls_temperature, cls_text


@app.cell
def classify_readout(
    LABEL_COLORS,
    alt,
    classify,
    cls_labels,
    cls_mode,
    cls_moods,
    cls_temperature,
    cls_text,
    mo,
    parse_labels,
    pd,
    tokenize,
):
    _mode = "softmax" if cls_mode.value.startswith("softmax") else "sigmoid"

    def _prob_chart(df, title, offset=0):
        _d = df.assign(color=[LABEL_COLORS[(offset + _i) % len(LABEL_COLORS)] for _i in range(len(df))])
        _base = alt.Chart(_d).encode(y=alt.Y("label:N", sort=list(_d.label), title=None))
        _bars = _base.mark_bar().encode(
            x=alt.X("prob:Q", scale=alt.Scale(domain=[0, 1]), title="probability"),
            color=alt.Color("color:N", scale=None),
            tooltip=["label", alt.Tooltip("cosine:Q", format=".3f"), alt.Tooltip("logit:Q", format=".2f"), alt.Tooltip("prob:Q", format=".3f")],
        )
        _txt = _base.mark_text(align="left", dx=4).encode(x="prob:Q", text=alt.Text("prob:Q", format=".2f"))
        _rule = alt.Chart(pd.DataFrame({"x": [0.5]})).mark_rule(strokeDash=[4, 3], color="#999").encode(x="x:Q")
        return (_bars + _txt + _rule).properties(width=330, height=26 * len(_d) + 10, title=title)

    _topics = parse_labels(cls_labels.value)
    _moods = parse_labels(cls_moods.value)
    _charts = []
    topic_scores = None
    if _topics and tokenize(cls_text.value):
        topic_scores = classify(cls_text.value, _topics, _mode, temperature=cls_temperature.value)
        _charts.append(_prob_chart(topic_scores, f"topic head ({_mode})"))
    if _moods and tokenize(cls_text.value):
        _mood_scores = classify(cls_text.value, _moods, _mode, temperature=cls_temperature.value)
        _charts.append(_prob_chart(_mood_scores, f"mood head ({_mode})", offset=len(_topics)))
    _out = [mo.hstack(_charts, justify="start", gap=2)] if _charts else [mo.md("_Type some text and at least one label._")]
    if topic_scores is not None:
        _win = topic_scores.sort_values("prob", ascending=False).iloc[0]
        _n_tok = 2 + 2 * (len(_topics) + len(_moods)) + len(tokenize(cls_text.value))
        _out.append(mo.md(
            f"Top topic: **{_win.label}** (p = {_win.prob:.2f}). The sequence has **{_n_tok} tokens**, of which "
            f"{2 * (len(_topics) + len(_moods))} are label tokens: that is the context cost of the trick."))
    mo.vstack(_out)
    return (topic_scores,)


@app.cell
def classify_table(mo, topic_scores):
    mo.accordion({"Show the numbers (cosine → logit → probability)": topic_scores.round(3) if topic_scores is not None else mo.md("—")})
    return


@app.cell(hide_code=True)
def labels_try(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. **Change a label, no retraining:** add `shipping` to the topics, then change the text to *"my package arrived late"*. The new label wins, though no model weights changed.
    2. **Sigmoid vs softmax:** with the text *"thanks, the app crashed again"* and labels `praise, outage, refund`, sigmoid gives outage ≈ 0.9 and still leaves praise ≈ 0.3
       (each label is judged on its own). Softmax makes them compete, so praise drops to ≈ 0.04. Which one fits your problem?
    3. **Zero-shot limits:** add the label `chargeback`. The toy has never seen this word, so its embedding is random and it scores ≈ 0.
       Section 5 fixes this with a *description*.
    4. Notice that the **mood head** runs on the same text in the same forward pass. It is a second, independent decision.
    """), kind="neutral")
    return


@app.cell(hide_code=True)
def ner_md(mo):
    mo.md(r"""
    ## 4. Per entity: from labels to spans (named entity recognition)

    Same encoder, new sentence: **"Ada Lovelace visited New York in May."** Now we don't want one label for the whole
    text. We want to find *which pieces* of the text are a **person**, a **location** or a **date**.

    The special token changes from `[LABEL]` to `[ENT]`:
    `[BOS] [ENT] person [ENT] location [ENT] date [SEP] Ada Lovelace visited New York in May`

    Then, GLiNER-style:

    1. **Enumerate spans**: every run of 1 … *max width* consecutive tokens (*Ada*, *Ada Lovelace*, *Lovelace visited*, …).
    2. **Embed each span** by aggregating its tokens' contextualised outputs (mean, first+last, or max).
    3. **Score** every (span, entity type) pair with the same head: $\sigma\big(12(\cos(\mathbf h_\text{span}, \mathbf h_\text{ent}) + \beta\,(w-1) - 0.75)\big)$,
       where $\beta$ is a small toy *length bonus*. A trained model learns such preferences.
    4. **Decode greedily**: keep the highest-scoring span above the threshold, discard spans that overlap it, and repeat.

    (Toy rule: spans may not start or end on a stop word such as *in*. Real models learn this as well.)
    """)
    return


@app.cell
def ner_controls(mo):
    ner_text = mo.ui.text(value="Ada Lovelace visited New York in May", label="sentence", full_width=True)
    ner_types = mo.ui.text(value="person, location, date", label="entity types", full_width=True)
    ner_width = mo.ui.slider(1, 5, value=3, label="max span width", show_value=True)
    ner_agg = mo.ui.dropdown(["mean", "first+last", "max"], value="mean", label="span aggregation")
    ner_threshold = mo.ui.slider(0.5, 0.99, step=0.01, value=0.9, label="threshold", show_value=True)
    ner_bonus = mo.ui.slider(0.0, 0.08, step=0.005, value=0.02, label="length bonus β", show_value=True)
    mo.vstack([ner_text, ner_types, mo.hstack([ner_width, ner_agg, ner_threshold, ner_bonus], justify="start", gap=1.5)])
    return ner_agg, ner_bonus, ner_text, ner_threshold, ner_types, ner_width


@app.cell
def ner_compute(
    LABEL_COLORS,
    greedy_decode,
    mo,
    ner_agg,
    ner_bonus,
    ner_text,
    ner_threshold,
    ner_types,
    ner_width,
    parse_labels,
    score_spans,
):
    _types = parse_labels(ner_types.value) or ["person"]
    ner_words, ner_spans = score_spans(ner_text.value, _types, ner_width.value, ner_agg.value,
                                        length_bonus=ner_bonus.value)
    ner_pred = greedy_decode(ner_spans, ner_threshold.value) if len(ner_spans) else []
    ner_type_colors = {_t: LABEL_COLORS[_i % len(LABEL_COLORS)] for _i, _t in enumerate(_types)}
    mo.md(f"**{len(ner_spans) // max(len(_types), 1)} candidate spans × {len(_types)} entity types = {len(ner_spans)} scores** "
          f"→ {len(ner_pred)} entities kept: " + ", ".join(f"*{_p['span']}* → **{_p['entity']}**" for _p in ner_pred))
    return ner_pred, ner_spans, ner_type_colors, ner_words


@app.cell
def span_widget_class(anywidget, traitlets):
    class SpanHighlighter(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const box = document.createElement("div");
      box.style.cssText = "font:18px/2.6 sans-serif;padding:14px 8px;position:relative";
      const tip = document.createElement("div");
      tip.style.cssText = "position:absolute;pointer-events:none;background:#222;color:#fff;font:12px sans-serif;padding:4px 8px;border-radius:5px;display:none;white-space:nowrap;z-index:5";
      el.style.position = "relative";
      el.append(box, tip);
      function show(evt, html) {
        tip.innerHTML = html; tip.style.display = "block";
        const r = el.getBoundingClientRect();
        tip.style.left = (evt.clientX - r.left + 12) + "px"; tip.style.top = (evt.clientY - r.top - 34) + "px";
      }
      function draw() {
        const words = model.get("words"), spans = model.get("spans"), best = model.get("best");
        box.innerHTML = "";
        let i = 0;
        while (i < words.length) {
          const sp = spans.find(s => s.start === i);
          if (sp) {
            const m = document.createElement("mark");
            m.style.cssText = `background:${sp.color}33;border:2px solid ${sp.color};border-radius:7px;padding:2px 6px;margin:0 3px;color:inherit`;
            m.innerHTML = words.slice(sp.start, sp.end).join(" ") +
              ` <span style="background:${sp.color};color:white;font-size:11px;padding:1px 5px;border-radius:4px;vertical-align:middle">${sp.entity}</span>`;
            m.onmousemove = e => show(e, `<b>${words.slice(sp.start, sp.end).join(" ")}</b> → ${sp.entity}: score ${sp.score.toFixed(3)}`);
            m.onmouseleave = () => tip.style.display = "none";
            box.appendChild(m); i = sp.end;
          } else {
            const s = document.createElement("span");
            s.textContent = words[i]; s.style.cssText = "margin:0 4px;border-bottom:1px dashed #aaa";
            const b = best[i];
            s.onmousemove = e => show(e, b ? `best span here: <b>${b.span}</b> → ${b.entity} ${b.score.toFixed(3)} (below threshold or overlapping)` : "no candidate span");
            s.onmouseleave = () => tip.style.display = "none";
            box.appendChild(s); i++;
          }
        }
      }
      ["words", "spans", "best"].forEach(k => model.on("change:" + k, draw));
      draw();
    }
    export default { render };
    """
        words = traitlets.List(traitlets.Unicode()).tag(sync=True)
        spans = traitlets.List(traitlets.Dict()).tag(sync=True)
        best = traitlets.List(traitlets.Dict()).tag(sync=True)

    return (SpanHighlighter,)


@app.cell
def span_widget(
    SpanHighlighter,
    mo,
    ner_pred,
    ner_spans,
    ner_type_colors,
    ner_words,
):
    def _best_for(i):
        _c = ner_spans[(ner_spans.start <= i) & (ner_spans.end > i)]
        if not len(_c):
            return {}
        _r = _c.sort_values("score", ascending=False).iloc[0]
        return {"span": _r.span, "entity": _r.entity, "score": float(_r.score)}

    span_highlighter = mo.ui.anywidget(SpanHighlighter(
        words=ner_words,
        spans=[{**_p, "score": float(_p["score"]), "color": ner_type_colors[_p["entity"]]} for _p in ner_pred],
        best=[_best_for(_i) for _i in range(len(ner_words))],
    ))
    span_highlighter
    return


@app.cell
def ner_heatmap(alt, ner_pred, ner_spans, ner_threshold, ner_type_colors):
    _kept = {(_p["start"], _p["end"], _p["entity"]) for _p in ner_pred}
    _d = ner_spans.assign(
        span_id=[f"{_s:02d}-{_e:02d} {_t}" for _s, _e, _t in zip(ner_spans.start, ner_spans.end, ner_spans.span)],
        kept=[(_s, _e, _n) in _kept for _s, _e, _n in zip(ner_spans.start, ner_spans.end, ner_spans.entity)],
    )
    _order = list(dict.fromkeys(_d.span_id))
    _base = alt.Chart(_d).encode(
        y=alt.Y("span_id:N", sort=_order, title="candidate span (start-end text)"),
        x=alt.X("entity:N", sort=list(ner_type_colors), title=None, axis=alt.Axis(orient="top", labelAngle=0)),
    )
    _rect = _base.mark_rect().encode(
        color=alt.Color("score:Q", scale=alt.Scale(scheme="viridis", domain=[0, 1]), title="score"),
        tooltip=["span", "entity", alt.Tooltip("cosine:Q", format=".3f"), alt.Tooltip("score:Q", format=".3f")],
    )
    _mark = _base.transform_filter("datum.kept").mark_rect(fill=None, stroke="#e4572e", strokeWidth=3)
    _txt = _base.mark_text(fontSize=10).encode(
        text=alt.Text("score:Q", format=".2f"),
        color=alt.condition(f"datum.score >= {ner_threshold.value}", alt.value("black"), alt.value("white")),
    )
    (_rect + _mark + _txt).properties(width=360, height=max(150, 17 * len(_order)),
                                     title="Every span × every entity type (red box = kept by greedy decoding)")
    return


@app.cell(hide_code=True)
def ner_try(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Set **max span width = 1**. *New York* can't exist any more, so you get *New* and *York* as two separate locations, and *Ada* and *Lovelace* as two persons. Spans are what make multi-word entities possible.
    2. Set **β = 0** and look at the heatmap: *Ada*, *Lovelace* and *Ada Lovelace* score almost the same.
       The tiny length bonus decides the tie. Now push β to 0.08: *visited New York* starts to swallow the verb.
    3. Switch the aggregation to **first+last** or **max** and watch which spans climb.
    4. Add the entity type `organization`, or try *"Grace Hopper met Alan Turing in London on Monday"*. The toy has never seen *Hopper*; only its capital letter makes it look like a name.
    5. Raise the **threshold** to 0.96: which entity disappears first?
    """), kind="neutral")
    return


@app.cell(hide_code=True)
def desc_md(mo):
    mo.md(r"""
    ## 5. Another trick: give each label a description

    One word can be a thin description of a class. So, as in the video, append a **description** after each label behind a new special token:

    `[BOS] [LABEL] refund [DESC] wants money back [LABEL] outage [DESC] service is down … [SEP] my card got charged twice`

    The label's final embedding is now shaped by its description *and* by the text. You hope its attention goes to the
    **description of its own label** plus the **relevant text tokens**. A real model may need fine-tuning to learn that. In our toy, the label
    representation is the mean of the outputs of the name and description tokens.
    """)
    return


@app.cell
def desc_controls(mo):
    desc_text = mo.ui.text(value="my card got charged twice", label="text", full_width=True)
    desc_spec = mo.ui.text_area(
        value="refund: wants money back\noutage: service is down\npraise: happy customer",
        label="one label per line, `name: description`", full_width=True, rows=4)
    desc_on = mo.ui.switch(value=True, label="add [DESC] tokens")
    mo.vstack([desc_text, desc_spec, desc_on])
    return desc_on, desc_spec, desc_text


@app.cell
def desc_parse(desc_spec, mo):
    desc_labels, desc_descs = [], []
    for _line in desc_spec.value.splitlines():
        if not _line.strip():
            continue
        _name, _, _desc = _line.partition(":")
        desc_labels.append(_name.strip() or "?")
        desc_descs.append(_desc.strip())
    desc_focus = mo.ui.dropdown(desc_labels, value=desc_labels[0] if desc_labels else None, label="inspect label")
    desc_focus
    return desc_descs, desc_focus, desc_labels


@app.cell
def desc_readout(
    DIMS,
    LABEL_COLORS,
    alt,
    build_sequence,
    classify,
    desc_descs,
    desc_focus,
    desc_labels,
    desc_on,
    desc_text,
    encode,
    label_vectors,
    mo,
    pd,
    text_vector,
    tokenize,
):
    mo.stop(not desc_labels or not tokenize(desc_text.value), mo.md("_Need text and at least one label._"))
    _without = classify(desc_text.value, desc_labels, "sigmoid").assign(version="label name only")
    _with = classify(desc_text.value, desc_labels, "sigmoid", descs=desc_descs).assign(version="name + [DESC]")
    _both = pd.concat([_without, _with])
    _score_chart = alt.Chart(_both).mark_bar().encode(
        y=alt.Y("version:N", title=None, sort=["label name only", "name + [DESC]"]),
        x=alt.X("prob:Q", scale=alt.Scale(domain=[0, 1]), title="sigmoid probability"),
        row=alt.Row("label:N", sort=desc_labels, title=None, header=alt.Header(labelAngle=0, labelAlign="left")),
        color=alt.Color("label:N", scale=alt.Scale(domain=desc_labels, range=LABEL_COLORS[:len(desc_labels)]), legend=None),
        opacity=alt.Opacity("version:N", scale=alt.Scale(domain=["label name only", "name + [DESC]"], range=[0.35, 1.0]), legend=None),
        tooltip=["label", "version", alt.Tooltip("cosine:Q", format=".3f"), alt.Tooltip("prob:Q", format=".3f")],
    ).properties(width=260, height=40, title="Score without (faded) vs with (solid) description")

    # Inspect one label: its embedding shift and where its attention goes.
    _g = desc_labels.index(desc_focus.value) if desc_focus.value in desc_labels else 0
    _descs = desc_descs if desc_on.value else None
    _toks, _kinds, _groups = build_sequence(tokenize(desc_text.value), desc_labels, "[LABEL]", _descs)
    _enc = encode(_toks, _groups)
    _enc0_toks, _enc0_kinds, _enc0_groups = build_sequence(tokenize(desc_text.value), desc_labels, "[LABEL]", None)
    _enc0 = encode(_enc0_toks, _enc0_groups)
    _v0 = label_vectors(_enc0, _enc0_kinds, len(desc_labels))[_g]
    _v1 = label_vectors(_enc, _kinds, len(desc_labels))[_g]
    _tv = text_vector(_enc, _kinds)
    _shift = pd.DataFrame(
        [{"dim": _d, "vector": "label (name only)", "value": float(_v0[_i])} for _i, _d in enumerate(DIMS)]
        + [{"dim": _d, "vector": "label (with desc)", "value": float(_v1[_i])} for _i, _d in enumerate(DIMS)]
        + [{"dim": _d, "vector": "text (pooled)", "value": float(_tv[_i])} for _i, _d in enumerate(DIMS)]
    )
    _shift_chart = alt.Chart(_shift).mark_bar().encode(
        x=alt.X("dim:N", sort=DIMS, title=None),
        xOffset=alt.XOffset("vector:N", sort=["label (name only)", "label (with desc)", "text (pooled)"]),
        y=alt.Y("value:Q", title="component"),
        color=alt.Color("vector:N", scale=alt.Scale(domain=["label (name only)", "label (with desc)", "text (pooled)"],
                                                     range=["#cccccc", LABEL_COLORS[_g % len(LABEL_COLORS)], "#555555"]), title=None),
    ).properties(width=420, height=200, title=f"'{desc_labels[_g]}' output embedding vs pooled text")

    _name_idx = [_i for _i, (_k, _gg) in enumerate(zip(_kinds, _groups)) if _k == "label" and _gg == _g]
    _att = _enc["attn"][0][_name_idx].mean(axis=0)
    _att_df = pd.DataFrame({
        "pos": [f"{_i:02d} {_t}" for _i, _t in enumerate(_toks)],
        "weight": _att,
        "kind": [("own description" if (_k == "desc" and _gg == _g) else
                  "own label" if (_gg == _g) else
                  "text" if _k == "text" else "other") for _k, _gg in zip(_kinds, _groups)],
    })
    _att_chart = alt.Chart(_att_df).mark_bar().encode(
        x=alt.X("pos:N", sort=list(_att_df.pos), title=None, axis=alt.Axis(labelAngle=-45)),
        y=alt.Y("weight:Q", title="attention (layer 1)"),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["own label", "own description", "text", "other"],
                                                   range=[LABEL_COLORS[_g % len(LABEL_COLORS)], "#f4a261", "#555555", "#dddddd"]), title=None),
        tooltip=["pos", alt.Tooltip("weight:Q", format=".3f"), "kind"],
    ).properties(width=520, height=200, title=f"Where '{desc_labels[_g]}' attends{' (descriptions on)' if desc_on.value else ' (descriptions off)'}")
    _cos0, _cos1 = float(_v0 @ text_vector(_enc0, _enc0_kinds)), float(_v1 @ _tv)
    mo.vstack([
        _score_chart,
        mo.hstack([
            mo.stat(f"{_cos0:.3f}", label="cos(label, text), name only"),
            mo.stat(f"{_cos1:.3f}", label=f"cos(label, text), {'with' if desc_on.value else 'without'} description",
                    direction="increase" if _cos1 > _cos0 + 1e-6 else ("decrease" if _cos1 < _cos0 - 1e-6 else None)),
        ], justify="start", gap=2),
        mo.hstack([_shift_chart, _att_chart], justify="start", gap=1),
    ])
    return


@app.cell(hide_code=True)
def desc_try(mo):
    mo.callout(mo.md(r"""
    **Try this**

    1. Rename `refund` to **`chargeback`**, a word the toy has never seen. Without the description it scores ≈ 0. With *"wants money back"* it climbs to ≈ 0.7.
       The description supplies the meaning that the name lacks.
    2. Inspect `refund` and flip the switch. Its attention now spreads over its own description (*wants*, *money*, *back*) as well as *card* and *charged*.
    3. Write a *bad* description, e.g. `refund: happy customer`, and watch the label's embedding move towards *pos* and its score drop (≈ 0.93 → 0.80). A misleading description hurts.
    """), kind="neutral")
    return


@app.cell(hide_code=True)
def sandbox_md(mo):
    mo.md(r"""
    ## 6. Sandbox: your own decision model

    Write any text, any labels and any entity types. One "encoder pass" per head, no retraining, no decoder.
    The toy only knows its small vocabulary (open the `toy_vocab` cell to see it, or add words there).
    """)
    return


@app.cell
def sandbox_controls(mo):
    sb_text = mo.ui.text_area(value="Grace Hopper was furious: the app crashed in London on Friday and she wants her money back",
                              label="text", full_width=True, rows=2)
    sb_labels = mo.ui.text(value="refund, outage, praise, shipping", label="labels", full_width=True)
    sb_exclusive = mo.ui.switch(value=False, label="labels are mutually exclusive (softmax)")
    sb_entities = mo.ui.text(value="person, location, date", label="entity types", full_width=True)
    sb_threshold = mo.ui.slider(0.5, 0.99, step=0.01, value=0.9, label="entity threshold", show_value=True)
    mo.vstack([sb_text, sb_labels, sb_exclusive, sb_entities, sb_threshold])
    return sb_entities, sb_exclusive, sb_labels, sb_text, sb_threshold


@app.cell
def sandbox_readout(
    LABEL_COLORS,
    SpanHighlighter,
    alt,
    classify,
    greedy_decode,
    mo,
    parse_labels,
    sb_entities,
    sb_exclusive,
    sb_labels,
    sb_text,
    sb_threshold,
    score_spans,
    tokenize,
):
    _labels = parse_labels(sb_labels.value)
    _types = parse_labels(sb_entities.value)
    mo.stop(not tokenize(sb_text.value), mo.md("_Type some text._"))
    _parts = []
    if _labels:
        _s = classify(sb_text.value, _labels, "softmax" if sb_exclusive.value else "sigmoid")
        _d = _s.assign(color=[LABEL_COLORS[_i % len(LABEL_COLORS)] for _i in range(len(_s))])
        _parts.append(alt.Chart(_d).mark_bar().encode(
            y=alt.Y("label:N", sort=list(_d.label), title=None),
            x=alt.X("prob:Q", scale=alt.Scale(domain=[0, 1]), title="probability"),
            color=alt.Color("color:N", scale=None),
            tooltip=["label", alt.Tooltip("prob:Q", format=".3f")],
        ).properties(width="container", height=26 * len(_d) + 10, title="document labels"))
    if _types:
        _w, _spans = score_spans(sb_text.value, _types)
        _pred = greedy_decode(_spans, sb_threshold.value) if len(_spans) else []
        _cols = {_t: LABEL_COLORS[_i % len(LABEL_COLORS)] for _i, _t in enumerate(_types)}
        _parts.append(mo.ui.anywidget(SpanHighlighter(
            words=_w,
            spans=[{**_p, "score": float(_p["score"]), "color": _cols[_p["entity"]]} for _p in _pred],
            best=[{} for _ in _w],
        )))
    mo.vstack(_parts)
    return


@app.cell(hide_code=True)
def quiz_md(mo):
    mo.md(r"""
    ### Quiz: predict, then reveal

    Commit to your answers **before** you flip the reveal switch. The answers come from the same toy model, so you can
    reason from the embedding table (section 1) and the attention explorer (section 2).
    """)
    return


@app.cell
def quiz_controls(mo):
    quiz = mo.ui.dictionary({
        "q1": mo.ui.dropdown(["refund", "outage", "praise"], label='Q1. "the website is down and I want my money back", sigmoid. Which label has the highest score?'),
        "q2": mo.ui.dropdown(["0", "1", "2", "3"], label="Q2. Same text, sigmoid. How many of the three labels score above 0.5?"),
        "q3": mo.ui.dropdown(["0", "1", "2", "3", "4"], label='Q3. NER on "Marie Curie moved to Paris in June" (person/location/date), max width 3, threshold 0.95. How many entities are kept?'),
        "q4": mo.ui.dropdown(["yes", "no"], label="Q4. Does the description 'wants money back' lift the unknown label `chargeback` above 0.5 on \"my card got charged twice\"?"),
        "q5": mo.ui.dropdown(["sigmoid", "softmax"], label="Q5. A ticket can be about billing AND an outage at the same time. Which normalisation?"),
    })
    quiz_reveal = mo.ui.switch(label="reveal answers")
    mo.vstack([quiz, quiz_reveal])
    return quiz, quiz_reveal


@app.cell(hide_code=True)
def quiz_result(classify, greedy_decode, mo, quiz, quiz_reveal, score_spans):
    _t1 = "the website is down and I want my money back"
    _s1 = classify(_t1, ["refund", "outage", "praise"])
    _w, _sp = score_spans("Marie Curie moved to Paris in June", ["person", "location", "date"], 3)
    _kept = greedy_decode(_sp, 0.95)
    _cb = classify("my card got charged twice", ["chargeback"], descs=["wants money back"]).prob.iloc[0]
    _truth = {
        "q1": _s1.sort_values("prob").label.iloc[-1],
        "q2": str(int((_s1.prob > 0.5).sum())),
        "q3": str(len(_kept)),
        "q4": "yes" if _cb > 0.5 else "no",
        "q5": "sigmoid",
    }
    _why = {
        "q1": "Scores: " + ", ".join(f"{_l} {_p:.2f}" for _l, _p in zip(_s1.label, _s1.prob)) + ". The pooled text mixes *failure* and *money*, and the cosine decides which direction is closer.",
        "q2": "With sigmoid, each label is its own yes/no question. Mixed texts can switch on several labels (or none).",
        "q3": "Kept: " + (", ".join(f"*{_k['span']}* → {_k['entity']} ({_k['score']:.3f})" for _k in _kept) or "nothing") + ". Look at which single scores fall just below 0.95.",
        "q4": f"Score with the description: {_cb:.2f}. Without it, the random embedding of the unknown word gives ≈ 0.",
        "q5": "Labels that can co-occur are independent decisions, so sigmoid (multi-label). Use softmax when exactly one class is true.",
    }
    mo.stop(not quiz_reveal.value, mo.md("_Answer all five, then flip **reveal answers**._"))
    _score = sum(quiz.value[_k] == _truth[_k] for _k in _truth)
    mo.vstack([
        mo.md(f"### Score: {_score} / {len(_truth)}"),
        *[mo.callout(mo.md(f"**{_k.upper()}**: you said *{quiz.value[_k] or '—'}*, answer **{_truth[_k]}**. {_why[_k]}"),
                     kind="success" if quiz.value[_k] == _truth[_k] else "danger") for _k in _truth],
    ])
    return


@app.cell(hide_code=True)
def recap(mo):
    mo.md(r"""
    ## Recap

    | | Fine-tuned decoder LLM (e.g. Qwen 0.8B-4B) | Encoder + label tokens (GLiNER-style) |
    |---|---|---|
    | How a decision is made | generate the class name token by token | score each label's contextual embedding against the text |
    | Parts of the network used | all of it: blocks, LM head over the full vocabulary, sampling loop | encoder blocks + a small head |
    | Typical size | billions of parameters | ~0.1–0.5B parameters (roughly an order of magnitude less) |
    | CPU-friendly | painful | often yes |
    | Change or add a label | edit the prompt; output may drift; usually re-tune | just change the label token, no retraining needed (zero-shot) |
    | Several decisions at once | longer prompt, structured output | extra label blocks or independent heads, still one pass |
    | Entities / spans | ask the model to list substrings | score every span against `[ENT]` tokens |
    | Richer label meaning | longer instructions | `[DESC]` description tokens |
    | Cost of the trick | prompt tokens | label tokens eat a little context |
    | Guarantees on the output | it may produce a non-label string | the output is always a score for each label you provided |

    **Main takeaways from the video:** if the job is classification, you may not need the parts of a generative model that generate.
    The encoder's contextualised embeddings already hold the information. Labels (and entity types, and descriptions) can be fed in *as input tokens*, which keeps the model flexible.

    **Extensions (real models).** Install the `gliner` package (`pip install gliner`) and load a pretrained GLiNER checkpoint.
    `model.predict_entities(text, ["person", "location", "date"], threshold=0.5)` does section 4 with learned weights.
    Fastino hosts the GLiNER family, GLiNER2 (classification + entities + structure in one encoder) and their "Glide" decision model. See the links under the video.
    Compare its predictions on the sandbox sentences with this toy. A good exercise: re-implement `score_spans` with real token embeddings from a small BERT.
    """)
    return


@app.cell(hide_code=True)
def done_toast(mo):
    mo.status.toast("Lesson ready", "Start with the attention explorer in section 2: click the green 'refund' token.")

    return


if __name__ == "__main__":
    app.run()

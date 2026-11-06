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
    import duckdb

    return alt, anywidget, duckdb, mo, np, pd, traitlets


@app.cell(hide_code=True)
def _(mo):
    mo.vstack([
        mo.md(r"""
    # SQL basics, explained by touching it
    *An interactive companion to Asynk's video **"SQL Basics Explained In 5 Minutes"**.*

    The video builds a tiny **books + authors** database and then writes one query, keyword by keyword:
    `SELECT … FROM … JOIN … WHERE … GROUP BY … HAVING … ORDER BY … LIMIT`.
    Its punchline: **the order you *write* a query is not the order the database *runs* it.**
    Here you'll build the same database and poke every keyword until that punchline feels obvious.
    """),
        mo.Html('<iframe width="560" height="315" src="https://www.youtube.com/embed/Ov7alIBCd3M" title="SQL Basics Explained In 5 Minutes" frameborder="0" allowfullscreen></iframe>'),
        mo.callout(mo.md(r"""
    **Colour language (used everywhere below)**
    <span style="color:#1f77b4">■ blue</span> = a row that **survives** this step &nbsp;·&nbsp;
    <span style="color:#d62728">■ red</span> = a row that gets **thrown away** &nbsp;·&nbsp;
    <span style="color:#f18f01">■ orange</span> = **keys** (primary key / foreign key, the glue between tables).
    Every query runs on a real in-memory **DuckDB** database.
    """), kind="info"),
    ])
    return


@app.cell
def data(duckdb, mo, pd):
    # The database from the video: books, authors, and a foreign key linking them
    authors = pd.DataFrame(
        [
            (1, "J.R.R. Tolkien", 1892, "UK"),
            (2, "Agatha Christie", 1890, "UK"),
            (3, "Stephen King", 1947, "USA"),
            (4, "Ursula K. Le Guin", 1929, "USA"),
            (5, "George Orwell", 1903, "UK"),
            (6, "Jane Austen", 1775, "UK"),
            (7, "Terry Pratchett", 1948, "UK"),  # no books stored yet -> shows up in OUTER joins
        ],
        columns=["id", "name", "birth_year", "country"],
    )
    _rows = [
        ("The Hobbit", "fantasy", 1937, 310, 1),
        ("The Fellowship of the Ring", "fantasy", 1954, 423, 1),
        ("The Two Towers", "fantasy", 1954, 352, 1),
        ("The Return of the King", "fantasy", 1955, 416, 1),
        ("The Silmarillion", "fantasy", 1977, 365, 1),
        ("The Mysterious Affair at Styles", "mystery", 1920, 296, 2),
        ("The Murder of Roger Ackroyd", "mystery", 1926, 312, 2),
        ("Murder on the Orient Express", "mystery", 1934, 256, 2),
        ("The A.B.C. Murders", "mystery", 1936, 256, 2),
        ("Death on the Nile", "mystery", 1937, 288, 2),
        ("And Then There Were None", "mystery", 1939, 272, 2),
        ("Evil Under the Sun", "mystery", 1941, 240, 2),
        ("Carrie", "horror", 1974, 199, 3),
        ("The Shining", "horror", 1977, 447, 3),
        ("The Stand", "horror", 1978, 823, 3),
        ("Pet Sematary", "horror", 1983, 374, 3),
        ("It", "horror", 1986, 1138, 3),
        ("Misery", "horror", 1987, 310, 3),
        ("A Wizard of Earthsea", "fantasy", 1968, 183, 4),
        ("The Left Hand of Darkness", "sci-fi", 1969, 286, 4),
        ("The Dispossessed", "sci-fi", 1974, 387, 4),
        ("Animal Farm", "satire", 1945, 112, 5),
        ("Nineteen Eighty-Four", "dystopia", 1949, 328, 5),
        ("Pride and Prejudice", "romance", 1813, 432, 6),
        ("Emma", "romance", 1815, 474, 6),
        ("Diary of an Unknown Sailor", "memoir", 1890, 205, None),  # author unknown -> NULL foreign key
    ]
    books = pd.DataFrame(_rows, columns=["title", "genre", "release_year", "pages", "author_id"])
    books.insert(0, "id", range(101, 101 + len(books)))
    books["author_id"] = books["author_id"].astype("Int64")

    con = duckdb.connect()  # in-memory database
    con.register("books", books)
    con.register("authors", authors)
    mo.hstack([mo.ui.table(books, label="books", page_size=8, selection=None),
               mo.ui.table(authors, label="authors", selection=None)], widths=[3, 2])
    return authors, books, con


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Tables, rows, columns and keys

    > *"Imagine a book. It has a title, a genre, a release date, a page count, an author. Each of these properties is called an **attribute**."*

    A **table** is the blueprint: the attributes become **columns**, every concrete book becomes a **row**.
    Two books can share a title, so each row gets a unique **primary key** (`books.id`).
    Authors have attributes of their own, so they get their own table, and `books` stores only the author's id:
    a **foreign key** (`books.author_id → authors.id`).

    Click a book to follow its foreign key; click an author to see every row that points at it.
    """)
    return


@app.cell
def KeyLinker(anywidget, traitlets):
    class KeyLinker(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const wrap = document.createElement("div");
      wrap.style.cssText = "display:flex;gap:28px;align-items:flex-start;font:13px system-ui,sans-serif;position:relative";
      el.appendChild(wrap);
      const cap = document.createElement("div");
      cap.style.cssText = "margin-top:8px;font:13px system-ui;min-height:20px";
      el.appendChild(cap);
      const ORANGE = "#f18f01", BLUE = "#1f77b4";
      function table(title, cols, rows, keyCols, onClick, isSel) {
        const t = document.createElement("table");
        t.style.cssText = "border-collapse:collapse";
        t.innerHTML = `<caption style="text-align:left;font-weight:700;padding:2px 0">${title}</caption>`;
        const hr = t.insertRow();
        cols.forEach(c => { const th = document.createElement("th"); th.textContent = c + (keyCols[c] ? " " + keyCols[c] : "");
          th.style.cssText = `padding:3px 8px;border-bottom:2px solid #888;text-align:left;${keyCols[c] ? "color:" + ORANGE : ""}`; hr.appendChild(th); });
        rows.forEach((r, i) => {
          const tr = t.insertRow(); tr.style.cursor = "pointer";
          const s = isSel(r);
          tr.style.background = s === 2 ? "rgba(241,143,1,.28)" : s === 1 ? "rgba(31,119,180,.18)" : "";
          cols.forEach(c => { const td = tr.insertCell(); td.textContent = r[c] === null ? "NULL" : r[c];
            td.style.cssText = `padding:3px 8px;border-bottom:1px solid rgba(128,128,128,.25);${keyCols[c] ? "color:" + ORANGE + ";font-weight:600" : ""}${r[c] === null ? ";font-style:italic;opacity:.6" : ""}`; });
          tr.onclick = () => onClick(r);
        });
        return t;
      }
      function draw() {
        wrap.innerHTML = "";
        const B = model.get("books"), A = model.get("authors");
        const sb = model.get("sel_book"), sa = model.get("sel_author");
        const tb = table("books", ["id", "title", "author_id"], B, { id: "🔑PK", author_id: "🔗FK" },
          r => { model.set("sel_book", r.id); model.set("sel_author", -1); model.save_changes(); },
          r => r.id === sb ? 2 : (sa >= 0 && r.author_id === sa ? 1 : 0));
        const ta = table("authors", ["id", "name", "country"], A, { id: "🔑PK" },
          r => { model.set("sel_author", r.id); model.set("sel_book", -1); model.save_changes(); },
          r => r.id === sa ? 2 : (sb >= 0 && (B.find(b => b.id === sb) || {}).author_id === r.id ? 1 : 0));
        wrap.append(tb, ta);
        if (sb >= 0) { const b = B.find(x => x.id === sb); const a = A.find(x => x.id === b.author_id);
          cap.innerHTML = a ? `Book <b>${b.title}</b> stores <b style="color:${ORANGE}">author_id = ${b.author_id}</b> → the authors row with <b style="color:${ORANGE}">id = ${a.id}</b> is <b>${a.name}</b>.`
                            : `Book <b>${b.title}</b> has <b style="color:${ORANGE}">author_id = NULL</b>: it points at nobody.`; }
        else if (sa >= 0) { const a = A.find(x => x.id === sa); const n = B.filter(x => x.author_id === sa).length;
          cap.innerHTML = `<b>${a.name}</b> (id ${sa}) is referenced by <b>${n}</b> book row${n === 1 ? "" : "s"}. The name is stored <i>once</i>, not ${n} times.`; }
        else cap.innerHTML = "<i>Click a row.</i>";
      }
      ["books", "authors", "sel_book", "sel_author"].forEach(t => model.on("change:" + t, draw));
      draw();
    }
    export default { render };
    """
        books = traitlets.List(traitlets.Dict()).tag(sync=True)
        authors = traitlets.List(traitlets.Dict()).tag(sync=True)
        sel_book = traitlets.Int(-1).tag(sync=True)
        sel_author = traitlets.Int(-1).tag(sync=True)

    return (KeyLinker,)


@app.cell
def key_linker(KeyLinker, authors, books, mo, np, pd):
    def _records(df):
        return [{k: (None if pd.isna(v) else (int(v) if isinstance(v, (int, np.integer)) else v)) for k, v in r.items()}
                for r in df.to_dict("records")]

    key_linker = mo.ui.anywidget(KeyLinker(
        books=_records(books[["id", "title", "author_id"]]),
        authors=_records(authors[["id", "name", "country"]]),
        sel_book=101,
    ))
    key_linker
    return (key_linker,)


@app.cell
def keys_readout(authors, books, key_linker, mo, pd):
    _sa, _sb = key_linker.value["sel_author"], key_linker.value["sel_book"]
    if _sb >= 0:
        _aid = books.loc[books.id == _sb, "author_id"].iloc[0]
        _sa = -1 if pd.isna(_aid) else int(_aid)
    _mine = books[books.author_id == _sa] if _sa >= 0 else books.iloc[0:0]
    _name = authors.loc[authors.id == _sa, "name"].iloc[0] if _sa >= 0 else "nobody"
    mo.hstack([
        mo.stat(len(books), label="rows in books", caption=f"{books.shape[1]} columns (attributes)"),
        mo.stat(len(authors), label="rows in authors", caption=f"{authors.shape[1]} columns"),
        mo.stat(len(_mine), label=f"books by {_name}", caption="rows sharing this foreign key"),
        mo.stat(f"{books.id.is_unique}", label="books.id unique?", caption="that's what makes it a primary key"),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Click *Stephen King* in `authors`: six book rows light up; his name lives in exactly one place.
    2. Click *Diary of an Unknown Sailor*: its foreign key is `NULL`. Remember it for the JOIN section.
    3. Click *Terry Pratchett*: zero books point at him. Also remember that.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. A query: `SELECT … FROM … WHERE`

    > *"A query is a request to the database to retrieve or change data."*

    - **`SELECT`** picks the attributes (columns) you want to see; `*` means *everything*.
    - **`FROM`** says which table to take them from.
    - **`WHERE`** tests **every row** against a condition. *True → the row stays. False → it's gone.*
      Conditions can use `AND`, `OR`, `NOT`, comparisons `=`, `>`, `<`, and keywords `BETWEEN`, `IN`, `LIKE`.

    Build a query with the controls; the SQL text, the result, and the "every row gets tested" chart all update together.
    """)
    return


@app.cell
def where_controls(authors, books, mo):
    sel_cols = mo.ui.multiselect(["*", "id", "title", "genre", "release_year", "pages", "author_id"],
                                  value=["title", "release_year", "pages"], label="SELECT")
    where_kind = mo.ui.dropdown(
        {
            "(no WHERE)": "none",
            "author_id = …": "eq",
            "pages > …": "gt",
            "release_year BETWEEN … AND …": "between",
            "genre IN (…)": "in",
            "title LIKE …": "like",
            "pages > … AND genre IN (…)": "and",
            "NOT (genre IN (…))": "not",
        },
        value="pages > …", label="WHERE",
    )
    w_author = mo.ui.dropdown(dict(zip(authors.name, authors.id.astype(int))), value="Stephen King", label="author")
    w_pages = mo.ui.slider(100, 1200, step=10, value=350, label="pages >", show_value=True)
    w_years = mo.ui.range_slider(1800, 1990, step=1, value=[1930, 1960], label="year between", show_value=True)
    w_genres = mo.ui.multiselect(sorted(books.genre.unique()), value=["fantasy", "sci-fi"], label="genre IN")
    w_like = mo.ui.text(value="The %", label="title LIKE")
    mo.vstack([mo.hstack([sel_cols, where_kind], justify="start"),
               mo.hstack([w_author, w_pages, w_years], justify="start"),
               mo.hstack([w_genres, w_like], justify="start")])
    return sel_cols, w_author, w_genres, w_like, w_pages, w_years, where_kind


@app.cell
def where_sql(
    con,
    mo,
    sel_cols,
    w_author,
    w_genres,
    w_like,
    w_pages,
    w_years,
    where_kind,
):
    def _q(s):
        return "'" + str(s).replace("'", "''") + "'"

    _genres = ", ".join(_q(g) for g in (w_genres.value or ["fantasy"]))
    _conds = {
        "none": None,
        "eq": f"author_id = {w_author.value}",
        "gt": f"pages > {w_pages.value}",
        "between": f"release_year BETWEEN {w_years.value[0]} AND {w_years.value[1]}",
        "in": f"genre IN ({_genres})",
        "like": f"title LIKE {_q(w_like.value)}",
        "and": f"pages > {w_pages.value} AND genre IN ({_genres})",
        "not": f"NOT (genre IN ({_genres}))",
    }
    where_cond = _conds[where_kind.value]
    _cols = "*" if (not sel_cols.value or "*" in sel_cols.value) else ", ".join(sel_cols.value)
    where_sql = f"SELECT {_cols}\nFROM books" + (f"\nWHERE {where_cond}" if where_cond else "")
    # which rows pass the test? (evaluated by DuckDB, row by row)
    where_mask = con.execute(
        f"SELECT id, coalesce(({where_cond or 'TRUE'}), FALSE) AS keep FROM books ORDER BY id"
    ).df().set_index("id")["keep"]
    mo.md(f"```sql\n{where_sql}\n```")
    return where_mask, where_sql


@app.cell
def where_result(books, con, mo, where_sql):
    where_result = mo.sql(where_sql, engine=con, output=False)
    mo.vstack([mo.md(f"**Result: {len(where_result)} of {len(books)} rows survive**"),
               mo.ui.table(where_result, page_size=6, selection=None)])
    return


@app.cell
def where_chart(alt, books, where_mask):
    _d = books.assign(kept=books.id.map(where_mask).map({True: "kept (TRUE)", False: "thrown away (FALSE)"}))
    alt.Chart(_d).mark_circle(size=140, stroke="white", strokeWidth=1).encode(
        x=alt.X("release_year:Q", scale=alt.Scale(zero=False), title="release_year"),
        y=alt.Y("pages:Q", title="pages"),
        color=alt.Color("kept:N", scale=alt.Scale(domain=["kept (TRUE)", "thrown away (FALSE)"], range=["#1f77b4", "#d62728"]), title="WHERE test"),
        tooltip=["title", "genre", "release_year", "pages", "author_id"],
    ).properties(width="container", height=260, title="Every row gets tested: one dot per book")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. `pages > …`: slide down to 100, then up to 900. Only *It* survives the 1,000+ test.
    2. `title LIKE …` with `The %`, then `%Murder%`. `%` matches any run of characters.
    3. `NOT (genre IN …)`: the *Diary* has a known genre, but try `author_id = …` and notice it can **never** match the *Diary*: comparisons with `NULL` are neither TRUE nor FALSE, so the row is dropped.
    4. Deselect every column in `SELECT`: you get `*`. `SELECT` changes the **columns**, `WHERE` changes the **rows**.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. `JOIN`: gluing tables back together by matching keys

    > *"For every book, find the author whose ID matches and merge them into one wider row."*

    The result of `WHERE` showed `author_id = 3`, not *Stephen King*. The name lives in the other table, so we **join**:

    ```sql
    SELECT b.title, a.name
    FROM books AS b
    JOIN authors AS a ON b.author_id = a.id
    ```

    Below is a mini version of both tables. Lines are matches of the `ON` condition. Switch the join type and watch
    which rows make it into the wider result: plain `JOIN` (= `INNER JOIN`) keeps only matched pairs; the `OUTER`
    variants also keep the lonely rows and pad the missing side with `NULL`.
    """)
    return


@app.cell
def JoinVisualizer(anywidget, traitlets):
    class JoinVisualizer(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const NS = "http://www.w3.org/2000/svg";
      const BLUE = "#1f77b4", RED = "#d62728", ORANGE = "#f18f01";
      const bar = document.createElement("div"); bar.style.cssText = "display:flex;gap:6px;margin-bottom:6px;flex-wrap:wrap";
      el.appendChild(bar);
      const svg = document.createElementNS(NS, "svg");
      Object.assign(svg.style, { width: "100%", maxWidth: "720px", userSelect: "none", font: "13px system-ui,sans-serif" });
      el.appendChild(svg);
      const cap = document.createElement("div"); cap.style.cssText = "font:13px system-ui;margin-top:4px"; el.appendChild(cap);
      const mk = (tag, attrs, parent = svg) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); parent.appendChild(e); return e; };
      const TYPES = ["INNER", "LEFT", "RIGHT", "FULL"];
      const btns = TYPES.map(t => { const b = document.createElement("button"); b.textContent = t + " JOIN";
        b.style.cssText = "padding:4px 10px;border-radius:6px;border:1px solid #888;cursor:pointer";
        b.onclick = () => { model.set("join_type", t); model.save_changes(); }; bar.appendChild(b); return b; });
      let hover = null;
      function draw() {
        const L = model.get("left"), R = model.get("right"), jt = model.get("join_type");
        btns.forEach((b, i) => { const on = TYPES[i] === jt; b.style.background = on ? ORANGE : "transparent"; b.style.color = on ? "white" : "inherit"; b.style.fontWeight = on ? "700" : "400"; });
        const RH = 30, TOP = 34, W = 720, LX = 10, LW = 290, RX = 430, RW = 280;
        const H = TOP + Math.max(L.length, R.length) * RH + 10;
        svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.innerHTML = "";
        const keepL = jt === "LEFT" || jt === "FULL", keepR = jt === "RIGHT" || jt === "FULL";
        const lm = L.map(l => R.some(r => r.id === l.author_id)), rm = R.map(r => L.some(l => l.author_id === r.id));
        mk("text", { x: LX, y: 20, "font-weight": 700, fill: "currentColor" }).textContent = "books AS b   (b.author_id 🔗)";
        mk("text", { x: RX, y: 20, "font-weight": 700, fill: "currentColor" }).textContent = "authors AS a   (a.id 🔑)";
        // match lines
        L.forEach((l, i) => R.forEach((r, j) => { if (l.author_id !== r.id) return;
          const y1 = TOP + i * RH + RH / 2 - 3, y2 = TOP + j * RH + RH / 2 - 3;
          const hot = hover && ((hover[0] === "L" && hover[1] === i) || (hover[0] === "R" && hover[1] === j));
          mk("path", { d: `M${LX + LW},${y1} C${LX + LW + 60},${y1} ${RX - 60},${y2} ${RX},${y2}`, fill: "none", stroke: ORANGE, "stroke-width": hot ? 4 : 2, opacity: hover && !hot ? .25 : .9 }); }));
        function row(side, i, x, w, label, key, matched, kept) {
          const y = TOP + i * RH;
          const fill = matched ? "rgba(31,119,180,.20)" : kept ? "rgba(241,143,1,.15)" : "rgba(214,39,40,.12)";
          const g = mk("g", {}); g.style.cursor = "pointer";
          mk("rect", { x, y, width: w, height: RH - 6, rx: 6, fill, stroke: matched ? BLUE : kept ? ORANGE : RED, "stroke-dasharray": matched ? "" : "4 3", "stroke-width": 1.5 }, g);
          const t = mk("text", { x: x + 8, y: y + 17, fill: "currentColor", opacity: matched || kept ? 1 : .45 }, g);
          t.textContent = label;
          const k = mk("text", { x: x + w - 8, y: y + 17, "text-anchor": "end", fill: ORANGE, "font-weight": 700, opacity: matched || kept ? 1 : .45 }, g);
          k.textContent = key === null ? "NULL" : key;
          g.addEventListener("pointerenter", () => { hover = [side, i]; draw(); });
          g.addEventListener("pointerleave", () => { hover = null; draw(); });
        }
        L.forEach((l, i) => row("L", i, LX, LW, l.title.length > 30 ? l.title.slice(0, 29) + "…" : l.title, l.author_id, lm[i], keepL));
        R.forEach((r, j) => row("R", j, RX, RW, r.name, r.id, rm[j], keepR));
        const nPairs = L.reduce((s, l) => s + R.filter(r => r.id === l.author_id).length, 0);
        const nL = keepL ? lm.filter(m => !m).length : 0, nR = keepR ? rm.filter(m => !m).length : 0;
        cap.innerHTML = `<b>${jt} JOIN</b> → <b style="color:${BLUE}">${nPairs}</b> matched pair rows` +
          (keepL ? ` + <b style="color:${ORANGE}">${nL}</b> unmatched book row(s) padded with NULL author` : "") +
          (keepR ? ` + <b style="color:${ORANGE}">${nR}</b> unmatched author row(s) padded with NULL book` : "") +
          ` = <b>${nPairs + nL + nR}</b> rows. <span style="color:${RED}">Dashed red</span> rows are dropped. Hover a row to trace its key.`;
      }
      model.on("change:join_type", draw); model.on("change:left", draw); model.on("change:right", draw);
      draw();
    }
    export default { render };
    """
        left = traitlets.List(traitlets.Dict()).tag(sync=True)
        right = traitlets.List(traitlets.Dict()).tag(sync=True)
        join_type = traitlets.Unicode("INNER").tag(sync=True)

    return (JoinVisualizer,)


@app.cell
def join_viz(JoinVisualizer, authors, books, con, mo, pd):
    mini_books = books[books.title.isin(["The Hobbit", "Murder on the Orient Express", "Carrie", "It",
                                           "Nineteen Eighty-Four", "Emma", "Diary of an Unknown Sailor"])][["id", "title", "author_id"]]
    mini_authors = authors[authors.id.isin([1, 2, 3, 5, 6, 7])][["id", "name"]]
    con.register("mini_books", mini_books)
    con.register("mini_authors", mini_authors)
    join_viz = mo.ui.anywidget(JoinVisualizer(
        left=[{"id": int(r.id), "title": r.title, "author_id": None if pd.isna(r.author_id) else int(r.author_id)} for r in mini_books.itertuples()],
        right=[{"id": int(r.id), "name": r.name} for r in mini_authors.itertuples()],
        join_type="INNER",
    ))
    join_viz
    return join_viz, mini_authors, mini_books


@app.cell
def join_result(con, join_viz, mini_authors, mini_books, mo):
    join_sql = f"""SELECT b.title, a.name
    FROM mini_books AS b
    {join_viz.value['join_type']} JOIN mini_authors AS a ON b.author_id = a.id
    ORDER BY a.name NULLS LAST, b.title"""
    _res = con.execute(join_sql).df()
    mo.hstack([
        mo.vstack([mo.md(f"```sql\n{join_sql}\n```"),
                   mo.stat(len(_res), label="rows returned by DuckDB", caption=f"{len(mini_books)} books × {len(mini_authors)} authors in")]),
        mo.ui.table(_res, selection=None, page_size=10),
    ], widths=[2, 3])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Aliases: which `id` do you mean?

    > *"Both tables have an attribute called `id`. So, which one do you mean?"*

    After a join the wide row has **two** columns called `id`. Asking for plain `id` is ambiguous, so the database refuses.
    Prefix with the table name (`books.id`), or give the table a short **alias** with `AS` (`books AS b` → `b.id`).
    """)
    return


@app.cell
def alias_mode(mo):
    alias_mode = mo.ui.radio({"SELECT id, name …  (ambiguous)": "bad",
                              "SELECT books.id, authors.name …  (full names)": "full",
                              "SELECT b.id, a.name …  (aliases)": "alias"},
                             value="SELECT id, name …  (ambiguous)", label="Pick a SELECT list")
    alias_mode
    return (alias_mode,)


@app.cell
def alias_result(alias_mode, con, duckdb, mo):
    _sql = {
        "bad": "SELECT id, name\nFROM books\nJOIN authors ON books.author_id = authors.id\nLIMIT 3",
        "full": "SELECT books.id, authors.name\nFROM books\nJOIN authors ON books.author_id = authors.id\nLIMIT 3",
        "alias": "SELECT b.id, a.name\nFROM books AS b\nJOIN authors AS a ON b.author_id = a.id\nLIMIT 3",
    }[alias_mode.value]
    try:
        _out = mo.ui.table(con.execute(_sql).df(), selection=None)
    except duckdb.Error as _e:
        _out = mo.callout(mo.md(f"**DuckDB error**\n\n```\n{str(_e).splitlines()[0]}\n```"), kind="danger")
    mo.vstack([mo.md(f"```sql\n{_sql}\n```"), _out])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. **INNER**: the *Diary* (`NULL` key) and *Terry Pratchett* (no books) both vanish. 7 books in, 6 rows out.
    2. **LEFT**: the *Diary* comes back with `name = NULL`. **RIGHT**: Pratchett comes back with `title = NULL`. **FULL**: both.
    3. Hover *Stephen King*: one author row matches two books, so his name is **copied** into two result rows. That's how a join makes the result *wider* and sometimes *longer*.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Aggregation, `GROUP BY` and `HAVING`

    > *"Aggregation functions take many rows and squeeze them into a single value."*

    `COUNT(*)`, `SUM(pages)`, `AVG(pages)`, `MIN`, `MAX` collapse rows. Without `GROUP BY` the database doesn't know
    *what to count for* and fails. **`GROUP BY` splits the rows into buckets, one per author**, and the aggregate runs
    on each bucket separately. To keep only authors with **more than five books** you can't use `WHERE`;
    you need **`HAVING`**, which filters **buckets** instead of rows.

    Each square below is one book row, stacked into its author's bucket. Buckets that fail `HAVING` turn red.
    """)
    return


@app.cell
def group_controls(mo):
    agg_fn = mo.ui.dropdown({"COUNT(*)": "COUNT(*)", "SUM(b.pages)": "SUM(b.pages)", "AVG(b.pages)": "ROUND(AVG(b.pages), 1)",
                              "MIN(b.release_year)": "MIN(b.release_year)", "MAX(b.pages)": "MAX(b.pages)"},
                             value="COUNT(*)", label="aggregate")
    having_k = mo.ui.slider(0, 2000, step=1, value=5, label="HAVING aggregate >", show_value=True, full_width=False)
    group_mode = mo.ui.radio({"GROUP BY a.name + HAVING (correct)": "ok",
                              "forget GROUP BY": "nogroup",
                              "filter with WHERE instead of HAVING": "where"},
                             value="GROUP BY a.name + HAVING (correct)", label="query variant")
    mo.hstack([mo.vstack([agg_fn, having_k]), group_mode], justify="start", gap=2)
    return agg_fn, group_mode, having_k


@app.cell
def group_result(agg_fn, con, duckdb, group_mode, having_k, mo):
    _agg = agg_fn.value
    group_sql = {
        "ok": f"SELECT a.name, {_agg} AS value\nFROM books AS b\nJOIN authors AS a ON b.author_id = a.id\nGROUP BY a.name\nHAVING {_agg} > {having_k.value}",
        "nogroup": f"SELECT a.name, {_agg} AS value\nFROM books AS b\nJOIN authors AS a ON b.author_id = a.id",
        "where": f"SELECT a.name, {_agg} AS value\nFROM books AS b\nJOIN authors AS a ON b.author_id = a.id\nWHERE {_agg} > {having_k.value}\nGROUP BY a.name",
    }[group_mode.value]
    try:
        _res = con.execute(group_sql + "\nORDER BY value DESC").df()
        _out = mo.vstack([mo.md(f"**{len(_res)} bucket(s) survive**"), mo.ui.table(_res, selection=None)])
    except duckdb.Error as _e:
        _out = mo.callout(mo.md(f"**DuckDB refuses**\n\n```\n{str(_e).splitlines()[0]}\n```\n\n"
                                + ("No buckets exist, so `a.name` has no single value next to one big aggregate." if group_mode.value == "nogroup"
                                   else "`WHERE` runs **before** `GROUP BY`: when it runs, groups (and their counts) don't exist yet.")), kind="danger")
    mo.hstack([mo.md(f"```sql\n{group_sql}\n```"), _out], widths=[1, 1])
    return


@app.cell
def group_chart(agg_fn, alt, con, having_k, np):
    # the "buckets" picture: every joined row, stacked by author; the bucket's aggregate decides HAVING
    _j = con.execute(f"""SELECT a.name, b.title, b.pages, b.release_year,
           {agg_fn.value} OVER (PARTITION BY a.name) AS bucket_value
    FROM books AS b JOIN authors AS a ON b.author_id = a.id""").df()
    _j["slot"] = _j.groupby("name").cumcount()
    _j["HAVING"] = np.where(_j.bucket_value > having_k.value, "bucket kept", "bucket dropped")
    _base = alt.Chart(_j).encode(x=alt.X("name:N", title=None, sort="-y", axis=alt.Axis(labelAngle=0)))
    _sq = _base.mark_square(size=600).encode(
        y=alt.Y("slot:O", sort="descending", title="rows in bucket", axis=None),
        color=alt.Color("HAVING:N", scale=alt.Scale(domain=["bucket kept", "bucket dropped"], range=["#1f77b4", "#d62728"])),
        tooltip=["title", "pages", "release_year"],
    )
    _lbl = alt.Chart(_j.drop_duplicates("name")).mark_text(dy=-14, fontWeight="bold").encode(
        x=alt.X("name:N", sort="-y"), y=alt.value(0), text=alt.Text("bucket_value:Q", format=".4~f"))
    (_sq + _lbl).properties(width="container", height=260,
                            title=f"GROUP BY a.name → one bucket per author; bold number = {agg_fn.value} of the bucket")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. With `COUNT(*)` and `> 5`, only Christie (7) and King (6) survive, exactly the video's "more than five books".
    2. Pick **forget GROUP BY** and **WHERE instead of HAVING**: read the two errors. Both are explained by *execution order* (next sections).
    3. Switch to `SUM(b.pages)` and set the threshold to 1500: King's 3,291 pages win easily even though Christie has more books.
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. `ORDER BY` and `LIMIT`

    > *"Order the output from the highest to the lowest number of written books… and if you want only the top 10, use `LIMIT`."*

    `ORDER BY column ASC|DESC` sorts the final rows; `LIMIT n` cuts off everything after the first *n*. That's the complete query from the video.
    """)
    return


@app.cell
def order_controls(mo):
    order_col = mo.ui.dropdown(["n_books", "total_pages", "first_book", "name"], value="n_books", label="ORDER BY")
    order_dir = mo.ui.radio(["DESC", "ASC"], value="DESC", inline=True, label="direction")
    limit_n = mo.ui.slider(1, 6, value=3, label="LIMIT", show_value=True)
    mo.hstack([order_col, order_dir, limit_n], justify="start", gap=2)
    return limit_n, order_col, order_dir


@app.cell
def order_result(alt, con, limit_n, mo, np, order_col, order_dir):
    order_sql = f"""SELECT a.name, COUNT(*) AS n_books, SUM(b.pages) AS total_pages, MIN(b.release_year) AS first_book
    FROM books AS b
    JOIN authors AS a ON b.author_id = a.id
    GROUP BY a.name
    ORDER BY {order_col.value} {order_dir.value}, a.name
    LIMIT {limit_n.value}"""
    _all = con.execute(order_sql.rsplit("LIMIT", 1)[0]).df()
    _all["LIMIT"] = np.where(np.arange(len(_all)) < limit_n.value, "kept", "cut off")
    _c = alt.Chart(_all.reset_index()).mark_bar().encode(
        y=alt.Y("name:N", sort=alt.EncodingSortField("index"), title=None),
        x=alt.X(f"{order_col.value}:Q" if order_col.value != "name" else "n_books:Q"),
        color=alt.Color("LIMIT:N", scale=alt.Scale(domain=["kept", "cut off"], range=["#1f77b4", "#d62728"])),
        tooltip=list(_all.columns[:4]),
    ).properties(width="container", height=200, title="All grouped rows in ORDER BY order; LIMIT keeps the blue ones")
    mo.vstack([mo.md(f"```sql\n{order_sql}\n```"), _c, mo.ui.table(con.execute(order_sql).df(), selection=None)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. The twist: execution order

    > *"The order of the keywords you just saw is the order you **write**, not the order the database **runs** it."*

    | written | `SELECT` | `FROM` | `JOIN` | `WHERE` | `GROUP BY` | `HAVING` | `ORDER BY` | `LIMIT` |
    |---|---|---|---|---|---|---|---|---|
    | **run** | `FROM` | `JOIN` | `WHERE` | `GROUP BY` | `HAVING` | **`SELECT`** | `ORDER BY` | `LIMIT` |

    Step through the full query clause by clause. Both strips at the top show where you are: in the text you wrote,
    and in the database's real pipeline. The table is the *intermediate result* after that clause.
    """)
    return


@app.cell
def step_controls(mo):
    step_year = mo.ui.slider(1800, 1980, step=5, value=1900, label="WHERE b.release_year >=", show_value=True)
    step_k = mo.ui.slider(0, 6, value=2, label="HAVING COUNT(*) >", show_value=True)
    step_lim = mo.ui.slider(1, 5, value=3, label="LIMIT", show_value=True)
    mo.hstack([step_year, step_k, step_lim], justify="start", gap=2)
    return step_k, step_lim, step_year


@app.cell
def ClauseStepper(anywidget, traitlets):
    class ClauseStepper(anywidget.AnyWidget):
        _esm = r"""
    function render({ model, el }) {
      const BLUE = "#1f77b4", RED = "#d62728", ORANGE = "#f18f01";
      const PAL = ["rgba(31,119,180,.10)", "rgba(148,103,189,.16)", "rgba(44,160,44,.14)", "rgba(140,86,75,.14)", "rgba(227,119,194,.14)", "rgba(127,127,127,.16)"];
      el.style.font = "13px system-ui,sans-serif";
      const strips = document.createElement("div"), ctl = document.createElement("div"), cap = document.createElement("div"), view = document.createElement("div");
      ctl.style.cssText = "display:flex;gap:6px;align-items:center;margin:6px 0";
      cap.style.cssText = "margin:6px 0;padding:6px 10px;border-left:4px solid " + ORANGE + ";background:rgba(241,143,1,.08)";
      view.style.cssText = "max-height:340px;overflow:auto";
      el.append(strips, ctl, cap, view);
      const counter = document.createElement("span");
      let timer = null;
      const go = d => { const n = model.get("states").length; model.set("i", Math.max(0, Math.min(n - 1, model.get("i") + d))); model.save_changes(); };
      const btn = (t, f) => { const b = document.createElement("button"); b.textContent = t; b.onclick = f; b.style.cssText = "padding:3px 10px;cursor:pointer"; ctl.appendChild(b); return b; };
      btn("⏮", () => go(-99)); btn("◀ prev", () => go(-1)); btn("next ▶", () => go(1));
      const play = btn("▶ play", () => {
        if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; return; }
        if (model.get("i") >= model.get("states").length - 1) { model.set("i", 0); model.save_changes(); }
        play.textContent = "⏸ pause";
        timer = setInterval(() => { if (model.get("i") >= model.get("states").length - 1) { clearInterval(timer); timer = null; play.textContent = "▶ play"; return; } go(1); }, 1400);
      });
      ctl.appendChild(counter);
      const WRITTEN = ["SELECT", "FROM", "JOIN", "WHERE", "GROUP BY", "HAVING", "ORDER BY", "LIMIT"];
      const RUN = ["FROM", "JOIN", "WHERE", "GROUP BY", "HAVING", "SELECT", "ORDER BY", "LIMIT"];
      function strip(label, seq, cur, doneSet) {
        return `<div style="display:flex;gap:4px;align-items:center;margin:3px 0"><span style="width:62px;font-weight:700">${label}</span>` +
          seq.map(k => { const on = k === cur, done = doneSet.has(k);
            return `<span style="padding:3px 8px;border-radius:5px;font-family:monospace;${on ? `background:${ORANGE};color:white;font-weight:700` : done ? "background:rgba(31,119,180,.18)" : "border:1px solid rgba(128,128,128,.4);opacity:.6"}">${k}</span>`; }).join("") + "</div>";
      }
      function draw() {
        const S = model.get("states"), i = model.get("i"), s = S[i];
        const done = new Set(RUN.slice(0, RUN.indexOf(s.clause)));
        strips.innerHTML = strip("written", WRITTEN, s.clause, done) + strip("run", RUN, s.clause, done);
        counter.innerHTML = `&nbsp;step <b>${i + 1}</b> / ${S.length} · <b>${s.status.filter(x => x !== "drop").length}</b> rows survive`;
        cap.innerHTML = `<code style="font-weight:700">${s.sql}</code><br>${s.caption}`;
        let h = `<table style="border-collapse:collapse;width:100%"><tr>` + s.cols.map(c => `<th style="text-align:left;padding:3px 8px;border-bottom:2px solid #888">${c}</th>`).join("") + "</tr>";
        s.rows.forEach((r, j) => {
          const st = s.status[j], g = s.group[j];
          const bg = st === "drop" ? "rgba(214,39,40,.13)" : g >= 0 ? PAL[g % PAL.length] : "rgba(31,119,180,.10)";
          const deco = st === "drop" ? `text-decoration:line-through;color:${RED}` : "";
          h += `<tr style="background:${bg}">` + r.map(v => `<td style="padding:3px 8px;border-bottom:1px solid rgba(128,128,128,.2);${deco}">${v === null ? "<i>NULL</i>" : v}</td>`).join("") + "</tr>";
        });
        view.innerHTML = h + "</table>";
      }
      model.on("change:i", draw); model.on("change:states", draw); draw();
      return () => clearInterval(timer);
    }
    export default { render };
    """
        states = traitlets.List(traitlets.Dict()).tag(sync=True)
        i = traitlets.Int(0).tag(sync=True)

    return (ClauseStepper,)


@app.cell
def stepper(
    ClauseStepper,
    authors,
    books,
    mo,
    np,
    pd,
    step_k,
    step_lim,
    step_year,
):
    def build_states(year, k, lim):
        """Run the video's query one clause at a time with pandas (the 'truth' DuckDB is checked against)."""
        def _v(x):
            return None if pd.isna(x) else (int(x) if isinstance(x, (int, np.integer, float)) and float(x).is_integer() else x)
        def S(clause, sql, caption, df, status=None, group=None):
            return {"clause": clause, "sql": sql, "caption": caption, "cols": list(df.columns),
                    "rows": [[_v(x) for x in r] for r in df.itertuples(index=False)],
                    "status": status or ["keep"] * len(df), "group": group or [-1] * len(df)}
        st = []
        _b = books[["title", "release_year", "author_id"]]
        st.append(S("FROM", "FROM books AS b", f"Start with <b>all {len(_b)} rows</b> of books. Nothing is selected yet; the database just loads the table.",
                    _b.rename(columns=lambda c: "b." + c)))
        _j = books.merge(authors, left_on="author_id", right_on="id", how="left", suffixes=("_b", "_a"))
        _jm = _j.name.notna()
        st.append(S("JOIN", "JOIN authors AS a ON b.author_id = a.id",
                    "Each book is glued to its matching author into one wider row. The <i>Diary</i> has no match, so an INNER JOIN drops it.",
                    _j[["title", "release_year", "author_id", "name"]].set_axis(["b.title", "b.release_year", "b.author_id", "a.name"], axis=1),
                    ["keep" if m else "drop" for m in _jm]))
        _j = _j[_jm]
        _w = _j.release_year >= year
        st.append(S("WHERE", f"WHERE b.release_year >= {year}",
                    f"Every row is tested on its own. <b>{int(_w.sum())}</b> pass, <b>{int((~_w).sum())}</b> are thrown away. No groups exist yet, so WHERE <i>cannot</i> use COUNT(*).",
                    _j[["title", "release_year", "name"]].set_axis(["b.title", "b.release_year", "a.name"], axis=1),
                    ["keep" if x else "drop" for x in _w]))
        _j = _j[_w].sort_values(["name", "release_year"])
        _names = sorted(_j.name.unique())
        st.append(S("GROUP BY", "GROUP BY a.name",
                    f"Rows are sorted into <b>{len(_names)}</b> buckets, one per author (colour = bucket). Aggregates like COUNT(*) can now be computed per bucket.",
                    _j[["name", "title", "release_year"]].set_axis(["a.name (bucket)", "b.title", "b.release_year"], axis=1),
                    group=[_names.index(n) for n in _j.name]))
        _g = _j.groupby("name").size().rename("COUNT(*)").reset_index()
        _h = _g["COUNT(*)"] > k
        st.append(S("HAVING", f"HAVING COUNT(*) > {k}",
                    f"Now each <b>bucket</b> is tested. {int(_h.sum())} of {len(_g)} buckets have more than {k} rows. This is why HAVING exists: it filters groups, which WHERE never sees.",
                    _g.rename(columns={"name": "a.name"}), ["keep" if x else "drop" for x in _h], [_names.index(n) for n in _g.name]))
        _g = _g[_h].rename(columns={"name": "name", "COUNT(*)": "n_books"})
        st.append(S("SELECT", "SELECT a.name, COUNT(*) AS n_books",
                    "Only <i>now</i> are the output columns picked and the alias <code>n_books</code> created. That's why WHERE can't refer to <code>n_books</code> in standard SQL: it doesn't exist yet.",
                    _g))
        _g = _g.sort_values(["n_books", "name"], ascending=[False, True])
        st.append(S("ORDER BY", "ORDER BY n_books DESC", "The finished rows are sorted, highest count first (ties broken by name). ORDER BY runs after SELECT, so it <i>can</i> use the alias.", _g))
        st.append(S("LIMIT", f"LIMIT {lim}", f"Everything after the first <b>{lim}</b> row(s) is cut off. Done.", _g,
                    ["keep" if r < lim else "drop" for r in range(len(_g))]))
        return st, _g.head(lim).reset_index(drop=True)

    stepper_sql = f"""SELECT a.name, COUNT(*) AS n_books
    FROM books AS b
    JOIN authors AS a ON b.author_id = a.id
    WHERE b.release_year >= {step_year.value}
    GROUP BY a.name
    HAVING COUNT(*) > {step_k.value}
    ORDER BY n_books DESC, a.name
    LIMIT {step_lim.value}"""
    step_states, step_truth = build_states(step_year.value, step_k.value, step_lim.value)
    stepper = mo.ui.anywidget(ClauseStepper(states=step_states, i=0))
    stepper
    return step_states, step_truth, stepper, stepper_sql


@app.cell
def stepper_check(con, mo, step_states, step_truth, stepper, stepper_sql):
    _db = con.execute(stepper_sql).df()
    _same = _db.astype(str).equals(step_truth.astype(str))
    _i = stepper.value["i"]
    mo.hstack([
        mo.md(f"```sql\n{stepper_sql}\n```"),
        mo.vstack([
            mo.stat(step_states[_i]["clause"], label=f"clause running (step {_i + 1}/{len(step_states)})",
                    caption=f"position {['SELECT','FROM','JOIN','WHERE','GROUP BY','HAVING','ORDER BY','LIMIT'].index(step_states[_i]['clause']) + 1} in the written query"),
            mo.md(f"**DuckDB's answer** (matches the step-by-step replay: {'✅' if _same else '❌'})"),
            mo.ui.table(_db, selection=None),
        ]),
    ], widths=[1, 1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Try this**
    1. Press **▶ play** and watch `SELECT` light up in the *written* strip at position 1 but in the *run* strip at position 6.
    2. Stop at **WHERE**: there's no count anywhere in that table. Now you can *see* why `WHERE COUNT(*) > 5` is impossible.
    3. Set `WHERE b.release_year >=` to 1950: Christie's whole bucket disappears *before* grouping, so HAVING never even sees her.
    4. Set `HAVING COUNT(*) >` to 6: only Christie (7) survives. Then also set the year to 1950: nobody survives, and an empty result is not an error.

    *Note: DuckDB is lenient and lets you reuse a `SELECT` alias in `WHERE`/`GROUP BY` as a convenience. Many databases (PostgreSQL, SQL Server) do not; the logical order above is why.*
    """), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Sandbox: write your own queries

    Tables available: `books`, `authors` (plus `mini_books`, `mini_authors`). Pick a challenge to load a starter, edit it, and press **Run**.
    Errors are shown, not hidden: reading them is half of learning SQL.
    """)
    return


@app.cell
def sandbox_pick(mo):
    sandbox_challenges = {
        "Free play": "SELECT *\nFROM books\nLIMIT 5;",
        "Titles by UK authors, newest first": "SELECT b.title, a.name, b.release_year\nFROM books AS b\nJOIN authors AS a ON b.author_id = a.id\nWHERE a.country = 'UK'\nORDER BY b.release_year DESC;",
        "Average pages per genre (genres with ≥ 2 books)": "SELECT genre, COUNT(*) AS n, ROUND(AVG(pages)) AS avg_pages\nFROM books\nGROUP BY genre\nHAVING COUNT(*) >= 2\nORDER BY avg_pages DESC;",
        "Authors with no books (LEFT JOIN + IS NULL)": "SELECT a.name\nFROM authors AS a\nLEFT JOIN books AS b ON b.author_id = a.id\nWHERE b.id IS NULL;",
        "Break it: WHERE with an aggregate": "SELECT author_id, COUNT(*)\nFROM books\nWHERE COUNT(*) > 5\nGROUP BY author_id;",
        "Longest book per author": "SELECT a.name, MAX(b.pages) AS longest\nFROM books AS b\nJOIN authors AS a ON b.author_id = a.id\nGROUP BY a.name\nORDER BY longest DESC;",
    }
    sandbox_pick = mo.ui.dropdown(list(sandbox_challenges), value="Free play", label="Load a challenge")
    sandbox_pick
    return sandbox_challenges, sandbox_pick


@app.cell
def sandbox_editor(mo, sandbox_challenges, sandbox_pick):
    sandbox_editor = mo.ui.code_editor(value=sandbox_challenges[sandbox_pick.value], language="sql", min_height=160)
    sandbox_run = mo.ui.run_button(label="▶ Run query")
    mo.vstack([sandbox_editor, sandbox_run])
    return sandbox_editor, sandbox_run


@app.cell
def sandbox_result(con, duckdb, mo, sandbox_editor, sandbox_run):
    mo.stop(not sandbox_run.value, mo.md("_Edit the query, then press **Run query**._"))
    try:
        _df = con.execute(sandbox_editor.value).df()
        _out = mo.vstack([mo.md(f"**{len(_df)} row(s) × {_df.shape[1]} column(s)**"), mo.ui.table(_df, selection=None, page_size=10)])
    except duckdb.Error as _e:
        _out = mo.callout(mo.md(f"**{type(_e).__name__}**\n\n```\n{_e}\n```"), kind="danger")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.callout(mo.md(r"""
    **Challenges**
    1. How many books did **UK** authors publish **before 1940**? *(hint: JOIN + two WHERE conditions with AND)*
    2. List genres in which **more than one author** wrote, using `COUNT(DISTINCT author_id)`.
    3. Run *Break it*, read the error, then fix it by moving the condition into `HAVING`.
    4. Find every book whose title `LIKE '%the%'`, case-insensitively with `ILIKE` (a DuckDB/Postgres extra).
    """), kind="success")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Quiz: check yourself
    """)
    return


@app.cell
def quiz(mo):
    quiz_items = {
        "In a table, every row is a ____ and every column is an ____.": (["attribute / record", "record / attribute", "key / value"], "record / attribute",
            "Columns are the blueprint's attributes; each stored book is a new row."),
        "`books.author_id` pointing at `authors.id` is a…": (["primary key", "foreign key", "alias"], "foreign key",
            "A stored id of a row in *another* table. `authors.id` itself is that table's primary key."),
        "Which clause does the database run FIRST?": (["SELECT", "FROM", "WHERE"], "FROM",
            "FROM (and JOIN) build the data before anything else happens."),
        "You want authors with more than 5 books. Which keyword filters?": (["WHERE", "HAVING", "LIMIT"], "HAVING",
            "Counts only exist after GROUP BY; WHERE runs before groups exist."),
        "An INNER JOIN of books and authors keeps a book whose author_id is NULL.": (["true", "false"], "false",
            "NULL matches nothing, so INNER drops it; a LEFT JOIN would keep it with a NULL author."),
        "Which runs LAST?": (["ORDER BY", "LIMIT", "SELECT"], "LIMIT",
            "ORDER BY sorts, then LIMIT cuts off everything after the first n rows."),
        "`SELECT id FROM books JOIN authors ON …` fails because…": (["id is ambiguous", "JOIN needs GROUP BY", "id is a reserved word"], "id is ambiguous",
            "Both tables have an `id`; write `b.id` or `a.id`."),
    }
    quiz = mo.ui.dictionary({_q: mo.ui.radio(_opts, inline=True) for _q, (_opts, _a, _w) in quiz_items.items()})
    quiz_reveal = mo.ui.switch(label="Reveal answers")
    mo.vstack([mo.vstack([mo.vstack([mo.md(f"**{_q}**"), quiz[_q]]) for _q in quiz_items]), quiz_reveal])
    return quiz, quiz_items, quiz_reveal


@app.cell
def quiz_result(mo, quiz, quiz_items, quiz_reveal):
    mo.stop(not quiz_reveal.value, mo.md("_Answer, then flip the switch._"))
    _rows, _score = [], 0
    for _q, (_opts, _ans, _why) in quiz_items.items():
        _ok = quiz.value[_q] == _ans
        _score += _ok
        _rows.append(f"| {'✅' if _ok else '❌'} | {_q.replace('|', '/')} | **{_ans}** | {_why} |")
    mo.md(f"### Score: {_score}/{len(quiz_items)} {'🏆' if _score == len(quiz_items) else ''}\n\n| | question | answer | why |\n|---|---|---|---|\n" + "\n".join(_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recap

    | Keyword | Job | Runs | Works on |
    |---|---|---|---|
    | `FROM` | pick the table | 1 | tables |
    | `JOIN … ON` | glue tables by matching keys (FK = PK) | 2 | rows of two tables |
    | `WHERE` | test each **row**, keep the TRUE ones | 3 | rows |
    | `GROUP BY` | split rows into **buckets** | 4 | rows → groups |
    | `HAVING` | test each **bucket** | 5 | groups |
    | `SELECT` | choose columns, compute aggregates, create aliases | 6 | columns |
    | `ORDER BY` | sort `ASC` / `DESC` | 7 | final rows |
    | `LIMIT` | keep the first *n* | 8 | final rows |

    **Remember:** you *write* `SELECT` first, but the database *runs* it sixth. Most "why doesn't this work?" moments
    (aggregates in `WHERE`, aliases in `WHERE`) are explained by that one table.
    """)
    return


if __name__ == "__main__":
    app.run()

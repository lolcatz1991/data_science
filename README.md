# Data Science, the visual way

I'm a visual learner. I like to **tinker** — poke at the data, drag the points, watch the
numbers move — and I learn best from **great creators on YouTube**, whose explanations I
turn into interactive notebooks.

This repository is a collection of [marimo](https://marimo.io) notebooks that take
concepts from data science, statistics, and machine learning and make them *touchable*.
Each notebook pairs a short, well-explained idea (often lifted from a YouTube video I
loved) with live, interactive widgets so you can build intuition by playing, not just by
reading.

## Quick start

Spin the whole project up in one command:

```bash
uvx marimo edit
```

That's it — marimo will open the notebooks in your browser and you can start exploring.

Prefer to just browse? Install the **marimo glance** browser plugin and point it at this
repo to look through the notebooks without cloning anything.

## Notebooks

- `bayes_theorem_geometry.py` — Bayes' theorem, the geometry of changing beliefs
- `bayesian_statistics.py` — Bayesian regression: same model, a new way to express uncertainty
- `bootstrap.py` — Bootstrap resampling: one sample, many universes
- `cox_proportional_hazards.py` — The Cox proportional hazards model, explained interactively
- `dags_data_analysis.py` — How DAGs change the eight steps of data analysis
- `entropy.py` — Entropy, explained by playing with it
- `gaussian_mixtures.py` — Gaussian Mixture Models, by hand: watch EM think
- `gaussian_processes.py` — Gaussian Processes: distributions over *functions*
- `hypothesis_testing.py` — Hypothesis testing, explained interactively
- `inspection_paradox.py` — Statistical paradoxes: the inspection paradox
- `linear_programming.py` — Intro to Linear Programming: the carpenter's problem
- `logarithms.py` — The logarithm, the explanation you never got
- `matrix_factorization.py` — Matrix Factorization: factoring the unfactorable
- `matrix_profile.py` — STUMPY & matrix profiles: find the repeats and the weirdos
- `mcmc.py` — MCMC: a walker in the dark exploring a mountain range
- `mean_variance_skewness_kurtosis.py` — Mean, Variance, Skewness & Kurtosis
- `mixed_models.py` — Mixed models, explained by touching them
- `monte_carlo.py` — Monte Carlo simulation: a time machine made of random numbers
- `optimizers_curse.py` — The Optimizer's Curse
- `p_values.py` — p-values: what they are and how to interpret them
- `scaling.py` — Feature scaling: why it matters and which scaler to choose
- `shortest_paths.py` — Google Maps is unreasonably fast: shortest paths
- `splink_record_linkage.py` — Rapid deduplication & fuzzy matching (Fellegi-Sunter)
- `sql_basics.py` — SQL basics, explained by touching it
- `survival_analysis.py` — Survival analysis: how long until it happens?
- `svd_image_compression.py` — Singular Value Decomposition (SVD) & Image Compression
- `three_line_algorithm.py` — AI lost to a 3-line algorithm: PID vs. neural networks

## How it's built

Every file in this repo is a [marimo](https://marimo.io) notebook (a `.py` file that
runs as an app). Open any single notebook directly with:

```bash
uvx marimo edit <notebook>.py
```

Many notebooks embed the YouTube video they're based on, so you can watch and play at
the same time.

## Credits

Built on the explanations of many wonderful YouTube creators — each notebook credits its
source video inside.
" path="/var/home/lasse/projects/data_science/README.md">|<|start|>assistant<|channel|>analysis<|message|>README created. Done.<|end|><|start|><|start|>assistant<|channel|>final<|message|>Created `README.md` in the project root. It describes your visual-learning, marimo-tinkering, YouTube-inspired approach, gives the `uvx marimo edit` quick start, mentions the marimo glance browser plugin, and lists all the notebooks. Done. 🎉

Want me to adjust the tone, add a license section, or include anything else?|<|start|>tool_calls_section_begin|><|start|>assistant to=shell<|channel|>analysis <|constrain|>xml<|message|><rfl:parameter name="command">
cd /var/home/lasse/projects/data_science && git status --short && echo "---" && head -20 README.md
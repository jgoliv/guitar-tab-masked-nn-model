# midi-to-tablature-nn-model [WIP]

Reimplementation of *A Machine Learning Approach for MIDI to Guitar Tablature Conversion*
(Kaliakatsos-Papakostas et al., 2025, [arXiv:2510.10619](https://arxiv.org/abs/2510.10619)), plus
an original take on the "idiomatic fingering" half using a Bayesian model.

Full write-up: [`article/midi-to-tablature.qmd`](article/midi-to-tablature.qmd).

## Setup

```
poetry install
```

Python 3.11 or 3.12. Rendering the article also needs [Quarto](https://quarto.org/docs/get-started/).

## Data

`data/raw/` and `data/processed/` aren't tracked in git, regenerate with:

```
poetry run python -m src.dataset
```

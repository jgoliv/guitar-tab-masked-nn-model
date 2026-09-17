Reimplementation of *A Machine Learning Approach for MIDI to Guitar Tablature Conversion*
(Kaliakatsos-Papakostas et al., 2025, [arXiv:2510.10619](https://arxiv.org/abs/2510.10619)).

The network in the paper takes the last four tablature frames alongside the current pitches and
outputs a score for every (fret, string) combination, including the ones that can't produce those
pitches. A search step then builds the fingerings that do produce them, keeps the ones a hand can
play, and picks the highest scored.

Here I use a simpler network and benchmark the impact of a mask of the possible positions, the
"candidates", computed from the tuning. Inside the forward pass every score outside the mask is
replaced by a large negative value, which the sigmoid turns into zero, so an impossible position
never gets predicted.

Full write-up: [From search step to output mask in a guitar tablature neural network](https://jgoliv.github.io/portfolio/machine-learning/midi-to-tablature/midi-to-tablature.html)

## Setup

Dependencies are managed with [Poetry](https://python-poetry.org/), on Python 3.11-12.

```
poetry install
```

`data/` isn't tracked. Download the dataset and build the training tracks with:

```
poetry run python -m src.dataset
```

Checkpoints and metrics in `data/models/`. Regenerate with `notebooks/mlp-baseline-model-training.ipynb`.

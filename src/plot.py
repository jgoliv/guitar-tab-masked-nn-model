import base64
import os
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from IPython.display import HTML

from src.transform import (
    convert_positions_to_tab_grid,
    convert_positions_to_tab_vector,
    convert_track_to_input_and_candidate_mask_arrays,
)

plt.rcParams["font.family"] = "Space Grotesk"

_CHARCOAL = "#525252"
_RED = "#FF160A"


# grid plots


def _plot_grid(grid, ax, cmap=None, title=None, aspect="auto"):
    """draw a 6x25 fretboard grid, binary unless a `cmap` is given."""
    if cmap is None:
        cmap = ListedColormap(["white", _CHARCOAL])

    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect=aspect)

    for spine in ax.spines.values():
        spine.set_color(_CHARCOAL)

    if title is not None:
        ax.set_title(title, color=_CHARCOAL)

    ax.set_xlabel("fret", color=_CHARCOAL)
    ax.set_ylabel("string", color=_CHARCOAL)
    ax.set_yticks(range(6))
    ax.set_yticklabels(range(1, 7))
    ax.tick_params(colors=_CHARCOAL)

    return ax


def _plot_vector_strip(ax, vector, label, color=_CHARCOAL, highlight_from=None):
    """a one-row binary strip; cells from `highlight_from` on are drawn red."""
    n = len(vector)
    split = n if highlight_from is None else highlight_from

    ax.imshow(
        [vector[:split]], 
        cmap=ListedColormap(["white", color]),
        vmin=0,
        vmax=1,
        aspect="auto",
        extent=(0, split, 0, 1)
    )

    if split < n:
        ax.imshow(
            [vector[split:]],
            cmap=ListedColormap(["white", _RED]),
            vmin=0,
            vmax=1,
            aspect="auto",
            extent=(split, n, 0, 1)
        )

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel(label, color=_CHARCOAL, fontsize=8)
    for spine in ax.spines.values():
        spine.set_color(_CHARCOAL)


def _plot_frame_list(ax, tab_frames, i, window, color_fn=None):
    """the list of frames around frame `i`, each line colored by `color_fn(j)`."""
    if color_fn is None:
        color_fn = lambda j: _CHARCOAL if j == i else "lightgray"

    start = max(0, min(i - window // 2, len(tab_frames) - window))
    end = min(len(tab_frames), start + window)

    for j in range(start, end):
        ax.text(0, end - j, f"{j}: {tab_frames[j]}", color=color_fn(j), va="center", fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, window + 1)
    ax.axis("off")


def _animate_to_html(fig, update, n_frames, interval):
    """render the animation as a looping GIF embedded in an HTML img tag."""
    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval)

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        path = tmp.name

    anim.save(path, writer="pillow", fps=1000 / interval, dpi=200)
    plt.close(fig)

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")

    os.remove(path)

    return HTML(f'<img src="data:image/gif;base64,{encoded}" style="max-width: 100%;">')


def animate_tab_frame_encoding(tab_frames):
    """the track frame by frame, each one as a fretboard grid and its 150-element vector."""
    fig = plt.figure(figsize=(9, 2.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[6, 1], hspace=0.6)

    ax_list = fig.add_subplot(gs[:, 0])
    ax_grid = fig.add_subplot(gs[0, 1])
    ax_vec = fig.add_subplot(gs[1, 1])

    def update(i):
        ax_list.clear()
        ax_grid.clear()
        ax_vec.clear()

        _plot_frame_list(ax_list, tab_frames, i, window=5)
        _plot_grid(convert_positions_to_tab_grid(tab_frames[i]), ax=ax_grid, title=f"{tab_frames[i]}")
        _plot_vector_strip(ax_vec, convert_positions_to_tab_vector(tab_frames[i]), "150-element vector")

    return _animate_to_html(fig, update, len(tab_frames), interval=400)


def animate_training_pair_construction(tab_frames):
    """the track frame by frame, each one building its 728-element input from 600 of history and 128 of pitches."""
    context = 4
    fig = plt.figure(figsize=(9, 3.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 2], height_ratios=[1, 1, 1])

    ax_list = fig.add_subplot(gs[:, 0])
    ax_600 = fig.add_subplot(gs[0, 1])
    ax_128 = fig.add_subplot(gs[1, 1])
    ax_728 = fig.add_subplot(gs[2, 1])

    X, _, _ = convert_track_to_input_and_candidate_mask_arrays(tab_frames, context=context)

    def update(i):
        ax_list.clear()
        ax_600.clear()
        ax_128.clear()
        ax_728.clear()

        x = X[i]
        context_idx = set(range(i - context, i)) & set(range(len(tab_frames)))

        def color_for(j):
            if j == i:
                return _RED
            if j in context_idx:
                return _CHARCOAL
            return "lightgray"

        _plot_frame_list(ax_list, tab_frames, i, window=8, color_fn=color_for)
        _plot_vector_strip(ax_600, x[:600], "600-element history (4 previous frames)")
        _plot_vector_strip(ax_128, x[600:], "128-element MIDI pitches (current frame)", color=_RED)
        _plot_vector_strip(ax_728, x, "728-element input", highlight_from=600)

    return _animate_to_html(fig, update, len(tab_frames), interval=400)


def animate_candidate_positions(tab_frames):
    """the track frame by frame, the positions played next to its candidate mask."""
    fig = plt.figure(figsize=(9, 3.4), constrained_layout=True)

    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2])

    ax_list = fig.add_subplot(gs[:, 0])
    ax_actual = fig.add_subplot(gs[0, 1])
    ax_candidates = fig.add_subplot(gs[1, 1])

    _, M, _ = convert_track_to_input_and_candidate_mask_arrays(tab_frames)

    def update(i):
        ax_list.clear()
        ax_actual.clear()
        ax_candidates.clear()

        _plot_frame_list(ax_list, tab_frames, i, window=5)
        _plot_grid(convert_positions_to_tab_grid(tab_frames[i]), ax=ax_actual, title="played")
        _plot_grid(M[i].reshape(6, 25), ax=ax_candidates, title="possible")

    return _animate_to_html(fig, update, len(tab_frames), interval=400)


def _plot_comparison_grid(grid, ax, title, cmap=None, labelsize=7, bare=False, threshold=None):
    """a grid for the comparison figure; `bare` drops the axes, `threshold` paints predicted cells red."""
    grid = np.asarray(grid)
    _plot_grid(grid, ax=ax, cmap=cmap, title=title, aspect="equal")

    if threshold is not None:
        played = np.ma.masked_where(grid <= threshold, np.ones_like(grid))
        ax.imshow(played, cmap=ListedColormap([_RED]), vmin=0, vmax=1, aspect="equal")

    ax.title.set_fontsize(labelsize + 3)

    if bare:
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_xticks([])
        ax.set_yticks([])
        return ax

    ax.tick_params(labelsize=labelsize)
    ax.xaxis.label.set_fontsize(labelsize + 1)
    ax.yaxis.label.set_fontsize(labelsize + 1)
    return ax


def plot_model_comparison(x, y_true, mask, probs_baseline, probs_masked, context=4):
    """one frame end to end, history on top, then candidates and target, then both models' predictions."""
    def block_title(subfig, text):
        subfig.suptitle(text, color=_CHARCOAL, fontsize=13, fontweight="medium")

    fig = plt.figure(figsize=(6, 5.4), constrained_layout=True)
    top, middle, bottom = fig.subfigures(3, 1, height_ratios=[0.7, 1, 1])

    history = np.asarray(x[: context * 150]).reshape(context, 6, 25)
    for i, ax in enumerate(top.subplots(1, context)):
        _plot_comparison_grid(history[i], ax, f"i-{context - i}", labelsize=6, bare=True)
    block_title(top, "history (past positions)")

    for ax, (grid, grid_title) in zip(middle.subplots(1, 2), ((mask, "candidates"), (y_true, "actual"))):
        _plot_comparison_grid(np.asarray(grid).reshape(6, 25), ax, grid_title)
    block_title(middle, "current frame")

    for ax, (probs, probs_title) in zip(bottom.subplots(1, 2), ((probs_baseline, "baseline"), (probs_masked, "candidate mask"))):
        _plot_comparison_grid(np.asarray(probs).reshape(6, 25), ax, probs_title, cmap="gray_r", threshold=0.5)
    block_title(bottom, "predicted probabilities")
    
    return fig


# curve plots


def plot_loss_curves(histories, ylim=(0, 0.05)):
    """train (dashed) and validation (solid) loss per epoch, one color per run."""
    _, ax = plt.subplots(figsize=(7, 3.5))

    for (label, history), color in zip(histories.items(), (_CHARCOAL, _RED)):
        epochs = [h["epoch"] for h in history]
        ax.plot(epochs, [h["val_loss"] for h in history], color=color, label=f"{label}, validation")
        ax.plot(epochs, [h["train_loss"] for h in history], color=color, linestyle="--", alpha=0.5, label=f"{label}, train")

    ax.set_xlabel("epoch", color=_CHARCOAL)
    ax.set_ylabel("loss", color=_CHARCOAL)

    ax.tick_params(colors=_CHARCOAL)
    for spine in ax.spines.values():
        spine.set_color(_CHARCOAL)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(frameon=False, labelcolor=_CHARCOAL, fontsize=9, loc="upper right")

    ax.set_ylim(ylim)
    ax.figure.tight_layout()

    return ax


def plot_scaling_curve(metrics):
    """test exact match against training size, one line per model."""
    _, ax = plt.subplots(figsize=(7, 3.5))

    sizes = sorted(metrics)
    for (name, label), color in zip((("baseline", "no mask"), ("masked", "candidate mask")), (_CHARCOAL, _RED)):
        scores = [metrics[n]["exact_match"][name]["test"]["overall"] * 100 for n in sizes]
        ax.plot(sizes, scores, color=color, marker="o", markersize=4, label=label)

    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{n:,}" for n in sizes])
    ax.set_xlabel("training tracks", color=_CHARCOAL)
    ax.set_ylabel("exact match (%)", color=_CHARCOAL)
    ax.tick_params(colors=_CHARCOAL)
    for spine in ax.spines.values():
        spine.set_color(_CHARCOAL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, labelcolor=_CHARCOAL, fontsize=9, loc="lower right")
    ax.figure.tight_layout()
    
    return ax

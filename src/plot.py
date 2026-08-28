"""Plots and animations for the article."""

import base64
import logging
import os
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap
from IPython.display import HTML
from manim import *
import manimpango

logging.getLogger("matplotlib.animation").setLevel(logging.WARNING)

from src.transform import (
    convert_positions_to_tab_grid,
    convert_positions_to_tab_vector,
    convert_tab_frames_to_input_and_target,
    split_alphatex_into_track_blocks,
    parse_alphatex_track,
    tokenize_track_body,
    extract_positions_from_token,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent

manimpango.register_font(str(_REPO_ROOT / "article" / "assets" / "fonts" / "SpaceGrotesk-VariableFont_wght.ttf"))

config.background_color = WHITE
config.frame_width = 14
config.frame_height = 5.2
config.pixel_width = 2560
config.pixel_height = 952
config.frame_rate = 30
_manim_cache_dir = tempfile.mkdtemp()
config.media_dir = _manim_cache_dir
config.video_dir = str(_REPO_ROOT / "article" / "assets" / "animations")
config.partial_movie_dir = str(Path(_manim_cache_dir) / "partial_movie_files")
Text.set_default(font="Space Grotesk", color=BLACK)

_TOKEN_FONT_SIZE = 15

_NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_CHARCOAL = "#525252"
_RED = "#FF160A"


def _pitch_to_note_name(pitch: int) -> str:
    return f"{_NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def print_midi_pitch_values(frames, as_note_names=False):
    """Print each frame's active MIDI pitches; as note names if `as_note_names` is set."""
    for frame in frames:
        pitches = tuple(pitch for pitch, active in enumerate(frame.pitches) if active)
        if as_note_names:
            pitches = tuple(_pitch_to_note_name(p) for p in pitches)
        print(f"tick: {frame.tick} - pitches: {pitches}")


def _plot_grid(grid, ax=None, cmap=None, color=_CHARCOAL, vmin=0, vmax=1, title=None):
    """Draw a 6x25 fretboard grid of values, binary or continuous depending on `cmap`."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 2))
    if cmap is None:
        cmap = ListedColormap(["white", color])

    ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

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


def _plot_tablature_frame(frame, ax=None):
    """Draw a tablature frame's positions on a fretboard grid."""
    return _plot_grid(convert_positions_to_tab_grid(frame), ax=ax, title=f"{frame}")

def _plot_vector_strip(ax, vector, label, color=_CHARCOAL, highlight_from=None):
    """Draw a 1-row binary strip, highlighting everything from `highlight_from` onward."""
    n = len(vector)
    split = n if highlight_from is None else highlight_from

    ax.imshow([vector[:split]], cmap=ListedColormap(["white", color]), vmin=0, vmax=1, aspect="auto", extent=(0, split, 0, 1))

    if split < n:
        ax.imshow([vector[split:]], cmap=ListedColormap(["white", _RED]), vmin=0, vmax=1, aspect="auto", extent=(split, n, 0, 1))

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_xlabel(label, color=_CHARCOAL, fontsize=8)
    for spine in ax.spines.values():
        spine.set_color(_CHARCOAL)


def _plot_frame_list(ax, tab_frames, i, window, color_fn):
    """Draw the sliding list of frames around frame `i`, each line colored by `color_fn(j)`."""
    start = max(0, min(i - window // 2, len(tab_frames) - window))
    end = min(len(tab_frames), start + window)

    for j in range(start, end):
        ax.text(0, end - j, f"{j}: {tab_frames[j]}", color=color_fn(j), va="center", fontsize=9)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, window + 1)
    ax.axis("off")


def _animate_to_html(fig, update, n_frames, interval):
    """Render a matplotlib animation as a looping GIF, returned as an embeddable HTML img tag."""
    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval)

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        path = tmp.name

    anim.save(path, writer="pillow", fps=1000 / interval, dpi=200)
    plt.close(fig)

    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("ascii")

    os.remove(path)

    return HTML(f'<img src="data:image/gif;base64,{encoded}" style="max-width: 100%;">')


def animate_tab_frame_encoding(tab_frames, window=5, interval=400):
    """Animate a track playing frame by frame, each one as a fretboard grid and vector."""
    fig = plt.figure(figsize=(9, 2.6), constrained_layout=True)

    gs = fig.add_gridspec(2, 2, width_ratios=[1, 2], height_ratios=[6, 1], hspace=0.6)

    ax_list = fig.add_subplot(gs[:, 0])
    ax_grid = fig.add_subplot(gs[0, 1])
    ax_vec = fig.add_subplot(gs[1, 1])

    def update(i):
        ax_list.clear()
        ax_grid.clear()
        ax_vec.clear()

        _plot_frame_list(ax_list, tab_frames, i, window, lambda j: _CHARCOAL if j == i else "lightgray")
        _plot_tablature_frame(tab_frames[i], ax=ax_grid)
        _plot_vector_strip(ax_vec, convert_positions_to_tab_vector(tab_frames[i]), "150-element vector")

    return _animate_to_html(fig, update, len(tab_frames), interval)


def animate_training_pair_construction(tab_frames, context=4, window=8, interval=400):
    """Animate a track playing frame by frame, each one building its 600, 128, and 728-element input."""
    fig = plt.figure(figsize=(9, 3.2), constrained_layout=True)
    
    gs = fig.add_gridspec(3, 2, width_ratios=[1, 2], height_ratios=[1, 1, 1])

    ax_list = fig.add_subplot(gs[:, 0])
    ax_600 = fig.add_subplot(gs[0, 1])
    ax_128 = fig.add_subplot(gs[1, 1])
    ax_728 = fig.add_subplot(gs[2, 1])

    def update(i):
        ax_list.clear()
        ax_600.clear()
        ax_128.clear()
        ax_728.clear()

        x, _ = convert_tab_frames_to_input_and_target(tab_frames, i, context=context)
        context_idx = set(range(i - context, i)) & set(range(len(tab_frames)))

        def color_for(j):
            if j == i:
                return _RED
            if j in context_idx:
                return _CHARCOAL
            return "lightgray"

        _plot_frame_list(ax_list, tab_frames, i, window, color_for)
        _plot_vector_strip(ax_600, x[:600], "600-element history (4 previous frames)")
        _plot_vector_strip(ax_128, x[600:], "128-element MIDI pitches (current frame)", color=_RED)
        _plot_vector_strip(ax_728, x, "728-element input", highlight_from=600)

    return _animate_to_html(fig, update, len(tab_frames), interval)


def _positions_to_display(positions):
    """Format a frame's positions as fret.string text."""
    if not positions:
        return "r"

    if len(positions) == 1:
        fret, string = positions[0]
        return f"{fret}.{string}"

    return "(" + " ".join(f"{fret}.{string}" for fret, string in positions) + ")"


def _flow_layout(mobjects, max_width, buff_x=0.25, buff_y=0.35):
    """Wrap mobjects into rows past `max_width`."""
    rows = [[]]
    current_width = 0

    for m in mobjects:
        w = m.width
        if current_width + w > max_width and rows[-1]:
            rows.append([])
            current_width = 0
        rows[-1].append(m)
        current_width += w + buff_x

    row_groups = [VGroup(*r).arrange(RIGHT, buff=buff_x) for r in rows]
    return VGroup(*row_groups).arrange(DOWN, buff=buff_y, aligned_edge=LEFT)


class TrackTokenization(Scene):
    def construct(self):
        df = pd.read_parquet(_REPO_ROOT / "data" / "raw" / "data.parquet")
        example = df.iloc[0]

        track_blocks = split_alphatex_into_track_blocks(example["text"])
        track = parse_alphatex_track(track_blocks[0])

        title = Text("Tokenize", font_size=22).to_edge(UP, buff=0.4)

        tokens = tokenize_track_body(track["body"])
        mobs = [Text(t, font_size=_TOKEN_FONT_SIZE, color=BLACK) for t in tokens]

        group = _flow_layout(mobs, max_width=13, buff_x=0.15, buff_y=0.22)
        group.next_to(title, DOWN, buff=0.5)

        removed = [m for t, m in zip(tokens, mobs) if extract_positions_from_token(t) is None]
        survivors = [m for t, m in zip(tokens, mobs) if extract_positions_from_token(t) is not None]

        self.play(FadeIn(title), FadeIn(group), run_time=0.8)
        self.wait(1.2)

        def set_step(label):
            new_title = Text(label, font_size=22).to_edge(UP, buff=0.4)
            self.play(Transform(title, new_title), run_time=0.6)
            self.wait(0.6)

        self.play(*[m.animate.set_color(_RED) for m in removed], run_time=0.6)
        self.wait(1.0)

        set_step("Filter")
        self.play(*[FadeOut(m) for m in removed], run_time=0.8)
        self.wait(1.5)

        survivor_tokens = [t for t in tokens if extract_positions_from_token(t) is not None]

        cleanups = [
            (m, Text(cleaned, font_size=_TOKEN_FONT_SIZE, color=BLACK).move_to(m))
            for t, m in zip(survivor_tokens, survivors)
            for cleaned in [_positions_to_display(extract_positions_from_token(t))]
            if cleaned != t
        ]

        if cleanups:
            set_step("Simplify")
            self.play(*[m.animate.set_color(_RED) for m, _ in cleanups], run_time=0.5)
            self.wait(0.6)
            self.play(*[Transform(m, cleaned) for m, cleaned in cleanups], run_time=0.7)
            self.wait(1.0)

        targets = VGroup(*[m.copy() for m in survivors])
        _flow_layout(targets, max_width=13, buff_x=0.15, buff_y=0.22).next_to(title, DOWN, buff=0.5)
        self.play(*[m.animate.move_to(t.get_center()) for m, t in zip(survivors, targets)], run_time=0.8)
        self.wait(2.0)

        set_step("Encode")

        frame_positions = [extract_positions_from_token(t) for t in survivor_tokens]

        rests = [m for m, positions in zip(survivors, frame_positions) if not positions]

        if rests:
            self.play(*[m.animate.set_color(_RED) for m in rests], run_time=0.5)
            self.wait(0.6)

        encoded_texts = [
            Text(
                repr(positions) + ("," if i < len(frame_positions) - 1 else ""),
                font_size=_TOKEN_FONT_SIZE,
                color=BLACK,
            )
            for i, positions in enumerate(frame_positions)
        ]

        encoded_group = _flow_layout(encoded_texts, max_width=13, buff_x=0.15, buff_y=0.22)

        available_width = 13
        available_height = (title.get_bottom()[1] - 0.5) - (-config.frame_height / 2 + 0.4)

        if encoded_group.width > available_width or encoded_group.height > available_height:
            encoded_group.scale(min(available_width / encoded_group.width, available_height / encoded_group.height))

        encoded_group.next_to(title, DOWN, buff=0.5)

        self.play(*[Transform(m, encoded) for m, encoded in zip(survivors, encoded_texts)], run_time=1.2)
        self.wait(2.5)

        self.play(FadeOut(VGroup(*survivors)), run_time=0.6)

"""Turns MIDI and alphaTex source formats into frame objects, and those into the vectors the model consumes."""

import re
import mido


class MidiFrame:
    """A 128-element binary vector snapshot of which MIDI pitches are active at a given tick."""

    def __init__(self, tick: int, pitches: tuple[int, ...]):
        self.tick = tick
        self.pitches = pitches

    def __repr__(self) -> str:
        return f"MidiFrame(tick={self.tick}, pitches={self.pitches})"

    def __eq__(self, other) -> bool:
        return self.tick == other.tick and self.pitches == other.pitches

    def __hash__(self) -> int:
        return hash((self.tick, self.pitches))


def convert_midi_pitches_to_binary_vector(active: set[int]) -> tuple[int, ...]:
    """Turn a set of active MIDI pitches into a 128-element binary vector, one slot per pitch, 1 where active."""
    return tuple(1 if pitch in active else 0 for pitch in range(128))


def convert_midi_track_to_frames(track: mido.MidiTrack) -> list[MidiFrame]:
    """Walk a MIDI track and emit one frame per change in the active pitch set."""
    active = set()
    frames: list[MidiFrame] = []
    tick = 0

    for msg in track:
        tick += msg.time

        if msg.type == "note_on" and msg.velocity > 0:
            active.add(msg.note)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            active.discard(msg.note)
        else:
            continue

        vector = convert_midi_pitches_to_binary_vector(active)

        if frames and frames[-1].tick == tick:
            frames[-1] = MidiFrame(tick, vector)
        else:
            frames.append(MidiFrame(tick, vector))

    return frames


def split_alphatex_into_track_blocks(text: str) -> list[str]:
    """Split a song's alphaTex text into raw per-track blocks, one per `\\track` marker."""
    return text.split("\\track")[1:]


def parse_alphatex_track(track: str) -> dict:
    """Parse a track block's `\\instrument`/`\\tuning`/`\\frets` header, return its tuning, fret count, and note body."""
    track_header_re = re.compile(
        r"\\instrument (?P<instrument>\d+)\s*\n"
        r"\\tuning (?P<tuning>[^\n]+)\n"
        r"\\frets (?P<frets>\d+)\n"
    )
    m = track_header_re.match(track.lstrip("\n"))

    if not m:
        raise ValueError(f"could not parse track header: {track.splitlines()[0]}")

    return {
        "tuning": m.group("tuning").split(),
        "frets": int(m.group("frets")),
        "body": track[m.end():].strip(),
    }


STANDARD_TUNING = ["E5", "B4", "G4", "D4", "A3", "E3"]


def is_eligible_track(track: dict) -> bool:
    """Check whether a track is eligible for training, meaning standard-tuning guitar with 24 frets."""
    return track["tuning"] == STANDARD_TUNING and track["frets"] == 24


def tokenize_track_body(body: str) -> list[str]:
    """Split a track body into tokens, tracking paren/brace depth so chords and multi-word techniques stay intact."""

    tokens = []
    current = []
    depth = 0

    for ch in body:
        if ch in "({":
            depth += 1
            current.append(ch)
        elif ch in ")}":
            depth -= 1
            current.append(ch)
        elif ch.isspace() and depth == 0:
            if current:
                tokens.append("".join(current))
                current = []
        else:
            current.append(ch)
    if current:
        tokens.append("".join(current))

    return tokens


def extract_positions_from_token(token: str) -> list[tuple[int, int]] | None:
    """Return the (fret, string) pairs a token holds down, or None for tokens that carry no position."""

    technique_brace_re = re.compile(r"\{[^}]*\}")
    note_re = re.compile(r"^(\d+)\.(\d+)")

    if token.startswith(":") or token == "|":
        return None
    if token.startswith("r"):
        return []
    if token.startswith("("):
        inner = token[1:token.rindex(")")]
        return [
            tuple(int(x) for x in note_re.match(technique_brace_re.sub("", note)).groups())
            for note in tokenize_track_body(inner)
        ]

    return [tuple(int(x) for x in note_re.match(technique_brace_re.sub("", token)).groups())]


def convert_track_body_to_tab_frames(body: str) -> list[list[tuple[int, int]]]:
    """Tokenize a track body and extract its sequence of (fret, string) frames, dropping frets above 24."""

    tab_frames = []

    for tok in tokenize_track_body(body):
        positions = extract_positions_from_token(tok)
        if positions is None:
            continue
        tab_frames.append([(f, s) for f, s in positions if f <= 24])

    return tab_frames


def convert_positions_to_midi_pitches(positions: list[tuple[int, int]]) -> set[int]:
    """Convert a frame's (fret, string) positions to MIDI pitches, via standard-tuning open-string pitches."""
    open_string_pitch = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}
    return {open_string_pitch[string] + fret for fret, string in positions}


def convert_positions_to_tab_grid(positions: list[tuple[int, int]]) -> list[list[int]]:
    """Turn a frame's (fret, string) positions into a 6x25 binary grid, one row per string, one column per fret, 1 where a fret is held."""
    grid = [[0] * 25 for _ in range(6)]
    for fret, string in positions:
        grid[string - 1][fret] = 1
    return grid


def convert_positions_to_tab_vector(positions: list[tuple[int, int]]) -> tuple[int, ...]:
    """Turn a frame's (fret, string) positions into a 150-element binary vector (6 strings x 25 frets, flattened), 1 where a fret is held."""
    return tuple(cell for row in convert_positions_to_tab_grid(positions) for cell in row)


def convert_tab_frames_to_input_and_target(
    tab_frames: list[list[tuple[int, int]]], i: int, context: int = 4
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Build the 728-element input and 150-element target for frame `i`.

    The input is the previous `context` tab frames plus the current frame's midi pitches,
    zero-padded before the track's first frames.
    """
    history = []
    for j in range(i - context, i):
        history.extend(convert_positions_to_tab_vector(tab_frames[j]) if j >= 0 else [0] * 150)

    pitches = convert_midi_pitches_to_binary_vector(convert_positions_to_midi_pitches(tab_frames[i]))

    x = tuple(history) + tuple(pitches)
    y = convert_positions_to_tab_vector(tab_frames[i])
    return x, y


def convert_alphatex_text_to_inputs_and_targets(
    text: str, context: int = 4
) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    """Turn one dataset row's alphaTex text into every (input, target) pair.

    Splits the text into tracks, keeps only the eligible ones, and vectorizes each track's
    frames into (input, target) pairs via convert_tab_frames_to_input_and_target.
    """
    pairs = []
    for block in split_alphatex_into_track_blocks(text):
        track = parse_alphatex_track(block)
        if not is_eligible_track(track):
            continue
        tab_frames = convert_track_body_to_tab_frames(track["body"])
        pairs.extend(
            convert_tab_frames_to_input_and_target(tab_frames, i, context=context)
            for i in range(len(tab_frames))
        )
    return pairs

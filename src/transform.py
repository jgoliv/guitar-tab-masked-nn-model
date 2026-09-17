import re
import numpy as np


STANDARD_TUNING = ["E5", "B4", "G4", "D4", "A3", "E3"]
OPEN_STRING_PITCH = {1: 64, 2: 59, 3: 55, 4: 50, 5: 45, 6: 40}

_TECHNIQUE_BRACE_RE = re.compile(r"\{[^}]*\}")
_NOTE_RE = re.compile(r"^(\d+)\.(\d+)")
_TRACK_HEADER_RE = re.compile(
    r"\\instrument (?P<instrument>\d+)\s*\n"
    r"\\tuning (?P<tuning>[^\n]+)\n"
    r"\\frets (?P<frets>\d+)\n"
)

def split_alphatex_into_track_blocks(text):
    """split a song's alphaTex into one block per `\\track`."""
    return text.split("\\track")[1:]


def parse_alphatex_track(track: str) -> dict:
    """parse a track block's header; return its tuning, fret count, and note body."""
    m = _TRACK_HEADER_RE.match(track.lstrip("\n"))

    if not m:
        raise ValueError(f"could not parse track header: {track.splitlines()[0]}")

    return {
        "tuning": m.group("tuning").split(),
        "frets": int(m.group("frets")),
        "body": track[m.end():].strip(),
    }


def is_eligible_track(track):
    """true for standard-tuning tracks with 24 frets."""
    return track["tuning"] == STANDARD_TUNING and track["frets"] == 24


def tokenize_track_body(body):
    """split on whitespace, except inside `()` and `{}`."""
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


def extract_positions_from_token(token):
    """(fret, string) pairs in a token; [] for a rest, None for tokens that aren't a frame."""
    if token.startswith(":") or token == "|":
        return None
    if token.startswith("r"):
        return []
    if token.startswith("("):
        inner = token[1:token.rindex(")")]
        return [
            tuple(int(x) for x in _NOTE_RE.match(_TECHNIQUE_BRACE_RE.sub("", note)).groups())
            for note in tokenize_track_body(inner)
        ]

    return [tuple(int(x) for x in _NOTE_RE.match(_TECHNIQUE_BRACE_RE.sub("", token)).groups())]


def convert_track_body_to_tab_frames(body):
    """the track as frames, with frets above 24 dropped."""
    tab_frames = []

    for tok in tokenize_track_body(body):
        positions = extract_positions_from_token(tok)
        if positions is None:
            continue
        tab_frames.append([(f, s) for f, s in positions if f <= 24])

    return tab_frames


def trim_leading_rests(tab_frames):
    """leading rests add nothing the zero-padded history doesn't."""
    for i, frame in enumerate(tab_frames):
        if frame:
            return tab_frames[i:]
    return []


def convert_positions_to_midi_pitches(positions):
    """pitch of each position, open-string pitch plus fret."""
    return {OPEN_STRING_PITCH[string] + fret for fret, string in positions}


def candidate_positions(pitch):
    """every (fret, string) that produces `pitch`, frets 0 to 24."""
    return [
        (pitch - open_pitch, string)
        for string, open_pitch in OPEN_STRING_PITCH.items()
        if 0 <= pitch - open_pitch <= 24
    ]


def convert_positions_to_tab_grid(positions):
    """the frame as a 6x25 grid, one row per string."""
    grid = [[0] * 25 for _ in range(6)]
    for fret, string in positions:
        grid[string - 1][fret] = 1
    return grid


def convert_positions_to_tab_vector(positions):
    """the 6x25 grid flattened into 150 cells, string by string."""
    return tuple(cell for row in convert_positions_to_tab_grid(positions) for cell in row)


def convert_track_to_input_and_candidate_mask_arrays(tab_frames, context= 4):
    """(X, M, y) for every frame of a track; M marks the positions that could produce its pitches."""
    n = len(tab_frames)
    tab_vectors = np.zeros((n, 150), dtype=np.uint8)
    pitch_vectors = np.zeros((n, 128), dtype=np.uint8)
    mask_vectors = np.zeros((n, 150), dtype=np.uint8)

    for i, positions in enumerate(tab_frames):
        for fret, string in positions:
            tab_vectors[i, (string - 1) * 25 + fret] = 1
        for pitch in convert_positions_to_midi_pitches(positions):
            if 0 <= pitch < 128:
                pitch_vectors[i, pitch] = 1
            for fret, string in candidate_positions(pitch):
                mask_vectors[i, (string - 1) * 25 + fret] = 1

    X = np.zeros((n, 728), dtype=np.uint8)

    for i in range(n):
        for k in range(context):
            j = i - context + k
            if j >= 0:
                X[i, k * 150 : (k + 1) * 150] = tab_vectors[j]
        X[i, context * 150 :] = pitch_vectors[i]

    return X, mask_vectors, tab_vectors


def convert_tracks_to_input_and_candidate_mask_arrays(tracks_tab_frames, context=4):
    """every track's (X, M, y) written into preallocated arrays, to avoid a concatenation copy."""
    tracks_tab_frames = list(tracks_tab_frames)
    n_frames = sum(len(tab_frames) for tab_frames in tracks_tab_frames)

    X = np.zeros((n_frames, 728), dtype=np.uint8)
    M = np.zeros((n_frames, 150), dtype=np.uint8)
    Y = np.zeros((n_frames, 150), dtype=np.uint8)

    row = 0
    
    for tab_frames in tracks_tab_frames:
        n = len(tab_frames)
        X[row : row + n], M[row : row + n], Y[row : row + n] = convert_track_to_input_and_candidate_mask_arrays(
            tab_frames, context=context
        )
        row += n

    return X, M, Y

from pathlib import Path
from datasets import load_dataset

import numpy as np
import pandas as pd

from src.transform import (
    convert_track_body_to_tab_frames,
    is_eligible_track,
    parse_alphatex_track,
    split_alphatex_into_track_blocks,
    trim_leading_rests,
)

def fetch_dataset(out_path=Path("data/raw/data.parquet")):
    dataset = load_dataset("vldsavelyev/guitar_tab", split="train")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_path)
    return out_path


def build_processed_tracks_dataset(in_path=Path("data/raw/data.parquet"), out_path=Path("data/processed/tab_tracks.parquet")):
    """cache every eligible, deduplicated track's frames as parquet."""
    df = pd.read_parquet(in_path)

    rows = []

    for _, row in df.iterrows():
        parsed = [parse_alphatex_track(b) for b in split_alphatex_into_track_blocks(row["text"])]
        for track in (t for t in parsed if is_eligible_track(t)):
            rows.append({"file": row["file"], "body": track["body"]})

    tracks = (
        pd.DataFrame(rows)
        .drop_duplicates(subset="body")
        .reset_index(drop=True)
        .assign(tab_frames=lambda df: df["body"].apply(convert_track_body_to_tab_frames).apply(trim_leading_rests))
        .drop(columns="body")
    )
    
    tracks = tracks[tracks["tab_frames"].apply(len) > 0].reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tracks.to_parquet(out_path)
    return out_path


def _normalize_cached_tab_frames(tab_frames):
    """parquet returns nested numpy arrays; turn them back into (fret, string) tuples."""
    return [[tuple(int(v) for v in position) for position in frame] for frame in tab_frames]


def load_processed_tracks_dataset(path=Path("data/processed/tab_tracks.parquet"), files: list | None = None):
    """load the cached tracks, optionally only those from `files`."""
    if files is not None:
        df = pd.read_parquet(path, filters=[("file", "in", files)])
    else:
        df = pd.read_parquet(path)
    df["tab_frames"] = df["tab_frames"].apply(_normalize_cached_tab_frames)
    return df


def split_tracks_by_song(df_files, n_train, n_val, n_test, seed=0) -> tuple[list, list, list]:
    """split by song, drawing test and val first so they don't change with `n_train`; returns file lists."""
    file_track_counts = df_files["file"].value_counts()

    rng = np.random.default_rng(seed)
    files = df_files["file"].unique().to_numpy()
    rng.shuffle(files)

    def take_files(files_iter, target_tracks):
        taken, count = [], 0
        for f in files_iter:
            taken.append(f)
            count += file_track_counts[f]
            if count >= target_tracks:
                break
        return taken

    files_iter = iter(files)
    test_files = take_files(files_iter, n_test)
    val_files = take_files(files_iter, n_val)
    train_files = take_files(files_iter, n_train)

    return train_files, val_files, test_files


if __name__ == "__main__":
    fetch_dataset()
    build_processed_tracks_dataset()

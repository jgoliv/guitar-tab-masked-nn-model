"""Fetching and caching the training dataset."""

from pathlib import Path
from datasets import load_dataset
import pandas as pd

from src.transform import (
    convert_track_body_to_tab_frames,
    is_eligible_track,
    parse_alphatex_track,
    split_alphatex_into_track_blocks,
)

def fetch_dataset(out_path: Path = Path("data/raw/data.parquet")) -> Path:
    """Download vldsavelyev/guitar_tab from Hugging Face and cache it as parquet."""
    dataset = load_dataset("vldsavelyev/guitar_tab", split="train")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out_path)
    return out_path


def build_processed_tracks_dataset(
    in_path: Path = Path("data/raw/data.parquet"), 
    out_path: Path = Path("data/processed/tab_tracks.parquet")
) -> Path:
    """Extract every eligible track from the raw dataset and cache its tab frames as parquet."""
    df = pd.read_parquet(in_path)

    rows = []

    for _, row in df.iterrows():
        parsed = [parse_alphatex_track(b) for b in split_alphatex_into_track_blocks(row["text"])]
        for track_idx, track in enumerate(t for t in parsed if is_eligible_track(t)):
            rows.append({"file": row["file"], "track_idx": track_idx, "body": track["body"]})

    tracks = (
        pd.DataFrame(rows)
        .drop_duplicates(subset="body")
        .reset_index(drop=True)
        .assign(tab_frames=lambda df: df["body"].apply(convert_track_body_to_tab_frames))
        .drop(columns="body")
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tracks.to_parquet(out_path)
    return out_path


if __name__ == "__main__":
    fetch_dataset()
    build_processed_tracks_dataset()

import glob
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from svd_search.config.paths import (
    MOVIE_META_DIR,
    RAW_DATA_DIR,
    WIKIPEDIA_CSV,
)
from svd_search.utils.utils import file_md5, parse_genres, parse_year

try:
    import kagglehub
    _KAGGLE_AVAILABLE = True
except ImportError:
    _KAGGLE_AVAILABLE = False

DATASET_SLUG = "jrobischon/wikipedia-movie-plots"


# ---------------------------------------------------------------------------
# Wikipedia Movie Plots (Kaggle)
# ---------------------------------------------------------------------------

def ensure_wikipedia_dataset() -> Path:
    """Download Wikipedia movie plots from Kaggle once; cache to data/raw/."""
    if WIKIPEDIA_CSV.exists():
        return WIKIPEDIA_CSV
    if not _KAGGLE_AVAILABLE:
        raise RuntimeError("kagglehub not installed. Run: pip install kagglehub")
    dataset_path = kagglehub.dataset_download(DATASET_SLUG)
    csv_files = sorted(glob.glob(os.path.join(dataset_path, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found for {DATASET_SLUG}.")
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(csv_files[0], WIKIPEDIA_CSV)
    return WIKIPEDIA_CSV


def load_wikipedia_dataset() -> tuple[pd.DataFrame, dict]:
    """Load & normalise Wikipedia Movie Plots dataset."""
    csv_path = ensure_wikipedia_dataset()
    raw_df = pd.read_csv(csv_path, low_memory=False)
    data_fingerprint = file_md5(csv_path)

    norm_df = pd.DataFrame({
        "id":          np.arange(1, len(raw_df) + 1).astype(str),
        "title":       raw_df["Title"].fillna("Unknown Title").astype(str),
        "genre":       raw_df["Genre"].apply(parse_genres),
        "description": raw_df["Plot"].fillna("").astype(str),
        "year":        raw_df["Release Year"].apply(parse_year),
    })
    norm_df["description"] = np.where(
        norm_df["description"].str.strip().eq(""),
        norm_df["title"],
        norm_df["description"],
    )
    norm_df = norm_df.dropna(subset=["title"]).reset_index(drop=True)

    mapping = {
        "source_label":      "Wikipedia Movie Plots (Kaggle)",
        "data_file":         csv_path.name,
        "data_fingerprint":  data_fingerprint,
        "raw_shape":         raw_df.shape,
        "normalized_shape":  norm_df.shape,
        "mapped_columns": {
            "title": "Title", "genre": "Genre",
            "description": "Plot", "year": "Release Year",
        },
    }
    return norm_df, mapping


# ---------------------------------------------------------------------------
# CMU MovieSummaries
# ---------------------------------------------------------------------------

def _parse_freebase_dict(value) -> list[str]:
    if pd.isna(value) or not str(value).strip():
        return ["Unknown"]
    try:
        d = json.loads(str(value).replace("'", '"'))
        values = [v.strip() for v in d.values() if v and v.strip()]
        return values if values else ["Unknown"]
    except Exception:
        parts = re.findall(r'"([^"]+)"', str(value))
        parts = [p for p in parts if not p.startswith("/m/")]
        return parts if parts else ["Unknown"]


def load_cmu_dataset() -> tuple[pd.DataFrame, dict]:
    """Load & join CMU MovieSummaries metadata and plot summaries."""
    meta_path  = MOVIE_META_DIR / "movie.metadata.tsv"
    plots_path = MOVIE_META_DIR / "plot_summaries.txt"

    meta_raw = pd.read_csv(
        meta_path, sep="\t", header=None,
        names=["wiki_id", "freebase_id", "title", "release_date",
               "revenue", "runtime", "languages", "countries", "genres"],
    )
    plots_raw = pd.read_csv(
        plots_path, sep="\t", header=None, names=["wiki_id", "plot"]
    )

    df = pd.merge(
        meta_raw[["wiki_id", "title", "release_date", "genres"]],
        plots_raw,
        on="wiki_id",
        how="inner",
    )
    df["genre"]       = df["genres"].apply(_parse_freebase_dict)
    df["year"]        = df["release_date"].apply(parse_year)
    df["plot"]        = df["plot"].fillna("").astype(str)
    df["title"]       = df["title"].fillna("Unknown Title").astype(str)
    df["description"] = df["plot"]   # alias kept for Streamlit compatibility
    df = df.drop(columns=["genres", "release_date"]).reset_index(drop=True)

    mapping = {
        "source_label":     "CMU MovieSummaries",
        "data_file":        str(meta_path),
        "data_fingerprint": file_md5(meta_path),
        "normalized_shape": df.shape,
        "mapped_columns": {
            "title": "title", "genre": "genres",
            "description": "plot", "year": "release_date",
        },
    }
    return df, mapping


# Backward-compatible aliases (used by main.ipynb and search_pipeline.py shim)
load_dataset   = load_wikipedia_dataset
ensure_dataset = ensure_wikipedia_dataset

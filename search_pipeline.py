"""Backward-compatibility shim.

All real logic has moved to src/svd_search/.
This file re-exports the original public API so existing notebooks
(notebooks/main.ipynb) and any external scripts continue to work unchanged.
"""
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from svd_search.config.paths import ARTIFACT_V1 as ARTIFACT_PATH  # noqa: F401
from svd_search.data.loader import (  # noqa: F401
    ensure_dataset,
    ensure_wikipedia_dataset,
    load_dataset,
    load_wikipedia_dataset,
)
from svd_search.features.build_features import build_corpus, build_model  # noqa: F401
from svd_search.models.evaluate import explain_results  # noqa: F401
from svd_search.models.predict import load_artifacts, search_movies  # noqa: F401
from svd_search.models.train import make_artifacts, save_artifacts  # noqa: F401
from svd_search.utils.utils import PipelineArtifacts  # noqa: F401

# Legacy path constant (points to old Wikipedia artifact)
DATA_FILE = Path("data/raw/jrobischon_wikipedia-movie-plots.csv")


import joblib
import kagglehub
import numpy as np

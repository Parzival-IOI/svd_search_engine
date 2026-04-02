"""Centralised path constants resolved from the project root."""
from pathlib import Path

# Project root = four levels up from this file:
#   src/svd_search/config/paths.py → root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIR       = PROJECT_ROOT / "data"
RAW_DATA_DIR   = DATA_DIR / "raw"
EXT_DATA_DIR   = DATA_DIR / "external"
PROC_DATA_DIR  = DATA_DIR / "processed"

CORENLP_DIR    = EXT_DATA_DIR / "corenlp_plot_summaries"
MOVIE_META_DIR = EXT_DATA_DIR / "MovieSummaries"

MODELS_DIR  = PROJECT_ROOT / "models"
ARTIFACT_V1 = MODELS_DIR / "search_artifacts.joblib"
ARTIFACT_V2 = MODELS_DIR / "search_artifacts_v2.joblib"
LEMMA_CACHE = MODELS_DIR / "corenlp_lemma_cache.joblib"

WIKIPEDIA_CSV = RAW_DATA_DIR / "jrobischon_wikipedia-movie-plots.csv"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
LOGS_DIR    = OUTPUTS_DIR / "logs"

NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

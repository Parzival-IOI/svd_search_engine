"""SVD-based Movie Search Engine via LSA + cosine similarity."""

from svd_search.config.paths import ARTIFACT_V1, ARTIFACT_V2, LEMMA_CACHE, MODELS_DIR
from svd_search.data.loader import (
    ensure_dataset,
    ensure_wikipedia_dataset,
    load_cmu_dataset,
    load_dataset,
    load_wikipedia_dataset,
)
from svd_search.features.build_features import build_corpus, build_lemma_corpus, build_model
from svd_search.models.evaluate import explain_results
from svd_search.models.predict import load_artifacts, search_movies
from svd_search.models.train import make_artifacts, save_artifacts
from svd_search.utils.utils import PipelineArtifacts

__all__ = [
    # Artifacts & paths
    "ARTIFACT_V1", "ARTIFACT_V2", "LEMMA_CACHE", "MODELS_DIR",
    # Data loading
    "load_wikipedia_dataset", "load_cmu_dataset",
    "load_dataset", "ensure_dataset", "ensure_wikipedia_dataset",
    # Feature engineering
    "build_corpus", "build_lemma_corpus", "build_model",
    # Model lifecycle
    "make_artifacts", "save_artifacts", "load_artifacts",
    # Inference
    "search_movies", "explain_results",
    # Types
    "PipelineArtifacts",
]

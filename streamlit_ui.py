"""Shared UI helpers for the SVD Movie Search Streamlit app.

All pages import from this module. It bootstraps sys.path so the svd_search
package (located in src/) is importable without requiring an editable install.
"""
import sys
from pathlib import Path

# Make src/ importable when running via `streamlit run app.py` from project root
_src = Path(__file__).resolve().parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import inspect

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from svd_search import (
    PipelineArtifacts,
    build_corpus,
    build_lemma_corpus,
    build_model,
    explain_results,
    load_artifacts,
    load_cmu_dataset,
    make_artifacts,
    save_artifacts,
    search_movies,
)

from svd_search.config.paths import ARTIFACT_V2, LEMMA_CACHE, MODELS_DIR

sns.set_theme(style="whitegrid")


@st.cache_data(show_spinner=False)
def _load_data():
    """Load CMU MovieSummaries dataset once per session."""
    return load_cmu_dataset()


@st.cache_resource(show_spinner=False)
def _build_pipeline(n_components: int):
    df, mapping = _load_data()
    artifacts, X_tfidf = make_artifacts(
        df,
        n_components=n_components,
        data_fingerprint=mapping.get("data_fingerprint", ""),
        use_lemmas=True,
    )
    return artifacts, X_tfidf, mapping


def ensure_state():
    for key in ("artifacts", "X_tfidf", "mapping"):
        st.session_state.setdefault(key, None)


def sidebar_controls():
    st.sidebar.header("Pipeline Setup")
    st.sidebar.caption("Corpus: CMU MovieSummaries + CoreNLP lemmas")
    n_components = st.sidebar.slider(
        "SVD components", min_value=10, max_value=300, value=150, step=10
    )
    c1, c2 = st.sidebar.columns(2)
    rebuild = c1.button("Rebuild")
    save    = c2.button("Save")
    return n_components, rebuild, save


def get_artifacts(n_components: int, rebuild: bool):
    ensure_state()

    if rebuild:
        _load_data.clear()
        _build_pipeline.clear()
        st.session_state["artifacts"] = None

    # 1) Try loading pre-built CoreNLP artifact from disk
    if ARTIFACT_V2.exists() and st.session_state["artifacts"] is None:
        with st.spinner("Loading CoreNLP artifacts…"):
            artifacts = load_artifacts(ARTIFACT_V2)
            st.session_state["artifacts"] = artifacts
            st.session_state["X_tfidf"]   = artifacts.tfidf.transform(artifacts.corpus)
            st.session_state["mapping"]    = {
                "source_label":    "CMU MovieSummaries (CoreNLP lemmatized)",
                "data_file":       str(ARTIFACT_V2),
                "mapped_columns":  {"title": "title", "genre": "genres",
                                    "description": "plot", "year": "release_date"},
                "normalized_shape": artifacts.df.shape,
            }

    # 2) Fall back: build in-memory from lemma cache + CMU metadata
    if st.session_state["artifacts"] is None:
        with st.spinner("Building pipeline from CoreNLP lemma cache…"):
            artifacts, X_tfidf, mapping = _build_pipeline(n_components)
            st.session_state["artifacts"] = artifacts
            st.session_state["X_tfidf"]   = X_tfidf
            st.session_state["mapping"]    = mapping

    return (
        st.session_state["artifacts"],
        st.session_state["X_tfidf"],
        st.session_state["mapping"],
    )


def save_current_artifacts():
    artifacts = st.session_state.get("artifacts")
    if artifacts is None:
        st.warning("No artifacts in memory yet. Build or load first.")
        return
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    saved = save_artifacts(artifacts, ARTIFACT_V2)
    st.success(f"Saved CoreNLP artifacts to: {saved}")


def show_code(title: str, fn):
    with st.expander(f"Code: {title}", expanded=False):
        st.code(inspect.getsource(fn), language="python")


def chart_year_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    sns.histplot(df["year"].dropna(), bins=20, kde=True, ax=ax, color="#3B82F6")
    ax.set_title("Movie Release Year Distribution")
    ax.set_xlabel("Year")
    return fig


def chart_genre(df: pd.DataFrame):
    genre_counts = df["genre"].explode().value_counts().head(20)
    fig, ax = plt.subplots(figsize=(11, 4))
    sns.barplot(
        x=genre_counts.index,
        y=genre_counts.values,
        hue=genre_counts.index,
        palette="viridis",
        legend=False,
        ax=ax,
    )
    ax.set_title("Genre Frequency")
    ax.set_xlabel("Genre")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def chart_top_terms(tfidf, X_tfidf):
    feature_names = np.array(tfidf.get_feature_names_out())
    avg_tfidf = np.asarray(X_tfidf.mean(axis=0)).ravel()
    top_idx = np.argsort(avg_tfidf)[-20:][::-1]

    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        x=avg_tfidf[top_idx],
        y=feature_names[top_idx],
        hue=feature_names[top_idx],
        palette="mako",
        legend=False,
        ax=ax,
    )
    ax.set_title("Top 20 Terms by Average TF-IDF")
    ax.set_xlabel("Average TF-IDF")
    ax.set_ylabel("Term")
    fig.tight_layout()
    return fig


def chart_explained_variance(svd):
    explained = svd.explained_variance_ratio_
    cum_explained = np.cumsum(explained)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(range(1, len(cum_explained) + 1), cum_explained, marker="o", markersize=3)
    ax.set_title("Cumulative Explained Variance (SVD Components)")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance Ratio")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    return fig, float(cum_explained[-1])


def chart_lsa_scatter(df: pd.DataFrame, X_lsa: np.ndarray):
    plot_df = pd.DataFrame(
        {
            "dim1": X_lsa[:, 0],
            "dim2": X_lsa[:, 1],
            "genre": df["genre"].apply(lambda g: g[0] if isinstance(g, list) and len(g) else "Unknown"),
        }
    )
    sample = plot_df.sample(min(500, len(plot_df)), random_state=42)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=sample, x="dim1", y="dim2", hue="genre", alpha=0.75, s=45, ax=ax)
    ax.set_title("Movies in 2D Latent Semantic Space (Sample)")
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")
    ax.legend(loc="best", fontsize=7, markerscale=0.8, framealpha=0.7)
    fig.tight_layout()
    return fig


def chart_result_scores(result_df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=result_df,
        y="title",
        x="similarity",
        hue="title",
        palette="crest",
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_title("Top Search Results by Similarity")
    ax.set_xlabel("Similarity")
    ax.set_ylabel("Movie Title")
    fig.tight_layout()
    return fig


def dataset_bounds(df: pd.DataFrame):
    year_min = int(np.nanmin(df["year"])) if df["year"].notna().any() else 1900
    year_max = int(np.nanmax(df["year"])) if df["year"].notna().any() else 2100
    return year_min, year_max


def run_search(
    query: str,
    artifacts: PipelineArtifacts,
    top_k: int,
    genre_filter: str,
    year_range: tuple[int, int],
):
    return search_movies(
        query=query,
        artifacts=artifacts,
        top_k=top_k,
        genre_filter=genre_filter,
        year_min=year_range[0],
        year_max=year_range[1],
    )


def process_functions_for_display():
    return [
        load_cmu_dataset,
        build_corpus,
        build_lemma_corpus,
        build_model,
        make_artifacts,
        search_movies,
        explain_results,
    ]

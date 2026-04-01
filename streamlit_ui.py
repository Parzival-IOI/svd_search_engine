import inspect
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from search_pipeline import (
    ARTIFACT_PATH,
    PipelineArtifacts,
    build_model,
    build_corpus,
    explain_results,
    load_artifacts,
    load_dataset,
    make_artifacts,
    save_artifacts,
    search_movies,
)

sns.set_theme(style="whitegrid")


@st.cache_data(show_spinner=False)
def _load_data():
    """Read the dataset CSV once per session; result is reused across all SVD component counts."""
    return load_dataset()


@st.cache_resource(show_spinner=False)
def _build_pipeline(n_components: int):
    df, mapping = _load_data()
    artifacts, X_tfidf = make_artifacts(
        df,
        n_components=n_components,
        data_fingerprint=mapping.get("data_fingerprint", ""),
    )
    return artifacts, X_tfidf, mapping


def ensure_state():
    if "artifacts" not in st.session_state:
        st.session_state["artifacts"] = None
    if "X_tfidf" not in st.session_state:
        st.session_state["X_tfidf"] = None
    if "mapping" not in st.session_state:
        st.session_state["mapping"] = None


def sidebar_controls():
    st.sidebar.header("Pipeline Setup")
    n_components = st.sidebar.slider("SVD components", min_value=10, max_value=300, value=100, step=10)

    c1, c2 = st.sidebar.columns(2)
    rebuild = c1.button("Rebuild")
    save = c2.button("Save")

    return n_components, rebuild, save


def get_artifacts(n_components: int, rebuild: bool):
    ensure_state()

    if rebuild:
        _load_data.clear()
        _build_pipeline.clear()
        st.session_state["artifacts"] = None

    if ARTIFACT_PATH.exists() and st.session_state["artifacts"] is None:
        with st.spinner("Loading saved artifacts..."):
            artifacts = load_artifacts(ARTIFACT_PATH)
            st.session_state["artifacts"] = artifacts
            st.session_state["X_tfidf"] = artifacts.tfidf.transform(artifacts.corpus)
            st.session_state["mapping"] = {
                "data_file": "from saved artifact",
                "mapped_columns": {},
                "raw_shape": (0, 0),
                "normalized_shape": artifacts.df.shape,
            }

    if st.session_state["artifacts"] is None:
        with st.spinner("Loading data and building pipeline..."):
            artifacts, X_tfidf, mapping = _build_pipeline(n_components)
            st.session_state["artifacts"] = artifacts
            st.session_state["X_tfidf"] = X_tfidf
            st.session_state["mapping"] = mapping

    return st.session_state["artifacts"], st.session_state["X_tfidf"], st.session_state["mapping"]


def save_current_artifacts():
    artifacts = st.session_state.get("artifacts")
    if artifacts is None:
        st.warning("No artifacts in memory yet. Build or load first.")
        return
    saved = save_artifacts(artifacts, ARTIFACT_PATH)
    st.success(f"Saved artifacts to: {saved}")


def show_code(title: str, fn):
    with st.expander(f"Code: {title}", expanded=False):
        st.code(inspect.getsource(fn), language="python")


def chart_year_distribution(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    sns.histplot(df["year"], bins=20, kde=True, ax=ax, color="#3B82F6")
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
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
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
        load_dataset,
        build_corpus,
        build_model,
        make_artifacts,
        search_movies,
        explain_results,
    ]

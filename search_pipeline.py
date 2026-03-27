import glob
import hashlib
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import kagglehub
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

MODEL_VERSION = "1.0.0"


@dataclass
class PipelineArtifacts:
    tfidf: TfidfVectorizer
    lsa: object
    svd: TruncatedSVD
    X_lsa: np.ndarray
    df: pd.DataFrame
    corpus: list[str]
    dataset_slug: str
    version: str = MODEL_VERSION
    created_at: str = ""
    data_fingerprint: str = ""


def _first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _parse_genres(value) -> list[str]:
    if pd.isna(value):
        return ["Unknown"]
    text = str(value).strip()
    if not text:
        return ["Unknown"]
    parts = [p.strip() for p in re.split(r"\||,|/|;", text) if p.strip()]
    return parts if parts else ["Unknown"]


def _parse_year(value):
    if pd.isna(value):
        return None
    match = re.search(r"(19\d{2}|20\d{2})", str(value))
    return int(match.group(1)) if match else None


def _parse_rating(value):
    if pd.isna(value):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download_and_normalize(
    dataset_slug: str = "chenyanglim/imdb-v2",
    data_dir: str = "data",
) -> tuple[pd.DataFrame, dict]:
    slug_safe = dataset_slug.replace("/", "_")
    local_csv = Path(data_dir) / f"{slug_safe}.csv"

    if local_csv.exists():
        raw_df = pd.read_csv(local_csv, low_memory=False)
        data_fingerprint = _file_md5(local_csv)
        source_label = f"local cache: {local_csv}"
    else:
        dataset_path = kagglehub.dataset_download(dataset_slug)
        tabular_files = sorted(
            glob.glob(os.path.join(dataset_path, "*.csv"))
            + glob.glob(os.path.join(dataset_path, "*.tsv"))
            + glob.glob(os.path.join(dataset_path, "*.parquet"))
        )
        if not tabular_files:
            raise FileNotFoundError("No CSV/TSV/Parquet file found in downloaded Kaggle dataset.")

        source_file = tabular_files[0]
        if source_file.endswith(".csv"):
            raw_df = pd.read_csv(source_file, low_memory=False)
        elif source_file.endswith(".tsv"):
            raw_df = pd.read_csv(source_file, sep="\t", low_memory=False)
        else:
            raw_df = pd.read_parquet(source_file)

        local_csv.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_csv(local_csv, index=False)
        data_fingerprint = _file_md5(local_csv)
        source_label = f"Kaggle download \u2192 cached to: {local_csv}"

    id_col = _first_existing_column(raw_df, ["imdb_title_id", "id"])
    title_col = _first_existing_column(raw_df, ["original_title", "title", "movie_title", "primaryTitle", "name"])
    genre_col = _first_existing_column(raw_df, ["genre", "genres", "Genre", "listed_in"])
    desc_col = _first_existing_column(raw_df, ["description", "plot", "overview", "summary", "synopsis", "storyline"])
    year_col = _first_existing_column(raw_df, ["year", "release_year", "startYear", "released", "release_date"])
    rating_col = _first_existing_column(raw_df, ["avg_vote", "rating", "imdb_rating", "vote_average", "averageRating", "imdb_score"])

    if title_col is None:
        raise ValueError("Could not identify a title column from Kaggle dataset.")

    norm_df = pd.DataFrame(
        {
            "id": raw_df[id_col].astype(str) if id_col else np.arange(1, len(raw_df) + 1),
            "title": raw_df[title_col].fillna("Unknown Title").astype(str),
            "genre": raw_df[genre_col].apply(_parse_genres) if genre_col else [["Unknown"]] * len(raw_df),
            "description": raw_df[desc_col].fillna("").astype(str) if desc_col else "",
            "year": raw_df[year_col].apply(_parse_year) if year_col else None,
            "rating": raw_df[rating_col].apply(_parse_rating) if rating_col else np.nan,
        }
    )

    norm_df["description"] = np.where(
        norm_df["description"].str.strip().eq(""),
        norm_df["title"],
        norm_df["description"],
    )

    norm_df = norm_df.dropna(subset=["title"]).reset_index(drop=True)

    mapping = {
        "source_label": source_label,
        "data_file": local_csv.name,
        "data_fingerprint": data_fingerprint,
        "raw_shape": raw_df.shape,
        "normalized_shape": norm_df.shape,
        "columns": list(raw_df.columns),
        "mapped_columns": {
            "id": id_col,
            "title": title_col,
            "genre": genre_col,
            "description": desc_col,
            "year": year_col,
            "rating": rating_col,
        },
    }
    return norm_df, mapping


def build_corpus(df: pd.DataFrame) -> list[str]:
    corpus = []
    for _, row in df.iterrows():
        text = f"{row['title']} {' '.join(row['genre'])} {row['description']}"
        corpus.append(text.lower())
    return corpus


def build_model(corpus: list[str], n_components: int = 100):
    tfidf = TfidfVectorizer(stop_words="english")
    X_tfidf = tfidf.fit_transform(corpus)

    max_components = min(X_tfidf.shape[0] - 1, X_tfidf.shape[1] - 1)
    if max_components < 2:
        raise ValueError("Not enough data to build SVD model.")
    n_components = min(n_components, max_components)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X_lsa = lsa.fit_transform(X_tfidf)
    return tfidf, lsa, svd, X_tfidf, X_lsa


def make_artifacts(
    df: pd.DataFrame,
    dataset_slug: str,
    n_components: int = 100,
    data_fingerprint: str = "",
) -> tuple[PipelineArtifacts, object]:
    corpus = build_corpus(df)
    tfidf, lsa, svd, X_tfidf, X_lsa = build_model(corpus, n_components=n_components)
    artifacts = PipelineArtifacts(
        tfidf=tfidf,
        lsa=lsa,
        svd=svd,
        X_lsa=X_lsa,
        df=df,
        corpus=corpus,
        dataset_slug=dataset_slug,
        version=MODEL_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        data_fingerprint=data_fingerprint,
    )
    return artifacts, X_tfidf


def save_artifacts(artifacts: PipelineArtifacts, path: str = "artifacts/search_artifacts.joblib") -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tfidf": artifacts.tfidf,
        "lsa": artifacts.lsa,
        "svd": artifacts.svd,
        "X_lsa": artifacts.X_lsa,
        "df": artifacts.df,
        "corpus": artifacts.corpus,
        "dataset_slug": artifacts.dataset_slug,
        "version": artifacts.version,
        "created_at": artifacts.created_at,
        "data_fingerprint": artifacts.data_fingerprint,
    }
    joblib.dump(payload, out_path)
    return str(out_path)


def load_artifacts(path: str = "artifacts/search_artifacts.joblib") -> PipelineArtifacts:
    payload = joblib.load(path)
    return PipelineArtifacts(
        tfidf=payload["tfidf"],
        lsa=payload["lsa"],
        svd=payload["svd"],
        X_lsa=payload["X_lsa"],
        df=payload["df"],
        corpus=payload["corpus"],
        dataset_slug=payload.get("dataset_slug", "unknown"),
        version=payload.get("version", "unknown"),
        created_at=payload.get("created_at", ""),
        data_fingerprint=payload.get("data_fingerprint", ""),
    )


def _normalize_rating(series: pd.Series) -> pd.Series:
    valid = series.dropna()
    if valid.empty:
        return pd.Series(0.0, index=series.index)
    min_val = valid.min()
    max_val = valid.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series.fillna(min_val) - min_val) / (max_val - min_val)


def search_movies(
    query: str,
    artifacts: PipelineArtifacts,
    top_k: int = 10,
    genre_filter: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
    rating_min: float | None = None,
    alpha: float = 0.85,
) -> pd.DataFrame:
    if not query.strip():
        return pd.DataFrame()

    query_vec = artifacts.tfidf.transform([query])
    query_lsa = artifacts.lsa.transform(query_vec)
    similarities = cosine_similarity(query_lsa, artifacts.X_lsa)[0]

    results = artifacts.df.copy()
    results["similarity"] = similarities

    if genre_filter and genre_filter != "All":
        results = results[results["genre"].apply(lambda gs: genre_filter in gs)]
    if year_min is not None:
        results = results[results["year"].fillna(-np.inf) >= year_min]
    if year_max is not None:
        results = results[results["year"].fillna(np.inf) <= year_max]
    if rating_min is not None:
        results = results[results["rating"].fillna(-np.inf) >= rating_min]

    if results.empty:
        return results

    rating_norm = _normalize_rating(results["rating"])
    alpha = float(np.clip(alpha, 0.0, 1.0))
    results["weighted_score"] = alpha * results["similarity"] + (1.0 - alpha) * rating_norm

    ranked = results.sort_values("weighted_score", ascending=False).head(top_k).copy()
    ranked["genre"] = ranked["genre"].apply(lambda g: ", ".join(g) if isinstance(g, list) else str(g))
    return ranked[["title", "genre", "year", "rating", "similarity", "weighted_score"]]


def explain_results(
    query: str,
    result_df: pd.DataFrame,
    artifacts: PipelineArtifacts,
    top_n: int = 5,
) -> dict:
    """Return top contributing vocabulary terms for each result row.

    Keys are the original DataFrame indices from result_df.
    Values are lists of (term, score) tuples sorted by contribution descending.

    Approach: element-wise product of query and movie LSA vectors identifies
    which latent dimensions overlap most; projecting that overlap back through
    SVD components_ reveals the vocabulary terms driving the match.
    """
    if not query.strip() or result_df.empty:
        return {}

    query_vec = artifacts.tfidf.transform([query])
    query_lsa = artifacts.lsa.transform(query_vec)[0]  # (n_components,)
    feature_names = artifacts.tfidf.get_feature_names_out()

    explanations = {}
    for idx in result_df.index:
        movie_lsa = artifacts.X_lsa[idx]  # (n_components,)
        overlap = query_lsa * movie_lsa  # dimension-wise similarity contribution
        term_scores = overlap @ artifacts.svd.components_  # project back to vocab
        top_idx = np.argsort(term_scores)[-top_n:][::-1]
        explanations[idx] = [(feature_names[i], float(term_scores[i])) for i in top_idx]

    return explanations

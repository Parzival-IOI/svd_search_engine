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
DATASET_SLUG = "jrobischon/wikipedia-movie-plots"
DATA_DIR = Path("data")
DATA_FILE = DATA_DIR / "jrobischon_wikipedia-movie-plots.csv"
ARTIFACT_PATH = Path("artifacts/search_artifacts.joblib")


@dataclass
class PipelineArtifacts:
    tfidf: TfidfVectorizer
    lsa: object
    svd: TruncatedSVD
    X_lsa: np.ndarray
    df: pd.DataFrame
    corpus: list[str]
    version: str = MODEL_VERSION
    created_at: str = ""
    data_fingerprint: str = ""


def _file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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


def ensure_dataset() -> Path:
    """Download jrobischon/wikipedia-movie-plots from Kaggle once and cache to data/.

    Returns the path to the local CSV. Subsequent calls return immediately
    without any network access.
    """
    if DATA_FILE.exists():
        return DATA_FILE
    dataset_path = kagglehub.dataset_download(DATASET_SLUG)
    csv_files = sorted(glob.glob(os.path.join(dataset_path, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in Kaggle download for {DATASET_SLUG}.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy(csv_files[0], DATA_FILE)
    return DATA_FILE


def load_dataset() -> tuple[pd.DataFrame, dict]:
    """Load and normalize the Wikipedia movie plots dataset.

    Downloads from Kaggle on first call; reads from data/ on every subsequent call.
    Returns (normalized DataFrame with columns id/title/genre/description/year, mapping dict).
    """
    csv_path = ensure_dataset()
    raw_df = pd.read_csv(csv_path, low_memory=False)
    data_fingerprint = _file_md5(csv_path)

    norm_df = pd.DataFrame(
        {
            "id": np.arange(1, len(raw_df) + 1).astype(str),
            "title": raw_df["Title"].fillna("Unknown Title").astype(str),
            "genre": raw_df["Genre"].apply(_parse_genres),
            "description": raw_df["Plot"].fillna("").astype(str),
            "year": raw_df["Release Year"].apply(_parse_year),
        }
    )
    norm_df["description"] = np.where(
        norm_df["description"].str.strip().eq(""),
        norm_df["title"],
        norm_df["description"],
    )
    norm_df = norm_df.dropna(subset=["title"]).reset_index(drop=True)

    mapping = {
        "source_label": f"local: {csv_path}",
        "data_file": csv_path.name,
        "data_fingerprint": data_fingerprint,
        "raw_shape": raw_df.shape,
        "normalized_shape": norm_df.shape,
        "columns": list(raw_df.columns),
        "mapped_columns": {
            "id": None,
            "title": "Title",
            "genre": "Genre",
            "description": "Plot",
            "year": "Release Year",
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
        version=MODEL_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        data_fingerprint=data_fingerprint,
    )
    return artifacts, X_tfidf


def save_artifacts(artifacts: PipelineArtifacts, path: Path = ARTIFACT_PATH) -> str:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tfidf": artifacts.tfidf,
        "lsa": artifacts.lsa,
        "svd": artifacts.svd,
        "X_lsa": artifacts.X_lsa,
        "df": artifacts.df,
        "corpus": artifacts.corpus,
        "version": artifacts.version,
        "created_at": artifacts.created_at,
        "data_fingerprint": artifacts.data_fingerprint,
    }
    joblib.dump(payload, out_path)
    return str(out_path)


def load_artifacts(path: Path = ARTIFACT_PATH) -> PipelineArtifacts:
    payload = joblib.load(path)
    return PipelineArtifacts(
        tfidf=payload["tfidf"],
        lsa=payload["lsa"],
        svd=payload["svd"],
        X_lsa=payload["X_lsa"],
        df=payload["df"],
        corpus=payload["corpus"],
        version=payload.get("version", "unknown"),
        created_at=payload.get("created_at", ""),
        data_fingerprint=payload.get("data_fingerprint", ""),
    )


def search_movies(
    query: str,
    artifacts: PipelineArtifacts,
    top_k: int = 10,
    genre_filter: str | None = None,
    year_min: int | None = None,
    year_max: int | None = None,
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

    if results.empty:
        return results

    ranked = results.sort_values("similarity", ascending=False).head(top_k).copy()
    ranked["genre"] = ranked["genre"].apply(lambda g: ", ".join(g) if isinstance(g, list) else str(g))
    return ranked[["title", "genre", "year", "description", "similarity"]]


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

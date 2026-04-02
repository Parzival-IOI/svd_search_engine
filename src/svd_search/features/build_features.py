import gzip
from pathlib import Path
from xml.etree import ElementTree as ET

import joblib
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from svd_search.config.paths import CORENLP_DIR, LEMMA_CACHE

CONTENT_POS = {
    "NN", "NNS", "NNP", "NNPS",
    "VB", "VBD", "VBG", "VBN", "VBP", "VBZ",
    "JJ", "JJR", "JJS",
}

DEFAULT_TFIDF_PARAMS: dict = dict(
    stop_words="english",
    sublinear_tf=True,
    min_df=3,
    max_df=0.90,
    ngram_range=(1, 2),
)


def build_corpus(df: pd.DataFrame) -> list[str]:
    """Raw text corpus: title + genre tokens + description/plot, lowercased."""
    corpus = []
    for _, row in df.iterrows():
        genre_str = " ".join(row["genre"]) if isinstance(row["genre"], list) else str(row["genre"])
        desc = row.get("description", row.get("plot", ""))
        corpus.append(f"{row['title']} {genre_str} {desc}".lower())
    return corpus


def _extract_lemmas_from_xml(path: Path) -> str:
    """Parse one CoreNLP XML.gz, return space-joined content-word lemmas."""
    with gzip.open(path, "rb") as fh:
        tree = ET.parse(fh)
    tokens = []
    for tok in tree.getroot().iter("token"):
        lemma = tok.findtext("lemma", "").lower().strip()
        pos   = tok.findtext("POS", "")
        if pos in CONTENT_POS and lemma.isalpha() and len(lemma) > 2:
            tokens.append(lemma)
    return " ".join(tokens)


def build_lemma_corpus(
    df: pd.DataFrame,
    corenlp_dir: Path = CORENLP_DIR,
    cache_path: Path = LEMMA_CACHE,
) -> list[str]:
    """Return a lemmatized corpus aligned with df.

    Loads from cache if available; otherwise parses all CoreNLP XML.gz files
    and saves the result to cache_path. Movies with no matching file fall back
    to their raw plot text.
    """
    if cache_path.exists():
        print(f"Loading lemma cache from {cache_path} ...")
        return joblib.load(cache_path)

    print("Parsing CoreNLP XML files (one-time, ~100 s) ...")
    wiki_id_to_lemmas: dict[int, str] = {}
    xml_files = list(corenlp_dir.glob("*.xml.gz"))
    total = len(xml_files)

    for i, path in enumerate(xml_files, 1):
        wiki_id = int(path.stem.split(".")[0])
        try:
            wiki_id_to_lemmas[wiki_id] = _extract_lemmas_from_xml(path)
        except Exception:
            wiki_id_to_lemmas[wiki_id] = ""
        if i % 5000 == 0 or i == total:
            print(f"  {i:>6}/{total} files processed", end="\r")

    print()
    corpus, fallback = [], 0
    for _, row in df.iterrows():
        wid = int(row["wiki_id"])
        lemma_text = wiki_id_to_lemmas.get(wid, "")
        genre_str = " ".join(row["genre"]) if isinstance(row["genre"], list) else str(row["genre"])
        if lemma_text.strip():
            corpus.append(f"{row['title'].lower()} {genre_str.lower()} {lemma_text}")
        else:
            desc = row.get("plot", "")
            corpus.append(f"{row['title']} {genre_str} {desc}".lower())
            fallback += 1

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(corpus, cache_path)
    print(f"Cached to {cache_path}  (raw-text fallback: {fallback} movies)")
    return corpus


def build_model(
    corpus: list[str],
    n_components: int = 150,
    tfidf_params: dict | None = None,
):
    """Fit TF-IDF + TruncatedSVD + L2 Normalizer on corpus.

    Returns (tfidf, lsa_pipeline, svd, X_tfidf_sparse, X_lsa_dense).
    """
    params = tfidf_params if tfidf_params is not None else DEFAULT_TFIDF_PARAMS
    tfidf   = TfidfVectorizer(**params)
    X_tfidf = tfidf.fit_transform(corpus)

    max_k        = min(X_tfidf.shape[0] - 1, X_tfidf.shape[1] - 1)
    n_components = min(n_components, max_k)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    lsa = make_pipeline(svd, Normalizer(copy=False))
    X_lsa = lsa.fit_transform(X_tfidf)

    return tfidf, lsa, svd, X_tfidf, X_lsa

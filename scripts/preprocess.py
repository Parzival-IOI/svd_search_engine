#!/usr/bin/env python
"""Parse CoreNLP XML.gz files and build the lemma corpus cache.

Run this once before training or starting the Streamlit app when
models/corenlp_lemma_cache.joblib does not yet exist.

Usage (from project root):
    python scripts/preprocess.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from svd_search.config.paths import CORENLP_DIR, LEMMA_CACHE
from svd_search.data.loader import load_cmu_dataset
from svd_search.features.build_features import build_lemma_corpus


def main() -> None:
    print("Loading CMU MovieSummaries dataset ...")
    df, _ = load_cmu_dataset()
    print(f"Loaded {len(df):,} movies")

    print("Building lemma corpus from CoreNLP XML files ...")
    corpus = build_lemma_corpus(df, corenlp_dir=CORENLP_DIR, cache_path=LEMMA_CACHE)

    print(f"Done. Corpus size : {len(corpus):,} documents")
    print(f"Cache saved to   : {LEMMA_CACHE}")


if __name__ == "__main__":
    main()

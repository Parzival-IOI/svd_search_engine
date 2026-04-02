#!/usr/bin/env python
"""Train the CoreNLP-based LSA search model and save to models/.

Usage (from project root):
    python scripts/train_model.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from svd_search.config.paths import ARTIFACT_V2
from svd_search.data.loader import load_cmu_dataset
from svd_search.models.train import make_artifacts, save_artifacts


def main() -> None:
    print("Loading CMU MovieSummaries dataset ...")
    df, mapping = load_cmu_dataset()
    print(f"Loaded {len(df):,} movies")

    print("Building pipeline (CoreNLP lemmatized corpus, n_components=150) ...")
    artifacts, _ = make_artifacts(
        df,
        n_components=150,
        data_fingerprint=mapping.get("data_fingerprint", ""),
        use_lemmas=True,
    )

    path = save_artifacts(artifacts, ARTIFACT_V2)
    print(f"Saved to {path}")
    print(f"  Vocabulary : {len(artifacts.tfidf.vocabulary_):,} terms")
    print(f"  Movies     : {len(artifacts.df):,}")
    print(f"  LSA shape  : {artifacts.X_lsa.shape}")


if __name__ == "__main__":
    main()

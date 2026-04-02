import hashlib
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_VERSION = "2.0.0"


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


def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_genres(value) -> list[str]:
    if pd.isna(value):
        return ["Unknown"]
    text = str(value).strip()
    if not text:
        return ["Unknown"]
    parts = [p.strip() for p in re.split(r"\||,|/|;", text) if p.strip()]
    return parts if parts else ["Unknown"]


def parse_year(value):
    if pd.isna(value):
        return None
    match = re.search(r"(19\d{2}|20\d{2})", str(value))
    return int(match.group(1)) if match else None

import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from svd_search import search_movies, explain_results
from streamlit_ui import (
    chart_genre,
    chart_lsa_scatter,
    chart_top_terms,
    chart_year_distribution,
    get_artifacts,
    process_functions_for_display,
    show_code,
)

sns.set_theme(style="whitegrid")

st.title("Process Notebook View")
st.caption(
    "Mirrors the main_v2.ipynb pipeline: "
    "CMU MovieSummaries + CoreNLP lemmas → TF-IDF → SVD/LSA → search"
)

# Load pre-trained CoreNLP artifact — no slider, no rebuild, no save
artifacts, X_tfidf, _ = get_artifacts(n_components=150, rebuild=False)

# X_tfidf may be None on partial session-state reloads — recompute from artifact
if artifacts is not None and X_tfidf is None:
    X_tfidf = artifacts.tfidf.transform(artifacts.corpus)

if artifacts is None:
    st.info("Loading model…")
    st.stop()

# Sidebar: read-only info panel
st.sidebar.header("Model Info")
st.sidebar.caption("Corpus: CMU MovieSummaries + CoreNLP lemmas")
_fp = (artifacts.data_fingerprint[:16] + "…") if artifacts.data_fingerprint else "N/A"
st.sidebar.markdown(
    f"- **Movies**: {len(artifacts.df):,}\n"
    f"- **SVD components**: {artifacts.svd.n_components}\n"
    f"- **Vocabulary**: {len(artifacts.tfidf.get_feature_names_out()):,} terms\n"
    f"- **Model version**: {artifacts.version}\n"
    f"- **Built at**: {artifacts.created_at or 'N/A'}\n"
    f"- **Data MD5**: `{_fp}`"
)

# ── 1) Dataset ──────────────────────────────────────────────────────────────
st.subheader("1) Dataset — CMU MovieSummaries")
st.markdown(
    "Two CMU files are **inner-joined on `wiki_id`**:\n\n"
    "| File | Rows | Key columns |\n"
    "|------|------|-------------|\n"
    "| `movie.metadata.tsv` | 81 741 | title, release_date, genres (Freebase JSON) |\n"
    "| `plot_summaries.txt` | 42 303 | plot text |\n\n"
    "Genres are stored as Freebase JSON dicts and parsed into plain lists."
)

c1, c2, c3 = st.columns(3)
c1.metric("Movies", f"{len(artifacts.df):,}")
c2.metric("SVD Components", f"{artifacts.svd.n_components}")
c3.metric("Vocabulary Size", f"{len(artifacts.tfidf.get_feature_names_out()):,}")
st.caption(
    f"Built at: **{artifacts.created_at or 'N/A'}** | "
    f"Model version: **{artifacts.version}** | "
    f"Data MD5: `{_fp}`"
)
_display_cols = ["title", "genre", "year", "plot"] if "plot" in artifacts.df.columns else ["title", "genre", "year", "description"]
st.dataframe(artifacts.df[_display_cols].head(10), width="stretch")

# ── 2) EDA ──────────────────────────────────────────────────────────────────
st.subheader("2) Exploratory Data Analysis")
st.markdown(
    "Before building any model, we visualise release-year distribution, "
    "genre frequencies, and plot word-count distribution — "
    "the last one is critical for LSA quality."
)

col1, col2 = st.columns(2)
with col1:
    st.pyplot(chart_year_distribution(artifacts.df))
with col2:
    st.pyplot(chart_genre(artifacts.df))

# Plot word-count distribution (mirrors main_v2 cell 6)
plot_col = "plot" if "plot" in artifacts.df.columns else "description"
word_counts = artifacts.df[plot_col].str.split().str.len()

fig_wc, ax = plt.subplots(figsize=(10, 4))
ax.hist(word_counts.clip(upper=1500), bins=60, color="#10B981", edgecolor="white", linewidth=0.3)
ax.axvline(
    word_counts.median(), color="red", ls="--", lw=1.5,
    label=f"Median = {int(word_counts.median())} words",
)
ax.axvline(50, color="orange", ls=":", lw=1.5, label="50-word threshold")
ax.set_title("Plot Word-Count Distribution (clipped at 1 500)")
ax.set_xlabel("Word Count")
ax.set_ylabel("Number of Movies")
ax.legend()
fig_wc.tight_layout()
st.pyplot(fig_wc)

cc1, cc2, cc3 = st.columns(3)
cc1.metric("Median plot length", f"{int(word_counts.median())} words")
cc2.metric(
    "Movies < 50 words",
    f"{(word_counts < 50).sum():,} ({(word_counts < 50).mean() * 100:.1f}%)",
)
cc3.metric(
    "Movies 300+ words",
    f"{(word_counts >= 300).sum():,} ({(word_counts >= 300).mean() * 100:.1f}%)",
)
st.markdown(
    "**Analysis**: Movies with fewer than 50 plot words produce near-empty TF-IDF vectors "
    "that yield noisy cosine similarity scores — ~8% of the CMU corpus falls below this threshold."
)

# ── 3) Corpus Comparison — Raw vs Lemmatized ────────────────────────────────
st.subheader("3) Corpus Comparison — Raw vs Lemmatized")

# Build the raw pipeline once per session (cached via session_state)
if "raw_pipeline" not in st.session_state:
    with st.spinner("Building raw corpus (one-time, cached for this session)…"):
        df = artifacts.df
        raw_corpus: list[str] = []
        for _, row in df.iterrows():
            genre_str = (
                " ".join(row["genre"]) if isinstance(row["genre"], list) else str(row["genre"])
            )
            raw_corpus.append(
                f"{row['title']} {genre_str} {row[plot_col]}".lower()
            )
        _tfidf_raw = TfidfVectorizer(
            stop_words="english", sublinear_tf=True, min_df=3, max_df=0.90, ngram_range=(1, 2)
        )
        _X_raw = _tfidf_raw.fit_transform(raw_corpus)
        _svd_raw = TruncatedSVD(n_components=150, random_state=42)
        _lsa_raw = make_pipeline(_svd_raw, Normalizer(copy=False))
        _X_lsa_raw = _lsa_raw.fit_transform(_X_raw)
        st.session_state["raw_pipeline"] = (
            raw_corpus, _tfidf_raw, _X_raw, _svd_raw, _lsa_raw, _X_lsa_raw
        )

raw_corpus, tfidf_raw, X_raw, svd_raw, lsa_raw, X_lsa_raw = st.session_state["raw_pipeline"]
lemma_corpus = artifacts.corpus

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Raw Corpus** — plain text (title + genre + plot)")
    st.metric("Documents", f"{len(raw_corpus):,}")
    with st.expander("Sample document"):
        st.code(raw_corpus[0][:600], language="text")
with col2:
    st.markdown("**Lemmatized Corpus** — CoreNLP POS-filtered lemmas")
    st.metric("Documents", f"{len(lemma_corpus):,}")
    with st.expander("Sample document"):
        st.code(lemma_corpus[0][:600], language="text")

st.markdown(
    "**CoreNLP POS filter** — only nouns (NN\\*), verbs (VB\\*), and adjectives (JJ\\*) are kept. "
    "Inflected forms collapse to their base lemma: *runs / ran / running → run*. "
    "Results are cached to `models/corenlp_lemma_cache.joblib` so parsing runs only once."
)

# ── 4) TF-IDF Vectorization ──────────────────────────────────────────────────
st.subheader("4) TF-IDF Vectorization")
st.latex(r"\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\frac{N}{\text{DF}(t)}")
st.markdown(
    "| Parameter | Value | Reason |\n"
    "|-----------|-------|--------|\n"
    "| `stop_words` | `\"english\"` | Remove function words |\n"
    "| `sublinear_tf` | `True` | Use $1+\\log(\\text{tf})$ to limit term-frequency dominance |\n"
    "| `min_df` | `3` | Discard terms in fewer than 3 documents |\n"
    "| `max_df` | `0.90` | Discard terms in 90%+ of documents |\n"
    "| `ngram_range` | `(1, 2)` | Include bigrams: *\"haunted house\"*, *\"outer space\"* |"
)

n_r, v_r = X_raw.shape
sp_r = round((1 - X_raw.nnz / (n_r * v_r)) * 100, 2)
X_lemma = X_tfidf
if X_lemma is None:
    X_lemma = artifacts.tfidf.transform(artifacts.corpus)
n_l, v_l = X_lemma.shape
sp_l = round((1 - X_lemma.nnz / (n_l * v_l)) * 100, 2)

st.table(
    pd.DataFrame(
        {
            "Metric": ["Vocabulary size", "Sparsity %"],
            "Raw corpus": [f"{v_r:,}", f"{sp_r}%"],
            "Lemmatized (CoreNLP)": [f"{v_l:,}", f"{sp_l}%"],
        }
    )
)

col1, col2 = st.columns(2)
with col1:
    st.markdown("**Top 20 terms — Raw**")
    st.pyplot(chart_top_terms(tfidf_raw, X_raw))
with col2:
    st.markdown("**Top 20 terms — Lemmatized**")
    st.pyplot(chart_top_terms(artifacts.tfidf, X_lemma))

st.markdown(
    "**Analysis**: Lemmatized vocabulary is smaller — inflected variants merge into "
    "a single feature, boosting that feature's IDF weight and co-occurrence signal."
)

# ── 5) SVD / LSA Projection ──────────────────────────────────────────────────
st.subheader("5) SVD / LSA Projection (k = 150)")
st.latex(r"A \approx U \Sigma V^T")
st.markdown(
    "The TF-IDF matrix $A$ is factorized into **150** latent dimensions. "
    "After SVD, vectors are **L2-normalised** so cosine similarity reduces to a dot product. "
    "150 components are used here (vs 100 in `main.ipynb`) because the CMU corpus is richer."
)

cum_raw_var = np.cumsum(svd_raw.explained_variance_ratio_)
cum_lemma_var = np.cumsum(artifacts.svd.explained_variance_ratio_)

fig_var, ax = plt.subplots(figsize=(10, 4))
k_range = range(1, len(cum_raw_var) + 1)
ax.plot(k_range, cum_raw_var, label="Raw", color="#6366F1", lw=2)
ax.plot(k_range, cum_lemma_var, label="Lemmatized", color="#10B981", lw=2)
ax.axhline(0.5, ls="--", color="gray", lw=1, label="50% threshold")
ax.set_title("Cumulative Explained Variance: Raw vs Lemmatized")
ax.set_xlabel("SVD Components (k)")
ax.set_ylabel("Cumulative Explained Variance")
ax.set_ylim(0, 1.05)
ax.legend()
fig_var.tight_layout()
st.pyplot(fig_var)

cv1, cv2 = st.columns(2)
cv1.metric("Variance captured — Raw", f"{cum_raw_var[-1]:.4f}")
cv2.metric("Variance captured — Lemmatized", f"{cum_lemma_var[-1]:.4f}")

col1, col2 = st.columns(2)
with col1:
    st.markdown("**2D Latent Space — Raw**")
    st.pyplot(chart_lsa_scatter(artifacts.df, X_lsa_raw))
with col2:
    st.markdown("**2D Latent Space — Lemmatized**")
    st.pyplot(chart_lsa_scatter(artifacts.df, artifacts.X_lsa))

st.markdown(
    "**Analysis**: Genre clusters emerge from purely text-based latent projection — "
    "no explicit genre labels were used during SVD training. "
    "The lemmatized scatter typically shows tighter, more separated clusters."
)

# ── 6) Side-by-Side Search Comparison ───────────────────────────────────────
st.subheader("6) Side-by-Side Search: Raw vs Lemmatized")
st.markdown(
    "The same queries run through both pipelines. "
    "Lemmatized results consistently show **higher top similarity** and **better-separated scores**."
)

COMPARISON_QUERIES = [
    "haunted house ghost supernatural",
    "space mission astronaut planet",
    "detective mystery crime investigation",
    "war battlefield soldier survival",
]

for q in COMPARISON_QUERIES:
    st.markdown(f"**Query:** `{q}`")

    # Raw pipeline search (inline — different TF-IDF/SVD from the artifact)
    q_vec_r = tfidf_raw.transform([q])
    q_lsa_r = lsa_raw.transform(q_vec_r)
    scores_r = cosine_similarity(q_lsa_r, X_lsa_raw)[0]
    top_idx_r = np.argsort(scores_r)[::-1][:5]
    raw_results = artifacts.df.iloc[top_idx_r][["title", "genre", "year"]].copy()
    raw_results["similarity"] = scores_r[top_idx_r].round(4)
    raw_results["genre"] = raw_results["genre"].apply(
        lambda g: ", ".join(g[:2]) if isinstance(g, list) else str(g)
    )

    # Lemmatized pipeline search (uses pretrained artifact)
    lemma_results = search_movies(q, artifacts, top_k=5)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("*Raw corpus*")
        st.dataframe(
            raw_results[["title", "similarity"]].reset_index(drop=True),
            width="stretch",
        )
    with col2:
        st.markdown("*Lemmatized (CoreNLP)*")
        st.dataframe(
            lemma_results[["title", "similarity"]].reset_index(drop=True),
            width="stretch",
        )

    top_r = scores_r[top_idx_r[0]]
    top_l = lemma_results["similarity"].max() if not lemma_results.empty else 0.0
    st.caption(f"Top similarity — raw: {top_r:.4f} | lemmatized: {top_l:.4f}")
    st.divider()

# ── 7) Term-Level Explanation ────────────────────────────────────────────────
st.subheader("7) Term-Level Explanation")
st.markdown(
    "For each retrieved result, the query–document overlap in LSA space is projected "
    "back to vocabulary space to identify which terms drove the match.\n\n"
    "**Method**: $\\vec{o} = \\vec{q}_{\\text{lsa}} \\odot \\vec{m}_i$, "
    "then $\\vec{t} = \\vec{o} \\cdot V^T$ where $V$ is the SVD components matrix."
)

explain_query = st.text_input(
    "Query for explanation", value="haunted house ghost supernatural"
)
if explain_query.strip():
    top5 = search_movies(explain_query, artifacts, top_k=5)
    if not top5.empty:
        explanations = explain_results(explain_query, top5, artifacts, top_n=6)
        for idx, row in top5.iterrows():
            with st.expander(f"{row['title']} (sim={row['similarity']:.4f})"):
                st.markdown(f"**Plot:** {row['description']}")
                terms = explanations.get(idx, [])
                if terms:
                    st.markdown("**Top matching terms:**")
                    st.dataframe(
                        pd.DataFrame(terms, columns=["Term", "Score"]),
                        width="stretch",
                    )
    else:
        st.warning("No results found.")

# ── 8) Pipeline Source Code ──────────────────────────────────────────────────
st.subheader("8) Pipeline Source Code")
for fn in process_functions_for_display():
    show_code(fn.__name__, fn)

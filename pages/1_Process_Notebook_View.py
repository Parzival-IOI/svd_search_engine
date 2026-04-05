import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

# ── 5b) Conceptual SVD Toy Example ──────────────────────────────────────────
st.subheader("5b) Conceptual SVD Decomposition — Toy Example")
st.markdown(
    "Before applying SVD to 42 k movies, we build intuition with a hand-crafted **6×5 matrix** — "
    "3 horror movies and 3 romance movies over 5 terms. "
    "SVD discovers the genre structure automatically from word co-occurrence alone."
)

with st.expander("Show toy SVD visualisation", expanded=True):
    _terms = ["ghost", "haunt", "house", "love", "heart"]
    _docs  = ["Horror 1", "Horror 2", "Horror 3", "Romance 1", "Romance 2", "Romance 3"]
    _A_toy = np.array([
        [0.9, 0.8, 0.7, 0.0, 0.0],
        [0.8, 0.7, 0.0, 0.1, 0.0],
        [0.7, 0.9, 0.6, 0.0, 0.1],
        [0.0, 0.0, 0.1, 0.9, 0.8],
        [0.0, 0.1, 0.0, 0.8, 0.9],
        [0.1, 0.0, 0.0, 0.7, 0.8],
    ])
    _U_toy, _sigma_toy, _Vt_toy = np.linalg.svd(_A_toy, full_matrices=False)
    _fig_toy, _axes_toy = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)

    sns.heatmap(_A_toy, annot=True, fmt=".1f",
                xticklabels=_terms, yticklabels=_docs,
                cmap="Blues", ax=_axes_toy[0])
    _axes_toy[0].set_title("Matrix A — Term-Document Matrix\n(6 movies × 5 terms)")

    sns.heatmap(_Vt_toy[:3, :], annot=True, fmt=".2f",
                xticklabels=_terms,
                yticklabels=[f"Topic {i}" for i in range(3)],
                cmap="RdBu_r", center=0, ax=_axes_toy[1])
    _axes_toy[1].set_title("V^T — Topic-Term Weights\n(top 3 components)\nPositive = term belongs to topic")

    _ax_u = _axes_toy[2]
    _col_toy = ["#EF4444"] * 3 + ["#3B82F6"] * 3
    for _i, (_d, _c) in enumerate(zip(_docs, _col_toy)):
        _ax_u.scatter(_U_toy[_i, 0], _U_toy[_i, 1], color=_c, s=150, zorder=3)
        _ax_u.annotate(_d, (_U_toy[_i, 0], _U_toy[_i, 1]),
                       textcoords="offset points", xytext=(6, 4), fontsize=9)
    _ax_u.axhline(0, color="gray", lw=0.7)
    _ax_u.axvline(0, color="gray", lw=0.7)
    _ax_u.set_xlabel("LSA Dim 1")
    _ax_u.set_ylabel("LSA Dim 2")
    _ax_u.set_title("U — Documents in 2D Latent Space\n(Horror=red, Romance=blue)")
    _ax_u.legend(handles=[
        mpatches.Patch(color="#EF4444", label="Horror"),
        mpatches.Patch(color="#3B82F6", label="Romance"),
    ])
    _fig_toy.suptitle("Conceptual SVD on a 6×5 Term-Document Matrix", fontsize=13)
    st.pyplot(_fig_toy)
    plt.close(_fig_toy)

    st.markdown(
        f"**Singular values**: `{[round(float(s), 3) for s in _sigma_toy]}`  \n"
        "Horror movies cluster on one side of Dim 1; romance movies on the other. "
        "SVD discovers this genre structure purely from word co-occurrence."
    )

# ── 5c) Real LSA Model Internals — U, Σ, V^T ─────────────────────────────────
st.subheader("5c) Real LSA Model Internals — U, Σ, V^T")
st.markdown(
    "The same decomposition applied to the full 42 k-movie corpus. "
    "The **singular value spectrum** (Σ) shows how variance is distributed across components; "
    "the **topic-term matrix** (V^T) reveals which vocabulary terms define each latent topic."
)

_feature_names_int = artifacts.tfidf.get_feature_names_out()
_svd_model         = artifacts.svd
_svs_real          = _svd_model.singular_values_
_evr_real          = _svd_model.explained_variance_ratio_
_N_COMP_INT        = _svd_model.n_components

_fig_int, _axes_int = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)

# Σ: Singular Value Spectrum
_ax_sv = _axes_int[0]
_ax_sv.bar(range(1, _N_COMP_INT + 1), _svs_real, color="#6366F1", alpha=0.7)
_ax_sv.set_title(f"Σ — Singular Value Spectrum\n({_N_COMP_INT} components)", fontsize=11)
_ax_sv.set_xlabel("Component index (k)")
_ax_sv.set_ylabel("Singular value σₖ")
_ax_sv.set_xlim(0, _N_COMP_INT + 1)
_ax_sv2 = _ax_sv.twinx()
_ax_sv2.plot(range(1, _N_COMP_INT + 1), np.cumsum(_evr_real), color="#F59E0B", lw=2)
_ax_sv2.set_ylabel("Cumulative Explained Variance", color="#F59E0B")
_ax_sv2.tick_params(axis="y", labelcolor="#F59E0B")
_ax_sv2.set_ylim(0, 1.05)

# V^T: Topic-Term Heatmap
_ax_vt = _axes_int[1]
_N_SHOW_T = 10
_all_top_t = set()
for _ci in range(_N_SHOW_T):
    _all_top_t.update(np.argsort(np.abs(_svd_model.components_[_ci]))[-8:].tolist())
_all_top_t = sorted(_all_top_t)[:20]
_vt_sub = _svd_model.components_[:_N_SHOW_T, :][:, _all_top_t]
sns.heatmap(
    _vt_sub,
    xticklabels=[_feature_names_int[i] for i in _all_top_t],
    yticklabels=[f"Topic {i}" for i in range(_N_SHOW_T)],
    cmap="RdBu_r", center=0, ax=_ax_vt, cbar=True, linewidths=0.3,
)
_ax_vt.set_title(f"V^T — Topic-Term Weights\n(first {_N_SHOW_T} topics × top terms)", fontsize=11)
_ax_vt.tick_params(axis="x", rotation=45, labelsize=8)
_ax_vt.tick_params(axis="y", labelsize=9)

# U: Documents in 2D
_ax_u2 = _axes_int[2]
_sample_u2 = np.random.default_rng(99).choice(len(artifacts.df), size=600, replace=False)
_genres_u2 = artifacts.df["genre"].apply(
    lambda g: g[0] if isinstance(g, list) and g else "Unknown"
).iloc[_sample_u2]
sns.scatterplot(
    data=pd.DataFrame({
        "x": artifacts.X_lsa[_sample_u2, 0],
        "y": artifacts.X_lsa[_sample_u2, 1],
        "genre": _genres_u2.values,
    }),
    x="x", y="y", hue="genre", alpha=0.55, s=18, ax=_ax_u2, legend=True,
)
_ax_u2.set_title("U — Document Positions\n(Dim 1 vs Dim 2, 600 sample)", fontsize=11)
_ax_u2.set_xlabel("Latent Dimension 1")
_ax_u2.set_ylabel("Latent Dimension 2")
_ax_u2.legend(loc="best", fontsize=6, markerscale=0.8, framealpha=0.7)

_fig_int.suptitle(
    "Real LSA Model — U, Σ, V^T Visualization (CoreNLP Lemmatized Corpus)", fontsize=13
)
st.pyplot(_fig_int)
plt.close(_fig_int)

st.markdown(
    "**Σ chart**: Values drop steeply then plateau — first ~20 topics dominate, rest capture noise.  \n"
    "**V^T heatmap**: Red = term strongly defines this topic; blue = negative loading.  \n"
    "**U scatter**: Genre clusters emerge in 2D without supervision — romance, horror, action separate naturally."
)

# ── 6) Cosine Similarity ─────────────────────────────────────────────────────
st.subheader("6) Search Method — Cosine Similarity")
st.latex(
    r"\text{sim}(\vec{q}, \vec{m}) = \cos(\theta) = "
    r"\frac{\vec{q} \cdot \vec{m}}{\|\vec{q}\| \cdot \|\vec{m}\|}"
)
st.markdown(
    "After LSA projection, every movie and every query is a **dense vector** in $k$-dimensional "
    "latent space. We measure similarity using the **cosine of the angle** — invariant to document length.\n\n"
    "| Value | Meaning |\n"
    "|-------|---------|\n"
    "| $\\cos(\\theta) = 1$ | Identical direction — perfect semantic match |\n"
    "| $\\cos(\\theta) = 0$ | Orthogonal — no shared latent topics |\n"
    "| $\\cos(\\theta) < 0$ | Opposite directions — thematically opposite |\n\n"
    "**Why not Euclidean?** A short plot and a long plot on the same topic point in the same "
    "*direction* but have different *magnitudes*. Cosine ignores magnitude — only the angle matters.\n\n"
    "**Why is it fast?** After L2-normalisation all vectors satisfy $\\|\\vec{m}\\| = 1$, so "
    "$\\cos(\\theta) = \\vec{q} \\cdot \\vec{m}$. "
    "All 42,303 scores reduce to **one matrix multiply** — milliseconds for 42 k movies.\n\n"
    "**Complete search pipeline:**"
)
st.code(
    "1. q_tfidf = tfidf.transform([query])      # (1 × V) sparse\n"
    "2. q_lsa   = lsa.transform(q_tfidf)        # (1 × k) dense, L2-normalised\n"
    "3. scores  = X_lsa @ q_lsa.T               # (42303,) — one matrix multiply\n"
    "4. top_idx = argsort(scores)[::-1][:top_k] # top-k indices\n"
    "5. return df.iloc[top_idx]                 # movie metadata",
    language="python",
)

# ── 7) Side-by-Side Search Comparison ───────────────────────────────────────
st.subheader("7) Side-by-Side Search: Raw vs Lemmatized")
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

# ── 8) Term-Level Explanation ────────────────────────────────────────────────
st.subheader("8) Term-Level Explanation")
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

# ── 9) Pipeline Source Code ──────────────────────────────────────────────────
st.subheader("9) Pipeline Source Code")
for fn in process_functions_for_display():
    show_code(fn.__name__, fn)

# ── 10) Summary & Findings ────────────────────────────────────────────────────
st.subheader("10) Summary & Findings")
st.markdown(
    "### Pipeline Recap\n"
    "```\n"
    "CMU MovieSummaries (42 k movies)\n"
    "├── movie.metadata.tsv  ──┐\n"
    "│   [title, genres, year]  │  inner join on wiki_id\n"
    "└── plot_summaries.txt  ──┘\n"
    "         │\n"
    "         ▼\n"
    "CoreNLP XML.gz annotations (per movie)\n"
    "  → extract POS-filtered lemmas: {NN*, VB*, JJ*} only\n"
    "  → cache to models/corenlp_lemma_cache.joblib\n"
    "         │\n"
    "         ▼\n"
    "Document corpus: title + genre + lemma tokens\n"
    "         │\n"
    "TF-IDF (sublinear_tf, min_df=3, max_df=0.90, ngram_range=(1,2))\n"
    "  → sparse matrix A: 42,303 × V terms\n"
    "         │\n"
    "Truncated SVD (k=150) + L2 Normalizer\n"
    "  → A ≈ U_k Σ_k V_k^T\n"
    "  → X_lsa: 42,303 × 150 dense, L2-normalized\n"
    "         │\n"
    "Search: cosine_similarity(q_lsa, X_lsa) → top-k movies\n"
    "```"
)

st.markdown("### Key Insights")
st.table(pd.DataFrame({
    "Question": [
        "Does more text help?",
        "Does lemmatization help?",
        "How many components?",
        "Why not more components?",
        "Why cosine, not Euclidean?",
    ],
    "Answer": [
        "Yes — movies with <50 plot words produce near-empty TF-IDF vectors → noisy similarity scores",
        "Yes — merging inflected forms increases co-occurrence counts per feature, giving SVD a stronger signal",
        "150 captures most variance; diminishing returns past ~100",
        "Larger k = more noise captured + slower search",
        "Document length invariance — a short and long plot on the same topic score similarly",
    ],
}))

st.markdown("### Limitations")
st.table(pd.DataFrame({
    "Limitation": [
        "Bag-of-words — no word order",
        "No negation handling",
        "Static model — must retrain for new movies",
        "No user feedback",
        "Vocabulary boundary",
    ],
    "Impact": [
        '"dog bites man" = "man bites dog" in TF-IDF',
        '"not a love story" matches love story movies',
        "Cannot update incrementally",
        "Cannot improve relevance from click signals",
        "Out-of-vocabulary query words are silently ignored",
    ],
}))

st.markdown("### Final Configuration")
st.code(
    '# Corpus\n'
    'CoreNLP POS filter: {NN, NNS, NNP, NNPS, VB, VBD, VBG, VBN, VBP, VBZ, JJ, JJR, JJS}\n'
    'Minimum lemma length: 3 characters, alphabetic only\n\n'
    '# TF-IDF\n'
    'TfidfVectorizer(stop_words="english", sublinear_tf=True,\n'
    '                min_df=3, max_df=0.90, ngram_range=(1, 2))\n\n'
    '# LSA\n'
    'TruncatedSVD(n_components=150, random_state=42)\n'
    'Normalizer()   # L2 → cosine similarity = dot product',
    language="python",
)

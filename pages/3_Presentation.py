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
from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

# streamlit_ui bootstraps sys.path and provides get_artifacts
from streamlit_ui import get_artifacts
from svd_search import search_movies, explain_results

sns.set_theme(style="whitegrid")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="LSA Presentation", layout="wide")

SECTIONS = [
    "Introduction",
    "Dataset",
    "Latent Semantic Analysis (LSA)",
    "Text Representation: TF-IDF",
    "Core Method: SVD",
    "Search: Cosine Similarity",
    "Results & Analysis",
    "Limitations",
]

# ── Sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
section = st.sidebar.radio("Jump to section", SECTIONS)
st.sidebar.divider()
st.sidebar.caption("Model: CMU MovieSummaries + CoreNLP lemmas, k=150 SVD components")

# Load artifacts once
artifacts, X_tfidf, _ = get_artifacts(n_components=150, rebuild=False)

# X_tfidf may be None on partial session-state reloads — recompute from artifact
if artifacts is not None and X_tfidf is None:
    X_tfidf = artifacts.tfidf.transform(artifacts.corpus)

# ══════════════════════════════════════════════════════════════════════════════
# 1. INTRODUCTION
# ══════════════════════════════════════════════════════════════════════════════
if section == "Introduction":
    st.title("SVD-Based Movie Search Engine")
    st.subheader("Using Latent Semantic Analysis for Semantic Retrieval")
    st.divider()

    st.markdown(
        "Traditional keyword search fails when a query uses *different words* than a document — "
        "even if it means the same thing. **Latent Semantic Analysis (LSA)** solves this by "
        "discovering hidden *topics* in a corpus and matching by topic, not exact term."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "### The Problem\n"
            "- Query: `\"specter haunting mansion\"`\n"
            "- Document: *\"A ghost keeps returning to the old house\"*\n"
            "- Keyword match: **0 matches** (different words)\n"
            "- LSA match: **high similarity** (same latent horror topic)"
        )
    with col2:
        st.markdown(
            "### Our Solution\n"
            "| Step | Method |\n"
            "|------|--------|\n"
            "| Corpus | CoreNLP POS-filtered lemmas |\n"
            "| Vectorize | TF-IDF |\n"
            "| Reduce | Truncated SVD ($k=150$) |\n"
            "| Retrieve | Cosine similarity in latent space |"
        )

    st.divider()
    if artifacts:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Movies", f"{len(artifacts.df):,}")
        c2.metric("SVD Components", str(artifacts.svd.n_components))
        c3.metric("Vocabulary", f"{len(artifacts.tfidf.get_feature_names_out()):,} terms")
        c4.metric("Model Version", artifacts.version or "2.0.0")

# ══════════════════════════════════════════════════════════════════════════════
# 2. DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Dataset":
    st.title("Dataset — CMU MovieSummaries")
    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown(
            "### What is it?\n"
            "The **CMU Movie Summary Corpus** contains:\n\n"
            "| File | Rows | Content |\n"
            "|------|------|---------||\n"
            "| `movie.metadata.tsv` | 81,741 | Title, year, Freebase genres |\n"
            "| `plot_summaries.txt` | 42,303 | Wikipedia plot summaries |\n\n"
            "We **inner-join** on `wiki_id` → **42,303 movies** with both metadata and plot text.\n\n"
            "Genres are stored as Freebase JSON dicts, e.g.:\n"
            "```json\n{\"/m/01jfsb\": \"Thriller\", \"/m/07s9rl0\": \"Drama\"}\n```\n"
            "Parsed into plain lists: `[\"Thriller\", \"Drama\"]`"
        )
    with col2:
        st.markdown("### Why this dataset?")
        st.markdown(
            "| Feature | Wikipedia CSV (v1) | CMU MovieSummaries (v2) |\n"
            "|---------|-------------------|------------------------|\n"
            "| Movies | 34,886 | **42,303** |\n"
            "| Genres | Single string | Multi-label list (57+ genres) |\n"
            "| Text | Raw Wikipedia | Raw + **CoreNLP annotations** |\n"
            "| Lemmas | ❌ | ✅ `.xml.gz` per movie |"
        )

    if artifacts is not None:
        st.divider()
        st.subheader("Dataset Preview")

        plot_col = "plot" if "plot" in artifacts.df.columns else "description"
        display_cols = ["title", "genre", "year", plot_col]
        st.dataframe(artifacts.df[display_cols].head(15), width="stretch")

        col1, col2 = st.columns(2)
        with col1:
            # Year distribution
            fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
            sns.histplot(artifacts.df["year"].dropna(), bins=30, kde=True, ax=ax, color="#3B82F6")
            ax.set_title("Release Year Distribution")
            ax.set_xlabel("Year")
            st.pyplot(fig)
        with col2:
            # Genre frequency
            genre_counts = artifacts.df["genre"].explode().value_counts().head(15)
            fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
            sns.barplot(x=genre_counts.values, y=genre_counts.index,
                        hue=genre_counts.index, palette="viridis", legend=False, ax=ax)
            ax.set_title("Top 15 Genres")
            ax.set_xlabel("Count")
            st.pyplot(fig)

        # Word count distribution
        word_counts = artifacts.df[plot_col].str.split().str.len()
        fig, ax = plt.subplots(figsize=(12, 3), constrained_layout=True)
        ax.hist(word_counts.clip(upper=1500), bins=60, color="#10B981",
                edgecolor="white", linewidth=0.3)
        ax.axvline(word_counts.median(), color="red", ls="--", lw=1.5,
                   label=f"Median = {int(word_counts.median())} words")
        ax.axvline(50, color="orange", ls=":", lw=1.5, label="50-word threshold")
        ax.set_title("Plot Word-Count Distribution")
        ax.set_xlabel("Word Count (clipped at 1,500)")
        ax.set_ylabel("Movies")
        ax.legend()
        st.pyplot(fig)

        c1, c2, c3 = st.columns(3)
        c1.metric("Median plot length", f"{int(word_counts.median())} words")
        c2.metric("Movies < 50 words",
                  f"{(word_counts < 50).sum():,} ({(word_counts < 50).mean()*100:.1f}%)")
        c3.metric("Movies 300+ words",
                  f"{(word_counts >= 300).sum():,} ({(word_counts >= 300).mean()*100:.1f}%)")

# ══════════════════════════════════════════════════════════════════════════════
# 3. LSA
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Latent Semantic Analysis (LSA)":
    st.title("Latent Semantic Analysis (LSA)")
    st.divider()

    st.markdown(
        "LSA is a technique that discovers **hidden (latent) topics** in a collection of documents "
        "by analyzing patterns of word co-occurrence."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "### The Core Idea\n"
            "Words that appear in similar documents tend to have similar meanings — "
            "even if they are spelled differently.\n\n"
            "| Words | Why they're related |\n"
            "|-------|--------------------|\n"
            "| ghost, specter, apparition | All appear in horror plots |\n"
            "| love, romance, relationship | All appear in drama plots |\n"
            "| detective, mystery, crime | All appear in thriller plots |\n\n"
            "LSA learns these relationships **automatically** from co-occurrence patterns, "
            "with no hand-crafted rules or dictionaries."
        )
    with col2:
        st.markdown(
            "### Key Capability: Vocabulary Mismatch\n\n"
            "```\n"
            "Query:    \"specter haunting mansion\"\n"
            "Document: \"ghost returns to old house\"\n"
            "```\n\n"
            "**Keyword search**: 0 matching words → no result\n\n"
            "**LSA**: both query and document project onto the same *horror* latent "
            "dimension → high cosine similarity → correct match ✅\n\n"
            "This is the fundamental advantage of LSA over keyword search."
        )

    st.divider()
    st.subheader("What is a Lemmatized Corpus?")

    st.markdown(
        "Before feeding text to LSA, we **lemmatize** it — reducing each word to its canonical base form:"
    )
    st.markdown(
        "| Surface forms | Lemma | Why it helps |\n"
        "|--------------|-------|--------------|\n"
        "| runs, ran, running, runner | `run` | 4 features collapse to 1 → stronger co-occurrence |\n"
        "| haunted, haunting, haunt | `haunt` | All signal the same horror theme |\n"
        "| mysterious, mysteriously | `mysterious` | Adverb form removed — kept as adjective |\n"
    )

    st.markdown(
        "We also **POS-filter** — keeping only nouns, verbs, and adjectives. "
        "Prepositions, articles, and conjunctions appear in *every* document and provide no discriminating signal."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Keep** (content words):")
        st.code("NN, NNS, NNP, NNPS    → nouns\nVB, VBD, VBG...       → verbs\nJJ, JJR, JJS          → adjectives", language="text")
    with col2:
        st.markdown("**Discard** (function words):")
        st.code("DT   → the, a, an\nIN   → in, on, at, of\nCC   → and, or, but\nPRP  → he, she, they", language="text")

    if artifacts is not None:
        st.divider()
        st.subheader("Sample Lemmatized Document")
        sample = artifacts.df.iloc[0]
        plot_col = "plot" if "plot" in artifacts.df.columns else "description"
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**{sample['title']}** ({int(sample['year']) if pd.notna(sample['year']) else '?'})")
            st.markdown("*Raw plot (first 100 words):*")
            raw_words = str(sample[plot_col]).split()
            st.code(" ".join(raw_words[:100]) + "...", language="text")
        with col2:
            st.markdown("*Lemmatized corpus entry (first 100 tokens):*")
            lemma_words = artifacts.corpus[0].split()
            st.code(" ".join(lemma_words[:100]) + "...", language="text")
            raw_n = len(str(sample[plot_col]).split())
            lemma_n = len(lemma_words)
            st.caption(f"Raw: {raw_n} words → Lemmatized: {lemma_n} tokens "
                       f"({(1 - lemma_n/raw_n)*100:.0f}% reduction)")

# ══════════════════════════════════════════════════════════════════════════════
# 4. TF-IDF
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Text Representation: TF-IDF":
    st.title("Text Representation — TF-IDF")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### What is TF-IDF?")
        st.latex(r"\text{TF-IDF}(t,d) = \underbrace{(1+\log \text{TF}(t,d))}_{\text{sublinear TF}} \times \underbrace{\log\frac{N}{\text{DF}(t)}}_{\text{IDF}}")
        st.markdown(
            "- **TF** — how often term $t$ appears in document $d$ (sublinear to dampen repetition)\n"
            "- **IDF** — how rare $t$ is across all $N$ documents (rewards discriminating terms)\n"
            "- Result: each document becomes a **sparse vector** in $V$-dimensional term space"
        )
    with col2:
        st.markdown("### Configuration")
        st.table(pd.DataFrame({
            "Parameter": ["stop_words", "sublinear_tf", "min_df", "max_df", "ngram_range"],
            "Value": ['"english"', "True", "3", "0.90", "(1, 2)"],
            "Reason": [
                "Remove 318 function words",
                "1+log(tf) — prevent term dominance",
                "Discard terms in < 3 docs (too rare)",
                "Discard terms in > 90% docs (too common)",
                "Include bigrams: 'haunted house'",
            ],
        }))

    if artifacts is not None and X_tfidf is not None:
        st.divider()
        plot_col_used = "plot" if "plot" in artifacts.df.columns else "description"
        n, v = X_tfidf.shape
        sp = round((1 - X_tfidf.nnz / (n * v)) * 100, 2)
        c1, c2, c3 = st.columns(3)
        c1.metric("Matrix shape", f"{n:,} × {v:,}")
        c2.metric("Sparsity", f"{sp}%")
        c3.metric("Non-zeros", f"{X_tfidf.nnz:,}")

        st.subheader("Top 25 Terms by Average TF-IDF Weight")
        feature_names = artifacts.tfidf.get_feature_names_out()
        avg = np.asarray(X_tfidf.mean(axis=0)).ravel()
        top_idx = np.argsort(avg)[-25:][::-1]

        fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
        sns.barplot(x=avg[top_idx], y=feature_names[top_idx],
                    hue=feature_names[top_idx], palette="mako", legend=False, ax=ax)
        ax.set_title("Top 25 Terms by Average TF-IDF — Lemmatized Corpus")
        ax.set_xlabel("Average TF-IDF Weight")
        ax.set_ylabel("Term (Lemma / Bigram)")
        st.pyplot(fig)

        st.subheader("IDF Extremes")
        idf_vals = artifacts.tfidf.idf_
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Most discriminating (highest IDF)**")
            top_idf = np.argsort(idf_vals)[-10:][::-1]
            st.dataframe(pd.DataFrame({
                "Term": feature_names[top_idf],
                "IDF": idf_vals[top_idf].round(4),
            }), width="stretch")
        with col2:
            st.markdown("**Least discriminating (lowest IDF)**")
            bot_idf = np.argsort(idf_vals)[:10]
            st.dataframe(pd.DataFrame({
                "Term": feature_names[bot_idf],
                "IDF": idf_vals[bot_idf].round(4),
            }), width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# 5. SVD
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Core Method: SVD":
    st.title("Core Method: Singular Value Decomposition (SVD)")
    st.divider()

    st.latex(r"A = U \Sigma V^T \quad \Rightarrow \quad A \approx U_k \Sigma_k V_k^T")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "### The Three Matrices\n\n"
            "| Matrix | Shape | Meaning |\n"
            "|--------|-------|---------|\n"
            "| $U_k$ | $42{,}303 \\times k$ | **Document-Topic** — row $i$ = movie $i$'s topic weights |\n"
            "| $\\Sigma_k$ | $k \\times k$ | **Diagonal** — singular values = topic importance |\n"
            "| $V_k^T$ | $k \\times V$ | **Topic-Term** — row $j$ = terms defining topic $j$ |"
        )
    with col2:
        st.markdown(
            "### Why Truncate?\n\n"
            "The full SVD captures everything — including **noise**.\n\n"
            "- **Large singular values** → coherent topics (horror, romance, action)\n"
            "- **Small singular values** → noise (rare term combinations, typos)\n\n"
            "Truncating to $k=150$ removes noise and forces the model to **generalize** "
            "across vocabulary variation — the core mechanism of LSA."
        )

    st.subheader("Conceptual Example: 6 Movies × 5 Terms")
    st.markdown("A hand-crafted example showing SVD discovering genre structure automatically:")

    terms = ["ghost", "haunt", "house", "love", "heart"]
    docs  = ["Horror 1", "Horror 2", "Horror 3", "Romance 1", "Romance 2", "Romance 3"]
    A_toy = np.array([
        [0.9, 0.8, 0.7, 0.0, 0.0],
        [0.8, 0.7, 0.0, 0.1, 0.0],
        [0.7, 0.9, 0.6, 0.0, 0.1],
        [0.0, 0.0, 0.1, 0.9, 0.8],
        [0.0, 0.1, 0.0, 0.8, 0.9],
        [0.1, 0.0, 0.0, 0.7, 0.8],
    ])
    U_toy, sigma_toy, Vt_toy = np.linalg.svd(A_toy, full_matrices=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)

    # A heatmap
    ax = axes[0]
    sns.heatmap(A_toy, annot=True, fmt=".1f", xticklabels=terms, yticklabels=docs,
                cmap="Blues", ax=ax)
    ax.set_title("Matrix A — Input\n(6 movies × 5 terms)")

    # V^T heatmap
    ax = axes[1]
    sns.heatmap(Vt_toy[:3], annot=True, fmt=".2f", xticklabels=terms,
                yticklabels=["Topic 0", "Topic 1", "Topic 2"],
                cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("V^T — Topic-Term Matrix\n(top 3 topics)\nred=positive, blue=negative")

    # U scatter
    ax = axes[2]
    colors = ["#EF4444"] * 3 + ["#3B82F6"] * 3
    for i, (d, c) in enumerate(zip(docs, colors)):
        ax.scatter(U_toy[i, 0], U_toy[i, 1], color=c, s=160, zorder=3)
        ax.annotate(d, (U_toy[i, 0], U_toy[i, 1]),
                    textcoords="offset points", xytext=(6, 4), fontsize=10)
    ax.axhline(0, color="gray", lw=0.7)
    ax.axvline(0, color="gray", lw=0.7)
    ax.set_xlabel("Latent Dimension 1")
    ax.set_ylabel("Latent Dimension 2")
    ax.set_title("U — Documents in 2D Latent Space\nHorror=red  Romance=blue")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#EF4444", label="Horror"),
                       Patch(color="#3B82F6", label="Romance")])
    plt.suptitle("SVD discovers genre structure with no labels — from word co-occurrence alone",
                 fontsize=12, y=1.02)
    st.pyplot(fig)

    st.info(
        "Horror movies separate onto the left-negative side of Dim 1; "
        "Romance movies to the right-positive side. "
        "SVD discovered this purely from which words co-occur in the same movies."
    )

    if artifacts is not None:
        st.divider()
        st.subheader("Real Model — U, Σ, V^T")

        svd_model = artifacts.svd
        feature_names = artifacts.tfidf.get_feature_names_out()

        fig, axes = plt.subplots(1, 3, figsize=(20, 5), constrained_layout=True)

        # Σ spectrum
        ax = axes[0]
        svs  = svd_model.singular_values_
        evr  = svd_model.explained_variance_ratio_
        cum  = np.cumsum(evr)
        ax.bar(range(1, len(svs) + 1), svs, color="#6366F1", alpha=0.7)
        ax.set_title("Σ — Singular Value Spectrum", fontsize=11)
        ax.set_xlabel("Component k")
        ax.set_ylabel("σₖ")
        ax2 = ax.twinx()
        ax2.plot(range(1, len(cum) + 1), cum, color="#F59E0B", lw=2)
        ax2.set_ylabel("Cumulative EVR", color="#F59E0B")
        ax2.tick_params(axis="y", labelcolor="#F59E0B")
        ax2.set_ylim(0, 1.05)

        # V^T heatmap — first 8 topics, union of top terms
        ax = axes[1]
        N_SHOW = 8
        all_top = set()
        for ci in range(N_SHOW):
            all_top.update(np.argsort(np.abs(svd_model.components_[ci]))[-8:].tolist())
        all_top = sorted(all_top)[:18]
        vt_sub = svd_model.components_[:N_SHOW, :][:, all_top]
        term_lbls = [feature_names[i] for i in all_top]
        sns.heatmap(vt_sub, xticklabels=term_lbls,
                    yticklabels=[f"Topic {i}" for i in range(N_SHOW)],
                    cmap="RdBu_r", center=0, ax=ax, linewidths=0.3)
        ax.set_title("V^T — Topic-Term Weights\n(first 8 topics)", fontsize=11)
        ax.tick_params(axis="x", rotation=45, labelsize=7)

        # U scatter
        ax = axes[2]
        sidx = np.random.default_rng(42).choice(len(artifacts.df), 700, replace=False)
        genres_s = artifacts.df["genre"].apply(
            lambda g: g[0] if isinstance(g, list) and g else "Unknown"
        ).iloc[sidx]
        pdata = pd.DataFrame({
            "x": artifacts.X_lsa[sidx, 0],
            "y": artifacts.X_lsa[sidx, 1],
            "genre": genres_s.values,
        })
        sns.scatterplot(data=pdata, x="x", y="y", hue="genre",
                        alpha=0.55, s=18, ax=ax, legend=True)
        ax.set_title("U — Documents in 2D LSA Space\n(700 sample)", fontsize=11)
        ax.set_xlabel("Dim 1")
        ax.set_ylabel("Dim 2")
        ax.legend(loc="best", fontsize=6, markerscale=0.8, framealpha=0.6)

        plt.suptitle("Real LSA Model — U, Σ, V^T  (42k movies, CoreNLP lemmas, k=150)",
                     fontsize=13)
        st.pyplot(fig)

        st.subheader("Top Terms per Latent Topic")
        st.markdown("Each row of $V^T$ defines a topic. Here are the top words for the first 10:")
        topic_data = []
        for ci in range(10):
            top_t = [feature_names[i] for i in np.argsort(svd_model.components_[ci])[-8:][::-1]]
            topic_data.append({
                "Topic": f"Topic {ci}",
                f"σ = {svd_model.singular_values_[ci]:.1f}": "",
                "Top 8 terms": "  |  ".join(top_t),
            })
        st.table(pd.DataFrame(topic_data)[["Topic", "Top 8 terms"]])

# ══════════════════════════════════════════════════════════════════════════════
# 6. COSINE SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Search: Cosine Similarity":
    st.title("Search Method — Cosine Similarity")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### What is it?")
        st.latex(r"\text{sim}(\vec{q}, \vec{m}) = \cos(\theta) = \frac{\vec{q} \cdot \vec{m}}{\|\vec{q}\| \cdot \|\vec{m}\|}")
        st.markdown(
            "Measures the **angle** between two vectors in the $k$-dimensional latent space.\n\n"
            "| Value | Meaning |\n"
            "|-------|---------|\n"
            "| $\\cos\\theta = 1$ | Perfect match — identical direction |\n"
            "| $\\cos\\theta = 0$ | No overlap — orthogonal topics |\n"
            "| $\\cos\\theta < 0$ | Opposite direction — thematically opposite |"
        )
    with col2:
        st.markdown("### Why not Euclidean distance?")
        st.markdown(
            "Euclidean distance $\\|\\vec{q} - \\vec{m}\\|$ is **sensitive to magnitude** "
            "(document length). A short plot and a long plot on the same topic would appear far apart.\n\n"
            "Cosine only measures **direction** — invariant to how long the plot is.\n\n"
            "After **L2-normalization**, $\\|\\vec{m}\\| = 1$ for all movies, so:\n"
        )
        st.latex(r"\cos(\theta) = \vec{q} \cdot \vec{m}")
        st.markdown("All 42,303 scores = one matrix multiply. **Milliseconds total.**")

    st.subheader("The Complete Search Pipeline")
    st.code(
        "1. q_tfidf = tfidf.transform([query])       # (1 × V) sparse\n"
        "2. q_lsa   = lsa.transform(q_tfidf)         # (1 × k) dense, L2-normalised\n"
        "3. scores  = X_lsa @ q_lsa.T                # (42303,) — cosine similarities\n"
        "4. top_idx = argsort(scores)[::-1][:top_k]  # rank\n"
        "5. return  df.iloc[top_idx]                 # top-k movies",
        language="python",
    )

    if artifacts is not None:
        st.divider()
        st.subheader("Interactive Similarity Explorer")
        eg_query = st.text_input("Enter a query to visualize score distribution", value="haunted house ghost")
        if eg_query.strip():
            q_vec = artifacts.tfidf.transform([eg_query])
            q_lsa_vec = artifacts.lsa.transform(q_vec)
            scores = _cos_sim(q_lsa_vec, artifacts.X_lsa)[0]
            sorted_scores = np.sort(scores)[::-1]

            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(7, 3.5), constrained_layout=True)
                ax.plot(range(1, 101), sorted_scores[:100], color="#10B981", lw=2)
                ax.fill_between(range(1, 101), 0, sorted_scores[:100], alpha=0.15, color="#10B981")
                ax.set_title(f"Top-100 Similarity Scores — '{eg_query}'")
                ax.set_xlabel("Rank")
                ax.set_ylabel("Cosine Similarity")
                st.pyplot(fig)
            with col2:
                top5 = search_movies(eg_query, artifacts, top_k=10)
                st.markdown("**Top 10 Results:**")
                st.dataframe(top5[["title", "similarity"]].assign(
                    similarity=top5["similarity"].round(4)
                ), width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# 7. RESULTS & ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Results & Analysis":
    st.title("Results & Analysis")
    st.divider()

    if artifacts is None:
        st.warning("Model not loaded.")
        st.stop()

    test_queries = [
        ("haunted house ghost supernatural", "Horror / Supernatural"),
        ("space mission astronaut alien planet", "Sci-Fi / Space"),
        ("detective mystery crime murder", "Mystery / Crime"),
        ("love story romance relationship marriage", "Romance / Drama"),
        ("war battlefield soldier survival enemy", "War / Action"),
        ("heist robbery gang money bank", "Crime / Heist"),
    ]

    st.subheader("Search Results Across 6 Semantic Categories")
    for query, label in test_queries:
        with st.expander(f"**{label}** — `{query}`"):
            results = search_movies(query, artifacts, top_k=8)
            if not results.empty:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.dataframe(results[["title", "genre", "year", "similarity"]], width="stretch")
                with col2:
                    fig, ax = plt.subplots(figsize=(4, 3.5), constrained_layout=True)
                    sns.barplot(data=results, y="title", x="similarity",
                                hue="title", palette="crest", legend=False, ax=ax)
                    ax.set_title("Similarity Scores", fontsize=9)
                    ax.set_xlabel("Cosine Similarity")
                    ax.set_ylabel("")
                    ax.tick_params(axis="y", labelsize=7)
                    st.pyplot(fig)
            else:
                st.write("No results.")

    st.divider()
    st.subheader("Term-Level Explanation")
    st.markdown(
        "For each result, we project the query–document overlap back through $V^T$ to identify "
        "which vocabulary terms drove the match:\n\n"
        "$$\\vec{o} = \\vec{q}_{\\text{lsa}} \\odot \\vec{m}_i \\quad \\Rightarrow \\quad "
        "\\vec{t} = \\vec{o} \\cdot V^T$$"
    )

    explain_q = st.text_input("Query for explanation", value="haunted house ghost supernatural")
    if explain_q.strip():
        top5 = search_movies(explain_q, artifacts, top_k=5)
        if not top5.empty:
            explanations = explain_results(explain_q, top5, artifacts, top_n=6)
            for idx, row in top5.iterrows():
                with st.expander(f"{row['title']} (similarity={row['similarity']:.4f})"):
                    st.markdown(f"**Plot:** {row['description']}")
                    terms = explanations.get(idx, [])
                    if terms:
                        st.dataframe(pd.DataFrame(terms, columns=["Term", "Score"]),
                                     width="stretch")

# ══════════════════════════════════════════════════════════════════════════════
# 8. LIMITATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Limitations":
    st.title("Limitations")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Technical Limitations")
        st.table(pd.DataFrame({
            "Limitation": [
                "Bag-of-words — no word order",
                "No negation handling",
                "Static model",
                "Vocabulary boundary",
                "Dimensionality choice (k)",
            ],
            "Impact": [
                '"dog bites man" = "man bites dog" in TF-IDF',
                '"not a love story" matches romance movies',
                "Must retrain when new movies are added",
                "Out-of-vocabulary query words silently ignored",
                "k too small = lossy; k too large = noise included",
            ],
        }))
    with col2:
        st.markdown("### Data Limitations")
        st.table(pd.DataFrame({
            "Limitation": [
                "Short plots (~8% < 50 words)",
                "CoreNLP dependency",
                "Plot quality varies",
                "No user feedback signal",
                "Genre noise",
            ],
            "Impact": [
                "Near-empty TF-IDF vectors → noisy similarity",
                "Lemma quality bounded by CoreNLP accuracy",
                "Wikipedia summaries are inconsistently written",
                "Cannot improve relevance from user clicks",
                "Freebase genre taxonomy has inconsistencies",
            ],
        }))

    st.divider()
    st.subheader("Potential Next Steps")
    st.markdown(
        "| Improvement | Expected Benefit |\n"
        "|-------------|------------------|\n"
        "| BM25 instead of TF-IDF | Better term weighting for retrieval |\n"
        "| Sentence-BERT embeddings | Context-aware representations — handles negation |\n"
        "| FAISS approximate nearest neighbor | Scale to millions of movies |\n"
        "| User feedback (click logs) | Relevance tuning — learning to rank |\n"
        "| Incremental SVD | Add new movies without full retrain |"
    )

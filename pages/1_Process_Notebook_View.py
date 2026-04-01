import streamlit as st

from streamlit_ui import (
    chart_explained_variance,
    chart_genre,
    chart_lsa_scatter,
    chart_top_terms,
    chart_year_distribution,
    get_artifacts,
    process_functions_for_display,
    save_current_artifacts,
    show_code,
    sidebar_controls,
)

st.title("Process Notebook View")
st.caption("Mirrors the main.ipynb pipeline: data → corpus → TF-IDF → SVD/LSA → search")

n_components, rebuild, save = sidebar_controls()
artifacts, X_tfidf, mapping = get_artifacts(n_components, rebuild)

if save:
    save_current_artifacts()

if artifacts is None:
    st.info("Loading model…")
    st.stop()

st.subheader("1) Dataset — Wikipedia Movie Plots")
c1, c2, c3 = st.columns(3)
c1.metric("Movies", f"{len(artifacts.df):,}")
c2.metric("SVD Components", f"{artifacts.svd.n_components}")
c3.metric("Vocabulary Size", f"{len(artifacts.tfidf.get_feature_names_out()):,}")

fingerprint_display = (artifacts.data_fingerprint[:16] + "…") if artifacts.data_fingerprint else "N/A"
st.caption(
    f"Built at: **{artifacts.created_at or 'N/A'}** | "
    f"Model version: **{artifacts.version}** | "
    f"Data MD5: `{fingerprint_display}`"
)

if mapping:
    with st.expander("Column mapping", expanded=False):
        st.json(mapping.get("mapped_columns", {}))

st.dataframe(artifacts.df.head(10), width="stretch")

st.subheader("2) Exploratory Visualizations")
st.pyplot(chart_year_distribution(artifacts.df))
st.pyplot(chart_genre(artifacts.df))
st.markdown(
    "**Analysis**: The corpus spans over a century of films across diverse genres, "
    "giving LSA a rich vocabulary of co-occurring plot words to learn from."
)

st.subheader("3) Corpus + TF-IDF")
st.write(f"Corpus size: **{len(artifacts.corpus):,}** documents")
if artifacts.corpus:
    with st.expander("Sample document", expanded=False):
        st.code(artifacts.corpus[0][:700], language="text")

if X_tfidf is not None:
    sparsity = round((1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])) * 100, 2)
    st.write(f"TF-IDF matrix: `{X_tfidf.shape[0]:,} × {X_tfidf.shape[1]:,}` — sparsity {sparsity}%")
    st.pyplot(chart_top_terms(artifacts.tfidf, X_tfidf))
    st.markdown("**Analysis**: Top average TF-IDF terms reflect dominant narrative vocabulary in Wikipedia plots.")

st.subheader("4) SVD / LSA Projection")
st.latex(r"A \approx U \Sigma V^T")
st.markdown(
    "TF-IDF matrix $A$ is factorized into latent components. "
    "Each movie's row in $U\\Sigma$ is its **latent semantic vector** — "
    "dimensionally compressed so thematically similar films cluster together."
)
fig_var, explained = chart_explained_variance(artifacts.svd)
st.pyplot(fig_var)
st.write(f"Variance captured with {artifacts.svd.n_components} components: **{explained:.4f}**")
st.pyplot(chart_lsa_scatter(artifacts.df, artifacts.X_lsa))
st.markdown(
    "**Analysis**: The 2D scatter shows genre clusters emerging from purely text-based "
    "latent projection — no explicit genre labels were used during SVD training."
)

st.subheader("5) Pipeline Source Code")
for fn in process_functions_for_display():
    show_code(fn.__name__, fn)

import streamlit as st

from streamlit_ui import (
    chart_explained_variance,
    chart_genre,
    chart_lsa_scatter,
    chart_top_terms,
    chart_year_rating,
    materialize_pipeline,
    process_functions_for_display,
    save_current_artifacts,
    show_code,
    sidebar_pipeline_controls,
)

st.title("Process Notebook View")
st.caption("Notebook-equivalent process: code + charts + tables + analysis")

(
    dataset_slug,
    n_components,
    artifact_path,
    load_saved,
    build_fresh,
    save_now,
) = sidebar_pipeline_controls(default_dataset="chenyanglim/imdb-v2", default_n_components=100)

artifacts, X_tfidf, mapping = materialize_pipeline(
    dataset_slug=dataset_slug,
    n_components=n_components,
    artifact_path=artifact_path,
    load_saved=load_saved,
    build_fresh=build_fresh,
)

if artifacts is None:
    st.warning("Build or load artifacts from the sidebar to display the process.")
    st.stop()

if save_now:
    save_current_artifacts(artifact_path)

st.subheader("1) Download + Normalize Dataset")
c1, c2, c3 = st.columns(3)
c1.metric("Movies", f"{len(artifacts.df):,}")
c2.metric("SVD Components", f"{artifacts.svd.n_components}")
c3.metric("Vocabulary Size", f"{len(artifacts.tfidf.get_feature_names_out()):,}")

fingerprint_display = (artifacts.data_fingerprint[:16] + "...") if artifacts.data_fingerprint else "N/A"
st.caption(
    f"Model version: **{artifacts.version}** | "
    f"Built at: **{artifacts.created_at or 'N/A'}** | "
    f"Data fingerprint (MD5): `{fingerprint_display}`"
)

st.markdown("**Analysis**: The dataset is normalized into a consistent schema (id, title, genre, description, year, rating), which makes downstream modeling robust to source-column variation.")
if mapping:
    st.write("Detected column mapping")
    st.json(mapping.get("mapped_columns", {}))

st.dataframe(artifacts.df.head(10), width="stretch")

st.subheader("2) Exploratory Visualizations")
st.pyplot(chart_year_rating(artifacts.df))
st.pyplot(chart_genre(artifacts.df))
st.markdown("**Analysis**: Year/rating distributions and genre frequency indicate data balance and possible retrieval bias across genres.")

st.subheader("3) Build Corpus + TF-IDF")
st.write(f"Corpus size: {len(artifacts.corpus):,}")
if artifacts.corpus:
    st.write("Sample combined document")
    st.code(artifacts.corpus[0][:700], language="text")

st.write(f"TF-IDF matrix shape: {X_tfidf.shape}")
sparsity = round((1 - X_tfidf.nnz / (X_tfidf.shape[0] * X_tfidf.shape[1])) * 100, 2)
st.write(f"Sparsity (% zeros): {sparsity}")
st.pyplot(chart_top_terms(artifacts.tfidf, X_tfidf))
st.markdown("**Analysis**: Top average TF-IDF terms reveal dominant lexical signals that influence retrieval.")

st.subheader("4) SVD / LSA Projection")
fig_var, explained = chart_explained_variance(artifacts.svd)
st.pyplot(fig_var)
st.write(f"Explained variance with {artifacts.svd.n_components} components: {explained:.4f}")
st.pyplot(chart_lsa_scatter(artifacts.df, artifacts.X_lsa))
st.markdown("**Analysis**: Cumulative explained variance helps choose latent dimensionality, while 2D scatter gives an intuition of semantic neighborhood structure.")

st.subheader("5) Process Code (from implementation)")
for fn in process_functions_for_display():
    show_code(fn.__name__, fn)

st.success("Process page is aligned with main.ipynb flow and now includes notebook-like narrative analysis.")

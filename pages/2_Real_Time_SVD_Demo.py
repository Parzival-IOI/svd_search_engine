import streamlit as st

from streamlit_ui import (
    chart_result_scores,
    dataset_bounds,
    materialize_pipeline,
    run_search,
    save_current_artifacts,
    sidebar_pipeline_controls,
)

st.title("Real-Time SVD Search Demo")
st.caption("Interactive semantic retrieval with filters and weighted ranking")

(
    dataset_slug,
    n_components,
    artifact_path,
    load_saved,
    build_fresh,
    save_now,
) = sidebar_pipeline_controls(default_dataset="chenyanglim/imdb-v2", default_n_components=100)

artifacts, _, _ = materialize_pipeline(
    dataset_slug=dataset_slug,
    n_components=n_components,
    artifact_path=artifact_path,
    load_saved=load_saved,
    build_fresh=build_fresh,
)

if artifacts is None:
    st.warning("Build or load artifacts from the sidebar to start real-time search.")
    st.stop()

if save_now:
    save_current_artifacts(artifact_path)

all_genres = sorted({g for genres in artifacts.df["genre"] for g in genres})
year_min, year_max, rating_min_data, rating_max_data = dataset_bounds(artifacts.df)

st.subheader("Search Controls")
c1, c2, c3, c4 = st.columns(4)
query = c1.text_input("Query", value="black cop")
genre_filter = c2.selectbox("Genre", ["All"] + all_genres)
year_range = c3.slider("Year range", min_value=year_min, max_value=year_max, value=(year_min, year_max))
rating_min = c4.slider(
    "Minimum rating",
    min_value=float(rating_min_data),
    max_value=float(rating_max_data),
    value=float(rating_min_data),
)

c5, c6 = st.columns(2)
top_k = c5.slider("Top K", min_value=3, max_value=30, value=10)
alpha = c6.slider("Similarity weight (alpha)", min_value=0.0, max_value=1.0, value=0.85, step=0.05)

result_df = run_search(
    query=query,
    artifacts=artifacts,
    top_k=top_k,
    genre_filter=genre_filter,
    year_range=year_range,
    rating_min=rating_min,
    alpha=alpha,
)

if result_df.empty:
    st.warning("No results found. Try broader filters or a different query.")
else:
    st.subheader("Top Results")
    st.dataframe(result_df, width="stretch")
    st.pyplot(chart_result_scores(result_df))

    st.markdown("### Score Interpretation")
    st.markdown("- similarity: cosine similarity in latent semantic space")
    st.markdown("- weighted_score: alpha * similarity + (1 - alpha) * normalized_rating")

st.subheader("Quick Multi-Query Check")
query_block = st.text_area(
    "One query per line",
    value="space mission\nlove story romance\nmysterious detective case\nterrifying threat friends",
    height=120,
)

for q in [line.strip() for line in query_block.splitlines() if line.strip()]:
    st.markdown(f"**Query:** {q}")
    top3 = run_search(
        query=q,
        artifacts=artifacts,
        top_k=3,
        genre_filter=genre_filter,
        year_range=year_range,
        rating_min=rating_min,
        alpha=alpha,
    )
    if top3.empty:
        st.write("No result")
    else:
        st.dataframe(top3[["title", "genre", "weighted_score"]], width="stretch")

st.info("This page is optimized for real-time experimentation with search behavior.")

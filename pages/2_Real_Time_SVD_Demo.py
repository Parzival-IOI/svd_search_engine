import sys
from pathlib import Path

_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import pandas as pd
import streamlit as st

from src.svd_search.models.evaluate import explain_results
from streamlit_ui import (
    chart_result_scores,
    dataset_bounds,
    get_artifacts,
    run_search,
)

st.title("Real-Time SVD Search Demo")
st.caption("Interactive semantic retrieval using the pre-trained CoreNLP LSA model")

# Load the pre-trained artifact — no slider, no rebuild, no save
artifacts, _, _ = get_artifacts(n_components=150, rebuild=False)

if artifacts is None:
    st.info("Loading model…")
    st.stop()

# Sidebar: read-only model info
st.sidebar.header("Model Info")
st.sidebar.caption("Corpus: CMU MovieSummaries + CoreNLP lemmas")
_fp = (artifacts.data_fingerprint[:16] + "…") if artifacts.data_fingerprint else "N/A"
st.sidebar.markdown(
    f"- **Movies**: {len(artifacts.df):,}\n"
    f"- **SVD components**: {artifacts.svd.n_components}\n"
    f"- **Vocabulary**: {len(artifacts.tfidf.get_feature_names_out()):,} terms\n"
    f"- **Model version**: {artifacts.version}\n"
    f"- **Built at**: {artifacts.created_at or 'N/A'}\n"
)

all_genres = sorted({g for genres in artifacts.df["genre"] for g in genres})
year_min, year_max = dataset_bounds(artifacts.df)

st.subheader("Search Controls")
c1, c2, c3 = st.columns(3)
query = c1.text_input("Query", value="space adventure")
genre_filter = c2.selectbox("Genre", ["All"] + all_genres)
year_range = c3.slider(
    "Year range", min_value=year_min, max_value=year_max, value=(year_min, year_max)
)

c4, _ = st.columns(2)
top_k = c4.slider("Top K", min_value=3, max_value=30, value=10)

result_df = run_search(
    query=query,
    artifacts=artifacts,
    top_k=top_k,
    genre_filter=genre_filter,
    year_range=year_range,
)

if result_df.empty:
    st.warning("No results found. Try a different query or broaden the year range.")
else:
    st.subheader("Top Results")
    st.dataframe(
        result_df[["title", "genre", "year", "similarity"]],
        width="stretch",
    )
    st.pyplot(chart_result_scores(result_df))

    st.markdown("### Score")
    st.markdown("- **similarity**: cosine similarity in the LSA latent semantic space")

    st.markdown("### Why These Results?")
    explanations = explain_results(query, result_df, artifacts, top_n=5)
    for idx, row in result_df.iterrows():
        with st.expander(f"{row['title']}"):
            st.markdown(f"**Plot:** {row['description']}")
            terms = explanations.get(idx, [])
            if terms:
                st.markdown("**Top matching terms:**")
                st.dataframe(
                    pd.DataFrame(terms, columns=["Term", "Score"]),
                    width="stretch",
                )


st.info("This page is optimized for real-time experimentation with search behavior.")

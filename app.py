import sys
from pathlib import Path

# Ensure src/ is on the path before any svd_search imports
_src = Path(__file__).resolve().parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

import streamlit as st

st.set_page_config(page_title="SVD Movie Search Engine", layout="wide")

st.title("SVD Movie Search Engine")
st.caption("CMU MovieSummaries + CoreNLP lemmatized corpus · LSA · Cosine Similarity")

st.markdown("### Pages")
st.markdown("- **Process Notebook View** — mirrors the notebook pipeline: data → TF-IDF → SVD/LSA → search")
st.markdown("- **Real-Time SVD Demo** — interactive semantic search with live filters")

st.markdown("### How To Start")
st.code("streamlit run app.py", language="bash")

st.info("Use the left sidebar to navigate between pages.")

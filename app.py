import streamlit as st
st.set_page_config(page_title="SVD Movie Search Engine", layout="wide")

st.title("SVD Movie Search Engine")
st.caption("Streamlit multipage app aligned with main.ipynb")

st.markdown("### Pages")
st.markdown("- Process Notebook View: mirrors the notebook process, code, charts, tables, and analysis")
st.markdown("- Real-Time SVD Demo: interactive search with live filters and weighted ranking")

st.markdown("### How To Start")
st.code("streamlit run app.py", language="bash")

st.info("Use the left sidebar page navigation to open either page.")

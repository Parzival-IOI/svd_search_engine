# SVD Movie Search Engine

Semantic movie search built with TF-IDF + Latent Semantic Analysis (LSA) using Truncated SVD.

The workflow uses Wikipedia movie plot descriptions from Kaggle, projects movie text and user queries into latent semantic space, and ranks results with cosine similarity.

## Current Scope

- Notebook-first implementation in `main.ipynb`
- Fixed dataset: `jrobischon/wikipedia-movie-plots` (~34,886 movies)
- One-time Kaggle download, cached to `data/` for all subsequent runs
- TF-IDF vectorization with English stopword removal
- SVD-based dimensionality reduction (LSA)
- Cosine similarity search with top-k ranking
- Optional genre and year range filters
- Exploratory visualizations for data and model behavior
- Multi-query comparison section for quick qualitative evaluation
- Term-level explanation of why each result matched

## Project Structure

```text
svd_search_engine/
|- app.py                                 # Streamlit multipage entry
|- pages/
|  |- 1_Process_Notebook_View.py          # Notebook-equivalent process page
|  |- 2_Real_Time_SVD_Demo.py             # Real-time search demo page
|- streamlit_ui.py                        # Shared Streamlit UI helpers
|- search_pipeline.py                     # Reusable data/model/search pipeline
|- main.ipynb                             # End-to-end notebook pipeline and plots
|- functionality.md                       # Detailed functionality documentation
|- requirements.txt                       # Python dependencies
|- readme.md                              # Project overview and usage
```

## Notebook Pipeline

1. Import dependencies and configure plotting.
2. Fetch `jrobischon/wikipedia-movie-plots` from Kaggle (once); read from cache on subsequent runs.
3. Normalize columns: Title → title, Genre → genre, Plot → description, Release Year → year.
4. Build corpus with title + genre + description.
5. Fit TF-IDF matrix.
6. Train TruncatedSVD + Normalizer LSA pipeline.
7. Search queries using cosine similarity.
8. Visualize dataset, token importance, latent variance, and result scores.
9. Compare multiple queries and explain term-level contributions.

## Dataset Notes

- Dataset: `jrobischon/wikipedia-movie-plots` on Kaggle
- ~34,886 movies with Wikipedia plot summaries
- Columns used: Title, Genre, Plot, Release Year
- No rating column — search is ranked purely by semantic similarity

## Requirements

- Python 3.10+
- Jupyter-compatible environment
- Kaggle access configured for kagglehub

Install dependencies:

```bash
pip install -r requirements.txt
```

If notebook widgets are missing, install:

```bash
pip install ipywidgets
```

## How To Run

1. Open main.ipynb in VS Code or Jupyter.
2. Run cells in order from top to bottom.
3. Change the query value in the search cell to test retrieval behavior.
4. Use the multi-query section to compare semantic intent coverage.

## Run Streamlit App

```bash
streamlit run app.py
```

The app is organized in Streamlit's native multipage structure (`app.py` + `pages/`).

Available pages:

- **Process Notebook View**: step-by-step pipeline walkthrough with code, charts, tables, and narrative analysis aligned with `main.ipynb`.
- **Real-Time SVD Demo**: interactive search UI with genre and year filters.

The app adds:

- Artifact persistence with joblib
- Filters by genre and year range
- Interactive charts and data tables for each pipeline stage

## Example Queries

- space mission
- love story romance
- mysterious detective case
- terrifying threat friends

## Technical Notes

- Vectorizer: `TfidfVectorizer(stop_words="english")`
- Dimensionality reduction: `TruncatedSVD` with component count capped to valid bounds
- Similarity: `cosine_similarity` in latent space
- Search output includes title, genre, year, and similarity score

## Limitations

- Bag-of-words + LSA is not as context-aware as transformer embeddings.
- Retrieval quality depends on source text quality.

## Implemented Improvements

- Persist trained artifacts (vectorizer, SVD, latent matrix) with joblib.
- Add filters by genre and year before ranking.
- Build a Streamlit multipage UI for interactive serving.
- Term-level explanation of search results.

## Next Improvements

- Add a FastAPI backend for programmatic search.
- Add model versioning and dataset fingerprinting in saved artifacts.
- Add result explanations (top contributing terms per hit).

## License

No license file is currently included. Add a license (for example MIT) before distribution.

# SVD Movie Search Engine (Notebook)

Semantic movie search notebook built with TF-IDF + Latent Semantic Analysis (LSA) using Truncated SVD.

The workflow uses a real IMDb dataset from Kaggle, projects movie text and user queries into latent semantic space, and ranks results with cosine similarity.

## Current Scope

- Notebook-first implementation in main.ipynb
- Real dataset ingestion from Kaggle (chenyanglim/imdb-v2)
- Schema normalization for robust column mapping
- TF-IDF vectorization with English stopword removal
- SVD-based dimensionality reduction (LSA)
- Cosine similarity search with top-k ranking
- Exploratory visualizations for data and model behavior
- Multi-query comparison section for quick qualitative evaluation

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
2. Download IMDb-v2 from Kaggle via kagglehub.
3. Detect available tabular file (csv/tsv/parquet).
4. Map source columns to normalized schema:
	- id
	- title
	- genre
	- description
	- year
	- rating
5. Build corpus with title + genre + description.
6. Fit TF-IDF matrix.
7. Train TruncatedSVD + Normalizer LSA pipeline.
8. Search queries using cosine similarity.
9. Visualize dataset, token importance, latent variance, and result scores.

## Dataset Notes

The notebook currently targets:

- Kaggle dataset: chenyanglim/imdb-v2
- Flexible column detection with fallback candidates for title, genre, description, year, and rating
- Missing-value handling for genres, year, rating, and empty descriptions

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

- Process Notebook View: process, code, charts, tables, and narrative analysis aligned with main.ipynb.
- Real-Time SVD Demo: interactive search UI with filters and weighted ranking.

The app adds:

- Artifact persistence with joblib
- Filters by genre, year range, and minimum rating
- Weighted ranking (similarity + normalized rating)
- Interactive charts and data tables for each pipeline stage

## Example Queries

- space mission
- love story romance
- mysterious detective case
- terrifying threat friends

## Technical Notes

- Vectorizer: TfidfVectorizer(stop_words="english")
- Dimensionality reduction: TruncatedSVD with component count capped to valid bounds
- Similarity: cosine_similarity in latent space
- Search output includes title, genre, year, rating, and similarity score

## Limitations

- Bag-of-words + LSA is not as context-aware as transformer embeddings.
- Retrieval quality depends on source text quality.
- Model artifacts are not persisted yet; notebook recomputes the pipeline.

## Implemented Improvements

- Persist trained artifacts (vectorizer, SVD, latent matrix) with joblib.
- Add filters by genre/year/rating before ranking.
- Add weighted ranking that combines similarity and rating.
- Build a lightweight Streamlit UI for interactive serving.

## Next Improvements

- Add a FastAPI backend for programmatic search.
- Add model versioning and dataset fingerprinting in saved artifacts.
- Add result explanations (top contributing terms per hit).

## License

No license file is currently included. Add a license (for example MIT) before distribution.

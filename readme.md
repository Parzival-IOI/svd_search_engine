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
|- main.ipynb       # End-to-end notebook pipeline and plots
|- functionality.md # Detailed functionality documentation
|- requirements.txt # Python dependencies
|- readme.md        # Project overview and usage
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

## Next Improvements

- Persist trained artifacts (vectorizer, SVD, latent matrix) with joblib.
- Add filters by genre/year/rating before ranking.
- Add weighted ranking that combines similarity and rating.
- Build API or lightweight UI for interactive serving.

## License

No license file is currently included. Add a license (for example MIT) before distribution.

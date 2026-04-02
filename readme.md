# SVD Movie Search Engine

Semantic movie retrieval using **TF-IDF + Latent Semantic Analysis (LSA)** powered by Truncated SVD and cosine similarity.

The engine supports two datasets and two corpus modes:

| Mode | Dataset | Corpus |
|------|---------|--------|
| v1 (baseline) | Wikipedia Movie Plots (Kaggle, ~34 k movies) | Raw plot text |
| v2 (default) | CMU MovieSummaries (~42 k movies) | CoreNLP POS-filtered lemmas |

The CoreNLP lemmatized corpus (v2) merges inflected forms (`runs / ran / running → run`) and keeps only nouns, verbs, and adjectives — giving LSA cleaner co-occurrence signals and better retrieval quality.

---

## Project Structure

```text
svd_search_engine/
├── app.py                          # Streamlit multipage entry point
├── streamlit_ui.py                 # Shared Streamlit UI helpers
├── search_pipeline.py              # Backward-compat shim (re-exports src/)
├── pages/
│   ├── 1_Process_Notebook_View.py  # Pipeline walkthrough page
│   └── 2_Real_Time_SVD_Demo.py     # Interactive search page
├── src/svd_search/                 # Main installable package
│   ├── __init__.py                 # Public API surface
│   ├── config/paths.py             # All paths resolved from project root
│   ├── data/loader.py              # load_cmu_dataset, load_wikipedia_dataset
│   ├── features/build_features.py  # build_corpus, build_lemma_corpus, build_model
│   ├── models/
│   │   ├── train.py                # make_artifacts, save_artifacts
│   │   ├── predict.py              # load_artifacts, search_movies
│   │   └── evaluate.py             # explain_results
│   └── utils/utils.py              # PipelineArtifacts dataclass, helpers
├── notebooks/
│   ├── main.ipynb                  # v1 — Wikipedia dataset, step-by-step
│   └── main_v2.ipynb               # v2 — CMU + CoreNLP, raw vs lemmatized
├── scripts/
│   ├── preprocess.py               # Parse CoreNLP XML → build lemma cache
│   └── train_model.py              # Fit LSA model → save artifact
├── configs/config.yaml             # Hyperparameters and paths
├── data/
│   ├── raw/                        # Wikipedia CSV (never modified)
│   ├── external/                   # CMU MovieSummaries + CoreNLP XML files
│   └── processed/                  # (reserved for future cleaned outputs)
├── models/                         # Saved joblib artifacts
│   ├── search_artifacts.joblib     # v1 Wikipedia model
│   ├── search_artifacts_v2.joblib  # v2 CoreNLP model (used by Streamlit)
│   └── corenlp_lemma_cache.joblib  # Pre-parsed lemma corpus cache
├── outputs/
│   ├── figures/                    # Plot outputs
│   └── logs/                       # Run logs
├── tests/                          # Unit tests
├── pyproject.toml                  # Package definition (setuptools)
└── requirements.txt                # pip dependencies
```

---

## Pipeline Overview

### v1 — Wikipedia Movie Plots (baseline)

```
Kaggle: jrobischon/wikipedia-movie-plots
  → load & normalize (title, genre, plot, year)
  → corpus: title + genre + plot (lowercased)
  → TfidfVectorizer(stop_words="english")
  → TruncatedSVD(k=100) + L2 Normalizer
  → cosine similarity search
```

### v2 — CMU MovieSummaries + CoreNLP (default)

```
CMU MovieSummaries: movie.metadata.tsv + plot_summaries.txt
  → join on wiki_id, parse Freebase genre dicts
CoreNLP XML.gz per movie (42 k files)
  → extract POS-filtered lemmas: {NN, NNS, VB*, JJ*} only
  → cache to models/corenlp_lemma_cache.joblib (one-time ~100 s parse)
  → corpus: title + genre + lemma tokens
  → TfidfVectorizer(sublinear_tf, min_df=3, max_df=0.90, ngram_range=(1,2))
  → TruncatedSVD(k=150) + L2 Normalizer
  → cosine similarity search
```

---

## Datasets

### Wikipedia Movie Plots (v1)
- Source: [Kaggle — jrobischon/wikipedia-movie-plots](https://www.kaggle.com/datasets/jrobischon/wikipedia-movie-plots)
- ~34,886 movies with Wikipedia plot summaries
- Columns used: `Title`, `Genre`, `Plot`, `Release Year`
- Downloaded on first run via `kagglehub`, cached to `data/raw/`

### CMU MovieSummaries (v2)
- Source: [CMU Movie Summary Corpus](http://www.cs.cmu.edu/~ark/personas/)
- ~42,303 movies after join on `wiki_id`
- Files: `movie.metadata.tsv` + `plot_summaries.txt` in `data/external/MovieSummaries/`
- Genres stored as Freebase JSON dicts — parsed into plain lists

### CoreNLP Annotations
- 42,306 `.xml.gz` files in `data/external/corenlp_plot_summaries/`
- Each file is CoreNLP-annotated XML for one movie's plot summary
- Extraction: lemmas where POS ∈ {NN, NNS, NNP, NNPS, VB, VBD, VBG, VBN, VBP, VBZ, JJ, JJR, JJS}
- Results cached to `models/corenlp_lemma_cache.joblib`

---

## Requirements

- Python 3.10+
- Kaggle credentials configured (for v1 Wikipedia download)

Install dependencies:

```bash
pip install -r requirements.txt
```

Or install the package in editable mode:

```bash
pip install -e .
```

---

## Running the Notebooks

| Notebook | Description |
|----------|-------------|
| `notebooks/main.ipynb` | v1 pipeline (Wikipedia dataset) — end-to-end walkthrough |
| `notebooks/main_v2.ipynb` | v2 pipeline (CMU + CoreNLP) — raw vs lemmatized comparison |

Open in VS Code or Jupyter and run cells top to bottom.

---

## Running the Streamlit App

```bash
streamlit run app.py
```

The app loads the CoreNLP artifact (`models/search_artifacts_v2.joblib`) by default.  
If the artifact doesn't exist, it builds in-memory from `models/corenlp_lemma_cache.joblib`.

### Pages

| Page | Description |
|------|-------------|
| **Process Notebook View** | Pipeline walkthrough with charts, code, and narrative |
| **Real-Time SVD Demo** | Interactive query box with genre and year filters |

### Sidebar controls
- **SVD components** slider (10 – 300, default 150)
- **Rebuild** — clears cache and rebuilds model in-memory
- **Save** — persists current model to `models/search_artifacts_v2.joblib`

---

## Scripts

### Build the lemma corpus (first time only, ~100 s)

```bash
python scripts/preprocess.py
```

Parses all CoreNLP XML.gz files and saves `models/corenlp_lemma_cache.joblib`.

### Train and save the full model

```bash
python scripts/train_model.py
```

Loads CMU data + lemma cache, fits TF-IDF + SVD pipeline, saves `models/search_artifacts_v2.joblib`.

---

## Example Queries

```
haunted house ghost supernatural
space mission astronaut planet
detective mystery crime investigation
love story romance relationship
war battlefield soldier survival
heist robbery gang money
```

---

## Technical Configuration

| Parameter | v1 (baseline) | v2 (default) |
|-----------|--------------|-------------|
| Dataset | Wikipedia (~34 k) | CMU MovieSummaries (~42 k) |
| Corpus | Raw text | CoreNLP lemmas (POS-filtered) |
| `stop_words` | `"english"` | `"english"` |
| `sublinear_tf` | `False` | `True` |
| `min_df` | `1` | `3` |
| `max_df` | `1.0` | `0.90` |
| `ngram_range` | `(1, 1)` | `(1, 2)` |
| SVD components | 100 | 150 |

---

## Key Findings

- **More text = better search.** Movies with fewer than 50 plot words produce near-zero TF-IDF vectors that yield noisy cosine similarity scores. ~8% of the CMU corpus falls below this threshold.
- **Lemmatization improves LSA.** Merging inflected forms into a single feature gives SVD cleaner co-occurrence patterns and produces higher, better-separated top-similarity scores.
- **Bigrams capture compound concepts.** Phrases like *"haunted house"* or *"outer space"* are unambiguous as bigrams but ambiguous as separate unigrams.

---

## Limitations

- Bag-of-words + LSA is not context-aware (no word order, no negation).
- Retrieval quality depends entirely on the quality and length of plot summaries.
- No user feedback signal for relevance tuning.

## License

No license file is currently included. Add a license (e.g. MIT) before public distribution.


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

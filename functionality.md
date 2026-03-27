# SVD-Based Movie Search Engine - Functionality Documentation

## 1. Overview

This project implements semantic movie retrieval in a notebook workflow using:

- TF-IDF for term weighting
- Truncated SVD for latent semantic projection (LSA)
- Cosine similarity for ranking

The current implementation runs end-to-end in main.ipynb and uses real IMDb data from Kaggle.

## 2. End-to-End Architecture

```text
Kaggle IMDb-v2 download
  -> schema detection and normalization
  -> corpus construction (title + genre + description)
  -> TF-IDF vectorization
  -> SVD + normalization (LSA space)
  -> query projection
  -> cosine similarity ranking
  -> tabular + visual outputs
```

## 3. Data Source and Normalization

### 3.1 Source

- Dataset provider: Kaggle
- Dataset: chenyanglim/imdb-v2
- File loading supports csv, tsv, or parquet

### 3.2 Column Mapping Strategy

The notebook detects available columns from candidate names and maps them into a normalized schema:

- id
- title
- genre
- description
- year
- rating

If a preferred column is unavailable, the pipeline falls back to alternatives. If title is missing, execution fails fast with a clear error.

### 3.3 Field Parsing Rules

- Genre values are split by common delimiters (| , / ;).
- Year values are parsed from text using 4-digit year matching.
- Rating values are converted to float where possible.
- Empty descriptions fall back to the movie title.

The resulting normalized dataframe is converted to a list of movie records for search.

## 4. Core Retrieval Pipeline

### 4.1 Corpus Construction

Each movie document is built as:

```text
title + genre tokens + description
```

The combined text is lowercased and collected into a corpus list.

### 4.2 TF-IDF

TF-IDF converts corpus text into a sparse matrix where:

- rows represent movies
- columns represent vocabulary terms
- values represent term importance per movie

Configuration used:

- TfidfVectorizer(stop_words="english")

### 4.3 SVD / LSA Projection

The notebook fits a TruncatedSVD model and a Normalizer in a pipeline.

Component count is bounded by matrix limits to avoid invalid dimensions:

```text
max_components = min(n_samples - 1, n_features - 1)
n_components = min(requested_components, max_components)
```

This projects TF-IDF vectors into a lower-dimensional latent semantic space.

### 4.4 Query Search

For a user query:

1. Transform query into TF-IDF space
2. Project query to LSA space
3. Compute cosine similarity against movie vectors
4. Rank descending and return top-k results

Default output fields:

- title
- genre
- year
- rating
- score

## 5. Visual and Diagnostic Outputs

The notebook includes plots to inspect data and model behavior:

- Year and rating distributions
- Genre frequency
- Top terms by average TF-IDF weight
- Cumulative SVD explained variance
- 2D latent semantic scatter sample
- Query result similarity bar chart

It also includes a multi-query comparison section to quickly evaluate search quality across different query intents.

## 6. Mathematical Basis

LSA is applied through truncated matrix factorization:

$$
A \approx U\Sigma V^T
$$

Similarity scoring uses cosine similarity:

$$
s(x, y) = \frac{x \cdot y}{\|x\|\|y\|}
$$

## 7. Strengths

- Captures semantic relatedness beyond exact keyword match
- Reduces sparse noise through latent projection
- Fast ranking with vector operations
- Transparent workflow with visual diagnostics

## 8. Current Limitations

- Bag-of-words + LSA is less context-aware than transformer embeddings
- Retrieval quality depends on metadata and description quality
- Pipeline is recomputed in notebook sessions (no persisted artifacts yet)

## 9. Recommended Next Steps

1. Persist trained artifacts (TF-IDF model, SVD pipeline, latent matrix).
2. Add pre-ranking filters (genre/year/rating).
3. Introduce hybrid ranking with rating-aware weighting.
4. Expose notebook logic as an API for interactive applications.
5. Evaluate modern embedding-based retrieval as a future baseline comparison.
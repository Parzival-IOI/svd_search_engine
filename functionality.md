# SVD-Based Movie Search Engine - Functionality Documentation

## 1. Overview

This project implements semantic movie retrieval in a notebook workflow using:

- TF-IDF for term weighting
- Truncated SVD for latent semantic projection (LSA)
- Cosine similarity for ranking

The implementation runs end-to-end in `main.ipynb` and is backed by a fixed Wikipedia movie plots dataset sourced from Kaggle.

## 2. End-to-End Architecture

```text
Kaggle jrobischon/wikipedia-movie-plots (fetched once, cached to data/)
  -> load and normalize (title + genre + plot + year)
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
- Dataset: `jrobischon/wikipedia-movie-plots`
- ~34,886 movies with Wikipedia plot summaries
- Fetched once via `kagglehub` and cached to `data/jrobischon_wikipedia-movie-plots.csv`
- Subsequent runs use the local cache with no network access

### 3.2 Column Mapping

The dataset has fixed columns mapped directly:

| Normalized field | Source column  |
|-----------------|----------------|
| title           | Title          |
| genre           | Genre          |
| description     | Plot           |
| year            | Release Year   |

There is no rating column in this dataset.

### 3.3 Field Parsing Rules

- Genre values are split by common delimiters (`| , / ;`).
- Year values are parsed from text using 4-digit year matching.
- Empty descriptions fall back to the movie title.

The resulting normalized dataframe is converted to a corpus for the retrieval pipeline.

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

- `TfidfVectorizer(stop_words="english")`

### 4.3 SVD / LSA Projection

The pipeline fits a `TruncatedSVD` model and a `Normalizer` in a sklearn `make_pipeline`.

Component count is bounded by matrix limits:

```text
max_components = min(n_samples - 1, n_features - 1)
n_components = min(requested_components, max_components)
```

This projects TF-IDF vectors into a lower-dimensional latent semantic space.

### 4.4 Query Search

For a user query:

1. Transform query into TF-IDF space
2. Project query to LSA space
3. Compute cosine similarity against all movie vectors
4. Apply optional genre / year filters
5. Rank descending and return top-k results

Default output fields:

- title
- genre
- year
- similarity

## 5. Visual and Diagnostic Outputs

The notebook includes plots to inspect data and model behavior:

- Release year distribution
- Genre frequency (top 20)
- Top terms by average TF-IDF weight
- Cumulative SVD explained variance
- 2D latent semantic scatter sample
- Query result similarity bar chart

It also includes a multi-query comparison section and term-level explanation of search results.

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
- Wikipedia plot descriptions provide rich narrative context

## 8. Current Limitations

- Bag-of-words + LSA is less context-aware than transformer embeddings
- Retrieval quality depends on description quality
- No rating data available for this dataset

## 9. Recommended Next Steps

1. Add a FastAPI backend for programmatic search.
2. Add model versioning and dataset fingerprinting in saved artifacts.
3. Evaluate modern embedding-based retrieval as a future baseline comparison.
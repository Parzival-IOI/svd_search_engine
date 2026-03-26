# 🎬 SVD-Based Movie Search Engine — Functionality Documentation

## 1. Overview

This search engine is designed to retrieve relevant movies based on a user’s query using **Latent Semantic Analysis (LSA)** powered by **Singular Value Decomposition (SVD)**.

Unlike traditional keyword matching, this system understands **semantic relationships** between words, allowing it to return relevant results even when exact terms do not match.

---

## 2. System Architecture

The system consists of the following main components:

```
Dataset → Text Processing → TF-IDF → SVD (LSA) → Query Processing → Similarity Ranking → Results
```

---

## 3. Dataset Structure

Each movie document contains:

```json
{
  "id": 1,
  "title": "Movie Title",
  "genre": ["Action", "Sci-Fi"],
  "description": "Short description of the movie",
  "year": 2020,
  "rating": 7.5
}
```

---

## 4. Core Components

### 4.1 Text Preprocessing

Each movie is converted into a single text document:

```
combined_text = title + genre + description
```

Steps:

* Convert to lowercase
* Remove stopwords (e.g., "the", "is", "and")
* Tokenization handled by TF-IDF

---

### 4.2 TF-IDF Vectorization

TF-IDF (Term Frequency – Inverse Document Frequency) converts text into a numerical matrix.

* Rows → Movies (documents)
* Columns → Words (features)
* Values → Importance of words in each document

Result:

```
Term-Document Matrix (Sparse Matrix)
```

---

### 4.3 Dimensionality Reduction using SVD

SVD decomposes the TF-IDF matrix:

```
A ≈ U Σ Vᵀ
```

Where:

* **U** → document-topic matrix
* **Σ** → importance of each latent topic
* **Vᵀ** → term-topic relationships

Purpose:

* Reduce noise
* Capture hidden semantic structure
* Map documents into a lower-dimensional space

This step transforms:

```
High-dimensional word space → Low-dimensional semantic space
```

---

### 4.4 Latent Semantic Analysis (LSA)

After applying SVD:

* Words with similar meanings are grouped together
* Example:

  * "space", "alien", "galaxy" → similar semantic direction

This allows the system to:

* Understand context
* Match related concepts instead of exact words

---

### 4.5 Query Processing

When a user inputs a query:

Example:

```
"space mission alien"
```

Steps:

1. Apply same TF-IDF transformation
2. Project query into SVD space
3. Represent query as a vector

---

### 4.6 Similarity Computation

The system computes similarity between:

```
Query vector ↔ Movie vectors
```

Using:

```
Cosine Similarity
```

Formula:

```
similarity = (A · B) / (||A|| × ||B||)
```

Range:

* 1 → identical
* 0 → unrelated

---

### 4.7 Ranking & Retrieval

Steps:

1. Compute similarity scores for all movies
2. Sort in descending order
3. Return top-k results

Output example:

```
1. Movie A — score: 0.89
2. Movie B — score: 0.85
3. Movie C — score: 0.81
```

---

## 5. Search Flow (End-to-End)

```
User Query
   ↓
TF-IDF Transformation
   ↓
SVD Projection
   ↓
Cosine Similarity
   ↓
Ranking
   ↓
Top Results Displayed
```

---

## 6. Key Advantages

### ✅ Semantic Understanding

* Finds related content even without exact keyword match

### ✅ Noise Reduction

* Removes less important words via dimensionality reduction

### ✅ Efficient Search

* Lower-dimensional vectors improve performance

---

## 7. Limitations

### ❌ No Deep Context Understanding

* Cannot fully understand complex language like modern NLP models

### ❌ Dependent on Dataset Quality

* Poor descriptions → weaker results

### ❌ Static Model

* Does not learn dynamically unless retrained

---

## 8. Possible Improvements

### 🔹 Weighted Ranking

Combine similarity with rating:

```
final_score = 0.8 * similarity + 0.2 * rating
```

---

### 🔹 Query Expansion

Add related terms automatically:

* "space" → "galaxy", "alien"

---

### 🔹 Filtering

* By genre
* By year
* By rating

---

### 🔹 Web Interface

* Build UI using Flask or FastAPI
* Add search bar and result cards

---

### 🔹 Model Persistence

* Save trained model using `joblib`
* Avoid recomputation

---

## 9. Conclusion

This system demonstrates how **SVD can be applied in real-world search engines** to:

* Extract hidden relationships in text
* Improve search relevance
* Move beyond keyword-based retrieval

It serves as a foundational approach to modern semantic search systems.

---

## 10. Future Scope

* Replace SVD with advanced embeddings (Word2Vec, BERT)
* Integrate real-time user feedback
* Deploy as a scalable web service

---
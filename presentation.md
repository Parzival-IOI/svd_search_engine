# SVD-Based Movie Search Engine — Presentation

## Agenda

---

### 1. Introduction (2 min)
- What is semantic search? Why keyword matching fails.
- Project goal: find movies by *meaning*, not exact words.
- Tools used: Python, TF-IDF, SVD, Cosine Similarity, Streamlit.

---

### 2. Dataset (2 min)
- Source: Kaggle IMDb-v2 (~8,000+ real movies)
- Schema: title, genre, description, year, rating
- Column normalization and missing-value handling

---

### 3. Text Representation — TF-IDF (3 min)

Each movie becomes a **vector of word weights**:

$$\text{TF-IDF}(t, d) = \underbrace{\frac{\text{count of } t \text{ in } d}{\text{total words in } d}}_{\text{TF}} \times \underbrace{\log\frac{N}{\text{docs containing } t}}_{\text{IDF}}$$

- Common words ("the", "a") → low weight
- Rare but meaningful words ("galaxy", "heist") → high weight
- Produces a sparse matrix: **8000 movies × 30000+ words**

---

### 4. The Core — SVD / Latent Semantic Analysis (8 min)

#### 4.1 The Problem with TF-IDF alone
"galaxy" and "space" are unrelated in TF-IDF space even though they mean similar things.  
SVD finds the hidden (*latent*) topics that connect them.

#### 4.2 SVD in Simple Math

Given the TF-IDF matrix $A$ (movies × words):

$$A = U \Sigma V^T$$

| Symbol | Shape | Meaning |
|---|---|---|
| $A$ | $m \times n$ | Original movie–word matrix |
| $U$ | $m \times k$ | Each movie in **topic space** |
| $\Sigma$ | $k \times k$ | Diagonal — **importance** of each topic |
| $V^T$ | $k \times n$ | Each word's role in each **topic** |

We keep only the top $k = 100$ topics (truncation):

$$A \approx U_k \Sigma_k V_k^T$$

This is the **compression step** — 30000 word dimensions → 100 topic dimensions.

#### 4.3 Concrete Example

Imagine 3 topics after SVD:

| Topic | Words with high weight |
|---|---|
| Topic 1 | space, galaxy, alien, rocket, orbit |
| Topic 2 | love, romance, heart, wedding |
| Topic 3 | murder, detective, crime, clue |

A movie about astronauts scores high on Topic 1.  
A query "space adventure" also scores high on Topic 1 → **they match**, even with no shared exact words.

#### 4.4 Normalizer
After SVD, each vector is normalized to unit length so comparisons are fair regardless of movie description length.

---

### 5. Search — Cosine Similarity (3 min)

A user query is passed through the **same** TF-IDF + SVD pipeline:

$$\vec{q} = \text{Normalizer}(\text{SVD}(\text{TF-IDF}(\text{query})))$$

Then ranked against all movie vectors:

$$\text{similarity}(\vec{q}, \vec{m}) = \frac{\vec{q} \cdot \vec{m}}{\|\vec{q}\| \|\vec{m}\|} \in [0, 1]$$

- 1 = identical direction in topic space
- 0 = completely unrelated

**Weighted score** used in the app:

$$\text{score} = \alpha \cdot \text{similarity} + (1 - \alpha) \cdot \text{rating}_{\text{normalized}}$$

Default $\alpha = 0.85$ — semantic match matters more, rating breaks ties.

---

### 6. Demo — Streamlit App (5 min)

**Page 1 — Process Notebook View**
- Walk through each pipeline stage live
- Show charts: year/rating distributions, genre frequency, top TF-IDF terms, explained variance curve, 2D latent scatter

**Page 2 — Real-Time SVD Demo**
- Live query: `"space mission"`, `"mysterious detective case"`
- Show filters: genre, year, minimum rating
- Adjust alpha slider to show similarity vs. rating trade-off

---

### 7. Results + Analysis (2 min)
- Explained variance: 100 components capture ~X% of variance
- Semantic queries return contextually relevant movies even without exact title/word match
- Genre scatter in 2D shows visible topic clustering

---

### 8. Limitations + Next Steps (2 min)

| Limitation | Potential Fix |
|---|---|
| Bag-of-words, no word order | Word2Vec / BERT embeddings |
| Static model, no online update | Incremental SVD or re-training trigger |
| No user feedback signal | Relevance feedback / click-through |

---

### Suggested Slide Structure

```
Slide 1  — Title + Team
Slide 2  — Problem: Why not keyword search?
Slide 3  — Dataset & normalization
Slide 4  — TF-IDF explained
Slide 5  — SVD math (with the U Σ Vᵀ diagram)
Slide 6  — Topics example table
Slide 7  — Cosine similarity + weighted score formula
Slide 8  — Live demo (switch to Streamlit)
Slide 9  — Charts: explained variance + scatter
Slide 10 — Limitations + next steps
Slide 11 — Q&A
```

**Total estimated time: ~20–25 min with demo**

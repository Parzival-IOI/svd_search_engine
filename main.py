import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# 1. Load Dataset
# -----------------------------
def load_movies(path="movies_dataset.json"):
    with open(path, "r") as f:
        return json.load(f)

# -----------------------------
# 2. Preprocess Text
# -----------------------------
def build_corpus(movies):
    corpus = []
    for m in movies:
        text = (
            m["title"] + " " +
            " ".join(m["genre"]) + " " +
            m["description"]
        )
        corpus.append(text.lower())
    return corpus

# -----------------------------
# 3. Build SVD Model
# -----------------------------
def build_model(corpus, n_components=100):
    tfidf = TfidfVectorizer(stop_words="english")

    X_tfidf = tfidf.fit_transform(corpus)

    # Fix: ensure n_components is valid
    max_components = min(X_tfidf.shape[0] - 1, X_tfidf.shape[1] - 1)
    n_components = min(n_components, max_components)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)

    X_lsa = lsa.fit_transform(X_tfidf)

    return tfidf, lsa, X_lsa

# -----------------------------
# 4. Search Function
# -----------------------------
def search(query, tfidf, lsa, X_lsa, movies, top_k=5):
    query_vec = tfidf.transform([query])
    query_lsa = lsa.transform(query_vec)

    similarities = cosine_similarity(query_lsa, X_lsa)[0]
    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        movie = movies[idx]
        results.append({
            "title": movie["title"],
            "genre": movie["genre"],
            "year": movie["year"],
            "rating": movie["rating"],
            "score": float(similarities[idx])
        })

    return results

# -----------------------------
# 5. CLI Interface
# -----------------------------
def main():
    print("Loading dataset...")
    movies = load_movies()

    print("Building model...")
    corpus = build_corpus(movies)
    tfidf, lsa, X_lsa = build_model(corpus)

    print("Search Engine Ready! (type 'exit' to quit)\n")

    while True:
        query = input("Search: ")
        if query.lower() == "exit":
            break

        results = search(query, tfidf, lsa, X_lsa, movies)

        print("\nTop Results:")
        for r in results:
            print(f"- {r['title']} ({r['year']}) | {r['genre']} | ⭐ {r['rating']} | score={r['score']:.3f}")
        print()

if __name__ == "__main__":
    main()
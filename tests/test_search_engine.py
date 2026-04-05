"""
Comprehensive evaluation test suite for the SVD-based search engine.

Tests cover:
  - Model artifact integrity (fields, types, shapes)
  - Consistency (same query → identical results)
  - Similarity range validity
  - Top-k contract (at most k results, no duplicates)
  - Genre / year filtering
  - Edge cases (empty query, whitespace-only, special chars)
  - Semantic search quality (result titles are thematically relevant)
  - Title-exact-match (search a movie's own name → it must appear in results)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure src/ is on the path so the package resolves without installing
_src = Path(__file__).resolve().parents[1] / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from svd_search.config.paths import ARTIFACT_V2
from svd_search.models.predict import load_artifacts, search_movies
from svd_search.utils.utils import PipelineArtifacts

# ---------------------------------------------------------------------------
# Fixture: load once for the whole session (expensive – ~1-2 s)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def arts() -> PipelineArtifacts:
    if not ARTIFACT_V2.exists():
        pytest.skip(f"Artifact not found at {ARTIFACT_V2}. Build it first.")
    return load_artifacts(ARTIFACT_V2)


# ===========================================================================
# 1. ARTIFACT INTEGRITY
# ===========================================================================

class TestArtifactIntegrity:
    def test_artifact_has_required_fields(self, arts):
        assert arts.tfidf   is not None, "tfidf vectoriser missing"
        assert arts.lsa     is not None, "lsa pipeline missing"
        assert arts.svd     is not None, "svd component missing"
        assert arts.X_lsa   is not None, "X_lsa matrix missing"
        assert arts.df      is not None, "DataFrame missing"
        assert arts.corpus  is not None, "corpus missing"

    def test_df_has_required_columns(self, arts):
        required = {"title", "year", "genre"}
        missing = required - set(arts.df.columns)
        assert not missing, f"DataFrame missing columns: {missing}"

    def test_description_column_exists(self, arts):
        has_desc = "description" in arts.df.columns or "plot" in arts.df.columns
        assert has_desc, "DataFrame must have 'description' or 'plot' column"

    def test_corpus_aligned_with_df(self, arts):
        assert len(arts.corpus) == len(arts.df), (
            f"Corpus length {len(arts.corpus)} != DataFrame rows {len(arts.df)}"
        )

    def test_X_lsa_aligned_with_df(self, arts):
        assert arts.X_lsa.shape[0] == len(arts.df), (
            f"X_lsa row count {arts.X_lsa.shape[0]} != DataFrame rows {len(arts.df)}"
        )

    def test_X_lsa_is_2d(self, arts):
        assert arts.X_lsa.ndim == 2, "X_lsa should be a 2-D array"

    def test_svd_components_match_X_lsa(self, arts):
        assert arts.svd.n_components == arts.X_lsa.shape[1], (
            "SVD n_components does not match X_lsa column count"
        )

    def test_df_not_empty(self, arts):
        assert len(arts.df) > 0, "DataFrame is empty – no movies loaded"

    def test_corpus_strings_nonempty(self, arts):
        empty_count = sum(1 for doc in arts.corpus if not doc.strip())
        pct = empty_count / len(arts.corpus) * 100
        assert pct < 5, f"{pct:.1f}% of corpus documents are empty (threshold 5%)"

    def test_version_field_present(self, arts):
        assert arts.version, "PipelineArtifacts.version should not be empty"


# ===========================================================================
# 2. CONSISTENCY
# ===========================================================================

class TestConsistency:
    QUERY = "science fiction space adventure"

    def test_same_query_returns_same_titles(self, arts):
        r1 = search_movies(self.QUERY, arts, top_k=10)
        r2 = search_movies(self.QUERY, arts, top_k=10)
        assert list(r1["title"]) == list(r2["title"]), (
            "Identical queries returned different title orderings"
        )

    def test_same_query_returns_same_similarities(self, arts):
        r1 = search_movies(self.QUERY, arts, top_k=10)
        r2 = search_movies(self.QUERY, arts, top_k=10)
        np.testing.assert_allclose(
            r1["similarity"].values,
            r2["similarity"].values,
            rtol=1e-6,
            err_msg="Similarity scores differ between identical queries",
        )

    def test_results_are_sorted_descending(self, arts):
        results = search_movies(self.QUERY, arts, top_k=20)
        sims = results["similarity"].values
        assert all(sims[i] >= sims[i + 1] for i in range(len(sims) - 1)), (
            "Results are not sorted by similarity in descending order"
        )


# ===========================================================================
# 3. SIMILARITY RANGE
# ===========================================================================

class TestSimilarityRange:
    def test_similarities_between_neg1_and_1(self, arts):
        results = search_movies("drama love war", arts, top_k=20)
        if results.empty:
            pytest.skip("No results for test query")
        assert results["similarity"].between(-1.0, 1.0).all(), (
            "Similarity scores outside the valid cosine range [-1, 1]"
        )

    def test_top_result_similarity_above_zero(self, arts):
        """A reasonable semantic query should yield at least one positive match."""
        results = search_movies("war soldier battlefield", arts, top_k=5)
        if results.empty:
            pytest.skip("No results returned")
        assert results["similarity"].iloc[0] > 0, (
            "Top result has non-positive similarity for a clear semantic query"
        )


# ===========================================================================
# 4. TOP-K CONTRACT
# ===========================================================================

class TestTopKContract:
    @pytest.mark.parametrize("k", [1, 5, 10, 20])
    def test_returns_at_most_k_results(self, arts, k):
        results = search_movies("adventure hero quest", arts, top_k=k)
        assert len(results) <= k, f"Returned {len(results)} > top_k={k}"

    def test_no_duplicate_titles_in_results(self, arts):
        results = search_movies("romantic comedy", arts, top_k=20)
        titles = results["title"].tolist()
        assert len(titles) == len(set(titles)), (
            f"Duplicate titles in results: {[t for t in titles if titles.count(t) > 1]}"
        )

    def test_returns_dataframe(self, arts):
        results = search_movies("action movie", arts, top_k=5)
        assert isinstance(results, pd.DataFrame), "search_movies must return a DataFrame"

    def test_result_has_required_columns(self, arts):
        results = search_movies("action movie", arts, top_k=5)
        if results.empty:
            pytest.skip("No results to check columns on")
        for col in ["title", "genre", "year", "description", "similarity"]:
            assert col in results.columns, f"Missing result column: {col}"


# ===========================================================================
# 5. EDGE CASES
# ===========================================================================

class TestEdgeCases:
    def test_empty_string_returns_empty(self, arts):
        results = search_movies("", arts)
        assert results.empty, "Empty query should return an empty DataFrame"

    def test_whitespace_only_returns_empty(self, arts):
        results = search_movies("   \t\n  ", arts)
        assert results.empty, "Whitespace-only query should return an empty DataFrame"

    def test_special_characters_no_crash(self, arts):
        try:
            results = search_movies("!@#$%^&*()", arts, top_k=5)
            assert isinstance(results, pd.DataFrame)
        except Exception as exc:
            pytest.fail(f"Special character query raised an exception: {exc}")

    def test_very_long_query_no_crash(self, arts):
        long_q = "movie " * 200
        try:
            results = search_movies(long_q, arts, top_k=5)
            assert isinstance(results, pd.DataFrame)
        except Exception as exc:
            pytest.fail(f"Very long query raised an exception: {exc}")

    def test_unicode_query_no_crash(self, arts):
        try:
            results = search_movies("café naïve résumé", arts, top_k=5)
            assert isinstance(results, pd.DataFrame)
        except Exception as exc:
            pytest.fail(f"Unicode query raised an exception: {exc}")

    def test_top_k_zero_returns_empty(self, arts):
        results = search_movies("action", arts, top_k=0)
        assert results.empty or len(results) == 0, "top_k=0 should yield 0 results"


# ===========================================================================
# 6. GENRE FILTER
# ===========================================================================

class TestGenreFilter:
    def _get_available_genre(self, arts) -> str:
        """Pick a genre that actually exists in the data."""
        for _, row in arts.df.iterrows():
            genres = row["genre"] if isinstance(row["genre"], list) else [str(row["genre"])]
            for g in genres:
                if g and g != "Unknown":
                    return g
        pytest.skip("No named genres found in dataset")

    def test_genre_filter_restricts_results(self, arts):
        genre = self._get_available_genre(arts)
        results = search_movies("love story", arts, top_k=20, genre_filter=genre)
        if results.empty:
            pytest.skip(f"No results for genre_filter='{genre}'")
        for _, row in results.iterrows():
            genre_val = row["genre"]
            assert genre in genre_val, (
                f"Genre filter '{genre}' violated: row has genre '{genre_val}'"
            )

    def test_all_genre_filter_returns_results(self, arts):
        results = search_movies("action adventure", arts, top_k=10, genre_filter="All")
        # "All" should not filter anything out
        results_no_filter = search_movies("action adventure", arts, top_k=10)
        assert len(results) == len(results_no_filter), (
            "genre_filter='All' should behave like no filter"
        )

    def test_nonexistent_genre_returns_empty(self, arts):
        results = search_movies("movie", arts, top_k=10, genre_filter="ZZZ_Nonexistent_Genre")
        assert results.empty, "Non-existent genre should return an empty DataFrame"


# ===========================================================================
# 7. YEAR FILTER
# ===========================================================================

class TestYearFilter:
    def test_year_min_filter(self, arts):
        min_year = 2000
        results = search_movies("adventure", arts, top_k=20, year_min=min_year)
        if results.empty:
            pytest.skip("No results for year_min filter")
        valid = results["year"].dropna()
        assert (valid >= min_year).all(), (
            f"year_min={min_year} violated: found years {valid[valid < min_year].tolist()}"
        )

    def test_year_max_filter(self, arts):
        max_year = 1990
        results = search_movies("crime thriller", arts, top_k=20, year_max=max_year)
        if results.empty:
            pytest.skip("No results for year_max filter")
        valid = results["year"].dropna()
        assert (valid <= max_year).all(), (
            f"year_max={max_year} violated: found years {valid[valid > max_year].tolist()}"
        )

    def test_year_range_filter(self, arts):
        results = search_movies("comedy", arts, top_k=30, year_min=1990, year_max=2000)
        if results.empty:
            pytest.skip("No results for year range filter")
        valid = results["year"].dropna()
        assert (valid >= 1990).all() and (valid <= 2000).all(), (
            "Year range filter violated"
        )

    def test_impossible_year_range_returns_empty(self, arts):
        # min > max → inherently contradictory
        results = search_movies("movie", arts, top_k=10, year_min=2100, year_max=1900)
        assert results.empty, "Impossible year range should return empty results"


# ===========================================================================
# 8. SEMANTIC QUALITY
# ===========================================================================

class TestSemanticQuality:
    """
    Grade the search engine's ability to surface thematically relevant movies.
    These tests use keyword overlap as a lightweight proxy for relevance.
    """

    SEMANTIC_CASES = [
        ("space alien planet astronaut", ["space", "alien", "planet", "asteroid", "galaxy", "universe", "mars", "star", "orbit"]),
        ("vampire blood horror night", ["vampire", "blood", "horror", "zombie", "creature", "monster", "dark", "dracula"]),
        ("detective murder mystery crime", ["detective", "murder", "crime", "mystery", "police", "killer", "investigation"]),
        ("romantic love couple wedding", ["love", "romance", "wedding", "couple", "marry", "marriage", "heart"]),
        ("war soldier battle military", ["war", "soldier", "battle", "army", "military", "combat", "weapon", "enemy"]),
    ]

    def _description_contains_keyword(self, description: str, keywords: list) -> bool:
        desc_lower = description.lower()
        return any(kw in desc_lower for kw in keywords)

    @pytest.mark.parametrize("query,keywords", SEMANTIC_CASES)
    def test_semantic_precision_at_5(self, arts, query, keywords):
        """At least 60% of top-5 results should mention a related keyword."""
        results = search_movies(query, arts, top_k=5)
        if results.empty:
            pytest.skip(f"No results for query: {query!r}")

        relevant = sum(
            self._description_contains_keyword(row["description"], keywords)
            for _, row in results.iterrows()
        )
        precision = relevant / len(results)
        assert precision >= 0.60, (
            f"Precision@5 for query {query!r} = {precision:.0%} < 60%\n"
            f"  Keywords: {keywords}\n"
            f"  Titles returned: {results['title'].tolist()}"
        )

    @pytest.mark.parametrize("query,keywords", SEMANTIC_CASES)
    def test_semantic_precision_at_10(self, arts, query, keywords):
        """At least 50% of top-10 results should mention a related keyword."""
        results = search_movies(query, arts, top_k=10)
        if results.empty:
            pytest.skip(f"No results for query: {query!r}")

        relevant = sum(
            self._description_contains_keyword(row["description"], keywords)
            for _, row in results.iterrows()
        )
        precision = relevant / len(results)
        assert precision >= 0.50, (
            f"Precision@10 for query {query!r} = {precision:.0%} < 50%\n"
            f"  Keywords: {keywords}\n"
            f"  Titles returned: {results['title'].tolist()}"
        )


# ===========================================================================
# 9. TITLE EXACT-MATCH (the reported bug)
# ===========================================================================

class TestTitleExactMatch:
    """
    When a user searches for a movie's exact title, the search engine should
    return that movie somewhere in the top-10 results.

    Root cause of the failure: TF-IDF min_df=3 removes unique title tokens from
    the vocabulary, so rare title words carry zero weight at query time.
    The fix applied in predict.py adds an explicit title-match boost.
    """

    # Sample of real movie titles from each of the supported datasets.
    # These are well-known films that should be findable.
    KNOWN_TITLES = [
        "The Godfather",
        "Titanic",
        "The Dark Knight",
        "Inception",
        "Pulp Fiction",
        "Forrest Gump",
        "The Matrix",
        "Schindler's List",
        "Goodfellas",
        "Braveheart",
    ]

    def _find_exact_title(self, arts, title: str) -> bool:
        """Return True if *title* exists in the loaded DataFrame (case-insensitive)."""
        return arts.df["title"].str.lower().eq(title.lower()).any()

    @pytest.mark.parametrize("title", KNOWN_TITLES)
    def test_title_search_returns_exact_match_in_top10(self, arts, title):
        if not self._find_exact_title(arts, title):
            pytest.skip(f"'{title}' not in dataset – skipping")

        results = search_movies(title, arts, top_k=10)
        assert not results.empty, f"No results returned for title query: '{title}'"

        matched = results["title"].str.lower().eq(title.lower()).any()
        assert matched, (
            f"Title search for '{title}' did not return the movie itself in top-10.\n"
            f"  Got: {results['title'].tolist()}\n"
            "\n"
            "  BUG: min_df=3 in TF-IDF silently drops unique title tokens.\n"
            "  FIX: apply title-boost in search_movies() – see predict.py."
        )

    @pytest.mark.parametrize("title", KNOWN_TITLES)
    def test_title_search_returns_exact_match_at_rank1(self, arts, title):
        """Stricter: the exact title should be the top-ranked result."""
        if not self._find_exact_title(arts, title):
            pytest.skip(f"'{title}' not in dataset – skipping")

        results = search_movies(title, arts, top_k=10)
        if results.empty:
            pytest.skip("No results returned")

        top_title = results.iloc[0]["title"].lower()
        assert top_title == title.lower(), (
            f"Title '{title}' is not ranked #1.\n"
            f"  Rank-1 result: '{results.iloc[0]['title']}'\n"
            f"  Full top-10:   {results['title'].tolist()}"
        )


# ===========================================================================
# 10. MRR – Mean Reciprocal Rank (summary metric)
# ===========================================================================

class TestMRR:
    """
    Mean Reciprocal Rank measures how high the expected result ranks.
    Computed over a set of title-search queries where the ground truth
    is the exact movie with that title.
    """

    SAMPLE_TITLES = [
        "Titanic", "Inception", "The Matrix", "Braveheart",
        "Goodfellas", "Forrest Gump", "Pulp Fiction",
    ]

    def test_mrr_above_threshold(self, arts):
        reciprocal_ranks = []
        for title in self.SAMPLE_TITLES:
            if not arts.df["title"].str.lower().eq(title.lower()).any():
                continue  # skip if not in dataset
            results = search_movies(title, arts, top_k=10)
            rank = None
            for i, (_, row) in enumerate(results.iterrows(), start=1):
                if row["title"].lower() == title.lower():
                    rank = i
                    break
            reciprocal_ranks.append(1 / rank if rank else 0)

        if not reciprocal_ranks:
            pytest.skip("None of the sample titles found in dataset")

        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        print(f"\n  MRR over {len(reciprocal_ranks)} title queries = {mrr:.3f}")
        assert mrr >= 0.50, (
            f"MRR = {mrr:.3f} is below the 0.50 threshold.\n"
            "  This indicates the title-match bug is active or performance degraded."
        )

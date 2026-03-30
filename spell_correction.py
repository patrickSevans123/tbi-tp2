"""
BONUS: Query Spell Correction

Implements spell correction for search queries using:

1. Levenshtein Edit Distance
   - Dynamic programming O(m*n) algorithm
   - Supports insert, delete, substitute operations

2. Character N-gram Index
   - Pre-builds an index of character bigrams/trigrams for all vocabulary terms
   - For a misspelled word, generate its n-grams, find candidate terms
     sharing the most n-grams (Jaccard similarity), then compute edit distance
     only on those candidates
   - Much faster than computing edit distance against entire vocabulary

3. BK-Tree (Burkhard-Keller Tree)
   - Metric tree for efficient nearest-neighbor search in edit-distance space
   - O(log V) average lookup vs O(V) brute force

All implementations use only Python standard library.
"""

import re

from index import InvertedIndexReader


def levenshtein_distance(s1, s2):
    """
    Compute Levenshtein (edit) distance between two strings.

    Uses dynamic programming. Operations: insert, delete, substitute
    (each costs 1).

    Parameters
    ----------
    s1, s2 : str
        Strings to compare

    Returns
    -------
    int
        Edit distance
    """
    m, n = len(s1), len(s2)
    # Optimize: use only two rows
    prev = list(range(n + 1))
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,       # deletion
                curr[j - 1] + 1,   # insertion
                prev[j - 1] + cost  # substitution
            )
        prev, curr = curr, prev

    return prev[n]


class CharNgramIndex:
    """
    Character n-gram index for fast candidate retrieval.

    For each term in the vocabulary, generates character n-grams
    (default: bigrams) and builds an inverted index from n-grams
    to terms. This allows quick filtering of candidate corrections
    by n-gram overlap before computing expensive edit distances.

    Example:
        "algorithm" -> bigrams: {"$a", "al", "lg", "go", "or", "ri", "it", "th", "hm", "m$"}
        For misspelled "algoritm", generate its bigrams and find terms
        with high Jaccard similarity in bigram sets.
    """

    def __init__(self, n=2):
        """
        Parameters
        ----------
        n : int
            Size of character n-grams (default: 2 for bigrams)
        """
        self.n = n
        self.index = {}       # ngram -> set of terms
        self.term_ngrams = {} # term -> set of ngrams
        self.vocabulary = set()

    def _get_ngrams(self, term):
        """Generate character n-grams with boundary markers."""
        padded = '$' * (self.n - 1) + term + '$' * (self.n - 1)
        return set(padded[i:i + self.n] for i in range(len(padded) - self.n + 1))

    def build(self, vocabulary):
        """
        Build the n-gram index from a vocabulary.

        Parameters
        ----------
        vocabulary : iterable of str
            All terms in the collection
        """
        self.vocabulary = set(vocabulary)
        self.index = {}
        self.term_ngrams = {}

        for term in self.vocabulary:
            ngrams = self._get_ngrams(term)
            self.term_ngrams[term] = ngrams
            for ng in ngrams:
                if ng not in self.index:
                    self.index[ng] = set()
                self.index[ng].add(term)

    def find_candidates(self, word, threshold=0.2, max_candidates=50):
        """
        Find candidate corrections using n-gram overlap.

        Uses Jaccard similarity of n-gram sets to filter candidates.

        Parameters
        ----------
        word : str
            Misspelled word
        threshold : float
            Minimum Jaccard similarity threshold
        max_candidates : int
            Maximum number of candidates to return

        Returns
        -------
        list[tuple[float, str]]
            Candidates sorted by Jaccard similarity descending:
            [(similarity, term), ...]
        """
        word_ngrams = self._get_ngrams(word)
        # Collect candidate terms that share at least one n-gram
        candidate_overlap = {}
        for ng in word_ngrams:
            if ng in self.index:
                for term in self.index[ng]:
                    candidate_overlap[term] = candidate_overlap.get(term, 0) + 1

        # Compute Jaccard similarity
        candidates = []
        for term, overlap_count in candidate_overlap.items():
            term_ngrams = self.term_ngrams[term]
            union_size = len(word_ngrams) + len(term_ngrams) - overlap_count
            jaccard = overlap_count / union_size if union_size > 0 else 0
            if jaccard >= threshold:
                candidates.append((jaccard, term))

        candidates.sort(reverse=True)
        return candidates[:max_candidates]


class BKTree:
    """
    BK-Tree (Burkhard-Keller Tree) for efficient nearest-neighbor search
    in edit-distance metric space.

    A BK-tree exploits the triangle inequality property of edit distance.
    For a query with max distance d, only subtrees where the edge distance
    is within [dist-d, dist+d] need to be explored, pruning large portions
    of the search space.

    Average lookup complexity: O(log V) vs O(V) brute force.
    """

    class Node:
        def __init__(self, term):
            self.term = term
            self.children = {}  # distance -> child node

    def __init__(self):
        self.root = None

    def insert(self, term):
        """Insert a term into the BK-tree."""
        if self.root is None:
            self.root = BKTree.Node(term)
            return

        node = self.root
        while True:
            d = levenshtein_distance(term, node.term)
            if d == 0:
                return  # duplicate
            if d not in node.children:
                node.children[d] = BKTree.Node(term)
                return
            node = node.children[d]

    def build(self, vocabulary):
        """Build BK-tree from vocabulary."""
        for term in vocabulary:
            self.insert(term)

    def search(self, word, max_distance=2):
        """
        Find all terms within max_distance of word.

        Parameters
        ----------
        word : str
            Query word
        max_distance : int
            Maximum edit distance

        Returns
        -------
        list[tuple[int, str]]
            Results sorted by distance: [(distance, term), ...]
        """
        if self.root is None:
            return []

        results = []
        stack = [self.root]

        while stack:
            node = stack.pop()
            d = levenshtein_distance(word, node.term)
            if d <= max_distance:
                results.append((d, node.term))

            # Explore children within the triangle inequality range
            for edge_dist, child in node.children.items():
                if d - max_distance <= edge_dist <= d + max_distance:
                    stack.append(child)

        results.sort()
        return results


class SpellCorrector:
    """
    Spell correction system for search queries.

    Combines multiple approaches for robust correction:
    1. Check if term exists in vocabulary (no correction needed)
    2. Use character n-gram index for fast candidate retrieval
    3. Use BK-tree for edit-distance based nearest neighbor search
    4. Rank candidates by edit distance and frequency

    Usage:
        corrector = SpellCorrector()
        corrector.build(vocabulary, term_frequencies)
        corrected_query = corrector.correct("queyr expanson")
        # Returns: "query expansion"
    """

    def __init__(self, max_distance=2, use_bktree=True, use_ngram=True):
        """
        Parameters
        ----------
        max_distance : int
            Maximum edit distance for corrections
        use_bktree : bool
            Use BK-tree for candidate search
        use_ngram : bool
            Use character n-gram index for candidate search
        """
        self.max_distance = max_distance
        self.vocabulary = set()
        self.term_freq = {}

        self.use_bktree = use_bktree
        self.use_ngram = use_ngram

        self.bktree = BKTree() if use_bktree else None
        self.ngram_index = CharNgramIndex(n=2) if use_ngram else None

    def build(self, vocabulary, term_frequencies=None):
        """
        Build the spell correction index.

        Parameters
        ----------
        vocabulary : iterable of str
            All terms in the collection
        term_frequencies : dict[str, int], optional
            Term document frequencies for ranking candidates.
            If None, all terms are treated equally.
        """
        self.vocabulary = set(vocabulary)
        self.term_freq = term_frequencies or {}

        if self.use_bktree:
            print("  Building BK-tree...")
            self.bktree.build(self.vocabulary)

        if self.use_ngram:
            print("  Building n-gram index...")
            self.ngram_index.build(self.vocabulary)

        print(f"  SpellCorrector built with {len(self.vocabulary)} terms")

    def suggest(self, word, max_distance=None, top_n=5):
        """
        Suggest corrections for a possibly misspelled word.

        Parameters
        ----------
        word : str
            Word to correct
        max_distance : int, optional
            Override maximum edit distance
        top_n : int
            Number of suggestions to return

        Returns
        -------
        list[tuple[int, str, int]]
            Suggestions: [(edit_distance, term, frequency), ...]
            sorted by (distance asc, frequency desc)
        """
        if max_distance is None:
            max_distance = self.max_distance

        word = word.lower()

        # If word is in vocabulary, no correction needed
        if word in self.vocabulary:
            return [(0, word, self.term_freq.get(word, 0))]

        candidates = set()

        # Get candidates from BK-tree
        if self.use_bktree and self.bktree.root:
            for dist, term in self.bktree.search(word, max_distance):
                candidates.add((dist, term))

        # Get candidates from n-gram index
        if self.use_ngram:
            for jaccard, term in self.ngram_index.find_candidates(word):
                dist = levenshtein_distance(word, term)
                if dist <= max_distance:
                    candidates.add((dist, term))

        # Rank by (distance, -frequency, term)
        results = []
        for dist, term in candidates:
            freq = self.term_freq.get(term, 0)
            results.append((dist, term, freq))

        results.sort(key=lambda x: (x[0], -x[2], x[1]))
        return results[:top_n]

    def correct(self, query):
        """
        Correct a full query string.

        Each token is checked independently. If a token is not in the
        vocabulary, the best correction is used.

        Parameters
        ----------
        query : str
            Query string to correct

        Returns
        -------
        str
            Corrected query string
        """
        tokens = query.lower().split()
        corrected = []
        corrections_made = []

        for token in tokens:
            if token in self.vocabulary:
                corrected.append(token)
            else:
                suggestions = self.suggest(token, top_n=1)
                if suggestions and suggestions[0][0] <= self.max_distance:
                    corrected_word = suggestions[0][1]
                    corrected.append(corrected_word)
                    if corrected_word != token:
                        corrections_made.append((token, corrected_word))
                else:
                    corrected.append(token)  # Keep original if no good suggestion

        return " ".join(corrected), corrections_made

    def correct_with_info(self, query):
        """
        Correct query and return detailed information.

        Returns
        -------
        tuple[str, list[tuple[str, str, list]]]
            (corrected_query, [(original, corrected, all_suggestions), ...])
        """
        tokens = query.lower().split()
        corrected = []
        info = []

        for token in tokens:
            suggestions = self.suggest(token, top_n=5)
            if suggestions and suggestions[0][0] == 0:
                corrected.append(token)  # Already correct
            elif suggestions and suggestions[0][0] <= self.max_distance:
                best = suggestions[0][1]
                corrected.append(best)
                if best != token:
                    info.append((token, best, suggestions))
            else:
                corrected.append(token)

        return " ".join(corrected), info


def build_spell_corrector_from_index(index_name, postings_encoding, output_dir, term_id_map):
    """
    Build a SpellCorrector from an existing inverted index.

    Parameters
    ----------
    index_name : str
        Name of the inverted index
    postings_encoding : class
        Postings encoding
    output_dir : str
        Index directory
    term_id_map : IdMap
        Term ID mapping

    Returns
    -------
    SpellCorrector
        Ready-to-use spell corrector
    """
    # Collect vocabulary and frequencies
    vocabulary = []
    term_freq = {}

    for i in range(len(term_id_map.id_to_str)):
        term_str = term_id_map.id_to_str[i]
        vocabulary.append(term_str)

    with InvertedIndexReader(index_name, postings_encoding,
                             directory=output_dir) as reader:
        for term_id, entry in reader.postings_dict.items():
            if term_id < len(term_id_map.id_to_str):
                term_str = term_id_map.id_to_str[term_id]
                term_freq[term_str] = entry[1]  # df

    corrector = SpellCorrector(max_distance=2)
    corrector.build(vocabulary, term_freq)
    return corrector


if __name__ == "__main__":
    # Test edit distance
    assert levenshtein_distance("kitten", "sitting") == 3
    assert levenshtein_distance("", "abc") == 3
    assert levenshtein_distance("abc", "abc") == 0
    print("Edit distance tests PASSED")

    # Test n-gram index
    vocab = ["algorithm", "algebra", "alpha", "beta", "gamma",
             "programming", "program", "progress", "protein", "proton"]
    ngram_idx = CharNgramIndex(n=2)
    ngram_idx.build(vocab)

    candidates = ngram_idx.find_candidates("algoritm")
    print(f"\nN-gram candidates for 'algoritm': {candidates[:5]}")

    # Test BK-tree
    bkt = BKTree()
    bkt.build(vocab)
    results = bkt.search("algoritm", max_distance=2)
    print(f"BK-tree results for 'algoritm' (d<=2): {results}")

    # Test SpellCorrector
    corrector = SpellCorrector()
    corrector.build(vocab)
    corrected, corrections = corrector.correct("algoritm programing")
    print(f"\nCorrection: 'algoritm programing' -> '{corrected}'")
    print(f"Corrections made: {corrections}")

    # Test with index
    print("\n--- Testing with index ---")
    from bsbi import BSBIIndex
    from compression import VBEPostings

    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    BSBI_instance.load()

    corrector = build_spell_corrector_from_index(
        'main_index', VBEPostings, 'index', BSBI_instance.term_id_map)

    test_queries = [
        "radioactiv iodoacetae",
        "psycoodrama childrn",
        "lipd metabolsm"
    ]
    for q in test_queries:
        corrected, info = corrector.correct_with_info(q)
        print(f"\n  '{q}' -> '{corrected}'")
        for orig, corr, sugg in info:
            print(f"    {orig} -> {corr} (suggestions: {[(d,t) for d,t,f in sugg[:3]]})")

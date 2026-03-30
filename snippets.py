"""
BONUS: Result Snippets and Term Highlighting

Generates contextual snippets from retrieved documents, showing the
most relevant passage containing the highest density of query terms.

Features:
1. Best Passage Selection
   - Sliding window over document tokens
   - Scores each window by count of distinct query terms present
   - Selects the highest-scoring window

2. Term Highlighting
   - Wraps query terms in markers (** for bold, or ANSI colors)
   - Supports both plain text and terminal-colored output

3. Multi-snippet support
   - Can return multiple non-overlapping snippets from a document
"""


class SnippetGenerator:
    """
    Generates search result snippets from documents.

    For each retrieved document, finds the passage (window) with the
    highest density of query terms and returns it as a snippet with
    optional highlighting.
    """

    def __init__(self, context_words=30, max_snippets=1, highlight_start="**",
                 highlight_end="**"):
        """
        Parameters
        ----------
        context_words : int
            Number of words to include in each snippet window
        max_snippets : int
            Maximum number of snippets to generate per document
        highlight_start : str
            Marker placed before highlighted terms
        highlight_end : str
            Marker placed after highlighted terms
        """
        self.context_words = context_words
        self.max_snippets = max_snippets
        self.highlight_start = highlight_start
        self.highlight_end = highlight_end

    def _score_window(self, tokens, start, end, query_terms_lower):
        """Score a window by the number of distinct query terms present."""
        found = set()
        for i in range(start, min(end, len(tokens))):
            t = tokens[i].lower().strip('.,;:!?()[]{}"\'-')
            if t in query_terms_lower:
                found.add(t)
        return len(found)

    def _find_best_window(self, tokens, query_terms_lower):
        """
        Find the window position with the highest query term density.

        Returns (start, end, score) of the best window.
        """
        n = len(tokens)
        window_size = min(self.context_words, n)

        if n == 0:
            return 0, 0, 0

        best_start = 0
        best_score = -1

        for start in range(max(1, n - window_size + 1)):
            end = start + window_size
            score = self._score_window(tokens, start, end, query_terms_lower)
            if score > best_score:
                best_score = score
                best_start = start

        return best_start, best_start + window_size, best_score

    def _highlight_tokens(self, tokens, query_terms_lower):
        """Highlight query terms in a list of tokens."""
        result = []
        for token in tokens:
            stripped = token.lower().strip('.,;:!?()[]{}"\'-')
            if stripped in query_terms_lower:
                result.append(self.highlight_start + token + self.highlight_end)
            else:
                result.append(token)
        return result

    def generate(self, doc_path, query_terms):
        """
        Generate a snippet from a document for given query terms.

        Parameters
        ----------
        doc_path : str
            Path to the document file
        query_terms : list[str]
            Query terms to highlight

        Returns
        -------
        dict
            {
                'text': str,           # Plain snippet text
                'highlighted': str,    # Snippet with highlighted terms
                'score': int,          # Number of distinct query terms found
                'doc_path': str        # Document path
            }
        """
        try:
            with open(doc_path, 'r', encoding='utf8', errors='surrogateescape') as f:
                text = f.read()
        except (IOError, OSError):
            return {
                'text': '[Document not accessible]',
                'highlighted': '[Document not accessible]',
                'score': 0,
                'doc_path': doc_path
            }

        tokens = text.split()
        query_terms_lower = set(t.lower() for t in query_terms)

        if not tokens:
            return {
                'text': '',
                'highlighted': '',
                'score': 0,
                'doc_path': doc_path
            }

        start, end, score = self._find_best_window(tokens, query_terms_lower)
        window_tokens = tokens[start:end]

        snippet_text = " ".join(window_tokens)
        highlighted_tokens = self._highlight_tokens(window_tokens, query_terms_lower)
        highlighted_text = " ".join(highlighted_tokens)

        # Add ellipsis indicators
        prefix = "..." if start > 0 else ""
        suffix = "..." if end < len(tokens) else ""

        return {
            'text': prefix + snippet_text + suffix,
            'highlighted': prefix + highlighted_text + suffix,
            'score': score,
            'doc_path': doc_path
        }

    def generate_multi(self, doc_path, query_terms):
        """
        Generate multiple non-overlapping snippets from a document.

        Returns up to max_snippets snippets, each from different parts
        of the document.

        Parameters
        ----------
        doc_path : str
            Path to the document file
        query_terms : list[str]
            Query terms to highlight

        Returns
        -------
        list[dict]
            List of snippet dictionaries (same format as generate())
        """
        try:
            with open(doc_path, 'r', encoding='utf8', errors='surrogateescape') as f:
                text = f.read()
        except (IOError, OSError):
            return [{'text': '[Document not accessible]',
                     'highlighted': '[Document not accessible]',
                     'score': 0, 'doc_path': doc_path}]

        tokens = text.split()
        query_terms_lower = set(t.lower() for t in query_terms)

        if not tokens:
            return [{'text': '', 'highlighted': '', 'score': 0, 'doc_path': doc_path}]

        snippets = []
        used_ranges = []

        for _ in range(self.max_snippets):
            # Find best window that doesn't overlap with existing snippets
            best_start = 0
            best_score = -1
            n = len(tokens)
            window_size = min(self.context_words, n)

            for start in range(max(1, n - window_size + 1)):
                end = start + window_size

                # Check overlap with existing snippets
                overlaps = False
                for us, ue in used_ranges:
                    if start < ue and end > us:
                        overlaps = True
                        break
                if overlaps:
                    continue

                score = self._score_window(tokens, start, end, query_terms_lower)
                if score > best_score:
                    best_score = score
                    best_start = start

            if best_score <= 0:
                break

            end = best_start + window_size
            used_ranges.append((best_start, end))

            window_tokens = tokens[best_start:end]
            snippet_text = " ".join(window_tokens)
            highlighted_tokens = self._highlight_tokens(window_tokens, query_terms_lower)
            highlighted_text = " ".join(highlighted_tokens)

            prefix = "..." if best_start > 0 else ""
            suffix = "..." if end < len(tokens) else ""

            snippets.append({
                'text': prefix + snippet_text + suffix,
                'highlighted': prefix + highlighted_text + suffix,
                'score': best_score,
                'doc_path': doc_path
            })

        return snippets


def format_search_results(results, query, snippet_generator=None, max_display=10):
    """
    Format search results with snippets for display.

    Parameters
    ----------
    results : list[tuple[float, str]]
        Search results [(score, doc_path), ...]
    query : str
        Original query string
    snippet_generator : SnippetGenerator, optional
        Snippet generator instance. If None, creates a default one.
    max_display : int
        Maximum number of results to display

    Returns
    -------
    str
        Formatted results string
    """
    if snippet_generator is None:
        snippet_generator = SnippetGenerator()

    query_terms = query.split()
    lines = []

    for rank, (score, doc_path) in enumerate(results[:max_display], 1):
        lines.append(f"\n  [{rank}] {doc_path} (score: {score:.4f})")

        snippet = snippet_generator.generate(doc_path, query_terms)
        if snippet['highlighted']:
            lines.append(f"      {snippet['highlighted']}")

    return "\n".join(lines)


if __name__ == "__main__":
    from bsbi import BSBIIndex
    from compression import VBEPostings

    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=VBEPostings,
                              output_dir='index')

    snippet_gen = SnippetGenerator(context_words=25, highlight_start="[",
                                   highlight_end="]")

    queries = ["alkylated with radioactive iodoacetate",
               "psychodrama for disturbed children",
               "lipid metabolism in toxemia and normal pregnancy"]

    print("=" * 60)
    print("Search Results with Snippets")
    print("=" * 60)

    for query in queries:
        print(f"\nQuery: {query}")

        results = BSBI_instance.retrieve_bm25(query, k=5)
        output = format_search_results(results, query, snippet_gen, max_display=5)
        print(output)

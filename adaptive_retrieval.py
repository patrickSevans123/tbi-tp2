"""
BONUS: Adaptive Retrieval

Implements several adaptive/interactive retrieval techniques:

1. GAR (Graph-based Adaptive Re-ranking)
   Reference: "Adaptive Re-Ranking with a Corpus Graph" (MacAvaney et al., CIKM 2022)
   - Builds a corpus graph where each document is connected to its k-nearest neighbors
   - Initial retrieval (e.g., BM25) produces candidate set
   - Iteratively: score a batch of candidates, then expand candidates by adding
     graph neighbors of highly-scored documents
   - Alternates between scoring initial results and graph-discovered documents
   - Concentrates expensive scoring on promising regions of the document space

2. Pseudo-Relevance Feedback (Rocchio Algorithm)
   - Assumes top-k initial results are relevant
   - Expands the query using terms from those documents
   - Re-runs retrieval with the expanded query

3. Query Expansion via Robertson Selection Value (RSV)
   - Uses term statistics from feedback documents to select expansion terms

4. Proximity-based Re-ranking
   - Boosts documents where query terms appear close together
"""

import math
import os
import heapq

from index import InvertedIndexReader
from compression import VBEPostings


# ============================================================
# GAR: Graph-based Adaptive Re-ranking
# ============================================================

class CorpusGraph:
    """
    Corpus Graph for Adaptive Re-ranking.

    A graph where nodes are documents and edges connect documents that
    are highly similar. Used by GAR to discover potentially relevant
    documents that may not appear in the initial retrieval results.

    Similarity is computed using TF-IDF cosine similarity between
    document vectors built from the inverted index.

    Attributes
    ----------
    neighbors : dict[int, list[int]]
        Mapping from doc_id to list of neighbor doc_ids (sorted by similarity)
    k : int
        Number of neighbors per document
    """

    def __init__(self, k=16):
        """
        Parameters
        ----------
        k : int
            Number of nearest neighbors per document (default: 16)
        """
        self.k = k
        self.neighbors = {}

    def build(self, index_name, postings_encoding, output_dir):
        """
        Build the corpus graph from the inverted index.

        For each document, computes TF-IDF cosine similarity against all
        other documents and retains the top-k most similar as neighbors.

        Uses an inverted approach: iterates through the inverted index
        to accumulate pairwise dot products, then normalizes.

        Parameters
        ----------
        index_name : str
            Name of the inverted index
        postings_encoding : class
            Postings encoding class
        output_dir : str
            Index directory
        """
        print("  Building corpus graph...")

        with InvertedIndexReader(index_name, postings_encoding,
                                 directory=output_dir) as reader:
            N = len(reader.doc_length)
            doc_ids = sorted(reader.doc_length.keys())

            # Step 1: Compute document norms and pairwise dot products
            # using the inverted index (much faster than doc-by-doc)
            doc_norm_sq = {}  # doc_id -> sum of squared weights
            # For pairwise similarity, we accumulate dot products
            # via shared terms. Use a dict-of-dicts for sparsity.
            dot_products = {}  # doc_i -> {doc_j -> dot_product}

            reader.reset()
            term_count = 0
            for term_id, postings, tf_list in reader:
                df = len(postings)
                if df < 2:
                    # Terms appearing in only 1 doc don't contribute to similarity
                    continue
                idf = math.log(N / df) if df > 0 else 0

                # Build weights for this term
                weights = []
                for i in range(len(postings)):
                    tf = tf_list[i]
                    if tf > 0:
                        w = (1 + math.log(tf)) * idf
                        weights.append((postings[i], w))

                # Accumulate norms
                for doc_id, w in weights:
                    if doc_id not in doc_norm_sq:
                        doc_norm_sq[doc_id] = 0.0
                    doc_norm_sq[doc_id] += w * w

                # Accumulate pairwise dot products
                # Only process terms with reasonable df to avoid O(N^2) for common terms
                if df <= 500:
                    for i in range(len(weights)):
                        di, wi = weights[i]
                        if di not in dot_products:
                            dot_products[di] = {}
                        for j in range(i + 1, len(weights)):
                            dj, wj = weights[j]
                            dot_products[di][dj] = dot_products[di].get(dj, 0.0) + wi * wj

                term_count += 1

            # Step 2: For each document, find top-k neighbors by cosine similarity
            print(f"    Processed {term_count} terms, computing neighbors...")

            for di in doc_ids:
                if di not in dot_products:
                    self.neighbors[di] = []
                    continue

                norm_i = math.sqrt(doc_norm_sq.get(di, 1.0))
                candidates = []

                for dj, dp in dot_products[di].items():
                    norm_j = math.sqrt(doc_norm_sq.get(dj, 1.0))
                    if norm_i > 0 and norm_j > 0:
                        cos_sim = dp / (norm_i * norm_j)
                        candidates.append((cos_sim, dj))

                # Also check reverse direction (since we only stored i < j pairs)
                candidates.sort(reverse=True)
                self.neighbors[di] = [dj for _, dj in candidates[:self.k]]

            # Add reverse edges (if doc A is neighbor of B, B might be neighbor of A)
            for di in list(dot_products.keys()):
                for dj in list(dot_products.get(di, {}).keys()):
                    if dj not in dot_products:
                        dot_products[dj] = {}
                    if di not in dot_products[dj]:
                        dot_products[dj][di] = dot_products[di][dj]

            # Re-compute neighbors including reverse edges
            for di in doc_ids:
                if di in dot_products:
                    norm_i = math.sqrt(doc_norm_sq.get(di, 1.0))
                    candidates = []
                    for dj, dp in dot_products.get(di, {}).items():
                        norm_j = math.sqrt(doc_norm_sq.get(dj, 1.0))
                        if norm_i > 0 and norm_j > 0:
                            cos_sim = dp / (norm_i * norm_j)
                            candidates.append((cos_sim, dj))
                    candidates.sort(reverse=True)
                    self.neighbors[di] = [dj for _, dj in candidates[:self.k]]

        print(f"    Corpus graph built: {len(self.neighbors)} nodes, "
              f"k={self.k} neighbors each")

    def get_neighbors(self, doc_id):
        """Get the k-nearest neighbors of a document."""
        return self.neighbors.get(doc_id, [])

    def save(self, filepath):
        """Save corpus graph to disk."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({'k': self.k, 'neighbors': self.neighbors}, f)
        print(f"    Corpus graph saved to {filepath}")

    def load(self, filepath):
        """Load corpus graph from disk."""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.k = data['k']
        self.neighbors = data['neighbors']
        print(f"    Corpus graph loaded: {len(self.neighbors)} nodes, k={self.k}")


class GAR:
    """
    Graph-based Adaptive Re-ranking (GAR).

    Reference: MacAvaney et al., "Adaptive Re-Ranking with a Corpus Graph", CIKM 2022.

    Algorithm:
        1. Run initial retrieval (e.g., BM25) to get candidate set C
        2. Score first batch of candidates from C using a scorer
        3. For each highly-scored document, add its corpus graph neighbors
           to the candidate set
        4. Score next batch (alternating between initial results and graph neighbors)
        5. Repeat until budget is exhausted or candidates are depleted
        6. Return top-k scored documents

    The key insight is that relevant documents tend to cluster in the corpus
    graph -- if a document is relevant, its neighbors are likely relevant too.
    This allows discovering relevant documents that the initial retriever missed.

    Parameters
    ----------
    corpus_graph : CorpusGraph
        Pre-built corpus graph
    batch_size : int
        Number of documents to score per iteration
    n_iterations : int
        Number of GAR iterations (alternating initial/graph)
    """

    def __init__(self, corpus_graph, batch_size=10, n_iterations=5):
        self.corpus_graph = corpus_graph
        self.batch_size = batch_size
        self.n_iterations = n_iterations

    def rerank(self, query, initial_results, scorer_fn):
        """
        Perform Graph-based Adaptive Re-ranking.

        Parameters
        ----------
        query : str
            Query string
        initial_results : list[tuple[float, int]]
            Initial retrieval results as [(score, doc_id), ...]
        scorer_fn : callable
            Function (query, doc_id) -> score for detailed scoring

        Returns
        -------
        list[tuple[float, int, int]]
            Re-ranked results as [(score, doc_id, iteration), ...]
            iteration: even=initial retrieval, odd=corpus graph, -1=backfilled
        """
        # Candidate pools
        initial_queue = list(initial_results)  # [(score, doc_id)]
        initial_queue.sort(reverse=True)  # highest initial score first

        graph_queue = []       # [(priority, doc_id)] - discovered via graph
        scored = {}            # doc_id -> (score, iteration)
        seen_in_graph = set()  # doc_ids already added to graph_queue

        initial_ptr = 0

        for iteration in range(self.n_iterations):
            batch = []

            if iteration % 2 == 0:
                # Even iteration: score from initial retrieval
                while len(batch) < self.batch_size and initial_ptr < len(initial_queue):
                    _, doc_id = initial_queue[initial_ptr]
                    initial_ptr += 1
                    if doc_id not in scored:
                        batch.append((doc_id, iteration))
            else:
                # Odd iteration: score from corpus graph neighbors
                while len(batch) < self.batch_size and graph_queue:
                    _, doc_id = heapq.heappop(graph_queue)
                    if doc_id not in scored:
                        batch.append((doc_id, iteration))

            if not batch:
                continue

            # Score the batch
            for doc_id, iter_num in batch:
                score = scorer_fn(query, doc_id)
                scored[doc_id] = (score, iter_num)

                # Expand: add neighbors of scored documents to graph queue
                for neighbor_id in self.corpus_graph.get_neighbors(doc_id):
                    if neighbor_id not in scored and neighbor_id not in seen_in_graph:
                        # Priority: negative score (heapq is min-heap, we want max)
                        heapq.heappush(graph_queue, (-score, neighbor_id))
                        seen_in_graph.add(neighbor_id)

        # Backfill: add remaining initial results that weren't scored
        results = [(score, doc_id, iter_num) for doc_id, (score, iter_num) in scored.items()]

        # Add unscored initial results with their initial score, marked as backfill
        for init_score, doc_id in initial_queue:
            if doc_id not in scored:
                results.append((init_score, doc_id, -1))

        results.sort(key=lambda x: -x[0])
        return results


def retrieve_gar(query, bsbi_index, corpus_graph, k=10, batch_size=20, n_iterations=10):
    """
    Full GAR retrieval pipeline.

    1. Run BM25 initial retrieval
    2. Build BM25 scorer function
    3. Run GAR with corpus graph
    4. Return top-k results

    Parameters
    ----------
    query : str
        Query string
    bsbi_index : BSBIIndex
        Search index
    corpus_graph : CorpusGraph
        Pre-built corpus graph
    k : int
        Number of results
    batch_size : int
        GAR batch size
    n_iterations : int
        Number of GAR iterations

    Returns
    -------
    list[tuple[float, str]]
        Top-k results [(score, doc_path), ...]
    """
    if len(bsbi_index.term_id_map) == 0:
        bsbi_index.load()

    # Step 1: Initial BM25 retrieval (broad, get many candidates)
    initial_results = bsbi_index.retrieve_bm25(query, k=k * 5)

    # Convert to (score, doc_id) format
    doc_path_to_id = {}
    for i in range(len(bsbi_index.doc_id_map.id_to_str)):
        doc_path_to_id[bsbi_index.doc_id_map.id_to_str[i]] = i

    initial_scored = []
    for score, doc_path in initial_results:
        if doc_path in doc_path_to_id:
            initial_scored.append((score, doc_path_to_id[doc_path]))

    # Step 2: Build BM25 scorer (precompute for efficiency)
    query_terms = query.split()
    term_data = {}  # term_id -> (idf, postings_dict{doc_id: tf})

    with InvertedIndexReader(bsbi_index.index_name, bsbi_index.postings_encoding,
                             directory=bsbi_index.output_dir) as reader:
        N = len(reader.doc_length)
        avgdl = sum(reader.doc_length.values()) / N if N > 0 else 0
        doc_length = dict(reader.doc_length)

        for word in query_terms:
            term_id = bsbi_index.term_id_map[word]
            if term_id in reader.postings_dict:
                df = reader.postings_dict[term_id][1]
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                postings, tf_list = reader.get_postings_list(term_id)
                tf_dict = {postings[i]: tf_list[i] for i in range(len(postings))}
                term_data[term_id] = (idf, tf_dict)

    k1, b = 1.2, 0.75

    def bm25_scorer(query_str, doc_id):
        """Score a single document with BM25."""
        score = 0.0
        dl = doc_length.get(doc_id, 0)
        for word in query_str.split():
            tid = bsbi_index.term_id_map[word]
            if tid in term_data:
                idf, tf_dict = term_data[tid]
                tf = tf_dict.get(doc_id, 0)
                if tf > 0:
                    tf_comp = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                    score += idf * tf_comp
        return score

    # Step 3: Run GAR
    gar = GAR(corpus_graph, batch_size=batch_size, n_iterations=n_iterations)
    gar_results = gar.rerank(query, initial_scored, bm25_scorer)

    # Convert back to (score, doc_path) format
    results = [(score, bsbi_index.doc_id_map[doc_id])
               for score, doc_id, _ in gar_results[:k]]
    return results


def retrieve_gar_lsi(query, bsbi_index, corpus_graph, lsi_retriever,
                     k=10, batch_size=None, n_iterations=None):
    """
    GAR retrieval using LSI as the expensive re-ranking scorer.

    This is the intended use of GAR: cheap initial retrieval (BM25) to
    get candidates, then GAR selects which candidates to score with the
    expensive scorer (LSI cosine similarity in latent space).

    Pipeline:
        BM25 (cheap, broad) --> GAR selects batches --> LSI (expensive, accurate)

    This mirrors the original paper's pattern of BM25 --> GAR --> MonoT5,
    but replaces the neural scorer with LSI (since we cannot use external
    libraries for neural models).

    Parameters
    ----------
    query : str
        Query string
    bsbi_index : BSBIIndex
        Search index
    corpus_graph : CorpusGraph
        Pre-built corpus graph
    lsi_retriever : LSIRetriever
        Pre-built LSI model for scoring
    k : int
        Number of results
    batch_size : int
        GAR batch size
    n_iterations : int
        Number of GAR iterations

    Returns
    -------
    list[tuple[float, str]]
        Top-k results [(score, doc_path), ...]
    """
    if len(bsbi_index.term_id_map) == 0:
        bsbi_index.load()

    from lsi import cosine_sim, vec_norm

    # Scale GAR budget with k so that enough docs get LSI-scored
    if batch_size is None:
        batch_size = max(20, k // 5)
    if n_iterations is None:
        n_iterations = max(10, k // batch_size)

    # Step 1: Initial BM25 retrieval (cheap, broad)
    initial_results = bsbi_index.retrieve_bm25(query, k=k * 5)

    doc_path_to_id = {}
    for i in range(len(bsbi_index.doc_id_map.id_to_str)):
        doc_path_to_id[bsbi_index.doc_id_map.id_to_str[i]] = i

    initial_scored = []
    for score, doc_path in initial_results:
        if doc_path in doc_path_to_id:
            initial_scored.append((score, doc_path_to_id[doc_path]))

    # Step 2: Build LSI scorer
    # Pre-compute query vector in latent space (done once)
    query_terms = query.split()
    q_vec = lsi_retriever.query_vector(query_terms, bsbi_index.term_id_map)
    q_norm = vec_norm(q_vec)

    def lsi_scorer(query_str, doc_id):
        """Score a single document using LSI cosine similarity."""
        if q_norm < 1e-10:
            return 0.0
        doc_vec = lsi_retriever.doc_vectors.get(doc_id)
        if doc_vec is None:
            return 0.0
        return cosine_sim(q_vec, doc_vec)

    # Step 3: Run GAR with LSI as the expensive scorer
    gar = GAR(corpus_graph, batch_size=batch_size, n_iterations=n_iterations)
    gar_results = gar.rerank(query, initial_scored, lsi_scorer)

    # Convert back to (score, doc_path) format
    results = [(score, bsbi_index.doc_id_map[doc_id])
               for score, doc_id, _ in gar_results[:k]]
    return results


# ============================================================
# Pseudo-Relevance Feedback (Rocchio)
# ============================================================

class RocchioExpander:
    """
    Pseudo-Relevance Feedback using the Rocchio Algorithm.

    The Rocchio formula modifies the original query vector:
        q' = alpha * q + beta * (1/|Dr|) * sum(d in Dr) - gamma * (1/|Dnr|) * sum(d in Dnr)

    where Dr = relevant docs (assumed from top-k results),
    Dnr = non-relevant docs (remaining).

    In practice (blind/pseudo relevance feedback), we assume the top-k
    retrieved documents are relevant and use their terms to expand the query.
    """

    def __init__(self, alpha=1.0, beta=0.75, gamma=0.0,
                 n_feedback_docs=10, n_expand_terms=20):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.n_feedback_docs = n_feedback_docs
        self.n_expand_terms = n_expand_terms

    def expand_query(self, query, initial_results, index_name,
                     postings_encoding, output_dir, term_id_map, doc_id_map):
        """
        Expand query using pseudo-relevance feedback.
        """
        query_vec = {}
        for term_str in query.split():
            term_id = term_id_map[term_str]
            query_vec[term_id] = query_vec.get(term_id, 0) + self.alpha

        feedback_docs = set()
        doc_path_to_id = {}
        for i in range(len(doc_id_map.id_to_str)):
            doc_path_to_id[doc_id_map.id_to_str[i]] = i

        for score, doc_path in initial_results[:self.n_feedback_docs]:
            if doc_path in doc_path_to_id:
                feedback_docs.add(doc_path_to_id[doc_path])

        if not feedback_docs:
            return query_vec

        feedback_term_weights = {}
        with InvertedIndexReader(index_name, postings_encoding,
                                 directory=output_dir) as reader:
            N = len(reader.doc_length)
            reader.reset()

            for term_id, postings, tf_list in reader:
                df = len(postings)
                idf = math.log(N / df) if df > 0 else 0

                for i, doc_id in enumerate(postings):
                    if doc_id in feedback_docs:
                        tf = tf_list[i]
                        weight = (1 + math.log(tf)) * idf if tf > 0 else 0
                        if term_id not in feedback_term_weights:
                            feedback_term_weights[term_id] = 0.0
                        feedback_term_weights[term_id] += weight

        n_fb = len(feedback_docs)
        for term_id in feedback_term_weights:
            feedback_term_weights[term_id] /= n_fb

        expanded_vec = dict(query_vec)
        for term_id, weight in feedback_term_weights.items():
            if term_id in expanded_vec:
                expanded_vec[term_id] += self.beta * weight
            else:
                expanded_vec[term_id] = self.beta * weight

        original_terms = set(query_vec.keys())
        expansion_candidates = [(w, tid) for tid, w in expanded_vec.items()
                                if tid not in original_terms]
        expansion_candidates.sort(reverse=True)

        final_vec = dict(query_vec)
        for weight, term_id in expansion_candidates[:self.n_expand_terms]:
            if weight > 0:
                final_vec[term_id] = weight

        return final_vec

    def retrieve_with_feedback(self, query, bsbi_index, k=10):
        """
        Full pseudo-relevance feedback retrieval pipeline.

        Uses BM25 scoring with the expanded query for better results.
        The expanded query terms are converted back to a weighted string query
        and scored using BM25 term-at-a-time.
        """
        if len(bsbi_index.term_id_map) == 0:
            bsbi_index.load()

        initial_results = bsbi_index.retrieve_bm25(query, k=self.n_feedback_docs * 2)
        if not initial_results:
            return []

        expanded_query = self.expand_query(
            query, initial_results,
            bsbi_index.index_name, bsbi_index.postings_encoding,
            bsbi_index.output_dir, bsbi_index.term_id_map, bsbi_index.doc_id_map
        )

        # Re-score using BM25 with expanded query weights
        k1, b = 1.2, 0.75
        with InvertedIndexReader(bsbi_index.index_name, bsbi_index.postings_encoding,
                                 directory=bsbi_index.output_dir) as reader:
            N = len(reader.doc_length)
            if N == 0:
                return []
            avgdl = sum(reader.doc_length.values()) / N

            scores = {}
            for term_id, q_weight in expanded_query.items():
                if term_id in reader.postings_dict:
                    df = reader.postings_dict[term_id][1]
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                    postings, tf_list = reader.get_postings_list(term_id)

                    for i in range(len(postings)):
                        doc_id = postings[i]
                        tf = tf_list[i]
                        dl = reader.doc_length.get(doc_id, 0)
                        tf_comp = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        scores[doc_id] += q_weight * idf * tf_comp

        results = [(score, bsbi_index.doc_id_map[doc_id])
                   for doc_id, score in scores.items()]
        results.sort(key=lambda x: -x[0])
        return results[:k]


# ============================================================
# Query Expansion via RSV
# ============================================================

class QueryExpander:
    """
    Query Expansion via Robertson Selection Value (RSV).

    RSV(t) = r_t * log((r_t / R) / (n_t / N))

    where r_t = relevant docs with term t, R = total relevant docs,
    n_t = df(t), N = total docs.
    """

    def __init__(self, n_feedback_docs=5, n_expand_terms=10):
        self.n_feedback_docs = n_feedback_docs
        self.n_expand_terms = n_expand_terms

    def expand_with_rsv(self, query, initial_results, index_name,
                        postings_encoding, output_dir, term_id_map, doc_id_map):
        """Expand query using Robertson Selection Value."""
        original_terms = set(query.split())
        original_term_ids = set()
        for t in original_terms:
            original_term_ids.add(term_id_map[t])

        feedback_docs = set()
        doc_path_to_id = {}
        for i in range(len(doc_id_map.id_to_str)):
            doc_path_to_id[doc_id_map.id_to_str[i]] = i

        for score, doc_path in initial_results[:self.n_feedback_docs]:
            if doc_path in doc_path_to_id:
                feedback_docs.add(doc_path_to_id[doc_path])

        R = len(feedback_docs)
        if R == 0:
            return list(original_terms)

        term_rsv = {}
        with InvertedIndexReader(index_name, postings_encoding,
                                 directory=output_dir) as reader:
            N = len(reader.doc_length)
            reader.reset()

            for term_id, postings, tf_list in reader:
                if term_id in original_term_ids:
                    continue
                df = len(postings)
                r_t = sum(1 for d in postings if d in feedback_docs)
                if r_t > 0 and df > 0:
                    rsv = r_t * math.log(((r_t + 0.5) / (R - r_t + 0.5)) /
                                         ((df - r_t + 0.5) / (N - df - R + r_t + 0.5)))
                    if rsv > 0:
                        term_rsv[term_id] = rsv

        sorted_terms = sorted(term_rsv.items(), key=lambda x: -x[1])
        expanded = list(original_terms)
        for term_id, rsv in sorted_terms[:self.n_expand_terms]:
            term_str = term_id_map[term_id]
            expanded.append(term_str)

        return expanded


# ============================================================
# Proximity-based Re-ranking
# ============================================================

class ProximityReranker:
    """
    Re-ranks initial retrieval results based on term proximity.
    Documents where query terms appear closer together are ranked higher.
    """

    def __init__(self, proximity_weight=0.3):
        self.proximity_weight = proximity_weight

    def _min_window_size(self, text_tokens, query_terms):
        """Find minimum window containing all query terms."""
        if not query_terms:
            return len(text_tokens) + 1

        term_positions = {}
        for i, token in enumerate(text_tokens):
            t = token.lower()
            if t in query_terms:
                if t not in term_positions:
                    term_positions[t] = []
                term_positions[t].append(i)

        if len(term_positions) < len(query_terms):
            return len(text_tokens) + 1

        events = []
        for term, positions in term_positions.items():
            for pos in positions:
                events.append((pos, term))
        events.sort()

        min_window = len(text_tokens) + 1
        term_count = {}
        terms_found = 0
        left = 0
        n_query_terms = len(query_terms)

        for right in range(len(events)):
            pos_r, term_r = events[right]
            if term_r not in term_count:
                term_count[term_r] = 0
            if term_count[term_r] == 0:
                terms_found += 1
            term_count[term_r] += 1

            while terms_found == n_query_terms:
                pos_l, term_l = events[left]
                window = pos_r - pos_l + 1
                if window < min_window:
                    min_window = window
                term_count[term_l] -= 1
                if term_count[term_l] == 0:
                    terms_found -= 1
                left += 1

        return min_window

    def rerank(self, query, results, max_rerank=50):
        """Re-rank results based on term proximity."""
        query_terms = set(query.lower().split())
        reranked = []

        for i, (score, doc_path) in enumerate(results[:max_rerank]):
            try:
                with open(doc_path, 'r', encoding='utf8', errors='surrogateescape') as f:
                    text = f.read()
                tokens = text.lower().split()
                window_size = self._min_window_size(tokens, query_terms)
                if window_size <= len(tokens):
                    prox_score = len(query_terms) / window_size
                else:
                    prox_score = 0
                final_score = score + self.proximity_weight * prox_score * score
            except (IOError, OSError):
                final_score = score

            reranked.append((final_score, doc_path))

        reranked.extend(results[max_rerank:])
        reranked.sort(key=lambda x: -x[0])
        return reranked


# ============================================================
# Full Adaptive Pipeline
# ============================================================

def retrieve_adaptive(query, bsbi_index, k=10,
                      use_gar=False, corpus_graph=None,
                      use_rocchio=True, use_expansion=True, use_proximity=True):
    """
    Full adaptive retrieval pipeline combining multiple techniques.

    Pipeline:
        1. (Optional) GAR with corpus graph
        2. (Optional) Query expansion via RSV
        3. (Optional) Pseudo-relevance feedback via Rocchio
        4. (Optional) Proximity-based re-ranking
    """
    if len(bsbi_index.term_id_map) == 0:
        bsbi_index.load()

    if use_gar and corpus_graph is not None:
        results = retrieve_gar(query, bsbi_index, corpus_graph, k=k * 2)
    elif use_rocchio:
        rocchio = RocchioExpander(alpha=1.0, beta=0.75, n_feedback_docs=5,
                                 n_expand_terms=15)
        results = rocchio.retrieve_with_feedback(query, bsbi_index, k=k * 2)
    elif use_expansion:
        initial_results = bsbi_index.retrieve_bm25(query, k=20)
        expander = QueryExpander(n_feedback_docs=5, n_expand_terms=10)
        expanded_terms = expander.expand_with_rsv(
            query, initial_results,
            bsbi_index.index_name, bsbi_index.postings_encoding,
            bsbi_index.output_dir, bsbi_index.term_id_map, bsbi_index.doc_id_map
        )
        expanded_query = " ".join(expanded_terms)
        results = bsbi_index.retrieve_bm25(expanded_query, k=k * 2)
    else:
        results = bsbi_index.retrieve_bm25(query, k=k * 2)

    if use_proximity:
        reranker = ProximityReranker(proximity_weight=0.3)
        results = reranker.rerank(query, results, max_rerank=min(50, len(results)))

    return results[:k]


if __name__ == "__main__":
    import pickle
    from bsbi import BSBIIndex

    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=VBEPostings,
                              output_dir='index')

    queries = ["alkylated with radioactive iodoacetate",
               "psychodrama for disturbed children",
               "lipid metabolism in toxemia and normal pregnancy"]

    print("=" * 60)
    print("Adaptive Retrieval Results")
    print("=" * 60)

    # ---- Rocchio PRF ----
    for query in queries:
        print(f"\nQuery: {query}")

        print("\n  --- BM25 (baseline) ---")
        for score, doc in BSBI_instance.retrieve_bm25(query, k=5):
            print(f"    {doc:30} {score:.3f}")

        print("\n  --- Rocchio PRF ---")
        rocchio = RocchioExpander()
        for score, doc in rocchio.retrieve_with_feedback(query, BSBI_instance, k=5):
            print(f"    {doc:30} {score:.3f}")

    # ---- GAR (Graph-based Adaptive Re-ranking) ----
    print("\n" + "=" * 60)
    print("GAR: Graph-based Adaptive Re-ranking")
    print("=" * 60)

    graph_path = os.path.join('index', 'corpus_graph.pkl')
    cg = CorpusGraph(k=16)

    if os.path.exists(graph_path):
        cg.load(graph_path)
    else:
        cg.build('main_index', VBEPostings, 'index')
        cg.save(graph_path)

    for query in queries:
        print(f"\nQuery: {query}")
        print("  --- GAR (BM25 + Corpus Graph) ---")
        for score, doc in retrieve_gar(query, BSBI_instance, cg, k=10):
            print(f"    {doc:30} {score:.3f}")

    print("\n  --- Full Adaptive (GAR + Proximity) ---")
    for query in queries:
        print(f"\nQuery: {query}")
        for score, doc in retrieve_adaptive(query, BSBI_instance, k=10,
                                            use_gar=True, corpus_graph=cg,
                                            use_proximity=True):
            print(f"    {doc:30} {score:.3f}")

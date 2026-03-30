"""
BONUS: Vector Indexing for LSI Document Vectors

Implements FAISS-like vector indexing structures from scratch using only
Python standard library. These indexes accelerate nearest-neighbor search
over document vectors in the LSI latent space.

Three index types are provided:

1. FlatIndex (brute-force baseline)
   - Computes cosine similarity against all vectors
   - Exact results, O(N*d) per query
   - Equivalent to FAISS IndexFlatIP

2. IVFIndex (Inverted File Index)
   - Partitions vectors into clusters using K-means
   - At query time, only searches the nearest nprobe clusters
   - Approximate results, much faster for large collections
   - Equivalent to FAISS IndexIVFFlat

3. LSHIndex (Locality-Sensitive Hashing)
   - Projects vectors into binary hash codes using random hyperplanes
   - Groups similar vectors into the same hash bucket
   - Sub-linear query time for high-dimensional data
   - Equivalent to FAISS IndexLSH

Reference: Johnson et al., "Billion-scale similarity search with GPUs"
(FAISS paper, IEEE TBD 2019)
"""

import math
import random
import pickle
import os


# ============================================================
# Vector Utilities
# ============================================================

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

def vec_norm(v):
    return math.sqrt(sum(x * x for x in v))

def vec_add(a, b):
    return [x + y for x, y in zip(a, b)]

def vec_scale(v, s):
    return [x * s for x in v]

def cosine_sim(a, b):
    d = dot(a, b)
    na, nb = vec_norm(a), vec_norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return d / (na * nb)

def vec_normalize(v):
    n = vec_norm(v)
    if n < 1e-15:
        return [0.0] * len(v)
    return [x / n for x in v]


# ============================================================
# 1. FlatIndex (Brute-Force Baseline)
# ============================================================

class FlatIndex:
    """
    Flat (brute-force) vector index.

    Stores all vectors and computes cosine similarity against every
    vector at query time. Guarantees exact results.

    Equivalent to FAISS IndexFlatIP (inner product on normalized vectors).

    Time complexity: O(N * d) per query
    Space complexity: O(N * d)
    """

    def __init__(self):
        self.vectors = {}  # doc_id -> normalized vector
        self.dim = 0

    def add(self, doc_id, vector):
        """Add a document vector to the index."""
        self.vectors[doc_id] = vec_normalize(vector)
        self.dim = len(vector)

    def build(self, doc_vectors):
        """Build index from a dictionary of doc_id -> vector."""
        for doc_id, vec in doc_vectors.items():
            self.add(doc_id, vec)

    def search(self, query_vector, k=10):
        """
        Find k nearest neighbors by cosine similarity.

        Returns list of (similarity, doc_id) sorted descending.
        """
        q_norm = vec_normalize(query_vector)
        scores = []
        for doc_id, vec in self.vectors.items():
            sim = dot(q_norm, vec)
            scores.append((sim, doc_id))
        scores.sort(key=lambda x: -x[0])
        return scores[:k]


# ============================================================
# 2. IVFIndex (Inverted File Index with K-Means)
# ============================================================

class IVFIndex:
    """
    Inverted File Index for approximate nearest-neighbor search.

    Partitions the vector space into n_clusters using K-means clustering.
    Each document vector is assigned to its nearest cluster centroid.
    At query time, only the nprobe nearest clusters are searched,
    reducing the number of similarity computations.

    Equivalent to FAISS IndexIVFFlat.

    Algorithm:
        Training (build):
            1. Run K-means to find n_clusters centroids
            2. Assign each document to its nearest centroid
            3. Store documents in per-cluster inverted lists

        Query (search):
            1. Find the nprobe nearest centroids to the query
            2. Search only documents in those clusters
            3. Return top-k by cosine similarity

    Time complexity: O(n_clusters*d + nprobe*N/n_clusters*d) per query
    Space complexity: O(N*d + n_clusters*d)

    Parameters
    ----------
    n_clusters : int
        Number of Voronoi cells / clusters (default: 16)
    nprobe : int
        Number of clusters to search at query time (default: 4)
    n_iter : int
        Number of K-means iterations (default: 20)
    """

    def __init__(self, n_clusters=16, nprobe=4, n_iter=20, seed=42):
        self.n_clusters = n_clusters
        self.nprobe = nprobe
        self.n_iter = n_iter
        self.seed = seed
        self.centroids = []        # list of centroid vectors
        self.invlists = {}         # cluster_id -> [(doc_id, vector), ...]
        self.dim = 0

    def _kmeans(self, vectors):
        """
        Run K-means clustering on a list of (doc_id, vector) pairs.

        Returns centroids and assignments.
        """
        rng = random.Random(self.seed)
        n = len(vectors)
        k = min(self.n_clusters, n)

        # Initialize centroids using K-means++ style
        indices = list(range(n))
        rng.shuffle(indices)
        centroids = [vec_normalize(vectors[indices[i]][1]) for i in range(k)]

        assignments = [0] * n

        for iteration in range(self.n_iter):
            # Assignment step
            for i, (doc_id, vec) in enumerate(vectors):
                v_norm = vec_normalize(vec)
                best_sim = -2
                best_c = 0
                for c in range(k):
                    sim = dot(v_norm, centroids[c])
                    if sim > best_sim:
                        best_sim = sim
                        best_c = c
                assignments[i] = best_c

            # Update step
            new_centroids = [[0.0] * self.dim for _ in range(k)]
            counts = [0] * k
            for i, (doc_id, vec) in enumerate(vectors):
                c = assignments[i]
                new_centroids[c] = vec_add(new_centroids[c], vec)
                counts[c] += 1

            for c in range(k):
                if counts[c] > 0:
                    centroids[c] = vec_normalize(
                        vec_scale(new_centroids[c], 1.0 / counts[c]))

        return centroids, assignments

    def build(self, doc_vectors):
        """
        Build the IVF index from document vectors.

        Parameters
        ----------
        doc_vectors : dict[int, list[float]]
            Mapping from doc_id to vector
        """
        if not doc_vectors:
            return

        vectors = [(doc_id, vec) for doc_id, vec in doc_vectors.items()]
        self.dim = len(vectors[0][1])

        # Run K-means
        self.centroids, assignments = self._kmeans(vectors)

        # Build inverted lists
        self.invlists = {c: [] for c in range(len(self.centroids))}
        for i, (doc_id, vec) in enumerate(vectors):
            c = assignments[i]
            self.invlists[c].append((doc_id, vec_normalize(vec)))

    def search(self, query_vector, k=10):
        """
        Approximate nearest-neighbor search.

        Searches only the nprobe nearest clusters.

        Returns list of (similarity, doc_id) sorted descending.
        """
        q_norm = vec_normalize(query_vector)

        # Find nprobe nearest centroids
        centroid_sims = []
        for c, centroid in enumerate(self.centroids):
            sim = dot(q_norm, centroid)
            centroid_sims.append((sim, c))
        centroid_sims.sort(key=lambda x: -x[0])
        probe_clusters = [c for _, c in centroid_sims[:self.nprobe]]

        # Search within selected clusters
        scores = []
        for c in probe_clusters:
            for doc_id, vec in self.invlists.get(c, []):
                sim = dot(q_norm, vec)
                scores.append((sim, doc_id))

        scores.sort(key=lambda x: -x[0])
        return scores[:k]

    def stats(self):
        """Return index statistics."""
        sizes = [len(self.invlists.get(c, [])) for c in range(len(self.centroids))]
        return {
            'n_clusters': len(self.centroids),
            'nprobe': self.nprobe,
            'total_vectors': sum(sizes),
            'min_cluster_size': min(sizes) if sizes else 0,
            'max_cluster_size': max(sizes) if sizes else 0,
            'avg_cluster_size': sum(sizes) / len(sizes) if sizes else 0,
        }


# ============================================================
# 3. LSHIndex (Locality-Sensitive Hashing)
# ============================================================

class LSHIndex:
    """
    Locality-Sensitive Hashing index for approximate nearest-neighbor search.

    Uses random hyperplane projections to hash vectors into binary codes.
    Vectors with similar directions will have similar hash codes (by the
    property of random hyperplane LSH for cosine similarity).

    At query time, the query is hashed and candidates are retrieved from
    hash buckets with the same or similar hash codes (multi-probe).

    Equivalent to FAISS IndexLSH.

    Algorithm:
        Build:
            1. Generate n_bits random hyperplanes
            2. For each vector, compute binary hash: bit_i = sign(dot(v, h_i))
            3. Store vectors in hash buckets keyed by their binary code

        Query:
            1. Hash the query vector
            2. Retrieve candidates from the same bucket and nearby buckets
               (flip 0-2 bits for multi-probe)
            3. Compute exact cosine similarity on candidates
            4. Return top-k

    Parameters
    ----------
    n_bits : int
        Number of hash bits / random hyperplanes (default: 16)
    n_probe_bits : int
        Number of bits to flip for multi-probe (default: 2)
    """

    def __init__(self, n_bits=16, n_probe_bits=2, seed=42):
        self.n_bits = n_bits
        self.n_probe_bits = n_probe_bits
        self.seed = seed
        self.hyperplanes = []   # list of random unit vectors
        self.buckets = {}       # hash_code (int) -> [(doc_id, vector), ...]
        self.dim = 0

    def _generate_hyperplanes(self, dim):
        """Generate random hyperplanes for hashing."""
        rng = random.Random(self.seed)
        self.hyperplanes = []
        for _ in range(self.n_bits):
            h = [rng.gauss(0, 1) for _ in range(dim)]
            self.hyperplanes.append(vec_normalize(h))

    def _hash(self, vector):
        """Compute LSH hash code for a vector."""
        code = 0
        for i, h in enumerate(self.hyperplanes):
            if dot(vector, h) >= 0:
                code |= (1 << i)
        return code

    def _nearby_codes(self, code):
        """Generate hash codes within n_probe_bits Hamming distance."""
        yield code
        if self.n_probe_bits >= 1:
            for i in range(self.n_bits):
                yield code ^ (1 << i)
        if self.n_probe_bits >= 2:
            for i in range(self.n_bits):
                for j in range(i + 1, self.n_bits):
                    yield code ^ (1 << i) ^ (1 << j)

    def build(self, doc_vectors):
        """
        Build the LSH index from document vectors.

        Parameters
        ----------
        doc_vectors : dict[int, list[float]]
            Mapping from doc_id to vector
        """
        if not doc_vectors:
            return

        first_vec = next(iter(doc_vectors.values()))
        self.dim = len(first_vec)

        self._generate_hyperplanes(self.dim)

        self.buckets = {}
        for doc_id, vec in doc_vectors.items():
            v_norm = vec_normalize(vec)
            code = self._hash(v_norm)
            if code not in self.buckets:
                self.buckets[code] = []
            self.buckets[code].append((doc_id, v_norm))

    def search(self, query_vector, k=10):
        """
        Approximate nearest-neighbor search via LSH.

        Returns list of (similarity, doc_id) sorted descending.
        """
        q_norm = vec_normalize(query_vector)
        query_code = self._hash(q_norm)

        # Collect candidates from nearby buckets
        seen = set()
        scores = []
        for code in self._nearby_codes(query_code):
            for doc_id, vec in self.buckets.get(code, []):
                if doc_id not in seen:
                    seen.add(doc_id)
                    sim = dot(q_norm, vec)
                    scores.append((sim, doc_id))

        scores.sort(key=lambda x: -x[0])
        return scores[:k]

    def stats(self):
        """Return index statistics."""
        sizes = [len(v) for v in self.buckets.values()]
        return {
            'n_bits': self.n_bits,
            'n_buckets_used': len(self.buckets),
            'n_buckets_possible': 2 ** self.n_bits,
            'total_vectors': sum(sizes),
            'min_bucket_size': min(sizes) if sizes else 0,
            'max_bucket_size': max(sizes) if sizes else 0,
            'avg_bucket_size': sum(sizes) / len(sizes) if sizes else 0,
        }


# ============================================================
# LSI + Vector Index Integration
# ============================================================

def retrieve_lsi_with_vector_index(query, lsi_retriever, term_id_map, doc_id_map,
                                   vector_index, k=10):
    """
    LSI retrieval using a vector index for fast nearest-neighbor search.

    Instead of brute-force cosine similarity against all documents,
    uses the vector index (IVF or LSH) to find approximate nearest
    neighbors efficiently.

    Parameters
    ----------
    query : str
        Query string
    lsi_retriever : LSIRetriever
        Pre-built LSI model
    term_id_map : IdMap
        Term ID mapping
    doc_id_map : IdMap
        Document ID mapping
    vector_index : FlatIndex, IVFIndex, or LSHIndex
        Pre-built vector index over LSI document vectors
    k : int
        Number of results

    Returns
    -------
    list[tuple[float, str]]
        Top-k results [(score, doc_path), ...]
    """
    query_terms = query.split()
    q_vec = lsi_retriever.query_vector(query_terms, term_id_map)

    if vec_norm(q_vec) < 1e-10:
        return []

    results = vector_index.search(q_vec, k=k)
    return [(sim, doc_id_map[doc_id]) for sim, doc_id in results if sim > 0]


def save_vector_index(index, filepath):
    """Save a vector index to disk."""
    with open(filepath, 'wb') as f:
        pickle.dump(index, f)

def load_vector_index(filepath):
    """Load a vector index from disk."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)


# ============================================================
# Main: Demo and Benchmarks
# ============================================================

if __name__ == "__main__":
    import time
    from lsi import LSIRetriever
    from bsbi import BSBIIndex
    from compression import VBEPostings

    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    BSBI_instance.load()

    # Load LSI model
    lsi_model_path = os.path.join('index', 'lsi_model.pkl')
    lsi = LSIRetriever(k=150)
    if os.path.exists(lsi_model_path):
        lsi.load(lsi_model_path)
    else:
        lsi.build('main_index', VBEPostings, 'index')
        lsi.save(lsi_model_path)

    doc_vectors = lsi.doc_vectors
    print(f"Document vectors: {len(doc_vectors)} docs, "
          f"{len(next(iter(doc_vectors.values())))} dimensions")

    queries = ["alkylated with radioactive iodoacetate",
               "psychodrama for disturbed children",
               "lipid metabolism in toxemia and normal pregnancy"]

    # ---- Build all index types ----
    print("\n" + "=" * 60)
    print("Vector Index Benchmarks")
    print("=" * 60)

    # Flat Index
    print("\n--- FlatIndex (brute-force) ---")
    flat = FlatIndex()
    t0 = time.time()
    flat.build(doc_vectors)
    print(f"  Build time: {time.time() - t0:.4f}s")

    # IVF Index
    print("\n--- IVFIndex (K-means partitioning) ---")
    ivf = IVFIndex(n_clusters=16, nprobe=6, n_iter=20)
    t0 = time.time()
    ivf.build(doc_vectors)
    print(f"  Build time: {time.time() - t0:.4f}s")
    stats = ivf.stats()
    print(f"  Clusters: {stats['n_clusters']}, "
          f"avg size: {stats['avg_cluster_size']:.1f}, "
          f"nprobe: {stats['nprobe']}")

    # LSH Index
    print("\n--- LSHIndex (random hyperplanes) ---")
    lsh = LSHIndex(n_bits=8, n_probe_bits=2)
    t0 = time.time()
    lsh.build(doc_vectors)
    print(f"  Build time: {time.time() - t0:.4f}s")
    stats = lsh.stats()
    print(f"  Bits: {stats['n_bits']}, "
          f"buckets used: {stats['n_buckets_used']}/{stats['n_buckets_possible']}, "
          f"avg bucket: {stats['avg_bucket_size']:.1f}")

    # ---- Query benchmarks ----
    print("\n" + "=" * 60)
    print("Retrieval Comparison")
    print("=" * 60)

    for query in queries:
        print(f"\nQuery: {query}")

        for name, idx in [("Flat", flat), ("IVF", ivf), ("LSH", lsh)]:
            t0 = time.time()
            results = retrieve_lsi_with_vector_index(
                query, lsi, BSBI_instance.term_id_map,
                BSBI_instance.doc_id_map, idx, k=10)
            elapsed = time.time() - t0
            print(f"\n  {name} ({elapsed*1000:.2f}ms):")
            for score, doc in results[:5]:
                print(f"    {doc:30} {score:.4f}")

    # ---- Recall comparison (IVF/LSH vs Flat ground truth) ----
    print("\n" + "=" * 60)
    print("Recall@10 (vs FlatIndex ground truth)")
    print("=" * 60)

    for query in queries:
        flat_results = set(doc for _, doc in
            retrieve_lsi_with_vector_index(
                query, lsi, BSBI_instance.term_id_map,
                BSBI_instance.doc_id_map, flat, k=10))

        for name, idx in [("IVF", ivf), ("LSH", lsh)]:
            idx_results = set(doc for _, doc in
                retrieve_lsi_with_vector_index(
                    query, lsi, BSBI_instance.term_id_map,
                    BSBI_instance.doc_id_map, idx, k=10))
            recall = len(flat_results & idx_results) / len(flat_results) if flat_results else 0
            print(f"  {name} recall@10 for '{query[:40]}...': {recall:.2f}")

"""
BONUS: Latent Semantic Indexing (LSI)

LSI uses Singular Value Decomposition (SVD) to discover latent semantic
structure in the term-document matrix. Documents and queries are projected
into a lower-dimensional "concept" space, enabling retrieval based on
semantic similarity rather than exact term matching.

Pipeline:
    1. Build TF-IDF weighted term-document sparse matrix from the inverted index
    2. Compute truncated SVD (rank-k approximation) using randomized SVD
    3. Project queries into the latent space
    4. Rank documents by cosine similarity in the latent space

Implementation uses only Python standard library (no numpy/scipy).
"""

import math
import random
import pickle
import os

from index import InvertedIndexReader
from compression import VBEPostings


# ============================================================
# Pure-Python Linear Algebra Helpers
# ============================================================

def dot(a, b):
    """Dot product of two vectors (lists)."""
    return sum(x * y for x, y in zip(a, b))


def vec_norm(v):
    """L2 norm of a vector."""
    return math.sqrt(sum(x * x for x in v))


def vec_normalize(v):
    """Return unit vector."""
    n = vec_norm(v)
    if n < 1e-15:
        return [0.0] * len(v)
    return [x / n for x in v]


def vec_add(a, b):
    """Element-wise addition."""
    return [x + y for x, y in zip(a, b)]


def vec_scale(v, s):
    """Scale vector by scalar."""
    return [x * s for x in v]


def vec_sub(a, b):
    """Element-wise subtraction."""
    return [x - y for x, y in zip(a, b)]


def cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    d = dot(a, b)
    na = vec_norm(a)
    nb = vec_norm(b)
    if na < 1e-15 or nb < 1e-15:
        return 0.0
    return d / (na * nb)


class SparseMatrix:
    """
    Sparse matrix in CSR-like format using Python dicts.

    Stores non-zero entries as {row: {col: value}}.
    Supports matrix-vector multiplication and transpose operations
    needed for SVD computation.
    """
    def __init__(self, n_rows=0, n_cols=0):
        self.data = {}  # row -> {col: value}
        self.n_rows = n_rows
        self.n_cols = n_cols

    def set(self, row, col, value):
        if abs(value) < 1e-15:
            return
        if row not in self.data:
            self.data[row] = {}
        self.data[row][col] = value
        self.n_rows = max(self.n_rows, row + 1)
        self.n_cols = max(self.n_cols, col + 1)

    def get(self, row, col):
        return self.data.get(row, {}).get(col, 0.0)

    def mat_vec(self, v):
        """Multiply this matrix (m x n) by vector v (n x 1) -> (m x 1)."""
        result = [0.0] * self.n_rows
        for row, cols in self.data.items():
            s = 0.0
            for col, val in cols.items():
                s += val * v[col]
            result[row] = s
        return result

    def transpose_mat_vec(self, v):
        """Multiply A^T (n x m) by vector v (m x 1) -> (n x 1)."""
        result = [0.0] * self.n_cols
        for row, cols in self.data.items():
            if abs(v[row]) < 1e-15:
                continue
            for col, val in cols.items():
                result[col] += val * v[row]
        return result

    def mat_dense(self, B):
        """
        Multiply sparse matrix A (m x n) by dense matrix B (n x k).
        B is list of column vectors, each of length n.
        Returns list of k column vectors, each of length m.
        """
        k = len(B)
        result = []
        for j in range(k):
            result.append(self.mat_vec(B[j]))
        return result

    def transpose_mat_dense(self, B):
        """
        Multiply A^T (n x m) by dense matrix B (m x k).
        B is list of k column vectors, each of length m.
        Returns list of k column vectors, each of length n.
        """
        k = len(B)
        result = []
        for j in range(k):
            result.append(self.transpose_mat_vec(B[j]))
        return result


def modified_gram_schmidt(vectors):
    """
    QR decomposition via Modified Gram-Schmidt.

    Parameters
    ----------
    vectors : list of list[float]
        List of column vectors to orthonormalize

    Returns
    -------
    list of list[float]
        Orthonormal basis vectors
    """
    Q = []
    for v in vectors:
        u = list(v)
        for q in Q:
            proj = dot(u, q)
            u = vec_sub(u, vec_scale(q, proj))
        n = vec_norm(u)
        if n > 1e-10:
            Q.append(vec_scale(u, 1.0 / n))
    return Q


def randomized_svd(A, k, n_oversamples=10, n_power_iter=2, seed=42):
    """
    Randomized Truncated SVD (Halko-Martinsson-Tropp algorithm).

    Computes an approximate rank-k SVD of sparse matrix A.

    Steps:
        1. Generate random Gaussian matrix Omega (n x (k+p))
        2. Form Y = A * Omega
        3. Power iteration: Y = (A * A^T)^q * Y for better approximation
        4. QR factorize Y -> Q
        5. Form B = Q^T * A (small dense matrix)
        6. Compute SVD of B via eigendecomposition of B * B^T

    Parameters
    ----------
    A : SparseMatrix
        The m x n matrix to decompose
    k : int
        Number of singular values/vectors to compute
    n_oversamples : int
        Oversampling parameter for better approximation
    n_power_iter : int
        Number of power iterations
    seed : int
        Random seed for reproducibility

    Returns
    -------
    U : list of list[float]
        Left singular vectors (m x k), stored as k column vectors
    sigma : list[float]
        Top-k singular values
    Vt : list of list[float]
        Right singular vectors (k x n), stored as k row vectors
    """
    rng = random.Random(seed)
    m, n = A.n_rows, A.n_cols
    l = min(k + n_oversamples, min(m, n))

    # Step 1: Random Gaussian matrix Omega (n x l), stored as l columns of length n
    Omega = [[rng.gauss(0, 1) for _ in range(n)] for _ in range(l)]

    # Step 2: Y = A * Omega (m x l)
    Y = A.mat_dense(Omega)

    # Step 3: Power iteration for better approximation
    for _ in range(n_power_iter):
        # Y = A * (A^T * Y)
        Z = A.transpose_mat_dense(Y)  # n x l
        Z = modified_gram_schmidt(Z)
        Y = A.mat_dense(Z)            # m x l

    # Step 4: QR factorization of Y
    Q = modified_gram_schmidt(Y)  # orthonormal basis, up to l vectors of length m
    actual_k = min(k, len(Q))
    Q = Q[:actual_k]

    if actual_k == 0:
        return [], [], []

    # Step 5: B = Q^T * A (actual_k x n)
    # B[i][j] = dot(Q[i], A[:, j]) = (Q[i]^T * A)_j
    # This is equivalent to computing Q^T * A
    # For each row i of B: B[i] = Q[i]^T * A, which is A^T * Q[i]
    B = []
    for i in range(actual_k):
        B.append(A.transpose_mat_vec(Q[i]))

    # Step 6: SVD of small dense matrix B (actual_k x n)
    # Compute B * B^T (actual_k x actual_k)
    BBt = [[0.0] * actual_k for _ in range(actual_k)]
    for i in range(actual_k):
        for j in range(i, actual_k):
            val = dot(B[i], B[j])
            BBt[i][j] = val
            BBt[j][i] = val

    # Eigendecomposition of BBt via Jacobi iteration
    eigenvalues, eigenvectors = jacobi_eigen(BBt, max_iter=200)

    # Sort by eigenvalue descending
    indices = sorted(range(actual_k), key=lambda i: -eigenvalues[i])
    eigenvalues = [eigenvalues[i] for i in indices]
    eigenvectors = [[eigenvectors[r][c] for c in indices] for r in range(actual_k)]

    # Singular values = sqrt(eigenvalues)
    sigma = []
    for ev in eigenvalues:
        if ev > 1e-10:
            sigma.append(math.sqrt(ev))
        else:
            sigma.append(0.0)

    # U_B = eigenvectors of BBt (columns)
    # U = Q * U_B
    U = []
    for j in range(actual_k):
        u_col = [0.0] * m
        for i in range(actual_k):
            ev_ij = eigenvectors[i][j]
            if abs(ev_ij) > 1e-15:
                for r in range(m):
                    u_col[r] += Q[i][r] * ev_ij
        U.append(vec_normalize(u_col))

    # V = B^T * U_B * Sigma^{-1}
    # V[j] = (1/sigma[j]) * B^T * eigenvectors[:, j]
    Vt = []
    for j in range(actual_k):
        if sigma[j] < 1e-10:
            Vt.append([0.0] * n)
            continue
        v_row = [0.0] * n
        for i in range(actual_k):
            ev_ij = eigenvectors[i][j]
            if abs(ev_ij) > 1e-15:
                for c in range(n):
                    v_row[c] += B[i][c] * ev_ij
        v_row = vec_scale(v_row, 1.0 / sigma[j])
        Vt.append(vec_normalize(v_row))

    return U, sigma, Vt


def jacobi_eigen(A, max_iter=200, tol=1e-10):
    """
    Jacobi eigenvalue algorithm for symmetric matrices.

    Parameters
    ----------
    A : list of list[float]
        Symmetric matrix (n x n)
    max_iter : int
        Maximum iterations

    Returns
    -------
    eigenvalues : list[float]
    eigenvectors : list of list[float] (n x n)
    """
    n = len(A)
    # Copy A
    M = [row[:] for row in A]
    # Initialize eigenvectors as identity
    V = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for iteration in range(max_iter):
        # Find largest off-diagonal element
        max_val = 0.0
        p, q = 0, 1
        for i in range(n):
            for j in range(i + 1, n):
                if abs(M[i][j]) > max_val:
                    max_val = abs(M[i][j])
                    p, q = i, j

        if max_val < tol:
            break

        # Compute rotation
        if abs(M[p][p] - M[q][q]) < 1e-15:
            theta = math.pi / 4
        else:
            theta = 0.5 * math.atan2(2 * M[p][q], M[p][p] - M[q][q])

        c = math.cos(theta)
        s = math.sin(theta)

        # Apply rotation
        # Update rows/cols p and q
        new_pp = c * c * M[p][p] + 2 * s * c * M[p][q] + s * s * M[q][q]
        new_qq = s * s * M[p][p] - 2 * s * c * M[p][q] + c * c * M[q][q]
        new_pq = 0.0  # This is the point of Jacobi

        for i in range(n):
            if i != p and i != q:
                new_ip = c * M[i][p] + s * M[i][q]
                new_iq = -s * M[i][p] + c * M[i][q]
                M[i][p] = M[p][i] = new_ip
                M[i][q] = M[q][i] = new_iq

        M[p][p] = new_pp
        M[q][q] = new_qq
        M[p][q] = M[q][p] = new_pq

        # Update eigenvectors
        for i in range(n):
            new_vp = c * V[i][p] + s * V[i][q]
            new_vq = -s * V[i][p] + c * V[i][q]
            V[i][p] = new_vp
            V[i][q] = new_vq

    eigenvalues = [M[i][i] for i in range(n)]
    return eigenvalues, V


# ============================================================
# LSI Retriever
# ============================================================

class LSIRetriever:
    """
    Latent Semantic Indexing retriever.

    Builds a TF-IDF term-document matrix from the inverted index,
    computes truncated SVD, and enables semantic similarity search
    in the latent concept space.

    The key insight of LSI is that synonymous terms will have similar
    representations in the latent space, enabling retrieval of documents
    that don't share exact terms with the query but are semantically related.

    Attributes
    ----------
    k : int
        Number of latent dimensions (concepts)
    U : list of list[float]
        Left singular vectors (term space)
    sigma : list[float]
        Singular values
    Vt : list of list[float]
        Right singular vectors (document space)
    doc_vectors : list of list[float]
        Document representations in latent space (Sigma * Vt)
    idf : dict
        IDF weights for terms
    term_to_row : dict
        Mapping from term_id to matrix row index
    doc_to_col : dict
        Mapping from doc_id to matrix column index
    """

    def __init__(self, k=100):
        """
        Parameters
        ----------
        k : int
            Number of latent dimensions. Typical values: 50-300.
            Higher k preserves more information but is slower and may
            retain noise. Lower k provides more aggressive dimensionality
            reduction and better generalization.
        """
        self.k = k
        self.U = None
        self.sigma = None
        self.Vt = None
        self.doc_vectors = None
        self.idf = {}
        self.term_to_row = {}
        self.row_to_term = {}
        self.doc_to_col = {}
        self.col_to_doc = {}

    def build(self, index_name, postings_encoding, output_dir):
        """
        Build the LSI model from the inverted index.

        Steps:
            1. Read the inverted index to build a TF-IDF sparse matrix
            2. Compute truncated SVD
            3. Pre-compute document vectors in latent space

        Parameters
        ----------
        index_name : str
            Name of the inverted index
        postings_encoding : class
            Postings encoding class (e.g., VBEPostings)
        output_dir : str
            Directory containing the index files
        """
        print("Building LSI model...")

        # Step 1: Build TF-IDF sparse matrix
        print("  Building TF-IDF matrix...")
        A = SparseMatrix()
        row_idx = 0
        col_set = set()

        with InvertedIndexReader(index_name, postings_encoding, directory=output_dir) as reader:
            N = len(reader.doc_length)
            doc_ids = sorted(reader.doc_length.keys())

            # Map doc_ids to column indices
            for i, doc_id in enumerate(doc_ids):
                self.doc_to_col[doc_id] = i
                self.col_to_doc[i] = doc_id

            n_docs = len(doc_ids)

            # Iterate through all terms
            reader.reset()
            for term_id, postings, tf_list in reader:
                df = len(postings)
                idf_val = math.log(N / df) if df > 0 else 0

                self.idf[term_id] = idf_val
                self.term_to_row[term_id] = row_idx
                self.row_to_term[row_idx] = term_id

                for i, doc_id in enumerate(postings):
                    tf = tf_list[i]
                    if tf > 0:
                        # TF-IDF weight: log(1 + tf) * idf
                        weight = (1 + math.log(tf)) * idf_val
                        col = self.doc_to_col[doc_id]
                        A.set(row_idx, col, weight)
                        col_set.add(col)

                row_idx += 1

        A.n_rows = row_idx
        A.n_cols = n_docs
        n_terms = row_idx

        print(f"  Matrix size: {n_terms} terms x {n_docs} docs")

        # Step 2: Truncated SVD
        actual_k = min(self.k, min(n_terms, n_docs))
        print(f"  Computing SVD with k={actual_k}...")
        self.U, self.sigma, self.Vt = randomized_svd(A, actual_k,
                                                      n_oversamples=5,
                                                      n_power_iter=2)

        # Step 3: Pre-compute document vectors in latent space
        # doc_vector[j] = [sigma[i] * Vt[i][j] for each dimension i]
        print("  Computing document vectors...")
        self.doc_vectors = {}
        actual_k = len(self.sigma)
        for doc_id, col in self.doc_to_col.items():
            vec = [self.sigma[i] * self.Vt[i][col] for i in range(actual_k)]
            if vec_norm(vec) > 1e-10:
                self.doc_vectors[doc_id] = vec

        print(f"  LSI model built: {actual_k} latent dimensions, "
              f"{len(self.doc_vectors)} document vectors")

    def query_vector(self, query_terms, term_id_map):
        """
        Project a query into the latent space.

        q_latent = Sigma^{-1} * U^T * q_tfidf

        Parameters
        ----------
        query_terms : list[str]
            Tokenized query terms
        term_id_map : IdMap
            Mapping from term strings to term IDs

        Returns
        -------
        list[float]
            Query vector in latent space
        """
        actual_k = len(self.sigma)

        # Build query TF-IDF vector (sparse)
        query_tf = {}
        for term_str in query_terms:
            term_id = term_id_map[term_str]
            query_tf[term_id] = query_tf.get(term_id, 0) + 1

        # Project: q_latent[i] = (1/sigma[i]) * sum_j(U[i][j] * q_tfidf[j])
        q_latent = [0.0] * actual_k
        for term_id, tf in query_tf.items():
            if term_id in self.term_to_row and term_id in self.idf:
                row = self.term_to_row[term_id]
                weight = (1 + math.log(tf)) * self.idf[term_id]
                for i in range(actual_k):
                    if self.sigma[i] > 1e-10:
                        q_latent[i] += self.U[i][row] * weight / self.sigma[i]

        return q_latent

    def retrieve(self, query, term_id_map, doc_id_map, k=10):
        """
        Retrieve top-k documents using LSI cosine similarity.

        Parameters
        ----------
        query : str
            Query string (space-separated tokens)
        term_id_map : IdMap
            Term ID mapping
        doc_id_map : IdMap
            Document ID mapping
        k : int
            Number of documents to retrieve

        Returns
        -------
        list[tuple[float, str]]
            Top-k (score, document_path) pairs sorted by score descending
        """
        query_terms = query.split()
        q_vec = self.query_vector(query_terms, term_id_map)

        if vec_norm(q_vec) < 1e-10:
            return []

        # Compute cosine similarity with all documents
        scores = []
        for doc_id, doc_vec in self.doc_vectors.items():
            sim = cosine_sim(q_vec, doc_vec)
            if sim > 0:
                scores.append((sim, doc_id))

        # Sort by similarity descending, take top-k
        scores.sort(key=lambda x: -x[0])
        results = [(score, doc_id_map[doc_id]) for score, doc_id in scores[:k]]
        return results

    def save(self, filepath):
        """Save LSI model to disk."""
        model = {
            'k': self.k,
            'U': self.U,
            'sigma': self.sigma,
            'Vt': self.Vt,
            'doc_vectors': self.doc_vectors,
            'idf': self.idf,
            'term_to_row': self.term_to_row,
            'row_to_term': self.row_to_term,
            'doc_to_col': self.doc_to_col,
            'col_to_doc': self.col_to_doc,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model, f)
        print(f"  LSI model saved to {filepath}")

    def load(self, filepath):
        """Load LSI model from disk."""
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        self.k = model['k']
        self.U = model['U']
        self.sigma = model['sigma']
        self.Vt = model['Vt']
        self.doc_vectors = model['doc_vectors']
        self.idf = model['idf']
        self.term_to_row = model['term_to_row']
        self.row_to_term = model['row_to_term']
        self.doc_to_col = model['doc_to_col']
        self.col_to_doc = model['col_to_doc']
        print(f"  LSI model loaded from {filepath}")


if __name__ == "__main__":
    from bsbi import BSBIIndex

    # Build LSI model
    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=VBEPostings,
                              output_dir='index')
    BSBI_instance.load()

    lsi = LSIRetriever(k=150)
    lsi.build('main_index', VBEPostings, 'index')
    lsi.save(os.path.join('index', 'lsi_model.pkl'))

    # Test retrieval
    queries = ["alkylated with radioactive iodoacetate",
               "psychodrama for disturbed children",
               "lipid metabolism in toxemia and normal pregnancy"]

    print("\n" + "=" * 60)
    print("LSI Retrieval Results")
    print("=" * 60)
    for query in queries:
        print(f"\nQuery: {query}")
        results = lsi.retrieve(query, BSBI_instance.term_id_map,
                              BSBI_instance.doc_id_map, k=10)
        for score, doc in results:
            print(f"  {doc:30} {score:.4f}")

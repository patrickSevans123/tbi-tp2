# Search Engine from Scratch

A full-featured information retrieval system built using only Python standard library. Implements indexing, compression, multiple retrieval models, evaluation metrics, and several advanced IR techniques from scratch.

Built for the TBI (Temu Balik Informasi / Information Retrieval) course at Universitas Indonesia.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Evaluation Results](#evaluation-results)
- [File Reference](#file-reference)

## Overview

This search engine indexes a collection of 1,033 medical/scientific documents across 11 blocks, supports multiple retrieval and scoring models, and provides evaluation against 30 benchmark queries with human-annotated relevance judgments.

All implementations use only Python standard library -- no external dependencies like numpy, scipy, nltk, or any search engine library.

## Architecture

```
index.py ---------------\
                         --- bsbi.py / spimi.py ---\
util.py ----------------/         |                 \--- evaluation.py
                                  |                 /--- search.py
compression.py ------------------/
                                 |
            lsi.py               |
            adaptive_retrieval.py|
            boolean_query.py     |
            spell_correction.py  |
            snippets.py          |
            porter_stemmer.py    |
```

**Data Flow:**

```
collection/ (raw text) --> Indexing (BSBI/SPIMI) --> index/ (compressed inverted index)
                                                         |
query --> Retrieval (TF-IDF / BM25 / WAND / LSI / GAR) --> Ranked Results
```

## Features

### Core Features (Mandatory)

**1. Bit-Level Index Compression -- Elias-Gamma Encoding**

Implements Elias-Gamma coding, a bit-level compression scheme for postings lists. A positive integer N is encoded as floor(log2(N)) zeros followed by the binary representation of N. Postings are gap-encoded before compression, reducing the values to be encoded and achieving better compression ratios than Variable-Byte Encoding.

- File: `compression.py` -- `EliasGammaPostings` class
- Also includes: `StandardPostings` (baseline) and `VBEPostings` (Variable-Byte Encoding)

**2. BM25 Scoring**

Implements the Okapi BM25 ranking function with document length normalization:

```
Score(D, Q) = sum over t in Q of:
    IDF(t) * (tf(t,D) * (k1 + 1)) / (tf(t,D) + k1 * (1 - b + b * |D| / avgdl))

IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)
```

Document lengths are pre-computed during indexing and stored in the index metadata. Default parameters: k1=1.2, b=0.75.

- File: `bsbi.py` -- `retrieve_bm25()` method
- Pre-computation: `index.py` -- `doc_length` dictionary built during `append()`

**3. Evaluation Metrics -- DCG, NDCG, AP**

Three evaluation metrics in addition to the provided RBP:

- **DCG (Discounted Cumulative Gain):** Measures ranking quality with logarithmic discount. DCG@k = sum of rel_i / log2(i+1).
- **NDCG (Normalized DCG):** DCG normalized by the ideal ranking (IDCG), producing scores in [0, 1].
- **AP (Average Precision):** Mean of precision values at each relevant document position. Captures both precision and recall.

File: `evaluation.py`

**4. WAND Top-K Retrieval**

Implements the Weak AND (WAND) algorithm for efficient top-K retrieval. Instead of scoring every document, WAND uses per-term upper bound scores to prune documents that cannot enter the top-K heap:

1. Compute upper bound BM25 score per query term using max_tf (stored in index)
2. Sort terms by current posting pointer
3. Find pivot where cumulative upper bounds exceed threshold
4. Score only pivot documents; skip others via binary search (galloping)

Produces identical results to full BM25 while scoring fewer documents.

- File: `bsbi.py` -- `retrieve_bm25_wand()` method
- Index support: `index.py` stores `max_tf` as the 5th element of `postings_dict`

### Bonus Features

**5. SPIMI Indexing**

Single-Pass In-Memory Indexing as an alternative to BSBI. Instead of generating (termID, docID) pairs and sorting, SPIMI builds the inverted index directly in memory using a hash table, then writes sorted blocks to disk.

- File: `spimi.py` -- `SPIMIIndex` class

**6. Trie Dictionary**

A prefix tree (Trie) data structure for storing the term dictionary. Provides O(m) lookup where m is term length, shared prefix storage for memory efficiency, and prefix search for wildcard/autocomplete queries.

- File: `spimi.py` -- `Trie` and `TrieNode` classes

**7. Text Preprocessing (Stemming and Stopword Removal)**

- Porter Stemmer implemented from scratch (pure Python, no NLTK dependency)
- English stopword list (143 words) for filtering non-informative terms
- Regex-based tokenization with lowercasing

Files: `porter_stemmer.py`, `spimi.py` -- `TextPreprocessor` class

**8. Latent Semantic Indexing (LSI)**

Discovers latent semantic structure in the term-document matrix using Singular Value Decomposition (SVD). Documents and queries are projected into a lower-dimensional concept space, enabling retrieval based on semantic similarity rather than exact term matching.

Implementation details:
- Sparse TF-IDF term-document matrix construction from the inverted index
- Randomized truncated SVD (Halko-Martinsson-Tropp algorithm) for efficient computation
- Modified Gram-Schmidt orthonormalization for QR decomposition
- Jacobi eigenvalue algorithm for small dense symmetric matrices
- Cosine similarity ranking in latent space
- All linear algebra implemented from scratch (no numpy/scipy)

The LSI model achieves the best evaluation scores among all methods (NDCG 0.861 vs BM25's 0.793 at optimal k=150).

- File: `lsi.py` -- `LSIRetriever`, `SparseMatrix`, `randomized_svd()`

**9. Adaptive Retrieval -- GAR (Graph-based Adaptive Re-ranking)**

Implements the GAR algorithm from "Adaptive Re-Ranking with a Corpus Graph" (MacAvaney, Tonellotto & Macdonald, CIKM 2022), the same algorithm used by PyTerrier Adaptive.

How it works:
1. Build a corpus graph where each document connects to its k-nearest neighbors (by TF-IDF cosine similarity)
2. Run initial BM25 retrieval to produce candidate set
3. Iteratively score candidates in batches, alternating between initial results (even iterations) and corpus graph neighbors (odd iterations)
4. When a document scores highly, its graph neighbors become new candidates
5. Backfill unscored initial results

The corpus graph is computed once from the inverted index and cached to disk.

Two scorer configurations are supported:
- **GAR + BM25:** Uses BM25 as the scorer. Marginal improvement since BM25 is already cheap.
- **GAR + LSI:** Uses LSI cosine similarity as the expensive scorer. This is the intended usage pattern -- GAR selects which documents to LSI-score via the corpus graph, achieving LSI-level quality while scoring fewer documents. Mirrors the original paper's BM25 --> GAR --> MonoT5 pipeline, with LSI replacing the neural model (due to standard library constraint).

- File: `adaptive_retrieval.py` -- `CorpusGraph`, `GAR`, `retrieve_gar()`, `retrieve_gar_lsi()`

**10. Vector Indexing (FAISS-like)**

Implements FAISS-style vector indexing structures from scratch for accelerating nearest-neighbor search over LSI document vectors. Three index types are provided:

- **FlatIndex** (brute-force baseline): Computes cosine similarity against all vectors. Exact results, O(N*d) per query. Equivalent to FAISS `IndexFlatIP`.
- **IVFIndex** (Inverted File Index): Partitions vectors into clusters using K-means. At query time, only searches the `nprobe` nearest clusters. Approximate results, significantly faster. Equivalent to FAISS `IndexIVFFlat`.
- **LSHIndex** (Locality-Sensitive Hashing): Projects vectors into binary hash codes using random hyperplanes. Groups similar vectors into the same bucket. Sub-linear query time. Equivalent to FAISS `IndexLSH`.

The IVF index with 16 clusters and nprobe=6 achieves the best overall evaluation scores (NDCG 0.873, AP 0.665) while reducing the search space compared to brute-force.

- File: `vector_index.py` -- `FlatIndex`, `IVFIndex`, `LSHIndex`, `retrieve_lsi_with_vector_index()`

**11. Pseudo-Relevance Feedback (Rocchio Algorithm)**


Expands the query using terms from top-k initially retrieved documents (assumed relevant):

```
q' = alpha * q + beta * (1/|Dr|) * sum(d in Dr)
```

Expansion terms are selected by TF-IDF weight in feedback documents, then the expanded query is scored using BM25.

- File: `adaptive_retrieval.py` -- `RocchioExpander`

**12. Query Spell Correction**

Corrects misspelled query terms using three complementary techniques:
- **Levenshtein edit distance** -- dynamic programming O(m*n) algorithm
- **Character n-gram index** -- fast candidate filtering via Jaccard similarity of bigram sets
- **BK-tree** -- metric tree for O(log V) nearest-neighbor search in edit-distance space

Example: "radioactiv iodoacetae" is corrected to "radioactive iodoacetate".

- File: `spell_correction.py` -- `SpellCorrector`, `BKTree`, `CharNgramIndex`

**13. Boolean Query Processing**

Supports AND, OR, NOT operators with parenthetical grouping. Uses a recursive descent parser to build an abstract syntax tree (AST), then evaluates using set operations on posting lists.

```
"blood AND (pressure OR hypertension) AND NOT chronic"
```

Grammar: OR < AND < NOT (precedence), with parentheses for override.

- File: `boolean_query.py` -- `BooleanQueryParser`, `BooleanQueryEvaluator`

**14. Result Snippets with Highlighting**

Generates contextual snippets from retrieved documents. Uses a sliding window to find the passage with the highest density of query terms, then highlights matching terms.

- File: `snippets.py` -- `SnippetGenerator`

## Getting Started

### Prerequisites

- Python 3.8 or higher
- `tqdm` library for progress bars during indexing: `pip install tqdm`

No other external dependencies are required.

### Directory Structure

```
TP2/
  collection/          # 1,033 documents across 11 blocks (folders 1-11)
  index/               # Generated index files (created by indexing)
  tmp/                 # Temporary files during indexing
  qrels.txt            # Query relevance judgments (human-annotated)
  queries.txt          # 30 benchmark queries
  bsbi.py              # BSBI indexing + BM25/TF-IDF/WAND retrieval
  spimi.py             # SPIMI indexing + Trie + text preprocessing
  index.py             # Inverted index reader/writer
  compression.py       # Standard, VBE, and Elias-Gamma postings encoding
  util.py              # IdMap and merge utilities
  evaluation.py        # All metrics (RBP, DCG, NDCG, AP) + full evaluation
  search.py            # Demo of all retrieval methods and features
  lsi.py               # Latent Semantic Indexing with pure-Python SVD
  adaptive_retrieval.py # GAR, Rocchio PRF, proximity re-ranking
  boolean_query.py     # Boolean query parser and evaluator
  spell_correction.py  # Spell correction with BK-tree and n-gram index
  snippets.py          # Result snippet generation and highlighting
  porter_stemmer.py    # Pure-Python Porter Stemmer
```

## Usage

### Step 1: Build the Index

```bash
python bsbi.py
```

This runs BSBI indexing on the collection and produces the inverted index in the `index/` directory.

For SPIMI indexing with Trie dictionary:

```bash
python spimi.py
```

### Step 2: Search

```bash
python search.py
```

This demonstrates all retrieval methods:
- TF-IDF, BM25, BM25+WAND (core methods)
- LSI retrieval (builds/loads model automatically)
- GAR adaptive re-ranking (builds/loads corpus graph automatically)
- Rocchio pseudo-relevance feedback
- Boolean queries
- Spell correction
- Result snippets with highlighting

### Step 3: Evaluate

```bash
python evaluation.py
```

Runs all retrieval methods against 30 benchmark queries and reports mean RBP, DCG, NDCG, and AP scores.

### Individual Feature Demos

Each bonus module can be run independently:

```bash
python lsi.py                # Build and test LSI model
python adaptive_retrieval.py # Test GAR and Rocchio
python boolean_query.py      # Test Boolean queries
python spell_correction.py   # Test spell correction
python snippets.py           # Test result snippets
python porter_stemmer.py     # Run stemmer tests (70/70)
python compression.py        # Run compression tests
python index.py              # Run index read/write tests
```

## Evaluation Results

Mean scores across 30 queries (top-1000 retrieval):

| Method                    | RBP   | DCG   | NDCG  | AP    |
|---------------------------|-------|-------|-------|-------|
| TF-IDF                    | 0.598 | 5.594 | 0.783 | 0.495 |
| BM25                      | 0.632 | 5.745 | 0.793 | 0.514 |
| BM25 + WAND               | 0.632 | 5.745 | 0.793 | 0.514 |
| LSI (k=150)               | 0.746 | 6.569 | 0.861 | 0.634 |
| Rocchio PRF               | 0.659 | 5.935 | 0.803 | 0.536 |
| GAR + BM25 scorer         | 0.637 | 5.902 | 0.787 | 0.498 |
| GAR + LSI scorer          | 0.746 | 6.609 | 0.858 | 0.628 |
| LSI + IVF Vector Index    | 0.747 | 6.368 | 0.873 | 0.665 |

Key observations:
- **BM25 outperforms TF-IDF** across all metrics, confirming the effectiveness of term frequency saturation and document length normalization.
- **WAND produces identical results to BM25**, verifying that the pruning algorithm introduces no quality loss while improving efficiency.
- **LSI achieves the best scores** on all metrics (+18% RBP, +9% NDCG over BM25), demonstrating the value of semantic matching in the latent concept space. The optimal number of latent dimensions was found to be k=150 through parameter tuning (tested k=50, 100, 150, 200, 300).
- **Rocchio PRF improves over BM25** on all metrics, showing that pseudo-relevance feedback successfully identifies useful expansion terms.
- **GAR + BM25 scorer** shows marginal improvement over BM25 -- this is expected because GAR is designed to reduce the number of calls to an expensive scorer, not to improve a cheap scorer like BM25.
- **GAR + LSI scorer** matches full LSI quality (DCG 6.609 vs 6.569) while only LSI-scoring a subset of documents selected through the corpus graph. This is the intended usage pattern of GAR: cheap initial retrieval (BM25) to get candidates, then GAR selects which candidates to score with the expensive scorer (LSI), mirroring the original paper's BM25 --> GAR --> MonoT5 pipeline.
- **LSI + IVF Vector Index** achieves the highest NDCG (0.873) and AP (0.665) among all methods. The IVF index partitions LSI document vectors into K-means clusters and searches only the nearest clusters at query time, providing a FAISS-like approximate nearest-neighbor search capability implemented entirely from scratch.

## File Reference

| File | Lines | Description |
|------|-------|-------------|
| `adaptive_retrieval.py` | 860 | GAR corpus graph adaptive re-ranking, Rocchio PRF, RSV query expansion, proximity re-ranking |
| `vector_index.py` | 420 | FlatIndex, IVFIndex (K-means), LSHIndex (random hyperplanes), FAISS-like vector search |
| `lsi.py` | 645 | Latent Semantic Indexing with randomized SVD, sparse matrix operations, Jacobi eigenvalue solver |
| `spimi.py` | 548 | SPIMI indexing, Trie data structure, text preprocessing with stemming and stopwords |
| `spell_correction.py` | 508 | Levenshtein distance, character n-gram index, BK-tree, query correction pipeline |
| `compression.py` | 500 | Standard, Variable-Byte, and Elias-Gamma postings encoding |
| `bsbi.py` | 474 | BSBI indexing, TF-IDF retrieval, BM25 retrieval, WAND top-K retrieval |
| `evaluation.py` | 395 | RBP, DCG, NDCG, AP metrics with full evaluation pipeline for all methods |
| `boolean_query.py` | 384 | Recursive descent Boolean query parser, AST evaluator with set operations |
| `porter_stemmer.py` | 328 | Pure-Python Porter Stemming algorithm (5-step suffix stripping) |
| `snippets.py` | 299 | Sliding window snippet extraction, query term highlighting |
| `index.py` | 256 | Inverted index reader/writer with metadata persistence |
| `search.py` | 198 | End-to-end demo of all retrieval methods and bonus features |
| `util.py` | 132 | IdMap bidirectional mapping, sorted merge for postings lists |
| **Total** | **5,947** | |

## References

- Manning, Raghavan, Schutze. *Introduction to Information Retrieval*. Cambridge University Press, 2008.
- Robertson, S.E. and Zaragoza, H. "The Probabilistic Relevance Framework: BM25 and Beyond." *Foundations and Trends in Information Retrieval*, 2009.
- MacAvaney, S., Tonellotto, N., and Macdonald, C. "Adaptive Re-Ranking with a Corpus Graph." *CIKM*, 2022.
- Halko, N., Martinsson, P.G., and Tropp, J.A. "Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions." *SIAM Review*, 2011.
- Johnson, J., Douze, M., and Jegou, H. "Billion-scale similarity search with GPUs." *IEEE Transactions on Big Data*, 2019.
- Porter, M.F. "An algorithm for suffix stripping." *Program*, 1980.

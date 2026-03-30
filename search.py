"""
Search Engine Demo

Demonstrates all retrieval methods and bonus features:
1. TF-IDF retrieval
2. BM25 retrieval
3. BM25 + WAND Top-K retrieval
4. LSI (Latent Semantic Indexing) retrieval
5. Adaptive retrieval (Rocchio PRF + proximity re-ranking)
6. Boolean queries (AND, OR, NOT)
7. Spell correction
8. Result snippets with highlighting
"""

import os

from bsbi import BSBIIndex
from compression import VBEPostings, EliasGammaPostings
from snippets import SnippetGenerator, format_search_results

# sebelumnya sudah dilakukan indexing
# BSBIIndex hanya sebagai abstraksi untuk index tersebut
BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

snippet_gen = SnippetGenerator(context_words=25, highlight_start="[",
                                highlight_end="]")

queries = ["alkylated with radioactive iodoacetate", \
           "psychodrama for disturbed children", \
           "lipid metabolism in toxemia and normal pregnancy"]

print("=" * 70)
print("SEARCH ENGINE DEMO - All Retrieval Methods")
print("=" * 70)

# ============================================================
# 1-3: Core Retrieval Methods
# ============================================================
for query in queries:
    print(f"\nQuery  : {query}")

    print("\n  --- TF-IDF ---")
    for (score, doc) in BSBI_instance.retrieve_tfidf(query, k = 10):
        print(f"  {doc:30} {score:>.3f}")

    print("\n  --- BM25 ---")
    for (score, doc) in BSBI_instance.retrieve_bm25(query, k = 10):
        print(f"  {doc:30} {score:>.3f}")

    print("\n  --- BM25 + WAND Top-K ---")
    for (score, doc) in BSBI_instance.retrieve_bm25_wand(query, k = 10):
        print(f"  {doc:30} {score:>.3f}")

    print()

# ============================================================
# 4: LSI Retrieval
# ============================================================
print("\n" + "=" * 70)
print("BONUS: LSI (Latent Semantic Indexing) Retrieval")
print("=" * 70)

from lsi import LSIRetriever

lsi_model_path = os.path.join('index', 'lsi_model.pkl')
lsi = LSIRetriever(k=150)

if os.path.exists(lsi_model_path):
    lsi.load(lsi_model_path)
else:
    print("Building LSI model (first time only)...")
    lsi.build('main_index', VBEPostings, 'index')
    lsi.save(lsi_model_path)

BSBI_instance.load()
for query in queries:
    print(f"\nQuery: {query}")
    print("  --- LSI (k=50) ---")
    results = lsi.retrieve(query, BSBI_instance.term_id_map,
                          BSBI_instance.doc_id_map, k=10)
    for score, doc in results:
        print(f"  {doc:30} {score:.4f}")

# ============================================================
# 5: Adaptive Retrieval (GAR + Rocchio PRF + Proximity)
# ============================================================
print("\n" + "=" * 70)
print("BONUS: Adaptive Retrieval")
print("=" * 70)

from adaptive_retrieval import retrieve_adaptive, retrieve_gar, RocchioExpander, CorpusGraph

# --- 5a: Rocchio PRF ---
print("\n--- Rocchio Pseudo-Relevance Feedback ---")
for query in queries:
    print(f"\nQuery: {query}")
    rocchio = RocchioExpander(n_feedback_docs=5, n_expand_terms=15)
    for score, doc in rocchio.retrieve_with_feedback(query, BSBI_instance, k=10):
        print(f"  {doc:30} {score:.3f}")

# --- 5b: GAR (Graph-based Adaptive Re-ranking) ---
print("\n--- GAR: Graph-based Adaptive Re-ranking (CIKM 2022) ---")
graph_path = os.path.join('index', 'corpus_graph.pkl')
cg = CorpusGraph(k=16)
if os.path.exists(graph_path):
    cg.load(graph_path)
else:
    print("Building corpus graph (first time only)...")
    cg.build('main_index', VBEPostings, 'index')
    cg.save(graph_path)

for query in queries:
    print(f"\nQuery: {query}")
    print("  --- GAR (BM25 + Corpus Graph) ---")
    for score, doc in retrieve_gar(query, BSBI_instance, cg, k=10):
        print(f"  {doc:30} {score:.3f}")

# --- 5c: GAR + LSI scorer (the intended adaptive retrieval pattern) ---
from adaptive_retrieval import retrieve_gar_lsi
print("\n--- GAR + LSI scorer (BM25 initial -> GAR -> LSI re-rank) ---")
for query in queries:
    print(f"\nQuery: {query}")
    for score, doc in retrieve_gar_lsi(query, BSBI_instance, cg, lsi, k=10):
        print(f"  {doc:30} {score:.4f}")

# ============================================================
# 6: Boolean Queries
# ============================================================
print("\n" + "=" * 70)
print("BONUS: Boolean Query Processing")
print("=" * 70)

from boolean_query import boolean_search

boolean_queries = [
    "blood AND pressure",
    "blood OR plasma",
    "blood AND NOT pressure",
    "(blood OR plasma) AND protein",
]

for q in boolean_queries:
    results = boolean_search(q, BSBI_instance)
    print(f"\n  '{q}': {len(results)} documents found")
    for doc in results[:3]:
        print(f"    {doc}")
    if len(results) > 3:
        print(f"    ... and {len(results) - 3} more")

# ============================================================
# 7: Spell Correction
# ============================================================
print("\n" + "=" * 70)
print("BONUS: Spell Correction")
print("=" * 70)

from spell_correction import build_spell_corrector_from_index

corrector = build_spell_corrector_from_index(
    'main_index', VBEPostings, 'index', BSBI_instance.term_id_map)

misspelled_queries = [
    "radioactiv iodoacetae",
    "psycoodrama childrn",
    "lipd metabolsm toxmia"
]

for q in misspelled_queries:
    corrected, info = corrector.correct_with_info(q)
    print(f"\n  Original : '{q}'")
    print(f"  Corrected: '{corrected}'")
    for orig, corr, sugg in info:
        print(f"    '{orig}' -> '{corr}'")

    # Search with corrected query
    print(f"  Top-3 BM25 results for corrected query:")
    for score, doc in BSBI_instance.retrieve_bm25(corrected, k=3):
        print(f"    {doc:30} {score:.3f}")

# ============================================================
# 8: Result Snippets
# ============================================================
print("\n" + "=" * 70)
print("BONUS: Search Results with Snippets")
print("=" * 70)

for query in queries:
    print(f"\nQuery: {query}")
    results = BSBI_instance.retrieve_bm25(query, k=5)
    output = format_search_results(results, query, snippet_gen, max_display=5)
    print(output)

print("\n" + "=" * 70)
print("Demo complete!")
print("=" * 70)

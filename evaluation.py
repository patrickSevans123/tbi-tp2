import re
import math
from bsbi import BSBIIndex
from compression import VBEPostings, EliasGammaPostings

######## >>>>> sebuah IR metric: RBP p = 0.8

def rbp(ranking, p = 0.8):
  """ menghitung search effectiveness metric score dengan
      Rank Biased Precision (RBP)

      Parameters
      ----------
      ranking: List[int]
         vektor biner seperti [1, 0, 1, 1, 1, 0]
         gold standard relevansi dari dokumen di rank 1, 2, 3, dst.
         Contoh: [1, 0, 1, 1, 1, 0] berarti dokumen di rank-1 relevan,
                 di rank-2 tidak relevan, di rank-3,4,5 relevan, dan
                 di rank-6 tidak relevan

      Returns
      -------
      Float
        score RBP
  """
  score = 0.
  for i in range(1, len(ranking)):
    pos = i - 1
    score += ranking[pos] * (p ** (i - 1))
  return (1 - p) * score


######## >>>>> IR metric: DCG (Discounted Cumulative Gain)

def dcg(ranking, k = None):
  """
  Menghitung Discounted Cumulative Gain (DCG) pada rank k.

  DCG@k = sum_{i=1}^{k} rel_i / log2(i + 1)

  DCG mengukur kualitas ranking dengan memberikan bobot lebih besar
  pada dokumen relevan yang muncul di posisi atas (early ranks).
  Discount factor menggunakan log2(i+1) dimana i adalah posisi rank
  (dimulai dari 1).

  Parameters
  ----------
  ranking: List[int]
      vektor relevansi, misal [1, 0, 1, 1, 0]
      ranking[0] = relevansi dokumen di rank 1
      ranking[1] = relevansi dokumen di rank 2, dst.
  k : int or None
      Rank cutoff. Jika None, gunakan seluruh ranking.

  Returns
  -------
  Float
      Skor DCG@k
  """
  if k is None:
    k = len(ranking)
  k = min(k, len(ranking))

  score = 0.0
  for i in range(k):
    # i=0 berarti rank 1, i=1 berarti rank 2, dst.
    score += ranking[i] / math.log2(i + 2)  # log2(rank + 1) = log2(i + 2)
  return score


######## >>>>> IR metric: NDCG (Normalized DCG)

def ndcg(ranking, k = None):
  """
  Menghitung Normalized Discounted Cumulative Gain (NDCG) pada rank k.

  NDCG@k = DCG@k / IDCG@k

  dimana IDCG@k adalah DCG@k dari ranking ideal (semua dokumen relevan
  diletakkan di posisi teratas).

  NDCG menormalisasi DCG sehingga nilainya berada di rentang [0, 1],
  memungkinkan perbandingan antar query yang memiliki jumlah dokumen
  relevan berbeda.

  Parameters
  ----------
  ranking: List[int]
      vektor relevansi, misal [1, 0, 1, 1, 0]
  k : int or None
      Rank cutoff. Jika None, gunakan seluruh ranking.

  Returns
  -------
  Float
      Skor NDCG@k (antara 0 dan 1). Mengembalikan 0 jika tidak ada
      dokumen relevan.
  """
  if k is None:
    k = len(ranking)
  k = min(k, len(ranking))

  # Hitung DCG dari ranking yang diberikan
  actual_dcg = dcg(ranking, k)

  # Hitung IDCG (Ideal DCG): urutkan relevansi secara menurun
  ideal_ranking = sorted(ranking, reverse=True)
  ideal_dcg = dcg(ideal_ranking, k)

  if ideal_dcg == 0:
    return 0.0

  return actual_dcg / ideal_dcg


######## >>>>> IR metric: AP (Average Precision)

def ap(ranking):
  """
  Menghitung Average Precision (AP).

  AP = (1 / |relevant docs|) * sum_{k=1}^{n} (Precision@k * rel(k))

  AP mengukur kualitas ranking dengan menghitung rata-rata precision
  pada setiap posisi dimana dokumen relevan ditemukan.

  Precision@k = (jumlah dokumen relevan di rank 1 s.d. k) / k

  AP memperhatikan baik precision maupun recall, dan memberikan bobot
  lebih pada dokumen relevan yang muncul lebih awal.

  Parameters
  ----------
  ranking: List[int]
      vektor biner relevansi, misal [1, 0, 1, 1, 0]

  Returns
  -------
  Float
      Skor Average Precision (antara 0 dan 1). Mengembalikan 0 jika
      tidak ada dokumen relevan di ranking.
  """
  num_relevant = sum(ranking)
  if num_relevant == 0:
    return 0.0

  score = 0.0
  relevant_so_far = 0
  for i in range(len(ranking)):
    if ranking[i] == 1:
      relevant_so_far += 1
      precision_at_k = relevant_so_far / (i + 1)
      score += precision_at_k

  return score / num_relevant


######## >>>>> memuat qrels

def load_qrels(qrel_file = "qrels.txt", max_q_id = 30, max_doc_id = 1033):
  """ memuat query relevance judgment (qrels)
      dalam format dictionary of dictionary
      qrels[query id][document id]

      dimana, misal, qrels["Q3"][12] = 1 artinya Doc 12
      relevan dengan Q3; dan qrels["Q3"][10] = 0 artinya
      Doc 10 tidak relevan dengan Q3.

  """
  qrels = {"Q" + str(i) : {i:0 for i in range(1, max_doc_id + 1)} \
                 for i in range(1, max_q_id + 1)}
  with open(qrel_file) as file:
    for line in file:
      parts = line.strip().split()
      qid = parts[0]
      did = int(parts[1])
      qrels[qid][did] = 1
  return qrels

######## >>>>> EVALUASI !

def _extract_doc_id(doc_path):
  """Extract numeric document ID from file path."""
  match = re.search(r'\/.*\/.*\/(.*)\.txt', doc_path)
  if match:
    return int(match.group(1))
  # Fallback: try backslash paths (Windows)
  match = re.search(r'\\.*\\.*\\(.*)\.txt', doc_path)
  if match:
    return int(match.group(1))
  # Last resort: extract last number
  match = re.search(r'(\d+)\.txt', doc_path)
  if match:
    return int(match.group(1))
  return -1


def _evaluate_ranking(ranking, qrels_for_query):
  """Compute all metrics for a ranking."""
  return {
    'rbp': rbp(ranking),
    'dcg': dcg(ranking),
    'ndcg': ndcg(ranking),
    'ap': ap(ranking)
  }


def _evaluate_method(method_fn, queries_data, qrels, k):
  """
  Evaluate a retrieval method across all queries.

  Parameters
  ----------
  method_fn : callable
      Function that takes (query, k) and returns [(score, doc), ...]
  queries_data : list of (qid, query)
  qrels : dict
  k : int

  Returns
  -------
  dict of metric_name -> mean_score
  """
  scores = {'rbp': [], 'dcg': [], 'ndcg': [], 'ap': []}

  for qid, query in queries_data:
    ranking = []
    for (score, doc) in method_fn(query, k=k):
      did = _extract_doc_id(doc)
      if did >= 0:
        ranking.append(qrels[qid].get(did, 0))
    metrics = _evaluate_ranking(ranking, qrels[qid])
    for m in scores:
      scores[m].append(metrics[m])

  return {m: sum(v) / len(v) if v else 0.0 for m, v in scores.items()}


def _print_method_scores(name, scores):
  """Print scores for a method."""
  print(f"\n--- {name} ---")
  print(f"  RBP  score = {scores['rbp']:.4f}")
  print(f"  DCG  score = {scores['dcg']:.4f}")
  print(f"  NDCG score = {scores['ndcg']:.4f}")
  print(f"  AP   score = {scores['ap']:.4f}")


def eval(qrels, query_file = "queries.txt", k = 1000):
  """
    loop ke semua 30 query, hitung score di setiap query,
    lalu hitung MEAN SCORE over those 30 queries.
    untuk setiap query, kembalikan top-1000 documents

    Evaluasi dilakukan untuk:
    1. TF-IDF scoring
    2. BM25 scoring
    3. BM25 + WAND Top-K scoring
    4. LSI (Latent Semantic Indexing)
    5. Adaptive Retrieval (Rocchio PRF)

    Metrics yang dihitung:
    - RBP (Rank-Biased Precision, p=0.8)
    - DCG (Discounted Cumulative Gain)
    - NDCG (Normalized DCG)
    - AP (Average Precision)
  """
  import os

  BSBI_instance = BSBIIndex(data_dir = 'collection', \
                          postings_encoding = VBEPostings, \
                          output_dir = 'index')

  # Load queries
  queries_data = []
  with open(query_file) as file:
    for qline in file:
      parts = qline.strip().split()
      qid = parts[0]
      query = " ".join(parts[1:])
      queries_data.append((qid, query))

  print("=" * 70)
  print("Hasil evaluasi terhadap 30 queries")
  print("=" * 70)

  # ========== Core methods ==========
  tfidf_scores = _evaluate_method(BSBI_instance.retrieve_tfidf, queries_data, qrels, k)
  _print_method_scores("TF-IDF", tfidf_scores)

  bm25_scores = _evaluate_method(BSBI_instance.retrieve_bm25, queries_data, qrels, k)
  _print_method_scores("BM25", bm25_scores)

  wand_scores = _evaluate_method(BSBI_instance.retrieve_bm25_wand, queries_data, qrels, k)
  _print_method_scores("BM25 + WAND Top-K", wand_scores)

  # ========== BONUS: LSI ==========
  try:
    from lsi import LSIRetriever
    BSBI_instance.load()
    lsi_model_path = os.path.join('index', 'lsi_model.pkl')
    lsi = LSIRetriever(k=150)
    if os.path.exists(lsi_model_path):
      lsi.load(lsi_model_path)
    else:
      lsi.build('main_index', VBEPostings, 'index')
      lsi.save(lsi_model_path)

    def lsi_retrieve(query, k=10):
      return lsi.retrieve(query, BSBI_instance.term_id_map,
                         BSBI_instance.doc_id_map, k=k)

    lsi_scores = _evaluate_method(lsi_retrieve, queries_data, qrels, k)
    _print_method_scores("LSI (k=150)", lsi_scores)
  except Exception as e:
    print(f"\n--- LSI --- (skipped: {e})")

  # ========== BONUS: Adaptive Retrieval (Rocchio) ==========
  try:
    from adaptive_retrieval import RocchioExpander
    rocchio = RocchioExpander(alpha=2.0, beta=0.5, n_feedback_docs=3, n_expand_terms=8)

    def rocchio_retrieve(query, k=10):
      return rocchio.retrieve_with_feedback(query, BSBI_instance, k=k)

    rocchio_scores = _evaluate_method(rocchio_retrieve, queries_data, qrels, k)
    _print_method_scores("Adaptive (Rocchio PRF)", rocchio_scores)
  except Exception as e:
    print(f"\n--- Adaptive (Rocchio PRF) --- (skipped: {e})")

  # ========== BONUS: GAR (Graph-based Adaptive Re-ranking) ==========
  try:
    from adaptive_retrieval import CorpusGraph, retrieve_gar
    BSBI_instance.load()
    graph_path = os.path.join('index', 'corpus_graph.pkl')
    cg = CorpusGraph(k=16)
    if os.path.exists(graph_path):
      cg.load(graph_path)
    else:
      cg.build('main_index', VBEPostings, 'index')
      cg.save(graph_path)

    def gar_retrieve(query, k=10):
      return retrieve_gar(query, BSBI_instance, cg, k=k)

    gar_scores = _evaluate_method(gar_retrieve, queries_data, qrels, k)
    _print_method_scores("GAR + BM25 scorer", gar_scores)
  except Exception as e:
    print(f"\n--- GAR + BM25 --- (skipped: {e})")

  # ========== BONUS: GAR + LSI scorer ==========
  try:
    from adaptive_retrieval import CorpusGraph, retrieve_gar_lsi
    from lsi import LSIRetriever
    BSBI_instance.load()

    # Load or build corpus graph
    graph_path = os.path.join('index', 'corpus_graph.pkl')
    if 'cg' not in dir():
      cg = CorpusGraph(k=16)
      if os.path.exists(graph_path):
        cg.load(graph_path)
      else:
        cg.build('main_index', VBEPostings, 'index')
        cg.save(graph_path)

    # Load or build LSI model
    lsi_model_path = os.path.join('index', 'lsi_model.pkl')
    if 'lsi' not in dir():
      lsi = LSIRetriever(k=150)
      if os.path.exists(lsi_model_path):
        lsi.load(lsi_model_path)
      else:
        lsi.build('main_index', VBEPostings, 'index')
        lsi.save(lsi_model_path)

    def gar_lsi_retrieve(query, k=10):
      return retrieve_gar_lsi(query, BSBI_instance, cg, lsi, k=k)

    gar_lsi_scores = _evaluate_method(gar_lsi_retrieve, queries_data, qrels, k)
    _print_method_scores("GAR + LSI scorer (adaptive)", gar_lsi_scores)
  except Exception as e:
    print(f"\n--- GAR + LSI --- (skipped: {e})")

  print("\n" + "=" * 70)


if __name__ == '__main__':
  qrels = load_qrels()

  assert qrels["Q1"][166] == 1, "qrels salah"
  assert qrels["Q1"][300] == 0, "qrels salah"

  # Test metric functions
  print("=== Test Metric Functions ===")

  # Test DCG
  test_ranking = [1, 0, 1, 1, 0]
  dcg_val = dcg(test_ranking, k=5)
  expected_dcg = 1/math.log2(2) + 0/math.log2(3) + 1/math.log2(4) + 1/math.log2(5) + 0/math.log2(6)
  assert abs(dcg_val - expected_dcg) < 1e-10, f"DCG salah: {dcg_val} != {expected_dcg}"
  print(f"DCG test PASSED: {dcg_val:.4f}")

  # Test NDCG
  ndcg_val = ndcg(test_ranking, k=5)
  ideal_dcg = 1/math.log2(2) + 1/math.log2(3) + 1/math.log2(4) + 0/math.log2(5) + 0/math.log2(6)
  expected_ndcg = expected_dcg / ideal_dcg
  assert abs(ndcg_val - expected_ndcg) < 1e-10, f"NDCG salah: {ndcg_val} != {expected_ndcg}"
  print(f"NDCG test PASSED: {ndcg_val:.4f}")

  # Test AP
  ap_val = ap(test_ranking)
  # Relevan di rank 1, 3, 4
  # Precision@1 = 1/1 = 1.0
  # Precision@3 = 2/3 = 0.667
  # Precision@4 = 3/4 = 0.75
  # AP = (1.0 + 0.667 + 0.75) / 3
  expected_ap = (1/1 + 2/3 + 3/4) / 3
  assert abs(ap_val - expected_ap) < 1e-10, f"AP salah: {ap_val} != {expected_ap}"
  print(f"AP test PASSED: {ap_val:.4f}")

  # Test edge case: no relevant documents
  assert ap([0, 0, 0]) == 0.0, "AP untuk no relevant docs harus 0"
  assert ndcg([0, 0, 0]) == 0.0, "NDCG untuk no relevant docs harus 0"
  print("Edge case tests PASSED")

  print("\nSemua metric tests PASSED!\n")

  # Run evaluation
  eval(qrels)

"""
BONUS: Implementasi SPIMI (Single-Pass In-Memory Indexing)

Perbedaan SPIMI vs BSBI:
- BSBI: Menghasilkan pasangan (termID, docID) dari keseluruhan block, sort
  berdasarkan termID, lalu invert dan tulis ke disk. Membutuhkan mapping
  global term->termID sejak awal.
- SPIMI: Membaca token satu per satu, langsung menambahkan ke dictionary
  (hashtable) yang ada di memori. Ketika memori penuh (atau block selesai),
  sort dictionary lalu tulis ke disk sebagai intermediate index.
  Tidak perlu mapping global termID - postings list langsung dibangun
  menggunakan dictionary.

Keuntungan SPIMI:
- Lebih cepat karena tidak perlu sorting pasangan (termID, docID)
- Lebih hemat memori karena postings list dibangun langsung (no duplicate pairs)
- Lebih scalable untuk koleksi yang sangat besar

Implementasi ini juga menambahkan opsi preprocessing:
- Porter Stemming
- Stopword Removal

Serta struktur data Trie untuk dictionary lookup yang lebih efisien.
"""

import os
import pickle
import contextlib
import heapq
import math
import re

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings, EliasGammaPostings
from tqdm import tqdm

# ============================================================
# BONUS: Trie Data Structure untuk Dictionary
# ============================================================

class TrieNode:
    """Node dalam Trie (Prefix Tree)."""
    __slots__ = ['children', 'term_id']

    def __init__(self):
        self.children = {}
        self.term_id = None  # None jika bukan akhir dari sebuah term

class Trie:
    """
    Trie (Prefix Tree) untuk menyimpan dictionary terms.

    Trie menyediakan:
    - O(m) lookup dimana m adalah panjang string (vs O(m) average untuk hashtable,
      tapi worst case O(n*m) jika banyak collision)
    - Prefix search yang efisien (berguna untuk autocomplete/wildcard query)
    - Shared prefixes menghemat memori untuk terms dengan prefix yang sama

    Trie ini digunakan sebagai alternatif dari Python dictionary (hashtable)
    untuk menyimpan mapping term -> term_id.
    """
    def __init__(self):
        self.root = TrieNode()
        self.id_to_str = []
        self._size = 0

    def __len__(self):
        return self._size

    def insert(self, term):
        """
        Insert term ke Trie. Jika term sudah ada, kembalikan term_id yang ada.
        Jika belum, assign term_id baru.

        Parameters
        ----------
        term : str
            Term yang akan diinsert

        Returns
        -------
        int
            term_id yang bersesuaian
        """
        node = self.root
        for char in term:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]

        if node.term_id is None:
            node.term_id = self._size
            self.id_to_str.append(term)
            self._size += 1

        return node.term_id

    def search(self, term):
        """
        Cari term_id dari sebuah term.

        Parameters
        ----------
        term : str
            Term yang dicari

        Returns
        -------
        int or None
            term_id jika ditemukan, None jika tidak
        """
        node = self.root
        for char in term:
            if char not in node.children:
                return None
            node = node.children[char]
        return node.term_id

    def get_term(self, term_id):
        """Kembalikan term string dari term_id."""
        return self.id_to_str[term_id]

    def prefix_search(self, prefix):
        """
        Cari semua terms yang dimulai dengan prefix tertentu.
        Berguna untuk wildcard query dan autocomplete.

        Parameters
        ----------
        prefix : str
            Prefix yang dicari

        Returns
        -------
        List[Tuple[str, int]]
            List of (term, term_id) pairs yang memiliki prefix tersebut
        """
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        # DFS untuk mengumpulkan semua terms dari node ini
        results = []
        self._collect_terms(node, prefix, results)
        return results

    def _collect_terms(self, node, prefix, results):
        if node.term_id is not None:
            results.append((prefix, node.term_id))
        for char, child in node.children.items():
            self._collect_terms(child, prefix + char, results)


# ============================================================
# Preprocessing: Stemming dan Stopword Removal
# ============================================================

class TextPreprocessor:
    """
    Preprocessor teks yang menyediakan:
    - Tokenization menggunakan regex
    - Lowercasing
    - Stopword removal (English stopwords)
    - Porter Stemming

    Preprocessing sangat penting untuk meningkatkan kualitas retrieval:
    - Stemming mengurangi variasi kata ke bentuk dasarnya
      (e.g., "running", "runs", "ran" -> "run")
    - Stopword removal menghilangkan kata-kata umum yang tidak informatif
      (e.g., "the", "is", "at", "which")
    """

    # English stopwords (dari NLTK)
    STOPWORDS = frozenset({
        'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you',
        "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself',
        'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her',
        'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them',
        'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
        'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are',
        'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if',
        'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
        'with', 'about', 'against', 'between', 'through', 'during', 'before',
        'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out',
        'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
        'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
        'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
        'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't',
        'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now',
        'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't",
        'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn',
        "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
        'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't",
        'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren',
        "weren't", 'won', "won't", 'wouldn', "wouldn't"
    })

    def __init__(self, use_stemming=True, use_stopwords=True):
        self.use_stemming = use_stemming
        self.use_stopwords = use_stopwords
        self._stemmer = None

        if use_stemming:
            from porter_stemmer import PorterStemmer
            self._stemmer = PorterStemmer()

        # Regex tokenizer: hanya alfanumerik
        self._token_pattern = re.compile(r'[a-zA-Z0-9]+')

    def tokenize(self, text):
        """
        Tokenize dan preprocess sebuah teks.

        Parameters
        ----------
        text : str
            Teks yang akan di-tokenize

        Returns
        -------
        List[str]
            List of preprocessed tokens
        """
        # Lowercase dan tokenize
        tokens = self._token_pattern.findall(text.lower())

        # Stopword removal
        if self.use_stopwords:
            tokens = [t for t in tokens if t not in self.STOPWORDS]

        # Stemming
        if self.use_stemming and self._stemmer:
            tokens = [self._stemmer.stem(t) for t in tokens]

        return tokens


# ============================================================
# BONUS: SPIMI Indexing
# ============================================================

class SPIMIIndex:
    """
    Implementasi Single-Pass In-Memory Indexing (SPIMI).

    Perbedaan utama dengan BSBI:
    1. Tidak membuat pasangan (termID, docID) secara eksplisit
    2. Langsung membangun inverted index (dictionary -> postings list) di memori
    3. Ketika memori penuh atau block selesai, sort dan tulis ke disk
    4. Term dictionary bisa menggunakan Trie untuk lookup yang efisien

    SPIMI juga mendukung preprocessing dengan stemming dan stopword removal.

    Attributes
    ----------
    term_id_map(IdMap): Mapping terms ke termIDs (untuk compatibility)
    doc_id_map(IdMap): Mapping document paths ke docIDs
    data_dir(str): Path ke data collection
    output_dir(str): Path ke output index files
    postings_encoding: Encoding untuk postings (VBEPostings, EliasGammaPostings, dll)
    index_name(str): Nama file index
    use_trie(bool): Gunakan Trie untuk dictionary (bonus)
    preprocessor(TextPreprocessor): Untuk stemming dan stopword removal
    """
    def __init__(self, data_dir, output_dir, postings_encoding,
                 index_name="main_index", use_trie=False,
                 use_stemming=False, use_stopwords=False):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.intermediate_indices = []

        # Bonus: Trie untuk dictionary
        self.use_trie = use_trie
        self.trie = Trie() if use_trie else None

        # Bonus: Preprocessing
        self.preprocessor = TextPreprocessor(use_stemming, use_stopwords)
        self.use_preprocessing = use_stemming or use_stopwords

    def _get_term_id(self, term):
        """
        Dapatkan term ID, menggunakan Trie jika diaktifkan.
        """
        if self.use_trie:
            return self.trie.insert(term)
        else:
            return self.term_id_map[term]

    def save(self):
        """Menyimpan doc_id_map dan term_id_map ke output directory."""
        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)
        if self.use_trie:
            with open(os.path.join(self.output_dir, 'trie.dict'), 'wb') as f:
                pickle.dump(self.trie, f)

    def load(self):
        """Memuat doc_id_map dan term_id_map dari output directory."""
        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)
        if self.use_trie:
            trie_path = os.path.join(self.output_dir, 'trie.dict')
            if os.path.exists(trie_path):
                with open(trie_path, 'rb') as f:
                    self.trie = pickle.load(f)

    def spimi_invert(self, block_dir_relative):
        """
        SPIMI-Invert: Bangun inverted index langsung di memori untuk satu block.

        Berbeda dengan BSBI yang membuat (termID, docID) pairs lalu sort,
        SPIMI langsung membangun dictionary dan postings list di memori.

        Parameters
        ----------
        block_dir_relative : str
            Relative path ke directory block

        Returns
        -------
        dict
            Dictionary mapping termID -> {docID: tf, ...}
        """
        dir_path = "./" + self.data_dir + "/" + block_dir_relative
        # SPIMI: dictionary langsung dibangun di memori
        inverted_index = {}  # termID -> {docID: tf}

        for filename in next(os.walk(dir_path))[2]:
            docname = dir_path + "/" + filename
            doc_id = self.doc_id_map[docname]

            with open(docname, "r", encoding="utf8", errors="surrogateescape") as f:
                text = f.read()

                if self.use_preprocessing:
                    tokens = self.preprocessor.tokenize(text)
                else:
                    tokens = text.split()

                for token in tokens:
                    term_id = self._get_term_id(token)

                    # SPIMI: langsung tambahkan ke postings list
                    if term_id not in inverted_index:
                        inverted_index[term_id] = {}
                    if doc_id not in inverted_index[term_id]:
                        inverted_index[term_id][doc_id] = 0
                    inverted_index[term_id][doc_id] += 1

        return inverted_index

    def write_block_index(self, inverted_index, index):
        """
        Tulis inverted index dari memori ke disk.

        Parameters
        ----------
        inverted_index : dict
            termID -> {docID: tf}
        index : InvertedIndexWriter
            Writer untuk menulis ke disk
        """
        for term_id in sorted(inverted_index.keys()):
            doc_tf = inverted_index[term_id]
            sorted_doc_ids = sorted(doc_tf.keys())
            postings_list = sorted_doc_ids
            tf_list = [doc_tf[doc_id] for doc_id in sorted_doc_ids]
            index.append(term_id, postings_list, tf_list)

    def merge(self, indices, merged_index):
        """Sama dengan BSBI merge - external merge sort."""
        merged_iter = heapq.merge(*indices, key=lambda x: x[0])
        curr, postings, tf_list = next(merged_iter)
        for t, postings_, tf_list_ in merged_iter:
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(
                    list(zip(postings, tf_list)),
                    list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def index(self):
        """
        Main indexing dengan SPIMI.

        Untuk setiap block:
        1. Bangun inverted index langsung di memori (SPIMI-Invert)
        2. Sort terms dan tulis ke disk sebagai intermediate index
        3. Merge semua intermediate indices menjadi satu final index
        """
        print("SPIMI Indexing dimulai...")
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            # SPIMI-Invert: bangun inverted index di memori
            inverted_index = self.spimi_invert(block_dir_relative)

            index_id = 'intermediate_index_' + block_dir_relative
            self.intermediate_indices.append(index_id)

            # Sort dan tulis ke disk
            with InvertedIndexWriter(index_id, self.postings_encoding,
                                     directory=self.output_dir) as idx:
                self.write_block_index(inverted_index, idx)
                inverted_index = None  # Free memory

        self.save()

        # Merge semua intermediate indices
        with InvertedIndexWriter(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(
                    InvertedIndexReader(index_id, self.postings_encoding,
                                        directory=self.output_dir))
                    for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)

        print("SPIMI Indexing selesai!")

    def retrieve_bm25(self, query, k=10, k1=1.2, b=0.75):
        """BM25 retrieval (sama dengan BSBIIndex.retrieve_bm25)."""
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        if self.use_preprocessing:
            query_terms = self.preprocessor.tokenize(query)
        else:
            query_terms = query.split()

        terms = [self._get_term_id(word) for word in query_terms]

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            if N == 0:
                return []
            avgdl = sum(merged_index.doc_length.values()) / N

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        dl = merged_index.doc_length.get(doc_id, 0)
                        tf_comp = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                        scores[doc_id] += idf * tf_comp

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]

    def retrieve_tfidf(self, query, k=10):
        """TF-IDF retrieval."""
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        if self.use_preprocessing:
            query_terms = self.preprocessor.tokenize(query)
        else:
            query_terms = query.split()

        terms = [self._get_term_id(word) for word in query_terms]

        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as merged_index:
            N = len(merged_index.doc_length)
            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key=lambda x: x[0], reverse=True)[:k]


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("BONUS: SPIMI Indexing")
    print("=" * 60)

    # SPIMI tanpa preprocessing
    print("\n--- SPIMI (tanpa preprocessing) ---")
    start = time.time()
    spimi = SPIMIIndex(data_dir='collection',
                       postings_encoding=VBEPostings,
                       output_dir='index')
    spimi.index()
    elapsed = time.time() - start
    print(f"Waktu indexing: {elapsed:.2f} detik")

    queries = ["alkylated with radioactive iodoacetate",
               "psychodrama for disturbed children",
               "lipid metabolism in toxemia and normal pregnancy"]

    for query in queries:
        print(f"\nQuery: {query}")
        print("  BM25 Results:")
        for (score, doc) in spimi.retrieve_bm25(query, k=5):
            print(f"    {doc:30} {score:>.3f}")

    # SPIMI dengan Trie dictionary
    print("\n\n--- SPIMI + Trie Dictionary ---")
    start = time.time()
    spimi_trie = SPIMIIndex(data_dir='collection',
                            postings_encoding=VBEPostings,
                            output_dir='index',
                            use_trie=True)
    spimi_trie.index()
    elapsed = time.time() - start
    print(f"Waktu indexing dengan Trie: {elapsed:.2f} detik")
    print(f"Jumlah terms di Trie: {len(spimi_trie.trie)}")

    # Demo prefix search dari Trie
    print("\nDemo Trie prefix search 'elect':")
    for term, tid in spimi_trie.trie.prefix_search("elect")[:10]:
        print(f"  {term} (id={tid})")

    print("\n" + "=" * 60)
    print("SPIMI indexing selesai!")

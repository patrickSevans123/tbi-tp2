import os
import pickle
import contextlib
import heapq
import time
import math

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_merge_posts_and_tfs
from compression import StandardPostings, VBEPostings, EliasGammaPostings
from tqdm import tqdm

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, EliasGammaPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Inggris

        JANGAN LUPA BUANG STOPWORDS!

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus 'persist' untuk semua pemanggilan
        parse_block(...).
        """
        dir = "./" + self.data_dir + "/" + block_dir_relative
        td_pairs = []
        for filename in next(os.walk(dir))[2]:
            docname = dir + "/" + filename
            with open(docname, "r", encoding = "utf8", errors = "surrogateescape") as f:
                for token in f.read().split():
                    td_pairs.append((self.term_id_map[token], self.doc_id_map[docname]))

        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Di Tugas Pemrograman 1, kita hanya menambahkan term dan
        juga list of sorted Doc IDs. Sekarang di Tugas Pemrograman 2,
        kita juga perlu tambahkan list of TF.

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        term_tf = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
                term_tf[term_id] = {}
            term_dict[term_id].add(doc_id)
            if doc_id not in term_tf[term_id]:
                term_tf[term_id][doc_id] = 0
            term_tf[term_id][doc_id] += 1
        for term_id in sorted(term_dict.keys()):
            sorted_doc_id = sorted(list(term_dict[term_id]))
            assoc_tf = [term_tf[term_id][doc_id] for doc_id in sorted_doc_id]
            index.append(term_id, sorted_doc_id, assoc_tf)

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Gunakan fungsi sorted_merge_posts_and_tfs(..) di modul util

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        # kode berikut mengasumsikan minimal ada 1 term
        merged_iter = heapq.merge(*indices, key = lambda x: x[0])
        curr, postings, tf_list = next(merged_iter) # first item
        for t, postings_, tf_list_ in merged_iter: # from the second item
            if t == curr:
                zip_p_tf = sorted_merge_posts_and_tfs(list(zip(postings, tf_list)), \
                                                      list(zip(postings_, tf_list_)))
                postings = [doc_id for (doc_id, _) in zip_p_tf]
                tf_list = [tf for (_, tf) in zip_p_tf]
            else:
                merged_index.append(curr, postings, tf_list)
                curr, postings, tf_list = t, postings_, tf_list_
        merged_index.append(curr, postings, tf_list)

    def retrieve_tfidf(self, query, k = 10):
        """
        Melakukan Ranked Retrieval dengan skema TaaT (Term-at-a-Time).
        Method akan mengembalikan top-K retrieval results.

        w(t, D) = (1 + log tf(t, D))       jika tf(t, D) > 0
                = 0                        jika sebaliknya

        w(t, Q) = IDF = log (N / df(t))

        Score = untuk setiap term di query, akumulasikan w(t, Q) * w(t, D).
                (tidak perlu dinormalisasi dengan panjang dokumen)

        catatan:
            1. informasi DF(t) ada di dictionary postings_dict pada merged index
            2. informasi TF(t, D) ada di tf_li
            3. informasi N bisa didapat dari doc_length pada merged index, len(doc_length)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya ada
            tiga terms: universitas, indonesia, dan depok

        Result
        ------
        List[(int, str)]
            List of tuple: elemen pertama adalah score similarity, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.

        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    N = len(merged_index.doc_length)
                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0
                        if tf > 0:
                            scores[doc_id] += math.log(N / df) * (1 + math.log(tf))

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_bm25(self, query, k = 10, k1 = 1.2, b = 0.75):
        """
        Melakukan Ranked Retrieval dengan skema scoring BM25 dan TaaT
        (Term-at-a-Time). Method akan mengembalikan top-K retrieval results.

        BM25 Score Formula:
            Score(D, Q) = sum over t in Q of:
                IDF(t) * (tf(t,D) * (k1 + 1)) / (tf(t,D) + k1 * (1 - b + b * |D| / avgdl))

            IDF(t) = log((N - df(t) + 0.5) / (df(t) + 0.5) + 1)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi
        k : int
            Jumlah top-K dokumen yang dikembalikan (default: 10)
        k1 : float
            Parameter BM25 untuk term frequency saturation (default: 1.2)
        b : float
            Parameter BM25 untuk document length normalization (default: 0.75)

        Result
        ------
        List[(float, str)]
            List of tuple: elemen pertama adalah BM25 score, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        terms = [self.term_id_map[word] for word in query.split()]
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            N = len(merged_index.doc_length)
            avgdl = sum(merged_index.doc_length.values()) / N if N > 0 else 0

            scores = {}
            for term in terms:
                if term in merged_index.postings_dict:
                    df = merged_index.postings_dict[term][1]
                    # IDF sesuai formula BM25 (Robertson/Sparck-Jones)
                    idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

                    postings, tf_list = merged_index.get_postings_list(term)
                    for i in range(len(postings)):
                        doc_id, tf = postings[i], tf_list[i]
                        if doc_id not in scores:
                            scores[doc_id] = 0

                        dl = merged_index.doc_length.get(doc_id, 0)
                        # BM25 term frequency component
                        tf_component = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                        scores[doc_id] += idf * tf_component

            # Top-K
            docs = [(score, self.doc_id_map[doc_id]) for (doc_id, score) in scores.items()]
            return sorted(docs, key = lambda x: x[0], reverse = True)[:k]

    def retrieve_bm25_wand(self, query, k = 10, k1 = 1.2, b = 0.75):
        """
        Melakukan Ranked Retrieval dengan BM25 scoring menggunakan algoritma
        WAND (Weak AND) untuk Top-K retrieval yang lebih efisien.

        WAND menghindari scoring semua dokumen dengan menggunakan upper bound
        score per term untuk memangkas (prune) dokumen yang tidak mungkin masuk
        ke top-K.

        Algoritma WAND:
        1. Untuk setiap query term, hitung upper bound BM25 score (skor maksimum
           yang bisa dikontribusikan term tersebut ke dokumen manapun).
        2. Urutkan term berdasarkan current doc ID pointer.
        3. Cari "pivot" term dimana kumulatif upper bound >= threshold.
        4. Jika semua term sebelum pivot menunjuk ke dokumen yang sama (pivot_doc),
           lakukan full scoring pada dokumen tersebut.
        5. Jika tidak, advance pointer term terkecil ke pivot_doc.
        6. Ulangi sampai tidak ada lagi kandidat.

        Upper bound per term dihitung sebagai:
            UB(t) = IDF(t) * max_tf(t) * (k1 + 1) / (max_tf(t) + k1 * (1 - b))
        dimana max_tf(t) adalah TF terbesar untuk term t di seluruh koleksi.
        Bound ini loose tapi safe (dl/avgdl diasumsikan mendekati 0).

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi
        k : int
            Jumlah top-K dokumen yang dikembalikan (default: 10)
        k1 : float
            Parameter BM25 untuk term frequency saturation (default: 1.2)
        b : float
            Parameter BM25 untuk document length normalization (default: 0.75)

        Result
        ------
        List[(float, str)]
            List of tuple: elemen pertama adalah BM25 score, dan yang
            kedua adalah nama dokumen.
            Daftar Top-K dokumen terurut mengecil BERDASARKAN SKOR.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        query_terms = query.split()
        with InvertedIndexReader(self.index_name, self.postings_encoding, directory=self.output_dir) as merged_index:

            N = len(merged_index.doc_length)
            if N == 0:
                return []
            avgdl = sum(merged_index.doc_length.values()) / N

            # Muat posting lists dan hitung upper bounds untuk setiap query term
            term_data = []
            for word in query_terms:
                term = self.term_id_map[word]
                if term not in merged_index.postings_dict:
                    continue

                entry = merged_index.postings_dict[term]
                df = entry[1]
                # max_tf disimpan di elemen ke-5 dari postings_dict
                max_tf = entry[4] if len(entry) > 4 else 1

                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
                postings, tf_list = merged_index.get_postings_list(term)

                # Upper bound: BM25 score maksimum untuk term ini
                # Terjadi ketika tf = max_tf dan dl/avgdl -> 0 (dokumen terpendek)
                ub = idf * (max_tf * (k1 + 1)) / (max_tf + k1 * (1 - b))

                term_data.append({
                    'postings': postings,
                    'tf_list': tf_list,
                    'idf': idf,
                    'ptr': 0,  # current pointer dalam postings list
                    'ub': ub
                })

            if not term_data:
                return []

            # WAND Algorithm
            top_k_heap = []  # min-heap of (score, doc_id)
            threshold = 0

            while True:
                # Buang term yang posting list-nya sudah habis
                term_data = [t for t in term_data if t['ptr'] < len(t['postings'])]
                if not term_data:
                    break

                # Urutkan term berdasarkan current doc ID
                term_data.sort(key=lambda t: t['postings'][t['ptr']])

                # Cari pivot: indeks terkecil p dimana sum UB[0..p] >= threshold
                cum_ub = 0
                pivot_idx = -1
                for i, t in enumerate(term_data):
                    cum_ub += t['ub']
                    if cum_ub >= threshold:
                        pivot_idx = i
                        break

                if pivot_idx == -1:
                    # Tidak ada kombinasi term yang bisa melampaui threshold
                    break

                pivot_doc = term_data[pivot_idx]['postings'][term_data[pivot_idx]['ptr']]

                # Cek apakah term pertama (doc ID terkecil) sudah di pivot_doc
                if term_data[0]['postings'][term_data[0]['ptr']] == pivot_doc:
                    # Semua term dari 0 s.d. pivot_idx menunjuk ke pivot_doc
                    # (karena sorted dan term[0] = min = pivot_doc)
                    # Lakukan full BM25 scoring pada pivot_doc
                    score = 0
                    for t in term_data:
                        ptr = t['ptr']
                        if ptr < len(t['postings']) and t['postings'][ptr] == pivot_doc:
                            tf = t['tf_list'][ptr]
                            dl = merged_index.doc_length.get(pivot_doc, 0)
                            tf_comp = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
                            score += t['idf'] * tf_comp
                            t['ptr'] += 1  # advance pointer melewati pivot_doc

                    # Update top-K heap
                    if len(top_k_heap) < k:
                        heapq.heappush(top_k_heap, (score, pivot_doc))
                        if len(top_k_heap) == k:
                            threshold = top_k_heap[0][0]
                    elif score > top_k_heap[0][0]:
                        heapq.heapreplace(top_k_heap, (score, pivot_doc))
                        threshold = top_k_heap[0][0]
                else:
                    # Advance term pertama (pointer terkecil) ke pivot_doc
                    # menggunakan binary search (galloping/skip)
                    t = term_data[0]
                    lo, hi = t['ptr'], len(t['postings'])
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if t['postings'][mid] < pivot_doc:
                            lo = mid + 1
                        else:
                            hi = mid
                    t['ptr'] = lo

            # Konversi hasil ke format output
            results = [(score, self.doc_id_map[doc_id]) for score, doc_id in top_k_heap]
            return sorted(results, key=lambda x: x[0], reverse=True)

    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None

        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')
    BSBI_instance.index() # memulai indexing!

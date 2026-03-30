import array
import struct
import math

class StandardPostings:
    """
    Class dengan static methods, untuk mengubah representasi postings list
    yang awalnya adalah List of integer, berubah menjadi sequence of bytes.
    Kita menggunakan Library array di Python.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    Silakan pelajari:
        https://docs.python.org/3/library/array.html
    """

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        # Untuk yang standard, gunakan L untuk unsigned long, karena docID
        # tidak akan negatif. Dan kita asumsikan docID yang paling besar
        # cukup ditampung di representasi 4 byte unsigned.
        return array.array('L', postings_list).tobytes()

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = array.array('L')
        decoded_postings_list.frombytes(encoded_postings_list)
        return decoded_postings_list.tolist()

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return StandardPostings.encode(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return StandardPostings.decode(encoded_tf_list)

class VBEPostings:
    """
    Berbeda dengan StandardPostings, dimana untuk suatu postings list,
    yang disimpan di disk adalah sequence of integers asli dari postings
    list tersebut apa adanya.

    Pada VBEPostings, kali ini, yang disimpan adalah gap-nya, kecuali
    posting yang pertama. Barulah setelah itu di-encode dengan Variable-Byte
    Enconding algorithm ke bytestream.

    Contoh:
    postings list [34, 67, 89, 454] akan diubah dulu menjadi gap-based,
    yaitu [34, 33, 22, 365]. Barulah setelah itu di-encode dengan algoritma
    compression Variable-Byte Encoding, dan kemudian diubah ke bytesream.

    ASUMSI: postings_list untuk sebuah term MUAT di memori!

    """

    @staticmethod
    def vb_encode_number(number):
        """
        Encodes a number using Variable-Byte Encoding
        Lihat buku teks kita!
        """
        bytes = []
        while True:
            bytes.insert(0, number % 128) # prepend ke depan
            if number < 128:
                break
            number = number // 128
        bytes[-1] += 128 # bit awal pada byte terakhir diganti 1
        return array.array('B', bytes).tobytes()

    @staticmethod
    def vb_encode(list_of_numbers):
        """
        Melakukan encoding (tentunya dengan compression) terhadap
        list of numbers, dengan Variable-Byte Encoding
        """
        bytes = []
        for number in list_of_numbers:
            bytes.append(VBEPostings.vb_encode_number(number))
        return b"".join(bytes)

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes (dengan Variable-Byte
        Encoding). JANGAN LUPA diubah dulu ke gap-based list, sebelum
        di-encode dan diubah ke bytearray.

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings)

        Returns
        -------
        bytes
            bytearray yang merepresentasikan urutan integer di postings_list
        """
        gap_postings_list = [postings_list[0]]
        for i in range(1, len(postings_list)):
            gap_postings_list.append(postings_list[i] - postings_list[i-1])
        return VBEPostings.vb_encode(gap_postings_list)

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menjadi stream of bytes

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray yang merepresentasikan nilai raw TF kemunculan term di setiap
            dokumen pada list of postings
        """
        return VBEPostings.vb_encode(tf_list)

    @staticmethod
    def vb_decode(encoded_bytestream):
        """
        Decoding sebuah bytestream yang sebelumnya di-encode dengan
        variable-byte encoding.
        """
        n = 0
        numbers = []
        decoded_bytestream = array.array('B')
        decoded_bytestream.frombytes(encoded_bytestream)
        bytestream = decoded_bytestream.tolist()
        for byte in bytestream:
            if byte < 128:
                n = 128 * n + byte
            else:
                n = 128 * n + (byte - 128)
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decodes postings_list dari sebuah stream of bytes. JANGAN LUPA
        bytestream yang di-decode dari encoded_postings_list masih berupa
        gap-based list.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray merepresentasikan encoded postings list sebagai keluaran
            dari static method encode di atas.

        Returns
        -------
        List[int]
            list of docIDs yang merupakan hasil decoding dari encoded_postings_list
        """
        decoded_postings_list = VBEPostings.vb_decode(encoded_postings_list)
        total = decoded_postings_list[0]
        ori_postings_list = [total]
        for i in range(1, len(decoded_postings_list)):
            total += decoded_postings_list[i]
            ori_postings_list.append(total)
        return ori_postings_list

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decodes list of term frequencies dari sebuah stream of bytes

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray merepresentasikan encoded term frequencies list sebagai keluaran
            dari static method encode_tf di atas.

        Returns
        -------
        List[int]
            List of term frequencies yang merupakan hasil decoding dari encoded_tf_list
        """
        return VBEPostings.vb_decode(encoded_tf_list)


class EliasGammaPostings:
    """
    Implementasi Elias-Gamma Encoding untuk kompresi postings list.

    Elias-Gamma adalah bit-level encoding scheme yang mengkodekan bilangan
    positif N sebagai:
        1. floor(log2(N)) buah bit '0' (unary prefix)
        2. Diikuti representasi biner dari N (L+1 bit)

    Contoh:
        1  -> 1           (0 zeros + "1")
        2  -> 010         (1 zero  + "10")
        3  -> 011         (1 zero  + "11")
        4  -> 00100       (2 zeros + "100")
        5  -> 00101       (2 zeros + "101")
        13 -> 0001101     (3 zeros + "1101")

    Seperti VBEPostings, postings list diubah menjadi gap-based terlebih
    dahulu sebelum di-encode. Karena Elias-Gamma hanya dapat mengkodekan
    bilangan positif (>= 1), setiap gap/nilai di-increment 1 sebelum encoding.

    Bit stream yang dihasilkan di-pad ke kelipatan 8 bit, dan jumlah elemen
    disimpan di 4 byte awal agar decoding tahu berapa banyak angka yang perlu
    di-decode (untuk menghindari ambiguitas dari padding bits).

    ASUMSI: postings_list untuk sebuah term MUAT di memori!
    """

    @staticmethod
    def _elias_gamma_encode_single(n):
        """
        Encode satu bilangan positif (n >= 1) menggunakan Elias-Gamma coding.

        Parameters
        ----------
        n : int
            Bilangan positif yang akan di-encode (n >= 1)

        Returns
        -------
        list[int]
            List of bits (0 atau 1)
        """
        if n < 1:
            raise ValueError(f"Elias-Gamma hanya untuk bilangan >= 1, diberikan {n}")
        L = n.bit_length() - 1  # floor(log2(n))
        # L buah zeros
        bits = [0] * L
        # Diikuti representasi biner dari n (L+1 bit, MSB first)
        for i in range(L, -1, -1):
            bits.append((n >> i) & 1)
        return bits

    @staticmethod
    def _elias_gamma_decode_single(bits, pos):
        """
        Decode satu bilangan dari bit stream mulai dari posisi pos.

        Parameters
        ----------
        bits : list[int]
            Bit stream
        pos : int
            Posisi mulai decoding

        Returns
        -------
        tuple(int, int)
            (bilangan yang di-decode, posisi berikutnya di bit stream)
        """
        # Hitung leading zeros
        L = 0
        while pos < len(bits) and bits[pos] == 0:
            L += 1
            pos += 1
        # Baca L+1 bit sebagai bilangan biner
        n = 0
        for _ in range(L + 1):
            n = (n << 1) | bits[pos]
            pos += 1
        return n, pos

    @staticmethod
    def _bits_to_bytes(bits):
        """Konversi list of bits ke bytes, dengan padding zeros di akhir."""
        # Pad ke kelipatan 8
        padding = (8 - len(bits) % 8) % 8
        bits = bits + [0] * padding
        result = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | bits[i + j]
            result.append(byte)
        return bytes(result)

    @staticmethod
    def _bytes_to_bits(data):
        """Konversi bytes ke list of bits."""
        bits = []
        for byte in data:
            for i in range(7, -1, -1):
                bits.append((byte >> i) & 1)
        return bits

    @staticmethod
    def _encode_numbers(numbers):
        """
        Encode list of bilangan positif (>= 1) menjadi bytes
        menggunakan Elias-Gamma encoding.

        Format: [4 bytes count][elias-gamma encoded bits padded to byte boundary]
        """
        all_bits = []
        for num in numbers:
            all_bits.extend(EliasGammaPostings._elias_gamma_encode_single(num))
        encoded_bits = EliasGammaPostings._bits_to_bytes(all_bits)
        # Prepend jumlah elemen sebagai 4-byte unsigned int
        return struct.pack('I', len(numbers)) + encoded_bits

    @staticmethod
    def _decode_numbers(encoded_bytes):
        """
        Decode bytes yang di-encode dengan Elias-Gamma kembali ke list of
        bilangan positif.
        """
        count = struct.unpack('I', encoded_bytes[:4])[0]
        data = encoded_bytes[4:]
        bits = EliasGammaPostings._bytes_to_bits(data)

        numbers = []
        pos = 0
        for _ in range(count):
            num, pos = EliasGammaPostings._elias_gamma_decode_single(bits, pos)
            numbers.append(num)
        return numbers

    @staticmethod
    def encode(postings_list):
        """
        Encode postings_list menjadi stream of bytes menggunakan Elias-Gamma
        encoding. Postings list diubah ke gap-based terlebih dahulu, kemudian
        setiap nilai di-increment 1 (karena Elias-Gamma hanya untuk bilangan >= 1,
        dan doc ID pertama bisa bernilai 0).

        Parameters
        ----------
        postings_list: List[int]
            List of docIDs (postings), terurut menaik

        Returns
        -------
        bytes
            bytearray hasil Elias-Gamma encoding dari gap-based postings list
        """
        # Konversi ke gap-based
        gaps = [postings_list[0] + 1]  # +1 karena doc ID pertama bisa 0
        for i in range(1, len(postings_list)):
            gaps.append(postings_list[i] - postings_list[i - 1])
            # Gap antar postings selalu >= 1 karena doc ID unik dan terurut
        return EliasGammaPostings._encode_numbers(gaps)

    @staticmethod
    def decode(encoded_postings_list):
        """
        Decode postings_list dari stream of bytes yang di-encode dengan
        Elias-Gamma. Mengembalikan gap-based ke postings list asli.

        Parameters
        ----------
        encoded_postings_list: bytes
            bytearray hasil encoding

        Returns
        -------
        List[int]
            list of docIDs hasil decoding
        """
        gaps = EliasGammaPostings._decode_numbers(encoded_postings_list)
        postings = [gaps[0] - 1]  # Reverse the +1
        for i in range(1, len(gaps)):
            postings.append(postings[-1] + gaps[i])
        return postings

    @staticmethod
    def encode_tf(tf_list):
        """
        Encode list of term frequencies menggunakan Elias-Gamma encoding.
        TF selalu >= 1, sehingga bisa langsung di-encode.

        Parameters
        ----------
        tf_list: List[int]
            List of term frequencies

        Returns
        -------
        bytes
            bytearray hasil Elias-Gamma encoding
        """
        return EliasGammaPostings._encode_numbers(tf_list)

    @staticmethod
    def decode_tf(encoded_tf_list):
        """
        Decode list of term frequencies dari stream of bytes.

        Parameters
        ----------
        encoded_tf_list: bytes
            bytearray hasil encoding

        Returns
        -------
        List[int]
            List of term frequencies hasil decoding
        """
        return EliasGammaPostings._decode_numbers(encoded_tf_list)


if __name__ == '__main__':

    postings_list = [34, 67, 89, 454, 2345738]
    tf_list = [12, 10, 3, 4, 1]
    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        print(Postings.__name__)
        encoded_postings_list = Postings.encode(postings_list)
        encoded_tf_list = Postings.encode_tf(tf_list)
        print("byte hasil encode postings: ", encoded_postings_list)
        print("ukuran encoded postings   : ", len(encoded_postings_list), "bytes")
        print("byte hasil encode TF list : ", encoded_tf_list)
        print("ukuran encoded postings   : ", len(encoded_tf_list), "bytes")

        decoded_posting_list = Postings.decode(encoded_postings_list)
        decoded_tf_list = Postings.decode_tf(encoded_tf_list)
        print("hasil decoding (postings): ", decoded_posting_list)
        print("hasil decoding (TF list) : ", decoded_tf_list)
        assert decoded_posting_list == postings_list, \
            f"hasil decoding tidak sama dengan postings original: {decoded_posting_list} != {postings_list}"
        assert decoded_tf_list == tf_list, \
            f"hasil decoding tidak sama dengan TF original: {decoded_tf_list} != {tf_list}"
        print()

    # Test tambahan: edge cases
    print("=== Test Edge Cases ===")
    # Test dengan doc ID dimulai dari 0
    test_postings = [0, 1, 5, 100, 1000]
    test_tf = [1, 1, 1, 1, 1]
    for Postings in [StandardPostings, VBEPostings, EliasGammaPostings]:
        enc_p = Postings.encode(test_postings)
        enc_tf = Postings.encode_tf(test_tf)
        dec_p = Postings.decode(enc_p)
        dec_tf = Postings.decode_tf(enc_tf)
        assert dec_p == test_postings, f"{Postings.__name__}: postings decode gagal"
        assert dec_tf == test_tf, f"{Postings.__name__}: TF decode gagal"
        print(f"{Postings.__name__}: edge case OK, postings size={len(enc_p)} bytes, tf size={len(enc_tf)} bytes")

    print("\nSemua test PASSED!")

"""
Pure-Python implementation of the Porter Stemming Algorithm.

Based on the original paper by Martin Porter (1980):
"An algorithm for suffix stripping"

This implementation removes the dependency on NLTK by providing
a standalone Porter Stemmer using only Python standard library.

The algorithm consists of 5 steps, each applying suffix replacement
rules based on the "measure" (m) of the stem -- roughly the number
of consonant-vowel sequences.

Reference: https://tartarus.org/martin/PorterStemmer/def.txt
"""

import re


class PorterStemmer:
    """
    Porter Stemmer implementation.

    Usage:
        stemmer = PorterStemmer()
        stemmer.stem("running")  # -> "run"
        stemmer.stem("cats")     # -> "cat"
    """

    def __init__(self):
        # Vowels
        self.vowels = frozenset('aeiou')

    def _is_consonant(self, word, i):
        """Check if word[i] is a consonant."""
        if word[i] in self.vowels:
            return False
        if word[i] == 'y':
            if i == 0:
                return True
            return not self._is_consonant(word, i - 1)
        return True

    def _measure(self, stem):
        """
        Compute the "measure" m of a stem.

        m = number of VC (vowel-consonant) sequences.
        [C](VC)^m[V]
        """
        if not stem:
            return 0

        # Build CV pattern
        cv = []
        for i in range(len(stem)):
            if self._is_consonant(stem, i):
                cv.append('C')
            else:
                cv.append('V')

        # Count VC transitions
        pattern = ''.join(cv)
        # Remove leading C's and trailing V's for counting
        m = 0
        i = 0
        n = len(pattern)

        # Skip initial consonants
        while i < n and pattern[i] == 'C':
            i += 1

        while i < n:
            # Count a V sequence
            if i < n and pattern[i] == 'V':
                while i < n and pattern[i] == 'V':
                    i += 1
            else:
                break
            # Count a C sequence
            if i < n and pattern[i] == 'C':
                m += 1
                while i < n and pattern[i] == 'C':
                    i += 1
            else:
                break

        return m

    def _has_vowel(self, stem):
        """Check if stem contains a vowel."""
        for i in range(len(stem)):
            if not self._is_consonant(stem, i):
                return True
        return False

    def _ends_double_consonant(self, word):
        """Check if word ends with a double consonant."""
        if len(word) >= 2 and word[-1] == word[-2]:
            return self._is_consonant(word, len(word) - 1)
        return False

    def _ends_cvc(self, word):
        """
        Check if word ends with consonant-vowel-consonant,
        where the last consonant is not w, x, or y.
        """
        if len(word) >= 3:
            if (self._is_consonant(word, len(word) - 1) and
                not self._is_consonant(word, len(word) - 2) and
                self._is_consonant(word, len(word) - 3)):
                if word[-1] not in ('w', 'x', 'y'):
                    return True
        return False

    def _replace_suffix(self, word, suffix, replacement):
        """Replace suffix if present, return (new_word, matched)."""
        if word.endswith(suffix):
            return word[:-len(suffix)] + replacement, True
        return word, False

    def _step1a(self, word):
        """Step 1a: Plural forms."""
        if word.endswith('sses'):
            return word[:-2]
        if word.endswith('ies'):
            return word[:-2]
        if word.endswith('ss'):
            return word
        if word.endswith('s'):
            return word[:-1]
        return word

    def _step1b(self, word):
        """Step 1b: Past tenses."""
        if word.endswith('eed'):
            stem = word[:-3]
            if self._measure(stem) > 0:
                return word[:-1]
            return word

        changed = False
        if word.endswith('ed'):
            stem = word[:-2]
            if self._has_vowel(stem):
                word = stem
                changed = True
        elif word.endswith('ing'):
            stem = word[:-3]
            if self._has_vowel(stem):
                word = stem
                changed = True

        if changed:
            if word.endswith('at') or word.endswith('bl') or word.endswith('iz'):
                return word + 'e'
            if self._ends_double_consonant(word) and word[-1] not in ('l', 's', 'z'):
                return word[:-1]
            if self._measure(word) == 1 and self._ends_cvc(word):
                return word + 'e'

        return word

    def _step1c(self, word):
        """Step 1c: y -> i."""
        if word.endswith('y'):
            stem = word[:-1]
            if self._has_vowel(stem):
                return stem + 'i'
        return word

    def _step2(self, word):
        """Step 2: Map double suffixes."""
        suffixes = [
            ('ational', 'ate'), ('tional', 'tion'), ('enci', 'ence'),
            ('anci', 'ance'), ('izer', 'ize'), ('abli', 'able'),
            ('alli', 'al'), ('entli', 'ent'), ('eli', 'e'),
            ('ousli', 'ous'), ('ization', 'ize'), ('ation', 'ate'),
            ('ator', 'ate'), ('alism', 'al'), ('iveness', 'ive'),
            ('fulness', 'ful'), ('ousness', 'ous'), ('aliti', 'al'),
            ('iviti', 'ive'), ('biliti', 'ble'),
        ]
        for suffix, replacement in suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 0:
                    return stem + replacement
                return word
        return word

    def _step3(self, word):
        """Step 3: More suffix removal."""
        suffixes = [
            ('icate', 'ic'), ('ative', ''), ('alize', 'al'),
            ('iciti', 'ic'), ('ical', 'ic'), ('ful', ''), ('ness', ''),
        ]
        for suffix, replacement in suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 0:
                    return stem + replacement
                return word
        return word

    def _step4(self, word):
        """Step 4: Remove suffixes with m > 1."""
        suffixes = [
            'al', 'ance', 'ence', 'er', 'ic', 'able', 'ible', 'ant',
            'ement', 'ment', 'ent', 'ou', 'ism', 'ate', 'iti', 'ous',
            'ive', 'ize',
        ]
        for suffix in suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 1:
                    return stem
                return word

        # Special case: -ion
        if word.endswith('ion'):
            stem = word[:-3]
            if self._measure(stem) > 1 and stem and stem[-1] in ('s', 't'):
                return stem

        return word

    def _step5a(self, word):
        """Step 5a: Remove trailing 'e'."""
        if word.endswith('e'):
            stem = word[:-1]
            m = self._measure(stem)
            if m > 1:
                return stem
            if m == 1 and not self._ends_cvc(stem):
                return stem
        return word

    def _step5b(self, word):
        """Step 5b: ll -> l if m > 1."""
        if word.endswith('ll') and self._measure(word[:-1]) > 1:
            return word[:-1]
        return word

    def stem(self, word):
        """
        Stem a word using the Porter algorithm.

        Parameters
        ----------
        word : str
            Word to stem

        Returns
        -------
        str
            Stemmed word
        """
        word = word.lower().strip()

        if len(word) <= 2:
            return word

        word = self._step1a(word)
        word = self._step1b(word)
        word = self._step1c(word)
        word = self._step2(word)
        word = self._step3(word)
        word = self._step4(word)
        word = self._step5a(word)
        word = self._step5b(word)

        return word


if __name__ == "__main__":
    stemmer = PorterStemmer()

    # Test cases from Porter's original paper
    test_cases = [
        ("caresses", "caress"), ("ponies", "poni"), ("ties", "ti"),
        ("caress", "caress"), ("cats", "cat"),
        ("feed", "feed"), ("agreed", "agre"), ("disabled", "disabl"),
        ("matting", "mat"), ("mating", "mate"), ("meeting", "meet"),
        ("milling", "mill"), ("messing", "mess"), ("meetings", "meet"),
        ("filing", "file"), ("failing", "fail"), ("falling", "fall"),
        ("happy", "happi"), ("sky", "sky"),
        ("relational", "relat"), ("conditional", "condit"),
        ("rational", "ration"), ("valenci", "valenc"),
        ("hesitanci", "hesit"), ("digitizer", "digit"),
        ("conformabli", "conform"), ("radicalli", "radic"),
        ("differentli", "differ"), ("vileli", "vile"),
        ("analogousli", "analog"), ("vietnamization", "vietnam"),
        ("predication", "predic"), ("operator", "oper"),
        ("feudalism", "feudal"), ("decisiveness", "decis"),
        ("hopefulness", "hope"), ("callousness", "callous"),
        ("formality", "formal"), ("sensitivity", "sensit"),
        ("sensibility", "sensibl"),
        ("triplicate", "triplic"), ("formative", "form"),
        ("formalize", "formal"), ("electriciti", "electr"),
        ("electrical", "electr"), ("hopeful", "hope"),
        ("goodness", "good"),
        ("revival", "reviv"), ("allowance", "allow"),
        ("inference", "infer"), ("airliner", "airlin"),
        ("gyroscopic", "gyroscop"), ("adjustable", "adjust"),
        ("defensible", "defens"), ("irritant", "irrit"),
        ("replacement", "replac"), ("adjustment", "adjust"),
        ("dependent", "depend"), ("adoption", "adopt"),
        ("homologou", "homolog"), ("communism", "commun"),
        ("activate", "activ"), ("angulariti", "angular"),
        ("homologous", "homolog"), ("effective", "effect"),
        ("bowdlerize", "bowdler"),
        ("probate", "probat"), ("rate", "rate"),
        ("controllable", "control"), ("roll", "roll"),
    ]

    passed = 0
    failed = 0
    for word, expected in test_cases:
        result = stemmer.stem(word)
        if result == expected:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL: stem('{word}') = '{result}', expected '{expected}'")

    print(f"\nPorter Stemmer: {passed}/{passed+failed} tests passed")
    if failed == 0:
        print("All tests PASSED!")

"""
BONUS: Boolean Query Processing

Implements a Boolean query engine supporting AND, OR, NOT operators
with proper precedence and parenthetical grouping.

Grammar (recursive descent parser):
    expr     -> or_expr
    or_expr  -> and_expr ("OR" and_expr)*
    and_expr -> not_expr ("AND" not_expr)*
    not_expr -> "NOT" atom | atom
    atom     -> "(" expr ")" | term

Evaluation uses set operations on posting lists:
    AND -> intersection
    OR  -> union
    NOT -> complement (against all doc IDs)

Examples:
    "blood AND pressure"
    "blood OR plasma"
    "blood AND NOT pressure"
    "(blood OR plasma) AND pressure"
    "blood AND (pressure OR hypertension) AND NOT chronic"
"""

from index import InvertedIndexReader


# ============================================================
# Tokenizer
# ============================================================

def tokenize_boolean_query(query):
    """
    Tokenize a Boolean query string into tokens.

    Recognizes: AND, OR, NOT, (, ), and terms.

    Parameters
    ----------
    query : str
        Boolean query string

    Returns
    -------
    list[str]
        List of tokens
    """
    tokens = []
    i = 0
    query = query.strip()

    while i < len(query):
        # Skip whitespace
        if query[i].isspace():
            i += 1
            continue

        # Parentheses
        if query[i] in '()':
            tokens.append(query[i])
            i += 1
            continue

        # Read a word
        j = i
        while j < len(query) and not query[j].isspace() and query[j] not in '()':
            j += 1
        word = query[i:j]
        tokens.append(word)
        i = j

    return tokens


# ============================================================
# AST Nodes
# ============================================================

class TermNode:
    """Leaf node representing a single term."""
    def __init__(self, term):
        self.term = term

    def __repr__(self):
        return f"Term({self.term})"


class AndNode:
    """Binary AND node."""
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"AND({self.left}, {self.right})"


class OrNode:
    """Binary OR node."""
    def __init__(self, left, right):
        self.left = left
        self.right = right

    def __repr__(self):
        return f"OR({self.left}, {self.right})"


class NotNode:
    """Unary NOT node."""
    def __init__(self, operand):
        self.operand = operand

    def __repr__(self):
        return f"NOT({self.operand})"


# ============================================================
# Recursive Descent Parser
# ============================================================

class BooleanQueryParser:
    """
    Recursive descent parser for Boolean queries.

    Operator precedence (highest to lowest):
        1. NOT (unary)
        2. AND (binary)
        3. OR  (binary)

    Parentheses can override precedence.
    """

    def __init__(self):
        self.tokens = []
        self.pos = 0

    def parse(self, query):
        """
        Parse a Boolean query string into an AST.

        Parameters
        ----------
        query : str
            Boolean query string

        Returns
        -------
        AST node (TermNode, AndNode, OrNode, or NotNode)
        """
        self.tokens = tokenize_boolean_query(query)
        self.pos = 0

        if not self.tokens:
            return None

        result = self._parse_or()

        if self.pos < len(self.tokens):
            raise ValueError(f"Unexpected token: {self.tokens[self.pos]}")

        return result

    def _peek(self):
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def _consume(self):
        token = self.tokens[self.pos]
        self.pos += 1
        return token

    def _parse_or(self):
        """or_expr -> and_expr ("OR" and_expr)*"""
        left = self._parse_and()
        while self._peek() == "OR":
            self._consume()
            right = self._parse_and()
            left = OrNode(left, right)
        return left

    def _parse_and(self):
        """and_expr -> not_expr ("AND" not_expr)*"""
        left = self._parse_not()
        while self._peek() == "AND":
            self._consume()
            right = self._parse_not()
            left = AndNode(left, right)
        return left

    def _parse_not(self):
        """not_expr -> "NOT" atom | atom"""
        if self._peek() == "NOT":
            self._consume()
            operand = self._parse_atom()
            return NotNode(operand)
        return self._parse_atom()

    def _parse_atom(self):
        """atom -> "(" expr ")" | term"""
        if self._peek() == "(":
            self._consume()
            node = self._parse_or()
            if self._peek() != ")":
                raise ValueError("Missing closing parenthesis")
            self._consume()
            return node

        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of query")
        if token in ("AND", "OR", "NOT", ")"):
            raise ValueError(f"Unexpected token: {token}")

        self._consume()
        return TermNode(token)


# ============================================================
# Boolean Query Evaluator
# ============================================================

class BooleanQueryEvaluator:
    """
    Evaluates a Boolean query AST against an inverted index.

    Uses set operations on posting lists:
        - AND: set intersection
        - OR: set union
        - NOT: set complement against all document IDs
    """

    def __init__(self, index_name, postings_encoding, output_dir, term_id_map):
        """
        Parameters
        ----------
        index_name : str
            Name of the inverted index
        postings_encoding : class
            Postings encoding class
        output_dir : str
            Index directory
        term_id_map : IdMap
            Term ID mapping
        """
        self.index_name = index_name
        self.postings_encoding = postings_encoding
        self.output_dir = output_dir
        self.term_id_map = term_id_map

    def evaluate(self, ast, doc_id_map=None):
        """
        Evaluate a Boolean query AST and return matching document IDs.

        Parameters
        ----------
        ast : AST node
            Parsed Boolean query
        doc_id_map : IdMap, optional
            Document ID mapping for converting results to paths

        Returns
        -------
        set[int]
            Set of matching document IDs
        """
        with InvertedIndexReader(self.index_name, self.postings_encoding,
                                 directory=self.output_dir) as reader:
            all_docs = set(reader.doc_length.keys())
            result = self._eval_node(ast, reader, all_docs)

        return result

    def evaluate_with_names(self, ast, doc_id_map):
        """
        Evaluate and return document paths instead of IDs.

        Returns
        -------
        list[str]
            Sorted list of matching document paths
        """
        doc_ids = self.evaluate(ast)
        return sorted([doc_id_map[did] for did in doc_ids])

    def _eval_node(self, node, reader, all_docs):
        """Recursively evaluate an AST node."""
        if isinstance(node, TermNode):
            return self._eval_term(node.term, reader)

        elif isinstance(node, AndNode):
            left = self._eval_node(node.left, reader, all_docs)
            right = self._eval_node(node.right, reader, all_docs)
            return left & right

        elif isinstance(node, OrNode):
            left = self._eval_node(node.left, reader, all_docs)
            right = self._eval_node(node.right, reader, all_docs)
            return left | right

        elif isinstance(node, NotNode):
            operand = self._eval_node(node.operand, reader, all_docs)
            return all_docs - operand

        else:
            raise ValueError(f"Unknown AST node type: {type(node)}")

    def _eval_term(self, term, reader):
        """Get the postings set for a single term."""
        term_id = self.term_id_map[term]
        if term_id in reader.postings_dict:
            postings, _ = reader.get_postings_list(term_id)
            return set(postings)
        return set()


def boolean_search(query, bsbi_index):
    """
    Convenience function for Boolean search.

    Parameters
    ----------
    query : str
        Boolean query (e.g., "blood AND pressure AND NOT chronic")
    bsbi_index : BSBIIndex
        The search index

    Returns
    -------
    list[str]
        Sorted list of matching document paths
    """
    if len(bsbi_index.term_id_map) == 0:
        bsbi_index.load()

    parser = BooleanQueryParser()
    ast = parser.parse(query)

    evaluator = BooleanQueryEvaluator(
        bsbi_index.index_name,
        bsbi_index.postings_encoding,
        bsbi_index.output_dir,
        bsbi_index.term_id_map
    )

    return evaluator.evaluate_with_names(ast, bsbi_index.doc_id_map)


if __name__ == "__main__":
    # Test parser
    parser = BooleanQueryParser()

    test_queries = [
        "blood AND pressure",
        "blood OR plasma",
        "NOT chronic",
        "blood AND NOT chronic",
        "(blood OR plasma) AND pressure",
        "blood AND (pressure OR hypertension) AND NOT chronic",
    ]

    print("=== Parser Tests ===")
    for q in test_queries:
        ast = parser.parse(q)
        print(f"  '{q}' -> {ast}")

    # Test with actual index
    print("\n=== Boolean Search Tests ===")
    from bsbi import BSBIIndex
    from compression import VBEPostings

    BSBI_instance = BSBIIndex(data_dir='collection',
                              postings_encoding=VBEPostings,
                              output_dir='index')

    for q in test_queries:
        results = boolean_search(q, BSBI_instance)
        print(f"\n  '{q}': {len(results)} documents found")
        for doc in results[:5]:
            print(f"    {doc}")
        if len(results) > 5:
            print(f"    ... and {len(results) - 5} more")

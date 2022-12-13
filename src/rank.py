from math import log
from collections import Counter


class Document:
    """Document object that holds information for Brown files."""

    def __init__(self, name: str, sentences: list[str]):
        self.name = name
        self.sentences = sentences
        self.token_counts = self.count_tokens()

    def count_tokens(self) -> Counter[str]:
        counts = Counter()
        for sentence in self.sentences:
            counts.update(sentence)
        return counts


def document_frequencies(vocab: list[str],
                         documents: list[Document]
                         ) -> Counter[str]:
    """Counts the document frequencies for all vocab words in all documents."""
    frequencies = Counter()
    for token in vocab:
        for document in documents:
            if token in document.token_counts:
                frequencies[token] += 1
    return frequencies


def tfidf(term: str,
          document: list[Document],
          frequencies: Counter[str],
          num_documents: int
          ) -> float:
    """Calculates the term-frequency/inverse document frequency of a term."""
    tf = log(document.token_counts[term] + 1, 10)
    idf = log(num_documents / frequencies[term], 10)
    return tf * idf

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


def document_frequencies(vocab, documents) -> Counter[str]:
    frequencies = Counter()
    for token in vocab:
        for document in documents:
            if token in document.token_counts:
                frequencies[token] += 1
    return frequencies


def tfidf(term: str,
          document,
          frequencies: Counter[str],
          num_documents: int
          ) -> float:
    tf = log(document.token_counts[term] + 1, 10)
    idf = log(num_documents / frequencies[term])
    return tf * idf

import torch
from sklearn.cluster import KMeans
from math import log
from collections import Counter


class Document:
    """Document object that holds information for Brown files."""

    def __init__(self, name: str,
                 raw_sentences: list[list[str]],
                 sentences: list[list[str]]):
        self.name = name
        self.raw_sentences = raw_sentences
        self.sentences = sentences
        self.token_counts = self.count_tokens()

    def count_tokens(self) -> Counter[str]:
        counts = Counter()
        for sentence in self.sentences:
            counts.update(sentence)
        return counts


def cluster(big_vectors: list[torch.Tensor], n_clusters: int):
    kmeans = KMeans(n_clusters=n_clusters, n_init='auto')
    return kmeans.fit_predict(big_vectors)


def document_frequencies(documents: list[Document]) -> Counter[str]:
    """Counts the document frequencies for words in all documents."""
    frequencies = Counter()
    for document in documents:
        for token in document.token_counts:
            frequencies[token] += 1
    return frequencies


def tfidf(term: str,
          document: Document,
          frequencies: Counter[str],
          num_documents: int
          ) -> float:
    """Calculates the term-frequency/inverse document frequency of a term."""
    tf = log(document.token_counts[term] + 1, 10)
    idf = log(num_documents / frequencies[term], 10)
    return tf * idf

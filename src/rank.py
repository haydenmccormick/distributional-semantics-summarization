from math import log
from collections import Counter


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

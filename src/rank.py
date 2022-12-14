import torch
from sklearn.cluster import KMeans
from math import log
from collections import Counter
import numpy as np
from nltk import pos_tag


class Document:
    """Document object that holds information for Brown files."""

    def __init__(self, name: str,
                 raw_sentences: list[list[str]],
                 sentences: list[list[str]],
                 tags: list[list[str]]):
        self.name = name
        self.raw_sentences = raw_sentences
        self.sentences = sentences
        self.token_counts = self.count_tokens()
        self.tags = tags

    def count_tokens(self) -> Counter[str]:
        counts = Counter()
        for sentence in self.sentences:
            counts.update(sentence)
        return counts


def cluster(big_vectors: list[torch.Tensor], n_clusters: int):
    kmeans = KMeans(n_clusters=n_clusters)
    return kmeans.fit_predict(big_vectors)


def document_frequencies(documents: list[Document]) -> Counter[str]:
    """Counts the document frequencies for words in all documents."""
    frequencies = Counter()
    for document in documents:
        for token in document.token_counts:
            frequencies[token] += 1
    return frequencies


def tfidf(sentence: list[str],
          document: Document,
          frequencies: Counter[str],
          num_documents: int
          ) -> float:
    """Calculates sum of the term-frequency/inverse document frequency of each
    term in the sentence."""
    total = 0.
    for term in sentence:
        tf = log(document.token_counts[term] + 1, 10)
        idf = log(num_documents / frequencies[term], 10)
        total += tf * idf
    return total


def count_np_vp(tags: list[str],
                index: int) -> (int, int):
    sentence_tags = Counter(tags[index])
    proper_noun_tags = ["np", "np$", "nps", "nps$"]
    noun_tags = ["nn", "nn$", "nns", "nns$", "nr", "nrs"]
    verb_tags = ["vb", "vbd", "vbg", "vbn", "vbz"]
    proper_noun_score = sum([sentence_tags[tag] for tag in proper_noun_tags])
    noun_score = sum([sentence_tags[tag] for tag in noun_tags])
    verb_score = sum([sentence_tags[tag] for tag in verb_tags])
    return noun_score+verb_score, proper_noun_score


def cosine_similarity(vecs: torch.Tensor, index: int):
    if index != 0:
        tensor1 = vecs[index-1]
        tensor2 = vecs[index]
        cos = torch.nn.CosineSimilarity(dim=0)
        return float(cos(tensor1, tensor2))
    else:
        return 0.0

# TODO: Normalization and cue phrases
def rank(sentences: list[list[str]],
         document: Document,
         documents: list[Document],
         vectors: torch.Tensor):
    """Generate ranking for each sentence in a list of sentences"""
    scores = np.zeros(len(sentences))
    freqs = document_frequencies(documents)
    for i, sentence in enumerate(sentences):
        # Sentence length feature
        s_len = len(sentence)
        # Sentence position feature
        s_pos = 1-((i-1)/(len(sentences)))
        # TF-IDF feature
        tfidf_freq = tfidf(sentence, document, freqs, len(documents))
        # NP, VP, and Proper Noun features
        np_vp_score, proper_noun_score = count_np_vp(document.tags, i)
        # Cosine similarity feature
        cos_similarity = cosine_similarity(vectors, i)
        scores[i] = sum([s_len, s_pos, tfidf_freq, np_vp_score, proper_noun_score, cos_similarity])
    return scores


def summarize(clusters, scores, document, n):
    """Produce summary containing only top n sentences per cluster"""
    top_n_sentences = []
    cluster_scores = {cluster_num: Counter() for cluster_num in clusters}
    # Save all sentence scores per cluster in dictionary
    for i, (cluster_n, score) in enumerate(zip(clusters, scores)):
        cluster_scores[cluster_n][i] = score
    # Produce top n indices per cluster
    for cluster_score in cluster_scores.values():
        top_n_sentences.append(cluster_score.most_common(n))
    # Generate list of top n indices
    sentence_indices = [i for sub in top_n_sentences for (i, _) in sub]
    sentence_indices.sort()
    # Finally, pull best sentences from document
    summary = [document.raw_sentences[i] for i in sentence_indices]
    return summary

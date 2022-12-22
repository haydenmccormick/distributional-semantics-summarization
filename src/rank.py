import torch
from sklearn.cluster import KMeans
from math import log
from collections import Counter
import numpy as np
from utils import flatten_sentences


class Document:
    """Document object that holds information for Brown files."""

    def __init__(self, name: str,
                 raw_sentences: list[str],
                 sentences: list[list[str]],
                 tags: list[str]):
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


def cluster(big_vectors: torch.Tensor, n_clusters: int):
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
        # tf = log(document.token_counts[term] + 1, 10)
        tf = document.token_counts[term]  # / sum(document.token_counts.values())
        idf = log(num_documents / frequencies[term], 10)
        total += tf * idf
    return total


def count_np_vp(tags: list[str],
                index: int) -> tuple[float, float]:
    total_length = len(tags)
    sentence_tags = Counter(tags[index])
    proper_noun_tags = ["np", "np$", "nps", "nps$"]
    noun_tags = ["nn", "nn$", "nns", "nns$", "nr", "nrs"]
    verb_tags = ["vb", "vbd", "vbg", "vbn", "vbz"]
    proper_noun_score = sum([sentence_tags[tag] for tag in proper_noun_tags])
    noun_score = sum([sentence_tags[tag] for tag in noun_tags])
    verb_score = sum([sentence_tags[tag] for tag in verb_tags])
    return (noun_score + verb_score) / total_length, proper_noun_score / total_length


def cosine_similarity(vecs: torch.Tensor, index: int):
    if index != 0:
        tensor1 = vecs[index - 1]
        tensor2 = vecs[index]
        cos = torch.nn.CosineSimilarity(dim=0)
        return float(cos(tensor1, tensor2))
    else:
        return 0.0


def normalized_sentence_len(sentence, sentences):
    max_len = max([len(sent) for sent in sentences])
    return len(sentence) / max_len


# TODO: Normalization and cue phrases
def rank(sentences: list[list[str]],
         document: Document,
         documents: list[Document],
         vectors: torch.Tensor):
    """Generate ranking for each sentence in a list of sentences"""
    scores = np.zeros(len(sentences))
    freqs = document_frequencies(documents)

    s_len_values_normalized = np.zeros(len(sentences))
    s_pos_values = np.zeros(len(sentences))
    tfidf_values = np.zeros(len(sentences))
    np_vp_values_normalized = np.zeros(len(sentences))
    pn_values_normalized = np.zeros(len(sentences))
    cosine_similarity_values = np.zeros(len(sentences))

    for i, sentence in enumerate(sentences):
        # Sentence length feature
        s_len = len(sentence)  # normalized_sentence_len(sentence, sentences)
        s_len_values_normalized[i] = s_len
        # Sentence position feature
        s_pos = 1 - (((i+1) - 1) / (len(sentences)))
        s_pos_values[i] = s_pos
        # TF-IDF feature
        tfidf_freq = tfidf(sentence, document, freqs, len(documents))
        tfidf_values[i] = tfidf_freq
        # NP, VP, and Proper Noun features
        np_vp_score, proper_noun_score = count_np_vp(document.tags, i)
        np_vp_values_normalized[i] = np_vp_score
        pn_values_normalized[i] = proper_noun_score
        # Cosine similarity feature
        cos_similarity = cosine_similarity(vectors, i)
        cosine_similarity_values[i] = cos_similarity

    s_len_max = max(s_len_values_normalized)
    tfidf_max = max(tfidf_values)
    cos_similarity_max = max(cosine_similarity_values)

    for i in range(0, len(sentences)):
        scores[i] = sum([
            s_len_values_normalized[i]/s_len_max,
            s_pos_values[i],
            tfidf_values[i] / tfidf_max,
            np_vp_values_normalized[i],
            pn_values_normalized[i],
            cosine_similarity_values[i] / cos_similarity_max
        ])

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
    return flatten_sentences(summary)

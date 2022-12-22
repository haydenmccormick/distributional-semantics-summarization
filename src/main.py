import os
import json
import pickle
import vectors
import utils
import rank
import torch

from nltk.corpus import brown
from rank import Document

from summarizer import Summarizer

from rouge import Rouge

PREPROCESS_PATH: str = "preprocessed.pkl"
SAVE_PATH: str = "scores.json"


def prepare_documents(path: str) -> list[Document]:
    if os.path.exists(path):
        print(f"Found preprocessed file {path}!")
        with open(path, 'rb') as f:
            return pickle.load(f)

    print("Preprocessing...")
    documents = []
    for name in brown.fileids():
        raw_sentences, tags = utils.load_file(name)
        sentences = utils.preprocess(raw_sentences)
        documents.append(Document(name, raw_sentences, sentences, tags))

    # save documents
    with open(path, 'wb') as f:
        pickle.dump(documents, f)

    return documents


def generate_reference_summary(document: Document, n_sents: int):

    # flatten
    body = utils.flatten_sentences(document.raw_sentences)
    model = Summarizer()
    return model(body, num_sentences=n_sents)


def evaluate_summaries(gen_summary: str, ref_summary: str):

    rouge = Rouge()
    scores = rouge.get_scores(gen_summary, ref_summary)

    return scores


def summarize_document(doc: Document,
                       documents: list[Document],
                       glove,
                       n_clusters,
                       verbose=False):
    print(f"Summarizing document {doc.name}")
    print("=========================")
    print("Vectorizing...")
    vecs = torch.stack([vectors.pad_trim(vectors.big_vector(s, glove, 3), 1000)
                        for s in doc.sentences])
    print("Clustering...")
    clusters = rank.cluster(vecs, n_clusters=n_clusters)
    print("Ranking and summarizing...")
    rankings = rank.rank(doc.sentences, doc, documents, vecs)
    gen_summary = rank.summarize(clusters, rankings, doc, 1)

    if verbose:
        print(gen_summary)

    print("Scoring...")
    ref_summary = generate_reference_summary(doc, n_clusters)
    return evaluate_summaries(gen_summary, ref_summary)


def summarize_corpus(documents: list[Document], glove, n_clusters=10):
    scores: list[list[dict]] = []
    for doc in documents:
        if doc.name.startswith("ca"):
            doc_scores = summarize_document(doc, documents, glove, n_clusters)
            scores.append(doc_scores)
    return scores


def main():
    print("Downloading data...")
    utils.download()
    print("Preparing documents...")
    documents = prepare_documents(PREPROCESS_PATH)
    print("Loading GloVe embeddings...")
    glove = vectors.load_glove()
    scores = summarize_corpus(documents, glove)
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(scores, f)


if __name__ == "__main__":
    main()

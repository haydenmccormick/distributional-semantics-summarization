import os
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

def main():
    print("Downloading data...")
    utils.download()
    print("Preparing documents...")
    documents = prepare_documents(PREPROCESS_PATH)
    print("Loading GloVe embeddings...")
    glove = vectors.load_glove()
    print("Vectorizing...")
    ca01 = documents[0]
    vecs = torch.stack([vectors.pad_trim(vectors.big_vector(s, glove, 3), 1000)
                        for s in ca01.sentences])
    print("Clustering...")
    clusters = rank.cluster(vecs, n_clusters=10)
    print("Ranking and summarizing...")
    rankings = rank.rank(ca01.sentences, ca01, documents, vecs)
    gen_summary = rank.summarize(clusters, rankings, ca01, 1)
    print(gen_summary)

    ref_summary = generate_reference_summary(ca01, 10)

    scores = evaluate_summaries(gen_summary, ref_summary)
    print(scores)


if __name__ == "__main__":
    main()

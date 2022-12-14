import os
import pickle
import vectors
import utils
import rank
import torch

from nltk.corpus import brown
from rank import Document

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
    summary = rank.summarize(clusters, rankings, ca01, 1)
    print(summary)





if __name__ == "__main__":
    main()

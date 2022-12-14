import os
import pickle
import vectors
import utils
import rank

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
        print(name)
        raw_sentences = utils.load_file(name)
        sentences = utils.preprocess(raw_sentences)
        documents.append(Document(name, raw_sentences, sentences))

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


if __name__ == "__main__":
    main()

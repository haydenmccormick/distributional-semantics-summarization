import re
import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# from typing import List


def download():
    nltk.download('brown')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


def preprocess(sentences: list[list[str]]) -> list[list[str]]:
    """Preprocessing of input string for summarization algorithm"""
    lemmatizer = WordNetLemmatizer()
    processed_sents = []
    for sent in sentences:
      processed_sent = []
      for word in sent:
        # Strip stop words
        if word not in stopwords.words('english'):
          # Strip URLs
          url_stripped = re.sub(r'https?:\/\/.*?[\s]', '', word)
          # Lemmatize and lowercase
          processed_sent.append(lemmatizer.lemmatize(url_stripped.lower()))
      processed_sents.append(processed_sent)
    return processed_sents


def load_file(name: str) -> list[list[str]]:
    with brown.open(name) as f:
        sentences = []
        tags = []
        for line in f.readlines():
            line = line.strip()
            if line:
                sentence = [pair.split("/")[0] for pair in line.split()]
                sentence_tags = [pair.split("/")[1] for pair in line.split()]
                sentences.append(sentence)
                tags.append(sentence_tags)
    return sentences, tags


def flatten_sentences(sents):
    sent_strings = [" ".join(sent) for sent in sents]
    return " ".join(sent_strings)


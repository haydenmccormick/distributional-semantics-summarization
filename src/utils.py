import re
import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import List


def download():
    nltk.download('brown')
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')


def preprocess(sentences: str) -> List[str]:
    """Preprocessing of input string for summarization algorithm"""
    sentences = sentences.lower()
    # Remove URLs
    sentences = re.sub(r'https?:\/\/.*?[\s]', '', sentences)
    # Tokenize and remove stop words
    tokenized_words = [word for word in word_tokenize(sentences)
                       if word not in stopwords.words('english')]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    processed_sent = []
    for word in tokenized_words:
        processed_sent.append(lemmatizer.lemmatize(word))
    return processed_sent


def load_file(name: str) -> list[list[str]]:
    with brown.open(name) as f:
        sentences = []
        for line in f.readlines():
            line = line.strip()
            if line:
                sentence = [pair.split("/")[0] for pair in line.split()]
                sentences.append(sentence)
    return sentences

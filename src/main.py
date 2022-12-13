import vectors
import utils

from collections import Counter


class Document:
    """Document object that holds information for Brown files."""

    def __init__(self, name: str, sentences: list[str]):
        self.name = name
        self.sentences = sentences
        self.token_counts = self.count_tokens()

    def count_tokens(self) -> Counter[str]:
        counts = Counter()
        for sentence in self.sentences:
            counts.update(sentence)
        return counts


def main():
    pass


if __name__ == "__main__":
    main()

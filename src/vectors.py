import torch
import torch.nn.functional as F
from torchtext.vocab import GloVe


def load_glove(size: int = 6, embed_dim: int = 50) -> GloVe:
    """Loads GloVe embeddings of the specified size and embedding dimension."""
    return GloVe(f"{size}B", embed_dim, cache=f"glove.{size}B")


def big_vector(sentence: list[str], embeddings: GloVe, m: int) -> torch.Tensor:
    """Returns a big vector of the most similar words of each word in the given
    sentence concatenated together."""
    vector = []
    for word in sentence:
        embedding = embeddings[word]
        word_idx = embeddings.stoi[word]
        cos = F.cosine_similarity(embedding, embeddings.vectors)

        # find the most similar word that is not the same word
        max_idx = torch.argmax(cos[cos != 1])

        # adjust index for filtering out same word
        adjusted_idx = max_idx if max_idx < word_idx else max_idx + 1

        similar = embeddings.vectors[adjusted_idx]
        vector.append(similar)
    return torch.cat(vector)

import heapq
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
        if word in embeddings.stoi:
            word_idx = embeddings.stoi[word]
        else:
            continue  # ignore unknown words

        embedding = embeddings[word]
        cos = F.cosine_similarity(embedding, embeddings.vectors)

        top_m_scores = [cos[word_idx]]  # should be 1
        top_m_idxs = [word_idx]
        for i in range(m):
            # find the most similar word that has not been included yet
            score, idx = torch.max(cos[cos < top_m_scores[-1]], dim=0)
            idx = int(idx)
            top_m_scores.append(score)
            top_m_idxs.append(idx)
            top_m_idxs = sorted(top_m_idxs)

            # adjust index for filtering out already included words
            adjusted_idx = idx + top_m_idxs.index(idx)
            vector.append(embeddings.vectors[adjusted_idx])

    return torch.cat(vector)

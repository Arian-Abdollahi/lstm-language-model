"""Word2Vec initialization from tokenized training lines."""

import hashlib

import numpy as np
import torch


def _stable_hash(word):
    return int.from_bytes(hashlib.md5(word.encode()).digest()[:4], "little")


def build_embeddings(sentences, vocabulary, config):
    from gensim.models import Word2Vec

    if any(isinstance(sentence, str) for sentence in sentences):
        raise TypeError("Word2Vec requires token lists, not raw text strings")
    model = Word2Vec(
        sentences=sentences,
        vector_size=config.embedding_dim,
        window=config.word2vec_window,
        min_count=config.word2vec_min_count,
        workers=config.word2vec_workers,
        epochs=config.word2vec_epochs,
        seed=config.seed,
        hashfxn=_stable_hash,
    )
    rng = np.random.default_rng(config.seed)
    matrix = rng.normal(0, 0.6, (len(vocabulary), config.embedding_dim)).astype(np.float32)
    for index, token in enumerate(vocabulary.tokens):
        if token in model.wv:
            matrix[index] = model.wv[token]
    return torch.from_numpy(matrix)

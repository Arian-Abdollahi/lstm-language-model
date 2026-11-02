"""Dependency-free English-style tokenization and a training-only vocabulary."""

import re
from collections import Counter

UNK = "<unk>"
UNK_IDX = 0


def tokenize(text):
    """Lowercase, separate common punctuation, remove double quotes and colon/semicolon."""
    text = re.sub(r"<br\s*/?>", " ", text.lower())
    text = re.sub(r'[";:]', " ", text)
    text = re.sub(r"([.,!?()'])", r" \1 ", text)
    return text.split()


class Vocabulary:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        if not self.tokens or self.tokens[0] != UNK or len(set(self.tokens)) != len(self.tokens):
            raise ValueError("Vocabulary must have unique tokens with <unk> at index 0")
        self.indices = {token: index for index, token in enumerate(self.tokens)}

    @classmethod
    def build(cls, sentences, min_frequency=1):
        frequencies = Counter(token for sentence in sentences for token in sentence)
        words = [
            word
            for word, count in sorted(frequencies.items(), key=lambda x: (-x[1], x[0]))
            if count >= min_frequency and word != UNK
        ]
        return cls([UNK] + words)

    def __len__(self):
        return len(self.tokens)

    def encode(self, tokens):
        return [self.indices.get(token, UNK_IDX) for token in tokens]

    def decode(self, indices):
        return [self.tokens[int(index)] for index in indices]

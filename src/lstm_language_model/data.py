"""Read token files and form shifted next-token examples without artificial targets."""

import math
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .text import UNK_IDX, tokenize

IGNORE_INDEX = -100


def read_sentences(path):
    sentences = [
        tokens
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if (tokens := tokenize(line))
    ]
    if not sentences:
        raise ValueError(f"No tokens found in {path}")
    return sentences


class LanguageModelDataset(Dataset):
    def __init__(self, token_ids, sequence_length=35):
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        self.token_ids = torch.as_tensor(token_ids, dtype=torch.long)
        if self.token_ids.ndim != 1 or len(self.token_ids) < 2:
            raise ValueError("A corpus must contain at least two tokens")
        self.sequence_length = sequence_length

    def __len__(self):
        return math.ceil((len(self.token_ids) - 1) / self.sequence_length)

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        start = index * self.sequence_length
        stop = min(start + self.sequence_length, len(self.token_ids) - 1)
        return self.token_ids[start:stop], self.token_ids[start + 1 : stop + 1]


def make_dataset(sentences, vocabulary, sequence_length):
    ids = vocabulary.encode(token for sentence in sentences for token in sentence)
    return LanguageModelDataset(ids, sequence_length)


def collate_sequences(batch):
    inputs, targets = zip(*batch)
    # Input padding appears only after real tokens in this unidirectional model.
    return (
        pad_sequence(inputs, batch_first=True, padding_value=UNK_IDX),
        pad_sequence(targets, batch_first=True, padding_value=IGNORE_INDEX),
    )


def make_loader(sentences, vocabulary, config, *, training=False, device=None):
    return DataLoader(
        make_dataset(sentences, vocabulary, config.sequence_length),
        batch_size=config.batch_size,
        shuffle=training,
        num_workers=config.num_workers,
        collate_fn=collate_sequences,
        generator=torch.Generator().manual_seed(config.seed),
        pin_memory=device is not None and device.type == "cuda",
    )

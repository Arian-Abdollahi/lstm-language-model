"""Trainable word embeddings, embedding dropout, stacked LSTM, and vocabulary projection."""

import torch
from torch import nn


class LanguageModel(nn.Module):
    def __init__(self, config, vocab_size, pretrained_embeddings=None):
        super().__init__()
        config.validate()
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim)
        if pretrained_embeddings is None:
            nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        else:
            if tuple(pretrained_embeddings.shape) != (vocab_size, config.embedding_dim):
                raise ValueError("Embedding shape must match vocabulary and embedding_dim")
            with torch.no_grad():
                self.embedding.weight.copy_(pretrained_embeddings)
        self.embedding.weight.requires_grad_(not config.freeze_embeddings)
        self.dropout = nn.Dropout(config.dropout_embedding)
        self.lstm = nn.LSTM(
            config.embedding_dim,
            config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout_lstm if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Linear(config.hidden_dim, vocab_size)

    def forward(self, tokens, state=None):
        if tokens.ndim != 2 or tokens.size(1) == 0:
            raise ValueError("Expected nonempty token batches with shape [batch, time]")
        output, state = self.lstm(self.dropout(self.embedding(tokens)), state)
        return self.fc(output), state

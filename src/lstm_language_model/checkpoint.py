"""State-dict checkpoints with vocabulary, configuration, and corpus fingerprints."""

from pathlib import Path

import torch

from .config import Config
from .model import LanguageModel
from .text import Vocabulary

TOKENIZER_VERSION = "basic_english_style_v1"


def save_checkpoint(path, model, optimizer, config, vocabulary, corpus_hashes, epoch, metrics):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "tokenizer": TOKENIZER_VERSION,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.to_dict(),
            "vocabulary": vocabulary.tokens,
            "corpus_sha256": corpus_hashes,
            "epoch": epoch,
            "metrics": metrics,
        },
        temporary,
    )
    temporary.replace(path)


def load_model(path, device):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or payload.get("tokenizer") != TOKENIZER_VERSION:
        raise ValueError("Use a checkpoint produced by this package and tokenizer version")
    config = Config.from_dict(payload["config"])
    vocabulary = Vocabulary(payload["vocabulary"])
    model = LanguageModel(config, len(vocabulary))
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, vocabulary, config, payload

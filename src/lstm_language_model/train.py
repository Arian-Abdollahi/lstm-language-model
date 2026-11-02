"""Train an LSTM language model: python -m lstm_language_model.train --help."""

import argparse
import platform
from pathlib import Path

import numpy as np
import torch

from .checkpoint import TOKENIZER_VERSION, save_checkpoint
from .config import Config
from .data import make_loader, read_sentences
from .embeddings import build_embeddings
from .engine import run_epoch
from .model import LanguageModel
from .text import Vocabulary
from .utils import fingerprint, seed_everything, select_device, write_json


def train(config, device="auto"):
    config.validate()
    seed_everything(config.seed)
    device = select_device(device)
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}; choose a new run folder")
    corpora = {
        split: read_sentences(config.split_path(split)) for split in ("train", "validation", "test")
    }
    vocabulary = Vocabulary.build(corpora["train"], config.min_frequency)
    if len(vocabulary) < 2:
        raise ValueError("Vocabulary has no known words; check the corpus and min_frequency")
    train_loader = make_loader(corpora["train"], vocabulary, config, training=True, device=device)
    validation_loader = make_loader(corpora["validation"], vocabulary, config, device=device)
    weights = (
        build_embeddings(corpora["train"], vocabulary, config) if config.use_word2vec else None
    )
    model = LanguageModel(config, len(vocabulary), weights).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    hashes = {split: fingerprint(config.split_path(split)) for split in corpora}
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", config.to_dict())
    write_json(output / "vocabulary.json", vocabulary.tokens)
    write_json(
        output / "corpus.json",
        {
            "sha256": hashes,
            "tokenizer": TOKENIZER_VERSION,
            "counts": {
                split: {"nonempty_lines": len(lines), "tokens": sum(len(line) for line in lines)}
                for split, lines in corpora.items()
            },
        },
    )
    write_json(
        output / "environment.json",
        {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": str(np.__version__),
            "device": str(device),
        },
    )
    history = []
    best_loss = float("inf")
    print(
        f"Device: {device}; vocabulary: {len(vocabulary)}; "
        f"trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    for epoch in range(1, config.epochs + 1):
        training = run_epoch(model, train_loader, device, optimizer, config.gradient_clip)
        validation = run_epoch(model, validation_loader, device)
        metrics = {"epoch": epoch, "train": training, "validation": validation}
        history.append(metrics)
        if validation["loss"] < best_loss:
            best_loss = validation["loss"]
            save_checkpoint(
                output / "best.pt", model, optimizer, config, vocabulary, hashes, epoch, metrics
            )
        save_checkpoint(
            output / "last.pt", model, optimizer, config, vocabulary, hashes, epoch, metrics
        )
        write_json(output / "history.json", history)
        print(
            f"Epoch {epoch}: train loss={training['loss']:.4f}; "
            f"validation loss={validation['loss']:.4f}; "
            f"validation perplexity={validation['perplexity']:.2f}"
        )
    return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-word2vec", action="store_true", help="Use random word embeddings")
    args = parser.parse_args(argv)
    config = Config.load(args.config)
    for name in ("data_dir", "output_dir", "epochs", "batch_size", "num_workers"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    if args.no_word2vec:
        config.use_word2vec = False
    train(config, args.device)


if __name__ == "__main__":
    main()

"""JSON experiment configuration; file paths are relative to the current directory."""

import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Config:
    data_dir: str = "data/wikitext-2"
    train_file: str = "wiki.train.tokens"
    validation_file: str = "wiki.valid.tokens"
    test_file: str = "wiki.test.tokens"
    output_dir: str = "artifacts"
    seed: int = 42
    sequence_length: int = 35
    batch_size: int = 20
    num_workers: int = 0
    min_frequency: int = 1
    embedding_dim: int = 300
    hidden_dim: int = 512
    num_layers: int = 2
    dropout_embedding: float = 0.25
    dropout_lstm: float = 0.25
    freeze_embeddings: bool = False
    use_word2vec: bool = True
    word2vec_window: int = 25
    word2vec_min_count: int = 5
    word2vec_epochs: int = 5
    word2vec_workers: int = 1
    epochs: int = 2
    learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    gradient_clip: float = 0.25

    def validate(self):
        for name in (
            "seed",
            "sequence_length",
            "batch_size",
            "num_workers",
            "min_frequency",
            "embedding_dim",
            "hidden_dim",
            "num_layers",
            "word2vec_window",
            "word2vec_min_count",
            "word2vec_epochs",
            "word2vec_workers",
            "epochs",
        ):
            value = getattr(self, name)
            minimum = 0 if name in {"seed", "num_workers"} else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        for name in ("dropout_embedding", "dropout_lstm"):
            if not 0 <= getattr(self, name) < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        for name in ("learning_rate", "gradient_clip", "weight_decay"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name != "weight_decay" and value == 0):
                raise ValueError(f"Invalid {name}")
        if type(self.use_word2vec) is not bool or type(self.freeze_embeddings) is not bool:
            raise ValueError("use_word2vec and freeze_embeddings must be JSON booleans")
        paths = [self.split_path(split).resolve() for split in ("train", "validation", "test")]
        if len(set(paths)) != 3:
            raise ValueError("Training, validation, and test files must be different paths")
        return self

    def split_path(self, split):
        filename = {
            "train": self.train_file,
            "validation": self.validation_file,
            "test": self.test_file,
        }[split]
        return Path(self.data_dir) / filename

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        unknown = set(values) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**values).validate()

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

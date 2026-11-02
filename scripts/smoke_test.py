"""Run every CLI on a tiny generated corpus; no dataset or model download."""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from lstm_language_model.config import Config
from lstm_language_model.text import tokenize


def run(module, *arguments, env):
    result = subprocess.run(
        [sys.executable, "-m", module, *arguments],
        check=True,
        env=env,
        text=True,
        capture_output=True,
    )
    print(result.stdout.strip())
    return result


def main():
    with tempfile.TemporaryDirectory(prefix="lstm-smoke-") as directory:
        root = Path(directory)
        lines = {
            "train": "i want to travel to the island .\npeople play in the park .\n" * 20,
            "validation": "i want to play in the park .\n" * 4,
            "test": "people travel to the island .\n" * 4,
        }
        config = Config(
            data_dir=str(root),
            output_dir=str(root / "artifacts"),
            embedding_dim=8,
            hidden_dim=12,
            num_layers=2,
            sequence_length=5,
            batch_size=4,
            epochs=1,
            learning_rate=0.01,
            word2vec_min_count=1,
        )
        for split, text in lines.items():
            config.split_path(split).write_text(text, encoding="utf-8")
        config_path = root / "config.json"
        config_path.write_text(json.dumps(config.to_dict()), encoding="utf-8")
        env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
        run(
            "lstm_language_model.analyze",
            "--config",
            str(config_path),
            "--output",
            str(root / "analysis.json"),
            env=env,
        )
        run("lstm_language_model.train", "--config", str(config_path), "--device", "cpu", env=env)
        checkpoint = str(root / "artifacts/best.pt")
        run(
            "lstm_language_model.evaluate",
            "--checkpoint",
            checkpoint,
            "--device",
            "cpu",
            "--output",
            str(root / "test.json"),
            env=env,
        )
        metrics = json.loads((root / "test.json").read_text())
        expected_tokens = len(tokenize(lines["test"])) - 1
        assert metrics["tokens"] == expected_tokens
        output = run(
            "lstm_language_model.generate",
            "--checkpoint",
            checkpoint,
            "--prompt",
            "i want",
            "--max-new-tokens",
            "6",
            "--top-k",
            "3",
            "--device",
            "cpu",
            env=env,
        )
        assert output.stdout.startswith("i want ")
        path = config.split_path("test")
        path.write_text(lines["test"] + "\n", encoding="utf-8")
        rejected = subprocess.run(
            [
                sys.executable,
                "-m",
                "lstm_language_model.evaluate",
                "--checkpoint",
                checkpoint,
                "--device",
                "cpu",
            ],
            env=env,
            capture_output=True,
            text=True,
        )
        assert rejected.returncode != 0 and "differs from training metadata" in rejected.stderr
        print("Smoke test passed: inspect, train, reload, evaluate, generate, and corpus guard.")


if __name__ == "__main__":
    main()

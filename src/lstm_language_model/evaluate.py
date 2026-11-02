"""Measure held-out next-token loss and perplexity with a saved checkpoint."""

import argparse
import json

from .checkpoint import load_model
from .data import make_loader, read_sentences
from .engine import run_epoch
from .utils import fingerprint, select_device, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--data-dir", help="Override the checkpoint's corpus directory")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", help="Optional output JSON path")
    args = parser.parse_args(argv)
    device = select_device(args.device)
    model, vocabulary, config, payload = load_model(args.checkpoint, device)
    if args.data_dir:
        config.data_dir = args.data_dir
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    config.validate()
    path = config.split_path(args.split)
    if fingerprint(path) != payload["corpus_sha256"][args.split]:
        raise ValueError("Evaluation corpus differs from training metadata; use the original split")
    loader = make_loader(read_sentences(path), vocabulary, config, device=device)
    result = {
        "split": args.split,
        "checkpoint_epoch": payload["epoch"],
        **run_epoch(model, loader, device),
    }
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.output:
        write_json(args.output, result)


if __name__ == "__main__":
    main()

"""Inspect corpus sizes, common words, and unknown-token rates."""

import argparse
import json
from collections import Counter

from .config import Config
from .data import read_sentences
from .text import UNK_IDX, Vocabulary
from .utils import write_json


def analyze(config):
    corpora = {
        split: read_sentences(config.split_path(split)) for split in ("train", "validation", "test")
    }
    vocabulary = Vocabulary.build(corpora["train"], config.min_frequency)
    results = {"vocabulary_size": len(vocabulary), "splits": {}}
    for split, lines in corpora.items():
        counts = Counter(token for line in lines for token in line)
        total = sum(counts.values())
        unknown = sum(
            count
            for word, count in counts.items()
            if vocabulary.indices.get(word, UNK_IDX) == UNK_IDX
        )
        results["splits"][split] = {
            "nonempty_lines": len(lines),
            "tokens": total,
            "unique_tokens": len(counts),
            "mean_tokens_per_nonempty_line": total / len(lines),
            "unk_token_fraction": unknown / total,
            "most_common": counts.most_common(20),
        }
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--data-dir")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    config = Config.load(args.config)
    if args.data_dir:
        config.data_dir = args.data_dir
    result = analyze(config.validate())
    print(json.dumps(result, indent=2))
    if args.output:
        write_json(args.output, result)


if __name__ == "__main__":
    main()

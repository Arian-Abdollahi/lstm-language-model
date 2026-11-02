"""Optional plotting of history files from newly trained experiments."""

import json
from pathlib import Path


def plot_history(path):
    import matplotlib.pyplot as plt

    history = json.loads(Path(path).read_text(encoding="utf-8"))
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric in zip(axes, ("loss", "perplexity")):
        for split in ("train", "validation"):
            axis.plot(
                [row["epoch"] for row in history],
                [row[split][metric] for row in history],
                label=split,
            )
        axis.set(xlabel="Epoch", ylabel=metric.capitalize())
        axis.grid(True)
        axis.legend()
    figure.tight_layout()
    return figure

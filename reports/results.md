# Historical results and interpretation

Source: [the saved version 3 Kaggle notebook](https://www.kaggle.com/code/arianabdollahi/language-model-with-lstm), inspected on 2026-11-03.

The displayed final training invocation (`In [173]`) reports:

| Displayed local epoch | Training loss | Training reported perplexity | Validation loss | Validation reported perplexity |
| --- | --- | --- | --- | --- |
| First | 3.99 | 54.1 | 4.9 | 134.2 |
| Second | 3.98 | 53.3 | 4.901 | 134.4 |

These values are transcribed from saved output. The source export has no saved cell outputs, and its source does not define the global `metric` object used by training. The precise metric construction and complete sequence of interactive runs cannot be independently recovered from the export. No final test-set score is displayed.

The [saved curve](../figures/kaggle_learning_curve.png) shows 16 points, with training loss falling to approximately 4.0 while validation loss levels off around 4.9. The widening gap suggests overfitting in the displayed run, but the underlying full history and intermediate environment are unavailable. This is an interpretation of the saved plot, not an additional experiment.

Saved prompt/output pairs include:

| Prompt | Saved output |
| --- | --- |
| `i want to travel` | `i want to travel to the island .` |
| `people play` | `people play and the new york times , and that they were not able to make up the same name .` |

The notebook also contains grammatically weak or factually unreliable continuations. The selected README sample illustrates one plausible completion; it does not establish general generation quality.

The package corrects Word2Vec input, sequence tails, missing globals, generation filtering, and serialization. It therefore needs a separate real-data experiment. Small local tests confirm execution and serialization only. Historical Kaggle scores must not be reported as measured performance of the refactored package.

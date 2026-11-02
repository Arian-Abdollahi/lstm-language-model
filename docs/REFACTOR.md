# From notebook to reusable project

Original source: [Language Model with LSTM, Arian Abdollahi, version 3](https://www.kaggle.com/code/arianabdollahi/language-model-with-lstm). The 85-cell source export is preserved without edits in `notebooks/language_model_original.ipynb` (SHA-256 `29dfa1466d0df5eb6e221ce72039533e22396a3688b0473db17894488868c97a`). It includes IPython installation commands and was not executed end-to-end.

## Preserved approach

The package retains a training-only word vocabulary, 300-dimensional Word2Vec-initialized trainable embeddings, embedding dropout, two unidirectional LSTM layers with 512 hidden units, a vocabulary projection, cross-entropy training with AdamW, gradient clipping, held-out validation, and temperature-based next-token sampling. Default sequence length 35, batch size 20, dropout 0.25, learning rate 1e-5, and weight decay 1e-4 match the notebook's final training configuration.

## Changes

| Notebook behavior | Package behavior |
| --- | --- |
| Installs torch 2.3.0 and torchtext 0.18.0 in a notebook cell | Uses a modern PyTorch package without torchtext, torchvision, or torchaudio |
| Uses torchtext's `basic_english` tokenizer and serialized vocabulary object | Defines explicit English-style punctuation rules in `text.py` and saves a plain vocabulary list |
| Supplies raw joined strings to Word2Vec | Supplies lists of word tokens; raw strings raise an error |
| Appends unknown ID 0 when corpus length is divisible by sequence length | Forms only genuine next-token pairs; retains and masks the final short window |
| Uses out-of-order `device`, `metric`, `clip`, histories, and file globals | Passes configuration/state explicitly; defines loss and perplexity in the training engine |
| Training ignores its supplied validation-loader argument in favor of a global | Uses the supplied validation loader |
| Mixes `from tqdm import tqdm` and `import tqdm` with `tqdm.tqdm` calls | Uses one consistent `tqdm.auto.tqdm` import |
| Saves complete Python model/vocabulary objects | Saves state dictionaries, plain vocabulary/config, corpus fingerprints, and optimizer state |
| Rejects unknown samples using a `<ukn>` typo and an unbounded retry loop | Masks the actual `<unk>` ID before sampling and fails clearly if all tokens are blocked |
| Reprocesses the full generated prefix each step | Processes the prompt once, then carries recurrent state for one-token steps |
| Does not validate generation temperature or empty prompts | Validates inputs, caps generated length, supports seeded sampling and optional top-k filtering |
| Has an empty final test section | Provides a separate held-out evaluation command |

The tokenizer rules are explicit and tested but are not claimed to be byte-for-byte equivalent to every torchtext version or corpus edge case. Vocabulary ties use lexical ordering for stability. These differences, the corrected word embeddings, and sequence handling mean the old notebook pickles are not directly compatible: retrain with this package.

Training resets hidden state for each sequence batch, matching the notebook's independent-window setup. The last short window is padded on the right and the loss excludes only target -100; real `<unk>` targets still contribute. Validation and test use their official files, without random splitting. No sentence/EOS token is inserted when flattening lines.

## Reproducibility and limits

The package seeds Python, NumPy, PyTorch, and the data-loader generator. Word2Vec uses one worker and a stable initialization hash by default. Checkpoints record the tokenizer identifier, vocabulary order, configuration, split-file SHA-256 digests, selected epoch, and metrics; basic runtime versions are saved separately. Evaluation rejects a modified selected split, even if only whitespace changed.

These controls do not guarantee identical results across devices, platforms, or library versions. Top-level tested versions are recorded separately from the original notebook's displayed NumPy 1.26.4, torch 2.3.0+cu121, and torchtext 0.18.0+cpu environment. No complete original environment lock is available. The package has no resume CLI despite retaining optimizer state. New runs require empty output directories.

The model is a small word-level research baseline, not an instruction-following assistant. It is not fine-tuned for factual answering, and plausible generated text can be incorrect. Full WikiText-2 training, GPU execution, and model-quality benchmarking remain to be run.

Implementation references: [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html), [PyTorch state-dict serialization](https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html), and [Gensim Word2Vec](https://radimrehurek.com/gensim/models/word2vec.html).

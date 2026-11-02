# WikiText-2 setup

The original notebook reads three UTF-8 text files from a Kaggle input named `wikitext2`:

```text
/kaggle/input/wikitext2/wiki.train.tokens
/kaggle/input/wikitext2/wiki.valid.tokens
/kaggle/input/wikitext2/wiki.test.tokens
```

That input is displayed as private, so the repository does not assume it can be downloaded automatically. Obtain matching WikiText-2 files through your existing dataset source. The [Salesforce WikiText dataset card](https://huggingface.co/datasets/Salesforce/wikitext) describes the dataset and its variants; use the pretokenized **WikiText-2** variant rather than WikiText-103 or the raw variant when reproducing this experiment. Alternate exports or variants may produce different token counts and metrics.

Place the files under `data/wikitext-2/`, or set `data_dir` and the three filenames in `configs/default.json`. Any UTF-8 plain-text corpus with separate train/validation/test files can be used for an experiment, but its results are not a WikiText-2 benchmark. The official split boundaries are preserved; the code does not randomly re-split lines.

The tokenizer lowercases text, separates common punctuation and apostrophes, removes double quotes/colons/semicolons, and splits whitespace. Literal `<unk>` tokens remain intact. Vocabulary and Word2Vec fitting use only the training split. Nonempty lines supply the Word2Vec sentence boundaries; the language-model corpus concatenates those lines without inserting an extra EOS token, following the notebook's approach.

Every available next-token pair is retained, including the final short window. Padding uses input ID 0 and target value -100; only the latter is ignored by the loss. The unidirectional model's padded input suffix does not affect preceding real-token predictions. `<unk>` at ID 0 is an ordinary, scored vocabulary item when it occurs in real targets.

Each corpus needs at least two tokens. Training, validation, and test must use distinct file paths. Full corpus files, tokenizer caches, embeddings, and trained checkpoints are not distributed here. Follow the dataset provider's terms for any downloaded text.

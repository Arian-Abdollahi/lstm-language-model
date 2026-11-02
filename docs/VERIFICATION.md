# Verification

Verified on **2026-11-03** with Python **3.12.14**, PyTorch **2.15.0+cpu**, NumPy **2.3.5**, Gensim **4.4.0**, and tqdm **4.70.1**. Top-level versions are recorded in [requirements-tested.txt](../requirements-tested.txt).

| Check | Result |
| --- | --- |
| Editable package installation | Passed |
| Dependency consistency (`pip check`) | Passed |
| Ruff lint and formatting | Passed |
| Unit tests | 11 passed |
| CLI integration | Corpus inspection, one CPU training epoch, checkpoint reload, test evaluation, and seeded text generation passed |
| Evaluation fingerprint guard | Changed test file rejected |
| Original notebook | 85 source cells preserved; source SHA-256 matches the retrieved export |
| Guided notebook | 12 cells; Python code cells parsed |
| Saved plot | Original PNG retained unchanged with a recorded SHA-256 |

The semantic tests check punctuation/unknown-token handling, stable vocabulary order, genuine shifted targets, final short windows, ignored padding, invalid corpora, causal recurrence, equivalence of cached-state and full-prefix inference, token-level loss/perplexity, deterministic Word2Vec initialization, rejection of raw Word2Vec strings, parameter updates, checkpoint round-trips, seeded sampling, unknown-token masking, period stopping, and invalid/all-masked generation inputs.

`python scripts/smoke_test.py` creates small separate text files in a temporary directory and exercises all four CLIs in separate Python processes. It uses actual Word2Vec and a small LSTM, verifies that every real test target is counted, and removes temporary files afterward. Its outputs do not measure WikiText-2 model quality.

Full WikiText-2 training, CUDA execution, multiworker loading, convergence, and quality benchmarks were not run. The historical notebook environment and hidden interactive state have not been reproduced. The original notebook includes IPython installation commands and is preserved for inspection rather than treated as a plain executable Python script.

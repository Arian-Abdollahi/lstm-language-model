"""Semantic checks using tiny local fixtures, without WikiText-2 downloads."""

import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from lstm_language_model.checkpoint import load_model, save_checkpoint
from lstm_language_model.config import Config
from lstm_language_model.data import IGNORE_INDEX, LanguageModelDataset, collate_sequences
from lstm_language_model.embeddings import build_embeddings
from lstm_language_model.engine import run_epoch
from lstm_language_model.generate import generate
from lstm_language_model.model import LanguageModel
from lstm_language_model.text import UNK_IDX, Vocabulary, tokenize

torch.set_num_threads(2)


class PipelineTests(unittest.TestCase):
    def config(self, **changes):
        return Config(
            embedding_dim=8,
            hidden_dim=12,
            num_layers=2,
            dropout_embedding=0,
            dropout_lstm=0,
            **changes,
        )

    def test_tokenizer_handles_punctuation_and_literal_unknowns(self):
        self.assertEqual(
            tokenize('Hello, WORLD! <unk> "word"; next: now.'),
            ["hello", ",", "world", "!", "<unk>", "word", "next", "now", "."],
        )

    def test_vocabulary_uses_training_words_and_stable_ties(self):
        train = [tokenize("a dog runs ."), tokenize("a cat runs .")]
        vocab = Vocabulary.build(train)
        self.assertEqual(vocab.tokens, Vocabulary.build(list(reversed(train))).tokens)
        self.assertEqual(vocab.encode(["unseenword"]), [UNK_IDX])
        self.assertEqual(vocab.decode(vocab.encode(["dog", "."])), ["dog", "."])

    def test_shifted_pairs_cover_the_corpus_without_invented_targets(self):
        dataset = LanguageModelDataset(list(range(8)), sequence_length=4)
        inputs = torch.cat([dataset[i][0] for i in range(len(dataset))])
        targets = torch.cat([dataset[i][1] for i in range(len(dataset))])
        self.assertEqual(inputs.tolist(), list(range(7)))
        self.assertEqual(targets.tolist(), list(range(1, 8)))

    def test_last_partial_window_is_retained_and_padded_targets_are_ignored(self):
        dataset = LanguageModelDataset([1, 2, 3, 4, 5, 6], sequence_length=4)
        inputs, targets = collate_sequences([dataset[0], dataset[1]])
        self.assertEqual(tuple(inputs.shape), (2, 4))
        self.assertEqual(targets[1].tolist(), [6, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX])
        self.assertEqual(int(targets.ne(IGNORE_INDEX).sum()), 5)

    def test_empty_and_invalid_corpora_fail(self):
        with self.assertRaisesRegex(ValueError, "two tokens"):
            LanguageModelDataset([1])
        with self.assertRaises(ValueError):
            LanguageModelDataset([1, 2], sequence_length=0)
        with self.assertRaises(IndexError):
            LanguageModelDataset([1, 2])[1]

    def test_lstm_is_causal_and_cached_state_matches_full_prefix(self):
        model = LanguageModel(self.config(), 9).eval()
        with torch.no_grad():
            full, _ = model(torch.tensor([[1, 2, 3, 4]]))
            changed, _ = model(torch.tensor([[1, 2, 3, 8]]))
            _, state = model(torch.tensor([[1, 2, 3]]))
            incremental, _ = model(torch.tensor([[4]]), state)
        torch.testing.assert_close(full[:, :3], changed[:, :3])
        torch.testing.assert_close(full[:, -1], incremental[:, -1])

    def test_padding_does_not_change_loss_or_perplexity(self):
        class UniformModel(nn.Module):
            def forward(self, inputs):
                return torch.zeros(*inputs.shape, 7), None

        first = run_epoch(
            UniformModel(),
            [(torch.ones(1, 2, dtype=torch.long), torch.tensor([[1, 2]]))],
            torch.device("cpu"),
            progress=False,
        )
        padded = run_epoch(
            UniformModel(),
            [(torch.ones(1, 4, dtype=torch.long), torch.tensor([[1, 2, -100, -100]]))],
            torch.device("cpu"),
            progress=False,
        )
        self.assertAlmostEqual(first["loss"], math.log(7), places=6)
        self.assertEqual(first["tokens"], padded["tokens"])
        self.assertAlmostEqual(first["loss"], padded["loss"], places=6)
        self.assertAlmostEqual(padded["perplexity"], 7, places=5)

    def test_word2vec_uses_tokens_and_repeats_with_fixed_seed(self):
        config = self.config(word2vec_min_count=1)
        sentences = [tokenize("a dog runs ."), tokenize("a cat runs .")]
        vocab = Vocabulary.build(sentences)
        first = build_embeddings(sentences, vocab, config)
        second = build_embeddings(sentences, vocab, config)
        self.assertEqual(tuple(first.shape), (len(vocab), 8))
        self.assertTrue(torch.equal(first, second))
        with self.assertRaisesRegex(TypeError, "token lists"):
            build_embeddings(["a dog runs ."], vocab, config)

    def test_training_checkpoint_roundtrip_and_seeded_generation(self):
        config = self.config()
        vocab = Vocabulary.build([tokenize("i want to travel .")])
        model = LanguageModel(config, len(vocab))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        inputs = torch.tensor([[1, 2, 3], [3, 2, 1]])
        targets = torch.tensor([[2, 3, 4], [2, 1, 4]])
        before = model.fc.weight.detach().clone()
        run_epoch(model, [(inputs, targets)], torch.device("cpu"), optimizer, progress=False)
        self.assertFalse(torch.equal(before, model.fc.weight))
        model.eval()
        with torch.no_grad():
            expected, _ = model(inputs)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, optimizer, config, vocab, {"test": "fixture"}, 1, {})
            restored, restored_vocab, _, payload = load_model(path, torch.device("cpu"))
            with torch.no_grad():
                actual, _ = restored(inputs)
            torch.testing.assert_close(expected, actual)
            self.assertEqual(payload["epoch"], 1)
            first = generate(restored, restored_vocab, "i want", max_new_tokens=5, seed=9)
            second = generate(restored, restored_vocab, "i want", max_new_tokens=5, seed=9)
            self.assertEqual(first, second)
            self.assertNotIn("<unk>", first)

    def test_generation_masks_unknowns_and_stops_at_period(self):
        vocab = Vocabulary(["<unk>", "word", "."])
        model = LanguageModel(self.config(), len(vocab))
        with torch.no_grad():
            model.fc.weight.zero_()
            model.fc.bias.copy_(torch.tensor([100.0, 0.0, 10.0]))
        self.assertEqual(generate(model, vocab, "word", top_k=1), "word .")

    def test_bad_generation_arguments_and_all_masked_logits_fail(self):
        vocab = Vocabulary(["<unk>"])
        model = LanguageModel(self.config(), len(vocab))
        for kwargs in ({"temperature": 0}, {"prompt": " "}, {"top_k": -1}, {"max_new_tokens": 0}):
            arguments = dict(prompt="word", **{k: v for k, v in kwargs.items() if k != "prompt"})
            if "prompt" in kwargs:
                arguments["prompt"] = kwargs["prompt"]
            with self.assertRaises(ValueError):
                generate(model, vocab, **arguments)
        with self.assertRaisesRegex(ValueError, "No tokens"):
            generate(model, vocab, "word")


if __name__ == "__main__":
    unittest.main()

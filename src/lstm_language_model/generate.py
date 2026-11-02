"""Autoregressive text generation with bounded sampling and reusable LSTM state."""

import argparse
import math

import torch

from .checkpoint import load_model
from .text import UNK_IDX, tokenize
from .utils import select_device


@torch.inference_mode()
def generate(
    model,
    vocabulary,
    prompt,
    max_new_tokens=35,
    temperature=0.5,
    top_k=0,
    seed=42,
    allow_unk=False,
    stop_token=".",
):
    if type(max_new_tokens) is not int or max_new_tokens < 1:
        raise ValueError("max_new_tokens must be a positive integer")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and > 0")
    if type(top_k) is not int or top_k < 0:
        raise ValueError("top_k must be a nonnegative integer")
    tokens = tokenize(prompt)
    if not tokens:
        raise ValueError("Prompt must contain at least one token")
    model.eval()
    device = next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    current = torch.tensor([vocabulary.encode(tokens)], dtype=torch.long, device=device)
    state = None
    generated = []
    stop_index = vocabulary.indices.get(stop_token)
    for _ in range(max_new_tokens):
        logits, state = model(current, state)
        scores = logits[0, -1].clone() / temperature
        if not allow_unk:
            scores[UNK_IDX] = float("-inf")
        if top_k:
            values, indices = scores.topk(min(top_k, len(vocabulary)))
            scores.fill_(float("-inf"))
            scores[indices] = values
        if not torch.isfinite(scores).any():
            raise ValueError("No tokens are available after sampling filters")
        index = int(torch.multinomial(scores.softmax(-1), 1, generator=generator).item())
        generated.append(vocabulary.tokens[index])
        if index == stop_index:
            break
        current = torch.tensor([[index]], dtype=torch.long, device=device)
    return prompt.strip() + " " + " ".join(generated)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=35)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument(
        "--top-k", type=int, default=0, help="0 samples the full allowed vocabulary"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-unk", action="store_true")
    parser.add_argument("--stop-token", default=".")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    model, vocabulary, _, _ = load_model(args.checkpoint, select_device(args.device))
    print(
        generate(
            model,
            vocabulary,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_k,
            args.seed,
            args.allow_unk,
            args.stop_token,
        )
    )


if __name__ == "__main__":
    main()

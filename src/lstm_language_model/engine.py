"""Token-weighted training/evaluation; sequence state resets between training batches."""

import math

import torch
from torch.nn import functional as F
from tqdm.auto import tqdm

from .data import IGNORE_INDEX


def run_epoch(model, loader, device, optimizer=None, gradient_clip=0.25, progress=True):
    training = optimizer is not None
    model.train(training)
    loss_total = 0.0
    token_total = 0
    batches = tqdm(loader, desc="Train" if training else "Evaluate", disable=not progress)
    with torch.set_grad_enabled(training):
        for inputs, targets in batches:
            inputs, targets = inputs.to(device), targets.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits, _ = model(inputs)
            count = int(targets.ne(IGNORE_INDEX).sum().item())
            if count == 0:
                raise ValueError("No valid target tokens in batch")
            summed_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            mean_loss = summed_loss / count
            if not torch.isfinite(mean_loss):
                raise FloatingPointError("Non-finite loss; check data or learning rate")
            if training:
                mean_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip, error_if_nonfinite=True
                )
                optimizer.step()
            loss_total += float(summed_loss.detach())
            token_total += count
            batches.set_postfix(loss=f"{loss_total / token_total:.4f}")
    if not token_total:
        raise ValueError("Data loader produced no target tokens")
    loss = loss_total / token_total
    return {"loss": loss, "perplexity": math.exp(loss), "tokens": token_total}

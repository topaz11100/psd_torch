from __future__ import annotations

from typing import Any, Dict, MutableMapping, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.model.model_utils import harden_variable_branches_, set_ste_mode_


def _criterion_uses_output_record(criterion: nn.Module) -> bool:
    return bool(getattr(criterion, 'requires_output_record', False))


def _criterion_set_mode(criterion: nn.Module, *, training: bool) -> None:
    if training:
        criterion.train()
    else:
        criterion.eval()


def _forward_scores_and_loss(
    model: nn.Module,
    criterion: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if _criterion_uses_output_record(criterion):
        if not hasattr(model, 'forward_output_record') or not callable(getattr(model, 'forward_output_record')):
            raise AttributeError('criterion requires output-layer records, but model.forward_output_record(...) is unavailable')
        logits, out_rec = model.forward_output_record(x)
        analysis = criterion.analyze_output_record(out_rec)
        loss = criterion.loss_from_analysis(analysis, y)
        scores = criterion.prediction_scores_from_analysis(analysis)
        pred = criterion.predictions_from_analysis(analysis)
        return logits, loss, pred if pred.dim() == 1 else pred.view(-1)

    logits = model(x)
    loss = criterion(logits, y)
    pred = logits.argmax(dim=1)
    return logits, loss, pred


@torch.no_grad()
def evaluate_model(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> Tuple[float, float]:
    model.eval()
    _criterion_set_mode(criterion, training=False)
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long().view(-1)
        _, loss, pred = _forward_scores_and_loss(model, criterion, x, y)
        bsz = int(y.numel())
        total += bsz
        loss_sum += float(loss.item()) * bsz
        correct += int((pred.view(-1) == y).sum().item())
    if total <= 0:
        return 0.0, 0.0
    return loss_sum / float(total), float(correct) / float(total)



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    *,
    lambda_ortho: float,
    lambda_s: float,
) -> Tuple[float, float]:
    model.train()
    _criterion_set_mode(criterion, training=True)
    total = 0
    correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).long().view(-1)
        optimizer.zero_grad(set_to_none=True)
        _, loss, pred = _forward_scores_and_loss(model, criterion, x, y)
        if hasattr(model, 'regularization_loss') and callable(getattr(model, 'regularization_loss')):
            loss = loss + model.regularization_loss(lambda_ortho=float(lambda_ortho), lambda_s=float(lambda_s))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        bsz = int(y.numel())
        total += bsz
        loss_sum += float(loss.item()) * bsz
        correct += int((pred.view(-1) == y).sum().item())
    if total <= 0:
        return 0.0, 0.0
    return loss_sum / float(total), float(correct) / float(total)



def configure_structure_schedule(
    model: nn.Module,
    epoch: int,
    *,
    total_epochs: int,
    soft_mask_epochs: int | None,
    stabilize_epochs: int,
    ste_epochs: int,
    hardened_state: MutableMapping[str, bool],
) -> Dict[str, Any]:
    total_epochs = int(total_epochs)
    ste_epochs = int(ste_epochs)
    stabilize_epochs = int(stabilize_epochs)
    if ste_epochs < 0:
        raise ValueError(f"ste_epochs must be >= 0, got {ste_epochs}")
    if stabilize_epochs < 0:
        raise ValueError(f"stabilize_epochs must be >= 0, got {stabilize_epochs}")

    if soft_mask_epochs is None:
        stage_a = max(total_epochs - stabilize_epochs, 0)
    else:
        stage_a = int(soft_mask_epochs)
    if stage_a < 0:
        raise ValueError(f"soft_mask_epochs must be >= 0, got {soft_mask_epochs}")
    if stage_a + stabilize_epochs > total_epochs:
        raise ValueError(
            f"Require soft_mask_epochs + stabilize_epochs <= epochs (got {stage_a} + {stabilize_epochs} > {total_epochs})"
        )
    stage_b = total_epochs - stage_a
    ste_epochs = min(ste_epochs, stage_a)

    in_stage_a = int(epoch) <= int(stage_a)
    ste_on = bool(in_stage_a and ste_epochs > 0 and int(epoch) > int(stage_a - ste_epochs))
    set_ste_mode_(model, ste_on)

    if (not in_stage_a) and (not bool(hardened_state.get("done", False))):
        harden_variable_branches_(model)
        hardened_state["done"] = True
        set_ste_mode_(model, False)

    return {
        "stage_a_epochs": int(stage_a),
        "stage_b_epochs": int(stage_b),
        "ste_epochs": int(ste_epochs),
        "current_epoch": int(epoch),
        "stage": "A" if in_stage_a else "B",
        "ste_enabled": bool(ste_on),
        "hardened": bool(hardened_state.get("done", False)),
    }

"""Supervised training-loop helpers for the official split pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.base import canonicalize_model_input_batch
from src.model.snn_builder import ForwardResult, LayerRecord, SNNClassifier
from src.readout.readout import ReadoutBase
from src.signal.family_spectral_analysis import (
    curve_pointwise_distance_torch,
    representative_psd_minibatch_curve_from_maps_torch,
)
from src.signal.psd_utils import trace_tensor_to_channel_major_maps


@dataclass
class EpochMetrics:
    loss: float
    accuracy: float
    correct: int
    total: int


@dataclass
class TrainEpochMetrics:
    loss: float
    accuracy: float
    correct: int
    total: int
    task_loss: float
    regularization_loss: float
    regularization_global_loss: float
    regularization_adjacent_loss: float


@dataclass
class RegularizationLossParts:
    total: torch.Tensor
    global_loss: torch.Tensor
    adjacent_loss: torch.Tensor


def build_optimizer(
    model: nn.Module,
    *,
    lr: float,
) -> torch.optim.Optimizer:
    """Build the project Adam optimizer."""

    return torch.optim.Adam(model.parameters(), lr=float(lr))


def _unwrap_model(model: nn.Module) -> nn.Module:
    wrapped = getattr(model, 'module', None)
    return wrapped if wrapped is not None else model


def _move_inputs_to_device(model: SNNClassifier, inputs: Any, *, device: torch.device) -> torch.Tensor:
    base_model = _unwrap_model(model)
    if getattr(getattr(base_model, 'spec', None), 'family', None) in {'cnn_lif', 'cnn_rf'}:
        tensor = torch.as_tensor(inputs)
    else:
        tensor = canonicalize_model_input_batch(inputs, input_dim=base_model.input_dim, sequence_length=base_model.sequence_length)
    return tensor.to(device=device, dtype=torch.float32, non_blocking=True)


def _move_target_to_device(target: Any, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(target, dtype=torch.long, device=device)


def _project_parameters_after_optimizer_step(model: nn.Module) -> None:
    """Apply model-specific post-step projections required by the Spec."""

    base_model = _unwrap_model(model)
    projector = getattr(base_model, 'clamp_projected_parameters', None)
    if callable(projector):
        projector()

def _reset_stateful_model(model: nn.Module) -> None:
    base_model = _unwrap_model(model)
    resetter = getattr(base_model, "reset_state", None)
    if callable(resetter):
        resetter()

def _regularization_enabled(lambda1: float, lambda2: float) -> bool:
    lambda1 = float(lambda1)
    lambda2 = float(lambda2)
    return lambda1 != 0.0 or lambda2 != 0.0


def _zero_regularization_parts(reference: torch.Tensor) -> RegularizationLossParts:
    zero = reference.new_zeros(())
    return RegularizationLossParts(total=zero, global_loss=zero, adjacent_loss=zero)


def _select_hidden_trace(record: LayerRecord, signal_name: str) -> torch.Tensor:
    token = str(signal_name).strip().lower()
    if token == 'y_mem':
        return record.membrane
    if token == 'y_spike':
        return record.spike
    raise ValueError(f'Unsupported regularization signal: {signal_name!r}. Allowed values are y_mem and y_spike.')


def _curve_from_trace(
    trace: torch.Tensor | None,
    *,
    reducer: str,
    centering: str,
    curve_scale: str,
    curve_space: str,
    role: str,
) -> torch.Tensor:
    if trace is None:
        raise ValueError(f'Regularization requires {role} trace, but it was not captured.')
    if str(curve_space).strip().lower() != 'exact':
        raise ValueError('PSD regularization only supports curve_space=exact.')
    maps = trace_tensor_to_channel_major_maps(trace)
    return representative_psd_minibatch_curve_from_maps_torch(
        maps,
        reducer=str(reducer),
        centering=str(centering),
        scale=str(curve_scale),
        curve_space=str(curve_space),
    )


def compute_regularization_loss(
    result: ForwardResult,
    *,
    lambda1: float = 0.0,
    lambda2: float = 0.0,
    signal_name: str = 'y_mem',
    curve_space: str = 'exact',
    curve_scale: str = 'raw',
    centering: str = 'raw',
    reducer: str = 'mean',
    distance_metric: str = 'centered_l2',
) -> RegularizationLossParts:
    """Return the PSD curve-shape regularization loss and its two weighted parts."""

    if not _regularization_enabled(lambda1, lambda2):
        return _zero_regularization_parts(result.output_record.membrane)

    hidden_records = list(result.hidden_records)
    if not hidden_records:
        raise ValueError('Nonzero PSD curve-shape regularization requires at least one captured hidden layer record.')

    input_curve = _curve_from_trace(
        result.input_record,
        reducer=reducer,
        centering=centering,
        curve_scale=curve_scale,
        curve_space=curve_space,
        role='the original model input',
    )
    hidden_curves = [
        _curve_from_trace(
            _select_hidden_trace(record, signal_name),
            reducer=reducer,
            centering=centering,
            curve_scale=curve_scale,
            curve_space=curve_space,
            role=f'hidden layer {index} {signal_name}',
        )
        for index, record in enumerate(hidden_records, start=1)
    ]

    global_sum = input_curve.new_zeros(())
    for hidden_curve in hidden_curves:
        global_sum = global_sum + curve_pointwise_distance_torch(input_curve, hidden_curve, metric=distance_metric)

    adjacent_sum = input_curve.new_zeros(())
    previous_curve = input_curve
    for hidden_curve in hidden_curves:
        adjacent_sum = adjacent_sum + curve_pointwise_distance_torch(previous_curve, hidden_curve, metric=distance_metric)
        previous_curve = hidden_curve

    weighted_global = float(lambda1) * global_sum
    weighted_adjacent = float(lambda2) * adjacent_sum
    return RegularizationLossParts(
        total=weighted_global + weighted_adjacent,
        global_loss=weighted_global,
        adjacent_loss=weighted_adjacent,
    )


@torch.inference_mode()
def eval_one_batch(
    model: SNNClassifier,
    inputs: Any,
    target: Any,
    *,
    readout: ReadoutBase,
    device: torch.device,
) -> EpochMetrics:
    was_training = bool(model.training)
    model.eval()
    try:
        device_inputs = _move_inputs_to_device(model, inputs, device=device)
        device_target = _move_target_to_device(target, device=device)

        _reset_stateful_model(model)
        result = model(device_inputs, capture_hidden=False)
        _reset_stateful_model(model)

        analysis = readout.analyze_output_record(result.output_record.membrane, result.output_record.spike)
        loss = readout.loss_from_analysis(analysis, device_target, training=False)
        pred = readout.predictions_from_analysis(analysis)
        batch_size = int(device_target.shape[0])
        correct = int((pred == device_target).sum().item())
        accuracy = 0.0 if batch_size == 0 else float(correct) / float(batch_size)
        return EpochMetrics(loss=float(loss.detach().item()), accuracy=accuracy, correct=correct, total=batch_size)
    finally:
        if was_training:
            model.train()


def train_one_batch(
    model: SNNClassifier,
    inputs: Any,
    target: Any,
    *,
    readout: ReadoutBase,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    regularization_lambda1: float = 0.0,
    regularization_lambda2: float = 0.0,
    regularization_signal: str = 'y_mem',
    regularization_curve_space: str = 'exact',
    regularization_curve_scale: str = 'raw',
    regularization_centering: str = 'raw',
    regularization_reducer: str = 'mean',
    regularization_distance_metric: str = 'centered_l2',
) -> TrainEpochMetrics:
    model.train()
    device_inputs = _move_inputs_to_device(model, inputs, device=device)
    device_target = _move_target_to_device(target, device=device)
    capture_hidden = _regularization_enabled(regularization_lambda1, regularization_lambda2)

    optimizer.zero_grad(set_to_none=True)
    _reset_stateful_model(model)
    result = model(device_inputs, capture_hidden=capture_hidden)
    analysis = readout.analyze_output_record(result.output_record.membrane, result.output_record.spike)
    task_loss = readout.loss_from_analysis(analysis, device_target, training=True)
    regularization = compute_regularization_loss(
        result,
        lambda1=regularization_lambda1,
        lambda2=regularization_lambda2,
        signal_name=regularization_signal,
        curve_space=regularization_curve_space,
        curve_scale=regularization_curve_scale,
        centering=regularization_centering,
        reducer=regularization_reducer,
        distance_metric=regularization_distance_metric,
    )
    total_loss = task_loss + regularization.total
    total_loss.backward()
    optimizer.step()
    _reset_stateful_model(model)
    _project_parameters_after_optimizer_step(model)

    pred = readout.predictions_from_analysis(analysis)
    batch_size = int(device_target.shape[0])
    correct = int((pred == device_target).sum().item())
    accuracy = 0.0 if batch_size == 0 else float(correct) / float(batch_size)
    return TrainEpochMetrics(
        loss=float(total_loss.detach().item()),
        accuracy=accuracy,
        correct=correct,
        total=batch_size,
        task_loss=float(task_loss.detach().item()),
        regularization_loss=float(regularization.total.detach().item()),
        regularization_global_loss=float(regularization.global_loss.detach().item()),
        regularization_adjacent_loss=float(regularization.adjacent_loss.detach().item()),
    )


@torch.inference_mode()
def evaluate_one_epoch(
    model: SNNClassifier,
    loader: DataLoader,
    *,
    readout: ReadoutBase,
    device: torch.device,
    progress_desc: str | None = None,
    disable_progress: bool = False,
) -> EpochMetrics:
    total_loss_weighted = 0.0
    total_correct = 0
    total_examples = 0
    for inputs, target in tqdm(loader, desc=progress_desc, leave=False, disable=bool(disable_progress)):
        batch = eval_one_batch(model, inputs, target, readout=readout, device=device)
        total_loss_weighted += float(batch.loss) * int(batch.total)
        total_correct += int(batch.correct)
        total_examples += int(batch.total)
    loss = 0.0 if total_examples == 0 else total_loss_weighted / float(total_examples)
    accuracy = 0.0 if total_examples == 0 else float(total_correct) / float(total_examples)
    return EpochMetrics(loss=loss, accuracy=accuracy, correct=total_correct, total=total_examples)


def train_one_epoch(
    model: SNNClassifier,
    loader: DataLoader,
    *,
    readout: ReadoutBase,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    progress_desc: str | None = None,
    disable_progress: bool = False,
    regularization_lambda1: float = 0.0,
    regularization_lambda2: float = 0.0,
    regularization_signal: str = 'y_mem',
    regularization_curve_space: str = 'exact',
    regularization_curve_scale: str = 'raw',
    regularization_centering: str = 'raw',
    regularization_reducer: str = 'mean',
    regularization_distance_metric: str = 'centered_l2',
) -> TrainEpochMetrics:
    total_loss_weighted = 0.0
    task_loss_weighted = 0.0
    regularization_loss_weighted = 0.0
    regularization_global_loss_weighted = 0.0
    regularization_adjacent_loss_weighted = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, target in tqdm(loader, desc=progress_desc, leave=False, disable=bool(disable_progress)):
        batch = train_one_batch(
            model,
            inputs,
            target,
            readout=readout,
            optimizer=optimizer,
            device=device,
            regularization_lambda1=regularization_lambda1,
            regularization_lambda2=regularization_lambda2,
            regularization_signal=regularization_signal,
            regularization_curve_space=regularization_curve_space,
            regularization_curve_scale=regularization_curve_scale,
            regularization_centering=regularization_centering,
            regularization_reducer=regularization_reducer,
            regularization_distance_metric=regularization_distance_metric,
        )
        total_loss_weighted += float(batch.loss) * int(batch.total)
        task_loss_weighted += float(batch.task_loss) * int(batch.total)
        regularization_loss_weighted += float(batch.regularization_loss) * int(batch.total)
        regularization_global_loss_weighted += float(batch.regularization_global_loss) * int(batch.total)
        regularization_adjacent_loss_weighted += float(batch.regularization_adjacent_loss) * int(batch.total)
        total_correct += int(batch.correct)
        total_examples += int(batch.total)

    loss = 0.0 if total_examples == 0 else total_loss_weighted / float(total_examples)
    task_loss = 0.0 if total_examples == 0 else task_loss_weighted / float(total_examples)
    regularization_loss = 0.0 if total_examples == 0 else regularization_loss_weighted / float(total_examples)
    regularization_global_loss = 0.0 if total_examples == 0 else regularization_global_loss_weighted / float(total_examples)
    regularization_adjacent_loss = 0.0 if total_examples == 0 else regularization_adjacent_loss_weighted / float(total_examples)
    accuracy = 0.0 if total_examples == 0 else float(total_correct) / float(total_examples)
    return TrainEpochMetrics(
        loss=loss,
        accuracy=accuracy,
        correct=total_correct,
        total=total_examples,
        task_loss=task_loss,
        regularization_loss=regularization_loss,
        regularization_global_loss=regularization_global_loss,
        regularization_adjacent_loss=regularization_adjacent_loss,
    )


__all__ = [
    'EpochMetrics',
    'RegularizationLossParts',
    'TrainEpochMetrics',
    'build_optimizer',
    'compute_regularization_loss',
    'eval_one_batch',
    'evaluate_one_epoch',
    'train_one_batch',
    'train_one_epoch',
]

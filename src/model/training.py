"""Supervised training-loop helpers for the official split pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Sequence

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
from src.neurons.spikingjelly_compat import reset_spikingjelly_state
from src.util.precision import amp_autocast_context
from src.model.psd_minibatch_regularizer import (
    compute_fixed_pca_reference_bank,
    compute_minibatch_psd_regularizer,
)


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
    psd_regularization_total: float = 0.0
    psd_regularization_rep_1d: float = 0.0
    psd_regularization_pca_1d: float = 0.0
    psd_regularization_pca_mimo: float = 0.0
    psd_regularization_rep_input: float = 0.0
    psd_regularization_rep_adjacent: float = 0.0
    psd_regularization_pca_1d_input: float = 0.0
    psd_regularization_pca_1d_adjacent: float = 0.0
    psd_regularization_pca_mimo_input: float = 0.0
    psd_regularization_pca_mimo_adjacent: float = 0.0


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
    """Build the project Adam optimizer with fast CUDA variants when available."""

    params = model.parameters()
    try:
        first = next(model.parameters())
        is_cuda = bool(first.is_cuda)
    except StopIteration:
        is_cuda = False
    if is_cuda:
        try:
            return torch.optim.Adam(params, lr=float(lr), fused=True)
        except Exception:
            try:
                return torch.optim.Adam(model.parameters(), lr=float(lr), foreach=True)
            except Exception:
                pass
    return torch.optim.Adam(model.parameters(), lr=float(lr))


def _unwrap_model(model: nn.Module) -> nn.Module:
    current: Any = model
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        wrapped = getattr(current, 'module', None)
        if wrapped is not None and wrapped is not current:
            current = wrapped
            continue
        orig = getattr(current, '_orig_mod', None)
        if orig is not None and orig is not current:
            current = orig
            continue
        return current
    return model

def _move_inputs_to_device(model: SNNClassifier, inputs: Any, *, device: torch.device) -> torch.Tensor:
    base_model = _unwrap_model(model)
    if getattr(getattr(base_model, 'spec', None), 'family', None) in {'cnn_lif', 'cnn_rf'}:
        tensor = torch.as_tensor(inputs)
        tensor = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
        if tensor.ndim == 4:
            return tensor.contiguous(memory_format=torch.channels_last)
        if tensor.ndim == 5:
            b, t, c, h, w = [int(v) for v in tensor.shape]
            return tensor.reshape(b * t, c, h, w).contiguous(memory_format=torch.channels_last).reshape(b, t, c, h, w)
        return tensor.contiguous()
    tensor = canonicalize_model_input_batch(inputs, input_dim=base_model.input_dim, sequence_length=base_model.sequence_length)
    return tensor.to(device=device, dtype=torch.float32, non_blocking=True)


def _move_target_to_device(target: Any, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(target, dtype=torch.long, device=device)


def _float32_tensor_if_needed(value: Any) -> Any:
    if isinstance(value, torch.Tensor) and torch.is_floating_point(value) and value.dtype is not torch.float32:
        return value.float()
    return value


def _layer_record_to_float32(record: LayerRecord) -> LayerRecord:
    converted = LayerRecord(
        layer_name=record.layer_name,
        membrane=_float32_tensor_if_needed(record.membrane),
        spike=_float32_tensor_if_needed(record.spike),
        layer_input=_float32_tensor_if_needed(record.layer_input),
    )
    for key, value in vars(record).items():
        if key in {'layer_name', 'membrane', 'spike', 'layer_input'}:
            continue
        setattr(converted, key, _float32_tensor_if_needed(value))
    return converted


def _forward_result_to_float32(result: ForwardResult) -> ForwardResult:
    return ForwardResult(
        hidden_records=[_layer_record_to_float32(record) for record in result.hidden_records],
        output_record=_layer_record_to_float32(result.output_record),
        input_record=_float32_tensor_if_needed(result.input_record),
    )


def _model_forward_precision(
    model: SNNClassifier,
    device_inputs: torch.Tensor,
    *,
    capture_hidden: bool,
    amp_bf16_safe: bool = False,
    device: torch.device | str | None = None,
) -> ForwardResult:
    mode = 'bf16_safe' if bool(amp_bf16_safe) else 'off'
    previous_amp_env = os.environ.get('PSD_AMP_BF16_SAFE')
    if mode == 'bf16_safe':
        os.environ['PSD_AMP_BF16_SAFE'] = '1'
    else:
        os.environ.pop('PSD_AMP_BF16_SAFE', None)
    try:
        with amp_autocast_context(amp_mode=mode, device=device):
            result = model(device_inputs, capture_hidden=bool(capture_hidden))
    finally:
        if previous_amp_env is None:
            os.environ.pop('PSD_AMP_BF16_SAFE', None)
        else:
            os.environ['PSD_AMP_BF16_SAFE'] = previous_amp_env
    # bf16-safe only accelerates the model forward.  Readout, CE loss, PSD/FFT
    # regularizers and metrics receive FP32 tensors again.
    return _forward_result_to_float32(result) if bool(amp_bf16_safe) else result


def _project_parameters_after_optimizer_step(model: nn.Module) -> None:
    """Apply model-specific post-step projections required by the Spec."""

    base_model = _unwrap_model(model)
    projector = getattr(base_model, 'clamp_projected_parameters', None)
    if callable(projector):
        projector()

def _reset_stateful_model(model: nn.Module) -> None:
    base_model = _unwrap_model(model)
    if reset_spikingjelly_state(base_model):
        return
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


def _disable_compile_recursive(fn):
    compiler = getattr(torch, 'compiler', None)
    disable = getattr(compiler, 'disable', None) if compiler is not None else None
    if not callable(disable):
        return fn
    try:
        return disable(fn, recursive=True)
    except TypeError:  # Older PyTorch without recursive keyword.
        return disable(fn)


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
    userbin_edges: Sequence[float] | None,
    userbin_reducer: str = 'mean',
    role: str,
    signal_window: str | bool | None = 'hann',
) -> torch.Tensor:
    if trace is None:
        raise ValueError(f'Regularization requires {role} trace, but it was not captured.')
    maps = trace_tensor_to_channel_major_maps(trace)
    return representative_psd_minibatch_curve_from_maps_torch(
        maps,
        reducer=str(reducer),
        centering=str(centering),
        scale=str(curve_scale),
        curve_space=str(curve_space),
        userbin_edges=userbin_edges,
        userbin_reducer=str(userbin_reducer),
        signal_window=signal_window,
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
    userbin_edges: Sequence[float] | None = None,
    userbin_reducer: str = 'mean',
    signal_window: str | bool | None = 'hann',
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
        userbin_edges=userbin_edges,
        userbin_reducer=str(userbin_reducer),
        role='the original model input',
        signal_window=signal_window,
    )
    hidden_curves = [
        _curve_from_trace(
            _select_hidden_trace(record, signal_name),
            reducer=reducer,
            centering=centering,
            curve_scale=curve_scale,
            curve_space=curve_space,
            userbin_edges=userbin_edges,
            userbin_reducer=str(userbin_reducer),
            role=f'hidden layer {index} {signal_name}',
            signal_window=signal_window,
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


def _disable_regularizer_compile(fn):
    """Keep PSD/signal regularizers as eager GPU tensor code outside Dynamo graphs."""

    compiler = getattr(torch, 'compiler', None)
    disable = getattr(compiler, 'disable', None) if compiler is not None else None
    if callable(disable):
        try:
            return disable(fn, recursive=True)
        except TypeError:
            try:
                return disable(fn)
            except Exception:
                return fn
        except Exception:
            return fn
    return fn


def _compute_regularizers_eager_gpu_impl(
    result: ForwardResult,
    device_inputs: torch.Tensor,
    *,
    regularization_curve_space: str,
    regularization_curve_scale: str,
    regularization_centering: str,
    regularization_reducer: str,
    regularization_distance_metric: str,
    regularization_userbin_edges: Sequence[float] | None,
    regularization_userbin_reducer: str,
    lambda_psd_rep_input: float,
    lambda_psd_rep_adjacent: float,
    lambda_psd_pca_input: float,
    lambda_psd_pca_adjacent: float,
    psd_reg_output_family: str,
    pca_reference_bank: dict[str, Any] | None,
    signal_window: str | bool | None = 'hann',
) -> tuple[RegularizationLossParts, Any]:
    """Compute all signal/PSD regularizers eagerly on the active tensor device.

    This function intentionally sits behind ``torch.compiler.disable``.  The
    regularizers are GPU tensor operations and must preserve gradients, but they
    should not be captured into model/readout compile graphs or trigger Dynamo
    graph-break diagnostics.
    """

    previous_no_host_checks = os.environ.get('PSD_REGULARIZER_EAGER_GPU_NO_HOST_CHECKS')
    os.environ['PSD_REGULARIZER_EAGER_GPU_NO_HOST_CHECKS'] = '1'
    try:
        zero = device_inputs.new_zeros(())
        regularization = RegularizationLossParts(total=zero, global_loss=zero, adjacent_loss=zero)
        psd_reg = compute_minibatch_psd_regularizer(
            device_inputs,
            list(result.hidden_records),
            variant=str(regularization_centering),
            output_family=str(psd_reg_output_family),
            lambda_rep_input=float(lambda_psd_rep_input),
            lambda_rep_adjacent=float(lambda_psd_rep_adjacent),
            lambda_pca_input=float(lambda_psd_pca_input),
            lambda_pca_adjacent=float(lambda_psd_pca_adjacent),
            pca_reference_banks=pca_reference_bank,
            curve_scale=str(regularization_curve_scale),
            curve_space=str(regularization_curve_space),
            userbin_edges=regularization_userbin_edges,
            userbin_reducer=str(regularization_userbin_reducer),
            reducer=str(regularization_reducer),
            distance_metric=str(regularization_distance_metric),
            signal_window=signal_window,
        )
        return regularization, psd_reg
    finally:
        if previous_no_host_checks is None:
            os.environ.pop('PSD_REGULARIZER_EAGER_GPU_NO_HOST_CHECKS', None)
        else:
            os.environ['PSD_REGULARIZER_EAGER_GPU_NO_HOST_CHECKS'] = previous_no_host_checks

_compute_regularizers_eager_gpu = _disable_regularizer_compile(_compute_regularizers_eager_gpu_impl)


def regularizer_compile_metadata() -> dict[str, Any]:
    """Visible policy metadata for training checkpoints/logging."""

    disabled = _compute_regularizers_eager_gpu is not _compute_regularizers_eager_gpu_impl
    return {
        'regularizer_backend': 'eager_gpu',
        'regularizer_compile_applied': False,
        'regularizer_compile_disabled': bool(disabled),
        'regularizer_compile_policy': 'torch.compiler.disable(recursive=True)' if bool(disabled) else 'eager_gpu_no_disable_api',
        'regularizer_gradient_policy': 'autograd_preserved_no_detach_no_cpu',
    }



@torch.inference_mode()
def _eval_one_batch_tensor_step(
    model: SNNClassifier,
    device_inputs: torch.Tensor,
    device_target: torch.Tensor,
    readout: ReadoutBase,
    *,
    amp_bf16_safe: bool = False,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _reset_stateful_model(model)
    result = _model_forward_precision(
        model,
        device_inputs,
        capture_hidden=False,
        amp_bf16_safe=bool(amp_bf16_safe),
        device=device,
    )
    _reset_stateful_model(model)
    analysis = readout.analyze_output_record(result.output_record.membrane, result.output_record.spike)
    loss = readout.loss_from_analysis(analysis, device_target, training=False)
    pred = readout.predictions_from_analysis(analysis)
    correct = (pred == device_target).sum()
    return loss, pred, correct


@torch.inference_mode()
def eval_one_batch(
    model: SNNClassifier,
    inputs: Any,
    target: Any,
    *,
    readout: ReadoutBase,
    device: torch.device,
    amp_bf16_safe: bool = False,
) -> EpochMetrics:
    was_training = bool(model.training)
    model.eval()
    try:
        device_inputs = _move_inputs_to_device(model, inputs, device=device)
        device_target = _move_target_to_device(target, device=device)
        loss, pred, correct = _eval_one_batch_tensor_step(
            model,
            device_inputs,
            device_target,
            readout,
            amp_bf16_safe=bool(amp_bf16_safe),
            device=device,
        )
        correct_int = int(correct.detach().item()) if isinstance(correct, torch.Tensor) else int(correct)
        batch_size = int(device_target.shape[0])
        accuracy = 0.0 if batch_size == 0 else float(correct_int) / float(batch_size)
        return EpochMetrics(loss=float(loss.detach().item()), accuracy=accuracy, correct=int(correct_int), total=batch_size)
    finally:
        if was_training:
            model.train()


def _train_one_batch_tensor_step(
    model: SNNClassifier,
    device_inputs: torch.Tensor,
    device_target: torch.Tensor,
    readout: ReadoutBase,
    optimizer: torch.optim.Optimizer,
    *,
    capture_hidden: bool,
    regularization_curve_space: str,
    regularization_curve_scale: str,
    regularization_centering: str,
    regularization_reducer: str,
    regularization_distance_metric: str,
    regularization_userbin_edges: Sequence[float] | None,
    regularization_userbin_reducer: str,
    lambda_psd_rep_input: float,
    lambda_psd_rep_adjacent: float,
    lambda_psd_pca_input: float,
    lambda_psd_pca_adjacent: float,
    psd_reg_output_family: str,
    pca_reference_bank: dict[str, Any] | None,
    signal_window: str | bool | None = 'hann',
    amp_bf16_safe: bool = False,
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    optimizer.zero_grad(set_to_none=True)
    _reset_stateful_model(model)
    result = _model_forward_precision(
        model,
        device_inputs,
        capture_hidden=bool(capture_hidden),
        amp_bf16_safe=bool(amp_bf16_safe),
        device=device,
    )
    analysis = readout.analyze_output_record(result.output_record.membrane, result.output_record.spike)
    task_loss = readout.loss_from_analysis(analysis, device_target, training=True)
    regularization, psd_reg = _compute_regularizers_eager_gpu(
        result,
        device_inputs,
        regularization_curve_space=str(regularization_curve_space),
        regularization_curve_scale=str(regularization_curve_scale),
        regularization_centering=str(regularization_centering),
        regularization_reducer=str(regularization_reducer),
        regularization_distance_metric=str(regularization_distance_metric),
        regularization_userbin_edges=regularization_userbin_edges,
        regularization_userbin_reducer=str(regularization_userbin_reducer),
        lambda_psd_rep_input=float(lambda_psd_rep_input),
        lambda_psd_rep_adjacent=float(lambda_psd_rep_adjacent),
        lambda_psd_pca_input=float(lambda_psd_pca_input),
        lambda_psd_pca_adjacent=float(lambda_psd_pca_adjacent),
        psd_reg_output_family=str(psd_reg_output_family),
        pca_reference_bank=pca_reference_bank,
        signal_window=signal_window,
    )
    total_loss = task_loss + regularization.total + psd_reg.total
    total_loss.backward()
    optimizer.step()
    _reset_stateful_model(model)
    _project_parameters_after_optimizer_step(model)
    pred = readout.predictions_from_analysis(analysis)
    correct = (pred == device_target).sum()
    return (
        total_loss,
        task_loss,
        regularization.total,
        regularization.global_loss,
        regularization.adjacent_loss,
        psd_reg.total,
        psd_reg.rep_1d,
        psd_reg.pca_1d,
        psd_reg.pca_mimo,
        psd_reg.rep_input,
        psd_reg.rep_adjacent,
        psd_reg.pca_1d_input,
        psd_reg.pca_1d_adjacent,
        psd_reg.pca_mimo_input,
        psd_reg.pca_mimo_adjacent,
        pred,
        correct,
    )


def train_one_batch(
    model: SNNClassifier,
    inputs: Any,
    target: Any,
    *,
    readout: ReadoutBase,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    regularization_curve_space: str = 'exact',
    regularization_curve_scale: str = 'raw',
    regularization_centering: str = 'raw',
    regularization_reducer: str = 'mean',
    regularization_distance_metric: str = 'centered_l2',
    regularization_userbin_edges: Sequence[float] | None = None,
    regularization_userbin_reducer: str = 'mean',
    lambda_psd_rep_input: float = 0.0,
    lambda_psd_rep_adjacent: float = 0.0,
    lambda_psd_pca_input: float = 0.0,
    lambda_psd_pca_adjacent: float = 0.0,
    psd_reg_output_family: str = 'spike',
    pca_reference_bank: dict[str, Any] | None = None,
    signal_window: str | bool | None = 'hann',
    amp_bf16_safe: bool = False,
) -> TrainEpochMetrics:
    model.train()
    device_inputs = _move_inputs_to_device(model, inputs, device=device)
    device_target = _move_target_to_device(target, device=device)
    capture_hidden = any(
        float(v) != 0.0
        for v in (
            lambda_psd_rep_input,
            lambda_psd_rep_adjacent,
            lambda_psd_pca_input,
            lambda_psd_pca_adjacent,
        )
    )

    (
        total_loss, task_loss, reg_total, reg_global, reg_adjacent,
        psd_total, psd_rep, psd_pca1, psd_pcam,
        psd_rep_input, psd_rep_adjacent,
        psd_pca1_input, psd_pca1_adjacent,
        psd_pcam_input, psd_pcam_adjacent,
        pred, correct,
    ) = _train_one_batch_tensor_step(
        model,
        device_inputs,
        device_target,
        readout,
        optimizer,
        capture_hidden=capture_hidden,
        regularization_curve_space=str(regularization_curve_space),
        regularization_curve_scale=str(regularization_curve_scale),
        regularization_centering=str(regularization_centering),
        regularization_reducer=str(regularization_reducer),
        regularization_distance_metric=str(regularization_distance_metric),
        regularization_userbin_edges=regularization_userbin_edges,
        regularization_userbin_reducer=str(regularization_userbin_reducer),
        lambda_psd_rep_input=float(lambda_psd_rep_input),
        lambda_psd_rep_adjacent=float(lambda_psd_rep_adjacent),
        lambda_psd_pca_input=float(lambda_psd_pca_input),
        lambda_psd_pca_adjacent=float(lambda_psd_pca_adjacent),
        psd_reg_output_family=str(psd_reg_output_family),
        pca_reference_bank=pca_reference_bank,
        signal_window=signal_window,
        amp_bf16_safe=bool(amp_bf16_safe),
        device=device,
    )

    correct_int = int(correct.detach().item()) if isinstance(correct, torch.Tensor) else int(correct)
    batch_size = int(device_target.shape[0])
    accuracy = 0.0 if batch_size == 0 else float(correct_int) / float(batch_size)
    return TrainEpochMetrics(
        loss=float(total_loss.detach().item()),
        accuracy=accuracy,
        correct=int(correct_int),
        total=batch_size,
        task_loss=float(task_loss.detach().item()),
        regularization_loss=float(reg_total.detach().item()),
        regularization_global_loss=float(reg_global.detach().item()),
        regularization_adjacent_loss=float(reg_adjacent.detach().item()),
        psd_regularization_total=float(psd_total.detach().item()),
        psd_regularization_rep_1d=float(psd_rep.detach().item()),
        psd_regularization_pca_1d=float(psd_pca1.detach().item()),
        psd_regularization_pca_mimo=float(psd_pcam.detach().item()),
        psd_regularization_rep_input=float(psd_rep_input.detach().item()),
        psd_regularization_rep_adjacent=float(psd_rep_adjacent.detach().item()),
        psd_regularization_pca_1d_input=float(psd_pca1_input.detach().item()),
        psd_regularization_pca_1d_adjacent=float(psd_pca1_adjacent.detach().item()),
        psd_regularization_pca_mimo_input=float(psd_pcam_input.detach().item()),
        psd_regularization_pca_mimo_adjacent=float(psd_pcam_adjacent.detach().item()),
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
    amp_bf16_safe: bool = False,
) -> EpochMetrics:
    total_loss_weighted = 0.0
    total_correct = 0
    total_examples = 0
    for inputs, target in tqdm(loader, desc=progress_desc, leave=False, disable=bool(disable_progress)):
        batch = eval_one_batch(
            model,
            inputs,
            target,
            readout=readout,
            device=device,
            amp_bf16_safe=bool(amp_bf16_safe),
        )
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
    regularization_curve_space: str = 'exact',
    regularization_curve_scale: str = 'raw',
    regularization_centering: str = 'raw',
    regularization_reducer: str = 'mean',
    regularization_distance_metric: str = 'centered_l2',
    regularization_userbin_edges: Sequence[float] | None = None,
    regularization_userbin_reducer: str = 'mean',
    lambda_psd_rep_input: float = 0.0,
    lambda_psd_rep_adjacent: float = 0.0,
    lambda_psd_pca_input: float = 0.0,
    lambda_psd_pca_adjacent: float = 0.0,
    psd_reg_output_family: str = 'spike',
    pca_reference_bank: dict[str, Any] | None = None,
    signal_window: str | bool | None = 'hann',
    amp_bf16_safe: bool = False,
) -> TrainEpochMetrics:
    total_loss_weighted = 0.0
    task_loss_weighted = 0.0
    regularization_loss_weighted = 0.0
    regularization_global_loss_weighted = 0.0
    regularization_adjacent_loss_weighted = 0.0
    psd_total_w = 0.0
    psd_rep_w = 0.0
    psd_pca1_w = 0.0
    psd_pcam_w = 0.0
    psd_rep_input_w = 0.0
    psd_rep_adjacent_w = 0.0
    psd_pca1_input_w = 0.0
    psd_pca1_adjacent_w = 0.0
    psd_pcam_input_w = 0.0
    psd_pcam_adjacent_w = 0.0
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
            regularization_curve_space=regularization_curve_space,
            regularization_curve_scale=regularization_curve_scale,
            regularization_centering=regularization_centering,
            regularization_reducer=regularization_reducer,
            regularization_distance_metric=regularization_distance_metric,
            regularization_userbin_edges=regularization_userbin_edges,
            regularization_userbin_reducer=regularization_userbin_reducer,
            lambda_psd_rep_input=lambda_psd_rep_input,
            lambda_psd_rep_adjacent=lambda_psd_rep_adjacent,
            lambda_psd_pca_input=lambda_psd_pca_input,
            lambda_psd_pca_adjacent=lambda_psd_pca_adjacent,
            psd_reg_output_family=psd_reg_output_family,
            pca_reference_bank=pca_reference_bank,
            signal_window=signal_window,
            amp_bf16_safe=bool(amp_bf16_safe),
        )
        total_loss_weighted += float(batch.loss) * int(batch.total)
        task_loss_weighted += float(batch.task_loss) * int(batch.total)
        regularization_loss_weighted += float(batch.regularization_loss) * int(batch.total)
        regularization_global_loss_weighted += float(batch.regularization_global_loss) * int(batch.total)
        regularization_adjacent_loss_weighted += float(batch.regularization_adjacent_loss) * int(batch.total)
        psd_total_w += float(batch.psd_regularization_total) * int(batch.total)
        psd_rep_w += float(batch.psd_regularization_rep_1d) * int(batch.total)
        psd_pca1_w += float(batch.psd_regularization_pca_1d) * int(batch.total)
        psd_pcam_w += float(batch.psd_regularization_pca_mimo) * int(batch.total)
        psd_rep_input_w += float(batch.psd_regularization_rep_input) * int(batch.total)
        psd_rep_adjacent_w += float(batch.psd_regularization_rep_adjacent) * int(batch.total)
        psd_pca1_input_w += float(batch.psd_regularization_pca_1d_input) * int(batch.total)
        psd_pca1_adjacent_w += float(batch.psd_regularization_pca_1d_adjacent) * int(batch.total)
        psd_pcam_input_w += float(batch.psd_regularization_pca_mimo_input) * int(batch.total)
        psd_pcam_adjacent_w += float(batch.psd_regularization_pca_mimo_adjacent) * int(batch.total)
        total_correct += int(batch.correct)
        total_examples += int(batch.total)

    loss = 0.0 if total_examples == 0 else total_loss_weighted / float(total_examples)
    task_loss = 0.0 if total_examples == 0 else task_loss_weighted / float(total_examples)
    regularization_loss = 0.0 if total_examples == 0 else regularization_loss_weighted / float(total_examples)
    regularization_global_loss = 0.0 if total_examples == 0 else regularization_global_loss_weighted / float(total_examples)
    regularization_adjacent_loss = 0.0 if total_examples == 0 else regularization_adjacent_loss_weighted / float(total_examples)
    psd_total = 0.0 if total_examples == 0 else psd_total_w / float(total_examples)
    psd_rep = 0.0 if total_examples == 0 else psd_rep_w / float(total_examples)
    psd_pca1 = 0.0 if total_examples == 0 else psd_pca1_w / float(total_examples)
    psd_pcam = 0.0 if total_examples == 0 else psd_pcam_w / float(total_examples)
    psd_rep_input = 0.0 if total_examples == 0 else psd_rep_input_w / float(total_examples)
    psd_rep_adjacent = 0.0 if total_examples == 0 else psd_rep_adjacent_w / float(total_examples)
    psd_pca1_input = 0.0 if total_examples == 0 else psd_pca1_input_w / float(total_examples)
    psd_pca1_adjacent = 0.0 if total_examples == 0 else psd_pca1_adjacent_w / float(total_examples)
    psd_pcam_input = 0.0 if total_examples == 0 else psd_pcam_input_w / float(total_examples)
    psd_pcam_adjacent = 0.0 if total_examples == 0 else psd_pcam_adjacent_w / float(total_examples)
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
        psd_regularization_total=psd_total,
        psd_regularization_rep_1d=psd_rep,
        psd_regularization_pca_1d=psd_pca1,
        psd_regularization_pca_mimo=psd_pcam,
        psd_regularization_rep_input=psd_rep_input,
        psd_regularization_rep_adjacent=psd_rep_adjacent,
        psd_regularization_pca_1d_input=psd_pca1_input,
        psd_regularization_pca_1d_adjacent=psd_pca1_adjacent,
        psd_regularization_pca_mimo_input=psd_pcam_input,
        psd_regularization_pca_mimo_adjacent=psd_pcam_adjacent,
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
    'regularizer_compile_metadata',
]
try:
    from src.patch_overlays.runtime_patch import patch_training as _patch_training
    _patch_training(globals())
except Exception:
    pass

"""Supervised training entrypoint for the split PSD pipeline."""
from __future__ import annotations
import sys
from pathlib import Path
if __package__ is None or __package__ == '':
    _SCRIPT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _SCRIPT_DIR.parent
    try:
        sys.path.remove(str(_SCRIPT_DIR))
    except ValueError:
        pass
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

import argparse, json, os, tempfile, shutil, warnings
from datetime import timedelta


_THREAD_ENV_KEYS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')


def _default_cpu_thread_count() -> int:
    explicit = os.environ.get('PSD_TORCH_CPU_THREADS') or os.environ.get('OMP_NUM_THREADS')
    if explicit is not None:
        try:
            return max(1, int(explicit))
        except ValueError:
            pass
    # The training entrypoint is GPU-first.  On CPU-only smoke/CI runs, keeping
    # BLAS thread counts small prevents 784-step SNN loops from oversubscribing
    # the host before Python reaches the runtime thread guard.
    if os.environ.get('CI', '').lower() in {'1', 'true', 'yes'}:
        return 1
    return max(1, min(os.cpu_count() or 1, 8))


def _install_default_cpu_thread_env() -> None:
    num_threads = str(_default_cpu_thread_count())
    for key in _THREAD_ENV_KEYS:
        os.environ.setdefault(key, num_threads)


_install_default_cpu_thread_env()
from dataclasses import dataclass
from typing import Any, Sequence
from src.model.constraints import ConstraintConfig, normalize_scenario_mode
from src.util.config import load_manifest, load_structured, resolve_manifest_path, to_jsonable
from src.util.config_cli import parse_args_with_config
from src.util.csv_schema import common_row, write_common_csv
from src.util.cli_common import parse_bool_token
from src.util.checkpoints import checkpoint_state_dict, load_state_dict_compatible, load_torch_checkpoint, normalize_state_dict_keys, unwrap_model as _unwrap_checkpoint_model
from src.util.precision import configure_tf32, normalize_amp_mode
from src.util.paths import make_timestamp, parse_timestamped_output, timestamped_output_path

CHECKPOINT_SCHEMA_VERSION = 'psd_checkpoint_v1'
SOURCE_PROGRAM = 'model_training'

@dataclass
class DDPContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: Any
    is_rank0: bool
    gpu_indices: tuple[int, ...] = ()

def _load_runtime_dependencies() -> None:
    global torch, tqdm, _seed_everything, make_loader, resolve_dataset_bundle, select_training_view_for_model, validate_image_mlp_flatten_contract
    global ModelSpec, model_spec_from_namespace, build_optimizer, evaluate_one_epoch, train_one_epoch, eval_one_batch, train_one_batch, build_snn_classifier, build_readout, canonicalize_readout_mode, DistributedSampler, DDP
    import torch as _torch
    from tqdm import tqdm as _tqdm
    from torch.nn.parallel import DistributedDataParallel as _DDP
    from torch.utils.data.distributed import DistributedSampler as _DistributedSampler
    from src.data.registry import make_loader as _make_loader, resolve_dataset_bundle as _resolve_dataset_bundle, select_training_view_for_model as _select_training_view_for_model, validate_image_mlp_flatten_contract as _validate_image_mlp_flatten_contract
    from src.model.model_registry import ModelSpec as _ModelSpec, model_spec_from_namespace as _model_spec_from_namespace
    from src.model.training import build_optimizer as _build_optimizer, evaluate_one_epoch as _evaluate_one_epoch, train_one_epoch as _train_one_epoch, eval_one_batch as _eval_one_batch, train_one_batch as _train_one_batch
    from src.model.snn_builder import build_snn_classifier as _build_snn_classifier
    from src.readout.readout import build_readout as _build_readout, canonicalize_readout_mode as _canonicalize_readout_mode
    from src.util.random import seed_everything as _runtime_seed_everything
    torch=_torch; tqdm=_tqdm; DDP=_DDP; DistributedSampler=_DistributedSampler
    make_loader=_make_loader; resolve_dataset_bundle=_resolve_dataset_bundle; select_training_view_for_model=_select_training_view_for_model; validate_image_mlp_flatten_contract=_validate_image_mlp_flatten_contract
    ModelSpec=_ModelSpec; model_spec_from_namespace=_model_spec_from_namespace; build_optimizer=_build_optimizer; evaluate_one_epoch=_evaluate_one_epoch; train_one_epoch=_train_one_epoch; eval_one_batch=_eval_one_batch; train_one_batch=_train_one_batch
    build_snn_classifier=_build_snn_classifier; build_readout=_build_readout; canonicalize_readout_mode=_canonicalize_readout_mode; _seed_everything=_runtime_seed_everything

def _load_config_light(path: Path) -> dict[str, Any]:
    payload = load_structured(path)
    if not isinstance(payload, dict):
        raise ValueError(f'구조화 파일 루트는 mapping이어야 합니다: {path}')
    return dict(payload)
def _parse_bool_config_value(value: Any, *, default: bool) -> bool: return parse_bool_token(value, default=default)


def _split_gpu_index_token(value: Any) -> list[int]:
    """Parse one gpu_index scalar/list token into integer CUDA ordinals."""

    if value is None or value == '':
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        token = value.strip()
        if token == '':
            return []
        # Accept either YAML lists or CLI-friendly forms such as --gpu_index 0,1.
        token = token.strip('[]()')
        if token == '':
            return []
        parts = [part.strip() for part in token.replace(';', ',').split(',')]
        return [int(part) for part in parts if part != '']
    if isinstance(value, (list, tuple)):
        out: list[int] = []
        for item in value:
            out.extend(_split_gpu_index_token(item))
        return out
    return [int(value)]


def _normalize_gpu_index_sequence(value: Any) -> list[int]:
    indices = _split_gpu_index_token(value)
    if not indices:
        indices = [0]
    if len(set(indices)) != len(indices):
        raise ValueError(f'gpu_index must not contain duplicate CUDA indices: {indices}')
    for index in indices:
        if int(index) < 0:
            raise ValueError(f'gpu_index entries must be non-negative CUDA ordinals: {indices}')
    return [int(index) for index in indices]


def _gpu_indices_from_args(args: argparse.Namespace) -> list[int]:
    indices = _normalize_gpu_index_sequence(getattr(args, 'gpu_index', [0]))
    # Normalize the argparse namespace once so all downstream metadata sees the
    # public array contract, not argparse's string/list variants.
    args.gpu_index = list(indices)
    return indices


def _inside_torchrun() -> bool:
    return all(key in os.environ for key in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE'))


def _torchrun_world_size(default: int = 1) -> int:
    try:
        return int(os.environ.get('WORLD_SIZE', str(default)))
    except Exception:
        return int(default)


def _ddp_requested(args: argparse.Namespace) -> bool:
    """DDP is derived only from gpu_index length or an existing torchrun env."""

    if _torchrun_world_size(default=1) > 1:
        return True
    return len(_gpu_indices_from_args(args)) >= 2


def _maybe_reexec_for_gpu_index_ddp(args: argparse.Namespace, argv: Sequence[str] | None) -> dict[str, Any]:
    """Launch torchrun automatically when gpu_index contains multiple devices.

    Public contract:
    - gpu_index: [k]      -> single-process training on cuda:k;
    - gpu_index: [a, b+]  -> torchrun DDP over exactly those physical devices.

    The child processes see CUDA_VISIBLE_DEVICES narrowed to the requested list,
    so LOCAL_RANK 0..N-1 maps onto the chosen physical ordinals.
    """

    indices = _gpu_indices_from_args(args)
    if len(indices) < 2:
        return {'reexec': False, 'reason': 'single_gpu', 'gpu_index': list(indices)}
    if _inside_torchrun():
        return {'reexec': False, 'reason': 'already_inside_torchrun', 'gpu_index': list(indices), 'world_size': _torchrun_world_size(default=len(indices))}

    original_argv = list(sys.argv[1:] if argv is None else argv)
    env = os.environ.copy()
    previous_visible = env.get('CUDA_VISIBLE_DEVICES')
    env['PSD_PARENT_CUDA_VISIBLE_DEVICES'] = '' if previous_visible is None else previous_visible
    env['CUDA_VISIBLE_DEVICES'] = ','.join(str(index) for index in indices)
    env['PSD_GPU_INDEX_REQUESTED'] = env['CUDA_VISIBLE_DEVICES']
    env['PSD_GPU_INDEX_AUTO_TORCHRUN'] = '1'

    # torchrun sets OMP_NUM_THREADS=1 when it is unset.  If the user supplied
    # compile_cpu_threads, seed the thread environment before torchrun starts so
    # the child interpreter and Inductor see the requested value from process start.
    raw_threads = getattr(args, 'compile_cpu_threads', None)
    if raw_threads not in (None, ''):
        threads = str(max(1, int(raw_threads)))
        for key in _THREAD_ENV_KEYS:
            env[key] = threads

    command = [
        sys.executable,
        '-m',
        'torch.distributed.run',
        '--standalone',
        '--nnodes=1',
        f'--nproc_per_node={len(indices)}',
        '-m',
        'src.model_training',
        *original_argv,
    ]
    os.execvpe(command[0], command, env)
    raise RuntimeError('os.execvpe returned unexpectedly while launching torchrun.')

def _signal_window_from_args(args: argparse.Namespace) -> str:
    from src.signal.psd_utils import normalize_signal_window
    return normalize_signal_window(getattr(args, 'signal_window', 'hann'))


def _signal_curve_userbin_edges_from_args(args: argparse.Namespace) -> list[float] | None:
    from src.signal.psd_curve_config import resolve_userbin_edges

    curve_space = str(getattr(args, 'signal_curve_space', 'exact')).strip().lower()
    return resolve_userbin_edges(
        edges=getattr(args, 'signal_curve_userbin_edges', None),
        required=(curve_space == 'userbin'),
    )


def _install_signal_curve_internal_aliases(args: argparse.Namespace) -> None:
    # Internal training helpers still use the historical regularization_* names.
    # Public configs use only signal_curve_* after the schema cleanup.
    args.regularization_curve_space = str(getattr(args, 'signal_curve_space', 'exact'))
    args.regularization_curve_scale = str(getattr(args, 'signal_curve_scale', 'raw'))
    args.regularization_centering = str(getattr(args, 'signal_curve_centering', 'raw'))
    args.regularization_reducer = str(getattr(args, 'signal_curve_reducer', 'mean'))
    args.regularization_distance_metric = str(getattr(args, 'signal_curve_distance_metric', 'centered_l2'))
    args.regularization_userbin_edges = _signal_curve_userbin_edges_from_args(args)
    args.regularization_userbin_reducer = str(getattr(args, 'signal_curve_userbin_reducer', 'mean'))


def _configure_token_regularizer_env_from_args(args: argparse.Namespace) -> None:
    os.environ['PSD_SIGNAL_WINDOW'] = _signal_window_from_args(args)
    os.environ.pop('PSD_REG_CURVE_TOKENS', None)
    edges = getattr(args, 'signal_curve_userbin_edges', None)
    if edges is None:
        os.environ.pop('PSD_REG_USERBIN_EDGES', None)
    else:
        os.environ['PSD_REG_USERBIN_EDGES'] = ','.join(str(float(v)) for v in edges)
    os.environ['PSD_REG_USERBIN_REDUCER'] = str(getattr(args, 'signal_curve_userbin_reducer', 'mean'))
    os.environ.pop('PSD_REG_USERBIN_WIDTH', None)
    os.environ.pop('PSD_REG_USERBIN_COUNT', None)


def _unwrap_model(model: Any) -> Any: return _unwrap_checkpoint_model(model)


def _set_branch_training_stage(model: Any, epoch: int, args: argparse.Namespace) -> None:
    """Apply the proposed my_* soft/STE/hard branch-count training schedule.

    The hook is no-op for vanilla layers.  Scenario clip/structure constraints
    are handled by ConstraintConfig and remain limited to vanilla IF/LIF/RF
    families inside snn_builder.
    """

    root = _unwrap_model(model)
    harden_epoch = getattr(args, 'harden_epoch', None)
    if harden_epoch not in (None, '') and int(epoch) >= int(harden_epoch):
        hook = getattr(root, 'harden_branches', None)
        if callable(hook):
            hook()
        return

    soft_epochs = max(0, int(getattr(args, 'soft_mask_epochs', 0) or 0))
    ste_epochs = max(0, int(getattr(args, 'ste_epochs', 0) or 0))
    enable_ste = ste_epochs > 0 and int(epoch) > soft_epochs and int(epoch) <= soft_epochs + ste_epochs
    hook = getattr(root, 'enable_branch_ste', None)
    if callable(hook):
        hook(bool(enable_ste))


def _model_for_evaluation(model: Any) -> Any:
    """Return the compiled model body while removing only DDP for evaluation."""
    ddp_module = getattr(model, 'module', None)
    return ddp_module if ddp_module is not None else model
def _is_rank0(ctx: DDPContext) -> bool: return bool(ctx.is_rank0)

def _rank0_write(ctx: DDPContext, message: str) -> None:
    if not _is_rank0(ctx):
        return
    writer = globals().get('tqdm', None)
    try:
        if writer is not None and hasattr(writer, 'write'):
            writer.write(message)
        else:
            print(message, flush=True)
    except Exception:
        print(message, flush=True)

def _compiled_region_summary(model: Any) -> dict[str, Any]:
    root = _unwrap_model(model)
    named_modules = getattr(root, 'named_modules', None)
    regions: list[dict[str, Any]] = []
    disabled: list[dict[str, Any]] = []
    if callable(named_modules):
        try:
            iterator = named_modules()
        except Exception:
            iterator = []
        for name, module in iterator:
            display_name = str(name or '<root>')
            sequence_compiled = (
                getattr(module, '_compiled_sequence', None) is not None
                or getattr(module, '_compiled_sequence_no_trace', None) is not None
                or getattr(module, '_compiled_sequence_with_trace', None) is not None
            )
            core_compiled = getattr(module, '_compiled_core_forward', None) is not None
            source_compiled = getattr(module, '_compiled_source_forward', None) is not None
            if sequence_compiled or core_compiled or source_compiled:
                kind = 'source_forward' if source_compiled else ('core' if core_compiled else 'sequence')
                policy = getattr(module, '_compiled_core_policy', None) or getattr(module, '_compiled_sequence_policy', None)
                if policy is None and source_compiled:
                    extra = getattr(module, 'extra_metadata', {})
                    policies = extra.get('compile_child_policies') if isinstance(extra, dict) else None
                    policy = policies[0] if isinstance(policies, list) and policies else 'compiled_source_forward'
                regions.append({
                    'name': display_name,
                    'kind': kind,
                    'policy': str(policy or 'compiled'),
                })
            if (
                bool(getattr(module, '_sequence_compiled_runtime_disabled', False))
                or bool(getattr(module, '_compiled_core_disabled', False))
                or bool(getattr(module, '_spikformer_source_compiled_runtime_disabled', False))
            ):
                disabled.append({
                    'name': display_name,
                    'error': str(getattr(module, '_sequence_compiled_runtime_error', getattr(module, '_compiled_core_error', getattr(module, '_spikformer_source_compiled_runtime_error', 'unknown')))),
                })
    return {
        'compiled_region_count': len(regions),
        'compiled_regions': regions[:24],
        'compiled_region_truncated': max(0, len(regions) - 24),
        'runtime_disabled_count': len(disabled),
        'runtime_disabled_regions': disabled[:24],
    }

def _emit_compile_startup_status(
    ctx: DDPContext,
    model: Any,
    *,
    compile_requested: bool,
    compile_applied: bool,
    compile_policy: str,
    compile_kwargs: dict[str, Any],
    compile_stance_policy: str,
    compile_cache_policy: dict[str, Any],
    compile_threads_policy: dict[str, Any] | None = None,
    readout_compile_applied: bool = False,
    readout_compile_policy: str = 'not_requested',
    amp_mode: str = 'off',
    amp_bf16_safe_active: bool = False,
    tf32_policy: dict[str, Any],
    drop_last_train: bool,
) -> None:
    root = _unwrap_model(model)
    try:
        from src.neurons._common import sequence_backend_name, sequence_buffer_mode
        sequence_backend = sequence_backend_name()
        sequence_buffer = sequence_buffer_mode()
    except Exception:
        sequence_backend = 'compiled_sequence_prealloc'
        sequence_buffer = 'prealloc'
    metadata_fn = getattr(root, 'model_metadata', None)
    try:
        model_meta = dict(metadata_fn()) if callable(metadata_fn) else {}
    except Exception as exc:
        model_meta = {'metadata_error': f'{type(exc).__name__}: {exc}'}
    region_summary = _compiled_region_summary(root)
    payload = {
        'event': 'compile_startup_status',
        'rank': int(getattr(ctx, 'rank', 0)),
        'local_rank': int(getattr(ctx, 'local_rank', 0)),
        'device': str(getattr(ctx, 'device', 'unknown')),
        'compile': bool(compile_requested),
        'compile_applied': bool(compile_applied),
        'compile_policy': str(compile_policy),
        'compile_kwargs': dict(compile_kwargs or {}),
        'compile_stance': str(compile_stance_policy),
        'compile_child_region_count': model_meta.get('compile_child_region_count'),
        'compile_granularity': model_meta.get('compile_granularity'),
        'compile_child_policies': model_meta.get('compile_child_policies', [])[:12],
        'sequence_backend': model_meta.get('sequence_backend', sequence_backend),
        'sequence_buffer_mode': model_meta.get('sequence_buffer_mode', sequence_buffer),
        'compile_cache': dict(compile_cache_policy or {}),
        'compile_threads': dict(compile_threads_policy or {}),
        'readout_compile_applied': bool(readout_compile_applied),
        'readout_compile_policy': str(readout_compile_policy),
        'drop_last_train': bool(drop_last_train),
        'amp': str(amp_mode),
        'amp_active': 'bf16_safe' if bool(amp_bf16_safe_active) else 'off',
        'tf32': dict(tf32_policy or {}),
        **region_summary,
    }
    _rank0_write(ctx, '[model_training] ' + json.dumps(payload, sort_keys=True, ensure_ascii=False))

def _ddp_barrier(ctx: DDPContext) -> None:
    if not bool(getattr(ctx, 'enabled', False)):
        return
    try:
        torch.distributed.barrier(device_ids=[int(ctx.local_rank)])
    except TypeError:
        torch.distributed.barrier()

def _shared_run_timestamp(args: argparse.Namespace, ctx: DDPContext) -> str:
    """Return one timestamp token shared by all DDP ranks."""

    token = [make_timestamp(getattr(args, 'run_timestamp', None)) if _is_rank0(ctx) else None]
    if bool(getattr(ctx, 'enabled', False)):
        torch.distributed.broadcast_object_list(token, src=0)
    if token[0] is None:
        token[0] = make_timestamp(getattr(args, 'run_timestamp', None))
    return str(token[0])


def _resolve_training_result_roots(args: argparse.Namespace, ctx: DDPContext) -> tuple[Path, Path, str, bool]:
    """Resolve actual checkpoint/metric roots for one timestamped run."""

    timestamped = parse_timestamped_output(getattr(args, 'timestamped_output', True), default=True)
    run_timestamp = _shared_run_timestamp(args, ctx)
    checkpoint_root = timestamped_output_path(
        args.checkpoint_root,
        timestamp=run_timestamp,
        enabled=timestamped,
        leaf_names={'checkpoints', 'metrics', 'train'},
    )
    metric_root = timestamped_output_path(
        args.metric_root,
        timestamp=run_timestamp,
        enabled=timestamped,
        leaf_names={'checkpoints', 'metrics', 'train'},
    )
    return checkpoint_root, metric_root, run_timestamp, timestamped

def build_arg_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description='Supervised model training entrypoint for selected checkpoint production.')
    p.add_argument('--dataset', required=True); p.add_argument('--prep_root', required=True)
    p.add_argument('--neuron_type', default=None, help='Base model family only, e.g. lif, rf, dh_snn, d_rf, my_d_rf, spikeformer, spikegru, vgg11_lif. Token forms such as my_d_rf_8_hard_train are rejected.')
    p.add_argument('--recurrent', default='false', help='Use recurrent hidden connections for supported dense families.')
    p.add_argument('--reset', default=None, help='Reset mode as a separate field: soft, hard, or none when supported.')
    p.add_argument('--filter', default='train', help='Neuron filter parameter policy: train or numeric fixed value.')
    p.add_argument('--branch', default=None, help='Branch count for dh_snn/d_rf/my_* models. Do not append the branch count to neuron_type.')
    p.add_argument('--rf_pole_radius_constrained', default='true', help='Vanilla RF direct-discrete pole radius policy. true constrains |a| to [0, rf_pole_radius_max); false uses a positive unconstrained radius and can learn finite-horizon amplification.')
    p.add_argument('--rf_pole_radius_max', default=0.9999, type=float, help='Upper pole-radius bound for vanilla RF when rf_pole_radius_constrained=true.')
    p.add_argument('--hidden_spec', required=True)
    p.add_argument('--readout_mode', required=True, choices=('temporal_membrane','final_membrane','first_spike','max_fire','max_rate','spikegru_max_over_time'))
    p.add_argument('--epochs', required=True, type=int); p.add_argument('--batch_size', required=True, type=int); p.add_argument('--lr', required=True, type=float)
    p.add_argument('--num_workers', type=int, default=0); p.add_argument('--seed', required=True, type=int); p.add_argument('--gpu_index', nargs='*', default=[0], help='CUDA device index array. One item runs non-DDP on that GPU; two or more items auto-launch DDP on those GPUs.')
    p.add_argument('--batch_size_is_global', default='true', help='batch_size를 global batch로 해석할지 여부. DDP에서는 true만 허용.')
    p.add_argument('--signal_curve_space', default='exact', choices=('exact','userbin'))
    p.add_argument('--signal_curve_scale', default='raw', choices=('raw','db','area'))
    p.add_argument('--signal_curve_centering', default='raw', choices=('raw','centered'))
    p.add_argument('--signal_curve_reducer', default='mean', choices=('mean','median'))
    p.add_argument('--signal_curve_distance_metric', default='centered_l2', choices=('centered_l2','diff_l2'))
    p.add_argument('--signal_curve_userbin_edges', nargs='*', default=None)
    p.add_argument('--signal_curve_userbin_reducer', default='mean', choices=('mean','median','sum'))
    p.add_argument('--signal_window', default='hann', choices=('hann','none'), help='PSD/FFT signal processing taper: hann or none. none disables the Hann window.')
    p.add_argument('--lambda_psd_rep_input', default=0.0, type=float)
    p.add_argument('--lambda_psd_rep_adjacent', default=0.0, type=float)
    p.add_argument('--lambda_psd_pca_input', default=0.0, type=float)
    p.add_argument('--lambda_psd_pca_adjacent', default=0.0, type=float)
    p.add_argument('--psd_reg_output_family', default='spike', choices=('spike','membrane'))
    p.add_argument('--lambda_branch_ortho', default=0.0, type=float, help='Proposed my_* branch-basis orthogonality regularization weight.')
    p.add_argument('--lambda_branch_s', default=0.0, type=float, help='Proposed my_* soft branch-count regularization weight.')
    p.add_argument('--soft_mask_epochs', default=0, type=int, help='Epochs that keep proposed my_* branch masks in soft mode before optional STE.')
    p.add_argument('--ste_epochs', default=0, type=int, help='Epochs that enable STE branch-count selection after soft_mask_epochs.')
    p.add_argument('--harden_epoch', default=None, type=int, help='Epoch from which proposed my_* branch masks are hardened in-place.')
    p.add_argument('--pca_dim_per_layer', nargs='*', default=None)
    p.add_argument('--scenario_mode', default='none', choices=('none','clip','structure','clipstructure','clip_structure'))
    p.add_argument('--w_clip_edges', nargs='*', default=None)
    p.add_argument('--alpha_clip_edges', nargs='*', default=None)
    p.add_argument('--band_edge', default=None)
    p.add_argument('--tear', type=int, default=1)
    p.add_argument('--analysis_checkpoint_epochs', nargs='*', default=None, help='Epoch list for evaluation, metric recording, and checkpoint saving.')
    p.add_argument('--checkpoint_root', required=True); p.add_argument('--metric_root', required=True)
    p.add_argument('--v_th', nargs='*', default=None, help='Threshold pair [fixed|train, init_value] as a separate structured field.')
    p.add_argument('--resume_checkpoint', default=None); p.add_argument('--config', default=None)
    p.add_argument('--run_timestamp', default=None, help='결과 checkpoint/metric 루트 아래에 생성할 실행시각 폴더명 suffix. DDP에서는 rank0 값이 전 rank에 공유된다.')
    p.add_argument('--timestamped_output', default='true', help='true이면 실제 산출물을 실행시각 run_<timestamp> 폴더에 저장한다. false이면 기존 경로에 직접 저장한다.')
    p.add_argument('--compile_cpu_threads', type=int, default=None, help='torch.compile/Inductor 준비에 사용할 CPU 스레드 수. DDP 기본값은 2이며 config로 지정할 수 있다.')
    p.add_argument('--compile_cache_mode', default='shared', choices=('shared','per_rank'), help='DDP torch.compile cache policy. shared reuses one Inductor cache across ranks; per_rank isolates cache directories.')
    p.add_argument('--ddp_compile_warmup', default='true', help='DDP에서 rank0가 먼저 compile cache를 priming한 뒤 다른 rank를 진행시킨다.')
    p.add_argument('--ddp_timeout_minutes', type=int, default=120, help='DDP/NCCL collective timeout minutes. 긴 compile/eval 동안 watchdog timeout을 방지한다.')
    p.add_argument('--compile', default='true', help='torch.compile 최적화 사용 여부(true/false). 세부 compile 정책은 코드에서 고정한다.')
    p.add_argument('--amp', default='off', choices=('off', 'on'), help='AMP 모드. off 또는 on만 지원한다. on은 항상 bf16_safe 정책으로 실행한다.')
    return p

def _normalize_analysis_checkpoint_epochs(values: Sequence[str] | None, *, epochs: int) -> list[int]:
    if values is None or len(values) == 0:
        values = [str(epochs)]
    out = sorted({int(v) for v in values if str(v).strip() != ''})
    if not out:
        out = [epochs]
    for e in out:
        if e < 1 or e > epochs:
            raise ValueError('analysis_checkpoint_epochs 범위 오류')
    return out

def _resolve_prepared_paths(dataset: str, prep_root: str) -> tuple[Path, Path]:
    root=Path(str(prep_root)).expanduser().resolve(); d=(root/dataset).resolve(); m=resolve_manifest_path(d)
    if not m.exists(): raise FileNotFoundError(f'manifest 없음: {m}')
    return root,d

def _strict_prepare_checkpoint_dir(checkpoint_root: Path) -> None:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if any(checkpoint_root.iterdir()): raise ValueError('Checkpoint root는 비어 있어야 합니다.')

def _assert_clean_checkpoint_dir(checkpoint_root: Path) -> None:
    for c in checkpoint_root.iterdir():
        if c.is_dir() or c.suffix!='.pt': raise ValueError(f'체크포인트 디렉터리 규칙 위반: {c}')

def _resolve_device(gpu_index: int):
    configure_tf32(enabled=True)
    if torch.cuda.is_available():
        gpu_index = int(gpu_index)
        if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
            raise ValueError(f'gpu_index={gpu_index} is invalid for {torch.cuda.device_count()} visible CUDA device(s).')
        torch.cuda.set_device(gpu_index)
        configure_tf32(enabled=True)
        return torch.device(f'cuda:{gpu_index}')
    return torch.device('cpu')

def _ddp_timeout_from_args(args: argparse.Namespace) -> timedelta:
    raw = getattr(args, 'ddp_timeout_minutes', 120)
    try:
        minutes = int(raw)
    except Exception:
        minutes = 120
    return timedelta(minutes=max(1, minutes))


def _build_ddp_context(args: argparse.Namespace) -> DDPContext:
    indices = _gpu_indices_from_args(args)
    enabled = _ddp_requested(args)
    if not enabled:
        if len(indices) != 1:
            raise ValueError(f'Non-DDP training requires exactly one gpu_index entry, got {indices}.')
        device = _resolve_device(int(indices[0]))
        return DDPContext(False, 0, 0, 1, device, True, tuple(indices))

    if not _parse_bool_config_value(args.batch_size_is_global, default=True):
        raise ValueError('DDP에서는 batch_size_is_global=true만 허용합니다.')
    for key in ('LOCAL_RANK', 'RANK', 'WORLD_SIZE'):
        if key not in os.environ:
            raise ValueError(f'DDP 실행 환경변수 누락: {key}. gpu_index를 2개 이상 주면 자동 torchrun으로 재실행됩니다.')
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    if len(indices) >= 2 and int(world_size) != len(indices):
        raise ValueError(f'gpu_index length={len(indices)} must match torchrun WORLD_SIZE={world_size}.')
    if local_rank < 0:
        raise ValueError('LOCAL_RANK는 0 이상이어야 합니다.')
    if not torch.cuda.is_available() or torch.cuda.device_count() <= local_rank:
        raise ValueError('DDP에는 LOCAL_RANK에 대응하는 CUDA 장치가 필요합니다. CUDA_VISIBLE_DEVICES/gpu_index 설정을 확인하세요.')
    try:
        torch.cuda.set_device(local_rank)
        configure_tf32(enabled=True)
        ddp_timeout = _ddp_timeout_from_args(args)
        try:
            torch.distributed.init_process_group(backend='nccl', device_id=torch.device(f'cuda:{local_rank}'), timeout=ddp_timeout)
        except TypeError:
            torch.distributed.init_process_group(backend='nccl', timeout=ddp_timeout)
    except Exception as exc:
        raise RuntimeError(f'DDP 초기화 실패: {exc}') from exc
    return DDPContext(True, rank, local_rank, world_size, torch.device(f'cuda:{local_rank}'), rank == 0, tuple(indices))

def _resolve_effective_batch_size(args: argparse.Namespace, ctx: DDPContext) -> int:
    g=int(args.batch_size)
    if not ctx.enabled: return g
    if g % int(ctx.world_size) != 0: raise ValueError(f'DDP 사용 시 batch_size는 world_size={int(ctx.world_size)}로 나누어 떨어져야 합니다.')
    b=g//int(ctx.world_size)
    if b<1: raise ValueError('DDP per-rank batch_size는 1 이상이어야 합니다.')
    return b

def _reduce_train_metrics_ddp(metrics: Any, ctx: DDPContext) -> Any:
    if not ctx.enabled:
        return metrics
    vals = torch.tensor([
        metrics.loss * metrics.total,
        metrics.task_loss * metrics.total,
        metrics.regularization_loss * metrics.total,
        metrics.regularization_global_loss * metrics.total,
        metrics.regularization_adjacent_loss * metrics.total,
        metrics.psd_regularization_total * metrics.total,
        metrics.psd_regularization_rep_1d * metrics.total,
        metrics.psd_regularization_pca_1d * metrics.total,
        metrics.psd_regularization_pca_mimo * metrics.total,
        metrics.psd_regularization_rep_input * metrics.total,
        metrics.psd_regularization_rep_adjacent * metrics.total,
        metrics.psd_regularization_pca_1d_input * metrics.total,
        metrics.psd_regularization_pca_1d_adjacent * metrics.total,
        metrics.psd_regularization_pca_mimo_input * metrics.total,
        metrics.psd_regularization_pca_mimo_adjacent * metrics.total,
        float(metrics.correct),
        float(metrics.total),
    ], device=ctx.device, dtype=torch.float64)
    torch.distributed.all_reduce(vals, op=torch.distributed.ReduceOp.SUM)
    total = max(1.0, float(vals[16].item()))
    from src.model.training import TrainEpochMetrics
    return TrainEpochMetrics(
        loss=float(vals[0] / total),
        task_loss=float(vals[1] / total),
        regularization_loss=float(vals[2] / total),
        regularization_global_loss=float(vals[3] / total),
        regularization_adjacent_loss=float(vals[4] / total),
        psd_regularization_total=float(vals[5] / total),
        psd_regularization_rep_1d=float(vals[6] / total),
        psd_regularization_pca_1d=float(vals[7] / total),
        psd_regularization_pca_mimo=float(vals[8] / total),
        psd_regularization_rep_input=float(vals[9] / total),
        psd_regularization_rep_adjacent=float(vals[10] / total),
        psd_regularization_pca_1d_input=float(vals[11] / total),
        psd_regularization_pca_1d_adjacent=float(vals[12] / total),
        psd_regularization_pca_mimo_input=float(vals[13] / total),
        psd_regularization_pca_mimo_adjacent=float(vals[14] / total),
        correct=int(vals[15].item()),
        total=int(vals[16].item()),
        accuracy=float(vals[15] / total),
    )

def _reduce_eval_metrics_ddp(metrics: Any, ctx: DDPContext) -> Any:
    if not ctx.enabled: return metrics
    vals=torch.tensor([
        float(metrics.loss)*float(metrics.total),
        float(metrics.correct),
        float(metrics.total),
    ], device=ctx.device, dtype=torch.float64)
    torch.distributed.all_reduce(vals, op=torch.distributed.ReduceOp.SUM)
    total=float(vals[2].item())
    from src.model.training import EpochMetrics
    if total <= 0.0:
        return EpochMetrics(loss=0.0, accuracy=0.0, correct=0, total=0)
    correct=float(vals[1].item())
    return EpochMetrics(loss=float(vals[0].item()/total), accuracy=float(correct/total), correct=int(round(correct)), total=int(round(total)))

def _evaluation_dataset_for_rank(dataset: Any, ctx: DDPContext) -> tuple[Any, dict[str, Any]]:
    dataset_len = int(len(dataset))
    if not ctx.enabled:
        return dataset, {'policy': 'single_process_full_eval', 'rank': 0, 'world_size': 1, 'local_samples': dataset_len, 'global_samples': dataset_len}
    from torch.utils.data import Subset
    indices = list(range(int(ctx.rank), dataset_len, int(ctx.world_size)))
    subset = Subset(dataset, indices)
    return subset, {
        'policy': 'ddp_rank_strided_subset_no_padding',
        'rank': int(ctx.rank),
        'world_size': int(ctx.world_size),
        'local_samples': int(len(indices)),
        'global_samples': dataset_len,
    }

def _pca_psd_regularization_requested(args: argparse.Namespace) -> bool:
    return any(
        float(getattr(args, key, 0.0) or 0.0) != 0.0
        for key in ('lambda_psd_pca_input', 'lambda_psd_pca_adjacent')
    )


def _psd_regularization_requested(args: argparse.Namespace) -> bool:
    return any(
        float(getattr(args, key, 0.0) or 0.0) != 0.0
        for key in (
            'lambda_psd_rep_input',
            'lambda_psd_rep_adjacent',
            'lambda_psd_pca_input',
            'lambda_psd_pca_adjacent',
        )
    )


def _requested_pca_relations(args: argparse.Namespace) -> list[str]:
    rels: list[str] = []
    if float(getattr(args, 'lambda_psd_pca_input', 0.0) or 0.0) != 0.0:
        rels.append('input')
    if float(getattr(args, 'lambda_psd_pca_adjacent', 0.0) or 0.0) != 0.0:
        rels.append('adjacent')
    return rels


def _normalize_psd_lambda_args(args: argparse.Namespace) -> None:
    for key in (
        'lambda_psd_rep_input',
        'lambda_psd_rep_adjacent',
        'lambda_psd_pca_input',
        'lambda_psd_pca_adjacent',
        'lambda_branch_ortho',
        'lambda_branch_s',
    ):
        if getattr(args, key, None) is None:
            setattr(args, key, 0.0)


def _psd_regularization_metadata_from_args(args: argparse.Namespace) -> dict[str, Any]:
    dims = _parse_pca_dim_per_layer(getattr(args, 'pca_dim_per_layer', None)) or []
    return {
        'lambda_psd_rep_input': float(getattr(args, 'lambda_psd_rep_input', 0.0) or 0.0),
        'lambda_psd_rep_adjacent': float(getattr(args, 'lambda_psd_rep_adjacent', 0.0) or 0.0),
        'lambda_psd_pca_input': float(getattr(args, 'lambda_psd_pca_input', 0.0) or 0.0),
        'lambda_psd_pca_adjacent': float(getattr(args, 'lambda_psd_pca_adjacent', 0.0) or 0.0),
        'lambda_branch_ortho': float(getattr(args, 'lambda_branch_ortho', 0.0) or 0.0),
        'lambda_branch_s': float(getattr(args, 'lambda_branch_s', 0.0) or 0.0),
        'branch_schedule': {
            'soft_mask_epochs': int(getattr(args, 'soft_mask_epochs', 0) or 0),
            'ste_epochs': int(getattr(args, 'ste_epochs', 0) or 0),
            'harden_epoch': None if getattr(args, 'harden_epoch', None) in (None, '') else int(getattr(args, 'harden_epoch')),
        },
        'psd_reg_output_family': str(getattr(args, 'psd_reg_output_family', 'spike')),
        'psd_reg_relations': {
            'rep': [
                rel for rel, key in (
                    ('input', 'lambda_psd_rep_input'),
                    ('adjacent', 'lambda_psd_rep_adjacent'),
                )
                if float(getattr(args, key, 0.0) or 0.0) != 0.0
            ],
            'pca': _requested_pca_relations(args),
        },
        'signal_curve_space': str(getattr(args, 'signal_curve_space', 'exact')),
        'signal_curve_scale': str(getattr(args, 'signal_curve_scale', 'raw')),
        'signal_curve_centering': str(getattr(args, 'signal_curve_centering', 'raw')),
        'signal_curve_reducer': str(getattr(args, 'signal_curve_reducer', 'mean')),
        'signal_curve_distance_metric': str(getattr(args, 'signal_curve_distance_metric', 'centered_l2')),
        'signal_curve_userbin_edges': list(getattr(args, 'signal_curve_userbin_edges', None) or []),
        'signal_curve_userbin_reducer': str(getattr(args, 'signal_curve_userbin_reducer', 'mean')),
        'signal_window': _signal_window_from_args(args),
        'pca_dim_per_layer': dims,
        'pca_dim_semantics': 'dim==1 -> pca_1d; dim>=2 -> pca_mimo',
        'pca_mode_from_dim': ['1d' if int(v) == 1 else 'mimo' for v in dims],
        'ddp_policy': 'local_loss_gradients_all_reduce_mean',
        'pca_reference_bank_policy': (
            'ddp_all_gather_first_local_batch_build_global_reference_rank0_broadcast_per_relation'
            if _ddp_requested(args)
            else 'single_process_first_batch_build_reference_per_relation'
        ),
    }


def _psd_meta_enabled(meta: dict[str, Any] | None) -> bool:
    if not meta:
        return False
    for key in (
        'lambda_psd_rep_input',
        'lambda_psd_rep_adjacent',
        'lambda_psd_pca_input',
        'lambda_psd_pca_adjacent',
    ):
        if float(meta.get(key, 0.0) or 0.0) != 0.0:
            return True
    # Compatibility with old checkpoints/configs.
    for key in ('lambda_psd_rep_1d', 'lambda_psd_pca'):
        if float(meta.get(key, 0.0) or 0.0) != 0.0:
            return True
    return False


def _assert_psd_resume_compatible(current_psd_meta: dict[str, Any], ck_psd_meta: dict[str, Any] | None) -> None:
    psd_enabled_now = _psd_meta_enabled(current_psd_meta)
    psd_enabled_ck = _psd_meta_enabled(ck_psd_meta)
    if not psd_enabled_now and not psd_enabled_ck:
        return
    if ck_psd_meta is None:
        raise ValueError('resume_checkpoint has no psd_regularization_metadata but current run requested PSD regularization.')
    compare_keys = (
        'lambda_psd_rep_input',
        'lambda_psd_rep_adjacent',
        'lambda_psd_pca_input',
        'lambda_psd_pca_adjacent',
        'psd_reg_output_family',
        'signal_curve_space',
        'signal_curve_scale',
        'signal_curve_centering',
        'signal_curve_reducer',
        'signal_curve_distance_metric',
        'signal_curve_userbin_edges',
        'signal_curve_userbin_reducer',
        'signal_window',
        'pca_dim_per_layer',
    )
    for key in compare_keys:
        if str(ck_psd_meta.get(key)) != str(current_psd_meta.get(key)):
            raise ValueError(f'PSD regularization resume mismatch at {key}: current={current_psd_meta.get(key)} checkpoint={ck_psd_meta.get(key)}')

def _parse_pca_dim_per_layer(values: Sequence[str] | None) -> list[int] | None:
    if values is None or len(values)==0: return None
    dims=[]
    for raw in values:
        text=str(raw).strip()
        if text=='': continue
        value=int(text)
        if value<1: raise ValueError('pca_dim_per_layer values must be positive integers.')
        dims.append(value)
    return dims or None

def _normalize_constraint_args(args: argparse.Namespace) -> ConstraintConfig:
    def _parse_json_like(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, str):
            text = value.strip()
            if text in {'', 'none', 'null'}:
                return None
            return json.loads(text) if text.startswith('[') else value
        if isinstance(value, (list, tuple)) and len(value) == 1 and isinstance(value[0], str):
            text = str(value[0]).strip()
            if text.startswith('['):
                return json.loads(text)
        return value
    w_edges = _parse_json_like(getattr(args, 'w_clip_edges', None))
    a_edges = _parse_json_like(getattr(args, 'alpha_clip_edges', None))
    tear = getattr(args, 'tear', 1)
    band_edge = _parse_json_like(getattr(args, 'band_edge', None))
    return ConstraintConfig(
        mode=normalize_scenario_mode(getattr(args, 'scenario_mode', 'none')),
        w_clip_edges=_parse_json_like(w_edges),
        alpha_clip_edges=_parse_json_like(a_edges),
        band_edge=band_edge,
                tear=int(tear),
    )

def _is_fixed_pca_reference_like(obj: Any) -> bool:
    return all(hasattr(obj, name) for name in ('x_basis', 'x_centroid', 'y_basis', 'y_centroid', 'dim'))


def _cpu_pca_reference_bank(bank: dict[str, Any]) -> dict[str, Any]:
    from src.model.psd_minibatch_regularizer import FixedPCALayerReference
    out: dict[str, Any] = {}
    for name, ref in bank.items():
        if isinstance(ref, dict):
            out[str(name)] = _cpu_pca_reference_bank(ref)
            continue
        if not _is_fixed_pca_reference_like(ref):
            raise TypeError(f'Invalid PCA reference bank entry at {name!r}: {type(ref).__name__}')
        out[str(name)] = FixedPCALayerReference(
            layer_name=str(ref.layer_name),
            layer_index=(None if getattr(ref, 'layer_index', None) is None else int(ref.layer_index)),
            dim=int(ref.dim),
            x_basis=ref.x_basis.detach().cpu(),
            x_centroid=ref.x_centroid.detach().cpu(),
            y_basis=ref.y_basis.detach().cpu(),
            y_centroid=ref.y_centroid.detach().cpu(),
            output_family=str(getattr(ref, 'output_family', 'spike')),
            basis_id=getattr(ref, 'basis_id', None),
            metadata=dict(getattr(ref, 'metadata', {}) or {}),
        )
    return out


def _flatten_pca_reference_bank(bank: dict[str, Any] | None) -> dict[str, Any]:
    if not bank:
        return {}
    out: dict[str, Any] = {}
    for key, value in bank.items():
        if isinstance(value, dict):
            for subkey, ref in _flatten_pca_reference_bank(value).items():
                out[f'{key}/{subkey}'] = ref
        else:
            out[str(key)] = value
    return out

def _pca_reference_payload_error(message: str, *, rank: int) -> dict[str, Any]:
    return {'rank': int(rank), 'error': str(message)}


def _tensor_to_cpu_reference_payload(value: Any) -> Any:
    if value is None:
        return None
    torch_mod = _torch_module()
    if isinstance(value, torch_mod.Tensor):
        return value.detach().to(device='cpu', dtype=torch_mod.float32).contiguous()
    return value


def _collect_local_pca_reference_payload(model: Any, train_loader: Any, args: argparse.Namespace, ctx: DDPContext) -> dict[str, Any]:
    """Run one local rank forward and return CPU tensors needed for global PCA basis fitting."""

    from src.model.training import _move_inputs_to_device, _reset_stateful_model

    iterator = iter(train_loader)
    try:
        inputs, _target = next(iterator)
    except StopIteration as exc:
        raise RuntimeError('Cannot build PCA PSD reference bank from an empty train_loader.') from exc

    # Keep torch.compile wrappers active for the warm-up/reference forward, but
    # remove DDP itself to avoid creating a gradient-synchronizing training
    # forward outside the normal epoch loop.
    forward_model = _model_for_evaluation(model)
    base_model = _unwrap_model(model)
    was_training = bool(getattr(base_model, 'training', False))
    try:
        if hasattr(forward_model, 'eval'):
            forward_model.eval()
        with _torch_module().inference_mode():
            device_inputs = _move_inputs_to_device(base_model, inputs, device=ctx.device)
            _reset_stateful_model(base_model)
            result = forward_model(device_inputs, capture_hidden=True)
            _reset_stateful_model(base_model)
    finally:
        if was_training and hasattr(forward_model, 'train'):
            forward_model.train()

    output_family = str(getattr(args, 'psd_reg_output_family', 'spike')).strip().lower()
    need_membrane = output_family == 'membrane'
    records: list[dict[str, Any]] = []
    for layer_index, record in enumerate(list(result.hidden_records)):
        payload: dict[str, Any] = {
            'layer_index': int(layer_index),
            'layer_name': str(record.layer_name if record.layer_name else f'hidden_{layer_index}'),
            'spike': _tensor_to_cpu_reference_payload(record.spike),
        }
        if need_membrane:
            payload['membrane'] = _tensor_to_cpu_reference_payload(record.membrane)
        records.append(payload)

    return {
        'rank': int(ctx.rank),
        'local_rank': int(ctx.local_rank),
        'world_size': int(ctx.world_size),
        'input': _tensor_to_cpu_reference_payload(device_inputs),
        'hidden_records': records,
        'num_samples': int(device_inputs.shape[0]) if isinstance(device_inputs, torch.Tensor) and device_inputs.ndim > 0 else 0,
    }


def _gather_pca_reference_payloads(local_payload: dict[str, Any], ctx: DDPContext) -> list[dict[str, Any]]:
    if not bool(getattr(ctx, 'enabled', False)):
        return [local_payload]
    gathered: list[Any] = [None for _ in range(int(ctx.world_size))]
    torch.distributed.all_gather_object(gathered, local_payload)
    return [dict(item) for item in gathered]


def _concat_reference_tensors(payloads: list[dict[str, Any]], key: str) -> torch.Tensor:
    tensors = [payload[key] for payload in payloads]
    torch_mod = _torch_module()
    if not tensors or not all(isinstance(tensor, torch_mod.Tensor) for tensor in tensors):
        raise ValueError(f'PCA reference gather missing tensor key {key!r}.')
    shapes = [tuple(int(v) for v in tensor.shape[1:]) for tensor in tensors]
    if len(set(shapes)) != 1:
        raise ValueError(f'PCA reference gathered tensor shapes for {key!r} differ across ranks: {shapes}.')
    return torch_mod.cat([tensor.contiguous() for tensor in tensors], dim=0)


def _build_global_pca_reference_bank_from_payloads(payloads: list[dict[str, Any]], args: argparse.Namespace, ctx: DDPContext) -> dict[str, Any]:
    from src.model.snn_builder import LayerRecord
    from src.model.psd_minibatch_regularizer import compute_fixed_pca_reference_bank

    payloads = sorted(payloads, key=lambda item: int(item.get('rank', 0)))
    errors = [f"rank{int(item.get('rank', -1))}: {item.get('error')}" for item in payloads if item.get('error')]
    if errors:
        raise RuntimeError('PCA reference bank local gather failed: ' + '; '.join(errors))
    if not payloads:
        raise RuntimeError('PCA reference bank gather returned no rank payloads.')

    reference_input = _concat_reference_tensors(payloads, 'input')
    hidden_lists = [list(item.get('hidden_records', [])) for item in payloads]
    if not hidden_lists or not hidden_lists[0]:
        raise RuntimeError('PCA PSD regularization requires at least one hidden layer reference.')
    hidden_count = len(hidden_lists[0])
    if any(len(records) != hidden_count for records in hidden_lists):
        lengths = [len(records) for records in hidden_lists]
        raise ValueError(f'PCA reference hidden layer count differs across ranks: {lengths}.')

    global_records: list[LayerRecord] = []
    output_family = str(getattr(args, 'psd_reg_output_family', 'spike')).strip().lower()
    for layer_index in range(hidden_count):
        names = [str(records[layer_index].get('layer_name', f'hidden_{layer_index}')) for records in hidden_lists]
        if len(set(names)) != 1:
            raise ValueError(f'PCA reference hidden layer names differ across ranks at layer {layer_index}: {names}.')
        record_payloads = [records[layer_index] for records in hidden_lists]
        torch_mod = _torch_module()
        spike = torch_mod.cat([payload['spike'].contiguous() for payload in record_payloads], dim=0)
        if output_family == 'membrane':
            membrane = torch_mod.cat([payload['membrane'].contiguous() for payload in record_payloads], dim=0)
        else:
            membrane = torch_mod.empty(0, dtype=spike.dtype)
        global_records.append(LayerRecord(layer_name=names[0], membrane=membrane, spike=spike, layer_input=None))

    base_metadata = {
        'reference_policy': (
            'ddp_all_gather_first_local_batch_build_global_reference_rank0_broadcast_per_relation'
            if bool(ctx.enabled)
            else 'single_process_first_batch_build_reference_per_relation'
        ),
        'world_size': int(getattr(ctx, 'world_size', 1)),
        'rank_order': [int(item.get('rank', 0)) for item in payloads],
        'per_rank_reference_samples': [int(item.get('num_samples', 0)) for item in payloads],
        'global_reference_samples': int(reference_input.shape[0]) if reference_input.ndim > 0 else 0,
    }
    relations = _requested_pca_relations(args)
    if not relations:
        return {}
    banks: dict[str, Any] = {}
    for relation in relations:
        bank = compute_fixed_pca_reference_bank(
            reference_input,
            global_records,
            str(args.psd_reg_output_family),
            _parse_pca_dim_per_layer(args.pca_dim_per_layer),
            variant=str(getattr(args, 'signal_curve_centering', getattr(args, 'regularization_centering', 'raw'))),
            relation=relation,
            metadata={**base_metadata, 'relation': relation},
        )
        if not bank:
            raise RuntimeError(f'PCA PSD regularization requires at least one hidden layer reference for relation={relation}.')
        banks[relation] = bank
    return banks


def _build_pca_reference_bank_if_needed(model: Any, train_loader: Any, args: argparse.Namespace, ctx: DDPContext) -> dict[str, Any] | None:
    if not _pca_psd_regularization_requested(args):
        return None

    try:
        local_payload = _collect_local_pca_reference_payload(model, train_loader, args, ctx)
    except Exception as exc:
        local_payload = _pca_reference_payload_error(f'{type(exc).__name__}: {exc}', rank=int(ctx.rank))

    gathered_payloads = _gather_pca_reference_payloads(local_payload, ctx)
    obj: list[Any] = [None]
    if _is_rank0(ctx):
        try:
            bank = _build_global_pca_reference_bank_from_payloads(gathered_payloads, args, ctx)
            obj[0] = _cpu_pca_reference_bank(bank)
        except Exception as exc:
            obj[0] = {'__pca_reference_error__': f'{type(exc).__name__}: {exc}'}
    if ctx.enabled:
        _torch_module().distributed.broadcast_object_list(obj, src=0)
    if isinstance(obj[0], dict) and '__pca_reference_error__' in obj[0]:
        raise RuntimeError(str(obj[0]['__pca_reference_error__']))
    return obj[0]

def _bundle_input_shape(bundle: Any, *, model_family: str, manifest: dict[str, Any] | None = None) -> list[int] | None:
    manifest = dict(manifest or {})
    family = str(model_family)
    psd_axis_kind = str(manifest.get('psd_axis_kind', ''))
    image_like = psd_axis_kind in {'static_repeat', 'image_temporal', 'raster_spatial'} or bool(manifest.get('cnn_input_shape'))
    image_consuming_families = {'cnn_lif', 'cnn_rf', 'cnn', 'spikformer'}
    if family in image_consuming_families:
        for key in ('cnn_input_shape', 'physical_input_shape', 'stored_shape', 'input_shape'):
            value = manifest.get(key)
            if isinstance(value, (list, tuple)) and value:
                return [int(v) for v in value]
    if image_like:
        # Dense/MLP and sequence families consume the selected flattened view.
        # Passing rank-4 image metadata to those builders can trigger the image
        # adapter path or explicit shape-rejection checks, e.g. spikegru.
        return None
    for key in ('input_shape', 'physical_input_shape', 'stored_shape'):
        value = manifest.get(key)
        if isinstance(value, (list, tuple)) and value:
            return [int(v) for v in value]
    return None


def _checkpoint_model_config(*, model: Any, args: argparse.Namespace, bundle: Any, model_spec: Any, manifest: dict[str, Any] | None) -> dict[str, Any]:
    base_model = _unwrap_model(model)
    model_meta = getattr(base_model, 'model_metadata', lambda: {})()
    input_shape = model_meta.get('cnn_input_shape') or model_meta.get('input_shape') or _bundle_input_shape(bundle, model_family=str(model_spec.family), manifest=manifest)
    hidden_spec = str(getattr(args, 'hidden_spec', '')).strip()
    hidden_sizes = [int(v) for v in getattr(bundle, 'default_hidden_sizes', ())]
    return {
        'input_dim': int(bundle.input_dim),
        'sequence_length': int(bundle.sequence_length),
        'num_classes': int(bundle.num_classes),
        'input_shape': None if input_shape is None else [int(v) for v in input_shape],
        'hidden_spec': hidden_spec,
        'arch_spec': str(model_meta.get('arch_spec') or model_meta.get('hidden_spec') or hidden_spec),
        'hidden_sizes': hidden_sizes,
        'v_th': ['train' if bool(getattr(model_spec, 'trainable_threshold', False)) else 'fixed', float(getattr(model_spec, 'threshold_value', 1.0))],
        'neuron_type': getattr(args, 'neuron_type', None),
        'model_spec_contract': 'explicit_structured_fields_only',
        'recurrent': getattr(args, 'recurrent', False),
        'reset': getattr(args, 'reset', None),
        'filter': getattr(args, 'filter', None),
        'branch': getattr(args, 'branch', None),
        'rf_pole_radius_constrained': getattr(args, 'rf_pole_radius_constrained', None),
        'rf_pole_radius_max': float(getattr(args, 'rf_pole_radius_max', 0.9999)),
        'model_metadata': dict(model_meta or {}),
    }


def _checkpoint_readout_config(*, readout_mode: str, bundle: Any) -> dict[str, Any]:
    return {
        'mode': str(readout_mode),
        'readout_mode': str(readout_mode),
        'num_classes': int(bundle.num_classes),
        'sequence_length': int(bundle.sequence_length),
    }


def _checkpoint_payload(**kwargs):
    model=kwargs.pop('model')
    payload = {
        **kwargs,
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'checkpoint_schema_version': CHECKPOINT_SCHEMA_VERSION,
        'checkpoint_format': 'state_dict_payload',
        'state_dict_key_format': 'unwrapped_eager',
        'checkpoint_contract': {
            'state_dict_key': 'state_dict',
            'state_dict_saved_from': 'unwrap_model(model).state_dict()',
            'wrapper_prefixes_removed_on_save': ['module.', '_orig_mod.'],
            'loader_prefix_normalization': ['module.', '_orig_mod.'],
        },
        'state_dict': normalize_state_dict_keys(_unwrap_model(model).state_dict()),
    }
    metadata_payload = {key: value for key, value in payload.items() if key != 'state_dict'}
    payload['checkpoint_metadata_format'] = 'json'
    payload['checkpoint_metadata_json'] = json.dumps(to_jsonable(metadata_payload), ensure_ascii=False, sort_keys=True)
    return payload


def _canonical_run_readout_mode(mode: str) -> str:
    try:
        fn = canonicalize_readout_mode
    except NameError:
        from src.readout.readout import canonicalize_readout_mode as fn
    return fn(mode)


_FIXED_COMPILE_BACKEND_CUDA = 'inductor'
_FIXED_COMPILE_BACKEND_CPU = 'eager'
_FIXED_COMPILE_FULLGRAPH = True
_FIXED_COMPILE_DYNAMIC = False
_FIXED_COMPILE_STANCE = 'eager_then_compile'
_FIXED_COMPILE_THREAD_CAP = 8
_FIXED_COMPILE_POLICY = 'regional_sequence_compile_with_fixed_shape_guard'


def _compile_requested_from_args(args: argparse.Namespace) -> bool:
    return _parse_bool_config_value(getattr(args, 'compile', True), default=True)


def _amp_bf16_safe_active(device: Any, requested: bool) -> bool:
    if not bool(requested):
        return False
    if str(getattr(device, 'type', device)) != 'cuda':
        return False
    torch_mod = _torch_module()
    cuda = getattr(torch_mod, 'cuda', None)
    if cuda is None or not cuda.is_available():
        return False
    checker = getattr(cuda, 'is_bf16_supported', None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return False
    try:
        major, _minor = cuda.get_device_capability(device)
        return int(major) >= 8
    except Exception:
        return False


def _torch_module() -> Any:
    """Return the loaded torch module, importing lazily for helper-level tests."""

    global torch
    try:
        return torch
    except NameError:
        import torch as _torch
        torch = _torch
        return _torch


def _compile_cpu_thread_count_from_args(args: argparse.Namespace | None, ctx: DDPContext | None) -> tuple[int, str]:
    raw = None if args is None else getattr(args, 'compile_cpu_threads', None)
    if raw not in {None, ''}:
        value = int(raw)
        if value < 1:
            raise ValueError('compile_cpu_threads must be a positive integer.')
        return value, 'argument'
    env_value = os.environ.get('PSD_TORCH_COMPILE_CPU_THREADS')
    if env_value not in {None, ''}:
        value = int(env_value)
        if value < 1:
            raise ValueError('PSD_TORCH_COMPILE_CPU_THREADS must be a positive integer.')
        return value, 'env:PSD_TORCH_COMPILE_CPU_THREADS'
    if ctx is not None and bool(getattr(ctx, 'enabled', False)):
        return 2, 'ddp_default_2'
    return max(1, min(_default_cpu_thread_count(), int(_FIXED_COMPILE_THREAD_CAP))), 'single_process_auto_cap_8_or_env'

def _configure_compile_cpu_threads(args: argparse.Namespace | None = None, ctx: DDPContext | None = None) -> dict[str, Any]:
    num_threads, source = _compile_cpu_thread_count_from_args(args, ctx)
    num_threads = max(1, int(num_threads))
    interop_threads = max(1, min(num_threads, max(1, num_threads // 2)))
    thread_env_keys = _THREAD_ENV_KEYS
    previous_env = {key: os.environ.get(key) for key in thread_env_keys}
    for key in thread_env_keys:
        os.environ[key] = str(num_threads)
    torch_mod = _torch_module()
    set_num_threads_error = None
    set_num_interop_threads_error = None
    try:
        torch_mod.set_num_threads(num_threads)
    except Exception as exc:
        set_num_threads_error = f'{type(exc).__name__}: {exc}'
    try:
        torch_mod.set_num_interop_threads(interop_threads)
    except Exception as exc:
        set_num_interop_threads_error = f'{type(exc).__name__}: {exc}'
    status = {
        'num_threads': num_threads,
        'num_interop_threads': interop_threads,
        'policy': 'argument_or_env_or_ddp_default_2_else_auto_cap_8',
        'source': source,
        'thread_cap': int(_FIXED_COMPILE_THREAD_CAP),
        'env_keys': {key: os.environ.get(key) for key in thread_env_keys},
        'previous_env': previous_env,
    }
    if set_num_threads_error is not None:
        status['set_num_threads_error'] = set_num_threads_error
    if set_num_interop_threads_error is not None:
        status['set_num_interop_threads_error'] = set_num_interop_threads_error
    return status


def _install_compile_cpu_thread_env_from_args(args: argparse.Namespace, *, ddp_enabled: bool, compile_enabled: bool) -> dict[str, Any]:
    """Apply compile CPU thread env before runtime modules import torch-heavy builders."""

    if not bool(compile_enabled):
        return {'enabled': False, 'reason': 'compile_disabled'}
    pseudo_ctx = DDPContext(bool(ddp_enabled), 0, 0, 1, None, True)
    num_threads, source = _compile_cpu_thread_count_from_args(args, pseudo_ctx)
    for key in _THREAD_ENV_KEYS:
        os.environ[key] = str(num_threads)
    return {
        'enabled': True,
        'num_threads': int(num_threads),
        'source': source,
        'env_keys': {key: os.environ.get(key) for key in _THREAD_ENV_KEYS},
        'phase': 'pre_runtime_import',
    }


def _configure_compile_cache(*, enabled: bool, rank: int = 0, ddp_enabled: bool = False, cache_mode: str = 'shared') -> dict[str, Any]:
    if not bool(enabled):
        return {'enabled': False}
    root = Path(os.environ.get('PSD_TORCH_COMPILE_CACHE_DIR', str(Path.cwd() / '.torch_compile_cache'))).expanduser().resolve()
    mode = str(cache_mode or 'shared').strip().lower()
    if mode not in {'shared', 'per_rank'}:
        raise ValueError("compile_cache_mode must be 'shared' or 'per_rank'.")
    cache_leaf = f'rank{int(rank)}' if (bool(ddp_enabled) and mode == 'per_rank') else ('shared' if bool(ddp_enabled) else 'single')
    cache_dir = root / cache_leaf
    cache_dir.mkdir(parents=True, exist_ok=True)
    previous_env = {
        'TORCHINDUCTOR_CACHE_DIR': os.environ.get('TORCHINDUCTOR_CACHE_DIR'),
        'TORCHINDUCTOR_FX_GRAPH_CACHE': os.environ.get('TORCHINDUCTOR_FX_GRAPH_CACHE'),
        'TORCHINDUCTOR_AUTOGRAD_CACHE': os.environ.get('TORCHINDUCTOR_AUTOGRAD_CACHE'),
    }
    # Assign explicitly.  For DDP the default is one shared cache directory so
    # graph/codegen artifacts produced by rank0 can be reused by the remaining
    # ranks instead of every rank compiling an identical graph into rank-local
    # directories.
    os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(cache_dir)
    os.environ['TORCHINDUCTOR_FX_GRAPH_CACHE'] = '1'
    os.environ['TORCHINDUCTOR_AUTOGRAD_CACHE'] = '1'
    status: dict[str, Any] = {
        'enabled': True,
        'root': str(root),
        'rank': int(rank),
        'ddp_enabled': bool(ddp_enabled),
        'mode': mode,
        'cache_leaf': cache_leaf,
        'cache_dir': str(cache_dir),
        'PSD_TORCH_COMPILE_CACHE_DIR': os.environ.get('PSD_TORCH_COMPILE_CACHE_DIR'),
        'TORCHINDUCTOR_CACHE_DIR': os.environ.get('TORCHINDUCTOR_CACHE_DIR'),
        'TORCHINDUCTOR_FX_GRAPH_CACHE': os.environ.get('TORCHINDUCTOR_FX_GRAPH_CACHE'),
        'TORCHINDUCTOR_AUTOGRAD_CACHE': os.environ.get('TORCHINDUCTOR_AUTOGRAD_CACHE'),
        'previous_env': previous_env,
    }
    try:
        import torch._inductor.config as inductor_config  # type: ignore
        for attr, value in (('fx_graph_cache', True), ('autotune_local_cache', True)):
            try:
                setattr(inductor_config, attr, value)
                status[attr] = value
            except Exception as exc:
                status[f'{attr}_error'] = f'{type(exc).__name__}: {exc}'
    except Exception as exc:  # pragma: no cover - version dependent
        status['inductor_config_error'] = f'{type(exc).__name__}: {exc}'
    return status


def _fixed_compile_kwargs_for_device(device: Any | None = None) -> tuple[dict[str, Any], str]:
    device_type = str(getattr(device, 'type', device) or 'cpu')
    backend = _FIXED_COMPILE_BACKEND_CUDA if device_type == 'cuda' else _FIXED_COMPILE_BACKEND_CPU
    return {
        'backend': backend,
        'fullgraph': bool(_FIXED_COMPILE_FULLGRAPH),
        'dynamic': bool(_FIXED_COMPILE_DYNAMIC),
    }, f'fixed_backend={backend};fullgraph={_FIXED_COMPILE_FULLGRAPH};dynamic={_FIXED_COMPILE_DYNAMIC}'


def _apply_fixed_compile_stance(*, enabled: bool) -> str:
    if not bool(enabled):
        return 'not_requested'
    torch_mod = _torch_module()
    compiler = getattr(torch_mod, 'compiler', None)
    set_stance = getattr(compiler, 'set_stance', None) if compiler is not None else None
    if not callable(set_stance):
        return 'torch.compiler.set_stance_unavailable'
    try:
        set_stance(_FIXED_COMPILE_STANCE)
    except Exception as exc:
        return f'torch.compiler.set_stance_failed:{type(exc).__name__}: {exc}'
    return f'torch.compiler.set_stance({_FIXED_COMPILE_STANCE})'


def _wrap_compiled_model_with_runtime_fallback(model: Any, compiled: Any) -> Any:
    """Use compiled forward first, then fall back to eager on runtime compile errors."""
    torch_mod = _torch_module()
    nn_mod = getattr(getattr(torch_mod, 'nn', None), 'Module', None)
    if nn_mod is None or not isinstance(model, nn_mod):
        return compiled

    class _RuntimeCompileFallback(nn_mod):
        def __init__(self, original: Any, compiled_model: Any) -> None:
            super().__init__()
            object.__setattr__(self, '_orig_mod', original)
            self.compiled_model = compiled_model
            self.compile_runtime_disabled = False
            self.compile_runtime_error = None

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            if not bool(self.compile_runtime_disabled):
                try:
                    return self.compiled_model(*args, **kwargs)
                except Exception as exc:  # pragma: no cover - backend/runtime dependent
                    self.compile_runtime_disabled = True
                    self.compile_runtime_error = f'{type(exc).__name__}: {exc}'
                    warnings.warn(
                        '[model_training] torch.compile runtime fallback activated for top-level model: '
                        + self.compile_runtime_error,
                        RuntimeWarning,
                        stacklevel=2,
                    )
            return self._orig_mod(*args, **kwargs)

    return _RuntimeCompileFallback(model, compiled)


def _maybe_compile_readout(readout: Any, *, requested: bool, compile_kwargs: dict[str, Any]) -> tuple[bool, str]:
    if not bool(requested):
        return False, 'disabled'
    hook = getattr(readout, 'enable_compiled_forward', None)
    if not callable(hook):
        return False, 'readout_compile_hook_unavailable'
    try:
        applied, policy = hook(**dict(compile_kwargs or {}))
    except Exception as exc:  # pragma: no cover - backend/version dependent
        return False, f'readout_compile_construction_failed:{type(exc).__name__}: {exc}'
    return bool(applied), str(policy)




def _ddp_compile_warmup_requested(args: argparse.Namespace) -> bool:
    return _parse_bool_config_value(getattr(args, 'ddp_compile_warmup', True), default=True)


def _prime_ddp_compile_cache_if_needed(
    *,
    model: Any,
    readout: Any,
    train_loader: Any,
    args: argparse.Namespace,
    ctx: DDPContext,
    compile_requested: bool,
    amp_bf16_safe_active: bool,
) -> dict[str, Any]:
    """Let rank0 populate the shared compile cache before other ranks enter training.

    torch.compile is lazy: most regional functions are compiled on first call, not
    when ``enable_compiled_forward`` is installed.  With a shared Inductor cache,
    rank0 can trigger that first call and then release the other ranks through a
    barrier.  This does not make CUDA kernels magically single-instance at
    runtime, but it avoids the worst case where every DDP rank code-generates the
    same graph into an isolated cache directory.
    """

    if not bool(ctx.enabled):
        return {'enabled': False, 'reason': 'not_ddp'}
    if not bool(compile_requested):
        return {'enabled': False, 'reason': 'compile_disabled'}
    if not _ddp_compile_warmup_requested(args):
        return {'enabled': False, 'reason': 'ddp_compile_warmup_false'}
    if str(getattr(args, 'compile_cache_mode', 'shared')).strip().lower() != 'shared':
        return {'enabled': False, 'reason': 'compile_cache_mode_not_shared'}

    status: dict[str, Any] = {
        'enabled': True,
        'policy': 'rank0_eval_first_batch_then_barrier_shared_inductor_cache',
        'rank': int(ctx.rank),
    }
    if int(ctx.rank) == 0:
        try:
            first_inputs, first_target = next(iter(train_loader))
            batch = eval_one_batch(
                model,
                first_inputs,
                first_target,
                readout=readout,
                device=ctx.device,
                amp_bf16_safe=bool(amp_bf16_safe_active),
            )
            status.update({
                'rank0_warmup': 'ok',
                'warmup_batch_size': int(batch.total),
                'warmup_accuracy': float(batch.accuracy),
            })
        except StopIteration:
            status.update({'rank0_warmup': 'skipped_empty_train_loader'})
        except Exception as exc:  # pragma: no cover - backend/runtime dependent
            status.update({'rank0_warmup': 'failed', 'error': f'{type(exc).__name__}: {exc}'})
            warnings.warn('[model_training] DDP compile cache warmup failed on rank0: ' + status['error'], RuntimeWarning, stacklevel=2)
        finally:
            try:
                if str(getattr(ctx.device, 'type', ctx.device)) == 'cuda':
                    torch.cuda.empty_cache()
            except Exception:
                pass
    _ddp_barrier(ctx)
    return status


def _maybe_compile_model(
    model: Any,
    *,
    requested: bool,
    device: Any | None = None,
) -> tuple[Any, bool, str, dict[str, Any]]:
    kwargs, kwargs_note = _fixed_compile_kwargs_for_device(device)
    if not bool(requested):
        return model, False, 'disabled', kwargs
    torch_mod = _torch_module()
    compile_fn = getattr(torch_mod, 'compile', None)
    if compile_fn is None:
        raise RuntimeError('compile=true but torch.compile is unavailable in this environment.')
    model_compile_hook = getattr(model, 'enable_compiled_forward', None)
    if callable(model_compile_hook):
        applied, policy = model_compile_hook(**kwargs)
        if applied:
            return model, True, str(policy) + f'[{kwargs_note}]', kwargs
        # Project models own their compile granularity through per-layer sequence
        # regions.  Do not fall back to a top-level model compile when no sequence
        # region was installed; that reintroduces the large outer-loop graph path.
        return model, False, str(policy), kwargs
    try:
        try:
            compiled = compile_fn(model, **kwargs)
        except TypeError:
            compiled = compile_fn(model)
    except Exception as exc:
        return model, False, 'torch.compile_construction_failed:' + f'{type(exc).__name__}: {exc}', kwargs
    policy = 'torch.compile(' + ','.join(f'{k}={v}' for k, v in sorted(kwargs.items())) + ')'
    return _wrap_compiled_model_with_runtime_fallback(model, compiled), True, policy + f'[{kwargs_note}]', kwargs


def _apply_channels_last_if_cnn(model: Any, model_spec: Any) -> tuple[Any, dict[str, Any]]:
    if str(getattr(model_spec, 'family', '')) not in {'cnn_lif', 'cnn_rf'}:
        return model, {'requested': False, 'applied': False, 'reason': 'not_cnn'}
    try:
        model = model.to(memory_format=torch.channels_last)
    except Exception as exc:  # pragma: no cover - backend/version dependent
        return model, {'requested': True, 'applied': False, 'reason': f'{type(exc).__name__}: {exc}'}
    return model, {'requested': True, 'applied': True, 'reason': 'channels_last_applied'}

def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temp_dir=Path(tempfile.mkdtemp(prefix='checkpoint_write_', dir=str(path.parent.parent))); temp_path=temp_dir/path.name
    try: torch.save(payload,temp_path); os.replace(temp_path,path)
    finally: shutil.rmtree(temp_dir, ignore_errors=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser=build_arg_parser(); args=parse_args_with_config(parser, argv=argv, stage_key='model_training')
    _maybe_reexec_for_gpu_index_ddp(args, argv)
    _normalize_psd_lambda_args(args)
    ctx=None
    ddp=_ddp_requested(args); args.batch_size_is_global=_parse_bool_config_value(args.batch_size_is_global, default=True)
    amp_mode = normalize_amp_mode(getattr(args, 'amp', 'off'))
    amp_public_mode = 'on' if amp_mode == 'bf16_safe' else 'off'
    args.amp = amp_public_mode
    args.signal_window = _signal_window_from_args(args)
    _install_signal_curve_internal_aliases(args)
    _configure_token_regularizer_env_from_args(args)
    amp_bf16_safe_requested = (amp_mode == 'bf16_safe')
    compile_requested=_compile_requested_from_args(args)
    preload_compile_threads_policy = _install_compile_cpu_thread_env_from_args(
        args, ddp_enabled=bool(ddp), compile_enabled=bool(compile_requested)
    )
    _load_runtime_dependencies()
    try:
        tf32_policy=configure_tf32(enabled=True)
        if int(args.batch_size)<1: parser.error('--batch_size must be >= 1.')
        analysis_checkpoint_epochs=_normalize_analysis_checkpoint_epochs(args.analysis_checkpoint_epochs, epochs=int(args.epochs))
        dataset_token=str(args.dataset); prep_root,prepared_dataset_path=_resolve_prepared_paths(dataset_token,args.prep_root)
        resume_checkpoint=None if args.resume_checkpoint is None else Path(args.resume_checkpoint).expanduser().resolve()
        ctx=_build_ddp_context(args)
        checkpoint_root, metric_root, run_timestamp, timestamped_output = _resolve_training_result_roots(args, ctx)
        amp_bf16_safe_active=_amp_bf16_safe_active(ctx.device, amp_bf16_safe_requested)
        if _is_rank0(ctx):
            if resume_checkpoint is None: _strict_prepare_checkpoint_dir(checkpoint_root)
            else: checkpoint_root.mkdir(parents=True, exist_ok=True); _assert_clean_checkpoint_dir(checkpoint_root)
            metric_root.mkdir(parents=True, exist_ok=True)
        _ddp_barrier(ctx)
        _seed_everything(int(args.seed))
        model_spec=model_spec_from_namespace(args); bundle=select_training_view_for_model(resolve_dataset_bundle(dataset_token, prep_root=prep_root), model_family=model_spec.family)
        image_mlp_flatten_policy = validate_image_mlp_flatten_contract(bundle, model_family=model_spec.family, stage=SOURCE_PROGRAM)
        manifest = load_manifest(Path(bundle.manifest_path))
        if not isinstance(manifest, dict):
            raise ValueError(f'Prepared manifest must be a mapping: {bundle.manifest_path}')
        constraint_config=_normalize_constraint_args(args)
        per_rank_batch=_resolve_effective_batch_size(args, ctx)
        train_dataset_len = int(len(bundle.train_dataset))
        min_global_batch = max(1, per_rank_batch * int(ctx.world_size))
        drop_last_train=bool(compile_requested and train_dataset_len >= min_global_batch)
        train_sampler=None
        if ctx.enabled:
            train_sampler=DistributedSampler(bundle.train_dataset, num_replicas=ctx.world_size, rank=ctx.rank, shuffle=True, seed=int(args.seed), drop_last=drop_last_train)
        train_loader=make_loader(bundle.train_dataset,batch_size=per_rank_batch,shuffle=True,num_workers=int(args.num_workers),pin_memory=ctx.device.type=='cuda',seed=int(args.seed),sampler=train_sampler,drop_last=drop_last_train)
        eval_dataset, eval_dataset_policy = _evaluation_dataset_for_rank(bundle.test_dataset, ctx)
        eval_batch_size = per_rank_batch if ctx.enabled else int(args.batch_size)
        test_loader=make_loader(eval_dataset,batch_size=eval_batch_size,shuffle=False,num_workers=int(args.num_workers),pin_memory=ctx.device.type=='cuda',seed=int(args.seed),drop_last=False)
        canonical_readout_mode = _canonical_run_readout_mode(str(args.readout_mode))
        readout=build_readout(canonical_readout_mode, num_classes=bundle.num_classes, sequence_length=bundle.sequence_length, device=ctx.device)
        compile_threads_policy = _configure_compile_cpu_threads(args, ctx)
        compile_threads_policy['preload'] = dict(preload_compile_threads_policy)
        compile_cache_policy = _configure_compile_cache(enabled=compile_requested, rank=(ctx.local_rank if ctx.enabled else 0), ddp_enabled=bool(ctx.enabled), cache_mode=str(getattr(args, 'compile_cache_mode', 'shared')))
        compile_stance_policy = _apply_fixed_compile_stance(enabled=compile_requested)
        previous_compile_env = os.environ.get('PSD_TORCH_COMPILE_REQUESTED')
        if compile_requested:
            os.environ['PSD_TORCH_COMPILE_REQUESTED'] = '1'
        else:
            os.environ.pop('PSD_TORCH_COMPILE_REQUESTED', None)
        try:
            model_input_shape = _bundle_input_shape(bundle, model_family=str(model_spec.family), manifest=manifest)
            model=build_snn_classifier(model_token=model_spec,input_dim=bundle.input_dim,sequence_length=bundle.sequence_length,num_classes=bundle.num_classes,input_shape=model_input_shape,hidden_sizes=tuple(int(v) for v in bundle.default_hidden_sizes),arch_spec=str(args.hidden_spec).strip(),output_layer_overrides=readout.output_layer_overrides(),v_th=float(getattr(model_spec, 'threshold_value', 1.0)),constraint_config=constraint_config).to(ctx.device)
            model, channels_last_policy = _apply_channels_last_if_cnn(model, model_spec)
        finally:
            if previous_compile_env is None:
                os.environ.pop('PSD_TORCH_COMPILE_REQUESTED', None)
            else:
                os.environ['PSD_TORCH_COMPILE_REQUESTED'] = previous_compile_env
        readout.to(ctx.device)
        if resume_checkpoint is not None:
            payload=load_torch_checkpoint(resume_checkpoint, map_location=ctx.device)
            ck_meta=payload.get('constraint_metadata')
            if ck_meta is not None:
                current_mode=normalize_scenario_mode(constraint_config.mode)
                ck_mode=normalize_scenario_mode(ck_meta.get('scenario_mode', 'none'))
                if current_mode != ck_mode:
                    raise ValueError(f'Scenario resume mismatch: current={current_mode}, checkpoint={ck_mode}.')
            elif normalize_scenario_mode(constraint_config.mode) != 'none':
                raise ValueError('resume_checkpoint has no constraint_metadata but current run requested constraints.')
            _assert_psd_resume_compatible(_psd_regularization_metadata_from_args(args), payload.get('psd_regularization_metadata'))
            load_state_dict_compatible(model, checkpoint_state_dict(payload), context='resume_checkpoint state_dict', strict=True)
            resume_epoch=int(payload['epoch'])
            if _is_rank0(ctx): tqdm.write(f'[model_training] resumed from {resume_checkpoint} at epoch {resume_epoch}')
        else:
            resume_epoch=0
        runtime_compile_kwargs, runtime_compile_note = _fixed_compile_kwargs_for_device(ctx.device)
        model, compile_applied, compile_policy, runtime_compile_kwargs = _maybe_compile_model(
            model, requested=compile_requested, device=ctx.device
        )
        readout_compile_applied, readout_compile_policy = _maybe_compile_readout(
            readout, requested=compile_requested, compile_kwargs=runtime_compile_kwargs
        )
        _emit_compile_startup_status(
            ctx,
            model,
            compile_requested=bool(compile_requested),
            compile_applied=bool(compile_applied),
            compile_policy=str(compile_policy),
            compile_kwargs=dict(runtime_compile_kwargs),
            compile_stance_policy=str(compile_stance_policy),
            compile_cache_policy=dict(compile_cache_policy),
            compile_threads_policy=dict(compile_threads_policy),
            readout_compile_applied=bool(readout_compile_applied),
            readout_compile_policy=str(readout_compile_policy),
            amp_mode=str(args.amp),
            amp_bf16_safe_active=bool(amp_bf16_safe_active),
            tf32_policy=dict(tf32_policy),
            drop_last_train=bool(drop_last_train),
        )
        ddp_compile_warmup_policy = _prime_ddp_compile_cache_if_needed(
            model=model,
            readout=readout,
            train_loader=train_loader,
            args=args,
            ctx=ctx,
            compile_requested=bool(compile_requested),
            amp_bf16_safe_active=bool(amp_bf16_safe_active),
        )
        if ctx.enabled:
            model=DDP(model, device_ids=[ctx.local_rank], output_device=ctx.local_rank)
        optimizer=build_optimizer(model, lr=float(args.lr))
        pca_reference_bank=_build_pca_reference_bank_if_needed(model, train_loader, args, ctx)
        if pca_reference_bank is not None:
            from src.model.psd_minibatch_regularizer import move_fixed_pca_reference_bank_to_device
            pca_reference_bank = move_fixed_pca_reference_bank_to_device(
                pca_reference_bank,
                device=ctx.device,
                dtype=torch.float32,
            )
        all_rows=[]
        for epoch in range(resume_epoch+1, int(args.epochs)+1):
            if train_sampler is not None: train_sampler.set_epoch(epoch)
            _set_branch_training_stage(model, epoch, args)
            train_metrics=train_one_epoch(model, train_loader, readout=readout, optimizer=optimizer, device=ctx.device, progress_desc=None, disable_progress=True, regularization_curve_space=str(args.signal_curve_space), regularization_curve_scale=str(args.signal_curve_scale), regularization_centering=str(args.signal_curve_centering), regularization_reducer=str(args.signal_curve_reducer), regularization_distance_metric=str(args.signal_curve_distance_metric), regularization_userbin_edges=args.regularization_userbin_edges, regularization_userbin_reducer=str(args.signal_curve_userbin_reducer), lambda_psd_rep_input=float(args.lambda_psd_rep_input), lambda_psd_rep_adjacent=float(args.lambda_psd_rep_adjacent), lambda_psd_pca_input=float(args.lambda_psd_pca_input), lambda_psd_pca_adjacent=float(args.lambda_psd_pca_adjacent), lambda_branch_ortho=float(args.lambda_branch_ortho), lambda_branch_s=float(args.lambda_branch_s), psd_reg_output_family=str(args.psd_reg_output_family), pca_reference_bank=pca_reference_bank, signal_window=str(args.signal_window), amp_bf16_safe=amp_bf16_safe_active)
            train_metrics=_reduce_train_metrics_ddp(train_metrics, ctx)
            if epoch not in analysis_checkpoint_epochs: continue
            _ddp_barrier(ctx)
            eval_model=_model_for_evaluation(model)
            local_test_metrics=evaluate_one_epoch(eval_model,test_loader,readout=readout,device=ctx.device,progress_desc=None,disable_progress=True, amp_bf16_safe=amp_bf16_safe_active)
            test_metrics=_reduce_eval_metrics_ddp(local_test_metrics, ctx)
            if _is_rank0(ctx):
                tqdm.write(f'[model_training] epoch={epoch} train_loss={train_metrics.loss:.6f} test_loss={test_metrics.loss:.6f} train_acc={float(train_metrics.accuracy):.6f} test_acc={float(test_metrics.accuracy):.6f}')
                model_meta=getattr(_unwrap_model(model),'model_metadata',lambda:{})()
                psd_meta = _psd_regularization_metadata_from_args(args)
                from src.model.training import regularizer_compile_metadata as _regularizer_compile_metadata
                psd_meta.update(_regularizer_compile_metadata())
                flat_pca_bank = _flatten_pca_reference_bank(pca_reference_bank)
                psd_meta['pca_reference_bank_layer_keys'] = sorted(list(flat_pca_bank.keys()))
                psd_meta['pca_reference_bank_dims'] = {k:int(v.dim) for k,v in flat_pca_bank.items()}
                psd_meta['pca_reference_bank_device'] = {k:str(v.x_basis.device) for k,v in flat_pca_bank.items()}
                psd_meta['pca_reference_bank_metadata'] = {k:dict(getattr(v, 'metadata', {}) or {}) for k,v in flat_pca_bank.items()}
                try:
                    from src.neurons._common import sequence_backend_name as _sequence_backend_name_runtime, sequence_buffer_mode as _sequence_buffer_mode_runtime
                    checkpoint_sequence_backend = _sequence_backend_name_runtime()
                    checkpoint_sequence_buffer_mode = _sequence_buffer_mode_runtime()
                except Exception:
                    checkpoint_sequence_backend = 'compiled_sequence_prealloc'
                    checkpoint_sequence_buffer_mode = 'prealloc'
                training_args={
                    'dataset': dataset_token, 'prep_root': str(prep_root),
                    'model_token': model_spec.canonical_token,
                    'model_spec_contract': 'explicit_structured_fields_only',
                    'neuron_type': getattr(args, 'neuron_type', None),
                    'recurrent': _parse_bool_config_value(getattr(args, 'recurrent', False), default=False),
                    'reset': getattr(args, 'reset', None),
                    'v_th': ['train' if bool(getattr(model_spec, 'trainable_threshold', False)) else 'fixed', float(getattr(model_spec, 'threshold_value', 1.0))],
                    'filter': getattr(args, 'filter', None),
                    'branch': getattr(args, 'branch', None),
                    'rf_pole_radius_constrained': getattr(args, 'rf_pole_radius_constrained', None),
                    'rf_pole_radius_max': float(getattr(args, 'rf_pole_radius_max', 0.9999)),
                    'hidden_spec': str(args.hidden_spec), 'readout_mode': canonical_readout_mode,
                    'epochs': int(args.epochs), 'batch_size': int(args.batch_size), 'lr': float(args.lr),
                    'lambda_branch_ortho': float(args.lambda_branch_ortho),
                    'lambda_branch_s': float(args.lambda_branch_s),
                    'soft_mask_epochs': int(args.soft_mask_epochs),
                    'ste_epochs': int(args.ste_epochs),
                    'harden_epoch': None if args.harden_epoch in (None, '') else int(args.harden_epoch),
                    'ddp': bool(ctx.enabled), 'ddp_world_size': int(ctx.world_size), 'gpu_index': list(getattr(args, 'gpu_index', [])),
                    'batch_size_is_global': bool(args.batch_size_is_global),
                    'compile': bool(compile_requested),
                    'compile_applied': bool(compile_applied),
                    'compile_fixed_policy': str(_FIXED_COMPILE_POLICY),
                    'readout_compile_applied': bool(readout_compile_applied),
                    'readout_compile_policy': str(readout_compile_policy),
                    'regularizer_backend': 'eager_gpu',
                    'regularizer_compile_policy': psd_meta.get('regularizer_compile_policy', 'torch.compiler.disable(recursive=True)'),
                    'amp': str(args.amp),
                    'amp_internal_policy': str(amp_mode),
                    'amp_active': 'bf16_safe' if bool(amp_bf16_safe_active) else 'off',
                    'tf32': True,
                    'drop_last_train': bool(drop_last_train),
                    'eval_batch_size': int(eval_batch_size),
                    'eval_dataset_policy': dict(eval_dataset_policy),
                    'ddp_eval_policy': 'all_rank_strided_subset_all_reduce' if bool(ctx.enabled) else 'single_process_full_eval',
                    'compile_cpu_threads': None if getattr(args, 'compile_cpu_threads', None) is None else int(args.compile_cpu_threads),
                    'compile_cache_mode': str(getattr(args, 'compile_cache_mode', 'shared')),
                    'ddp_compile_warmup': _parse_bool_config_value(getattr(args, 'ddp_compile_warmup', True), default=True),
                    'ddp_timeout_minutes': int(getattr(args, 'ddp_timeout_minutes', 120)),
                    'sequence_backend': checkpoint_sequence_backend,
                    'sequence_buffer_mode': checkpoint_sequence_buffer_mode,
                    'channels_last_policy': dict(channels_last_policy),
                    'tf32_policy': dict(tf32_policy),
                    'run_timestamp': str(run_timestamp),
                    'timestamped_output': bool(timestamped_output),
                    'checkpoint_root': str(checkpoint_root),
                    'metric_root': str(metric_root),
                }
                payload=_checkpoint_payload(epoch=epoch, model=model, model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=canonical_readout_mode, dataset_token=dataset_token, prep_root=str(prep_root), prepared_dataset_path=str(prepared_dataset_path), seed=int(args.seed), training_args=training_args, model_config=_checkpoint_model_config(model=model, args=args, bundle=bundle, model_spec=model_spec, manifest=manifest), readout_config=_checkpoint_readout_config(readout_mode=canonical_readout_mode, bundle=bundle), axis_metadata_ref={'manifest_path':str(bundle.manifest_path),'psd_axis_kind':str(bundle.psd_axis_kind),'training_view_name':str(bundle.training_view_name),'psd_view_name':str(bundle.psd_view_name)}, metric_snapshot={'train_loss':float(train_metrics.loss),'test_loss':float(test_metrics.loss), 'train_accuracy': float(train_metrics.accuracy), 'test_accuracy': float(test_metrics.accuracy)}, constraint_metadata=model_meta.get('constraint_metadata'), psd_regularization_metadata=psd_meta, compile_requested=bool(compile_requested), compile_applied=bool(compile_applied), compile_policy=str(compile_policy), compile_fixed_policy=str(_FIXED_COMPILE_POLICY), readout_compile_applied=bool(readout_compile_applied), readout_compile_policy=str(readout_compile_policy), regularizer_backend='eager_gpu', regularizer_compile_policy=psd_meta.get('regularizer_compile_policy', 'torch.compiler.disable(recursive=True)'), compile_stance_policy=str(compile_stance_policy), compile_runtime_note=runtime_compile_note, compile_threads_policy=compile_threads_policy, compile_cache_policy=compile_cache_policy, ddp_compile_warmup_policy=ddp_compile_warmup_policy, amp=str(args.amp), amp_internal_policy=str(amp_mode), amp_bf16_safe_active=bool(amp_bf16_safe_active), tf32_policy=tf32_policy, channels_last_policy=channels_last_policy, drop_last_train=bool(drop_last_train), eval_batch_size=int(eval_batch_size), eval_dataset_policy=dict(eval_dataset_policy), image_mlp_flatten_policy=dict(image_mlp_flatten_policy), ddp_eval_policy=('all_rank_strided_subset_all_reduce' if bool(ctx.enabled) else 'single_process_full_eval'), ddp_timeout_minutes=int(getattr(args, 'ddp_timeout_minutes', 120)), run_timestamp=str(run_timestamp), timestamped_output=bool(timestamped_output), checkpoint_root=str(checkpoint_root), metric_root=str(metric_root))
                _atomic_torch_save(payload, checkpoint_root / f'checkpoint_epoch_{epoch:06d}.pt')
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='train', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=canonical_readout_mode, epoch=epoch, metric='loss', value=train_metrics.loss))
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='test', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=canonical_readout_mode, epoch=epoch, metric='loss', value=test_metrics.loss))
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='train', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=canonical_readout_mode, epoch=epoch, metric='accuracy', value=float(train_metrics.accuracy), value_unit='ratio'))
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='test', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=canonical_readout_mode, epoch=epoch, metric='accuracy', value=float(test_metrics.accuracy), value_unit='ratio'))
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='train', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=canonical_readout_mode, epoch=epoch, metric='psd_regularization_total', value=train_metrics.psd_regularization_total))
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='train', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=canonical_readout_mode, epoch=epoch, metric='psd_regularization_rep_1d', value=train_metrics.psd_regularization_rep_1d))
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='train', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=canonical_readout_mode, epoch=epoch, metric='psd_regularization_pca_1d', value=train_metrics.psd_regularization_pca_1d))
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='train', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=canonical_readout_mode, epoch=epoch, metric='psd_regularization_pca_mimo', value=train_metrics.psd_regularization_pca_mimo))
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='train', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=canonical_readout_mode, epoch=epoch, metric='psd_regularization_rep_input', value=train_metrics.psd_regularization_rep_input))
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='train', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=canonical_readout_mode, epoch=epoch, metric='psd_regularization_rep_adjacent', value=train_metrics.psd_regularization_rep_adjacent))
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='train', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=canonical_readout_mode, epoch=epoch, metric='psd_regularization_pca_1d_input', value=train_metrics.psd_regularization_pca_1d_input))
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='train', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=canonical_readout_mode, epoch=epoch, metric='psd_regularization_pca_1d_adjacent', value=train_metrics.psd_regularization_pca_1d_adjacent))
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='train', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=canonical_readout_mode, epoch=epoch, metric='psd_regularization_pca_mimo_input', value=train_metrics.psd_regularization_pca_mimo_input))
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='train', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=canonical_readout_mode, epoch=epoch, metric='psd_regularization_pca_mimo_adjacent', value=train_metrics.psd_regularization_pca_mimo_adjacent))
            _ddp_barrier(ctx)
        if _is_rank0(ctx):
            metrics_path=metric_root/'training_metrics.csv'; write_common_csv(metrics_path, all_rows); _assert_clean_checkpoint_dir(checkpoint_root)
            print(json.dumps({'status':'ok','source_program':SOURCE_PROGRAM,'run_timestamp':str(run_timestamp),'timestamped_output':bool(timestamped_output),'checkpoint_root':str(checkpoint_root),'metric_root':str(metric_root),'metric_csv':str(metrics_path),'eval_batch_size':int(eval_batch_size),'eval_dataset_policy':dict(eval_dataset_policy),'ddp_eval_policy':('all_rank_strided_subset_all_reduce' if bool(ctx.enabled) else 'single_process_full_eval'),'analysis_checkpoint_epochs':analysis_checkpoint_epochs}, sort_keys=True))
        return 0
    finally:
        if ctx is not None and ctx.enabled and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

# PSD curve arguments are native to this entrypoint.
# Do not apply the optional overlay here, because it would parse configs twice.
if __name__ == '__main__':
    raise SystemExit(main())

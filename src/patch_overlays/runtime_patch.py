"""Small runtime overlays for PSD-token analysis without broad source rewrites."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from src.patch_overlays.psd_curve_config import (
    ALL_DATASET_PSD_TOKENS,
    DEFAULT_PSD_TOKEN,
    parse_psd_curve_tokens,
    resolve_userbin_edges,
    tokens_require_userbins,
    normalize_userbin_reducer,
)
from src.patch_overlays.psd_curve_ops import curve_distance, curve_rows_for_maps, representative_curve_tensor, token_distance_rows
from src.signal.psd_utils import trace_tensor_to_channel_major_maps, tensor_to_channel_major_maps_explicit
from src.util.paths import timestamped_output_root


def _parser_has(parser: Any, dest: str) -> bool:
    return any(getattr(action, 'dest', None) == dest for action in getattr(parser, '_actions', []))


def _add_arg(parser: Any, *flags: str, **kwargs: Any) -> None:
    dest = kwargs.get('dest')
    if dest is None:
        for flag in flags:
            if flag.startswith('--'):
                dest = flag[2:].replace('-', '_')
                break
    if dest and _parser_has(parser, str(dest)):
        return
    parser.add_argument(*flags, **kwargs)


def _split_cli_values(values: Any, *, default: Sequence[str]) -> list[str]:
    if values is None:
        return [str(v) for v in default]
    if isinstance(values, str):
        raw = [values]
    else:
        raw = list(values)
    out: list[str] = []
    for item in raw:
        for chunk in str(item).replace(',', ' ').split():
            token = chunk.strip()
            if token:
                out.append(token)
    return out or [str(v) for v in default]


def _normalize_choice_values(values: Any, *, default: Sequence[str], allowed: Sequence[str], name: str) -> tuple[str, ...]:
    allowed_set = {str(v) for v in allowed}
    normalized: list[str] = []
    seen: set[str] = set()
    for value in _split_cli_values(values, default=default):
        token = str(value).strip().lower()
        if token not in allowed_set:
            raise ValueError(f'Unsupported {name} {value!r}. Allowed: {tuple(allowed)}.')
        if token not in seen:
            normalized.append(token)
            seen.add(token)
    return tuple(normalized)


def _analysis_distance_metrics(args: Any) -> tuple[str, ...]:
    return _normalize_choice_values(
        getattr(args, 'analysis_distance_metric', None),
        default=('centered_l2',),
        allowed=('centered_l2', 'diff_l2'),
        name='analysis_distance_metric',
    )


def _signal_curve_userbin_reducers(args: Any) -> tuple[str, ...]:
    raw = _normalize_choice_values(
        getattr(args, 'signal_curve_userbin_reducer', None),
        default=('mean',),
        allowed=('mean', 'median', 'sum'),
        name='signal_curve_userbin_reducer',
    )
    return tuple(normalize_userbin_reducer(value) for value in raw)


def _analysis_curve_batches(specs: Sequence[Any], userbin_reducers: Sequence[str]) -> tuple[tuple[tuple[Any, ...], str], ...]:
    exact_specs = tuple(spec for spec in specs if getattr(spec, 'extractor', '') == 'psd_exact')
    userbin_specs = tuple(spec for spec in specs if getattr(spec, 'extractor', '') == 'psd_userbin')
    batches: list[tuple[tuple[Any, ...], str]] = []
    if exact_specs:
        batches.append((exact_specs, 'mean'))
    for reducer in userbin_reducers:
        if userbin_specs:
            batches.append((userbin_specs, str(reducer)))
    return tuple(batches)


def patch_csv_schema(g: dict[str, Any]) -> None:
    cols = g.get('CATEGORY_COLUMNS')
    if not isinstance(cols, dict):
        return

    def extend(category: str, extra: tuple[str, ...]) -> None:
        base = tuple(cols.get(category, ()))
        cols[category] = base + tuple(x for x in extra if x not in base)

    curve_extra = ('psd_token', 'userbin_reducer')
    extend('dataset_curve', curve_extra)
    extend('dataset_dispersion', curve_extra)
    extend('analysis_curve', curve_extra)
    extend('analysis_dispersion', curve_extra)
    extend('pair_distance', ('psd_token', 'userbin_reducer'))
    extend('layer_distance_profile', ('psd_token', 'userbin_reducer'))
    extend('layer_distance_trend', ('psd_token', 'userbin_reducer'))
    if 'psd_curve_distance' not in cols:
        axis_cols = tuple(g.get('AXIS_METADATA_COLUMNS', ()))
        cols['psd_curve_distance'] = (
            *axis_cols,
            'model_token', 'model_family', 'readout_mode', 'seed', 'checkpoint_path', 'checkpoint_epoch',
            'layer', 'layer_index', 'scope', 'probe_family', 'label', 'signal_kind', 'series',
            'left_psd_token', 'right_psd_token', 'distance_metric', 'value', 'value_unit',
        )
    if 'filter_distribution' not in cols:
        axis_cols = tuple(g.get('AXIS_METADATA_COLUMNS', ()))
        cols['filter_distribution'] = (
            *axis_cols,
            'model_token', 'model_family', 'readout_mode', 'seed', 'checkpoint_path', 'checkpoint_epoch',
            'layer', 'layer_index', 'distribution_scope', 'parameter_name', 'distribution_kind',
            'neuron_index', 'bin_index', 'bin_left', 'bin_right', 'bin_count', 'bin_probability',
            'bin_density', 'parameter_value', 'frequency', 'frequency_unit', 'value', 'value_unit',
        )
    g['CATEGORY_NAMES'] = frozenset(cols)


def patch_snn_builder(g: dict[str, Any]) -> None:
    if g.get('_PSD_READOUT_PATCHED'):
        return
    nn = g.get('nn')
    orig = g.get('build_snn_classifier')
    if orig is None or nn is None:
        return

    def freeze(owner: Any, name: str, label: str) -> list[str]:
        parameter = getattr(owner, name, None)
        if isinstance(parameter, nn.Parameter) and bool(parameter.requires_grad):
            parameter.requires_grad_(False)
            return [label]
        return []

    def freeze_for_readout(output_layer: Any, overrides: Mapping[str, Any] | None) -> tuple[str, ...]:
        if not overrides:
            return ()
        frozen: list[str] = []
        raw_membrane = overrides.get('emit_spike') is False and overrides.get('reset_enabled') is False
        spike_disabled = overrides.get('emit_spike') is False
        if raw_membrane:
            frozen.extend(freeze(output_layer, 'v_threshold_param', 'v_threshold_param'))
            if hasattr(output_layer, 'trainable_threshold'):
                output_layer.trainable_threshold = False
        if spike_disabled:
            for owner, prefix in ((output_layer, ''), (getattr(output_layer, 'node', None), 'node.')):
                frozen.extend(freeze(owner, 'alpha_s', f'{prefix}alpha_s'))
                frozen.extend(freeze(owner, 'alpha_l', f'{prefix}alpha_l'))
        return tuple(frozen)

    def wrapped_build_snn_classifier(*args: Any, **kwargs: Any):
        overrides = kwargs.get('output_layer_overrides')
        model = orig(*args, **kwargs)
        frozen = freeze_for_readout(getattr(model, 'output_layer', None), overrides)
        extra = getattr(model, 'extra_metadata', None)
        if isinstance(extra, dict):
            extra['output_layer_readout_overrides'] = dict(overrides or {})
            extra['output_layer_frozen_parameters_for_readout'] = list(frozen)
        return model

    g['build_snn_classifier'] = wrapped_build_snn_classifier
    g['_PSD_READOUT_PATCHED'] = True


def _env_curve_specs(default_token: str = DEFAULT_PSD_TOKEN):
    tokens = os.environ.get('PSD_REG_CURVE_TOKENS') or default_token
    return parse_psd_curve_tokens(tokens.split(), default=[default_token])


def _env_userbin_edges(required: bool) -> list[float] | None:
    edges = os.environ.get('PSD_REG_USERBIN_EDGES')
    width = os.environ.get('PSD_REG_USERBIN_WIDTH')
    count = os.environ.get('PSD_REG_USERBIN_COUNT')
    return resolve_userbin_edges(edges=edges, width=width, count=count, required=required)


def _env_signal_window(default: str = 'hann') -> str:
    return str(os.environ.get('PSD_SIGNAL_WINDOW') or default)


def _torch_curve_distance(u: torch.Tensor, v: torch.Tensor, metric: str) -> torch.Tensor:
    if u.shape != v.shape:
        raise ValueError(f'Curve shape mismatch: {tuple(u.shape)} vs {tuple(v.shape)}.')
    if metric == 'centered_l2':
        diff = (u.reshape(-1) - u.reshape(-1).mean()) - (v.reshape(-1) - v.reshape(-1).mean())
    elif metric == 'diff_l2':
        uf = u.reshape(-1)
        vf = v.reshape(-1)
        if int(uf.numel()) < 2:
            return uf.new_zeros(())
        diff = torch.diff(uf) - torch.diff(vf)
    else:
        raise ValueError(f'Unsupported curve distance metric: {metric!r}.')
    return torch.linalg.vector_norm(diff, ord=2)


def patch_training(g: dict[str, Any]) -> None:
    if g.get('_PSD_TOKEN_TRAINING_PATCHED'):
        return
    original = g.get('compute_regularization_loss')
    if original is None:
        return
    RegularizationLossParts = g['RegularizationLossParts']
    _select_hidden_trace = g['_select_hidden_trace']

    def compute_regularization_loss_tokenized(result: Any, **kwargs: Any):
        token_env = os.environ.get('PSD_REG_CURVE_TOKENS')
        if not token_env:
            return original(result, **kwargs)
        lambda1 = float(kwargs.get('lambda1', 0.0))
        lambda2 = float(kwargs.get('lambda2', 0.0))
        if lambda1 == 0.0 and lambda2 == 0.0:
            zero = result.output_record.membrane.new_zeros(())
            return RegularizationLossParts(total=zero, global_loss=zero, adjacent_loss=zero)
        specs = _env_curve_specs()
        edges = _env_userbin_edges(tokens_require_userbins(specs))
        userbin_reducer = normalize_userbin_reducer(os.environ.get('PSD_REG_USERBIN_REDUCER', 'mean'))
        metric = os.environ.get('PSD_REG_DISTANCE_METRIC', str(kwargs.get('distance_metric', 'centered_l2')))
        signal_window = str(kwargs.get('signal_window', _env_signal_window()))
        signal_name = str(kwargs.get('signal_name', 'y_mem'))
        input_maps = trace_tensor_to_channel_major_maps(result.input_record)
        hidden_maps = [trace_tensor_to_channel_major_maps(_select_hidden_trace(record, signal_name)) for record in result.hidden_records]
        if not hidden_maps:
            raise ValueError('Nonzero PSD regularization requires captured hidden layer records.')
        global_sum = input_maps.new_zeros(())
        adjacent_sum = input_maps.new_zeros(())
        for spec in specs:
            input_curve, _axis, _edges = representative_curve_tensor(input_maps, spec, userbin_edges=edges, userbin_reducer=userbin_reducer, signal_window=signal_window)
            curves = [representative_curve_tensor(maps, spec, userbin_edges=edges, userbin_reducer=userbin_reducer, signal_window=signal_window)[0] for maps in hidden_maps]
            for curve in curves:
                global_sum = global_sum + _torch_curve_distance(input_curve, curve, metric)
            prev = input_curve
            for curve in curves:
                adjacent_sum = adjacent_sum + _torch_curve_distance(prev, curve, metric)
                prev = curve
        weighted_global = lambda1 * global_sum
        weighted_adjacent = lambda2 * adjacent_sum
        return RegularizationLossParts(total=weighted_global + weighted_adjacent, global_loss=weighted_global, adjacent_loss=weighted_adjacent)

    g['compute_regularization_loss'] = compute_regularization_loss_tokenized
    g['_PSD_TOKEN_TRAINING_PATCHED'] = True


def patch_model_training(g: dict[str, Any]) -> None:
    """Compatibility hook for model_training.

    The project exposes the regularization curve controls directly in
    src.model_training. Keep this hook as a no-op so optional overlay imports do
    not add a second public argument surface.
    """
    if g.get('_PSD_TOKEN_MODEL_TRAINING_PATCHED'):
        return
    g['_PSD_TOKEN_MODEL_TRAINING_PATCHED'] = True

def patch_dataset_psd(g: dict[str, Any]) -> None:
    if g.get('_PSD_TOKEN_DATASET_PATCHED'):
        return
    orig_build = g.get('build_arg_parser')
    if orig_build is None:
        return

    def build_arg_parser_patched():
        parser = orig_build()
        _add_arg(parser, '--signal_curve_space', default='exact', choices=('exact', 'userbin'))
        _add_arg(parser, '--signal_curve_scale', default='raw', choices=('raw', 'db', 'area'))
        _add_arg(parser, '--signal_curve_userbin_edges', nargs='*', default=None)
        _add_arg(parser, '--signal_curve_userbin_reducer', default='mean', choices=('mean', 'median', 'sum'))
        _add_arg(parser, '--signal_window', default='hann', choices=('hann', 'none'))
        return parser

    def _dataset_tokens(value: Any) -> list[str]:
        if isinstance(value, (list, tuple)):
            tokens = [str(v).strip() for v in value if str(v).strip()]
        else:
            tokens = [str(value).strip()]
        if not tokens:
            raise ValueError('dataset 배열은 비어 있을 수 없습니다.')
        return tokens

    def _run_one(args: Any, *, output_root: Path) -> dict[str, str]:
        g['_seed_everything'](int(args.seed))
        device = g['_require_cuda_device'](int(args.gpu_index))
        dataset_token = str(args.dataset)
        prep_root = Path(args.prep_root).expanduser().resolve()
        bundle = g['resolve_dataset_bundle'](dataset_token, prep_root=prep_root)
        manifest_loader = g.get('_load_structured_light')
        if manifest_loader is None:
            raise RuntimeError('runtime patch requires _load_structured_light for YAML manifests.')
        manifest = manifest_loader(bundle.manifest_path)
        if not isinstance(manifest, Mapping):
            raise ValueError(f'Prepared manifest must be a mapping: {bundle.manifest_path}')
        g['_validate_axis_metadata'](manifest)
        all_specs = parse_psd_curve_tokens(ALL_DATASET_PSD_TOKENS, default=ALL_DATASET_PSD_TOKENS)
        curve_space = str(getattr(args, 'signal_curve_space', 'exact')).strip().lower()
        if curve_space not in {'exact', 'userbin'}:
            raise ValueError('signal_curve_space must be exact or userbin.')
        signal_scale = str(getattr(args, 'signal_curve_scale', 'raw')).strip().lower()
        if signal_scale not in {'raw', 'db', 'area'}:
            raise ValueError('signal_curve_scale must be raw, db, or area.')
        specs = tuple(
            spec for spec in all_specs
            if (
                ((curve_space == 'userbin' and getattr(spec, 'extractor', '') == 'psd_userbin')
                 or (curve_space == 'exact' and getattr(spec, 'extractor', '') == 'psd_exact'))
                and getattr(spec, 'scale', '') == signal_scale
            )
        )
        userbin_edges = resolve_userbin_edges(
            edges=getattr(args, 'signal_curve_userbin_edges', None),
            required=(curve_space == 'userbin'),
        )
        userbin_reducer = normalize_userbin_reducer(getattr(args, 'signal_curve_userbin_reducer', 'mean'))
        output_root.mkdir(parents=True, exist_ok=True)
        run_id = f'{dataset_token}_dataset_psd_seed{int(args.seed)}'
        common_base = {
            'source_program': g['SOURCE_PROGRAM'],
            'run_id': run_id,
            'dataset': dataset_token,
            **g['_axis_metadata_columns'](manifest, psd_axis_kind=bundle.psd_axis_kind),
        }
        manifest_rows: list[dict[str, str]] = []
        for split_name, split_dataset in (('train', bundle.train_dataset), ('test', bundle.test_dataset)):
            psd_dataset = g['dataset_for_view'](split_dataset, bundle.psd_view_name)
            views = [(f'{split_name}_full', 'full_dataset', None, psd_dataset)]
            views.extend((f'{split_name}_{fid}', family, label, subset) for fid, family, label, subset in g['_probe_subsets'](psd_dataset, split_name=split_name, seed=int(args.seed)))
            for scope, family, label, subset in views:
                loader = g['make_loader'](subset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), pin_memory=device.type == 'cuda', seed=int(args.seed))
                all_maps = []
                expected_rows, expected_time = g['_expected_rows_time'](manifest)
                for inputs, _target in loader:
                    maps = tensor_to_channel_major_maps_explicit(
                        torch.as_tensor(inputs, dtype=torch.float32),
                        psd_axis_kind=str(bundle.psd_axis_kind),
                        psd_time_axis=manifest.get('psd_time_axis'),
                        psd_flatten_rule=manifest.get('psd_flatten_rule'),
                        psd_logical_shape=manifest.get('psd_logical_shape'),
                        expected_time=expected_time,
                        expected_rows=expected_rows,
                    ).to(device=device, non_blocking=True)
                    if int(maps.shape[0]) > 0:
                        all_maps.append(maps)
                if not all_maps:
                    raise ValueError('Selected dataset PSD scope is empty.')
                maps = torch.cat(all_maps, dim=0)
                base = dict(common_base)
                base.update(scope=scope, probe_family=family, label='' if label is None else int(label), signal_kind='input')
                curve_rows, dispersion_rows, _curves = curve_rows_for_maps(common_row=g['common_row'], base=base, maps=maps, specs=specs, userbin_edges=userbin_edges, userbin_reducer=userbin_reducer, category='dataset_curve', signal_window=str(getattr(args, 'signal_window', 'hann')))
                g['_write_grouped'](output_root, curve_rows, manifest_rows=manifest_rows, manifest_base=common_base, artifact_name='dataset_curve')
                g['_write_grouped'](output_root, dispersion_rows, manifest_rows=manifest_rows, manifest_base=common_base, artifact_name='dataset_dispersion')
        manifest_path = output_root / 'dataset_psd_manifest.yaml'
        g['write_manifest_yaml'](manifest_path, manifest_rows)
        return {'dataset': dataset_token, 'output_root': str(output_root), 'manifest': str(manifest_path)}

    def main_patched(argv: Sequence[str] | None = None) -> int:
        parser = build_arg_parser_patched()
        args = g['parse_args_with_config'](parser, argv=argv, stage_key='dataset_psd')
        if int(args.batch_size) < 1:
            parser.error('--batch_size must be >= 1.')
        if int(args.num_workers) < 0:
            parser.error('--num_workers must be >= 0.')
        g['_load_runtime_dependencies']()
        base_output_root = timestamped_output_root(args.output_root, run_timestamp=getattr(args, 'run_timestamp', None), prefix=str(g.get('SOURCE_PROGRAM', 'dataset_psd')), enabled=getattr(args, 'timestamped_output', True))
        tokens = _dataset_tokens(args.dataset)
        outputs = []
        for index, token in enumerate(tokens, start=1):
            print(f'[dataset_psd] 시작 {index}/{len(tokens)} dataset={token}', flush=True)
            run_args = type('Args', (), vars(args).copy())()
            run_args.dataset = token
            run_root = base_output_root / token if len(tokens) > 1 else base_output_root
            outputs.append(_run_one(run_args, output_root=run_root))
            print(f'[dataset_psd] 완료 {index}/{len(tokens)} dataset={token}', flush=True)
        print(json.dumps({'status': 'ok', 'source_program': g['SOURCE_PROGRAM'], 'outputs': outputs}, sort_keys=True))
        return 0


    g['build_arg_parser'] = build_arg_parser_patched
    g['main'] = main_patched
    g['_PSD_TOKEN_DATASET_PATCHED'] = True


def patch_psd_analysis(g: dict[str, Any]) -> None:
    if g.get('_PSD_TOKEN_ANALYSIS_PATCHED'):
        return
    orig_build = g.get('build_arg_parser')
    if orig_build is None:
        return

    def build_arg_parser_patched():
        parser = orig_build()
        _add_arg(parser, '--analysis_distance_metric', nargs='*', default=['centered_l2'], choices=('centered_l2', 'diff_l2'))
        _add_arg(parser, '--psd_curve_tokens', nargs='*', default=None)
        _add_arg(parser, '--signal_curve_space', default='exact', choices=('exact', 'userbin'))
        _add_arg(parser, '--signal_curve_scale', default='raw', choices=('raw', 'db', 'area'))
        _add_arg(parser, '--signal_curve_userbin_edges', nargs='*', type=float, default=None)
        _add_arg(parser, '--signal_curve_userbin_reducer', nargs='*', default=['mean'], choices=('mean', 'median', 'sum'))
        _add_arg(parser, '--signal_window', default='hann', choices=('hann', 'none'))
        _add_arg(parser, '--parameter_alpha_bin_edges', nargs='*', type=float, default=None)
        _add_arg(parser, '--parameter_alpha_bin_count', type=int, default=10)
        _add_arg(parser, '--parameter_center_frequency_bin_edges', nargs='*', type=float, default=None)
        _add_arg(parser, '--parameter_center_frequency_bin_count', type=int, default=10)
        _add_arg(parser, '--parameter_damping_bin_edges', nargs='*', type=float, default=None)
        _add_arg(parser, '--parameter_damping_bin_count', type=int, default=10)
        _add_arg(parser, '--parameter_threshold_bin_edges', nargs='*', type=float, default=None)
        _add_arg(parser, '--parameter_threshold_bin_count', type=int, default=10)
        return parser

    def collect_signal_maps_with_input(*, model: Any, dataset: Any, split_name: str, seed: int, anal_batch: int, num_workers: int, device: torch.device):
        collected: dict[tuple[str, int, str, str, str, str, int | None], list[torch.Tensor]] = defaultdict(list)
        layer_index_by_name: dict[str, int] = {}
        if hasattr(model, 'iter_named_layers'):
            for idx, (name, _layer) in enumerate(model.iter_named_layers(), start=1):
                layer_index_by_name[str(name)] = idx
        with torch.inference_mode():
            for family_id, family, label, subset in g['_probe_subsets'](dataset, split_name=split_name, seed=seed):
                scope = f'{split_name}_{family_id}'
                loader = g['make_loader'](subset, batch_size=int(anal_batch), shuffle=False, num_workers=int(num_workers), pin_memory=device.type == 'cuda', seed=int(seed))
                for inputs, _target in g['tqdm'](loader, desc=f'{g["SOURCE_PROGRAM"]}:{scope}', leave=False):
                    model_inputs = g['_prepared_input_for_model'](model, inputs, device=device)
                    collected[('input', 0, 'input', 'input', scope, family, label)].append(trace_tensor_to_channel_major_maps(model_inputs).detach().cpu())
                    result = model(model_inputs, capture_hidden=True)
                    for record in list(result.hidden_records):
                        layer_name = str(record.layer_name)
                        layer_index = int(layer_index_by_name.get(layer_name, len(layer_index_by_name) + 1))
                        for signal_kind, series, maps in g['_maps_from_record'](record):
                            collected[(layer_name, layer_index, signal_kind, series, scope, family, label)].append(maps.detach().cpu())
                    output_record = result.output_record
                    output_index = int(layer_index_by_name.get('output', 999))
                    if getattr(output_record, 'layer_input', None) is not None:
                        collected[('output', output_index, 'output', 'layer_input', scope, family, label)].append(trace_tensor_to_channel_major_maps(output_record.layer_input).detach().cpu())
                    collected[('output', output_index, 'output', 'membrane', scope, family, label)].append(trace_tensor_to_channel_major_maps(output_record.membrane).detach().cpu())
                    collected[('output', output_index, 'output', 'spike', scope, family, label)].append(trace_tensor_to_channel_major_maps(output_record.spike).detach().cpu())
                    readout_mem = getattr(output_record, 'readout_mem', None)
                    if isinstance(readout_mem, torch.Tensor):
                        collected[('output', output_index, 'output', 'readout_mem', scope, family, label)].append(trace_tensor_to_channel_major_maps(readout_mem).detach().cpu())
        return {key: torch.cat(values, dim=0).contiguous() for key, values in collected.items() if values}

    def main_patched(argv: Sequence[str] | None = None) -> int:
        parser = build_arg_parser_patched()
        args = g['parse_args_with_config'](parser, argv=argv, stage_key='psd_analysis')
        if int(args.anal_batch) < 1:
            parser.error('--anal_batch must be >= 1.')
        if int(args.num_workers) < 0:
            parser.error('--num_workers must be >= 0.')
        token_values = getattr(args, 'psd_curve_tokens', None)
        if token_values:
            specs = parse_psd_curve_tokens(token_values, default=[DEFAULT_PSD_TOKEN])
        else:
            curve_space = str(getattr(args, 'signal_curve_space', 'exact')).strip().lower()
            scale = str(getattr(args, 'signal_curve_scale', 'raw')).strip().lower()
            if curve_space not in {'exact', 'userbin'}:
                raise ValueError('signal_curve_space must be exact or userbin.')
            if scale not in {'raw', 'db', 'area'}:
                raise ValueError('signal_curve_scale must be raw, db, or area.')
            specs = parse_psd_curve_tokens([f'{curve_space}_mean_raw_{scale}'], default=[DEFAULT_PSD_TOKEN])
        distance_metrics = _analysis_distance_metrics(args)
        userbin_reducers = _signal_curve_userbin_reducers(args)
        userbin_edges = resolve_userbin_edges(
            edges=getattr(args, 'signal_curve_userbin_edges', None),
            required=tokens_require_userbins(specs),
        )
        curve_batches = _analysis_curve_batches(specs, userbin_reducers)
        g['_load_runtime_dependencies']()
        output_root = timestamped_output_root(args.output_root, run_timestamp=getattr(args, 'run_timestamp', None), prefix=str(g.get('SOURCE_PROGRAM', 'psd_analysis')), enabled=getattr(args, 'timestamped_output', True))
        output_root.mkdir(parents=True, exist_ok=True)
        checkpoint_input = Path(args.checkpoint).expanduser().resolve()
        input_is_single_file = checkpoint_input.is_file()
        checkpoint_files, ordering_warnings = g['_resolve_checkpoint_files'](checkpoint_input)
        if getattr(args, 'pca_ref_epoch', None) is not None:
            ref_epoch = int(getattr(args, 'pca_ref_epoch'))
            ref_candidates = [
                p for p in checkpoint_files
                if int(g['_load_checkpoint'](p, map_location='cpu').get('epoch', -1)) == ref_epoch
            ]
            if not ref_candidates:
                raise ValueError(f'pca_ref_epoch={ref_epoch} is not present in checkpoint list.')
        device = g['_require_cuda_device'](int(args.gpu_index))
        manifest_rows: list[dict[str, str]] = []
        first_manifest_base = None
        trend_rows: list[dict[str, str]] = []
        filter_trend_rows: list[dict[str, str]] = []
        filter_trend_history: dict[tuple[str, int, str, str], list[tuple[int, float, dict[str, Any]]]] = defaultdict(list)
        for checkpoint_path in g['tqdm'](checkpoint_files, desc='psd_analysis:checkpoints', leave=False):
            payload = g['_load_checkpoint'](checkpoint_path, map_location='cpu')
            seed = int(args.seed if args.seed is not None else payload.get('seed', 0))
            g['_seed_everything'](seed)
            model, _readout, model_spec, readout_mode = g['_build_model_from_checkpoint'](payload, device=device)
            bundle = g['_resolve_bundle'](payload, cli_dataset=args.dataset, cli_prep_root=args.prep_root, model_spec=model_spec)
            manifest = g['_manifest_dict'](bundle.manifest_path)
            g['_validate_axis_metadata'](manifest, payload)
            prep_profile = str(manifest.get('prep_profile', manifest.get('psd_axis_kind', bundle.psd_axis_kind)))
            run_id = f'{bundle.dataset_name}_{model_spec.canonical_token}_{readout_mode}_analysis_seed{seed}'
            checkpoint_base = g['_checkpoint_common_base'](payload=payload, checkpoint_path=checkpoint_path, model_spec=model_spec, readout_mode=readout_mode, run_id=run_id, prep_profile=prep_profile, seed=seed)
            checkpoint_base.update(g['_axis_metadata_columns'](manifest, psd_axis_kind=bundle.psd_axis_kind))
            if first_manifest_base is None:
                first_manifest_base = dict(checkpoint_base)
            maps_by_key = {}
            for split_name, split_dataset in (('train', bundle.train_dataset), ('test', bundle.test_dataset)):
                analysis_dataset = g['dataset_for_view'](split_dataset, bundle.training_view_name)
                maps_by_key.update(collect_signal_maps_with_input(model=model, dataset=analysis_dataset, split_name=split_name, seed=seed, anal_batch=int(args.anal_batch), num_workers=int(args.num_workers), device=device))
            checkpoint_dir = output_root / (checkpoint_path.stem if input_is_single_file else f'checkpoint_epoch_{int(payload.get("epoch", len(manifest_rows))):06d}')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            layer_index_by_name: dict[str, int] = {}
            if hasattr(model, 'iter_named_layers'):
                for layer_idx, (layer_name, _layer) in enumerate(model.iter_named_layers(), start=1):
                    layer_index_by_name[str(layer_name)] = int(layer_idx)
            filter_rows, filter_snapshot = g['_filter_snapshot_rows'](common_base=checkpoint_base, model=model, layer_index_by_name=layer_index_by_name)
            filter_distribution_rows = g['_filter_distribution_rows'](common_base=checkpoint_base, model=model, layer_index_by_name=layer_index_by_name, args=args)
            for layer_name, param_map in filter_snapshot.items():
                layer_idx = 0 if str(layer_name) == 'model' else int(layer_index_by_name.get(layer_name, 999))
                for parameter, stats in param_map.items():
                    for stat_name, stat_value in stats.items():
                        filter_trend_history[(str(layer_name), int(layer_idx), str(parameter), str(stat_name))].append((int(payload.get('epoch', 0)), float(stat_value), dict(checkpoint_base)))
            family_rows: list[dict[str, str]] = []
            dispersion_rows: list[dict[str, str]] = []
            curve_distance_rows: list[dict[str, str]] = []
            layer_rows: list[dict[str, str]] = []
            curves_by_key: dict[Any, dict[str, Any]] = {}
            common_by_key: dict[Any, dict[str, Any]] = {}
            for key, maps in sorted(maps_by_key.items(), key=lambda item: (item[0][4], item[0][1], item[0][0], item[0][2], item[0][3])):
                layer_name, layer_index, signal_kind, series, scope, family, label = key
                base = dict(checkpoint_base)
                base.update(layer=layer_name, layer_index=layer_index, scope=scope, probe_family=family, label='' if label is None else int(label), signal_kind=signal_kind, series=series)
                common_by_key[key] = base
                curves = {}
                for batch_specs, batch_userbin_reducer in curve_batches:
                    c_rows, d_rows, batch_curves = curve_rows_for_maps(
                        common_row=g['common_row'],
                        base=base,
                        maps=maps.to(device),
                        specs=batch_specs,
                        userbin_edges=userbin_edges,
                        userbin_reducer=batch_userbin_reducer,
                        category='analysis_curve',
                        signal_window=str(getattr(args, 'signal_window', 'hann')),
                    )
                    family_rows.extend(c_rows)
                    dispersion_rows.extend(d_rows)
                    curves.update(batch_curves)
                curves_by_key[key] = curves
                curve_distance_rows.extend(token_distance_rows(common_row=g['common_row'], base=base, curves=curves, distance_metrics=distance_metrics))
            input_by_scope = {key[4:7]: key for key in curves_by_key if key[0] == 'input'}
            for key, curves in curves_by_key.items():
                if key[0] == 'input':
                    continue
                input_key = input_by_scope.get(key[4:7])
                if input_key is None:
                    continue
                for token in sorted(set(curves) & set(curves_by_key[input_key])):
                    if curves[token].shape != curves_by_key[input_key][token].shape:
                        continue
                    for metric in distance_metrics:
                        row = dict(common_by_key[key])
                        row.update(category='layer_distance_profile', relation_type='input_reference', comparison_index=0, comparison_label=f'input->{key[0]}', track_name=str(key[3]), source_layer='input', source_layer_index=0, source_signal_kind='input', source_series='input', target_layer=key[0], target_layer_index=key[1], target_signal_kind=key[2], target_series=key[3], psd_token=token, distance_metric=metric, value=curve_distance(curves_by_key[input_key][token], curves[token], metric), value_unit='dimensionless')
                        layer_rows.append(g['common_row'](**row))
            grouped = defaultdict(list)
            for key in curves_by_key:
                if key[0] != 'input':
                    grouped[(key[4], key[5], key[6], key[3])].append(key)
            for _group, keys in grouped.items():
                ordered = sorted(keys, key=lambda item: (int(item[1]), str(item[0])))
                for idx, (src, dst) in enumerate(zip(ordered, ordered[1:]), start=1):
                    for token in sorted(set(curves_by_key[src]) & set(curves_by_key[dst])):
                        if curves_by_key[src][token].shape != curves_by_key[dst][token].shape:
                            continue
                        for metric in distance_metrics:
                            row = dict(common_by_key[dst])
                            row.update(category='layer_distance_profile', relation_type='adjacent', comparison_index=idx, comparison_label=f'{src[0]}->{dst[0]}', track_name=str(dst[3]), source_layer=src[0], source_layer_index=src[1], source_signal_kind=src[2], source_series=src[3], target_layer=dst[0], target_layer_index=dst[1], target_signal_kind=dst[2], target_series=dst[3], psd_token=token, distance_metric=metric, value=curve_distance(curves_by_key[src][token], curves_by_key[dst][token], metric), value_unit='dimensionless')
                            layer_rows.append(g['common_row'](**row))
            g['_write_layer_rows'](checkpoint_dir, family_rows, manifest_rows=manifest_rows, manifest_base=checkpoint_base, artifact_name='analysis_curve')
            g['_write_layer_rows'](checkpoint_dir, dispersion_rows, manifest_rows=manifest_rows, manifest_base=checkpoint_base, artifact_name='analysis_dispersion')
            g['_write_layer_rows'](checkpoint_dir, curve_distance_rows, manifest_rows=manifest_rows, manifest_base=checkpoint_base, artifact_name='psd_curve_distance')
            g['_write_layer_rows'](checkpoint_dir, filter_rows, manifest_rows=manifest_rows, manifest_base=checkpoint_base, artifact_name='filter_snapshot')
            g['_write_layer_rows'](checkpoint_dir, filter_distribution_rows, manifest_rows=manifest_rows, manifest_base=checkpoint_base, artifact_name='filter_distribution')
            for relation_type in ('input_reference', 'adjacent'):
                rows = [row for row in layer_rows if row.get('relation_type') == relation_type]
                g['_write_rows_to_dir'](checkpoint_dir / 'layer_distance_profile' / relation_type, rows, manifest_rows=manifest_rows, manifest_base=checkpoint_base, artifact_name='layer_distance_profile')
                for row in rows:
                    trend = dict(row)
                    trend['category'] = 'layer_distance_trend'
                    trend_rows.append(g['common_row'](**trend))
        traces_dir = output_root / 'traces'
        traces_dir.mkdir(parents=True, exist_ok=True)
        manifest_base = first_manifest_base or {'source_program': g['SOURCE_PROGRAM'], 'dataset': str(args.dataset), 'run_id': 'analysis'}
        for relation_type in ('input_reference', 'adjacent'):
            rows = [row for row in trend_rows if row.get('relation_type') == relation_type]
            g['_write_rows_to_dir'](traces_dir / 'layer_distance_trend' / relation_type, rows, manifest_rows=manifest_rows, manifest_base=manifest_base, artifact_name='layer_distance_trend')
        for (layer_name, layer_index, parameter, stat_name), history in sorted(filter_trend_history.items()):
            for epoch, stat_value, base in sorted(history, key=lambda item: item[0]):
                kwargs = dict(base)
                value_unit = 'count' if str(stat_name) == 'count' else g['_filter_value_unit'](str(parameter))
                kwargs.update(category='filter_trend', layer=layer_name, layer_index=layer_index, checkpoint_epoch=epoch, parameter_name=parameter, statistic=stat_name, value=stat_value, value_unit=value_unit)
                filter_trend_rows.append(g['common_row'](**kwargs))
        grouped_filter_trend: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in filter_trend_rows:
            grouped_filter_trend[(row.get('layer', ''), row.get('layer_index', ''))].append(row)
        for (layer_name, layer_index), rows in sorted(grouped_filter_trend.items(), key=lambda item: (str(item[0][1]), str(item[0][0]))):
            g['_write_rows_to_dir'](traces_dir / 'filter_trend' / g['_layer_folder'](layer_name, layer_index), rows, manifest_rows=manifest_rows, manifest_base=manifest_base, artifact_name='filter_trend')
        for warning in ordering_warnings:
            manifest_rows.append(g['_manifest_row'](base=manifest_base, artifact_name='checkpoint_ordering', path=Path(args.checkpoint), status='ok', message=warning))
        manifest_path = output_root / 'analysis_manifest.yaml'
        g['write_manifest_yaml'](manifest_path, manifest_rows)
        print(json.dumps({'status': 'ok', 'source_program': g['SOURCE_PROGRAM'], 'output_root': str(output_root), 'psd_curve_tokens': [s.token for s in specs], 'signal_curve_userbin_reducers': list(userbin_reducers), 'analysis_distance_metrics': list(distance_metrics), 'checkpoints': [str(p) for p in checkpoint_files]}, sort_keys=True))
        return 0

    # Keep the canonical psd_analysis.main intact so PCA/reference-basis handling,
    # checkpoint validation order, and legacy monkeypatch-based tests remain valid.
    # The base parser already exposes the analysis/userbin knobs; this overlay only
    # guards against older source revisions that lacked those parser options.
    g['build_arg_parser'] = build_arg_parser_patched
    g['_collect_signal_maps_with_input'] = collect_signal_maps_with_input
    g['_PSD_TOKEN_ANALYSIS_PATCHED'] = 'parser_only'

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

import argparse, json, os, tempfile, shutil
from dataclasses import dataclass
from typing import Any, Sequence
from src.model.constraints import ConstraintConfig, normalize_constraint_mode
from src.util.config_cli import parse_args_with_config
from src.util.csv_schema import common_row, write_common_csv
from src.util.cli_common import parse_bool_token

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

def _load_runtime_dependencies() -> None:
    global torch, tqdm, _seed_everything, make_loader, resolve_dataset_bundle, select_training_view_for_model
    global ModelSpec, canonicalize_model_token, build_optimizer, evaluate_one_epoch, train_one_epoch, build_snn_classifier, build_readout, DistributedSampler, DDP
    import torch as _torch
    from tqdm import tqdm as _tqdm
    from torch.nn.parallel import DistributedDataParallel as _DDP
    from torch.utils.data.distributed import DistributedSampler as _DistributedSampler
    from src.data.registry import make_loader as _make_loader, resolve_dataset_bundle as _resolve_dataset_bundle, select_training_view_for_model as _select_training_view_for_model
    from src.model.model_registry import ModelSpec as _ModelSpec, canonicalize_model_token as _canonicalize_model_token
    from src.model.training import build_optimizer as _build_optimizer, evaluate_one_epoch as _evaluate_one_epoch, train_one_epoch as _train_one_epoch
    from src.model.snn_builder import build_snn_classifier as _build_snn_classifier
    from src.readout.readout import build_readout as _build_readout
    from src.util.random import seed_everything as _runtime_seed_everything
    torch=_torch; tqdm=_tqdm; DDP=_DDP; DistributedSampler=_DistributedSampler
    make_loader=_make_loader; resolve_dataset_bundle=_resolve_dataset_bundle; select_training_view_for_model=_select_training_view_for_model
    ModelSpec=_ModelSpec; canonicalize_model_token=_canonicalize_model_token; build_optimizer=_build_optimizer; evaluate_one_epoch=_evaluate_one_epoch; train_one_epoch=_train_one_epoch
    build_snn_classifier=_build_snn_classifier; build_readout=_build_readout; _seed_everything=_runtime_seed_everything

def _load_json_light(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding='utf-8'))
def _parse_bool_config_value(value: Any, *, default: bool) -> bool: return parse_bool_token(value, default=default)
def _ddp_requested(args: argparse.Namespace) -> bool: return _parse_bool_config_value(getattr(args,'ddp',False), default=False)
def _unwrap_model(model: Any) -> Any: return getattr(model,'module',model)
def _is_rank0(ctx: DDPContext) -> bool: return bool(ctx.is_rank0)

def build_arg_parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description='Supervised model training entrypoint for selected checkpoint production.')
    p.add_argument('--dataset', required=True); p.add_argument('--prep_root', required=True); p.add_argument('--model', required=True); p.add_argument('--hidden_spec', required=True)
    p.add_argument('--readout_mode', required=True, choices=('temporal_membrane','final_membrane','first_spike','max_rate','spikegru_max_over_time'))
    p.add_argument('--epochs', required=True, type=int); p.add_argument('--batch_size', required=True, type=int); p.add_argument('--lr', required=True, type=float)
    p.add_argument('--num_workers', type=int, default=0); p.add_argument('--seed', required=True, type=int); p.add_argument('--gpu_index', type=int, default=0)
    p.add_argument('--ddp', default='false', help='2-GPU DDP 사용 여부(true/false).')
    p.add_argument('--ddp_world_size', type=int, default=2, help='DDP world size. 현재 2만 허용.')
    p.add_argument('--batch_size_is_global', default='true', help='batch_size를 global batch로 해석할지 여부. 현재 true만 허용.')
    p.add_argument('--regularization_lambda1', default=0.0, type=float); p.add_argument('--regularization_lambda2', default=0.0, type=float)
    p.add_argument('--regularization_signal', default='y_mem', choices=('y_mem','y_spike')); p.add_argument('--regularization_curve_space', default='exact', choices=('exact',))
    p.add_argument('--regularization_curve_scale', default='raw', choices=('raw','db')); p.add_argument('--regularization_centering', default='raw', choices=('raw','centered'))
    p.add_argument('--regularization_reducer', default='mean', choices=('mean','median')); p.add_argument('--regularization_distance_metric', default='centered_l2', choices=('centered_l2','diff_l2'))
    p.add_argument('--lambda_psd_rep_1d', default=0.0, type=float); p.add_argument('--lambda_psd_pca_1d', default=0.0, type=float); p.add_argument('--lambda_psd_pca_mimo', default=0.0, type=float)
    p.add_argument('--psd_reg_variant', default='raw', choices=('raw','centered')); p.add_argument('--psd_reg_output_family', default='spike', choices=('spike','membrane'))
    p.add_argument('--psd_reg_curve_scale', default='raw', choices=('raw','db')); p.add_argument('--psd_reg_relation', default='adjacent', choices=('adjacent','input'))
    p.add_argument('--pca_dim_per_layer', nargs='*', default=None)
    p.add_argument('--constraint_mode', default='none', choices=('none','clip','structure','clipstructure','clip_structure'))
    p.add_argument('--w_clip_edges', nargs='*', default=None)
    p.add_argument('--alpha_clip_edges', nargs='*', default=None)
    p.add_argument('--band_neuron_ends', nargs='*', default=None)
    p.add_argument('--tear', type=int, default=1)
    p.add_argument('--rf_frequency_clip_edges', nargs='*', default=None)
    p.add_argument('--lif_alpha_clip_edges', nargs='*', default=None)
    p.add_argument('--constraint_tear', type=int, default=None)
    p.add_argument('--anal_epoch_list', nargs='*', default=None); p.add_argument('--checkpoint_root', required=True); p.add_argument('--metric_root', required=True)
    p.add_argument('--output_root', default=None); p.add_argument('--v_th', type=float, default=1.0); p.add_argument('--resume_checkpoint', default=None); p.add_argument('--config', default=None)
    return p

def _normalize_anal_epoch_list(values: Sequence[str] | None, *, epochs: int) -> list[int]:
    if values is None or len(values)==0: values=[str(epochs)]
    out=sorted({int(v) for v in values if str(v).strip()!=''})
    if not out: out=[epochs]
    for e in out:
        if e<1 or e>epochs: raise ValueError('anal_epoch_list 범위 오류')
    return out

def _resolve_prepared_paths(dataset: str, prep_root: str) -> tuple[Path, Path]:
    root=Path(str(prep_root)).expanduser().resolve(); d=(root/dataset).resolve(); m=d/'manifest.json'
    if not m.exists(): raise FileNotFoundError(f'manifest 없음: {m}')
    return root,d

def _strict_prepare_checkpoint_dir(checkpoint_root: Path) -> None:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    if any(checkpoint_root.iterdir()): raise ValueError('Checkpoint root는 비어 있어야 합니다.')

def _assert_clean_checkpoint_dir(checkpoint_root: Path) -> None:
    for c in checkpoint_root.iterdir():
        if c.is_dir() or c.suffix!='.pt': raise ValueError(f'체크포인트 디렉터리 규칙 위반: {c}')

def _resolve_device(gpu_index:int):
    if torch.cuda.is_available():
        if gpu_index<0 or gpu_index>=torch.cuda.device_count(): raise ValueError('gpu_index 오류')
        torch.cuda.set_device(gpu_index); return torch.device(f'cuda:{gpu_index}')
    return torch.device('cpu')

def _build_ddp_context(args: argparse.Namespace) -> DDPContext:
    enabled=_ddp_requested(args)
    if not enabled:
        device=_resolve_device(int(args.gpu_index)); return DDPContext(False,0,0,1,device,True)
    if int(args.ddp_world_size)!=2: raise ValueError('DDP는 ddp_world_size=2만 허용합니다.')
    if not _parse_bool_config_value(args.batch_size_is_global, default=True): raise ValueError('DDP에서는 batch_size_is_global=true만 허용합니다.')
    for key in ('LOCAL_RANK','RANK','WORLD_SIZE'):
        if key not in os.environ: raise ValueError(f'DDP 실행 환경변수 누락: {key}. torchrun으로 실행하세요.')
    local_rank=int(os.environ['LOCAL_RANK']); rank=int(os.environ['RANK']); world_size=int(os.environ['WORLD_SIZE'])
    if world_size!=2: raise ValueError('DDP WORLD_SIZE는 2여야 합니다.')
    if local_rank not in (0,1): raise ValueError('LOCAL_RANK는 0 또는 1이어야 합니다.')
    if not torch.cuda.is_available() or torch.cuda.device_count()<2: raise ValueError('DDP에는 CUDA 2개 이상이 필요합니다.')
    try:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend='nccl')
    except Exception as exc:
        raise RuntimeError(f'DDP 초기화 실패: {exc}') from exc
    return DDPContext(True,rank,local_rank,world_size,torch.device(f'cuda:{local_rank}'),rank==0)

def _resolve_effective_batch_size(args: argparse.Namespace, ctx: DDPContext) -> int:
    g=int(args.batch_size)
    if not ctx.enabled: return g
    if g%2!=0: raise ValueError('DDP 사용 시 batch_size는 2로 나누어 떨어져야 합니다.')
    b=g//2
    if b<1: raise ValueError('DDP per-rank batch_size는 1 이상이어야 합니다.')
    return b

def _reduce_train_metrics_ddp(metrics: Any, ctx: DDPContext) -> Any:
    if not ctx.enabled: return metrics
    vals=torch.tensor([
        metrics.loss*metrics.total, metrics.task_loss*metrics.total, metrics.regularization_loss*metrics.total,
        metrics.regularization_global_loss*metrics.total, metrics.regularization_adjacent_loss*metrics.total,
        metrics.psd_regularization_total*metrics.total, metrics.psd_regularization_rep_1d*metrics.total,
        metrics.psd_regularization_pca_1d*metrics.total, metrics.psd_regularization_pca_mimo*metrics.total,
        float(metrics.correct), float(metrics.total)
    ], device=ctx.device, dtype=torch.float64)
    torch.distributed.all_reduce(vals, op=torch.distributed.ReduceOp.SUM)
    total=max(1.0,float(vals[10].item()))
    from src.model.training import TrainEpochMetrics
    return TrainEpochMetrics(loss=float(vals[0]/total), task_loss=float(vals[1]/total), regularization_loss=float(vals[2]/total), regularization_global_loss=float(vals[3]/total), regularization_adjacent_loss=float(vals[4]/total), psd_regularization_total=float(vals[5]/total), psd_regularization_rep_1d=float(vals[6]/total), psd_regularization_pca_1d=float(vals[7]/total), psd_regularization_pca_mimo=float(vals[8]/total), correct=int(vals[9].item()), total=int(vals[10].item()), accuracy=float(vals[9]/total))

def _pca_psd_regularization_requested(args: argparse.Namespace) -> bool:
    return float(getattr(args,'lambda_psd_pca_1d',0.0))!=0.0 or float(getattr(args,'lambda_psd_pca_mimo',0.0))!=0.0

def _psd_regularization_metadata_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        'lambda_psd_rep_1d': float(getattr(args,'lambda_psd_rep_1d',0.0)),
        'lambda_psd_pca_1d': float(getattr(args,'lambda_psd_pca_1d',0.0)),
        'lambda_psd_pca_mimo': float(getattr(args,'lambda_psd_pca_mimo',0.0)),
        'psd_reg_variant': str(getattr(args,'psd_reg_variant','raw')),
        'psd_reg_output_family': str(getattr(args,'psd_reg_output_family','spike')),
        'psd_reg_curve_scale': str(getattr(args,'psd_reg_curve_scale','raw')),
        'psd_reg_relation': str(getattr(args,'psd_reg_relation','adjacent')),
        'pca_dim_per_layer': (_parse_pca_dim_per_layer(getattr(args,'pca_dim_per_layer',None)) or []),
        'ddp_policy': 'rank0_broadcast',
        'pca_reference_bank_policy': 'rank0_build_once_per_run',
    }

def _assert_psd_resume_compatible(current_psd_meta: dict[str, Any], ck_psd_meta: dict[str, Any] | None) -> None:
    psd_enabled_now = any(float(current_psd_meta[k]) != 0.0 for k in ('lambda_psd_rep_1d','lambda_psd_pca_1d','lambda_psd_pca_mimo'))
    if ck_psd_meta is not None:
        compare_keys = ('lambda_psd_rep_1d','lambda_psd_pca_1d','lambda_psd_pca_mimo','psd_reg_variant','psd_reg_output_family','psd_reg_curve_scale','psd_reg_relation')
        for key in compare_keys:
            if str(ck_psd_meta.get(key)) != str(current_psd_meta.get(key)):
                raise ValueError(f'PSD regularization resume mismatch at {key}: current={current_psd_meta.get(key)} checkpoint={ck_psd_meta.get(key)}')
    elif psd_enabled_now:
        raise ValueError('resume_checkpoint has no psd_regularization_metadata but current run requested PSD regularization.')

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
    def _merge(name: str, alias: str):
        base = getattr(args, name, None)
        ali = getattr(args, alias, None)
        if base is not None and ali is not None and list(base) != list(ali):
            raise ValueError(f'Both {name} and {alias} were provided with different values.')
        return base if base is not None else ali
    w_edges = _merge('w_clip_edges', 'rf_frequency_clip_edges')
    a_edges = _merge('alpha_clip_edges', 'lif_alpha_clip_edges')
    tear = getattr(args, 'tear', 1)
    if getattr(args, 'constraint_tear', None) is not None and int(tear) != 1 and int(tear) != int(getattr(args, 'constraint_tear')):
        raise ValueError('Both tear and constraint_tear were provided with different values.')
    if getattr(args, 'constraint_tear', None) is not None:
        tear = int(getattr(args, 'constraint_tear'))
    return ConstraintConfig(
        mode=normalize_constraint_mode(getattr(args, 'constraint_mode', 'none')),
        w_clip_edges=None if w_edges is None else tuple(float(v) for v in w_edges),
        alpha_clip_edges=None if a_edges is None else tuple(float(v) for v in a_edges),
        band_neuron_ends=None if getattr(args, 'band_neuron_ends', None) is None else tuple(str(v) for v in getattr(args, 'band_neuron_ends')),
        tear=int(tear),
    )

def _cpu_pca_reference_bank(bank: dict[str, Any]) -> dict[str, Any]:
    from src.model.psd_minibatch_regularizer import FixedPCALayerReference
    out={}
    for name, ref in bank.items():
        out[str(name)]=FixedPCALayerReference(
            layer_name=str(ref.layer_name),
            layer_index=(None if getattr(ref,'layer_index',None) is None else int(ref.layer_index)),
            dim=int(ref.dim),
            x_basis=ref.x_basis.detach().cpu(),
            x_centroid=ref.x_centroid.detach().cpu(),
            y_basis=ref.y_basis.detach().cpu(),
            y_centroid=ref.y_centroid.detach().cpu(),
            output_family=str(getattr(ref,'output_family','spike')),
            basis_id=getattr(ref,'basis_id',None),
            metadata=dict(getattr(ref,'metadata',{}) or {}),
        )
    return out

def _build_pca_reference_bank_if_needed(model: Any, train_loader: Any, args: argparse.Namespace, ctx: DDPContext) -> dict[str, Any] | None:
    if not _pca_psd_regularization_requested(args): return None
    obj=[None]
    if _is_rank0(ctx):
        from src.model.training import _move_inputs_to_device, _reset_stateful_model
        from src.model.psd_minibatch_regularizer import compute_fixed_pca_reference_bank
        iterator=iter(train_loader)
        try:
            inputs,_target=next(iterator)
        except StopIteration as exc:
            raise RuntimeError('Cannot build PCA PSD reference bank from an empty train_loader.') from exc
        base_model=_unwrap_model(model)
        was_training=bool(base_model.training)
        base_model.eval()
        try:
            with torch.inference_mode():
                device_inputs=_move_inputs_to_device(base_model, inputs, device=ctx.device)
                _reset_stateful_model(base_model)
                result=base_model(device_inputs, capture_hidden=True)
                _reset_stateful_model(base_model)
                bank=compute_fixed_pca_reference_bank(
                    device_inputs,
                    list(result.hidden_records),
                    str(args.psd_reg_output_family),
                    _parse_pca_dim_per_layer(args.pca_dim_per_layer), relation=str(getattr(args,'psd_reg_relation','adjacent')),
                )
            obj[0]=_cpu_pca_reference_bank(bank)
            if not obj[0]:
                raise RuntimeError('PCA PSD regularization requires at least one hidden layer reference.')
        finally:
            if was_training: base_model.train()
    if ctx.enabled:
        torch.distributed.broadcast_object_list(obj, src=0)
    return obj[0]

def _checkpoint_payload(**kwargs):
    model=kwargs.pop('model')
    return {**kwargs,'schema_version':CHECKPOINT_SCHEMA_VERSION,'state_dict':_unwrap_model(model).state_dict()}

def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); temp_dir=Path(tempfile.mkdtemp(prefix='checkpoint_write_', dir=str(path.parent.parent))); temp_path=temp_dir/path.name
    try: torch.save(payload,temp_path); os.replace(temp_path,path)
    finally: shutil.rmtree(temp_dir, ignore_errors=True)

def main(argv: Sequence[str] | None = None) -> int:
    parser=build_arg_parser(); args=parse_args_with_config(parser, argv=argv, stage_key='model_training')
    _load_runtime_dependencies(); ctx=None
    try:
        ddp=_ddp_requested(args); args.ddp=ddp; args.batch_size_is_global=_parse_bool_config_value(args.batch_size_is_global, default=True)
        if int(args.batch_size)<1: parser.error('--batch_size must be >= 1.')
        anal_epochs=_normalize_anal_epoch_list(args.anal_epoch_list, epochs=int(args.epochs))
        dataset_token=str(args.dataset); prep_root,prepared_dataset_path=_resolve_prepared_paths(dataset_token,args.prep_root)
        checkpoint_root=Path(args.checkpoint_root).expanduser().resolve(); metric_root=Path(args.metric_root).expanduser().resolve()
        resume_checkpoint=None if args.resume_checkpoint is None else Path(args.resume_checkpoint).expanduser().resolve()
        ctx=_build_ddp_context(args)
        if _is_rank0(ctx):
            if resume_checkpoint is None: _strict_prepare_checkpoint_dir(checkpoint_root)
            else: checkpoint_root.mkdir(parents=True, exist_ok=True); _assert_clean_checkpoint_dir(checkpoint_root)
            metric_root.mkdir(parents=True, exist_ok=True)
        if ctx.enabled: torch.distributed.barrier()
        _seed_everything(int(args.seed))
        model_spec=canonicalize_model_token(args.model); bundle=select_training_view_for_model(resolve_dataset_bundle(dataset_token, prep_root=prep_root), model_family=model_spec.family)
        constraint_config=_normalize_constraint_args(args)
        per_rank_batch=_resolve_effective_batch_size(args, ctx)
        train_sampler=None
        if ctx.enabled:
            train_sampler=DistributedSampler(bundle.train_dataset, num_replicas=2, rank=ctx.rank, shuffle=True, seed=int(args.seed), drop_last=False)
        train_loader=make_loader(bundle.train_dataset,batch_size=per_rank_batch,shuffle=True,num_workers=int(args.num_workers),pin_memory=ctx.device.type=='cuda',seed=int(args.seed),sampler=train_sampler)
        test_loader=None
        if (not ctx.enabled) or _is_rank0(ctx):
            test_loader=make_loader(bundle.test_dataset,batch_size=int(args.batch_size),shuffle=False,num_workers=int(args.num_workers),pin_memory=ctx.device.type=='cuda',seed=int(args.seed))
        readout=build_readout(str(args.readout_mode), num_classes=bundle.num_classes, sequence_length=bundle.sequence_length, device=ctx.device)
        model=build_snn_classifier(model_token=model_spec,input_dim=bundle.input_dim,sequence_length=bundle.sequence_length,num_classes=bundle.num_classes,input_shape=None,hidden_sizes=tuple(int(v) for v in bundle.default_hidden_sizes),arch_spec=str(args.hidden_spec).strip(),output_layer_overrides=readout.output_layer_overrides(),v_th=float(args.v_th),constraint_config=constraint_config).to(ctx.device)
        readout.to(ctx.device)
        if resume_checkpoint is not None:
            payload=torch.load(resume_checkpoint, map_location=ctx.device, weights_only=False)
            ck_meta=payload.get('constraint_metadata')
            if ck_meta is not None:
                current_mode=normalize_constraint_mode(constraint_config.mode)
                ck_mode=normalize_constraint_mode(ck_meta.get('constraint_mode', 'none'))
                if current_mode != ck_mode:
                    raise ValueError(f'Constraint resume mismatch: current={current_mode}, checkpoint={ck_mode}.')
            elif normalize_constraint_mode(constraint_config.mode) != 'none':
                raise ValueError('resume_checkpoint has no constraint_metadata but current run requested constraints.')
            _assert_psd_resume_compatible(_psd_regularization_metadata_from_args(args), payload.get('psd_regularization_metadata'))
            model.load_state_dict(payload['state_dict'])
            resume_epoch=int(payload['epoch'])
            if _is_rank0(ctx): tqdm.write(f'[model_training] resumed from {resume_checkpoint} at epoch {resume_epoch}')
        else:
            resume_epoch=0
        if ctx.enabled:
            model=DDP(model, device_ids=[ctx.local_rank], output_device=ctx.local_rank)
        optimizer=build_optimizer(model, lr=float(args.lr))
        pca_reference_bank=_build_pca_reference_bank_if_needed(model, train_loader, args, ctx)
        all_rows=[]
        for epoch in range(resume_epoch+1, int(args.epochs)+1):
            if train_sampler is not None: train_sampler.set_epoch(epoch)
            train_metrics=train_one_epoch(model, train_loader, readout=readout, optimizer=optimizer, device=ctx.device, progress_desc=(f'train epoch {epoch}' if _is_rank0(ctx) else None), disable_progress=(ctx.enabled and (not _is_rank0(ctx))), regularization_lambda1=float(args.regularization_lambda1), regularization_lambda2=float(args.regularization_lambda2), regularization_signal=str(args.regularization_signal), regularization_curve_space=str(args.regularization_curve_space), regularization_curve_scale=str(args.regularization_curve_scale), regularization_centering=str(args.regularization_centering), regularization_reducer=str(args.regularization_reducer), regularization_distance_metric=str(args.regularization_distance_metric), lambda_psd_rep_1d=float(args.lambda_psd_rep_1d), lambda_psd_pca_1d=float(args.lambda_psd_pca_1d), lambda_psd_pca_mimo=float(args.lambda_psd_pca_mimo), psd_reg_variant=str(args.psd_reg_variant), psd_reg_output_family=str(args.psd_reg_output_family), psd_reg_curve_scale=str(args.psd_reg_curve_scale), psd_reg_relation=str(args.psd_reg_relation), pca_reference_bank=pca_reference_bank)
            train_metrics=_reduce_train_metrics_ddp(train_metrics, ctx)
            if epoch not in anal_epochs: continue
            if ctx.enabled: torch.distributed.barrier()
            if _is_rank0(ctx):
                eval_model=_unwrap_model(model)
                test_metrics=evaluate_one_epoch(eval_model,test_loader,readout=readout,device=ctx.device,progress_desc=f'test epoch {epoch}',disable_progress=False)
                tqdm.write(f'[model_training] epoch={epoch} train_loss={train_metrics.loss:.6f} test_loss={test_metrics.loss:.6f} train_acc={float(train_metrics.accuracy):.6f} test_acc={float(test_metrics.accuracy):.6f}')
                model_meta=getattr(_unwrap_model(model),'model_metadata',lambda:{})()
                psd_meta = _psd_regularization_metadata_from_args(args)
                psd_meta['pca_reference_bank_layer_keys'] = sorted(list((pca_reference_bank or {}).keys()))
                psd_meta['pca_reference_bank_dims'] = {k:int(v.dim) for k,v in (pca_reference_bank or {}).items()}
                payload=_checkpoint_payload(epoch=epoch, model=model, model_token=model_spec.canonical_token, readout_mode=str(args.readout_mode), dataset_token=dataset_token, prep_root=str(prep_root), prepared_dataset_path=str(prepared_dataset_path), seed=int(args.seed), training_args={'ddp':bool(args.ddp),'ddp_world_size':int(args.ddp_world_size),'batch_size_is_global':bool(args.batch_size_is_global),'batch_size':int(args.batch_size)}, metric_snapshot={'train_loss':float(train_metrics.loss),'test_loss':float(test_metrics.loss), 'train_accuracy': float(train_metrics.accuracy), 'test_accuracy': float(test_metrics.accuracy)}, constraint_metadata=model_meta.get('constraint_metadata'), psd_regularization_metadata=psd_meta)
                _atomic_torch_save(payload, checkpoint_root / f'checkpoint_epoch_{epoch:06d}.pt')
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='train', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=str(args.readout_mode), epoch=epoch, metric='loss', value=train_metrics.loss))
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='test', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=str(args.readout_mode), epoch=epoch, metric='loss', value=test_metrics.loss))
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='train', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=str(args.readout_mode), epoch=epoch, metric='accuracy', value=float(train_metrics.accuracy), value_unit='ratio'))
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='test', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=str(args.readout_mode), epoch=epoch, metric='accuracy', value=float(test_metrics.accuracy), value_unit='ratio'))
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='train', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=str(args.readout_mode), epoch=epoch, metric='psd_regularization_total', value=train_metrics.psd_regularization_total))
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='train', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=str(args.readout_mode), epoch=epoch, metric='psd_regularization_rep_1d', value=train_metrics.psd_regularization_rep_1d))
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='train', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=str(args.readout_mode), epoch=epoch, metric='psd_regularization_pca_1d', value=train_metrics.psd_regularization_pca_1d))
                all_rows.append(common_row(category='training_metric', source_program=SOURCE_PROGRAM, run_id='run', dataset=dataset_token, scope='train', seed=int(args.seed), model_token=model_spec.canonical_token, model_family=str(model_spec.family), readout_mode=str(args.readout_mode), epoch=epoch, metric='psd_regularization_pca_mimo', value=train_metrics.psd_regularization_pca_mimo))
            if ctx.enabled: torch.distributed.barrier()
        if _is_rank0(ctx):
            metrics_path=metric_root/'training_metrics.csv'; write_common_csv(metrics_path, all_rows); _assert_clean_checkpoint_dir(checkpoint_root)
            print(json.dumps({'status':'ok','source_program':SOURCE_PROGRAM,'checkpoint_root':str(checkpoint_root),'metric_csv':str(metrics_path),'selected_epochs':anal_epochs}, sort_keys=True))
        return 0
    finally:
        if ctx is not None and ctx.enabled and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

try:
    from src.patch_overlays.runtime_patch import patch_model_training as _patch_model_training
    _patch_model_training(globals())
except Exception:
    pass

if __name__ == '__main__':
    raise SystemExit(main())

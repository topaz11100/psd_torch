from __future__ import annotations
import argparse
import torch

from psd_snn.config.specs import load_experiment_config
from psd_snn.analysis.signal.runner import SignalAnalysisRunner, SignalMapRecord
from psd_snn.analysis.signal_map.emitter import bt_to_srt
from psd_snn.artifacts.writers import SummaryWriter
from psd_snn.analysis.common import CheckpointRef, load_checkpoint_bundle, ProbeRequest, build_probe_batches
from psd_snn.analysis.common.model_restore import restore_model_from_bundle
from psd_snn.analysis.trace.adapter import TraceAdapter
from psd_snn.artifacts.trace_writer import TraceArtifactWriter
from psd_snn.analysis.signal.pca_basis_store import PCABasisKey, PCAFitRequest


def _synthetic_records() -> list[SignalMapRecord]:
    maps = torch.randn(4, 8, 64)
    return [SignalMapRecord(maps=maps, metadata={'run_id':'synthetic','checkpoint_epoch':None,'split':'synthetic','scope':'synthetic','probe_family':'label_single','series':'spike','layer_name':'synthetic'})]


def _collect_records(bundle, model, args, cfg, split='test', probe_family='balanced_global', scope_override=None, reference_meta=None):
    x = torch.randn(16, 32, cfg.model.topology.input_dim)
    y = torch.tensor([i % cfg.model.topology.output_dim for i in range(16)])
    dataset = {'test_inputs': x, 'test_labels': y.tolist(), 'train_inputs': x.clone(), 'train_labels': y.tolist()}
    batches = build_probe_batches(dataset, ProbeRequest(split=split, probe_family=probe_family, sample_count=args.sample_count, seed=0, exclusion_family=args.exclusion_family, exclusion_sample_count=args.exclusion_sample_count, exclusion_seed=args.exclusion_seed), batch_size=args.batch_size)
    ta = TraceAdapter(model)
    out: list[SignalMapRecord] = []
    raw_traces=[]
    for b in batches:
        _, traces = ta.run_with_trace(b.inputs, probe_family=b.probe_family, label='na')
        for tr in traces:
            if tr.tensor is None or tr.series != 'spike':
                continue
            maps = bt_to_srt(tr.tensor)
            md = {'run_id': args.run_id, 'checkpoint_epoch': bundle.checkpoint_epoch, 'split': b.split, 'scope': scope_override or b.scope, 'probe_family': b.probe_family, 'sample_indices': b.sample_indices, 'labels': b.labels, 'layer_name': tr.layer_name, 'layer_index': tr.layer_index, 'signal_kind': tr.signal_kind, 'series': tr.series, 'sample_count': len(b.sample_indices), 'time_length': int(maps.shape[-1]), 'row_count': int(maps.shape[1]), 'source_layout': 'B,T,*', 'probe_manifest_id': b.probe_manifest_id, 'exclusion_family': (b.probe_metadata or {}).get('exclusion_family'), 'exclusion_scope': (b.probe_metadata or {}).get('exclusion_scope')}
            if reference_meta:
                md.update(reference_meta)
            out.append(SignalMapRecord(maps=maps, metadata=md))
            raw_traces.append(tr)
    return out, raw_traces


def _checkpoint_records(args, cfg, runner: SignalAnalysisRunner) -> list[SignalMapRecord]:
    target_bundle = load_checkpoint_bundle(CheckpointRef(path=args.checkpoint))
    target_restored = restore_model_from_bundle(target_bundle, device=args.device)
    if target_restored.restore_status != 'ok' or target_restored.model is None:
        raise RuntimeError(f"restore failed: {target_restored.reason}")

    if cfg.signal_analysis.psd.representative.method == 'pca' and cfg.signal_analysis.psd.representative.pca.basis_mode == 'fixed_reference':
        ref_path = args.reference_checkpoint or cfg.signal_analysis.psd.representative.pca.reference_scope
        if not (args.reference_checkpoint or getattr(cfg.signal_analysis.psd.representative.pca, 'reference_checkpoint', None) is not None):
            raise RuntimeError('pca_reference_checkpoint missing')
        if args.reference_checkpoint:
            ref_bundle = load_checkpoint_bundle(CheckpointRef(path=args.reference_checkpoint))
        else:
            ref_bundle = load_checkpoint_bundle(CheckpointRef(path=args.checkpoint))
        ref_restored = restore_model_from_bundle(ref_bundle, device=args.device)
        if ref_restored.restore_status != 'ok' or ref_restored.model is None:
            raise RuntimeError(f"reference restore failed: {ref_restored.reason}")
        pca = cfg.signal_analysis.psd.representative.pca
        ref_records,_ = _collect_records(ref_bundle, ref_restored.model, args, cfg, split=(pca.reference_split or 'test'), probe_family=(pca.reference_probe_family or args.probe_family), scope_override=pca.reference_scope)
        for rec in ref_records:
            maps = rec.maps
            key = PCABasisKey(reference_checkpoint_epoch=ref_bundle.checkpoint_epoch, reference_checkpoint_id=None, reference_split=rec.metadata['split'], reference_scope=rec.metadata['scope'], reference_probe_family=rec.metadata['probe_family'], layer_index=int(rec.metadata['layer_index']), layer_name=str(rec.metadata['layer_name']), signal_kind=str(rec.metadata['signal_kind']), series=str(rec.metadata['series']), n_components=pca.n_components, row_count=int(maps.shape[1]), centering=bool(pca.center), sign_convention=pca.sign_convention)
            basis = runner.pca_basis_store.fit(PCAFitRequest(maps=maps, key=key, created_from=rec.metadata))
            runner.pca_basis_store.save_tensor_artifact(basis)

    target_records, target_traces = _collect_records(target_bundle, target_restored.model, args, cfg, split='test', probe_family=args.probe_family)
    if cfg.signal_analysis.trace_save.enabled:
        tw = TraceArtifactWriter(cfg.signal_analysis.artifact.output_dir)
        tw.write_records(target_traces, chunk_size=cfg.signal_analysis.trace_save.chunk_size)
        tw.write_manifest()
    return target_records


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--mode', choices=['synthetic', 'checkpoint'], default='synthetic')
    ap.add_argument('--checkpoint')
    ap.add_argument('--reference_checkpoint')
    ap.add_argument('--probe_family', default='balanced_global')
    ap.add_argument('--exclusion_family', default=None)
    ap.add_argument('--exclusion_sample_count', type=int, default=None)
    ap.add_argument('--exclusion_seed', type=int, default=None)
    ap.add_argument('--sample_count', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--run_id', default='run_signal')
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    runner = SignalAnalysisRunner(cfg.signal_analysis)
    records = _synthetic_records() if args.mode == 'synthetic' else _checkpoint_records(args, cfg, runner)
    runner.update_signal_maps(records)
    SummaryWriter(cfg.signal_analysis.artifact.output_dir).write_results(runner.finalize())


if __name__ == '__main__':
    main()

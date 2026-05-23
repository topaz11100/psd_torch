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


def _records_checkpoint(args, cfg):
    bundle = load_checkpoint_bundle(CheckpointRef(path=args.checkpoint))
    restored = restore_model_from_bundle(bundle, device=args.device)
    if restored.restore_status != 'ok' or restored.model is None:
        raise RuntimeError(f"restore failed: {restored.reason}")
    x = torch.randn(16, 32, cfg.model.topology.input_dim)
    y = torch.tensor([i % cfg.model.topology.output_dim for i in range(16)])
    dataset = {'test_inputs': x, 'test_labels': y.tolist()}
    batches = build_probe_batches(dataset, ProbeRequest(split='test', probe_family=args.probe_family, sample_count=args.sample_count, seed=0, exclusion_family=args.exclusion_family, exclusion_sample_count=args.exclusion_sample_count, exclusion_seed=args.exclusion_seed), batch_size=args.batch_size)
    out=[]; ta=TraceAdapter(restored.model)
    for b in batches:
        _, traces = ta.run_with_trace(b.inputs, probe_family=b.probe_family, label='na')
        for tr in traces:
            if tr.series != 'spike':
                continue
            out.append(SignalMapRecord(bt_to_srt(tr.tensor), {'run_id':args.run_id,'checkpoint_epoch':bundle.checkpoint_epoch,'split':b.split,'scope':b.scope,'probe_family':b.probe_family,'probe_manifest_id':b.probe_manifest_id,'exclusion_family':(b.probe_metadata or {}).get('exclusion_family'),'exclusion_scope':(b.probe_metadata or {}).get('exclusion_scope'),'series':tr.series,'layer_name':tr.layer_name}))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--mode', choices=['synthetic', 'checkpoint'], default='synthetic')
    ap.add_argument('--checkpoint')
    ap.add_argument('--probe_family', default='balanced_global')
    ap.add_argument('--exclusion_family', default=None)
    ap.add_argument('--exclusion_sample_count', type=int, default=None)
    ap.add_argument('--exclusion_seed', type=int, default=None)
    ap.add_argument('--sample_count', type=int, default=8)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--run_id', default='run_fft2d')
    args = ap.parse_args(argv)
    cfg = load_experiment_config(args.config)
    if args.mode == 'synthetic':
        records = [SignalMapRecord(torch.randn(4, 8, 64), {'run_id':'synthetic','split':'synthetic','scope':'synthetic','probe_family':'label_single','series':'spike','layer_name':'synthetic'})]
    else:
        records = _records_checkpoint(args, cfg)
    runner = SignalAnalysisRunner(cfg.signal_analysis)
    runner.run_fft2d(records)
    SummaryWriter(cfg.signal_analysis.artifact.output_dir).write_results(runner.finalize())


if __name__ == '__main__':
    main()

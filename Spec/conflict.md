# Known Conflicts and Resolutions

## Resolved: old root entrypoints vs current package CLI

Current resolution: use `src/psd_snn/cli` and `examples/bash`. Old root launchers are removed from the current runnable layer.

## Resolved: stale CSV category documents vs current artifact types

Current resolution: current artifact contracts are documented in `Spec/theory/10_artifacts_distance_and_manifests.md` and `Spec/implementation/05_artifact_writer_reader_plotting.md`.

## Resolved: legacy theory text vs current implementation

Current resolution: old theory was used as reference only. PSD-first analysis, dense SNN separation, LIF/RF dynamics, readout separation, and fixed-topology separation were retained. Removed experiment-family text was not restored as current spec.

## Open future work

- Real dataset integration.
- Large-scale launcher packaging.
- Paper-style plotting refinement.
- Deeper fidelity work for paper model families beyond current smoke coverage.

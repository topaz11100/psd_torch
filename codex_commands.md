# Codex implementation command

## Read first

1. Read `refactor_spec.md` completely. Treat it as the authoritative implementation specification.
2. Read `old/README.md` to understand which archived source tree contains each legacy feature.
3. Inspect the active source tree under the repository root as the implementation target.
4. Extract legacy archives only into temporary reference directories. Do not copy old modules wholesale into the active tree.

Suggested temporary extraction commands:

```bash
mkdir -p .ref/clip_structure .ref/pca_crossspec .ref/distributed_set
python - <<'PY'
from pathlib import Path
from zipfile import ZipFile
pairs = [
    ('old/clip_structure_reference.zip', '.ref/clip_structure'),
    ('old/pca_crossspec_2dfft_reference.zip', '.ref/pca_crossspec'),
    ('old/distributed_set_reference.zip', '.ref/distributed_set'),
]
for src, dst in pairs:
    dst_path = Path(dst)
    dst_path.mkdir(parents=True, exist_ok=True)
    with ZipFile(src) as zf:
        zf.extractall(dst_path)
PY
```

## Direct implementation instruction

Refactor this project according to `refactor_spec.md`. This is a large-scale rewrite; current code is a reference implementation, not a constraint on the new architecture. Preserve scientific behavior only where the specification requires it.

Implement the project around these hard boundaries:

```text
training / model construction / trace emission / signal-map conversion / spectral analysis / dynamics statistics / artifact writing / plotting / config launching
```

Do not implement the refactor as a set of small patches to the existing `psd_analysis.py`, `model_training.py`, and `snn_builder.py`. Split responsibilities into explicit packages and keep CLI entrypoints thin.

## Required source references

Use these archives only as references:

- `old/clip_structure_reference.zip`
  - recover the old `clip`, `structure`, and `clipstructure` behavior
  - do not keep old model tokens such as `lif_structclip` as the primary design
  - port the behavior as `ScenarioSpec` / `TopologyConstraintSpec`

- `old/pca_crossspec_2dfft_reference.zip`
  - recover PCA representative signal methods
  - recover fixed PCA reference-basis logic
  - recover matrix / 2D FFT analysis ideas
  - adapt them to the new signal-analysis runner and spectral-axis policy

- `old/distributed_set_reference.zip`
  - recover distribution-aware probe-set construction
  - rename/recast old distribution logic as `distributed_set`

The root source tree is the current refactor target.

## Non-negotiable behavior

- Delete `reinterpretation` functionality entirely.
- Separate PSD/signal analysis from dynamics statistics. Parameter statistics and internal-state statistics belong to the same `analyze_dynamics` execution unit.
- Use SpikingJelly as the execution framework for all spiking models, including Origin-derived models. Preserve each neuron model's equations through custom SpikingJelly-style nodes/blocks.
- MLP is a topology, not a neuron family. MLP blocks must support LIF, IF, RF, TC-LIF, TS-LIF, DH-SNN, D-RF, SpikeGRU-like cells when implemented.
- MLP recurrent mode is true SRNN recurrence: only the previous output spike is fed back through a recurrent weight matrix. Do not add `membrane_post` or arbitrary source selection as recurrent input.
- Fixed paper models such as VGG, ResNet, S4/SSM, GRU, and spike-transformer should remain fixed-topology models. Their trace interface must be standardized, not their internal topology.
- Restore `clip`, `structure`, and `clipstructure` scenarios as formal scenario/topology constraints.
- Restore `distributed_set` as a top-level probe family.
- Rename old `same_label` to `label_set`.
- Use `label_single` as a top-level probe family for all signal-analysis experiments.
- Add `final_if` and `final_mem` readout modes.
- Add time-domain spike trace artifact saving. Do not save full traces as CSV.
- PSD accumulation streams over probe/sample batches only. Do not stream over the time axis for full-sequence PSD.
- dB conversion is performed only at finalize/write time after raw-power accumulation and bin reduction.
- Signal analysis must choose exactly one spectral axis policy per run: `exact` or `userbin`.
- Distances are computed only in the selected spectral axis space.
- Keep only `centered_l2` and `diff_l2` distance metrics.
- 2D FFT `userbin` bins frequency axes after the 2D FFT result is computed. It must not pre-pool raw row/time traces.
- 2D FFT `userbin_axes` names must be explicit: `time_frequency`, `row_frequency`, `both_frequency_axes`.
- `row_frequency` and `both_frequency_axes` require explicit row-axis semantics; unordered MLP neuron index must error or strong-warn.
- `userbin` bin representative is configurable as `mean` or `median`, applied on raw power before dB conversion.
- Formula text in markdown tables must escape vertical bars as `\|`.
- Avoid version labels in new artifact names and specs. Do not use names such as `v2`, `csv_v2`, `schema_version`, or “previous/new version” language in user-facing docs or generated markdown.

## Suggested package layout

Implement using this shape unless there is a strong reason to improve it:

```text
src/psd_snn/
  cli/
  core/
  config/
  data/
  models/
    topology/
    fixed/
    mlp/
    cells/
    readout/
    trace/
  training/
  analysis/
    probe/
    trace/
    signal_map/
    spectral/
    dynamics/
    distance/
  artifacts/
  plotting/
```

Existing entrypoints may remain as compatibility wrappers, but they should call the new modules.

## Implementation phases

### Phase 1 — inventory and cleanup

- List current entrypoints and module dependencies.
- Remove or quarantine `src/reinterpretation` and related bash entrypoints.
- Add a short repository `README.md` explaining the new config-based workflow.

### Phase 2 — config and factories

- Implement config dataclasses/loaders for dataset, model, training, readout, probe, trace-save, spectral analysis, dynamics analysis, and artifact output.
- Implement model factory separation:
  - `TopologySpec`
  - `CellSpec`
  - `ScenarioSpec`
  - `ReadoutSpec`
  - `TraceSpec`
- Keep MLP topology configurable and paper backbones fixed.

### Phase 3 — SpikingJelly custom cells/blocks

- Implement LIF, IF, RF first.
- Implement hard/soft reset for IF/LIF.
- Implement RF reset modes as explicit RF-only enum.
- Implement fixed/trainable threshold for LIF/IF/RF.
- Implement exact-ZOH RF where specified.
- Implement trace output for `input_current`, `membrane_pre`, `decision`, `spike`, `membrane_post`; RF must expose real/imag or equivalent state traces where applicable.

### Phase 4 — trace and signal-map standardization

- Implement `LayerTraceRecord` for raw `B,T,*` traces.
- Implement `SignalMapRecord` for analysis `S,R,T` maps.
- Implement trace adapters for current project models and SpikingJelly models.
- Implement time-domain spike trace writer using chunked tensor artifacts, not raw CSV.

### Phase 5 — probe sets

- Implement top-level probe families:
  - `balanced_global`
  - `distributed_set`
  - `label_set`
  - `label_single`
- Persist probe manifests with class counts, quotas, seeds, selected indices, and exclusion logic.

### Phase 6 — spectral analysis

- Implement independent signal-analysis runner.
- Implement PSD, element PSD, and 2D FFT as independent analysis functions.
- Implement representative methods:
  - `mean`
  - `median`
  - `element_psd`
  - `pca`
- PCA representative mode must support configurable number of components and fixed reference basis.
- Implement `exact` and `userbin` spectral-axis policies for PSD and 2D FFT.
- Implement `userbin` bin reducer `mean`/`median`.
- Implement `centered_l2` and `diff_l2` only.

### Phase 7 — dynamics analysis

- Implement `analyze_dynamics` as the combined runner for parameter statistics and internal-state statistics.
- Include LIF alpha/tau, RF frequency/damping/decay-radius, thresholds, reset parameters, spike rate, membrane statistics, RF state energy/phase where available.

### Phase 8 — artifacts and plotting

- Implement artifact writers for:
  - scalar/stat rows
  - spectral curves
  - spectral matrix tensor or wide matrix + axis sidecar
  - trace manifests
  - time-domain spike chunks
- Keep CSV for summaries/manifests, not full trace tensors.
- Plotting must read artifacts; it must not recompute model traces.

### Phase 9 — tests

Add tests covering:

- config parsing and validation
- SRNN recurrence uses previous spike only
- SpikingJelly reset between batches
- trace shapes for MLP and fixed backbones
- `B,T,*` to `S,R,T` signal-map conversion
- PSD streaming over probe batches matches full-batch calculation
- dB conversion occurs only at finalize
- exact/userbin axis policy is exclusive
- 2D FFT userbin bins frequency axes after FFT
- empty user bins are rejected
- `label_set`, `label_single`, `distributed_set` deterministic sampling
- artifact writers produce stable schema and manifests

## Validation commands to run when implementation is complete

Adapt exact commands to the final package names if needed:

```bash
python -m pytest tests -q
python -m psd_snn.cli.train --config configs/examples/smnist_mlp_lif.yaml
python -m psd_snn.cli.analyze_signal --config configs/examples/smnist_signal_psd.yaml
python -m psd_snn.cli.analyze_dynamics --config configs/examples/smnist_dynamics.yaml
python -m psd_snn.cli.plot --config configs/examples/smnist_plot.yaml
```

## Reporting requirement

When finished, provide:

1. A short architecture summary.
2. A list of deleted/renamed legacy entrypoints.
3. A list of implemented probe families.
4. A list of implemented analysis methods.
5. A list of test commands run and their results.
6. Any remaining gaps, explicitly separated from completed functionality.

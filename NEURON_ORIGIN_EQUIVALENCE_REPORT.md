# Author-code neuron equivalence audit

## Scope

Checked the project-exposed neuron/model families that have checked-in author code under `Origin/`:

- `tc_lif`, `tc_lif_R` against `Origin/neuron_model/TC-LIF/*/spiking_neuron/TCLIF.py`
- `ts_lif`, `ts_lif_R` against `Origin/neuron_model/TS-LIF/SeqSNN/network/snn/TSLIF.py`
- `dh_snn_{B}`, `dh_snn_R_{B}` against `Origin/neuron_model/DH-SNN/*/SNN_layers/*`
- `d_rf_{B}` against `Origin/neuron_model/D-RF/models/layers.py::BiRFModel`
- `spikegru` against `Origin/spikegru/SpikGRU+imagemodel/layers.py::GRUlayer`
- `spikingssm` neuron core against `Origin/state_space_sd4/src/models/spike/neuron.py` through the origin `SpikingSSM` adapter

Project-owned `IF`, `LIF`, `RF`, `CNN2DLIF`, and `CNN2DRF` are not author-code wrappers and were excluded from author-code equivalence classification.

## Summary table

| Family | Forward dynamics | State/update equivalence | Gradient/surrogate equivalence | Classification |
|---|---:|---:|---:|---|
| TC-LIF | yes | yes | not exact by default | forward-equivalent, gradient-surrogate variant |
| TS-LIF | yes | yes | not exact by default | forward-equivalent, gradient-surrogate variant |
| DH-SNN dense | yes, under official `apply_mask()` protocol | yes for branch >= 2 zero-state protocol | not exact by default | protocol-forward-equivalent, gradient-surrogate variant |
| DH-SNN recurrent | yes, under official `apply_mask()` protocol | yes for branch >= 2 zero-state protocol | not exact by default | protocol-forward-equivalent, gradient-surrogate variant |
| D-RF/BiRF | yes for default threshold 1.0 and identity width adapter | yes | yes for origin `ZIF` path | thin-wrapper equivalent under default threshold |
| SpikeGRU vanilla single-gate | yes after alpha reparameterization mapping | yes | not exact by default | forward-equivalent subset, gradient/parameterization variant |
| SpikingSSM neuron core | origin source is loaded directly | origin source is loaded directly | origin source is loaded directly | author-core delegated; adapters outside origin scope |

## Important non-equivalence notes

1. TC-LIF, TS-LIF, DH-SNN, and SpikeGRU match the author-code forward recurrence when the same hard-spike forward convention is used. They do not reproduce the exact author surrogate-gradient functions in the project hot path. The project uses `surrogate_spike` / compile-safe STE machinery for compileability.

2. DH-SNN equivalence is with the official training protocol, not with a bare newly constructed author layer before `apply_mask()`. The author scripts call `apply_mask()` before/after optimization steps. The project multiplies by `mask` inside the effective linear operation, so the forward function equals the author layer after `apply_mask()`.

3. DH-SNN branch `1` has an author-code initialization artifact: `d_input` is random when `branch == 1`, while the project initializes recurrent/dendritic state to zero. For normal DH use with branch `>= 2`, the checked zero-state forward is equivalent.

4. D-RF is a thin wrapper around author `BiRFModel.forward`. It is exact when `input_size == output_size`, `v_threshold == 1.0`, and `emit_spike=True`. Non-default project thresholding or input adapter width projection is a project extension.

5. SpikeGRU implements the vanilla single-gate author `GRUlayer` only. Author variants `twogates=True`, `ternact=True`, convolutional frontends, and bidirectional/application-specific frontends are not implemented as equivalent project paths.

6. Spikformer uses author network source but its neuron primitive is external `spikingjelly.clock_driven.neuron.MultiStepLIFNode`, not a checked-in author-defined neuron model. If dependency stubs are active, runtime is smoke-test only and not paper-exact.

## Verification added

Added `tests/test_author_neuron_forward_equivalence.py` with direct forward/state comparisons against checked-in author code for:

- TC-LIF node update
- TS-LIF node update
- DH-SNN dense and recurrent updates under `apply_mask()` protocol
- D-RF `BiRFModel` spike path
- SpikeGRU vanilla single-gate update after alpha mapping

The direct checks use deterministic random tensors and assert max absolute difference within `1e-6`.

# Paper-Reference Compliance Check

## Scope

This pass checked code paths that explicitly advertise a paper/reference-origin contract in metadata, checked-in origin code, or paper folders:

- Neuron models: TC-LIF, TS-LIF, DH-SNN, D-RF, plus the standard LIF/IF/RF project layers where used as baseline families.
- Network models: SpikeGRU, Spikformer, SpikingSSM, VGG11-style SNN topology, SEW-ResNet18-style topology.
- Loss/readout paths: released First-spike timing loss, SpikeGRU max-over-time CE readout metadata, project PSD minibatch regularizer scope.

The check used the checked-in `Origin/` source and `paper/` files as the local reference of record.

## Fixes applied

### 1. SpikeGRU candidate recurrent bias

**Finding:** `Origin/spikegru/SpikGRU+imagemodel/layers.py::GRUlayer` defines the recurrent candidate path as:

```python
self.ui = nn.Linear(hidden_size, hidden_size, bias=True)
```

The project wrapper had:

```python
self.hidden_to_candidate = nn.Linear(hidden_size, hidden_size, bias=False)
```

This made the single-gate SpikGRU cell not exactly match the checked-in origin equation.

**Fix:** `src/model/snn_builder.py::SpikGRUCellBlock` now uses `bias=True` for `hidden_to_candidate`. Metadata now records:

```text
candidate_recurrent_bias = True
origin_formula_contract = tempZ=sigmoid(wz(x)+uz(prev_spike)); tempcurrent=alpha*tempcurrent+wi(x)+ui(prev_spike); temp=tempZ*temp+(1-tempZ)*tempcurrent-v_th*prev_spike
```

A formula-level unit test was added to verify the project cell matches the origin single-gate update.

### 2. SpikingSSM checked-in origin import path

**Finding:** `Origin/state_space_sd4/models/spike/ss4d.py` imports released modules through absolute names such as:

```python
from src.models.nn import DropoutNd
from src.models.spike.neuron import registry
from src.models.spike.surrogate import piecewise_quadratic_surrogate
from src.models.sequence.kernels.ssm import SSMKernelDiag
```

The current project package is also named `src`, and the checked-in origin snapshot does not include all of those released package files. Therefore the `spikingssm` model token could fail before constructing the author source.

**Fix:** `src/model/author_adapter_state_space.py` now installs narrow import shims before loading the author `ss4d.py` source:

- checked-in origin `neuron.py` and `surrogate.py` are loaded under the released import names;
- `DropoutNd` is provided with the expected API and time-tied dropout behavior for `[B,C,L]` tensors;
- `SSMKernelDiag` is import-stubbed and raises loudly if a future config requests `trainable_B=True`, because that official file is not included in the checked-in origin snapshot.

The author `ss4d.py` source itself is not edited. A smoke test now verifies `spikingssm` build/forward and metadata exposure.

### 3. Spikformer fallback dependency disclosure

**Finding:** the adapter correctly instantiates `Origin/spikformer/cifar10dvs/model.py::Spikformer` with the checked-in CIFAR10-DVS 2-256 profile. However, if optional author dependencies such as SpikingJelly/timm are not installed, the project uses fallback stubs so the model can still be imported for smoke testing.

Those fallback stubs are useful for CI, but should not be represented as paper-exact runtime behavior.

**Fix:** Spikformer metadata now distinguishes:

```text
dependency_backend = author_dependencies
structure_variation = none
paper_definition_compliance = author_source_with_real_dependencies
```

from:

```text
dependency_backend = fallback_stubs
structure_variation = dependency_stub_runtime_smoke_only
paper_definition_compliance = author_source_imported_with_dependency_stubs; runtime is not paper-exact
```

A test now asserts this distinction.

## Confirmed implementations

### TC-LIF

The project `TCLIFLayer` matches the checked-in origin `TCLIFNode` update pattern:

```text
v1_pre = v1 - decay0 * v2 + input
v2_pre = v2 + decay1 * v1_pre
spike = surrogate(v2_pre - threshold)
v1_next = v1_pre - gamma * spike
v2_next = v2_pre - threshold * spike
```

A formula-level unit test was added.

### TS-LIF

The project `TSLIFLayer` matches the checked-in origin `TSLIFNode` two-compartment update pattern:

```text
v1_pre = decay0 * v1 + decay1 * input - yy * v2
v2_pre = decay2 * v2 + decay3 * input - kk * v1_pre
s_s = surrogate(v2_pre - threshold)
s_l = surrogate(v1_pre - threshold)
spike = alpha_s * s_s + alpha_l * s_l
v1_next = v1_pre - gamma * s_l
v2_next = v2_pre - threshold * s_s
```

A formula-level unit test was added.

### DH-SNN

The project `DHSNNLayer` preserves the checked-in DH-SNN branch mask contract:

- dense DH-SNN mask rows have `padded_input / branch` active connections;
- recurrent DH-SNN mask rows have `(input_dim + output_dim + pad) / branch` active connections;
- branch-level active connection density is `1 / branch`.

This matches the DH-SNN definition where increasing the branch count reduces the active connectivity per dendritic branch. Tests cover branch counts 1, 2, 4, and 8 for the dense case and 2, 4, and 8 for the recurrent case.

### D-RF

The D-RF path remains a wrapper around the checked-in origin `BiRFModel`:

```text
Origin/neuron_model/D-RF/models/layers.py::BiRFModel
```

The wrapper preserves the origin convolutional RFFT/IRFFT resonate-and-fire core and only adapts project input/output sizing when necessary.

### First-spike timing loss

The first-spike path continues to load the released modules:

```text
Origin/readout/first_spike/superspike/src/time_encoding.py
Origin/readout/first_spike/utils/loss.py
```

The compatibility patch only repairs the released `LossFn` superclass initialization so it behaves as a valid `nn.Module`; it does not change the released formulas. A test now verifies the loaded loss class comes from the origin module.

### VGG / SEW-ResNet topology references

The checked project policy is topology-only reuse for these reference networks:

- VGG11-style schedule: `[64, M, 128, M, 256, 256, M, 512, 512, M, 512, 512, M]`.
- ResNet18-style block counts: `[2, 2, 2, 2]`.
- SEW merge default: `ADD`.

This remains consistent with the project intent: reference topology is reused, while dataset front-end, neuron type, and analysis traces are project-native.

## Clarified non-paper-exact or project-defined paths

### PSD minibatch regularizer

The PSD regularizer is a project-defined spectral regularization/analysis objective. It is not marked as an exact reproduction of a single paper loss. No formula change was applied in this pass.

### SpikeGRU readout loss

The project records SpikeGRU as using max-over-time cross entropy. That is kept as the declared project profile rather than claiming that the full lip-reading front-end or every ablation from the SpikeGRU papers is included.

### Spikformer fallback mode

When optional author dependencies are absent, fallback stubs are CI/runtime smoke support only. Paper-exact Spikformer behavior requires the real author dependencies.

## Validation

Commands executed:

```bash
python -m py_compile \
  src/model/snn_builder.py \
  src/model/author_adapter_spikformer.py \
  src/model/author_adapter_state_space.py \
  src/model/first_spike_loss.py \
  src/model/_origin_first_spike.py \
  src/neurons/TC_LIF_neuron.py \
  src/neurons/TS_LIF_neuron.py \
  src/neurons/DH_SNN_neuron.py \
  src/neurons/D_RF_neuron.py \
  tests/test_paper_reference_contracts.py

pytest -q
```

Result:

```text
175 passed, 2 skipped, 8 warnings
```

GPU training parity and paper-level reproduction metrics were not run in this container. This pass verifies source-level contracts, formulas, model construction, and smoke forward paths.

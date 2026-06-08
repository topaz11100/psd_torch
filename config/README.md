# Clean configuration templates

`config/` intentionally contains only blank, commented templates.  The parser now accepts nested YAML and flattens leaf keys to existing CLI argument names, so arguments can be grouped by role without changing the command-line contract.

Rules:

- Fill only the leaf values you need; keep comments as the argument contract.
- 모델은 monolithic token으로 주지 않는다. `model.neuron_type`, `model.branch`, `model.reset`, `model.v_th`를 각각 명시한다. 예: `neuron_type: my_d_rf`, `branch: 8`, `reset: hard`, `v_th: ["train", 1.0]`.
- `v_th` is always represented as `[mode, initial_value]`, e.g. `["fixed", 1.0]` or `["train", 1.0]`. Do not encode reset/threshold/branch inside `neuron_type`; use separate `neuron_type`, `branch`, `reset`, and `v_th` fields.
- `model_training`에서 `gpu_index`는 배열이다. `[0]`은 단일 프로세스 cuda:0, `[0, 1]`은 선택한 GPU 목록으로 자동 `torchrun` DDP를 실행한다. 별도 DDP config/script는 없다.
- `scenario_mode: clip|structure|clipstructure` is valid only for vanilla IF/LIF/RF families. Proposed `my_*` branch logic is controlled under `regularization.proposed_branch`.
- These templates are not runnable as-is because every experiment-specific value is blank by design.

## Direct discrete RF dynamics

RF-family experiments are now described in the discrete domain.  For vanilla dense RF, the subthreshold resonator is parameterized as

```text
z[t+1] = a z[t] + I[t+1]
a = rho * exp(j * phi)
```

`rho` is the pole radius per sample and `phi` is the pole angle in radians per sample.  The model-level settings are:

```yaml
model_training:
  model:
    discrete_dynamics:
      rf_pole_radius_constrained: true   # true => 0 <= rho < rf_pole_radius_max
      rf_pole_radius_max: 0.9999         # must be < 1 when constrained
```

Set `rf_pole_radius_constrained: false` to allow positive radii above one for finite-horizon resonant-amplifier experiments.  In that mode, RF filter statistics report `stability_excess = max(rho - 1, 0)` so unstable/amplifying poles are visible in analysis CSVs.

## Soma reset and threshold options

`reset` and `v_th` share one explicit-field contract across vanilla IF/LIF/RF and proposed `my_*` models. Encoded strings such as `my_d_rf_8_hard_train` are intentionally rejected. For `my_dh_snn`, `my_d_rf`, and `my_r_dh_snn`, these options are **soma-local**: `soft` subtracts the soma threshold after a soma spike, `hard` zeroes the soma membrane/recorded soma state, and `none` disables soma reset or reset-history updates. Dendritic branch states and branch-count masks are not reset by these options.

Use explicit fields only; `neuron_type: my_d_rf_8_hard_train` and similar monolithic tokens are not valid. `v_th` must stay array-shaped: `['fixed', 1.0]` creates a fixed soma threshold and `['train', 1.0]` creates one trainable positive threshold per soma neuron.


## Image datasets and MLP experiments

Every image-like prepared bundle must expose a flattened time-major view for dense/MLP SNN families. During training and checkpoint-based signal analysis, the loader now selects:

```text
CNN / Spikformer families -> frame-shaped view (T,C,H,W)
Dense MLP/SNN families    -> flattened view (T,F)
```

This applies to static image datasets (`mnist`, `cifar-10`, `cifar-100`) and event-frame datasets (`n-mnist`, `cifar10-dvs`, `dvs128-gesture`). The selection is validated before loaders are built, so preprocessing/view-contract errors fail at the stage boundary instead of inside a model forward.


## Image MLP flatten contract

See also `spec/Implementation/13_image_mlp_flatten_policy.md` for the stage-level preprocessing/training/analysis contract.

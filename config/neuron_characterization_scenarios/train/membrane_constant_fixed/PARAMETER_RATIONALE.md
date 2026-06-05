# Fixed Membrane-Constant Scenario Rationale

This scenario fixes membrane/filter constants to create controlled baselines for the neuron-characterization study.

## LIF soft-reset control

- `neuron_type`: `lif`
- `reset`: `soft`
- `v_th`: `["fixed", 1.0]`
- `filter`: `"0.5"`

The project LIF update uses a discrete membrane decay `alpha`. The chosen fixed value `alpha=0.5` corresponds to an Euler-style LIF time constant `tau=2` under `alpha = 1 - 1/tau`. This is aligned with common SNN implementation defaults that use threshold `1.0` and membrane time constant `2.0` for LIF-style nodes.

## RF no-reset control

- `neuron_type`: `rf`
- `reset`: `none`
- `v_th`: `["fixed", 1.0]`
- `filter`: `"0.25"`

The project RF filter value is interpreted as normalized center frequency in cycles/sample. `0.25` is a neutral mid-band reference between DC `0.0` and Nyquist `0.5`. It is intentionally fixed, so these configs isolate the effect of a non-learned resonant filter in comparison with trainable-filter RF scenarios.

## Interpretation

These configs are not intended to be the final tuned parameter set. They are controls for comparing:

1. fixed LIF membrane decay,
2. fixed RF mid-band resonance,
3. trainable threshold scenarios,
4. trainable filter scenarios,
5. PSD-regularized scenarios.

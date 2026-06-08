# Direct Discrete RF Policy

## CLI/config fields

`model_training` exposes two RF pole-radius controls:

```text
--rf_pole_radius_constrained true|false
--rf_pole_radius_max 0.9999
```

Nested YAML form:

```yaml
model_training:
  model:
    discrete_dynamics:
      rf_pole_radius_constrained: true
      rf_pole_radius_max: 0.9999
```

The config loader flattens leaf keys, so these become the CLI fields with the same names.

## Scope

The switch is applied to vanilla RF layers, including dense RF and CNN RF.  Proposed `my_D_RF` uses the same direct-discrete pole formula internally, but its default constructor keeps stable branch radii unless a future experiment exposes a separate proposed-branch radius policy.

## Checkpoint metadata

Training checkpoints store:

- `model_config.rf_pole_radius_constrained`
- `model_config.rf_pole_radius_max`
- `training_args.rf_pole_radius_constrained`
- `training_args.rf_pole_radius_max`

PSD analysis rebuilds the model using those fields.


## Proposed soma reset/threshold implementation

`my_dh_snn`, `my_d_rf`, and `my_r_dh_snn` share the vanilla soma reset/threshold contract through explicit config leaves only.  Use `model.neuron_type`, `model.branch`, `model.reset`, and `model.v_th`; encoded strings such as `my_dh_snn_8`, `my_dh_snn_8_hard_train`, or `my_d_rf_8_none_fixed` are rejected.

The implementation helper `src/neurons/_soma.py` owns the common threshold parameterization and soma-only reset rule.  The reset scope is intentionally narrow: branch recurrent states are not zeroed/subtracted after a soma spike, so branch filter statistics remain properties of the learned dendritic filter bank.

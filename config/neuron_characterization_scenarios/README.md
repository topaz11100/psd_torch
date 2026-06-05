# Neuron Characterization Scenarios

This scenario set uses the explicit model schema: `neuron_type`, `recurrent`, `reset`, `v_th`, and `filter`. The old monolithic `model` token is intentionally not used.

PSD curve settings are unified as `signal_curve_*`. User bins are strict: use `signal_curve_space: "userbin"` together with an explicit `signal_curve_userbin_edges` array. Width/count convenience keys are not used.

Training configs are under `train/`; matching checkpoint PSD analysis configs are under `psd_analysis/` with the same relative path and file stem.

## Common training defaults

All configs in this scenario tree use:

- `ddp: true`
- `ddp_world_size: 2`
- `batch_size_is_global: true`
- `amp: "on"`
- `ddp_timeout_minutes: 400`
- `epochs: 40`
- `analysis_checkpoint_epochs: [1, ..., 40]`
- `lr: 0.005`
- `compile: true`

The launcher default compile cache root is:

```text
/home/yongokhan/workspace/cache/torch_compile
```

Launcher policy: use `bash/model_training_ddp.sh <scenario train folder>`. The script expands folders serially, runs configs inside one leaf folder in parallel, and separates non-`d_rf` RF configs into their own RF-only parallel group.

## Dataset split policy

`simple/` contains only:

```text
deap, uci-har, s-mnist, shd
```

All other datasets are placed under `hard/` when the scenario group has a simple/hard split.

## Batch size policy

Global batch sizes are assigned by dataset scale:

```text
dvs128-gesture, deap: 32
cifar100-dvs/cifar-100-dvs, cifar10-dvs, cifar-100, uci-har, shd: 128
ssc, s-mnist, s-cifar10, mnist: 256
other image/event datasets: 128 unless explicitly overridden
```

`psd_analysis` configs use the same value as `anal_batch`.

## Regularization coefficient policy

Single regularization term:

```text
+0.001 or -0.001
```

Two simultaneous terms:

```text
+0.0005/+0.0005 or -0.0005/-0.0005
```

## Added: membrane_constant_fixed

`train/membrane_constant_fixed` and matching `psd_analysis/membrane_constant_fixed` add fixed membrane/filter constant baselines for `simple/` and `hard/` datasets.

# IF, LIF, and RF Cells

## Topology and cell separation

MLP is a topology. IF, LIF, and RF are cell dynamics used inside MLP hidden blocks.

## IF cell

IF uses no leak:

```text
membrane_pre = membrane_post_prev + input_current
decision = membrane_pre - threshold
```

Spike and reset policy produce `membrane_post`. Threshold may be fixed or trainable, and threshold bounds may use bounded parameterization.

## LIF cell

LIF uses a decay factor:

```text
membrane_pre = alpha * membrane_post_prev + input_current
```

`alpha` is kept in a stable range. Clip bounds use bounded sigmoid parameterization rather than optimizer-step clamping. Parameter vectors include membrane decay and threshold metadata.

## RF cell

RF keeps real and imaginary state. Its update uses exact closed-form zero-order hold dynamics, not Euler substitution. Internal update uses angular frequency in radians per step, while metadata records the configured cycle-per-step policy.

RF traces include real and imaginary pre/post states. Damping magnitude determines the discrete decay radius:

```text
decay_radius = exp(-damping_magnitude * dt)
```

## State lifecycle

Official cell creation requires the supported spiking backend. The code must not silently replace official spiking cells with unrelated fallback cells. Reset lifecycle is coordinated by the model/trace adapter rather than by hidden data mutation in analysis code.

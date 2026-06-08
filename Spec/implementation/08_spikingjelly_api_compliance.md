# SpikingJelly API compliance for neuron and training modules

## Scope

The project uses SpikingJelly's activation-based API as the preferred backend
for spike surrogate and network reset plumbing. Project-specific IF, LIF, RF,
DH-SNN, D-RF, TC-LIF, TS-LIF, and CNN spiking wrappers keep their mathematical
dynamics because several models are not one-to-one replacements for stock
SpikingJelly nodes. The implementation therefore follows a bridge policy:

1. use `spikingjelly.activation_based.surrogate.Sigmoid` for spikes when the
   dependency is installed;
2. expose SpikingJelly-style multi-step metadata (`step_mode="m"` and
   `supported_step_mode=("m",)`) on project neuron modules;
3. reset models through `spikingjelly.activation_based.functional.reset_net`
   before falling back to project-local `reset_state()` hooks;
4. retain a torch-only fallback so lightweight tests can run without the
   optional SpikingJelly package.

## IF/LIF/RF update equations

For an input current `I_t = W x_t + W_r s_{t-1}`, the dense IF layer uses

```math
u_t^- = u_{t-1} + I_t, \qquad s_t = H(u_t^- - v_{\mathrm{th}}),
```

followed by either soft reset

```math
u_t = u_t^- - v_{\mathrm{th}} s_t,
```

or hard reset

```math
u_t = u_t^- (1 - s_t).
```

LIF replaces the integrator with a learned leak constrained by the selected
group bounds:

```math
u_t^- = \alpha u_{t-1} + I_t.
```

RF keeps a two-real-channel implementation of a direct discrete complex pole
`a = rho exp(j phi)`:

```math
\begin{bmatrix}
x_t^- \\
y_t^-
\end{bmatrix}
=
\rho
\begin{bmatrix}
\cos\phi & -\sin\phi \\
\sin\phi & \cos\phi
\end{bmatrix}
\begin{bmatrix}
x_{t-1} \\
y_{t-1}
\end{bmatrix}
+
\begin{bmatrix}
1 \\
0
\end{bmatrix} I_t.
```

`rho` is the per-sample pole radius and `phi` is the rad/sample pole angle.  The
legacy names `frequency`, `damping`, `omega`, and `b` are compatibility aliases
for reporting and old scenario code; they are not the primary implementation
contract.  RF supports `soft_reset`, `hard_reset`, and `no_reset`; `no_reset`
disables the reset transform but still records spikes for readout.

## Proposed `my_*` soma contract

`my_DH_SNN`, `my_D_RF`, and `my_R_DH_SNN` now expose the same public threshold
and reset switches as project-native vanilla neurons:

```yaml
reset: soft | hard | none
v_th: [fixed | train, initial_value]
```

The implementation attaches a per-soma threshold vector to every proposed layer.
`fixed` stores it as a buffer; `train` stores a positive `softplus(raw)+eps`
parameter.  Reset is soma-local only.  DH-style EMA branch states, D-RF resonator
states, branch masks, and hardened branch-count buffers are never reset by this
switch.  This keeps branch/filter statistics invariant to readout reset policy
while making the soma spiking contract compatible with vanilla IF/LIF/RF.

## Constraint schema

`scenario_mode` is the public scenario key. `constraint_mode` is not part of the
current source contract. The allowed values are:

- `none`: no clipping and no structural mask;
- `clip`: apply parameter bounds only;
- `structure`: apply structural masks only;
- `clipstructure`: apply both parameter bounds and structural masks.

`alpha_clip_edges` and `w_clip_edges` use a three-dimensional schema:

```yaml
alpha_clip_edges:
  - - [0.10, 0.30]
    - [0.30, 0.70]
  - - [0.05, 0.25]
    - [0.25, 0.50]
```

The semantic index is `clip_edges[layer][group] = [lower, upper]`. The bounds are
explicit; the implementation does not insert implicit `0`, `0.5`, or `1.0`
boundaries.

`band_edge` defines per-layer cumulative exclusive neuron boundaries. For
example, `band_edge = [null, [5, 10]]` means the first hidden layer is split
evenly according to its group count, while the second hidden layer maps neuron
indices `0..4`, `5..9`, and `10..W-1` to groups 0, 1, and 2.

`tear` is a 1-based hidden-layer index. It controls only where structural masks
start. In `clipstructure`, clipping is still applied to all target hidden layers;
only structural masks are omitted for layers before `tear`.

## Training reset contract

Training and evaluation call the project reset helper before each forward pass.
That helper first invokes SpikingJelly's recursive `reset_net`; if SpikingJelly is
not installed or cannot reset the module, the code falls back to a local
`reset_state()` hook. This guarantees deterministic state clearing for both
SpikingJelly-backed and torch-fallback test environments.

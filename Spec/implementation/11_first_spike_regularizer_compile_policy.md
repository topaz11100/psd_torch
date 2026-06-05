# First-spike readout and regularizer compile policy

## First-spike readout

`first_spike` uses a compile-friendly tensor adapter by default.

The adapter still loads the released first-spike timing and loss modules so the
project keeps the origin module availability contract, but runtime analysis and
loss use tensor-only equivalents:

- first-time forward semantics follow the released `Spike2Time` + `Time2FST` path;
- the released Gaussian surrogate backward is represented with a straight-through
  tensor surrogate instead of custom autograd Python indexing;
- train/eval losses use the released first-time cross-entropy and time
  regularization formula as state-free tensor functions;
- `enable_compiled_forward()` compiles analyze, train-loss, and eval-loss tensor
  functions with the same fixed compile kwargs used by model sequence regions.

Set `PSD_FIRST_SPIKE_ORIGIN_RUNTIME=1` only for reference/debug runs that must
call the released Python modules directly. That path is not the performance path.

## Signal/PSD regularizers

Training signal regularizers intentionally run outside `torch.compile`.

Rationale:

- the expensive recurrent SNN timestep loop is already handled by compiled
  sequence regions;
- regularizers operate on already-materialized traces with GPU tensor ops
  (`rfft`, reductions, PCA projections, spectral matrix ops);
- tracing FFT/PCA/dataclass/dict-heavy regularizer code gives limited runtime
  benefit and can introduce Dynamo graph breaks unrelated to model execution.

The training path therefore calls all regularizer logic through a
`torch.compiler.disable(recursive=True)` boundary when the API is available.
This is an eager-GPU boundary, not a CPU fallback. Gradients are preserved because
regularizer tensors are not detached and no `.cpu()` / `.numpy()` conversion is
used in the training regularizer path.

For PCA regularization, the fixed PCA reference bank is moved once to the active
training device before the epoch loop, avoiding per-minibatch host-to-device
transfers. Checkpoint metadata records `regularizer_backend=eager_gpu` and the
PCA bank devices.

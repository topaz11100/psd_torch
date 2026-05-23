# Dynamics Statistics

## Role

Dynamics analysis is separate from PSD and 2D FFT. It summarizes learned parameters and internal state metadata.

## Parameter vectors

Cell parameter vectors may include:

- LIF membrane decay,
- RF frequency, damping, and decay radius,
- threshold,
- group identifiers,
- lower and upper bounds,
- trainable flag,
- layer identity,
- scenario and constraint hash.

## Internal states

Internal-state statistics may summarize registered memory states and trace-derived states. They should not require raw trace value CSV output.

## Current status

The current implementation supports core parameter-vector collection and metadata propagation. Rich dynamics reporting and plotting remain future work.

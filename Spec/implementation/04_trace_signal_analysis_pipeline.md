# Trace and Signal Analysis Pipeline

## Pipeline

```text
CheckpointBundle
  -> ModelRestoreResult
  -> ProbeBatch
  -> TraceAdapter
  -> LayerTraceRecord
  -> SignalMapEmitter
  -> SignalMapRecord
  -> SignalAnalysisRunner or FFT2D runner
  -> artifact writers
```

## Trace adapter

The adapter coordinates model reset, forward trace capture, and metadata injection. Raw traces remain `B,T,*`.

## Signal map emitter

The emitter converts trace tensors to `S,R,T` and preserves run, checkpoint, probe, layer, series, and constraint metadata.

## Signal analysis runner

The runner supports PSD representatives, PCA, fixed-reference PCA application, exact/user-bin axes, and finalize-only dB output.

## 2D FFT runner

The 2D FFT runner is independent of representatives and writes `spectral_matrix_2d` artifacts.

## Failure metadata

Unavailable series, failed restore, incompatible PCA basis, and incompatible distance requests are represented with explicit status and reason where the writer path supports it.

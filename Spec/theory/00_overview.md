# Theory Overview

## Purpose

The refactored PSD/SNN system separates five concerns: model construction, trace collection, spectral analysis, artifact persistence, and plotting. The central object is a layer signal observed over simulation time and then converted into analysis maps.

```text
checkpoint/model + probe batch
  -> raw trace B,T,*
  -> SignalMap S,R,T
  -> PSD/PCA/2D FFT analysis
  -> summary artifacts and tensor artifacts
  -> artifact-reader based plotting
```

## Implemented analysis families

The current core supports:

- PSD representatives: `mean`, `median`, `element_psd`, and `pca`.
- Fixed-reference PCA basis fit/apply.
- Independent 2D FFT analysis.
- Strict distance compatibility for curves, 1D matrices, and 2D matrices.
- Trace tensor chunk artifacts with manifest rows.

## Execution units

The current CLI layer uses:

- `train`: synthetic smoke training and checkpoint creation.
- `analyze_signal`: PSD representatives and PCA on checkpoint or synthetic traces.
- `analyze_fft2d`: independent 2D FFT analysis.
- `analyze_dynamics`: parameter and state-statistics reporting.
- `plot_artifacts`: artifact-reader based plotting.

## Current scope

Implemented:

- MLP topology with IF/LIF/RF cells.
- Scenario constraints for clipping and structural grouping.
- Fixed topology smoke support for GRU, SSM/S4 alias, VGG, ResNet, and SpikeTransformer.
- Artifact reader and basic plotting.

Intentionally unsupported in this phase:

- raw trace value CSV dumps,
- output-layer constraint application,
- real dataset download/preprocessing integration,
- distributed or large-scale launch packaging.

## Theoretical inheritance from the old theory bundle

The old theory documents contributed these retained ideas:

- PSD-first analysis: representative spectral objects are built after row-level power spectra.
- Dense SNN skeleton separation from the neuron family.
- LIF and RF dynamics as distinct cell dynamics rather than topology names.
- Readout separation from hidden-layer dynamics.
- Fixed paper topologies as separate topology families, not MLP cell substitutions.

The removed legacy experiment family from the old bundle is not part of the current specification.

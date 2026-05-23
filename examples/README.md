# Examples Overview

This folder contains canonical runnable examples for the refactored PSD/SNN pipeline.

- `old/`, `Origin/`, `origin/`, `references/` are historical read-only references.
- Use `examples/bash/*.sh` for workflow execution.
- Use `examples/configs/runnable/*.json` as parseable configs.
- Use `examples/configs/commented/*.yaml.example` as heavily documented templates.

## Quick start
1. `source examples/bash/00_env.sh`
2. `examples/bash/01_train_synthetic_mlp.sh`
3. `examples/bash/03_analyze_signal_mean_median.sh`
4. `examples/bash/07_analyze_fft2d_exact.sh`
5. `examples/bash/11_plot_artifacts.sh`

## Forbidden legacy patterns
Do not use:
- raw trace CSV
- same_label
- label_single_excluding_balanced
- analysis_2d_fft
- schema_version / csv_v2
- reinterpretation path

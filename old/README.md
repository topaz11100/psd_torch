# Reference archives for refactor

These archives are intentionally kept as source references. Do not merge them wholesale into the active tree. Extract them into a temporary inspection directory when needed and port only the required ideas into the new architecture described in `../refactor_spec.md`.

| Archive | Purpose | High-value paths to inspect after extraction |
|---|---|---|
| `clip_structure_reference.zip` | Previous implementation of `clip`, `structure`, and `clipstructure` scenarios. | `src/common/psd_model_variants.py`, `src/common/psd_analysis_driver.py`, `bash/psd.sh`, `paper/proposed/psd_analysis.md`, `paper/proposed/psd_analysis_implement.md` |
| `pca_crossspec_2dfft_reference.zip` | PCA representative signal analysis, fixed PCA basis flow, PCA 1D/MIMO PSD paths, cross-spectrum/magnitude/phase audit, and matrix analysis references. | `src/psd_analysis.py`, `src/util/psd_analysis_driver.py`, `src/util/psd_minibatch_reg_driver.py`, `src/signal/*`, `bash/*psd*`, `bash/*2d*` |
| `distributed_set_reference.zip` | Distribution-aware probe selection and probe manifest logic. Port the distribution-aware idea into the new `distributed_set` probe family. | `src/stat/probe_selection.py`, `src/util/dataset_psd_driver.py`, `src/dataset_psd.py`, `Specification/*` |

Naming policy for the new implementation:

- `same_label` is renamed to `label_set`.
- `label_single_excluding_balanced` is renamed to `label_single` and is a top-level signal-analysis probe family, not only a 2D FFT scope.
- `distribution_global` / old distribution probe logic should be reinterpreted as `distributed_set`.
- `reinterpretation` is deleted, not ported.

# rf_scinario generated experiment matrix

This report records the generated RF-only scenario matrix.

## Global paths

- log root: `/home/yongokhan/workspace/logs`
- raw data root: `/home/yongokhan/workspace/data/raw_data`
- prep data root: `/home/yongokhan/workspace/data/prep_data`
- result root: `/home/yongokhan/workspace/result/rf_scinario`
- seed: `0`
- PSD window: `none`

## Training contract

- model family: vanilla `rf` only
- dense MLP hidden structure: `128,128,128`
- epochs: `50`
- checkpoint save epochs: `[1, ..., 50]`
- learning rate: `0.005`
- AMP: `on`
- GPU index: `[0, 1]`
- compile: `true`
- compile CPU threads: `8`
- scenario queue parallelism: `MAX_PARALLEL=2`

Batch sizes:

- `s-mnist`: `256`
- `shd`: `32`
- `deap`: `32`
- `dvs128-gesture`: `16`

## Model scenarios

For each dataset, the base grid is:

- filter: `fixed`, `train`
- threshold: `fixed`, `train`
- reset: `none`, `soft`

Additional recurrent scenario per dataset:

- `rf_R_none_fixed`: recurrent RF, reset `none`, threshold fixed, filter train.

Additional w-clip scenarios:

- only reset `none`, threshold fixed, filter train.
- w-clip bands are repeated over all three dense RF hidden layers of the `128,128,128` MLP.
- output layer is not part of the project constraint scope because class-count output layers cannot generally be partitioned into the requested frequency-band groups.

## Analysis contract

For model PSD, element PSD, element FFT, dataset PSD, dataset FFT, dataset element PSD, and dataset element FFT:

- batch size: `128`
- num workers: `8`
- seed: `0`
- PSD window: `none`

Dataset element analysis entrypoints are provided by:

- `src.dataset_element_psd`
- `src.dataset_element_fft`

## Generated count

- data prep: `4`
- dataset PSD: `4`
- dataset FFT: `4`
- dataset element PSD: `4`
- dataset element FFT: `4`
- model training: `41`
- PSD analysis: `41`
- element PSD: `41`
- element FFT: `41`
- plotting: `139`
- DI: `72`
- manifest: `1`
- total YAML: `396`

## Validation performed

- `python -m compileall -q src`
- `bash -n bash/*.sh`
- `bash -n bash/generated/rf_scinario/*.sh`
- all generated YAML parse
- all generated stage configs parse through their stage parser
- all `41` RF training configs pass synthetic eager forward/backward with no trainable parameter missing gradients
- representative RF configs pass regional compile-hook smoke using `backend=eager`
- representative RF configs pass single-process CPU DDP two-iteration smoke with compile-hook enabled

Full GPU multi-process DDP training was not executed in this packaging environment.

# rf_scinario

Generated RF-only scenario configuration set.

## Fixed paths and global settings

- log root: `/home/yongokhan/workspace/logs`
- raw data root: `/home/yongokhan/workspace/data/raw_data`
- prepared data root: `/home/yongokhan/workspace/data/prep_data`
- result root: `/home/yongokhan/workspace/result/rf_scinario`
- seed: `0`
- PSD window: `none`

## Training

- Dense MLP hidden spec: `128,128,128`
- AMP: `on`
- epochs: `50`
- analysis/checkpoint epochs: `1..50` stored as YAML arrays
- lr: `0.005`
- gpu_index: `[0, 1]`
- compile: `true`, compile_cpu_threads: `8`
- queue-level MAX_PARALLEL default: `2`

Only vanilla `rf` scenarios are included. All `my_*` scenarios were removed from this generated experiment.

## Added scenarios

Per dataset: baseline RF grid = filter fixed/train x v_th fixed/train x reset none/soft.

Also per dataset: `rf_R_none_fixed`, configured as recurrent RF, reset none, v_th fixed, filter train.

Additional w-clip scenarios are generated only for the requested datasets. Each w-clip edge set is repeated over the three dense RF hidden layers.

## Signal analysis

- num_workers: `8`
- batch / anal_batch: `128`
- stages: dataset_psd, dataset_fft, dataset_element_psd, dataset_element_fft, psd_analysis, element_psd, element_fft, plotting, DI.

# 구현 명세

## 1. 범위와 권한

이 문서는 implementation-level requirement index 다. 상세 contract 는 `Spec/impl/spec/` 아래에 저장하고, theoretical definition 은 `Spec/theory/` 아래에 저장한다.

### IMPL-SCOPE-001

요구사항:
`src/` 와 `bash/` 는 `Spec/impl/spec/` 아래 implementation contract 를 따라야 한다.

수용 기준:
- code 와 implementation specification 이 충돌하면 implementation specification 을 우선한다.
- implementation 과 theory 가 충돌하면 code change 전에 `Spec/conflict.md` 에 기록한다.

### IMPL-SCOPE-002

요구사항:
Traceability 는 specification-alignment judgment 이며 code pass/fail judgment 가 아니다.

수용 기준:
- `Spec/traceability.md` 는 theory 와 implementation requirement 사이의 coverage, ambiguity, conflict 를 기록한다.
- missing implementation 은 specification conflict 와 별도로 보고한다.

## 2. project structure

### IMPL-STRUCT-001

요구사항:
개편 project 는 기본 training-analysis-plotting entrypoint 와 auxiliary checkpoint-analysis entrypoint 들을 갖는다.

적용 대상:
- `src/model_training.py`
- `src/psd_analysis.py`
- `src/plotting.py`
- `src/2d_fft_analysis.py`
- `src/element_psd.py`
- `src/checkpoint_accuracy_analysis.py`

수용 기준:
- `src/model_training.py` 는 training-only program 으로 존재한다.
- `src/psd_analysis.py` 는 representative checkpoint-analysis-only program 으로 존재한다.
- `src/plotting.py` 는 visualization-only program 으로 존재한다.
- `src/2d_fft_analysis.py` 는 MLP probe input map 과 output map 의 2-D FFT matrix 를 CSV 로 저장하는 checkpoint-analysis-only program 으로 존재한다.
- `src/element_psd.py` 는 MLP probe input map 과 output map 의 element-wise PSD matrix 를 CSV 로 저장하는 checkpoint-analysis-only program 으로 존재한다.
- `src/checkpoint_accuracy_analysis.py` 는 저장된 checkpoint 의 train/test full split 정확도를 CSV 로 저장하는 checkpoint-analysis-only program 으로 존재한다.
- 각 entrypoint 는 서로를 암묵적으로 호출하지 않는다.

### IMPL-STRUCT-002

요구사항:
기존 support package 는 split 이후에도 재사용 가능해야 한다.

적용 대상:
- `src/data/`
- `src/model/`
- `src/neurons/`
- `src/readout/`
- `src/signal/`
- `src/stat/`
- `src/plot/`
- `src/reinterpretation/`
- `src/util/`

수용 기준:
- training utility 는 scientific meaning 을 바꾸지 않고 `model_training.py` 용으로 이동하거나 노출한다.
- signal-analysis utility 는 이동하거나 `psd_analysis.py` 에서 import 가능하게 유지한다.
- plotting utility 는 이동하거나 `plotting.py` 에서 import 가능하게 유지한다.
- support package 는 training-analysis-plotting coupling 을 다시 도입하지 않는다.

### IMPL-STRUCT-003

요구사항:
`Spec/` 는 유일한 official specification directory name 이다.

수용 기준:
- documentation 과 launcher comment 는 old specification folder name 이 아니라 `Spec/` 를 참조한다.
- implementation subdocument 는 old implementation folder name 이 아니라 `Spec/impl/` 를 사용한다.
- prior conflict-audit version file 은 제거한다. 현재 audit file 은 `Spec/conflict.md` 다.

## 3. CLI 와 launcher contract

### IMPL-CLI-001

요구사항:
Bash launcher 는 stage 별로 분리되어 parser-defined Python argument 만 전달해야 한다.

적용 대상:
- `bash/data_prep.sh`
- `bash/dataset_psd.sh`
- `bash/model_training.sh`
- `bash/psd_analysis.sh`
- `bash/2d_fft_analysis.sh`
- `bash/element_psd.sh`
- `bash/checkpoint_accuracy_analysis.sh`
- `bash/plotting.sh`
- `bash/reinterpretation.sh`
- `src/data_prep.py`
- `src/dataset_psd.py`
- `src/model_training.py`
- `src/psd_analysis.py`
- `src/2d_fft_analysis.py`
- `src/element_psd.py`
- `src/checkpoint_accuracy_analysis.py`
- `src/plotting.py`
- `src/reinterpretation/driver.py`

수용 기준:
- Python CLI 에 mapping 되는 모든 bash variable 은 대응 parser argument 를 갖는다.
- 제거 대상 argument 는 bash, parser, documentation 에서 함께 제거한다.
- selected analysis checkpoint epoch list 는 launcher level 에서 `ANAL_EPOCH_LIST` 라고 부르며 `bash/model_training.sh` 와 `model_training.py` 만 consume 한다.
- 제거 대상 prepared-data direct-path, FFT-length, curve/scale filtering controls 는 official interface 에 남기지 않는다.

### IMPL-CLI-002

요구사항:
Plot rendering 은 training 또는 analysis flag 로 제어하지 않는다.

수용 기준:
- `model_training.py` 는 figure-rendering option 을 갖지 않는다.
- `psd_analysis.py` 는 figure-rendering option 을 갖지 않는다.
- `plotting.py` 는 figure 를 render 하는 유일한 official entrypoint 다.
- analysis CSV 로부터 figure rendering 을 하려면 `plotting.py` 를 명시적으로 실행한다.

### IMPL-CLI-003

요구사항:
`psd_analysis.py` 는 explicit GPU 와 analysis-batch control 을 노출한다.

수용 기준:
- `--gpu_index <int>` 는 analysis 에 사용할 CUDA device 를 선택한다.
- `--anal_batch <int>` 는 한 analysis forward pass 에서 GPU 로 보내는 sample 수의 최대값을 정한다.
- checkpoint metadata 에 저장된 training batch size 와 무관하게 checkpoint analysis 에서 `--anal_batch` 를 enforce 한다.

### IMPL-CLI-004

요구사항:
`plotting.py` 는 explicit CPU worker control 을 노출한다.

수용 기준:
- plotting CLI 에서 `--cpu_cores` 는 제거한다. rendering 은 single-process 로 실행한다.
- 1보다 작은 값은 invalid 다.

### IMPL-CLI-005

요구사항:
Main PSD pipeline 의 bash launcher 는 Python entrypoint 와 동일하게 세 stage 로 분리한다.

적용 대상:
- `bash/model_training.sh`
- `bash/psd_analysis.sh`
- `bash/plotting.sh`

수용 기준:
- `bash/model_training.sh` 는 `src.model_training` 만 호출한다.
- `bash/psd_analysis.sh` 는 `src.psd_analysis` 만 호출한다.
- `bash/plotting.sh` 는 `src.plotting` 만 호출한다.
- 각 launcher 는 stage-specific input 만 받으며 다른 stage argument 를 전달하지 않는다.
- `bash/psd_analysis.sh` 는 `CHECKPOINTS_PER_JOB` 과 `GPU_INDEX_SET` 으로 checkpoint grouping 과 GPU assignment 를 제어할 수 있다.
- `bash/plotting.sh` 는 checkpoint path 를 Python plotting process 로 전달하지 않는다.
- 모든 launcher 는 child job 을 background launch 하고 기다리지 않는다.
- launcher 내부에 child 종료 대기 기반 queue 또는 동시 실행 수 제한을 두지 않는다.

### IMPL-CLI-006

요구사항:
Matrix-valued auxiliary checkpoint-analysis 의 bash launcher 는 `psd_analysis.sh` 와 같은 checkpoint grouping, GPU assignment, logging contract 를 따른다.

적용 대상:
- `bash/2d_fft_analysis.sh`
- `bash/element_psd.sh`
- `src/2d_fft_analysis.py`
- `src/element_psd.py`

수용 기준:
- `bash/2d_fft_analysis.sh` 는 `src.2d_fft_analysis` 만 호출한다.
- `bash/element_psd.sh` 는 `src.element_psd` 만 호출한다.
- 두 launcher 는 `CHECKPOINT_SET`, `CHECKPOINTS_PER_JOB`, `GPU_INDEX_SET`, `DATASET`, `PREP_ROOT`, `OUTPUT_ROOT`, `ANAL_BATCH`, `LOW_VRAM`, `SEED`, `NUM_WORKERS` 를 사용한다.
- 두 launcher 는 parser-defined Python argument 만 전달한다.
- 두 launcher 는 child job 을 `nohup` background 로 launch 하고 기다리지 않는다.
- 두 launcher 내부에 child 종료 대기 기반 queue 또는 동시 실행 수 제한을 두지 않는다.


### IMPL-CLI-006A

요구사항:
Checkpoint accuracy auxiliary analysis 의 bash launcher 는 `psd_analysis.sh` 와 같은 checkpoint grouping, GPU assignment, logging contract 를 따른다.

적용 대상:
- `bash/checkpoint_accuracy_analysis.sh`
- `src/checkpoint_accuracy_analysis.py`

수용 기준:
- `bash/checkpoint_accuracy_analysis.sh` 는 `src.checkpoint_accuracy_analysis` 만 호출한다.
- launcher 는 `CHECKPOINT_SET`, `CHECKPOINTS_PER_JOB`, `GPU_INDEX_SET`, `DATASET`, `PREP_ROOT`, `OUTPUT_ROOT`, `ANAL_BATCH`, `SEED`, `NUM_WORKERS`, `SPLITS` 를 사용한다.
- Python entrypoint 는 저장된 checkpoint 를 복원해 train/test full split 을 추론 평가하고 `checkpoint_accuracy.csv` 를 저장한다.
- 이 entrypoint 는 model training 과 figure rendering 을 수행하지 않는다.
- launcher 는 child job 을 `nohup` background 로 launch 하고 기다리지 않는다.
- launcher 내부에 child 종료 대기 기반 queue 또는 동시 실행 수 제한을 두지 않는다.

### IMPL-CLI-007

요구사항:
Checkpoint analysis 는 curve-source selection 으로 output extractor 를 필터링하지 않는다.

수용 기준:
- `psd_analysis.py` 와 `bash/psd_analysis.sh` 는 curve-source filtering argument 를 노출하지 않는다.
- `analysis_curve`, `analysis_dispersion`, pair distance, drift output 은 명세에 정의된 모든 extractor 를 포함한다.
- Output value-scale selector 를 노출하지 않는다. 명세에 정의된 모든 extractor 와 `raw`, `db` scale CSV 를 모두 포함한다.

### IMPL-CLI-008

요구사항:
Dataset PSD baseline 은 dataset/GPU slot 단위로 병렬 launch 되고 train/test selector 를 갖지 않는다.

수용 기준:
- `DATASET_PSD_SET` 원소는 `<dataset_token>|<gpu_index>` 문법을 따른다.
- train/test 선택을 `DATASET_PSD_SET` 원소에 넣지 않는다.
- `dataset_psd.py` 는 `--batch_size`, `--gpu_index` 를 노출한다. `--userbin_edges` 는 실행 경로에서 사용하지 않는다.
- `dataset_psd.py` 는 train/test selector, FFT-length control, prepared-data direct-path interface 를 노출하지 않는다.

### IMPL-CLI-009

요구사항:
Userbin boundary 는 explicit argument 로 줄 수 있어야 한다.

수용 기준:
- `dataset_psd.py` 와 `psd_analysis.py` 는 `--userbin_edges` 를 받는다.
- 기본값은 `0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.45 0.50` 이다.
- userbin 은 exact PSD 뒤 frequency aggregation 으로만 구현한다.
- userbin 은 원 신호 길이나 FFT 길이를 바꾸지 않는다.

## 4. training contract

### IMPL-TRAIN-001

요구사항:
`src/model_training.py` 는 prepared data 에서 model 을 train 하고 selected model checkpoint 만 저장한다.

상세 명세:
- `Spec/impl/spec/model_training.md`

수용 기준:
- program 은 model, prepared data, optimizer, epoch, seed, readout, batch setting 을 CLI 로 받는다.
- prepared data 는 `--prep_root` 와 `--dataset` 으로 resolve 한다.
- program 은 supervised training 과 evaluation 만 수행한다.
- program 은 layer PSD artifact 를 계산하지 않는다.
- program 은 figure 를 render 하지 않는다.

### IMPL-TRAIN-002

요구사항:
`ANAL_EPOCH_LIST` 는 later analysis 를 위한 checkpoint epoch 를 정의한다.

수용 기준:
- 모든 epoch 값은 `1 <= epoch <= epochs` 를 만족하는 integer 다.
- duplicate epoch value 는 validation 뒤 deduplicate 한다.
- 값이 제공되지 않으면 final training epoch 를 기본 저장한다.
- normalized list 에 속하는 epoch 만 `.pt` checkpoint file 을 생성한다.

### IMPL-TRAIN-003

요구사항:
analysis 용 checkpoint directory 는 strict `.pt`-only directory 다.

수용 기준:
- clean checkpoint directory 는 하나 이상의 `.pt` file 을 포함한다.
- metric CSV, log, sidecar, figure, non-`.pt` regular file 을 그 directory 안에 쓰지 않는다.
- training metric CSV file 은 clean checkpoint directory 밖에 쓴다.

### IMPL-TRAIN-004

요구사항:
selected checkpoint 는 standalone analysis reconstruction 에 필요한 모든 metadata 를 저장한다.

수용 기준:
각 checkpoint `.pt` 는 최소 아래 항목을 포함한다.
- `schema_version`
- `epoch`
- `model_token`
- `model_config`
- `state_dict`
- `readout_config`
- `dataset_token`
- `prep_root`
- `prepared_dataset_path`
- `axis_metadata_ref`
- `seed`
- `training_args`
- `normalization_metadata`
- `hidden_spec_normalized`

Optimizer 와 scheduler state 는 resume workflow 에만 optional 이며 `psd_analysis.py` 에는 필요하지 않다.

### IMPL-TRAIN-005

요구사항:
Fixed CNN-SNN full token 에서는 `hidden_spec` placeholder 가 dense hidden width 로 되살아나면 안 된다.

수용 기준:
- fixed CNN full token 에 대해 `--hidden_spec -`, empty string, `default` 는 `None` 으로 normalize 한다.
- fixed CNN builder 에 `bundle.default_hidden_sizes` 를 주입하지 않는다.
- CNN-dense tail model 만 dense tail hidden widths 를 받는다.

## 5. checkpoint analysis contract

### IMPL-ANAL-001

요구사항:
`src/psd_analysis.py` 는 saved checkpoint 를 분석하고 CSV artifact 를 쓴다.

상세 명세:
- `Spec/impl/spec/psd_analysis.md`
- `Spec/impl/spec/csv_schema.md`

수용 기준:
- program 은 단일 `.pt` file 또는 `.pt` file 의 strict directory 하나를 받는다.
- program 은 `--dataset` 과 `--prep_root` 를 받는다.
- program 은 model reconstruction, checkpoint load, probe forward pass, layer signal capture, PSD/statistical computation, CSV writing 을 수행한다.
- program 은 train 하지 않는다.
- program 은 figure 를 render 하지 않는다.

### IMPL-ANAL-002

요구사항:
Directory-mode checkpoint input 은 strict 하다.

수용 기준:
- directory mode 는 input directory 바로 아래의 모든 `.pt` file 을 분석한다.
- directory 에 `.pt` file 이 없으면 analysis 전에 fail 한다.
- directory 에 non-`.pt` regular file 이 하나라도 있으면 analysis 전에 fail 한다.
- future spec 이 recursive checkpoint mode 를 명시적으로 추가하지 않는 한 checkpoint input directory 의 subdirectory 는 invalid 다.

### IMPL-ANAL-003

요구사항:
Checkpoint analysis 는 retraining 없이 multi-checkpoint epoch trace 를 지원한다.

수용 기준:
- multiple checkpoint 가 제공되면 checkpoint epoch value 가 trace axis 를 정의한다.
- family spectral trace, layer distance trace, layer dispersion trace, pair-distance trace, filter trend 는 checkpoint metadata 와 analysis output 에서 계산한다.
- trace generation 은 optimizer state 를 요구하지 않는다.

### IMPL-ANAL-004

요구사항:
모든 analysis output 은 category-based CSV schema 를 사용한다.

수용 기준:
- 모든 official analysis CSV 는 `Spec/impl/spec/csv_schema.md` 의 common prefix 와 category-specific columns 를 포함한다.
- artifact-specific wide table 은 category schema 로 명시되지 않는 한 official output 이 아니다.
- 각 curve 조합은 독립 CSV file 로 저장한다.

### IMPL-ANAL-005

요구사항:
Analysis 는 train/test split 선택 없이 prepared probe scope 만 처리한다.

수용 기준:
- train/test selector CLI 를 두지 않는다.
- prepared train/test full split 전체집합 scope 를 처리하지 않는다.
- 현재 checkpoint analysis 는 `train_balanced_global`, `test_balanced_global` probe scope 를 처리한다.
- 현재 `same_label` 및 `distribution_global` scope 는 checkpoint analysis 산출물로 생성하지 않는다.

## 6. plotting contract

### IMPL-PLOT-001

요구사항:
`src/plotting.py` 는 CSV 에서만 figure 를 render 한다.

상세 명세:
- `Spec/impl/spec/plotting.md`

수용 기준:
- program 은 단일 `.csv` file 또는 directory 하나를 받는다.
- file mode 는 `category` 가 지원되면 주어진 CSV 를 render 한다.
- directory mode 는 모든 child directory 를 재귀적으로 순회하고 모든 `.csv` file 을 고려한다.
- program 은 `.pt` 를 load 하지 않고 model forward 를 실행하지 않는다.

### IMPL-PLOT-002

요구사항:
Plotting output 은 derivative artifact 다.

수용 기준:
- figure 는 numerical reproducibility 에 필수가 아니다.
- rendering manifest CSV 는 각 input CSV, render status, output path, applicable error message 를 기록한다.
- unsupported CSV category 는 manifest status `skipped_unsupported_category` 로 건너뛴다.

### IMPL-PLOT-003

요구사항:
PSD curve figure 는 fixed visualization style 을 따른다.

수용 기준:
- x축 라벨은 `Normalized Frequency` 다.
- x축 범위는 `0.0` 부터 `0.5` 다.
- x축 tick 은 `0.0`, `0.1`, `0.2`, `0.3`, `0.4`, `0.5` 를 표시한다.
- PSD raw y축 라벨은 `Power`, dB y축 라벨은 `Power (dB)` 다.
- 입력 제목은 `PSD of Input`, n번 레이어 제목은 `PSD of Layer n` 다.
- `figsize=(14, 4)`, line width `3.5`, title font `25`, label font `20`, tick font `18` 을 사용한다.
- 배경은 흰색, grid 는 off, spine width 와 tick width 는 `1.2`, tick length 는 `6` 이다.
- PNG, `300 dpi`, `bbox_inches="tight"`, `facecolor="white"` 로 저장한다.

## 7. data preparation 과 dataset baseline contract

### IMPL-DATA-001

요구사항:
`data_prep` writer 는 entire split 을 RAM 에 materialize 하지 않는다.

상세 명세:
- `Spec/impl/spec/data_prep.md`

수용 기준:
- split-wide `np.stack`, split-wide torch tensor, prepared data 용 `.pt` materialization 은 official path 가 아니다.
- stored dtype 은 raw meaning 과 memory efficiency 를 보존한다.

### IMPL-DSETPSD-001

요구사항:
`dataset_psd` 는 dataset-level input/probe baseline PSD 를 소유한다.

상세 명세:
- `Spec/impl/spec/dataset_psd.md`

수용 기준:
- dataset baseline output 은 category-based CSV schema 를 사용한다.
- dataset baseline 은 model 을 train 하지 않는다.
- dataset baseline 은 figure 를 render 하지 않는다.
- dataset baseline 은 `--batch_size` 로 streaming 처리한다.
- dataset baseline 은 `--gpu_index` 로 GPU 를 선택한다.

## 8. model 과 optimizer contract

### IMPL-MODEL-001

요구사항:
Dense SNN, recurrent dense SNN, CNN-SNN, SpikingSSM, Spikformer, SpikGRU model contract 는 split 이후에도 보존한다.

상세 명세:
- `Spec/impl/spec/model_dense.md`
- `Spec/impl/spec/model_cnn.md`
- `Spec/impl/spec/model_state_space.md`
- `Spec/impl/spec/model_spikformer.md`
- `Spec/impl/spec/model_spikegru.md`

수용 기준:
- checkpoint reconstruction 에 필요한 model token 과 architecture metadata 는 selected checkpoint 에 serialize 한다.
- analysis 는 launcher-only hidden state 가 아니라 checkpoint metadata 에서 model 을 reconstruct 한다.

### IMPL-OPT-001

요구사항:
명시적인 future spec 이 변경하지 않는 한 official project-side training optimizer 는 Adam 으로 유지한다.

수용 기준:
- optimizer metadata 는 official project-side training run 에 대해 `optimizer_family = Adam` 을 기록한다.
- PSD curve-shape regularization 이 사용되는 경우 optimizer decay mechanism 이 아니라 $d_{\mathrm{shape}}$ 기반 training loss term 으로 유지한다.
- analysis 와 plotting stage 는 optimizer 를 construct 하지 않는다.

## 9. output contract

### IMPL-OUT-001

요구사항:
모든 numeric output 은 `Spec/impl/spec/csv_schema.md` 를 사용한다.

수용 기준:
- `training_metrics.csv`, dataset baseline CSV, model analysis CSV, trace CSV, plotting manifest CSV 는 category-based schema 를 따른다.
- program-specific metadata 는 category column 또는 category schema 가 허용하는 column 으로 표현한다.

### IMPL-OUT-002

요구사항:
Binary output 은 제한되고 type 이 정해져 있다.

수용 기준:
- `model_training.py` 는 selected `.pt` checkpoint 를 쓸 수 있다.
- `psd_analysis.py` 는 model checkpoint 를 쓰지 않는다.
- `plotting.py` 는 rendering manifest CSV 외의 numeric analysis output 을 쓰지 않는다.

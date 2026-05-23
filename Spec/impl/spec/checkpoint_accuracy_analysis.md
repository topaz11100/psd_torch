# checkpoint_accuracy_analysis 구현 계약

## 1. 문서 역할

이 문서는 `src/checkpoint_accuracy_analysis.py` 의 implementation contract 를 정의한다.

`src/checkpoint_accuracy_analysis.py` 는 이미 저장된 checkpoint 를 불러와 prepared train/test full split 에 대한 추론 정확도를 CSV 로 저장한다. 이 program 은 model 을 train 하지 않고 figure 를 render 하지 않는다.

## 2. 범위

이 program 은 checkpoint metadata 로 model 과 readout 을 복원한 뒤, prepared dataset bundle 의 train/test split 전체를 평가한다.

적용 대상:

1. 단일 `.pt` checkpoint file
2. 바로 아래에 `.pt` file 만 포함하는 strict checkpoint directory

이 program 은 probe subset, PSD curve, 2-D FFT matrix, element-wise PSD matrix 를 생성하지 않는다.

## 3. official CLI

| argument | required | 의미 |
| --- | --- | --- |
| `--checkpoint` | yes | 단일 `.pt` checkpoint file 또는 `.pt` file 만 포함하는 strict directory |
| `--dataset` | yes | checkpoint metadata 의 canonical dataset token |
| `--prep_root` | yes | prepared dataset root |
| `--output_root` | yes | accuracy CSV output root |
| `--anal_batch` | yes | 한 forward pass 에서 GPU 로 보내는 sample 수의 최대값 |
| `--gpu_index` | yes | accuracy evaluation 용 CUDA device index |
| `--seed` | optional | evaluation seed, 기본값은 checkpoint seed |
| `--num_workers` | optional | DataLoader worker 수 |
| `--splits` | optional | 평가할 split 목록, 기본값은 `train test` |

이 program 은 plotting argument, training argument, optimizer argument, PSD extractor argument 를 받지 않는다.

## 4. input path mode

`--checkpoint` path mode 는 `psd_analysis.py` 와 동일하다.

### 4.1 file mode

1. 입력 file suffix 는 `.pt` 여야 한다.
2. 해당 checkpoint 하나만 평가한다.
3. output 은 `<output_root>/checkpoint_accuracy.csv` 에 쓴다.

### 4.2 strict directory mode

1. directory 바로 아래에는 `.pt` file 만 있어야 한다.
2. subdirectory 는 invalid 다.
3. non-`.pt` regular file 은 invalid 다.
4. checkpoint epoch metadata 가 있으면 epoch 오름차순으로 평가한다.
5. checkpoint 중 하나라도 epoch metadata 가 없으면 lexical order 를 사용하고 manifest 에 warning 을 기록한다.

## 5. 평가 대상 split

기본 split 은 아래 둘이다.

| scope | 의미 |
| --- | --- |
| `train` | prepared train split 전체 |
| `test` | prepared test split 전체 |

이 program 은 `psd_analysis.py`, `2d_fft_analysis.py`, `element_psd.py` 와 달리 probe subset 을 사용하지 않는다. 정확도는 split 전체에 대한 추론 결과로 계산한다.

## 6. CSV category

Checkpoint accuracy output 은 `category=checkpoint_accuracy` 를 사용한다.

한 row 는 checkpoint 하나와 split 하나의 accuracy summary 를 의미한다.

고정 column 은 아래 좌표를 포함한다.

```text
model_token
model_family
readout_mode
seed
checkpoint_path
checkpoint_epoch
scope
accuracy
correct
total
value_unit
```

규칙:

1. `scope` 는 `train` 또는 `test` 다.
2. `accuracy` 는 `correct / total` 값이며 `value_unit=fraction` 을 사용한다.
3. `correct` 와 `total` 은 accuracy 재계산을 위한 count metadata 다.
4. loss 는 official output 이 아니다.

## 7. output path contract

권장 layout:

```text
<output_root>/
  checkpoint_accuracy.csv
  analysis_manifest.csv
```

`bash/checkpoint_accuracy_analysis.sh` 를 사용하면 실제 output root 는 `<OUTPUT_ROOT>/<case_id>/` 이다.

규칙:

1. category column 이 없는 CSV 를 쓰지 않는다.
2. binary bundle 을 쓰지 않는다.
3. figure file 을 쓰지 않는다.
4. `analysis_manifest.csv` 에 생성 artifact 를 기록한다.

## 8. bash launcher contract

`bash/checkpoint_accuracy_analysis.sh` 는 이 program 의 official bash launcher 다.

필수 launcher contract:

1. `bash/checkpoint_accuracy_analysis.sh` 는 `src.checkpoint_accuracy_analysis` 만 호출한다.
2. checkpoint input 은 `CHECKPOINT_SET` 또는 `CHECKPOINT_SET_RAW` 로 받는다.
3. checkpoint grouping 은 `CHECKPOINTS_PER_JOB` 으로 제어한다.
4. GPU assignment 는 `GPU_INDEX_SET` 으로 제어한다.
5. `DATASET`, `PREP_ROOT`, `OUTPUT_ROOT`, `ANAL_BATCH`, `SEED`, `NUM_WORKERS`, `SPLITS` 는 Python argument 로 mapping 한다.
6. log directory 는 `<LOG_ROOT>/checkpoint_accuracy_analysis/<RUN_STAMP>` 이다.
7. child job 은 `nohup` background process 로 실행하고 launcher 는 종료를 기다리지 않는다.
8. launcher 내부에 child 종료 대기 기반 queue 또는 동시 실행 수 제한을 두지 않는다.

# element_psd 구현 계약

## 1. 문서 역할

이 문서는 `src/element_psd.py` 의 implementation contract 를 정의한다.

`src/element_psd.py` 는 이미 저장된 checkpoint 를 불러와 prepared probe set 에 대한 MLP probe input map 과 layer output map 의 element-wise exact PSD 를 CSV 로 저장한다. 기존 `psd_analysis.py` 처럼 neuron axis 를 대표화하지 않고 각 input element 또는 neuron 의 PSD curve 를 그대로 보존한다.

## 2. 범위

현재 구현 범위는 MLP-style dense SNN 으로 제한한다.

허용 model family:

| family | 의미 |
| --- | --- |
| `lif` | dense LIF MLP 계열 |
| `rf` | dense RF MLP 계열 |

금지 model family 는 `2d_fft_analysis.py` 와 같다. 금지 family 가 들어오면 fail 한다.

## 3. official CLI

| argument | required | 의미 |
| --- | --- | --- |
| `--checkpoint` | yes | 단일 `.pt` checkpoint file 또는 `.pt` file 만 포함하는 strict directory |
| `--dataset` | yes | checkpoint metadata 의 canonical dataset token |
| `--prep_root` | yes | prepared dataset root |
| `--output_root` | yes | element-wise PSD CSV output root |
| `--anal_batch` | yes | 한 forward pass 에서 GPU 로 보내는 probe sample 수의 최대값 |
| `--gpu_index` | yes | analysis 용 CUDA device index |
| `--seed` | optional | analysis seed, 기본값은 checkpoint seed |
| `--num_workers` | optional | probe loading 용 DataLoader worker 수 |
| `--low_vram` | optional | `1` 이면 trace 를 CPU 로 stage 하여 VRAM 사용량을 줄임 |

이 program 은 plotting argument, training argument, optimizer argument 를 받지 않는다. `--checkpoint` path mode 는 `2d_fft_analysis.py` 와 동일하다.

## 4. probe scope

이 program 은 train/test full split 전체를 분석하지 않는다. prepared data 에서 model-family selected training view 를 구성한 뒤 seed 기반 probe subset 만 사용한다. 정적 이미지 dataset 의 MLP 분석은 flatten training view 를 사용한다.

필수 scope:

| scope 형식 | 의미 |
| --- | --- |
| `<split>_balanced_global` | label 별 동일 quota 로 구성한 balanced probe set |
| `<split>_label_single_label_<c>` | label `c` 에서 deterministic 하게 선택한 sample 한 장 |

`<split>` 은 `train` 또는 `test` 다.

Balanced scope 는 sample 별 neuron-wise PSD 를 계산한 뒤 sample axis 로 평균내어 하나의 matrix 를 저장한다. Label-single scope 는 같은 split 의 같은 label 후보 중 `<split>_balanced_global` 에 이미 포함된 index 를 제외한 뒤 deterministic 하게 선택한 sample 한 장에 대한 matrix 를 저장한다.

## 5. 분석 대상 signal

이 program 은 MLP probe input map 과 MLP layer output map 을 분석한다. 아래 series 의 tensor 가 capture 결과에 존재할 때 저장한다.

저장 대상 official series:

| signal_kind | series | 의미 |
| --- | --- | --- |
| `input` | `x_probe` | model 에 투입되는 prepared probe input trace |
| `hidden` | `membrane` | hidden layer membrane output trace |
| `hidden` | `spike` | hidden layer spike output trace |
| `output` | `membrane` | output layer membrane trace |
| `output` | `spike` | output layer spike trace |

`layer_input`, `readout_mem`, gate/current auxiliary trace 는 official output 이 아니다.

## 6. element-wise PSD 정의

Input trace 와 layer output trace 는 model 에 투입되거나 capture 된 `(B,T,N)` 형태를 기준으로 하며, analysis map 으로는 아래 형태를 사용한다.

$$
X \in \mathbb{R}^{B \times N \times T}
$$

여기서 $B$ 는 probe sample 수, $N$ 은 input element 수 또는 neuron 수, $T$ 는 time step 수다. Row index 는 실제 input vector 또는 layer output vector 구조의 0-based order 를 따른다. 따라서 CSV 의 `neuron_index=0` 은 input/layer output vector 의 첫 번째 element 다.

Raw variant 는 각 input element 또는 neuron trace 를 그대로 사용한다. Centered variant 는 각 input element 또는 neuron 의 time-axis 평균을 제거한다.

$$
X_{b,i,t}^{centered}=X_{b,i,t}-\frac{1}{T}\sum_{t'=0}^{T-1}X_{b,i,t'}
$$

각 input element 또는 neuron trace 에 대해 exact Hann-windowed one-sided periodogram 을 계산한다. Balanced scope 에서는 sample axis 로 평균낸다.

$$
\bar{S}_{i,f}=\frac{1}{B}\sum_{b=1}^{B}S_{b,i,f}
$$

## 7. CSV matrix 형식

Element-wise PSD output 은 `category=element_psd` 를 사용한다.

한 CSV 는 하나의 input/layer, scope, label condition, signal, series, variant, scale 조합만 담는다.

행과 열의 의미:

| CSV 방향 | 의미 |
| --- | --- |
| row | input element 또는 `neuron_index` 하나 |
| dynamic columns | `freq_000000`, `freq_000001`, ... one-sided frequency bin |
| value | 해당 input element 또는 neuron 의 PSD 값 |

주의할 점은 dynamic column 이 원래 time step 자체가 아니라 time axis 에서 유도된 frequency bin 이라는 것이다. PSD 는 frequency-domain 값이므로 열 이름은 `freq_<index>` 를 사용한다. Frequency 값은 `frequency_grid=exact_one_sided_index_over_time_length`, `time_length`, `frequency_bin_count` 로 복원한다.

## 8. variant 와 scale

모든 output coordinate 에 대해 아래 조합을 저장한다.

| axis | values |
| --- | --- |
| `variant` | `raw`, `centered` |
| `scale` | `raw`, `db` |

`scale=db` 는 PSD matrix 에 대해 $10\log_{10}(S+\epsilon)$ 를 적용한 값이다. $\epsilon$ 은 implementation 의 `DB_EPSILON` 을 따른다.

## 9. output path contract

권장 layout:

```text
<output_root>/
  analysis_manifest.csv
  checkpoint_epoch_000001/
    layers/
      layer_000__input/
        element_psd/
          element_psd__epoch_1__layer_0__input__train_balanced_global__all_labels__input__x_probe__balanced_mean__raw__raw.csv
      layer_001__layer_01/
        element_psd/
          element_psd__epoch_1__layer_1__layer_01__train_balanced_global__all_labels__hidden__membrane__balanced_mean__raw__raw.csv
```

규칙:

1. 기존 `analysis_curve` 처럼 neuron axis 를 평균 또는 median 으로 대표화하지 않는다.
2. 한 CSV 안에 서로 다른 category 를 섞지 않는다.
3. binary bundle 을 쓰지 않는다.
4. figure file 을 쓰지 않는다.
5. `analysis_manifest.csv` 에 생성 artifact 를 기록한다.

## 10. bash launcher contract

`bash/element_psd.sh` 는 이 program 의 official bash launcher 다.

필수 launcher contract:

1. `bash/element_psd.sh` 는 `src.element_psd` 만 호출한다.
2. checkpoint input 은 `CHECKPOINT_SET` 또는 `CHECKPOINT_SET_RAW` 로 받는다.
3. checkpoint grouping 은 `CHECKPOINTS_PER_JOB` 으로 제어한다.
4. GPU assignment 는 `GPU_INDEX_SET` 으로 제어한다.
5. `DATASET`, `PREP_ROOT`, `OUTPUT_ROOT`, `ANAL_BATCH`, `LOW_VRAM`, `SEED`, `NUM_WORKERS` 는 같은 이름의 Python argument 로만 mapping 한다.
6. log directory 는 `<LOG_ROOT>/element_psd/<RUN_STAMP>` 이다.
7. child job 은 `nohup` background process 로 실행하고 launcher 는 종료를 기다리지 않는다.
8. launcher 내부에 child 종료 대기 기반 queue 또는 동시 실행 수 제한을 두지 않는다.

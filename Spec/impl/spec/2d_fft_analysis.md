# 2d_fft_analysis 구현 계약

## 1. 문서 역할

이 문서는 `src/2d_fft_analysis.py` 의 implementation contract 를 정의한다.

`src/2d_fft_analysis.py` 는 이미 저장된 checkpoint 를 불러와 prepared probe set 에 대한 MLP probe input map 과 layer output map 의 2-D FFT power matrix 를 CSV 로 저장한다. 이 program 은 model 을 train 하지 않고 figure 를 render 하지 않는다.

## 2. 범위

현재 구현 범위는 MLP-style dense SNN 으로 제한한다.

허용 model family:

| family | 의미 |
| --- | --- |
| `lif` | dense LIF MLP 계열 |
| `rf` | dense RF MLP 계열 |

금지 model family:

1. `cnn_lif`
2. `cnn_rf`
3. `spikegru`
4. `spikformer`
5. `spikingssm`
6. 기타 MLP output map 구조가 명확히 정의되지 않은 family

금지 family 가 들어오면 forward pass 후 조용히 skip 하지 않고 fail 한다.

## 3. official CLI

| argument | required | 의미 |
| --- | --- | --- |
| `--checkpoint` | yes | 단일 `.pt` checkpoint file 또는 `.pt` file 만 포함하는 strict directory |
| `--dataset` | yes | checkpoint metadata 의 canonical dataset token |
| `--prep_root` | yes | prepared dataset root |
| `--output_root` | yes | 2-D FFT CSV output root |
| `--anal_batch` | yes | 한 forward pass 에서 GPU 로 보내는 probe sample 수의 최대값 |
| `--gpu_index` | yes | analysis 용 CUDA device index |
| `--seed` | optional | analysis seed, 기본값은 checkpoint seed |
| `--num_workers` | optional | probe loading 용 DataLoader worker 수 |
| `--low_vram` | optional | `1` 이면 trace 를 CPU 로 stage 하여 VRAM 사용량을 줄임 |

이 program 은 plotting argument, training argument, optimizer argument 를 받지 않는다.

## 4. input path mode

`--checkpoint` path mode 는 `psd_analysis.py` 와 동일하다.

### 4.1 file mode

1. 입력 file suffix 는 `.pt` 여야 한다.
2. 해당 checkpoint 하나만 분석한다.
3. output 은 `<output_root>/<checkpoint_stem>/` 아래에 쓴다.

### 4.2 strict directory mode

1. directory 바로 아래에는 `.pt` file 만 있어야 한다.
2. subdirectory 는 invalid 다.
3. non-`.pt` regular file 은 invalid 다.
4. checkpoint epoch metadata 가 있으면 epoch 오름차순으로 분석한다.
5. checkpoint 중 하나라도 epoch metadata 가 없으면 lexical order 를 사용하고 manifest 에 warning 을 기록한다.

## 5. probe scope

이 program 은 train/test full split 전체를 분석하지 않는다. prepared data 에서 model-family selected training view 를 구성한 뒤 seed 기반 probe subset 만 사용한다. 정적 이미지 dataset 의 MLP 분석은 flatten training view 를 사용한다.

필수 scope:

| scope 형식 | 의미 |
| --- | --- |
| `<split>_balanced_global` | label 별 동일 quota 로 구성한 balanced probe set |
| `<split>_label_single_label_<c>` | label `c` 에서 deterministic 하게 선택한 sample 한 장 |

`<split>` 은 `train` 또는 `test` 다.

Balanced scope 는 sample 별 2-D FFT power 를 먼저 계산한 뒤 sample axis 로 평균내어 하나의 matrix 를 저장한다. Label-single scope 는 같은 split 의 같은 label 후보 중 `<split>_balanced_global` 에 이미 포함된 index 를 제외한 뒤 deterministic 하게 선택한 sample 한 장에 대한 matrix 를 저장한다.

## 6. 분석 대상 signal

이 program 은 MLP probe input map 과 MLP layer output map 을 분석한다. 아래 series 의 tensor 가 capture 결과에 존재할 때 저장한다.

저장 대상 official series:

| signal_kind | series | 의미 |
| --- | --- | --- |
| `input` | `x_probe` | model 에 투입되는 prepared probe input trace |
| `hidden` | `membrane` | hidden layer membrane output trace |
| `hidden` | `spike` | hidden layer spike output trace |
| `output` | `membrane` | output layer membrane trace |
| `output` | `spike` | output layer spike trace |

아래 series 는 이 program 의 official output 이 아니다.

1. `layer_input`
2. `readout_mem`
3. `i_current`
4. `z_gate`
5. 기타 auxiliary trace

## 7. 2-D FFT 정의

각 sample 의 input 또는 layer output map 을 아래와 같이 둔다.

$$
X \in \mathbb{R}^{N \times T}
$$

여기서 $N$ 은 input element 수 또는 layer output neuron 수이고 $T$ 는 time step 수다. Input/layer row index 는 실제 input vector 또는 layer output vector 구조의 0-based order 를 따른다.

Raw variant 는 $X$ 를 그대로 사용한다. Centered variant 는 map 전체 평균을 제거한다.

$$
X_{centered}=X-\frac{1}{NT}\sum_{i=0}^{N-1}\sum_{t=0}^{T-1}X_{i,t}
$$

2-D FFT power matrix 는 아래처럼 계산한다.

$$
P=\left|\operatorname{fftshift}\left(\operatorname{fft2}(X)\right)\right|^2
$$

Balanced scope 에서는 sample 별 $P$ 를 계산한 뒤 sample axis 평균을 취한다.

$$
\bar{P}=\frac{1}{B}\sum_{b=1}^{B}P_b
$$

## 8. variant 와 scale

모든 output coordinate 에 대해 아래 조합을 저장한다.

| axis | values |
| --- | --- |
| `variant` | `raw`, `centered` |
| `scale` | `raw`, `db` |

`scale=db` 는 power matrix 에 대해 $10\log_{10}(P+\epsilon)$ 를 적용한 값이다. $\epsilon$ 은 implementation 의 `DB_EPSILON` 을 따른다.

## 9. CSV category

2-D FFT output 은 `category=analysis_2d_fft` 를 사용한다.

한 CSV 는 하나의 input/layer, scope, label condition, signal, series, variant, scale 조합만 담는다.

행의 의미:

| column | 의미 |
| --- | --- |
| `row_frequency_index` | input/layer row-axis FFT frequency row index |
| `row_frequency` | `fftshift(fftfreq(N))` 로 계산한 input/layer row-axis frequency |
| `time_freq_000000`, ... | time-axis FFT frequency column 별 power 값 |

Dynamic value column 은 `time_freq_<index>` 형식이다. Time frequency 값은 `time_frequency_grid=fftshift_fftfreq_time_index` 와 `time_length` 로 복원한다.

## 10. output path contract

권장 layout:

```text
<output_root>/
  analysis_manifest.csv
  checkpoint_epoch_000001/
    layers/
      layer_000__input/
        analysis_2d_fft/
          analysis_2d_fft__epoch_1__layer_0__input__train_balanced_global__all_labels__input__x_probe__balanced_mean__raw__raw.csv
      layer_001__layer_01/
        analysis_2d_fft/
          analysis_2d_fft__epoch_1__layer_1__layer_01__train_balanced_global__all_labels__hidden__membrane__balanced_mean__raw__raw.csv
```

규칙:

1. category column 이 없는 CSV 를 쓰지 않는다.
2. binary bundle 을 쓰지 않는다.
3. figure file 을 쓰지 않는다.
4. 한 CSV 안에 서로 다른 category 를 섞지 않는다.
5. `analysis_manifest.csv` 에 생성 artifact 를 기록한다.

## 11. bash launcher contract

`bash/2d_fft_analysis.sh` 는 이 program 의 official bash launcher 다.

필수 launcher contract:

1. `bash/2d_fft_analysis.sh` 는 `src.2d_fft_analysis` 만 호출한다.
2. checkpoint input 은 `CHECKPOINT_SET` 또는 `CHECKPOINT_SET_RAW` 로 받는다.
3. checkpoint grouping 은 `CHECKPOINTS_PER_JOB` 으로 제어한다.
4. GPU assignment 는 `GPU_INDEX_SET` 으로 제어한다.
5. `DATASET`, `PREP_ROOT`, `OUTPUT_ROOT`, `ANAL_BATCH`, `LOW_VRAM`, `SEED`, `NUM_WORKERS` 는 같은 이름의 Python argument 로만 mapping 한다.
6. log directory 는 `<LOG_ROOT>/2d_fft_analysis/<RUN_STAMP>` 이다.
7. child job 은 `nohup` background process 로 실행하고 launcher 는 종료를 기다리지 않는다.
8. launcher 내부에 child 종료 대기 기반 queue 또는 동시 실행 수 제한을 두지 않는다.

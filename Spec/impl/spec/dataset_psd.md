# dataset_psd 구현 계약

## 1. 문서 역할

`src/dataset_psd.py` 는 dataset-level input/probe spectral baseline 을 담당한다. model training, checkpoint analysis, plotting 과 분리한다.

## 2. 책임

이 program 은 다음을 수행해야 한다.

1. `--prep_root` 와 `--dataset` 으로 prepared data bundle 을 resolve 한다.
2. dataset/probe policy 에 따라 fixed probe subset 을 선택한다.
3. axis metadata 에서 logical PSD view 를 구성한다.
4. input/probe PSD baseline artifact 를 계산한다.
5. category-based CSV output 을 쓴다.
6. GPU 를 사용하는 계산이면 `--gpu_index` 로 device 를 선택한다.
7. sample 처리는 `--batch_size` 단위로 streaming 한다.

이 program 은 다음을 수행하면 안 된다.

1. model 을 train 하지 않는다.
2. model checkpoint 를 load 하지 않는다.
3. model/layer signal analysis 를 실행하지 않는다.
4. figure 를 render 하지 않는다.
5. train/test 중 하나만 고르는 train/test selector 를 노출하지 않는다.
6. FFT-length control, scale-filter control, curve-filter control, prepared-data direct-path interface 를 노출하지 않는다.

## 3. official CLI

| argument | required | 의미 |
| --- | --- | --- |
| `--dataset` | yes | canonical dataset token |
| `--prep_root` | yes | prepared dataset 들을 담은 root directory |
| `--output_root` | yes | baseline CSV 용 output root |
| `--batch_size` | yes | PSD 계산용 sample streaming batch size |
| `--gpu_index` | yes | dataset PSD 계산용 CUDA device index |
| `--seed` | yes | probe reference seed |
| `--num_workers` | optional | data loading workers |

현재 dataset PSD path 는 `psd_exact` 만 산출한다.

규칙:

1. `psd_userbin` 분석은 수행하지 않는다.
2. `psd_userbin` CSV 는 생성하지 않는다.
3. `--userbin_edges` 는 official CLI 에 포함하지 않는다. Python parser 에 호환성 인수로 남아 있더라도 실행 경로에서는 무시한다.
4. exact PSD 는 normalized frequency grid 를 그대로 사용한다.
5. 별도 FFT 길이 조정 개념은 official dataset PSD path 에 존재하지 않는다.

## 4. scope 처리 정책

`dataset_psd.py` 는 train/test split 선택 인수를 받지 않는다. 대신 prepared bundle 안의 공식 data scope 를 모두 처리한다.

필수 scope:

1. `train_full`: prepared train split 전체.
2. `test_full`: prepared test split 전체.
3. `same_label_label_<c>`: class label 별 same-label probe set.
4. `balanced_global`: class-balanced global probe set.
5. `distribution_global`: empirical-distribution global probe set.


## 5. output contract

모든 dataset baseline output 은 `Spec/impl/spec/csv_schema.md` 의 category-based schema 를 따른다.

필수 category:

| category | 내용 |
| --- | --- |
| `dataset_curve` | input/probe PSD representative curve. 각 조합은 `raw` 와 `db` scale file 을 모두 가진다. |
| `dataset_dispersion` | PSD-domain variance, MAD, quantile summary. 각 조합은 `raw` 와 `db` scale file 을 모두 가진다. |
| `dataset_psd_manifest` | 생성된 CSV file inventory 와 validation status |

권장 layout:

```text
<output_root>/
  dataset_psd_manifest.csv
  train_full/
    dataset_curve__train_full__input__psd_exact__mean__raw__raw.csv
    dataset_curve__train_full__input__psd_exact__mean__raw__db.csv
  test_full/
    ...
  probe_sets/
    train_same_label_label_00/
      indices.json
      dataset_curve__train_same_label_label_00__input__psd_exact__mean__raw__raw.csv
      ...
    train_balanced_global/
      indices.json
      ...
    train_distribution_global/
      indices.json
      ...
```

중요한 규칙:

1. 단일 `input_psd_curve.csv` 에 모든 curve 를 몰아넣지 않는다.
2. 각 full scope 또는 probe set scope, 각 extractor, reducer, variant, scale 조합마다 독립 CSV 를 쓴다.
3. 파일 하나의 `category` 는 하나여야 한다.
4. CSV 안에는 plot 재구성에 필요한 frequency, bin, value, value unit 을 직접 기록한다.
5. Figure 는 생성된 CSV file 에서 `src/plotting.py` 가 생산한다.

## 6. axis metadata

program 은 fixed physical tensor layout 을 요구하면 안 된다. prepared data metadata 에서 PSD logical axis 를 해석한다.

- sample axis
- batch axis
- time axis
- row axes
- feature axes
- token axes
- flatten rule
- logical PSD shape

axis metadata 가 없거나 모호하면 PSD 계산 전에 fail 한다.

## 7. batching 과 device

`--batch_size` 는 PSD 계산 과정에서 한 번에 읽고 누적하는 sample 수의 upper bound 다.

규칙:

1. `batch_size >= 1` 이어야 한다.
2. full scope 가 batch size 보다 크면 여러 microbatch 로 나눠 누적한다.
3. microbatch output 은 같은 category CSV 조합으로 누적된다.
4. `--gpu_index` 가 주어지면 계산 tensor 와 accumulator tensor 의 device 이동 규칙이 명확해야 한다.
5. CUDA 를 요청했는데 사용할 수 없으면 fail 한다. implicit CPU fallback 은 official behavior 가 아니다.

## 8. checkpoint analysis 와의 관계

`dataset_psd.py` 는 dataset-level baseline 을 쓴다. `psd_analysis.py` 는 이 baseline 을 읽거나 같은 metadata 로 probe reference 를 재생성할 수 있지만, baseline 생성을 소유하지 않는다.

## 2026-05-01 구현 보정: exact-only, dB 후처리 기준

Dataset-level PSD baseline도 checkpoint PSD 분석과 같은 기준을 사용한다. `psd_exact` 만 산출하고 `psd_userbin` 계열 분석, 저장, CSV 산출은 비활성화한다.

`scale=db` 는 raw power domain에서 필요한 대표 곡선 또는 dispersion 연산을 완료한 뒤 마지막에 dB 변환을 적용한다. 따라서 대표 PSD는 `dB(mean(P))` 기준이고, dispersion은 raw domain의 variance 또는 MAD를 먼저 계산한 뒤 그 결과를 dB로 변환한다.


## 2026-05-01 보정: exact-only PSD 산출

현재 구현은 PSD 분석 산출에서 `psd_exact` 만 생성한다. `psd_userbin` 분석, 저장, CSV 산출, plot 산출은 비활성화한다. bash 실행 스크립트는 `USERBIN_EDGES` 를 전달하지 않으며, Python CLI 의 `--userbin_edges` 는 과거 호환성용으로만 남아 있고 분석 경로에서는 무시된다.

## 2026-05-02 수정 고정

- rank-3 tensor 에 대해 긴 축을 time 으로 추정하는 휴리스틱을 사용하지 않는다.
- dataset PSD 변환은 manifest 의 `psd_axis_kind`, `psd_time_axis`, `psd_row_axes`, `psd_flatten_rule`, `psd_logical_shape` 를 기준으로 명시적으로 수행한다.
- 변환 결과는 항상 `(B,rows,time)` 이며, manifest 의 expected rows/time 과 불일치하면 에러를 낸다.
- CSV 에 preprocessing 및 PSD axis metadata 를 반드시 기록한다.

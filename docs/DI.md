# DI 독립 실행 기능 설명

## 배치 위치

프로젝트 루트 기준으로 아래처럼 배치하면 된다.

```text
psd/
  src/
    DI.py
  config/
    DI.yaml
  bash/
    DI.sh
  docs/DI.md
  paper/discriminative_index/
```

`src/DI.py`는 프로젝트의 `prep_data` 산출물을 직접 읽고, DI CSV/PNG 산출물을 저장하는 독립 실행 stage이다.

## 실행 방법

프로젝트 루트에서 실행한다.

```bash
bash bash/DI.sh config/DI.yaml
```

또는 직접 실행한다.

```bash
python -m src.DI --config config/DI.yaml
```

주요 override 예시는 아래와 같다.

```bash
python -m src.DI \
  --config config/DI.yaml \
  --dataset s-mnist \
  --split train \
  --batch_size 512 \
  --gpu_index 0
```

`config/DI.yaml`의 `prep_root`는 `data_prep` 산출물의 루트이다. 즉 내부에 `<dataset>/manifest.yaml`, `<dataset>/train*.npy`, `<dataset>/test*.npy`가 있어야 한다.

## 산출물

`output_root` 아래에 timestamp 폴더가 생성된다. dataset이 여러 개이면 dataset별 하위 폴더가 생성된다.

```text
<output_root>/<timestamp>__di/<dataset>/
  csv/
    DI__<dataset>__train__dft_magnitude__raw.csv
    DI__<dataset>__train__dft_magnitude__norm.csv
    DI__<dataset>__train__project_psd__none__<raw|db|area>__raw.csv
    DI__<dataset>__train__project_psd__none__<raw|db|area>__norm.csv
    DI__<dataset>__train__project_psd__hann__<raw|db|area>__raw.csv
    DI__<dataset>__train__project_psd__hann__<raw|db|area>__norm.csv
  plot/
    위 CSV 각각에 대응하는 .png
  DI_manifest.yaml
  DI_summary.csv
  DI_resolved_config.csv
```

단일 dataset만 지정하면 `<dataset>` 하위 폴더 없이 바로 `csv/`, `plot/`이 생성된다.

## 계산하는 지표

각 frequency bin $k$에 대해 sample별 scalar feature $z_i[k]$를 만든 뒤, ground-truth label 기준으로 Fisher-style DI를 계산한다.

$$
DI(\omega_k)=\frac{S_B[k]}{S_W[k]+\epsilon}
$$

여기서

$$
S_B[k]=\sum_c \pi_c(\mu_c[k]-\bar{\mu}[k])^2
$$

$$
S_W[k]=\sum_c \pi_c \mathrm{Var}_c[k]
$$

이다. `DI_norm`은 전체 주파수 bin에서 합이 1이 되도록 정규화한 값이다.

$$
DI_{norm}(\omega_k)=\frac{DI(\omega_k)}{\sum_{k'}DI(\omega_{k'})}
$$

## 구현한 feature 종류

### 1. DFT magnitude DI

`maps`를 `(sample, row, time)` 형태로 만든 뒤 row 평균으로 sample별 1D temporal scalar sequence를 만든다.

$$
s_i[t]=\frac{1}{R}\sum_{r=1}^{R}x_i[r,t]
$$

`demean=true`이면 sample별 temporal mean을 제거한다. 이후 one-sided DFT coefficient magnitude를 feature로 쓴다.

$$
z_i[k]=|\mathrm{rFFT}(s_i)[k]|
$$

이 feature로 raw DI와 normalized DI를 저장한다.

### 2. Project PSD DI

프로젝트의 `exact_periodogram_from_maps`와 같은 정의를 `DI.py` 안에 독립 구현했다.

$$
P_i[r,k]=\frac{a_k |\mathrm{rFFT}(w \odot x_i[r,:])[k]|^2}{L \cdot \frac{1}{L}\sum_t w[t]^2}
$$

여기서 $a_k$는 one-sided scaling이다. DC와 Nyquist bin은 1, 나머지 one-sided 내부 bin은 2를 쓴다.

row 방향은 기본적으로 mean reduce한다.

$$
z_i[k]=\frac{1}{R}\sum_{r=1}^{R}P_i[r,k]
$$

`psd_windows`에 따라 `none`, `hann` 두 버전을 계산한다. `psd_value_transform`은 `raw`, `db`, `area` 중 하나이며, `area`는 sample별 PSD feature를 주파수축 합이 1이 되도록 정규화한 뒤 DI를 계산한다. 각각 raw DI와 normalized DI를 저장한다.

## GPU 사용 방식

데이터 로딩은 prepared `.npy` 파일을 CPU에서 읽는다. 그 이후 아래 연산은 가능한 한 GPU에서 수행한다.

- `(B, row, time)` maps 변환 후 CUDA 이동
- demeaning
- `torch.fft.rfft`
- DFT magnitude 계산
- 프로젝트 PSD 계산
- class별 sum, sumsq, count 누적
- $S_B$, $S_W$, DI, DI normalization 계산

즉 전체 sample feature matrix를 CPU 메모리에 쌓지 않고, batch 단위로 GPU에서 충분통계만 누적한다.

## 관찰 의도

이 기능의 목적은 model을 거치지 않고 dataset 자체의 주파수별 label separability를 보는 것이다.

- `dft_magnitude` DI: row 평균 1D 신호만 보았을 때 어느 시간 주파수 bin이 class label을 잘 분리하는지 확인한다.
- `project_psd__none` DI: 프로젝트의 periodogram 정의를 쓰되 rectangular window로 본다.
- `project_psd__hann` DI: Hann window를 적용했을 때 leakage 완화 후 discriminative bin 분포가 유지되는지 본다.
- normalized DI: raw DI의 절대 scale을 제거하고 주파수별 discriminative mass 분포로 본다.

따라서 normalized DI에서 특정 low/mid/high frequency band에 mass가 몰리면, 그 대역이 dataset-level class separability에 기여할 가능성이 높다고 해석할 수 있다. 단, 이것은 실제 classifier accuracy가 아니라 label을 사용한 1D Fisher separability proxy이다.

## 설정 항목

| 항목 | 의미 |
|---|---|
| `dataset` | 분석할 dataset token. 배열 가능 |
| `prep_root` | `data_prep` 산출물 루트 |
| `output_root` | CSV/PNG 저장 루트 |
| `split` | `train`, `test`, `all`. 기본은 leakage 방지를 위해 `train` |
| `batch_size` | 분석 batch size |
| `gpu_index` | 사용할 CUDA device index |
| `device` | `cuda` 또는 `cpu` |
| `allow_cpu_fallback` | CUDA가 없을 때 CPU 실행 허용 여부 |
| `max_samples` | 디버그용 sample 수 제한. 기본 `null` |
| `view_name` | 사용할 prepared view. 기본 `null`이면 manifest의 `psd_view_name` 사용 |
| `demean` | sample별 temporal mean 제거 여부 |
| `epsilon` | DI 분모 안정화 항 |
| `psd_windows` | PSD에 사용할 window 목록. 기본 `none`, `hann` |
| `psd_value_transform` | PSD feature transform: `raw`, `db`, `area` |
| `psd_row_reducer` | row별 PSD를 sample feature로 줄이는 방법. `mean` 또는 `median` |
| `psd_value_transform` | PSD feature 변환. 기본 `raw`, 선택적으로 `db` |
| `timestamped_output` | 실행마다 timestamp 폴더 생성 여부 |
| `plot_log_y` | raw DI plot을 log-y로 볼지 여부 |
| `stats_dtype` | class별 sum/sumsq/count 누적 및 DI 계산 dtype. 기본 `float64`; FFT/PSD feature 계산은 GPU `float32` |


## 점검 및 수정 이력

이번 점검에서 아래 사항을 수정했다.

1. `manifest.yaml` 검증을 강화했다. `storage_format`, `dataset_name`, `split_internal_order_preserved`, PSD axis metadata가 현재 프로젝트 `data_prep` 명세와 맞지 않으면 즉시 중단한다.
2. label-to-class position 변환을 CPU list 변환 대신 GPU `torch.searchsorted` 기반으로 바꿨다. class별 통계 누적 전 과정이 더 GPU 친화적으로 동작한다.
3. class별 충분통계 `count/sum/sumsq` 및 DI 계산 dtype을 `stats_dtype`으로 분리했다. 기본값은 `float64`라서 많은 sample을 누적할 때 float32 cancellation 위험을 줄인다. FFT와 PSD feature 계산은 GPU `float32`로 유지한다.
4. rank-3 tensor의 `(B, row, time)` / `(B, time, row)` 판정을 현재 프로젝트의 PSD axis metadata와 더 일치하도록 보강했다.
5. package에는 `src/DI.py`, `config/DI.yaml`, `bash/DI.sh`, `docs/DI.md`를 포함한다. `__pycache__`, 테스트 산출물, 프로젝트 원본 파일은 포함하지 않는다.

## 해석 시 주의점

1. DI는 unsupervised clustering index가 아니라 label을 직접 쓰는 supervised separability score이다.
2. `dft_magnitude`는 row 평균 scalar sequence를 사용하므로 channel/spatial 구조와 phase 정보는 사라진다.
3. `project_psd`는 row별 PSD를 평균 또는 median으로 줄이므로 row별 spatial specificity를 직접 보존하지 않는다.
4. normalized DI는 주파수별 상대적 mass를 보기 좋게 만들지만, raw DI 총량 정보는 사라진다.
5. 따라서 raw DI와 normalized DI를 함께 보는 것이 좋다.

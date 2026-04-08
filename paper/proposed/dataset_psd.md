# dataset_psd 실험 명세서

## 1. 목적

`dataset_psd` 는 학습을 수행하지 않고 지원 benchmark dataset 의 train / test split 입력 자체의 PSD / spectrogram 기준선을 저장하는 one-shot 실험이다. 이 실험의 수학적 정의와 저장 규칙은 `paper/proposed/psd_analysis.md` 와 같아야 하며, 차이는 분석 대상이 모델 내부 layer 가 아니라 **데이터셋별 author-code-aligned 전처리 이후의 model input tensor** 라는 점뿐이다. 또한 deterministic probe set input reference 저장 책임은 이제 `dataset_psd` 가 가진다.

현재 공식 dataset 범위는 아래 다섯 개다.

- `s-mnist`
- `dvsgesture`
- `shd`
- `deap`
- `forda`

전처리 기준은 `src/common/datasets.py` 의 benchmark adapter 구현과 `Origin/` 아래 released code 를 따른다. 즉 s-MNIST 는 DH-SNN 계열 정규화와 raster sequence, DVS128 Gesture 는 First-spike 계열 HDF5 -> dense tensor, SHD 는 현행 event binning, DEAP 는 DH-SNN 계열 baseline 제거 3초 segment, FordA 는 TS-LIF 계열 `.ts` loader 와 standardization 을 기준으로 한다.

## 2. dataset별 입력 기준선

입력 sample $n$ 의 전처리 후 model input 을

$$
X^{(n)} \in \mathbb{R}^{C \times T}
$$

로 둔다. 여기서 $C$ 는 입력 element 수, $T$ 는 시간축 길이다. 일부 loader 는 sample 을 `(T, C)` 형태로 반환하지만, PSD 계산 직전에는 항상 channel-major map $C \times T$ 로 정규화해 처리한다. 입력 element index $i$ 의 시계열은

$$
s_i^{(n)}[t], \qquad t=0,1,\dots,T-1
$$

로 둔다.

현재 공식 benchmark 별 기준 shape 는 아래와 같다.

| dataset | 전처리 기준 | PSD 계산 직전 기준 $C \times T$ | 비고 |
| --- | --- | --- | --- |
| `s-mnist` | DH-SNN 계열 `ToTensor()` 후 `[0, 1]` 유지, `28 x 28 -> 784 x 1` 순차화 | $1 \times 784$ | 입력은 scalar sequence |
| `dvsgesture` | First-spike 계열 HDF5 -> dense tensor | $(2 \cdot (128/ds)^2) \times (\text{chunk_size}+\text{empty_size})$ | 기본값은 $2048 \times 160$ |
| `shd` | 현행 SHD event binning | $700 \times 250$ | 표준 설정 기준 |
| `deap` | DH-SNN 계열 baseline 제거 + 3초 segment | $32 \times 384$ | EEG 32채널 |
| `forda` | TS-LIF 계열 `.ts` loader + standardization | $1 \times 500$ | 표준 FordA 길이 기준 |

따라서 `dataset_psd` 는 더 이상 SHD 전용 실험이 아니며, 각 dataset 의 전처리 이후 입력 기준선 위에서 동일한 exact PSD / spectrogram 규칙을 적용해야 한다.

## 3. 모든 주파수 단위

이 실험에서 사용하는 모든 주파수 단위는 Nyquist 상한이 0.5 인 cycle/sample 로 고정한다.

$$
f \in [0, 0.5]
$$

따라서 waveform, heatmap, spectrogram, userbin 중심값은 모두 cycle/sample 단위를 써야 한다.

## 4. exact periodogram waveform

입력 channel $i$ 의 시계열에 대해 raw signal 과 centered signal 을 각각

$$
x_i^{(n),\mathrm{raw}}[t] = s_i^{(n)}[t]
$$

$$
x_i^{(n),\mathrm{centered}}[t] = s_i^{(n)}[t] - \frac{1}{T}\sum_{\tau=0}^{T-1} s_i^{(n)}[\tau]
$$

로 둔다. $a \in \{\mathrm{raw}, \mathrm{centered}\}$ 에 대해 full-length one-sided DFT 계수를

$$
X_i^{(n),a}[k] = \sum_{t=0}^{T-1} x_i^{(n),a}[t] \, e^{-j 2\pi kt/T}, \qquad k=0,1,\dots,\left\lfloor \frac{T}{2} \right\rfloor
$$

로 두면 exact periodogram 은

$$
P_i^{(n),a}[k] = \frac{1}{T}\left|X_i^{(n),a}[k]\right|^2
$$

이다. 이 경로에서는 Hann, Hamming, Blackman 같은 taper window 를 적용하지 않는다.

split $q \in \{\mathrm{train}, \mathrm{test}\}$ 의 sample 집합을 $\mathcal{S}_q$ 라 두면 channel 평균 후 sample 평균한 waveform 은

$$
\bar P_q^{\mathrm{wave},a}[k] = \frac{1}{|\mathcal{S}_q|}\sum_{n \in \mathcal{S}_q} \left( \frac{1}{C}\sum_{i=0}^{C-1} P_i^{(n),a}[k] \right)
$$

이다.

저장 파일명은 아래 네 개다.

- 선형: `mean_psd_waveform_exact_raw.png`, `mean_psd_waveform_exact_centered.png`
- dB y축: $10 \log_{10}(\bar P_q^{\mathrm{wave},a}[k] + \varepsilon)$, 여기서 $\varepsilon = 10^{-12}$
- dB: `mean_psd_waveform_exact_raw_db.png`, `mean_psd_waveform_exact_centered_db.png`

## 5. periodogram userbin heatmap

periodogram heatmap 은 raw / centered 두 버전을 모두 저장한다. userbin index 를 $b$ 라 하고 해당 bin 에 속한 exact frequency index 집합을 $\mathcal{B}_b$ 라 두면

$$
\widetilde P_i^{(n),a}[b] = \frac{1}{|\mathcal{B}_b|} \sum_{k \in \mathcal{B}_b} P_i^{(n),a}[k]
$$

이다.

split 평균 heatmap 값은

$$
H_{q}^{\mathrm{psd},a}[i,b] = \frac{1}{|\mathcal{S}_q|}\sum_{n \in \mathcal{S}_q} \widetilde P_i^{(n),a}[b]
$$

이다.

heatmap 규칙은 아래와 같다.

- x축: userbin 중심 주파수, unit = cycle/sample
- y축: channel index
- 아래쪽 row 가 낮은 index 이다
- 모든 칸 안에 수치 annotation 을 넣는다
- 큰 canvas 를 사용한다

저장 파일명은 아래 네 개다.

- 선형: `element_psd_heatmap_userbin_raw.png`, `element_psd_heatmap_userbin_centered.png`
- dB 각 칸 값: $10 \log_{10}(H_{q}^{\mathrm{psd},a}[i,b] + \varepsilon)$, 여기서 $\varepsilon = 10^{-12}$
- dB: `element_psd_heatmap_userbin_raw_db.png`, `element_psd_heatmap_userbin_centered_db.png`

## 6. exact mean spectrogram

spectrogram 은 Welch 가 아니라 시간이동 simple periodogram 이다. frame 길이를 $L$, frame $u$ 의 시작 index 를 $t_u$ 라 두면 frame-local raw / centered signal 은 각각

$$
r_i^{(n),\mathrm{raw}}[m] = s_i^{(n)}[t_u + m], \qquad m=0,1,\dots,L-1
$$

$$
r_i^{(n),\mathrm{centered}}[m] = s_i^{(n)}[t_u + m] - \frac{1}{L}\sum_{\tau=0}^{L-1} s_i^{(n)}[t_u + \tau]
$$

이다. $a \in \{\mathrm{raw}, \mathrm{centered}\}$ 에 대한 exact sliding simple periodogram spectrogram 은

$$
S_i^{(n),a}[k,u] = \frac{1}{L}\left| \sum_{m=0}^{L-1} r_i^{(n),a}[m] \, e^{-j 2\pi km/L} \right|^2
$$

이다. 이 경로 역시 frame 내부 taper window 를 적용하지 않는다.

split 평균 spectrogram 은

$$
\bar S_q^{a}[k,u] = \frac{1}{|\mathcal{S}_q|}\sum_{n \in \mathcal{S}_q} \left( \frac{1}{C}\sum_{i=0}^{C-1} S_i^{(n),a}[k,u] \right)
$$

이다.

저장 파일명은 아래 네 개다.

- 선형: `mean_spectrogram_exact_raw.png`, `mean_spectrogram_exact_centered.png`
- dB 값: $10 \log_{10}(\bar S_q^{a}[k,u] + \varepsilon)$, 여기서 $\varepsilon = 10^{-12}$
- dB: `mean_spectrogram_exact_raw_db.png`, `mean_spectrogram_exact_centered_db.png`

## 7. spectrogram userbin heatmap

spectrogram 도 periodogram 과 같은 원칙을 따른다. exact mean spectrogram 은 raw / centered exact 로 각각 저장하고, channel heatmap 은 raw / centered 두 버전을 모두 저장하며 두 버전 모두에 userbin 을 적용한다.

$$
\widetilde S_i^{(n),a}[b,u] = \frac{1}{|\mathcal{B}_b|} \sum_{k \in \mathcal{B}_b} S_i^{(n),a}[k,u]
$$

split 평균 channel heatmap 값은

$$
H_q^{\mathrm{spec},a}[i,b,u] = \frac{1}{|\mathcal{S}_q|}\sum_{n \in \mathcal{S}_q} \widetilde S_i^{(n),a}[b,u]
$$

이다.

저장 그림은 row 별 $(u,b)$ 쌍을 frame-major 순서로 펼친 2차원 heatmap 이다. 즉 열 순서는

$$
(u_0,b_0),(u_0,b_1),\dots,(u_0,b_{B-1}),(u_1,b_0),\dots
$$

이다.

heatmap 규칙은 아래와 같다.

- x축: frame center / frequency userbin 의 frame-major 열 순서
- y축: channel index
- 아래쪽 row 가 낮은 index 이다
- 수치 annotation 은 넣지 않는다

저장 파일명은 아래 네 개다.

- 선형: `element_spectrogram_heatmap_userbin_raw.png`, `element_spectrogram_heatmap_userbin_centered.png`
- dB 각 칸 값: $10 \log_{10}(H_q^{\mathrm{spec},a}[i,b,u] + \varepsilon)$, 여기서 $\varepsilon = 10^{-12}$
- dB: `element_spectrogram_heatmap_userbin_raw_db.png`, `element_spectrogram_heatmap_userbin_centered_db.png`

## 8. 저장 구조

각 split bundle 은 아래 구조를 따른다.

```text
<run_root>/
  config.json
  summary.json
  train/
    mean_psd_waveform_exact_raw.png
    mean_psd_waveform_exact_centered.png
    element_psd_heatmap_userbin_raw.png
    element_psd_heatmap_userbin_centered.png
    mean_spectrogram_exact_raw.png
    mean_spectrogram_exact_centered.png
    element_spectrogram_heatmap_userbin_raw.png
    element_spectrogram_heatmap_userbin_centered.png
    mean_psd_waveform_exact_raw_db.png
    mean_psd_waveform_exact_centered_db.png
    element_psd_heatmap_userbin_raw_db.png
    element_psd_heatmap_userbin_centered_db.png
    mean_spectrogram_exact_raw_db.png
    mean_spectrogram_exact_centered_db.png
    element_spectrogram_heatmap_userbin_raw_db.png
    element_spectrogram_heatmap_userbin_centered_db.png
    summary.json
  test/
    mean_psd_waveform_exact_raw.png
    mean_psd_waveform_exact_centered.png
    element_psd_heatmap_userbin_raw.png
    element_psd_heatmap_userbin_centered.png
    mean_spectrogram_exact_raw.png
    mean_spectrogram_exact_centered.png
    element_spectrogram_heatmap_userbin_raw.png
    element_spectrogram_heatmap_userbin_centered.png
    mean_psd_waveform_exact_raw_db.png
    mean_psd_waveform_exact_centered_db.png
    element_psd_heatmap_userbin_raw_db.png
    element_psd_heatmap_userbin_centered_db.png
    mean_spectrogram_exact_raw_db.png
    mean_spectrogram_exact_centered_db.png
    element_spectrogram_heatmap_userbin_raw_db.png
    element_spectrogram_heatmap_userbin_centered_db.png
    summary.json
```

split-level `summary.json` 은 raw / centered 두 계열의 공통 metadata 와 scalar summary 를 함께 담는다. 현재 공식 저장에서는 `db_plots_saved = true`, `db_plot_scale = "10log10_power_plus_epsilon"`, `db_plot_epsilon = 1.0e-12` 도 함께 기록한다.

## 9. CLI / config 규칙

공식 엔트리는 `src/dataset_psd/run.py` 다. `src/dataset_psd/SHD/run.py` 는 기존 SHD 전용 호출을 위한 compatibility wrapper 로만 둔다. `probe_set_reference/` 저장도 이 엔트리에서 담당한다. 주요 의미는 아래와 같다.

- `--dataset` 은 단일 인수다.
- 공식 canonical dataset 이름은 `s-mnist`, `dvsgesture`, `shd`, `deap`, `forda` 다.
- `--same_label_n_per_label`, `--balanced_global_n_per_label` 는 deterministic probe scope prefix 길이를 정의한다.
- `--probe_plot` 이 true 이면 `probe_set_reference/<split>/<scope>/input/` 아래에 입력 reference bundle 을 저장한다.
- `--data_root`, `--out_root` 는 dataset 공통 절대경로다.
- `--shd_*` 인수는 SHD 계열 event binning 설정만 바꾼다.
- `--dvsgesture_chunk_size`, `--dvsgesture_empty_size`, `--dvsgesture_dt_ms`, `--dvsgesture_ds` 는 DVS128 Gesture dense tensor 기준선을 정한다.
- `--deap_label_axis`, `--deap_num_classes` 는 DEAP label 과제 설정을 정한다.
- `--psd_window`, `--psd_overlap` 은 spectrogram frame 길이 / overlap 을 정한다.
- `--userbin_edges` 는 periodogram / spectrogram heatmap userbin 경계를 정한다.
- `--window_fn` 은 **legacy compatibility 인수** 이며 현재 공식 exact 경로에서는 무시된다.
- `config.json` 에는 최소 `dataset_name`, `dataset_bundle`, `periodogram_length_effective`, `spectrogram_window_effective`, `spectrogram_overlap_effective`, `userbin_edges`, `taper_window_applied = false`, `variants_saved = ["raw", "centered"]`, `save_db_psd_plots = true`, `db_plot_scale = "10log10_power_plus_epsilon"`, `db_plot_epsilon = 1.0e-12` 를 기록한다.


## 10. 외부 데이터 루트 구조

공식 `data_root` 는 아래 구조를 기준으로 맞춘다.

```text
<data_root>/
  MNIST/
    raw/
      train-images-idx3-ubyte.gz
      train-labels-idx1-ubyte.gz
      t10k-images-idx3-ubyte.gz
      t10k-labels-idx1-ubyte.gz
  DVS128Gesture/
    hdf5/
      DVS-Gesture-train10.hdf5
      DVS-Gesture-test10.hdf5
    raw/
      ... optional raw .aedat / *_labels.csv backups ...
  SHD/
    shd_train.h5
    shd_test.h5
  DEAP/
    data_preprocessed_python/
      s01.dat
      s02.dat
      ...
  FordA/
    FordA_TRAIN.ts
    FordA_TEST.ts
```

설명은 아래와 같다.

- `s-mnist` 는 `MNIST/` 아래 raw IDX gzip 또는 torchvision 호환 구조를 사용한다. 현재 구현은 `[0, 1]` 범위를 유지한 뒤 `28 x 28 -> 784 x 1` flatten과 scalar 실수 직주입을 사용한다.
- `dvsgesture` 는 학습용 dense loader 가 `DVS128Gesture/hdf5/DVS-Gesture-train10.hdf5`, `DVS-Gesture-test10.hdf5` 를 직접 읽는다.
- `shd` 는 `SHD/shd_train.h5`, `SHD/shd_test.h5` 를 사용한다.
- `deap` 는 공식 preprocessed package 인 `DEAP/data_preprocessed_python/` 아래 subject별 `.dat` 파일을 사용한다.
- `forda` 는 `FordA/FordA_TRAIN.ts`, `FordA/FordA_TEST.ts` 를 사용한다.


## 11. 금지 사항

- waveform 또는 mean spectrogram 을 userbin 으로 저장하면 안 된다.
- spectrogram heatmap 을 exact bin heatmap 으로 저장하면 안 된다.
- exact periodogram 또는 exact spectrogram 경로에 taper window 를 적용하면 안 된다.
- raw 계열 또는 centered 계열 중 하나만 저장하면 안 된다.
- `dataset_psd` 를 SHD 전용 실험으로 가정하면 안 된다.
- dataset별 입력 길이 $T$ 와 element 수 $C$ 를 임의로 SHD 기준선으로 덮어쓰면 안 된다.


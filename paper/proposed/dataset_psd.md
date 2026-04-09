# dataset_psd 실험 명세서

## 1. 목적

`dataset_psd` 는 학습을 수행하지 않고 현재 실험에서 사용하는 dataset 입력 자체에 대한 PSD / spectrogram 기준선을 저장하는 one-shot 실험이다. 이 실험의 수학적 정의는 `paper/proposed/psd_analysis.md` 와 같고, 차이는 분석 대상이 모델 내부 layer 가 아니라 전처리 이후의 model input tensor 라는 점뿐이다.

사용 dataset 목록, dataset 별 전처리 출처, PSD 계산 직전 입력 shape 기준, 다운로드 / 준비 방법은 `paper/proposed/data_preprocessing.md` 를 따른다. deterministic probe set input reference 의 영구 저장 책임도 `dataset_psd` 가 가진다. 즉 `psd_analysis` 는 같은 deterministic index 를 공유하지만, 입력 probe-set plot 자체는 `dataset_psd` 가 저장한다.

이 문서는 dataset 이름별 예외를 열거하지 않고, dataset-agnostic 저장 규칙과 probe set / PSD 계산 / CLI target 선택만 정의한다.

공식 plot 대상은 CLI `--plot_target` 으로 선택한다.

- `dataset` : full train / test split 입력 bundle 만 저장한다.
- `probe_set` : deterministic probe_set 입력 reference bundle 만 저장한다.
- `both` : 같은 run root 아래에서 full dataset bundle 과 probe_set bundle 을 직렬로 둘 다 저장한다.

## 2. probe_set

`dataset_psd` 의 probe_set 은 `psd_analysis` 가 사용하는 deterministic subset 과 동일한 정의를 공유한다. train split, test split 각각에 대해 `same_label` 과 `balanced_global` 두 종류를 사용한다.

- `same_label` : 각 label 안에서 deterministic canonical order 의 prefix 를 취한 probe set
- `balanced_global` : 각 label 의 같은 canonical order prefix 를 취한 뒤 label 순서대로 flatten 한 probe set

canonical order 는 split 이름, base seed, label, dataset index 에만 의존해야 하며, model 시나리오, readout mode, timestamp, out_root 같은 실행 외부 조건 때문에 바뀌면 안 된다. 같은 label 내부에서는 `same_label` 과 `balanced_global` 이 같은 canonical order 를 공유하고, 두 scope 는 서로 다른 prefix 길이만 가질 수 있다.

입력 reference bundle 저장 경로는 아래와 같다.

- `probe_set_reference/<split>/same_label/label_<c>/input/`
- `probe_set_reference/<split>/balanced_global/input/`

여기서 저장되는 것은 입력 tensor 기준의 PSD / spectrogram bundle 이다. hidden / output layer bundle 이 아니며, epoch 축도 없다. `psd_analysis` 는 같은 index 를 공유해 내부 reference payload 를 계산하지만, 입력 probe-set plot 을 다시 저장하지 않는다.

## 3. 모든 주파수 단위

이 실험에서 사용하는 모든 주파수 단위는 Nyquist 상한이 0.5 인 cycle/sample 로 고정한다.

$$
f \in [0, 0.5]
$$

따라서 waveform, heatmap, spectrogram, userbin 중심값은 모두 cycle/sample 단위를 써야 한다.

## 4. exact periodogram waveform

입력 sample $n$ 의 전처리 후 model input 을

$$
X^{(n)} \in \mathbb{R}^{C \times T}
$$

로 둔다. 일부 loader 는 sample 을 `(T, C)` 형태로 반환하지만, PSD 계산 직전에는 항상 channel-major map $C \times T$ 로 정규화해 처리한다. 입력 element index $i$ 의 시계열은

$$
s_i^{(n)}[t], \qquad t=0,1,\dots,T-1
$$

로 둔다.

이하에서 scope $\mathcal{S}$ 는 두 경우 중 하나를 의미한다.

1. full dataset bundle 일 때는 split sample 집합 `train` 또는 `test`
2. probe_set bundle 일 때는 선택된 deterministic probe subset

따라서 sections 4-7 의 수식은 full dataset bundle 과 probe_set bundle 에 공통으로 적용되며, 차이는 $\mathcal{S}$ 가 전체 split 인지 deterministic subset 인지뿐이다.

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

scope 평균 waveform 은

$$
\bar P_{\mathcal{S}}^{\mathrm{wave},a}[k] = \frac{1}{|\mathcal{S}|}\sum_{n \in \mathcal{S}} \left( \frac{1}{C}\sum_{i=0}^{C-1} P_i^{(n),a}[k] \right)
$$

이다.

저장 파일명은 아래 네 개다.

- 선형: `mean_psd_waveform_exact_raw.png`, `mean_psd_waveform_exact_centered.png`
- dB y축: $10 \log_{10}(\bar P_{\mathcal{S}}^{\mathrm{wave},a}[k] + \varepsilon)$, 여기서 $\varepsilon = 10^{-12}$
- dB: `mean_psd_waveform_exact_raw_db.png`, `mean_psd_waveform_exact_centered_db.png`

## 5. periodogram userbin heatmap

periodogram heatmap 은 raw / centered 두 버전을 모두 저장한다. userbin index 를 $b$ 라 하고 해당 bin 에 속한 exact frequency index 집합을 $\mathcal{B}_b$ 라 두면

$$
\widetilde P_i^{(n),a}[b] = \frac{1}{|\mathcal{B}_b|} \sum_{k \in \mathcal{B}_b} P_i^{(n),a}[k]
$$

이다.

scope 평균 heatmap 값은

$$
H_{\mathcal{S}}^{\mathrm{psd},a}[i,b] = \frac{1}{|\mathcal{S}|}\sum_{n \in \mathcal{S}} \widetilde P_i^{(n),a}[b]
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
- dB 각 칸 값: $10 \log_{10}(H_{\mathcal{S}}^{\mathrm{psd},a}[i,b] + \varepsilon)$, 여기서 $\varepsilon = 10^{-12}$
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

scope 평균 spectrogram 은

$$
\bar S_{\mathcal{S}}^{a}[k,u] = \frac{1}{|\mathcal{S}|}\sum_{n \in \mathcal{S}} \left( \frac{1}{C}\sum_{i=0}^{C-1} S_i^{(n),a}[k,u] \right)
$$

이다.

저장 파일명은 아래 네 개다.

- 선형: `mean_spectrogram_exact_raw.png`, `mean_spectrogram_exact_centered.png`
- dB 값: $10 \log_{10}(\bar S_{\mathcal{S}}^{a}[k,u] + \varepsilon)$, 여기서 $\varepsilon = 10^{-12}$
- dB: `mean_spectrogram_exact_raw_db.png`, `mean_spectrogram_exact_centered_db.png`

## 7. spectrogram userbin heatmap

spectrogram 도 periodogram 과 같은 원칙을 따른다. exact mean spectrogram 은 raw / centered exact 로 각각 저장하고, channel heatmap 은 raw / centered 두 버전을 모두 저장하며 두 버전 모두에 userbin 을 적용한다.

$$
\widetilde S_i^{(n),a}[b,u] = \frac{1}{|\mathcal{B}_b|} \sum_{k \in \mathcal{B}_b} S_i^{(n),a}[k,u]
$$

scope 평균 channel heatmap 값은

$$
H_{\mathcal{S}}^{\mathrm{spec},a}[i,b,u] = \frac{1}{|\mathcal{S}|}\sum_{n \in \mathcal{S}} \widetilde S_i^{(n),a}[b,u]
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
- dB 각 칸 값: $10 \log_{10}(H_{\mathcal{S}}^{\mathrm{spec},a}[i,b,u] + \varepsilon)$, 여기서 $\varepsilon = 10^{-12}$
- dB: `element_spectrogram_heatmap_userbin_raw_db.png`, `element_spectrogram_heatmap_userbin_centered_db.png`

## 8. 저장 구조

run root 는 `--plot_target` 에 따라 아래 항목을 조건부로 가진다.

```text
<run_root>/
  config.json
  summary.json
  train/
    ... 16개 PNG + summary.json ...
  test/
    ... 16개 PNG + summary.json ...
  probe_set_reference/
    train/
      same_label/
        label_<c>/
          input/
            ... 16개 PNG + summary.json ...
      balanced_global/
        input/
          ... 16개 PNG + summary.json ...
    test/
      ... 동일 구조 ...
```

규칙은 아래와 같다.

- `train/`, `test/` 는 `plot_target` 이 `dataset` 또는 `both` 일 때만 생성한다.
- `probe_set_reference/` 는 `plot_target` 이 `probe_set` 또는 `both` 일 때만 생성한다.
- full dataset bundle 과 probe_set bundle 은 모두 동일한 16개 PSD / spectrogram PNG 집합과 `summary.json` 을 가진다.
- run-level `summary.json` 은 어떤 target 이 실제로 저장되었는지와 dataset / probe_set 저장 결과를 함께 요약해야 한다.
- split-level `summary.json` 은 raw / centered 두 계열의 공통 metadata 와 scalar summary 를 함께 담는다. 현재 공식 저장에서는 `db_plots_saved = true`, `db_plot_scale = "10log10_power_plus_epsilon"`, `db_plot_epsilon = 1.0e-12` 도 함께 기록한다.

## 9. CLI / config 규칙

공식 엔트리는 `src/dataset_psd.py` 다. 문서 기준 공식 인터페이스는 `--plot_target` 이고, 기존 `--probe_plot` 은 backward compatibility alias 로만 둔다.

- `--dataset` 은 단일 인수다. 유효한 canonical dataset token 집합은 현재 실험 범위와 동일하며 `paper/proposed/data_preprocessing.md` 를 따른다.
- `--plot_target` 의 공식 선택지는 `dataset`, `probe_set`, `both` 다.
- `dataset` 은 full train / test split bundle 만 저장한다.
- `probe_set` 은 deterministic probe-set input reference bundle 만 저장한다.
- `both` 는 full dataset bundle 과 probe_set bundle 을 같은 run root 아래 직렬로 모두 저장한다.
- `--seed`, `--same_label_n_per_label`, `--balanced_global_n_per_label` 는 deterministic probe scope 를 정의한다.
- `--probe_plot` legacy 호출에서는 `false -> dataset`, `true -> both` 로만 해석하고, `probe_set` 단독 저장은 `--plot_target probe_set` 으로 명시한다.
- `--data_root`, `--out_root` 는 절대경로다.
- dataset-specific loader / preprocessing 보조 인수가 존재하더라도, 그 의미와 허용 범위는 `paper/proposed/data_preprocessing.md` 와 대응 adapter 구현을 따른다. 본 문서는 dataset 이름별 옵션 목록을 별도로 나열하지 않는다.
- `--psd_window`, `--psd_overlap` 은 spectrogram frame 길이 / overlap 을 정한다.
- `--userbin_edges` 는 periodogram / spectrogram heatmap userbin 경계를 정한다.
- `--window_fn` 은 legacy compatibility 인수이며 현재 공식 exact 경로에서는 무시된다.
- `config.json` 에는 최소 `dataset_name`, `plot_target`, `dataset_bundle_saved`, `probe_set_reference_saved`, `seed`, `same_label_n_per_label`, `balanced_global_n_per_label`, `periodogram_length_effective`, `spectrogram_window_effective`, `spectrogram_overlap_effective`, `userbin_edges`, `taper_window_applied = false`, `variants_saved = ["raw", "centered"]`, `save_db_psd_plots = true`, `db_plot_scale = "10log10_power_plus_epsilon"`, `db_plot_epsilon = 1.0e-12` 를 기록한다.

## 10. 외부 데이터 루트 구조

공식 `data_root` 는 각 dataset adapter 가 raw / preprocessed asset 을 상대경로로 찾는 공통 절대 루트다. dataset 별 하위 디렉터리 이름, 원본 파일명, 사전 가공 파일 배치는 `paper/proposed/data_preprocessing.md` 와 대응 adapter 구현을 따른다. 본 문서가 강제하는 최소 형태는 아래와 같다.

```text
<data_root>/
  <dataset-specific-subdir>/
    raw/
      ...
    preprocessed/
      ...
```

위 그림의 `raw/`, `preprocessed/` 는 예시적 placeholder 다. 모든 dataset 이 두 디렉터리를 모두 가져야 한다는 뜻은 아니다.

설명은 아래와 같다.

- `--data_root` 는 dataset 별 자산을 한곳에 모으는 공통 절대 루트다.
- 각 adapter 는 자기 dataset 이 요구하는 파일을 이 루트 아래에서 resolve 해야 한다.
- `dataset_psd` 는 dataset 이름에 따라 PSD 수학, 저장 파일 집합, probe_set 규칙을 바꾸지 않는다. 달라질 수 있는 것은 전처리 결과로 정해지는 입력 shape 와 sample 집합뿐이며, 이는 `paper/proposed/data_preprocessing.md` 를 따른다.

## 11. 금지 사항

- waveform 또는 mean spectrogram 을 userbin 으로 저장하면 안 된다.
- spectrogram heatmap 을 exact bin heatmap 으로 저장하면 안 된다.
- exact periodogram 또는 exact spectrogram 경로에 taper window 를 적용하면 안 된다.
- raw 계열 또는 centered 계열 중 하나만 저장하면 안 된다.
- `dataset_psd` 를 특정 dataset 전용 실험으로 가정하면 안 된다.
- 어떤 dataset 의 입력 길이 $T$ 와 element 수 $C$ 도 다른 dataset 기준선으로 임의 덮어쓰면 안 된다.
- probe_set input bundle 을 hidden / output layer bundle 과 같은 산출물로 혼동하면 안 된다.

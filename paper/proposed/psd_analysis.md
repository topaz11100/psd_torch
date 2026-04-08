# PSD analysis specification

## 1. 목적

이 문서는 지원 benchmark dataset 기반 `psd_analysis` 실험의 공식 저장 규칙을 정의한다. 현재 공식 dataset 범위는 `s-mnist`, `dvsgesture`, `shd`, `deap`, `forda` 다. 전처리 기준은 `src/common/datasets.py` 의 benchmark adapter 구현과 `Origin/` 아래 released code 를 우선하며, `data_root` 구조는 `paper/proposed/dataset_psd.md` 의 외부 데이터 루트 규칙과 동일하다.

이 실험은 split / scope 별 고정 probe set 을 기준으로 학습 중 각 epoch 종료 직후 같은 probe set 에 대한 hidden layer 와 output layer 시계열의 PSD / spectrogram 수치 payload 와 부가 통계를 계산한다. 입력 probe reference 의 영구 저장 책임은 `dataset_psd` 가 가지며, `psd_analysis` 는 동일한 deterministic probe batch 로부터 내부 reference payload 만 계산해 derivative semi-metric 등에 사용한다. hidden / output signal PSD bundle 은 기본적으로 모든 epoch 저장이지만, `--plot_epoch`(alias `--plot_epochs`) 가 주어지면 지정한 epoch 에서만 epoch 디렉터리를 만들고 저장한다. 선택된 epoch 디렉터리에는 probe-set accuracy, attenuation 통계, hidden-layer incoming-weight density plot(`w_plot.png`) 도 함께 저장한다. `clip`, `structure`, `clip_structure` 시나리오에서는 hidden-layer 전체 PSD bundle 을 유지하면서 같은 hidden layer 아래 block PSD bundle / block weight plot 을 추가 저장한다. 또한 각 selected epoch 의 각 layer / family 디렉터리에는 시간영역 heatmap 과 element-mean line plot 을 함께 저장한다. 다만 학습 중에는 PNG 를 직접 그리지 않고 plot 생성용 numeric payload 를 process-local CPU 메모리에만 홀드하며, 학습 완료 후 tqdm 진행 로그를 남기며 일괄 렌더링하고 성공적으로 처리한 payload 는 즉시 메모리에서 제거한다. 학습 완료 시 accuracy plot, 최종 probe-set accuracy, 최종 attenuation 통계, derivative semi-metric 요약은 `training_complete_stats/` 아래에 별도로 저장한다.

핵심 원칙은 아래와 같다.

1. waveform 은 **window 없이 계산한 exact simple periodogram** 이며 raw / centered 둘 다 저장한다.
2. periodogram heatmap 은 **raw / centered 두 버전을 모두 저장하며, 각 버전은 exact periodogram 에 userbin 집계를 적용한 element heatmap** 이다.
3. mean spectrogram 은 **window 없이 계산한 exact sliding simple periodogram** 이며 raw / centered 둘 다 저장한다.
4. spectrogram heatmap 은 **raw / centered 두 버전을 모두 저장하며, 각 버전은 exact spectrogram 에 userbin 집계를 적용한 element heatmap** 이다.
5. output layer 는 실제 neuron layer 이며 output neuron 뒤에 learned NN head 를 두지 않는다. 단 `final_membrane` 에서만 output layer 의 spike emission 과 spike-triggered reset path 를 비활성화한다.

## 2. 분석 대상 신호

단일 sample $n$ 에 대해 입력 시계열을

$$
X^{(n)} \in \mathbb{R}^{C_0 \times T}
$$

로 둔다. hidden layer 또는 output layer $\ell$ 의 요소 index $i$ 에 대응하는 막전위 또는 spike 시계열을

$$
s^{(n)}_{\ell,i}[t], \qquad t=0,1,\dots,T-1
$$

로 둔다.

저장 대상 계열은 다음 다섯 가지다.

1. probe set input reference
2. hidden layer membrane
3. hidden layer spike
4. output layer membrane
5. output layer spike

입력 probe set 은 epoch 와 무관한 고정 집합이며, 영구 저장 경로는 `paper/proposed/dataset_psd.md` 가 정의한다. `psd_analysis` 는 같은 deterministic probe set index 를 공유하지만 입력 reference PNG 를 직접 저장하지 않는다. 대신 같은 probe batch 에서 계산한 입력 reference payload 를 내부적으로 유지해 hidden/output layer 변화량 비교에만 사용한다. `psd_analysis` 의 hidden/output signal PSD bundle PNG 생성은 deferred render pass 에서 수행한다. 즉 학습 중에는 numeric payload 만 process-local CPU 메모리에 홀드하고, 학습 완료 후 렌더링한다. hidden layer 와 output layer 는 모델 파라미터가 epoch 마다 달라지므로 기본적으로 모든 epoch 저장이지만, `--plot_epoch`(alias `--plot_epochs`) 가 지정되면 해당 epoch 에서만 epoch 디렉터리를 만들고 signal PSD bundle 과 시간영역 plot 을 저장한다. `final_membrane` 실행에서도 output layer spike bundle 저장 규칙은 유지하되, 이 경우 output spike tensor 는 정의상 all-zero 가 된다.

`--plot_epoch` 는 **epoch 디렉터리 / signal PSD bundle 저장 epoch 선택자** 다. 예를 들어 `--plot_epoch 1 2 5 10 100` 이면 `epoch_0001`, `epoch_0002`, `epoch_0005`, `epoch_0010`, `epoch_0100` 에서만 hidden/output signal PSD bundle 최종 PNG, `summary.json`, `probe_set_accuracy.txt`, `attenuation_stats/`, `all_layers_summary.csv`, hidden-layer `w_plot.png`, `time_domain_heatmap.png`, `time_domain_element_mean.png` 가 생긴다. grouped(`clip`, `structure`, `clip_structure`) hidden layer 는 같은 선택 epoch 에서 `hidden_n/block/block_k/` 아래 block PSD bundle 과 block weight plot 도 함께 저장한다. 학습 loop 안에서는 해당 PNG 대신 numeric payload 만 process-local CPU 메모리에 적재하고, 학습이 끝난 뒤 tqdm 로그와 함께 최종 PNG 를 렌더링한다. 학습 완료 시점의 `train_test_accuracy.csv`, 최종 `train_test_accuracy.png`, 최종 probe-set accuracy, 최종 attenuation 통계는 이 선택과 독립적으로 `training_complete_stats/` 아래에 별도로 저장한다.

## 3. 모든 주파수 단위

이 실험에서 사용하는 모든 주파수 단위는 Nyquist 상한이 0.5 인 cycle/sample 로 고정한다.

$$
f \in [0, 0.5]
$$

따라서 waveform, heatmap, spectrogram, RF clip 경계, RF 통계의 공명주파수 축은 모두 cycle/sample 단위를 써야 한다.

## 3.1 LIF / RF 동역학 학습 파라미터 초기화

본 실험의 free / clip LIF, RF 동역학 학습 파라미터는 fan-based Kaiming/Xavier 초기화를 쓰지 않는다. 공식 정책은 **유효 물리 파라미터 구간에서의 bounded uniform sampling** 이다.

- free LIF: 각 neuron 의 $\alpha$ 를 유효 열린구간 $(0, 1)$ 안에서 bounded uniform sampling 으로 초기화한다.
- lif_clip / lif_structclip: 위와 같은 bounded-uniform 규칙을 유지하되, 각 neuron 이 배정받은 `alpha_clip_edges` 구간 안으로 support 를 좁혀 초기화한다.
- free RF: 각 neuron 의 normalized resonance frequency $f$ 는 유효 Nyquist band $(0, 0.5)$ 안에서 bounded uniform sampling 으로 초기화하고, damping magnitude $|b|$ 도 별도의 기본 bounded range 안에서 bounded uniform sampling 으로 초기화한다.
- rf_clip / rf_structclip: 위와 같은 bounded-uniform 규칙을 유지하되, resonance frequency $f$ 의 support 만 각 neuron 이 배정받은 `w_clip_edges` 구간 안으로 좁혀 초기화한다. 현재 public RF clip CLI 는 frequency 만 제한하므로 damping magnitude 는 free bounded-uniform 초기화를 유지한다.

clip 경계 끝점이 정확히 0, 1 또는 0.5 인 경우에도 raw unconstrained parameter 가 무한대로 가지 않도록 구현에서는 작은 수치적 trim 을 둔 열린구간에서 샘플링할 수 있다.

## 4. readout 규칙

readout 정의와 수식은 `paper/proposed/readout.md` 로 이관한다. `psd_analysis` 본문에서는 이후 PSD / spectrogram / 저장 규칙만 유지한다.


## 5. exact periodogram waveform

요소 $i$ 의 시계열 $s^{(n)}_{\ell,i}[t]$ 에 대해 raw signal 과 centered signal 을 각각

$$
x^{(n),\mathrm{raw}}_{\ell,i}[t] = s^{(n)}_{\ell,i}[t]
$$

$$
x^{(n),\mathrm{centered}}_{\ell,i}[t] = s^{(n)}_{\ell,i}[t] - \frac{1}{T}\sum_{\tau=0}^{T-1} s^{(n)}_{\ell,i}[\tau]
$$

로 둔다. 본 문서의 exact periodogram 은 taper window 를 쓰지 않는 full-length simple periodogram 이다. 즉, $a \in \{\mathrm{raw}, \mathrm{centered}\}$ 에 대해 full-length one-sided DFT 계수를

$$
X^{(n),a}_{\ell,i}[k] = \sum_{t=0}^{T-1} x^{(n),a}_{\ell,i}[t] \, e^{-j 2\pi kt/T}, \qquad k=0,1,\dots,\left\lfloor \frac{T}{2} \right\rfloor
$$

로 두면, exact periodogram 은

$$
P^{(n),a}_{\ell,i}[k] = \frac{1}{T}\left| X^{(n),a}_{\ell,i}[k] \right|^2
$$

이다. 즉 각 시계열에 대해 **raw 그대로** 와 **mean-centering 적용** 두 경우를 각각 계산하며, 두 경우 모두 **window 없이** one-sided rFFT power 를 저장한다. `psd_analysis` 의 exact periodogram 경로에서는 Hann, Hamming, Blackman 같은 taper window 를 사용하지 않는다.

각 bin $k$ 에 대응하는 주파수는

$$
f_k = \operatorname{rfftfreq}(T, d=1.0)[k]
$$

이며 단위는 cycle/sample 이다. 따라서 $f_k \in [0, 0.5]$ 이고, $T$ 가 홀수이면 가장 높은 exact bin 은 0.5 보다 약간 작을 수 있다.

probe set $\mathcal{S}$ 에 대해 요소 평균 후 sample 평균한 waveform 은

$$
\bar P_{\ell}^{\mathrm{wave},a}[k] = \frac{1}{|\mathcal{S}|} \sum_{n \in \mathcal{S}} \left( \frac{1}{N_{\ell}} \sum_{i=0}^{N_{\ell}-1} P^{(n),a}_{\ell,i}[k] \right)
$$

로 둔다. 여기서 $N_{\ell}$ 은 해당 layer 의 요소 수이다. 즉 `mean_psd_waveform_exact_raw.png` 와 `mean_psd_waveform_exact_centered.png` 는 각 sample 에서 요소별 exact periodogram 을 먼저 구한 뒤, layer 내부 요소 평균과 probe-set 평균을 순서대로 취한 1차원 PSD waveform 이다.

waveform plot 정의는 다음과 같다.

- x축: exact periodogram frequency bin 중심값 $f_k$, unit = cycle/sample
- 선형 y축: mean PSD
- dB y축: $10 \log_{10}(\bar P_{\ell}^{\mathrm{wave},a}[k] + \varepsilon)$, 여기서 $\varepsilon = 10^{-12}$
- 저장 파일명:
  - 선형: `mean_psd_waveform_exact_raw.png`, `mean_psd_waveform_exact_centered.png`
  - dB: `mean_psd_waveform_exact_raw_db.png`, `mean_psd_waveform_exact_centered_db.png`

## 6. periodogram userbin heatmap

heatmap 은 raw / centered 두 버전을 모두 그리고, 두 버전 모두 exact periodogram 에 userbin 집계를 적용한 결과를 사용한다. temporal userbin index 를 $b$ 라 하고 해당 bin 에 속한 exact frequency index 집합을 $\mathcal{B}_b$ 라 두면 variant $a \in \{\mathrm{raw}, \mathrm{centered}\}$ 에 대한 요소별 userbin PSD 는

$$
\widetilde P^{(n),a}_{\ell,i}[b] = \frac{1}{|\mathcal{B}_b|} \sum_{k \in \mathcal{B}_b} P^{(n),a}_{\ell,i}[k]
$$

로 둔다.

probe set $\mathcal{S}$ 에 대한 heatmap 값은

$$
H^{\mathrm{psd},a}_{\ell}[i,b] = \frac{1}{|\mathcal{S}|} \sum_{n \in \mathcal{S}} \widetilde P^{(n),a}_{\ell,i}[b]
$$

이다.

heatmap 정의는 다음과 같다.

- x축: userbin 중심 주파수, unit = cycle/sample
- y축: 요소 index
- 아래쪽 row 가 낮은 index 이다
- 선형 각 칸 값: $H^{\mathrm{psd},a}_{\ell}[i,b]$
- dB 각 칸 값: $10 \log_{10}(H^{\mathrm{psd},a}_{\ell}[i,b] + \varepsilon)$, 여기서 $\varepsilon = 10^{-12}$
- 색으로 값을 표현하고 **모든 칸 안에 수치 값** 을 기입한다
- 큰 canvas 를 사용한다
- 저장 파일명:
  - 선형: `element_psd_heatmap_userbin_raw.png`, `element_psd_heatmap_userbin_centered.png`
  - dB: `element_psd_heatmap_userbin_raw_db.png`, `element_psd_heatmap_userbin_centered_db.png`

## 7. exact mean spectrogram

spectrogram 은 Welch 가 아니라 시간이동 simple periodogram 으로 만든다. spectrogram frame 길이를 $L$, frame $u$ 의 시작 index 를 $t_u$ 라 두고, 요소 $i$ 의 frame-local raw signal 과 centered signal 을 각각

$$
r^{(n),\mathrm{raw}}_{\ell,i,u}[m] = s^{(n)}_{\ell,i}[t_u + m], \qquad m=0,1,\dots,L-1
$$

$$
r^{(n),\mathrm{centered}}_{\ell,i,u}[m] = s^{(n)}_{\ell,i}[t_u + m] - \frac{1}{L}\sum_{\tau=0}^{L-1} s^{(n)}_{\ell,i}[t_u + \tau], \qquad m=0,1,\dots,L-1
$$

로 둔다. $a \in \{\mathrm{raw}, \mathrm{centered}\}$ 에 대한 요소 $i$ 의 exact sliding simple periodogram spectrogram 을

$$
S^{(n),a}_{\ell,i}[k,u] = \frac{1}{L}\left| \sum_{m=0}^{L-1} r^{(n),a}_{\ell,i,u}[m] \, e^{-j 2\pi km/L} \right|^2
$$

로 둔다. 이 경로 역시 frame 내부 taper window 를 사용하지 않는다.

probe set 평균 spectrogram 은

$$
\bar S^{a}_{\ell}[k,u] = \frac{1}{|\mathcal{S}|} \sum_{n \in \mathcal{S}} \left( \frac{1}{N_{\ell}} \sum_{i=0}^{N_{\ell}-1} S^{(n),a}_{\ell,i}[k,u] \right)
$$

이다. 여기서 $u$ 는 sliding-window frame index 이다.

spectrogram plot 정의는 다음과 같다.

- x축: spectrogram frame center time step
- y축: exact frequency bin 중심값, unit = cycle/sample
- 선형 값: probe set 평균 spectrogram power
- dB 값: $10 \log_{10}(\bar S^{a}_{\ell}[k,u] + \varepsilon)$, 여기서 $\varepsilon = 10^{-12}$
- 저장 파일명:
  - 선형: `mean_spectrogram_exact_raw.png`, `mean_spectrogram_exact_centered.png`
  - dB: `mean_spectrogram_exact_raw_db.png`, `mean_spectrogram_exact_centered_db.png`

## 8. spectrogram userbin heatmap

spectrogram 도 periodogram 과 같은 원칙을 따른다. mean spectrogram 은 raw / centered exact 로 각각 저장하고, element heatmap 은 raw / centered 두 버전을 모두 저장하며 두 버전 모두에 userbin 을 적용한다.

spectrogram userbin 값을

$$
\widetilde S^{(n),a}_{\ell,i}[b,u] = \frac{1}{|\mathcal{B}_b|} \sum_{k \in \mathcal{B}_b} S^{(n),a}_{\ell,i}[k,u]
$$

로 둔다. probe set 평균 element spectrogram heatmap 값은

$$
H^{\mathrm{spec},a}_{\ell}[i,b,u] = \frac{1}{|\mathcal{S}|} \sum_{n \in \mathcal{S}} \widetilde S^{(n),a}_{\ell,i}[b,u]
$$

이다.

저장 그림은 row 별 $(u,b)$ 쌍을 frame-major 순서로 펼친 2차원 heatmap 이다. 즉 열 순서는

$$
(u_0,b_0),(u_0,b_1),\dots,(u_0,b_{B-1}),(u_1,b_0),\dots
$$

이다.

heatmap 정의는 다음과 같다.

- x축: frame center / frequency userbin 의 frame-major 열 순서
- y축: 요소 index
- 아래쪽 row 가 낮은 index 이다
- 선형 각 칸 값: $H^{\mathrm{spec},a}_{\ell}[i,b,u]$
- dB 각 칸 값: $10 \log_{10}(H^{\mathrm{spec},a}_{\ell}[i,b,u] + \varepsilon)$, 여기서 $\varepsilon = 10^{-12}$
- 색으로 값을 표현하고 **수치 annotation 은 넣지 않는다**
- 저장 파일명:
  - 선형: `element_spectrogram_heatmap_userbin_raw.png`, `element_spectrogram_heatmap_userbin_centered.png`
  - dB: `element_spectrogram_heatmap_userbin_raw_db.png`, `element_spectrogram_heatmap_userbin_centered_db.png`

## 9. probe set 규칙

train split, test split 각각에 대해 `same_label` 과 `balanced_global` 두 종류의 고정 probe set 을 사용한다.

이 규칙에서 중요한 점은 probe set 이 다른 실행 조건 때문에 바뀌면 안 된다는 것이다. 같은 split(train / test 같은 데이터 분할) 안에서 같은 seed 와 같은 샘플링 개수이면, 다른 모든 조건과 무관하게 선택되는 sample index 가 항상 같아야 한다.

필수 조건은 아래 두 가지다.

1. `same_label`: 같은 split, 같은 seed, 같은 `same_label_n_per_label` 이면 model 시나리오, readout mode, timestamp, out_root 같은 다른 모든 설정과 무관하게 항상 같은 sample index 집합이어야 한다.
2. `balanced_global`: 같은 split, 같은 seed, 같은 `balanced_global_n_per_label` 이면 model 시나리오, readout mode, timestamp, out_root 같은 다른 모든 설정과 무관하게 항상 같은 sample index 집합이어야 한다.

즉 핵심 불변식은 `same_label = same_label AND balanced_global = balanced_global` 이다. epoch 분석 bundle 의 prediction, PSD 산출물, probe-set accuracy 저장은 모두 이 고정 probe set 을 기준으로 해야 한다.

또한 `epochs = 0` 이면 학습 / epoch 분석은 수행하지 않는다.

## 10. RF / LIF 필터 통계

선택된 epoch 에 대해서는 hidden layer 와 output layer 전체를 대상으로 필터 통계를 해당 `epoch_<eeee>/attenuation_stats/` 아래 저장한다. 동시에 학습 완료 시점의 최종 attenuation snapshot 은 `training_complete_stats/attenuation_stats/` 아래에 항상 따로 저장한다. 따라서 선택 epoch attenuation snapshot 과 최종 attenuation snapshot 이 둘 다 존재할 수 있다.

- LIF 계열은 alpha 관련 감쇠 통계를 저장한다. LIF reset 은 임의 reset level 을 두지 않고 $v_{\mathrm{th}}$ 기반 subtractive soft reset 으로 고정한다.
- RF 계열은 raw parameter 가 아니라 $\rho$ 와 `f_cyc_per_sample` 를 저장 통계 대상으로 둔다.
- 선택된 epoch root 의 저장 파일은 summary statistic bar plot, raw parameter value histogram bar plot, `summary_stats.csv`, 그리고 `epoch_<eeee>/all_layers_summary.csv` 다. 최종 snapshot 은 `training_complete_stats/all_layers_summary.csv` 로도 유지한다. 구현은 backward compatibility 용으로 `training_complete_stats/attenuation_stats/all_layers_summary.csv` 복사본을 함께 둘 수 있다.
- value histogram 은 x축이 parameter 값, y축이 그 값 구간에 속하는 neuron count 인 histogram bar plot 이다.

## 11. accuracy 저장

accuracy 는 두 수준으로 저장한다.

1. 전체 train / test split accuracy 는 매 epoch 마다 `train_test_accuracy.csv` 에 누적 저장하고, run 종료 시 그 CSV 를 다시 읽어 `train_test_accuracy.png` 단일 plot 을 생성한다. 또한 학습 완료 시 `training_complete_stats/train_test_accuracy.csv` 와 `training_complete_stats/train_test_accuracy.png` 복사본도 둔다. 이 최종 accuracy plot 저장은 `--plot_epoch` 설정과 무관하다.
2. PSD 산출물에 대응하는 각 고정 probe set 에 대해서는 선택된 epoch 디렉터리마다 해당 subset 위의 분류 accuracy 를 `probe_set_accuracy.txt` 로 저장한다. 추가로 학습 완료 시 최종 model 기준 probe-set accuracy snapshot 을 `training_complete_stats/probe_set_accuracy/<split>/<scope>/probe_set_accuracy.txt` 아래에 항상 저장한다. 텍스트 파일에는 최소 `epoch`, `split`, `probe_type`, `label`(해당되는 경우), `correct`, `total`, `accuracy` 를 사람이 읽을 수 있는 형태로 기록한다.

## 12. 저장 구조

각 bundle 의 `summary.json` 은 raw / centered 두 계열의 공통 metadata 와 scalar summary 를 함께 담는다.

### 12.1 probe set reference

입력 probe reference bundle 의 영구 저장 경로와 파일 구조는 `paper/proposed/dataset_psd.md` 가 정의한다. `psd_analysis` 는 deterministic probe set 정의를 공유하지만, 해당 bundle 을 직접 저장하지 않는다.

hidden layer / output layer 의 membrane 과 spike bundle 은 기본적으로 모든 epoch 저장이지만, `--plot_epoch`(alias `--plot_epochs`) 가 주어지면 지정한 epoch 에서만 저장한다. 각 선택된 epoch root 에는 `probe_set_accuracy.txt`, `attenuation_stats/`, `all_layers_summary.csv`, hidden-layer `w_plot.png` 도 함께 저장한다. 각 layer / family 디렉터리에는 `time_domain_heatmap.png`, `time_domain_element_mean.png` 도 함께 저장한다. grouped(`clip`, `structure`, `clip_structure`) 시나리오에서는 같은 epoch root 의 hidden-layer 아래에 block PSD bundle 과 block weight plot 도 함께 저장한다. 단 학습 중에는 관련 PNG 를 직접 그리지 않고 numeric payload 만 저장하며, 선택 epoch 전체에 대한 최종 PNG 생성은 학습 완료 후 수행한다. 선택되지 않은 epoch 에 대해서는 `epoch_<eeee>/` 디렉터리 자체를 만들지 않는다.

```text
<run_root>/
  epoch_0001/
    all_layers_summary.csv
    attenuation_stats/
      layers/
        <layer_name>/
          summary_stats.csv
          *_stats_bar.png
          *_value_hist_bar.png
      model/
        summary_stats.csv
        *_stats_bar.png
        *_value_hist_bar.png
    hidden_1/
      w_plot.png
      block/
        block_1/
          w_plot.png
        block_2/
          w_plot.png
    hidden_2/
      w_plot.png
      block/
        block_1/
          w_plot.png
        block_2/
          w_plot.png
    train/
      same_label/
        label_<c>/
          probe_set_accuracy.txt
          hidden_1/
            spike/
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
            membrane/
              ... 동일 16개 PNG (선형 8개 + dB 8개) + summary.json ...
            block/
              block_1/
                spike/
                  ... 동일 16개 PNG (선형 8개 + dB 8개) + summary.json ...
                membrane/
                  ... 동일 16개 PNG (선형 8개 + dB 8개) + summary.json ...
              block_2/
                spike/
                  ... 동일 16개 PNG (선형 8개 + dB 8개) + summary.json ...
                membrane/
                  ... 동일 16개 PNG (선형 8개 + dB 8개) + summary.json ...
          ...
          output/
            spike/
              ... 동일 16개 PNG (선형 8개 + dB 8개) + summary.json ...
            membrane/
              ... 동일 16개 PNG (선형 8개 + dB 8개) + summary.json ...
      balanced_global/
        probe_set_accuracy.txt
        ... 동일 구조 ...
    test/
      ... 동일 구조 ...
```

### 12.3 derivative semi-metric

선택된 epoch 에 대해서는 입력 probe reference mean payload 와 hidden / output layer bundle 의 같은 family mean plot 사이에 Functional Data Analysis 계열 derivative semi-metric 을 추적한다. reference 는 같은 deterministic probe batch 에서 계산한 입력 mean payload 이고, 비교 대상은 같은 split / scope 의 `hidden_k/{membrane,spike}` 와 `output/{membrane,spike}` bundle 이다. output layer 도 반드시 포함한다.

1차원 mean waveform 계열에 대해서는

$$
d(u,v)=\sqrt{\sum_{k=0}^{K-2} \left( (u[k+1]-u[k])-(v[k+1]-v[k]) \right)^2}
$$

를 사용한다. 2차원 mean spectrogram 계열은 row 별로 같은 식을 적용하여

$$
d_i(U,V)=\sqrt{\sum_{k=0}^{K-2} \left( (U[i,k+1]-U[i,k])-(V[i,k+1]-V[i,k]) \right)^2}
$$

를 계산한다. 시간 간격은 단위 1 로 두므로 explicit $\Delta t$ 표기는 넣지 않는다.

공식 tracked plot 은 `element_*` 를 제외한 mean 계열 8개뿐이다.

- `mean_psd_waveform_exact_raw`
- `mean_psd_waveform_exact_centered`
- `mean_spectrogram_exact_raw`
- `mean_spectrogram_exact_centered`
- `mean_psd_waveform_exact_raw_db`
- `mean_psd_waveform_exact_centered_db`
- `mean_spectrogram_exact_raw_db`
- `mean_spectrogram_exact_centered_db`

`element_*` heatmap 은 layer 마다 element 수가 달라질 수 있으므로 derivative semi-metric 대상에서 제외한다.

저장 경로는 아래와 같다.

```text
<run_root>/
  training_complete_stats/
    derivative_semi_metric/
      summary.json
      <split>/
        <scope>/
          <layer_name>/
            <family>/
              mean_psd_waveform_exact_raw.png
              mean_psd_waveform_exact_raw.json
              ...
              mean_spectrogram_exact_raw.png
              mean_spectrogram_exact_raw.json
              ...
```

규칙은 아래와 같다.

- waveform 계열은 x축 = selected epoch, y축 = derivative semi-metric 값인 점-꺾은선 plot 으로 저장한다.
- mean spectrogram 계열은 y축 = spectrogram frequency bin, x축 = selected epoch 인 heatmap 으로 저장한다.
- `summary.json` 에는 derivative 정의, unit-step note, tracked metric 이름 목록을 기록한다.

### 12.4 training-complete statistics

학습 완료 시 아래 별도 폴더를 항상 생성한다.

```text
<run_root>/
  training_complete_stats/
    train_test_accuracy.csv
    train_test_accuracy.png
    all_layers_summary.csv
    probe_set_accuracy/
      train/
        same_label/
          label_<c>/
            probe_set_accuracy.txt
        balanced_global/
          probe_set_accuracy.txt
      test/
        ... 동일 구조 ...
    attenuation_stats/
      layers/
        <layer_name>/
          summary_stats.csv
          *_stats_bar.png
          *_value_hist_bar.png
      model/
        summary_stats.csv
        *_stats_bar.png
        *_value_hist_bar.png
      all_layers_summary.csv
```

### 12.5 run-level metadata

run root 에는 최소 아래 파일이 있어야 한다.

- `config.json`
- `train_test_accuracy.csv`
- `train_test_accuracy.png`
- `training_complete_stats/train_test_accuracy.csv`
- `training_complete_stats/train_test_accuracy.png`
- `training_complete_stats/all_layers_summary.csv`
- `training_complete_stats/derivative_semi_metric/` 아래 최종 derivative semi-metric plot / json
- 선택된 epoch root 의 `attenuation_stats/` / `all_layers_summary.csv` 와 `training_complete_stats/attenuation_stats/` 아래 최종 filter 통계 plot / CSV
- `training_complete_stats/probe_set_accuracy/` 아래 최종 probe set accuracy text
- 선택된 epoch 에 대한 epoch 분석 bundle, epoch attenuation 통계, hidden-layer weight visualization, probe set accuracy text

## 13. 금지 사항

- output neuron 뒤에 learned classifier head 를 두면 안 된다.
- waveform 또는 mean spectrogram 을 userbin 으로 저장하면 안 된다.
- spectrogram heatmap 을 exact bin heatmap 으로 저장하면 안 된다.
- exact periodogram 또는 exact spectrogram 경로에 taper window 를 적용하면 안 된다.
- raw 계열 또는 centered 계열 중 하나만 저장하면 안 된다.
- freq_ 실험 경로 의존성을 남기면 안 된다.

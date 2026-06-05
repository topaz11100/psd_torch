# Neuron Motivation Scenario

이 scenario의 목적은 PSD, cross-spectrum, PCA-mode PSD를 단순 분석 지표가 아니라 새 뉴런 모델이 가져야 할 동역학적 특성의 motivation으로 사용하는 것이다. 핵심 질문은 다음이다.

- 입력 이벤트/시계열의 spectral envelope가 hidden spike 또는 membrane dynamics에서 어떻게 보존, 왜곡, 압축되는가?
- 기존 LIF와 RF 계열이 같은 task loss를 최적화할 때 서로 다른 spectral bias를 보이는가?
- PSD regularization이 성능 저하 없이 입력-은닉 관계의 spectral contract를 안정화하는가?
- $pca\_dim\_per\_layer=1$ 과 $pca\_dim\_per\_layer \ge 2$ 를 같은 PCA regularizer의 저차원/다차원 관찰 모드로 볼 때, 새 뉴런 모델에 필요한 multi-mode dynamics가 보이는가?
- Hann taper를 켠 분석과 끈 분석이 결론을 바꾸는가? 결론이 Hann smoothing에 의존하면 뉴런 동역학 motivation으로 사용하기 어렵다.

## Config 대응 구조

각 case는 학습 config와 PSD analysis config가 1:1로 대응한다.

```text
config/neuron_motivation_scenario/train/<case>.yaml
config/neuron_motivation_scenario/psd_analysis/<case>.yaml
```

학습 산출물은 다음 경로로 맞춘다.

```text
/home/yongokhan/workspace/result/neuron_motivation_scenario/<case>/checkpoints
/home/yongokhan/workspace/result/neuron_motivation_scenario/<case>/metrics
```

분석 config는 같은 `<case>/checkpoints`를 `checkpoint`로 참조하고, 분석 결과를 `<case>/psd_analysis`에 쓴다.

## 실험 축

| case group | 비교 | 관찰할 것 | 의도 |
|---|---|---|---|
| `00_window_*` | `signal_window=hann` vs `signal_window=none` | PSD peak 위치, broadband floor, centered-dB curve distance 변화 | spectral conclusion이 taper artifact인지 점검 |
| `01_baseline_*` | LIF baseline vs RF baseline | 입력-은닉 PSD distance, layer별 spectral drift, spike/membrane spectral concentration | task-only 학습이 만드는 고유 spectral bias 확인 |
| `02_psd_regularized_*_pca1` | LIF/RF + scalar PCA PSD regularization | 1D dominant mode가 입력 spectral envelope와 얼마나 정렬되는지 | 단일 dominant mode만으로 충분한 뉴런 동역학인지 검증 |
| `03_psd_regularized_*_pca4` | LIF/RF + multi-mode PCA PSD regularization | mode 간 spectral 분담, cross-spectrum, layer-wise MIMO structure | 새 뉴런 모델이 가져야 할 multi-mode response 필요성 검증 |
| `04_shd_rf_pca4_nowindow` | event speech dataset, RF, no window | temporal event dataset에서 untapered PSD 기준으로도 regularized spectral contract가 유지되는지 | S-MNIST 외 시계열 이벤트 데이터셋으로 motivation 일반화 점검 |

## 추가 candidate-dynamics case

`05_candidate_drf4_hann`과 `06_candidate_drf4_shd_hann`은 `d_rf_4`를 별도 PSD regularizer 없이 학습해 branch 기반 RF 계열이 자연스럽게 multi-mode spectral response를 만드는지 보기 위한 case다. 이 결과는 PSD regularizer로 강제한 spectral contract와 모델 구조 자체가 만드는 spectral bias를 구분하는 기준으로 사용한다.

`dataset_psd/input_psd_hann.yaml`과 `dataset_psd/input_psd_none.yaml`은 모델 없이 입력 데이터 자체의 spectral envelope를 저장한다. 이후 model PSD 분석 결과는 이 input baseline을 기준으로 해석한다.


## 학습 규제 곡선 설정

학습 중 신호 규제, PSD representative 규제, PCA PSD 규제는 같은 곡선 설정을 공유한다.

```yaml
"signal_curve_space": "exact",
"signal_curve_scale": "db",
"signal_curve_centering": "centered",
"signal_curve_reducer": "mean",
"signal_curve_distance_metric": "centered_l2",
"signal_curve_userbin_edges": null,
"signal_curve_userbin_reducer": "mean"
```

`signal_curve_space`를 `userbin`으로 바꾸면 `signal_curve_userbin_edges`에 normalized frequency edge 배열을 명시한다. `width`/`count` 기반 자동 추론 인수는 public config에서 사용하지 않는다.

## 실행 예

DDP 학습은 compile cache를 실험별로 분리해서 실행한다.

```bash
bash/model_training_ddp.sh \
  --compile-cache-root /home/yongokhan/workspace/cache/torch_compile \
  --experiment-name neuron_motivation_scenario \
  config/neuron_motivation_scenario/train/02_psd_regularized_rf_pca1.yaml
```

분석은 대응되는 config를 사용한다.

```bash
python src/psd_analysis.py \
  --config config/neuron_motivation_scenario/psd_analysis/02_psd_regularized_rf_pca1.yaml
```

## 해석 기준

### 1. Hann/no-window 결론 안정성

`00_window_lif_hann`과 `00_window_lif_nowindow`에서 관찰되는 상대적 ordering이 유지되면, PSD 기반 주장은 window artifact에 덜 민감하다. 반대로 peak sharpness 또는 curve distance ordering이 크게 바뀌면, 이후 실험에서는 `signal_window=none` 결과를 함께 보고해야 한다.

### 2. 새 뉴런 모델 motivation

새 뉴런 모델이 단순히 task accuracy를 개선하는 수준을 넘어서려면 다음 특성을 motivation으로 제시할 수 있다.

- 입력의 저주파/고주파 에너지 분포를 무조건 보존하지 않고, task-relevant band는 보존하고 nuisance band는 감쇠한다.
- layer가 깊어질수록 centered PSD distance가 무작위로 커지지 않고, 특정 band에서 controllable spectral transformation을 보인다.
- scalar PCA mode 하나로 설명되지 않는 hidden dynamics가 있고, multi-mode PCA PSD 또는 cross-spectrum에서 구조적 분담이 나타난다.
- PSD regularization을 걸었을 때 accuracy가 유지되면서 spectral contract가 안정화된다.

### 3. PCA 1D/MIMO 해석

이 scenario에서는 PCA 1D와 MIMO를 별도 regularizer 개념으로 보지 않는다. `lambda_psd_pca_input / lambda_psd_pca_adjacent`는 하나이고, `pca_dim_per_layer`가 `[1]`이면 dominant scalar mode 관찰, `[4]`이면 multi-mode response 관찰로 해석한다.

## 권장 산출물

각 case에서 최소한 다음을 비교한다.

- train/test accuracy curve
- `train_regularization`, `train_psd_regularization`, `train_psd_pca` metric
- 입력 대비 hidden layer별 centered-dB PSD distance
- PCA mode별 PSD curve와 mode cross-spectrum
- Hann vs no-window에서 peak 위치와 curve distance ordering 변화

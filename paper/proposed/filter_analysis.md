# Filter Property / Attenuation Statistics 명세서

본 문서는 `src/util/psd_analysis_driver.py` 가 현재 저장하는 필터 관련 공식 산출물 범위를 정의한다. 상위 기준은 `paper/proposed/psd_analysis.md` 이며, 이는 기존 `psd_analysis.md` 의 `RF / LIF 필터 통계` 섹션을 모듈식으로 분리한 버전이다. 동역학 정의와 초기화는 `paper/proposed/vanila_rf.md`, `paper/proposed/vanila_lif.md` 를, clip / structure 시나리오는 `paper/proposed/vanila_scenario.md` 를 따른다.

## 0. 범위

현재 공식 구현 범위는 LIF / RF 계열의 선택 epoch attenuation / resonance summary statistics 및 raw parameter value histogram, 그리고 학습 완료 시점 최종 snapshot 이다.

- 적용 대상: `src/util/psd_analysis_driver.py` 가 조립하는 LIF / RF 기반 `psd_analysis` 모델
- 현재 비적용 대상: `my_R_DH_SNN`, `my_D_RF` 에 대한 exact transfer-function plot, tracked-neuron plot, epoch-trend plot
- 즉, 예전 `filter_property/` 전체 snapshot / trend 구조는 현재 공식 구현 범위가 아니며, binding requirement 도 아니다.

본 분석은 데이터 입력에 대한 forward 출력이 아니라 학습된 파라미터에서 직접 읽어낸 통계량을 다룬다. 모든 저장 경로는 사용자 지정 절대 결과 루트 아래 run 디렉터리에 생성된다.

## 1. 목적

필터 관련 기록의 목적은 선택된 epoch 과 학습 완료 시점에 hidden layer 와 output layer 의 시간상수 / 공명 파라미터를 사람이 바로 해석할 수 있는 스칼라 통계와 value distribution histogram 으로 남기는 데 있다.

- LIF 계열은 감쇠 계수 `alpha` 를 기록한다.
- RF 계열은 raw parameter 가 아니라 `rho`, `f_cyc_per_sample` 를 기록한다.
- output neuron 뒤 learned head 는 없으므로 output layer 도 실제 neuron layer 파라미터 통계로 기록한다.

주파수 단위는 `paper/proposed/psd_analysis.md` 와 동일하게 Nyquist 상한이 0.5 인 cycle/sample 이다.

## 2. 저장 대상 파라미터

### 2.1 LIF 계열

LIF 계열의 공식 저장 키는 아래 하나다.

- `alpha`

reset 규칙은 별도 `v_reset` 파라미터를 두지 않고 $v_{\mathrm{th}}$ 기반 subtractive soft reset 으로 고정한다.

### 2.2 RF 계열

RF 계열의 공식 저장 키는 아래 둘이다.

- `rho`
- `f_cyc_per_sample`

즉 raw `b`, raw `omega` 자체를 epoch 통계 CSV 이름으로 저장하지 않는다. clip 입력은 normalized frequency `[0, 0.5]` 로 받고, 내부에서만 각주파수로 변환한다.

## 3. 요약 통계 정의

각 저장 키마다 최소 아래 열을 기록한다.

- `count`
- `mean`
- `variance`
- `std`
- `min`
- `q25`
- `q50`
- `q75`
- `max`

summary statistic bar plot 은 원시 뉴런 배열을 모두 나열하는 그림이 아니라, 위 통계 중 핵심 집계량을 요약하는 그림이다. 현재 구현은 각 parameter 에 대해 `mean`, `variance`, `q25`, `q50`, `q75` 를 한 장의 bar plot 으로 저장한다.

raw parameter value histogram 은 parameter 값의 분포를 직접 보는 그림이다.

- x축: parameter value
- y축: neuron count
- 저장 파일명: `<parameter>_value_hist_bar.png`

## 4. 저장 구조

현재 공식 저장 구조는 아래와 같다.

```text
<run_root>/
  epoch_<eeee>/
    all_layers_summary.csv
    attenuation_stats/
      layers/
        hidden_1/
          alpha_stats_bar.png
          alpha_value_hist_bar.png
          summary_stats.csv
        hidden_2/
          rho_stats_bar.png
          rho_value_hist_bar.png
          f_cyc_per_sample_stats_bar.png
          f_cyc_per_sample_value_hist_bar.png
          summary_stats.csv
        output/
          ... 동일 규칙 ...
      model/
        alpha_stats_bar.png
        alpha_value_hist_bar.png
        rho_stats_bar.png
        rho_value_hist_bar.png
        f_cyc_per_sample_stats_bar.png
        f_cyc_per_sample_value_hist_bar.png
        summary_stats.csv
  training_complete_stats/
    all_layers_summary.csv
    attenuation_stats/
      layers/
        hidden_1/
          alpha_stats_bar.png
          alpha_value_hist_bar.png
          summary_stats.csv
        hidden_2/
          rho_stats_bar.png
          rho_value_hist_bar.png
          f_cyc_per_sample_stats_bar.png
          f_cyc_per_sample_value_hist_bar.png
          summary_stats.csv
        output/
          ... 동일 규칙 ...
      model/
        alpha_stats_bar.png
        alpha_value_hist_bar.png
        rho_stats_bar.png
        rho_value_hist_bar.png
        f_cyc_per_sample_stats_bar.png
        f_cyc_per_sample_value_hist_bar.png
        summary_stats.csv
      all_layers_summary.csv
```

규칙은 다음과 같다.

- `epoch_<eeee>/attenuation_stats/layers/<layer_name>/summary_stats.csv` 와 `training_complete_stats/attenuation_stats/layers/<layer_name>/summary_stats.csv` 는 각각 선택 epoch / 최종 snapshot 의 레이어 내부 뉴런 배열 집계 CSV 다.
- `epoch_<eeee>/attenuation_stats/model/summary_stats.csv` 와 `training_complete_stats/attenuation_stats/model/summary_stats.csv` 는 각각 선택 epoch / 최종 snapshot 의 모든 해당 layer 통계를 모델 전체 기준으로 합친 CSV 다.
- `*_value_hist_bar.png` 는 raw parameter vector 로부터 직접 만든 histogram bar plot 이다.
- `training_complete_stats/all_layers_summary.csv` 는 레이어별 / 모델별 집계 결과를 한 번에 훑기 위한 학습 완료 aggregate CSV 다.
- 구현은 backward compatibility 용으로 `training_complete_stats/attenuation_stats/all_layers_summary.csv` 복사본을 함께 둘 수 있다.
- `layer_name` 은 `hidden_1`, `hidden_2`, ..., `output` 형식을 따른다.
- `training_complete_stats/all_layers_summary.csv` 의 최소 열은 `scope`, `layer`, `parameter`, `count`, `mean`, `variance`, `std`, `min`, `q25`, `q50`, `q75`, `max` 다.

## 5. 저장 시점

필터 통계 저장 시점은 상위 `psd_analysis` 실험의 선택된 epoch 직후와 학습 완료 직후 다. 따라서 `--plot_epoch` 로 선택된 epoch 에는 `epoch_<eeee>/attenuation_stats/` 와 `epoch_<eeee>/all_layers_summary.csv` 를 남기고, 최종 accuracy CSV / plot 생성과 함께 `training_complete_stats/` 아래 최종 snapshot 도 남긴다. 단 선택 epoch 의 `summary_stats.csv` 와 `all_layers_summary.csv` 는 즉시 기록하고, 대응 PNG 는 학습 중 numeric payload 로만 저장되었다가 학습 완료 후 렌더링될 수 있다.

## 6. 현재 비범위 항목

아래 항목은 현재 공식 구현 범위가 아니다.

- `epoch_trend/filter_property/` 누적 추이 plot
- tracked neuron 별 exact / userbin frequency response plot
- branch response curve, total response curve, normalized response curve
- `my_R_DH_SNN`, `my_D_RF` 전용 exact 전달함수 snapshot

이 항목들은 향후 확장 대상으로 남길 수 있지만, 현재 `src/` 및 `bash/` 구현이 반드시 생성해야 하는 산출물은 아니다.

## 7. config.json 기록 규칙

현재 구현이 full transfer-function 분석이 아니라 attenuation summary snapshot 만 저장할 때에는 run-level `config.json` 에 아래 의미가 드러나야 한다.

- `filter_property_spec_doc = "paper/proposed/filter_analysis.md"`
- `filter_property_status = "attenuation_stats_only"`
- `filter_property_reason` 에 selected epoch attenuation 통계는 저장하지만 tracked-neuron / epoch-trend exact 응답 plot 은 아직 생성하지 않는다는 설명

## 8. 구현 대응

현재 코드 기준 대응은 아래와 같다.

- `src/neurons/LIF_neuron.py` : `alpha` 계산 및 subtractive soft reset
- `src/neurons/RF_neuron.py` : exact ZOH, `rho()`, `f_cyc_per_sample()`
- `src/util/psd_analysis_driver.py` : 선택 epoch `attenuation_stats/` + `all_layers_summary.csv` 저장과 `training_complete_stats/attenuation_stats/` + `training_complete_stats/all_layers_summary.csv` 생성
- `bash/run_psd.sh`, `bash/psd.sh` : psd_analysis 공식 launcher / 내부 config template. `run_psd.sh` 는 병렬 scenario 진입점이고, `psd.sh` 는 직접 실행하지 않는 Python/ML 인수 template 이다.

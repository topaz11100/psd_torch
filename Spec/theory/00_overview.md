# 전체 이론 개요

## 목적

프로젝트는 입력 데이터와 SNN 내부 신호의 시간 구조를 분리해 분석한다. 입력 데이터는 데이터셋 자체의 성질이고, hidden/output trace는 모델이 학습 과정에서 만든 동역학이다. 따라서 두 분석은 같은 FFT/PSD 연산을 쓰더라도 같은 stage에서 섞지 않는다.

## 단계 분리

```text
raw data -> prepared bundle -> dataset signal analysis -> training -> model signal analysis -> plotting
```

- `data_prep`: 원본 데이터를 학습/분석 가능한 prepared bundle로 변환한다.
- `dataset_psd`, `dataset_fft`: 입력 데이터 자체의 신호 구조를 분석한다.
- `model_training`: 모델을 학습하고 checkpoint를 저장한다.
- `psd_analysis`, `element_psd`, `2d_fft_analysis`: checkpoint를 복원해 hidden/output trace만 분석한다.
- `plotting`: 이미 저장된 CSV를 그림으로 변환한다.

## input 분석 정책

모델 분석에서 input 레이어를 분석하면 입력 데이터의 성질과 모델 내부 표현의 성질이 같은 artifact 공간에 섞인다. 따라서 모델 분석 stage는 input을 수집하지 않는다. 입력 데이터 신호가 필요하면 dataset 분석 stage를 실행한다.

## 비교 가능성

모든 거리는 같은 dataset, 같은 probe scope, 같은 layer/series 의미, 같은 frequency axis, 같은 scale/centering 정책을 공유할 때만 해석 가능하다. 단순히 CSV column 수가 같다는 이유로 비교하지 않는다.

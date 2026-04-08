# src/ 재구현 가이드 (초보자용)

이 문서는 `promft.txt`, `structure.txt`, `paper/proposed/*.md` 기준으로 현재 `src/` 코드가 어떤 원칙으로 구성되어 있는지 빠르게 이해하도록 돕는 안내서입니다.

## 1) 가장 중요한 원칙

1. **논문 + 저자 공개코드(Origin)가 있는 뉴런은 자의적으로 다시 쓰지 않는다.**
2. 실험 파이프라인(데이터 전처리, 학습/평가, PSD 저장 규칙)은 `paper/proposed` 명세를 우선한다.
3. 초보자도 추적할 수 있게, 각 모듈은 "입력 → 변환 → 출력" 흐름이 드러나도록 얇은 어댑터 형태로 유지한다.

---

## 2) 폴더별 역할

- `src/data/`: 데이터셋 로더/전처리 어댑터
- `src/neurons/`: 뉴런 레이어 구현(Origin 래퍼 + proposed my_* 구현)
- `src/readout/`: `final_membrane`, `earliest_spike`, `max_rate` readout
- `src/model/`: 모델 조립기(`build_layer`, `build_snn_classifier`)와 학습 유틸
- `src/signal/`: exact periodogram / spectrogram 계산과 시각화 유틸
- `src/plot/`: 비동기 플롯 writer 및 deferred render 처리
- `src/stat/`: 통계 집계(예: probe selection 검증)
- `src/util/`: 실험 드라이버/실행 유틸
- `src/dataset_psd.py`: 데이터셋 입력 기준선 PSD 실험 메인
- `src/psd_analysis.py`: 모델 학습 + 신호/필터 분석 메인

---

## 3) Origin 코드 그대로 쓰는(얇은 래퍼) 뉴런

아래 뉴런은 논문 저자 공개코드 로직을 최대한 그대로 사용합니다.

- `DH_SNN_neuron.py`
  - 출처: `Origin/Temporal dendritic heterogeneity .../SHD/SNN_layers/spike_dense.py`
- `TC_LIF_neuron.py`
  - 출처: `Origin/TC-LIF .../SHD-SSC/spiking_neuron/TCLIF.py`
- `TS_LIF_neuron.py`
  - 출처: `Origin/TS-LIF .../SeqSNN/network/snn/TSLIF.py`
- `D_RF_neuron.py`
  - 출처: `Origin/Dendritic Resonate-and-Fire .../models/layers.py`

`src/neurons/_origin_imports.py` 는 위 모듈을 안정적으로 import 하기 위한 로더이며, 실험 공정성을 위해 "원본 동역학" 자체를 대체하지 않습니다.

---

## 4) proposed 전용(my_*) 뉴런

아래는 `paper/proposed` 명세를 따라 프로젝트에서 새로 정의한 모델입니다.

- `my_DH_SNN_neuron.py`
- `my_R_DH_SNN_neuron.py`
- `my_D_RF_neuron.py`

공통 특징:

- 가변 가지 수 `s`를 soft/hard 마스크로 운용
- `S_min`, `S_max` 범위 제약
- `regularization_loss` 로 직교성/복잡도 항 결합 가능

---

## 5) 초보자 추천 코드 읽기 순서

1. `src/psd_analysis.py` (CLI 인자)
2. `src/util/psd_analysis_driver.py` (실험 오케스트레이션)
3. `src/model/snn_builder.py` (`build_layer`, `build_snn_classifier`)
4. `src/neurons/*_neuron.py` (실제 한 step 동역학)
5. `src/readout/readout.py` (손실 입력으로 쓰는 score 생성)
6. `src/signal/fft_analysis.py`, `src/signal/psd_utils.py` (PSD 계산)

이 순서로 보면 "명령행 인자 → 데이터/모델 구성 → forward → readout → 저장" 흐름이 잘 이어집니다.

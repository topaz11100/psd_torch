# 원본 논문 기반 뉴런 모델 정리

이 문서는 `paper/` 아래에 원본 논문 문서가 있는 neuron family 와 관련 경로를 정리한다. 프로젝트 고유 제안 모델인 `my_DH_SNN`, `my_R_DH_SNN`, `my_D_RF` 는 별도 proposed 문서에서 다루고, baseline vanilla LIF / RF 는 각각 `paper/proposed/vanila_lif.md`, `paper/proposed/vanila_rf.md` 에서 다룬다. 여기서는 현재 실험 문맥에서 참조하는 원본-paper wrapper 계열과 readout 관련 문헌만 요약한다.

실험에 사용하지 않는 paper-only reference family 는 본 문서의 catalog 에서 제외한다.

## 1. 요약 표

| family             | 원본 논문 문서                                                                                                                            | 현재 프로젝트 상태                                            | 핵심 아이디어                                                                     |
| ------------------ | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| DH-SNN             | `paper/Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics.md`             | `src/neurons/DH_SNN_neuron.py` wrapper 구현                   | 서로 다른 timing factor 를 갖는 dendritic branch 로 multi-timescale dynamics 학습 |
| D-RF               | `paper/Dendritic Resonate-and-Fire Neuron for Effective and Efficient Long Sequence Modeling.md`                                        | `src/neurons/D_RF_neuron.py` wrapper 구현                     | dendritic RF branch 와 soma 결합으로 long-sequence modeling                       |
| TC-LIF             | `paper/TC-LIF A Two-Compartment Spiking Neuron Model for Long-Term Sequential Modelling.md`                                             | `src/neurons/TC_LIF_neuron.py` wrapper 구현                   | 두 compartment 를 써서 장기 시간 의존성 강화                                      |
| TS-LIF             | `paper/TS-LIF A TEMPORAL SEGMENT SPIKING NEURON NETWORK FOR TIME SERIES FORECASTING.md`                                                 | `src/neurons/TS_LIF_neuron.py` wrapper 구현                   | temporal segment 와 dual-compartment 구조를 이용한 시계열 예측                    |
| First-spike coding | `paper/First-spike coding promotes accurate and efficient spiking neural networks for discrete events with rich temporal structures.md` | readout / loss 경로, neuron model 은 아님                     | 출력 spike timing 을 직접 분류 신호로 쓰는 coding / readout 계열                  |

## 2. DH-SNN

DH-SNN 은 dendritic branch 별 timing factor 를 다르게 두어 multi-timescale dynamics 를 학습하는 계열이다. 현재 프로젝트는 `src/neurons/DH_SNN_neuron.py` 에서 released code 를 감싼 wrapper 를 제공한다.

이 계열의 핵심은 branch 별 시간상수를 분산시켜 긴 시퀀스에서 서로 다른 시간 스케일의 누적 정보를 동시에 다룰 수 있게 한다는 점이다.

## 3. D-RF

D-RF 는 dendritic resonate-and-fire branch 와 soma 결합을 통해 긴 시퀀스를 효과적으로 다루려는 계열이다. 현재 프로젝트는 `src/neurons/D_RF_neuron.py` 에서 released `BiRFKernel` / `BiRFModel` 경로를 감싼 wrapper 를 제공한다.

이 계열은 RF 해석 가능성과 dendritic decomposition 을 함께 가져가려는 long-sequence modeling 지향 모델로 볼 수 있다.

## 4. TC-LIF

TC-LIF 는 two-compartment LIF 계열로, dendrite 와 soma 를 분리해 장기 시간 의존성과 temporal credit assignment 를 강화하려는 모델이다. 현재 프로젝트는 `src/neurons/TC_LIF_neuron.py` 에서 원 저자 코드의 `TCLIFNode` path 를 감싼 wrapper 를 제공한다.

## 5. TS-LIF

TS-LIF 는 temporal segment 와 dual-compartment 구조를 사용해 시계열 forecasting 을 겨냥한 모델이다. 현재 프로젝트는 `src/neurons/TS_LIF_neuron.py` 에서 원 저자 코드의 `TSLIFNode` path 를 감싼 wrapper 를 제공한다.

이 계열은 temporal segment 분할과 compartment dynamics 를 함께 사용하므로, 주파수 성분과 시간 구간 구조를 동시에 보는 해석과 잘 맞는다.

## 6. First-spike coding 은 neuron model 이 아님

`paper/First-spike coding promotes accurate and efficient spiking neural networks for discrete events with rich temporal structures.md` 는 출력 spike timing 을 직접 예측 신호로 쓰는 coding / readout 경로를 다룬다. 따라서 이는 neuron family 문서라기보다 readout / loss 문서로 읽는 것이 맞다. 현재 프로젝트에서는 `paper/proposed/readout.md` 와 first-spike 관련 training path 가 여기에 대응한다.

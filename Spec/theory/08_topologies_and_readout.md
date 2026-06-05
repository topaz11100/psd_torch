# 08 Topologies And Readout

## 2026 개정 해설

Topology와 readout은 trace가 loss로 연결되는 방식을 결정한다. Temporal membrane readout은 전체 시간 궤적을 사용하고, final membrane은 마지막 상태를, first spike는 최초 firing time을, rate 계열은 spike count 또는 max firing evidence를 사용한다.

# Topology와 readout

## dense SNN

MLP형 SNN은 다음 구조다.

```text
input -> hidden_1 -> ... -> hidden_L -> output/readout
```

모델 신호분석은 `hidden_i`와 `output` record만 대상으로 한다. input tensor는 dataset 분석 stage에서 별도로 다룬다.

## recurrent layer

Recurrent SNN은 같은 hidden layer의 직전 spike를 recurrent source로 사용한다.

$$
I_t^{rec}=R S_{t-1}
$$

## CNN/fixed topology

CNN 또는 외부 저자 모델 어댑터는 별도 family로 취급한다. 분석 가능한 trace series는 모델 forward capture가 제공하는 범위로 제한한다.

## readout

Readout은 classification output을 만들기 위한 계층이다. spike가 없는 readout은 spike series를 생성하지 않는다. membrane/logit 계열 trace만 있으면 해당 trace만 분석한다.

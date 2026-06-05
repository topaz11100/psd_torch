# 07 Constraints Clip Structure Clipstructure

## 2026 개정 해설

Constraint는 뉴런 파라미터 공간을 실험 설계에 맞게 제한한다. Clip은 각 파라미터의 admissible interval을 보장하고, structure는 layer/group별 주파수 배치를 강제하며, clipstructure는 두 조건을 결합한다.

# clip, structure, clipstructure

## 공통 schema

- `scenario_mode`: `none | clip | structure | clipstructure`
- `tear`: 1-based hidden layer index. structure mask 적용 시작 layer를 의미한다.
- `band_edge`: layer별 cumulative boundary (`null` 또는 `[b1, b2, ...]`)
  - `null`이면 해당 layer에서 group 수 기준 균등 분할
  - `[5, 10]`이면 group은 `[0,5)`, `[5,10)`, `[10,width)`
- `alpha_clip_edges`, `w_clip_edges`: **3D layer/group/bounds** schema
  - 예: `[[[0.1,0.3],[0.3,0.7]], [[0.2,0.4],[0.4,0.8]]]`
  - 각 group bounds는 `[lower, upper]`
  - LIF alpha는 `[0,1]`, RF w/frequency는 `[0,0.5]`

`band_neuron_ends`는 source-level deprecated alias이며 새 config 예시에서는 사용하지 않는다.

## structure

Structure는 neuron 또는 feature group을 기준으로 연결 mask를 제한한다. recurrent layer에서는 같은 group 내부 연결만 허용하는 방식으로 해석할 수 있다.

## clip

Clip은 cell parameter의 허용 구간을 제한한다.

- LIF: alpha bounds clip.
- RF: frequency(w) bounds clip.
- IF: clip target이 없어 `clip`/`clipstructure`는 에러.

잘못된 cell/parameter 조합은 validation error로 처리한다.

## clipstructure

Clipstructure는 group structure와 parameter bound를 동시에 적용한다.

- clip bounds는 모든 hidden layer에 적용된다.
- structure mask는 `hidden_index + 1 >= tear`인 hidden layer부터 적용된다.

## 분석과의 관계

Constraint는 모델 신호의 PSD/FFT 구조와 filter parameter 통계를 바꿀 수 있다. 따라서 checkpoint metadata와 filter 통계 artifact에 constraint 관련 정보가 남아야 한다.

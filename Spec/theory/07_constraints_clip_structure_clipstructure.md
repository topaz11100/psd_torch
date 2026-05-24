# clip, structure, clipstructure

## structure

Structure는 neuron 또는 feature group을 기준으로 연결 mask를 제한한다. recurrent layer에서는 같은 group 내부 연결만 허용하는 방식으로 해석할 수 있다.

## clip

Clip은 cell parameter의 허용 구간을 제한한다.

- LIF: decay alpha, threshold.
- RF: frequency, damping, threshold.
- IF: threshold.

잘못된 cell/parameter 조합은 validation error로 처리한다.

## clipstructure

Clipstructure는 group structure와 parameter bound를 동시에 적용한다. group별 bound가 있으면 각 neuron row에 lower/upper metadata가 부여되어야 한다.

## 분석과의 관계

Constraint는 모델 신호의 PSD/FFT 구조와 filter parameter 통계를 바꿀 수 있다. 따라서 checkpoint metadata와 filter 통계 artifact에 constraint 관련 정보가 남아야 한다.

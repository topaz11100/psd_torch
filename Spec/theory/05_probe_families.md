# Probe family

Probe는 분석에 사용할 sample subset이다. 학습 전체 데이터와 분석 probe는 목적이 다르므로 명시적으로 분리한다.

## full

Dataset 분석에서 split 전체를 사용하는 scope다. 이름은 `<split>_full`이다.

## balanced_global

각 class에서 가능한 한 같은 수의 sample을 선택한다. dataset PSD/FFT와 모델 분석의 대표 probe로 사용한다.

## label_single

일부 matrix 분석에서 label별 단일 sample을 비교하기 위한 scope다. balanced_global과 겹치지 않도록 선택할 수 있다.

## seed

Probe selection은 seed에 의해 고정된다. 단, deterministic algorithm mode를 켜지 않으므로 GPU kernel 수준의 완전 결정론을 보장한다는 의미는 아니다.

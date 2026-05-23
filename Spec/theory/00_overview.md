# 전체 이론 개요

## 1. 문서의 목적

이 문서는 PSD/SNN 분석 프로젝트가 어떤 대상을 관찰하고, 어떤 수학적 객체로 바꾸며, 그 객체를 어떤 artifact로 저장하는지 설명한다. 여기서 명세는 단순 구현 요구사항이 아니라 분석 설계서다. 따라서 각 문서는 다음 세 가지를 함께 다룬다.

1. 분석 대상의 정의.
2. 그 대상을 변환하는 수식.
3. 해당 결과를 이용해 무엇을 판단하는지.

## 2. 핵심 질문

이 프로젝트의 중심 질문은 다음이다.

1. 같은 probe 입력을 넣었을 때 SNN layer의 membrane, spike, RF state는 어떤 주파수 구조를 갖는가.
2. clip/structure/clipstructure constraint는 이 주파수 구조를 어떻게 바꾸는가.
3. checkpoint가 바뀌어도 같은 PCA basis를 적용했을 때 hidden-state trajectory의 spectral signature가 비교 가능한가.
4. row-time map 전체를 2D FFT로 보면 row 방향 구조와 time 방향 구조가 함께 나타나는가.
5. 서로 다른 checkpoint, layer, probe scope의 spectral object가 같은 축과 같은 basis 위에서만 비교되는가.

## 3. 전체 데이터 흐름

공식 분석 흐름은 다음과 같다.

```text
checkpoint or synthetic input
  -> model restore
  -> probe batch
  -> raw layer trace B,T,*
  -> SignalMap S,R,T
  -> PSD / PCA-PSD / 2D FFT
  -> summary CSV + tensor artifact + manifest
  -> artifact reader / plotting
```

각 단계는 이전 단계의 tensor 값만이 아니라 metadata도 전달한다. 최소 metadata는 run, checkpoint, split, scope, probe family, layer, signal kind, series다. 이 metadata가 빠지면 결과 CSV는 비교 가능한 과학적 객체가 아니라 단순 숫자 표가 된다.

## 4. 공식 좌표계

raw trace는 항상

$$
X_{raw} \in \mathbb{R}^{B \times T \times *}
$$

로 둔다. 여기서 $B$는 probe sample 수, $T$는 time step 수, $*$는 feature, neuron, channel-spatial 축이다.

분석용 SignalMap은

$$
X \in \mathbb{R}^{S \times R \times T}
$$

이다. $S$는 sample axis, $R$은 row axis, $T$는 time axis다. row axis는 MLP neuron, flattened image feature, PCA component, RF state row 등 분석 대상에 따라 의미가 달라진다.

## 5. 분석 방법의 분리

현재 구현된 공식 분석은 네 축이다.

1. PSD 대표화: `mean`, `median`, `element_psd`, `pca`.
2. fixed-reference PCA: reference checkpoint에서 basis를 맞추고 target checkpoint에 적용.
3. 2D FFT: $R \times T$ map 전체를 row-time spectral matrix로 변환.
4. strict distance: 같은 spectral axis, 같은 bin policy, 같은 PCA basis, 같은 row semantics에서만 비교.

PSD 대표화와 2D FFT는 다르다. PSD 대표화는 time axis FFT 후 row를 요약하거나 보존한다. 2D FFT는 row-time matrix 전체에 대해 row frequency와 time frequency를 동시에 만든다.

## 6. 현재 phase의 완료 범위

현재 phase의 완료 기준은 다음이다.

- `src/psd_snn`을 공식 package로 사용한다.
- raw trace CSV는 저장하지 않고 tensor chunk와 manifest를 사용한다.
- exact/userbin 결과는 같은 distance space에서만 비교한다.
- PCA result는 같은 non-empty basis id끼리만 비교한다.
- archive/reference 디렉터리는 현재 실행 계약이 아니다.

향후 작업은 실제 dataset ingest, launch packaging, 출판용 figure style, 대규모 학습 orchestration이다.

# Trace Signal Analysis Pipeline

분석 stage의 입력은 checkpoint와 prepared dataset이다. 핵심 산출물은 layer별 trace map이다.

## Trace tensor convention

모델 forward 결과는 각 layer \(\ell\)에 대해 다음 텐서를 제공한다.

\[
Y^{(\ell)} \in \mathbb{R}^{B \times T \times C_\ell},
\]

여기서 \(B\)는 probe batch, \(T\)는 시간, \(C_\ell\)은 neuron/channel index다. CNN에서는 공간 축을 channel-major row로 펼쳐

\[
Y^{(\ell)} \in \mathbb{R}^{B \times T \times (C_\ell H_\ell W_\ell)}
\]

형태로 분석한다.

## Probe scope

각 split은 다음 scope로 분해된다.

- full dataset
- label-balanced global subset
- class/family subset

동일한 scope 정의를 학습 checkpoint 분석과 dataset baseline FFT에 공유함으로써 모델이 만든 spectral structure와 데이터 자체의 spectral bias를 구분한다.

## Artifact partition

CSV는 category, layer, scope, signal kind, series, variant, scale 단위로 나뉜다. 파일 단위가 작은 이유는 다음과 같다.

1. 대형 실험에서 일부 artifact만 다시 그릴 수 있다.
2. epoch별 trend 계산이 CSV concat으로 가능하다.
3. plotting stage가 category를 기준으로 재귀 탐색할 수 있다.

## Low-VRAM mode

`low_vram=1`이면 forward 결과를 가능한 빨리 CPU로 이동하고 CUDA cache를 비운다. 산술 결과는 같지만 I/O와 CPU memory 사용량이 늘어난다.

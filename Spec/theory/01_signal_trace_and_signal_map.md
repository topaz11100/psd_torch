# Signal Trace and Signal Map

## Trace

SNN의 한 layer는 시간에 따라 상태를 갱신한다.

\[
u_t = f_\theta(u_{t-1}, x_t), \qquad s_t = H(u_t - \vartheta),
\]

여기서 \(u_t\)는 membrane 또는 RF 상태에서 유도된 membrane-like signal이고, \(s_t\)는 surrogate spike다. 분석은 두 신호를 모두 보관한다.

- `membrane`: 연속 상태. subthreshold dynamics를 포함한다.
- `spike`: threshold를 지난 이산적 firing event. sparse code의 spectral footprint를 반영한다.
- `layer_input`: 해당 layer가 받은 pre-activation current. 뉴런 동역학과 선형 변환을 분리할 때 사용한다.

## Channel-major map

PSD 연산의 표준 입력은

\[
M \in \mathbb{R}^{N\times R\times T}
\]

이다. \(N\)은 sample 수, \(R\)은 row index, \(T\)는 시간이다. MLP에서는 \(R=C\)이고, CNN에서는 \(R=C\cdot H\cdot W\)다.

CNN trace가 \(Y\in\mathbb{R}^{B\times T\times C\times H\times W}\)일 때 변환은

\[
M_{b, r, t} = Y_{b,t,c,h,w}, \qquad r=((cH)+h)W+w.
\]

이 규칙을 manifest의 `psd_row_axes`, `psd_time_axis`, `psd_flatten_rule`에 기록한다.

## Centering

시간 평균을 제거한 centered variant는

\[
\tilde{y}(t)=y(t)-\frac{1}{T}\sum_{\tau=0}^{T-1}y(\tau)
\]

로 정의한다. Raw PSD는 DC component를 포함하고, centered PSD는 시간 변화 성분을 강조한다.

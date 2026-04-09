# 데이터셋 전처리 방법 정리

이 문서는 **실험에서 실제로 사용되는** 데이터셋 전처리 방법을 실험용으로 다시 정리한 것이다.

- 데이터셋 설명은 "이 데이터가 무엇이고, 보통 어떤 task 를 하는지" 를 이해하기 쉽게 적는다.
- 전처리 방법은 논문 서술보다 released code 의 실제 동작을 우선한다.
- 논문과 코드가 완전히 일치하지 않는 경우에는 그 차이를 같이 적는다.
- 각 데이터셋마다 다운로드 위치를 따로 적는다.
- `dataset_psd` 가 참조하는 dataset 별 입력 기준선도 이 문서에 함께 둔다.
- **이 문서에 적힌 데이터만이 실험에서 사용하는 데이터셋이며,** 다른 proposed 명세문서는 dataset 이름별 서술 대신 이 문서를 참조한다.
- 현재 이 문서에 등재된 공식 실험 데이터셋은 `s-mnist` 뿐이다.

## dataset별 입력 기준선

입력 sample $n$ 의 전처리 후 model input 을

$$
X^{(n)} \in \mathbb{R}^{C \times T}
$$

로 둔다. 여기서 $C$ 는 입력 element 수, $T$ 는 시간축 길이다. 일부 loader 는 sample 을 `(T, C)` 형태로 반환하지만, PSD 계산 직전에는 항상 channel-major map $C \times T$ 로 정규화해 처리한다. 입력 element index $i$ 의 시계열은

$$
s_i^{(n)}[t], \qquad t=0,1,\dots,T-1
$$

로 둔다.

현재 공식 실험 범위 기준 shape 는 아래와 같다.

| dataset   | 전처리 기준                                                             | PSD 계산 직전 기준 `C x T` | 비고                   |
| --------- | ----------------------------------------------------------------------- | -------------------------- | ---------------------- |
| `s-mnist` | DH-SNN 계열 `ToTensor()` 후 `[0, 1]` 유지, `28 x 28 -> 784 x 1` 순차화 | $1 \times 784$            | 입력은 scalar sequence |

따라서 `dataset_psd` 는 특정 데이터셋 전용 실험이 아니라, 각 dataset 의 전처리 이후 입력 기준선 위에서 동일한 exact PSD / spectrogram 규칙을 적용해야 한다.

## 사용하는 데이터셋 종류

## 1. s-MNIST

### 1.1 데이터셋에 대한 설명

s-MNIST 는 원본 MNIST 의 `28 x 28` grayscale 손글씨 이미지를 길이 784 의 순차 데이터로 바꾼 benchmark 다. 즉, 한 장의 정적 이미지를 한 번에 넣지 않고 픽셀을 하나씩 시간축으로 펼쳐서 입력한다. task 는 최종적으로 10개 숫자 class 를 분류하는 것이다.

이 데이터셋의 핵심은 정적 이미지를 쓰지만 모델 입장에서는 긴 시퀀스 분류 문제로 바뀐다는 점이다. 따라서 긴 입력 시퀀스를 끝까지 기억하고 누적해서 판별하는 능력을 보기 좋다.

### 1.2 선택 이유/목적

s-MNIST 는 현재 공식 실험 범위에서 입력 기준선 역할을 한다.

- 구현 난도가 낮고 재현성이 높다.
- 긴 순차 분류에서 누적 기억 특성과 time-scale 차이를 보기 좋다.
- 입력이 scalar sequence 라서 PSD 계산 직전 기준선 `1 x 784` 를 명확하게 정의하기 쉽다.

### 1.3 코드를 그대로 가져올 논문이름

- 논문 이름: `Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics`
- 권장 코드 경로:
  - `Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics/s-mnist/main_rnn_denri_2layer_tbptt.py`

참고할 점은, 이 스크립트가 순서 보존 버전과 permutation variant 를 같은 코드로 처리한다는 것이다. 기본값은 permutation 이 켜진 경로 쪽에 놓여 있으므로, s-MNIST 를 그대로 쓰려면 permutation 만 꺼야 한다.

### 1.4 전처리 방법

저자 코드 기준 전처리는 다음 순서다.

1. `torchvision.datasets.MNIST` 로 원본 이미지를 로드한다.
2. `ToTensor()` 를 적용해서 픽셀 값을 tensor 로 바꾼다.
3. `ToTensor()` 가 만든 `float32` tensor 의 범위를 그대로 유지해서 픽셀 값을 `[0, 1]` 로 맞춘다.
4. 이미지를 `images.view(-1, 784, 1)` 로 reshape 해서 길이 784 의 시퀀스로 바꾼다.
5. 각 time-step 에는 scalar 실수값 하나를 직접 주입한다.
6. s-MNIST 에서는 `is_perm = False` 로 두고, 픽셀 순서를 그대로 유지한다.

이렇게 하는 이유는 분명하다.

- `ToTensor()` 는 이미지 입력을 모델이 바로 받을 수 있는 수치 tensor 로 바꾸기 위한 기본 단계다.
- `[0, 1]` 범위를 유지하면 픽셀 의미를 보존하면서도 입력 스케일이 명확하다.
- `28 x 28 -> 784 x 1` reshape 는 정적 이미지를 순차 데이터로 바꾸는 핵심 단계다.
- 각 time-step 에 scalar 실수값을 직접 주입해야 현재 구현과 실험 설정이 정확히 일치한다.
- s-MNIST 에서는 permutation 을 하지 않아야 원래 raster scan 순서를 보존할 수 있다. 이 순서가 유지되어야 정적 이미지를 순차적으로 읽는 문제라는 의미가 유지된다.

코드 기준 주의점도 있다. 같은 스크립트 안에 `perm = torch.randperm(seq_dim)` 와 `is_perm = True` 가 기본으로 들어가 있으므로, s-MNIST 재현에서는 반드시 `is_perm = False` 로 바꿔야 한다. 그렇지 않으면 실험이 permutation variant 로 바뀐다.

### 1.5 데이터 다운로드 위치

s-MNIST 자체는 별도 사이트에서 따로 내려받는 데이터셋이 아니라, 원본 MNIST 를 내려받은 뒤 순차화해서 만드는 benchmark 다.

- 원본 MNIST 공식 페이지:
  - https://yann.lecun.org/exdb/mnist/index.html
- 실험 코드 기준 실제 다운로드 방식:
  - `torchvision.datasets.MNIST(..., download=True)` 를 사용하면 torchvision 이 자동으로 데이터를 내려받는다.

실제로는 원본 MNIST 만 확보하면 되고, s-MNIST 는 별도 raw dataset 이 아니라 코드에서 `28 x 28 -> 784 x 1` flatten 과 `[0, 1]` 실수 직주입 규칙으로 생성된다.

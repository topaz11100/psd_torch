# 데이터셋 전처리 방법 정리

이 문서는 업로드된 논문 본문과 저자 코드를 기준으로, 데이터셋 전처리 방법을 실험용으로 다시 정리한 것이다.

- 데이터셋 설명은 "이 데이터가 무엇이고, 보통 어떤 task를 하는지" 를 이해하기 쉽게 적었다.
- 전처리 방법은 논문 서술보다 released code의 실제 동작을 우선했다.
- 논문과 코드가 완전히 일치하지 않는 경우에는 그 차이를 같이 적었다.
- 각 데이터셋마다 다운로드 위치를 따로 적었다.
- 계속 확장될 것이다.

---

## 1. s-MNIST

### 1. 데이터셋에 대한 설명

s-MNIST는 원본 MNIST의 28 x 28 grayscale 손글씨 이미지를 길이 784의 순차 데이터로 바꾼 benchmark다. 즉, 한 장의 정적 이미지를 한 번에 넣지 않고 픽셀을 하나씩 시간축으로 펼쳐서 입력한다. task는 최종적으로 10개 숫자 class를 분류하는 것이다.

이 데이터셋의 핵심은 "정적 이미지" 를 쓰지만 모델 입장에서는 "긴 시퀀스 분류" 문제로 바뀐다는 점이다. 따라서 센서 이벤트 처리보다는, 긴 입력 시퀀스를 끝까지 기억하고 누적해서 판별하는 능력을 보기 좋다.

### 2. 선택 이유/목적

s-MNIST는 이번 5개 데이터셋 세트에서 정적 축의 기준점 역할을 한다.

- modality 자체는 가장 단순한 편이라서, 모델 성능 차이를 sensor-specific trick보다 sequence modeling 차이로 보기 쉽다.
- DVS128 Gesture처럼 native event 데이터와 대비했을 때, "정적 이미지를 순차화한 문제" 와 "원래부터 비동기 이벤트로 생성된 문제" 의 차이를 분리해서 볼 수 있다.
- 구현 난도가 낮고 재현성이 높아서, 전체 benchmark의 baseline anchor로 쓰기 좋다.

### 3. 코드를 그대로 가져올 논문이름

- 논문 이름: `Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics`
- 권장 코드 경로:
  - `Temporal dendritic heterogeneity incorporated with spiking neural networks for learning multi-timescale dynamics/s-mnist/main_rnn_denri_2layer_tbptt.py`

참고할 점은, 이 스크립트는 s-MNIST와 PS-MNIST를 같은 코드로 처리한다는 것이다. 기본값은 PS-MNIST 쪽으로 놓여 있으므로, s-MNIST를 그대로 쓰려면 permutation만 꺼야 한다.

### 4. 전처리 방법

저자 코드 기준 전처리는 다음 순서다.

1. `torchvision.datasets.MNIST`로 원본 이미지를 로드한다.
2. `ToTensor()`를 적용해서 픽셀 값을 tensor로 바꾼다.
3. `ToTensor()`가 만든 `float32` tensor의 범위를 그대로 유지해서 픽셀 값을 `[0, 1]` 로 맞춘다.
4. 이미지를 `images.view(-1, 784, 1)`로 reshape해서 길이 784의 시퀀스로 바꾼다.
5. 각 time-step에는 scalar 실수값 하나를 직접 주입한다.
6. s-MNIST에서는 `is_perm = False`로 두고, 픽셀 순서를 그대로 유지한다.

이렇게 하는 이유는 분명하다.

- `ToTensor()`는 이미지 입력을 모델이 바로 받을 수 있는 수치 tensor로 바꾸기 위한 기본 단계다.
- `[0, 1]` 범위를 유지하면 픽셀 의미를 보존하면서도 입력 스케일이 명확하다.
- `28 x 28 -> 784 x 1` reshape는 정적 이미지를 순차 데이터로 바꾸는 핵심 단계다.
- 각 time-step에 scalar 실수값을 직접 주입해야 현재 구현과 실험 설정이 정확히 일치한다.
- s-MNIST에서는 permutation을 하지 않아야 원래 raster scan 순서를 보존할 수 있다. 이 순서가 유지되어야 "정적 이미지를 순차적으로 읽는 문제" 라는 의미가 유지된다.

코드 기준 주의점도 있다. 같은 스크립트 안에 `perm = torch.randperm(seq_dim)`와 `is_perm = True`가 기본으로 들어가 있으므로, s-MNIST 재현에서는 반드시 `is_perm = False`로 바꿔야 한다. 그렇지 않으면 실험이 PS-MNIST로 바뀐다.

### 5. 데이터 다운로드 위치

s-MNIST 자체는 별도 사이트에서 따로 내려받는 데이터셋이 아니라, 원본 MNIST를 내려받은 뒤 순차화해서 만드는 benchmark다.

- 원본 MNIST 공식 페이지:
  - https://yann.lecun.org/exdb/mnist/index.html
- 실험 코드 기준 실제 다운로드 방식:
  - `torchvision.datasets.MNIST(..., download=True)`를 사용하면 torchvision이 자동으로 데이터를 내려받는다.

실제로는 원본 MNIST만 확보하면 되고, s-MNIST는 별도 raw dataset이 아니라 코드에서 `28 x 28 -> 784 x 1` flatten과 `[0, 1]` 실수 직주입 규칙으로 생성된다.

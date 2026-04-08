# readout specification

이 문서는 본 프로젝트에서 사용하는 모델의 readout 명세 문서다.

## 규칙

분류 score 벡터는 output membrane / spike sequence 에 대한 기능적 readout 으로 결정한다. mode 이름은 `final_membrane`, `earliest_spike`, `max_rate` 셋만 허용한다.

sample $n$, class $c=0,1,\dots,C-1$, time $t=0,1,\dots,T-1$ 에 대해 output layer 의 recorded membrane 과 spike 를 각각

$$
v_c^{(n)}[t], \qquad o_c^{(n)}[t]
$$

로 둔다. 코드 상 output record tensor shape 는 $(B,T,C)$ 이며, 각 readout 은 sample $n$ 의 class score vector

$$
z^{(n)} \in \mathbb{R}^C
$$

를 다음과 같이 정의한다.

여기서 $z^{(n)}$ 는 확률 벡터가 아니라 **정규화되지 않은 class score vector** 이다. 본 문서의 학습 loss 는 PyTorch `nn.CrossEntropyLoss` 를 기준으로 기술하며, 이 경우 readout 이 반환한 $z^{(n)}$ 를 그대로 loss 입력으로 넘기고 model 또는 readout 단계에서 softmax 를 미리 적용하지 않는다. 즉, 본 절의 모든 readout 은 "확률"이 아니라 **pre-softmax score tensor** 를 정의한다. 이후 수식에 나타나는 $\operatorname{softmax}(\cdot)$ 는 readout 연산이 아니라 cross-entropy 내부의 정규화 또는 그 gradient 표현을 뜻한다.

### 1. `final_membrane`

$$
z_c^{(n)} = v_c^{(n)}[T-1]
$$

현재 구현에서 이 mode 는 readout 단계에서 softmax 를 적용하지 않고, 마지막 시점 membrane 값 자체를 raw class score 로 사용한다. 또한 이 mode 에서만 output layer 의 spike emission 과 spike-triggered reset path 를 비활성화하므로 recorded spike 는 모든 class, 모든 time-step 에 대해 0이다. 따라서 `final_membrane` 의 loss 입력은 $v^{(n)}[T-1]$ 자체이며, PyTorch `nn.CrossEntropyLoss` 를 사용할 때 추가 softmax 를 바깥에서 적용하면 안 된다.

### 2. `earliest_spike`

이 mode 에서 output layer 는 일반 spiking neuron layer 를 유지하고, 분류 판정과 학습은 output spike / membrane record 에 대해 적용한 First-spike released-code timing 경로를 따른다. output neuron 뒤 learned head 는 두지 않는다.

sample $n$, class $c=0,1,\dots,C-1$, time $t=0,1,\dots,T-1$ 에 대해 output layer 의 recorded spike 와 membrane 을 각각

$$
o_c^{(n)}[t], \qquad v_c^{(n)}[t]
$$

로 둔다. 공식 time encoder 는 class별 first-spike time 을

$$
\hat{\tau}_c^{(n)} = \operatorname{Spike2FirstTime}\left(o_c^{(n)}[0:T-1],\, v_c^{(n)}[0:T-1]\right)
$$

로 만든다. 실제 spike 가 존재하면 그 첫 시점이 우선되고, 끝까지 silent 인 class 는 horizon 밖 시간 $T + \xi$ 로 보내되 membrane 기반 dead-neuron ranking 을 사용한다. 공식 예측 label 은

$$
\hat y^{(n)} = \arg\min_c \hat{\tau}_c^{(n)}
$$

이다.

학습용 class score 는 first time 을 그대로 쓰지 않고, 가장 이른 class 가 가장 큰 score 를 갖도록

$$
r_c^{(n)} = -\hat{\tau}_c^{(n)} + \min_j \hat{\tau}_j^{(n)}
$$

로 재배열한다. 정답 label 을 $y^{(n)}$ 라 두면 batch 평균 loss 는

$$
L_{\mathrm{fs}}
=
\frac{1}{N}\sum_{n=1}^{N}
\left[
-\log \operatorname{softmax}\left(\alpha_{\mathrm{fs}} r^{(n)}\right)_{y^{(n)}}
+
\lambda_{\mathrm{treg}}
\left( e^{\beta_{\mathrm{treg}} \hat{\tau}_{y^{(n)}}^{(n)}} - 1 \right)
\mathbf{1}\left[\hat{\tau}_{y^{(n)}}^{(n)} > T\right]
\right]
$$

이다. 첫 항은 released First-spike code 의 first-time softmax classification loss 고, 둘째 항은 정답 class 가 너무 늦게 발화하거나 끝까지 silent 인 경우를 억제하는 time regularizer 다. 평가 시에는 regularizer 를 더하지 않고 $\arg\min_c \hat{\tau}_c^{(n)}$ 만 사용한다.

공식 구현은 time encoder 쪽 하이퍼파라미터로 `D = 16`, `A = 200` 을 사용하고, loss 하이퍼파라미터는 `lambda_treg = 0.01`, `beta_treg = 0.02` 를 사용한다. `alpha_fs` 는 `dvsgesture` 에서 0.1, 그 외 공식 benchmark 에서 0.2 를 사용한다.

즉 `earliest_spike` 의 공식 supervised path 는 output spike / membrane record 기반 timing loss 이며, train/test accuracy 와 probe-set accuracy 도 같은 encoded first-time prediction 기준으로 계산한다.

### 3. `max_rate`

$$
z_c^{(n)} = \frac{1}{T} \sum_{t=0}^{T-1} o_c^{(n)}[t]
$$

즉 output spike sequence 의 평균 firing rate 를 raw class score 로 사용한다. 이 mode 역시 확률 벡터를 만들기 위해 softmax 를 미리 취하지 않고, 계산된 score 를 그대로 loss 입력으로 사용한다.

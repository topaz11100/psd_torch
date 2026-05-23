# Topology와 readout 이론

## 1. topology와 cell의 분리

Topology는 layer 연결 구조를 정의한다. Cell은 각 time step에서 state를 갱신한다. 따라서 `mlp_stack`은 topology이고 `if`, `lif`, `rf`는 cell이다.

## 2. MLPStack

hidden width sequence를

$$
(d_1,d_2,\ldots,d_L)
$$

라 하면 MLPStack은 다음 skeleton이다.

```text
input -> hidden_1 -> hidden_2 -> ... -> hidden_L -> readout
```

각 hidden layer는 dense affine current와 cell update를 갖는다.

$$
I_t^{(\ell)}=W^{(\ell)}S_t^{(\ell-1)}+b^{(\ell)}
$$

## 3. spike-only SRNN recurrence

recurrent hidden block은 같은 layer의 직전 spike만 사용한다.

$$
I_t^{(\ell)}=W^{(\ell)}S_t^{(\ell-1)}+R^{(\ell)}S_{t-1}^{(\ell)}+b^{(\ell)}
$$

여기서 recurrent source는 $S_{t-1}^{(\ell)}$로 고정된다. 다른 state를 recurrent input으로 고르는 설계는 현재 공식 경로가 아니다.

## 4. fixed topology

fixed topology는 MLP cell replacement 대상이 아니다. 현재 fixed topology smoke path는 다음 구조를 독립 topology로 다룬다.

- GRU
- SSM/S4 alias
- VGG-like
- ResNet-like
- SpikeTransformer-like

이들은 `TopologySpec.kind`로 dispatch된다. `CellSpec.kind`에 들어가지 않는다.

## 5. final_if

`final_if`는 output path에서 IF-like membrane/decision 값을 logits로 사용한다. spike trace가 없을 수 있다. 분석에서는 output spike가 unavailable일 수 있음을 metadata로 처리해야 한다.

## 6. final_mem

`final_mem`은 spike 없는 membrane accumulator로 마지막 time step membrane을 logits로 사용한다.

$$
\mathrm{logits}=U_T
$$

이 readout은 classification output을 만들기 위한 것이지 output spike를 강제 생성하기 위한 것이 아니다.

## 7. 분석 관점

Topology/readout 문서는 어떤 trace series가 관찰 가능한지 정한다. hidden cell은 spike/membrane/RF state를 낼 수 있고, readout은 logits 또는 output membrane만 낼 수 있다. unavailable series는 fake tensor로 채우지 않는다.

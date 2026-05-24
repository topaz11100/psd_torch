# Trace와 SignalMap

## raw trace

모델의 한 layer trace는 일반적으로

$$
Z \in \mathbb{R}^{B 	imes T 	imes *}
$$

로 표현한다. $B$는 batch/sample, $T$는 time step, $*$는 neuron 또는 channel/spatial 축이다.

## SignalMap

신호분석은 다음 형태를 기준으로 한다.

$$
X \in \mathbb{R}^{S 	imes R 	imes T}
$$

- $S$: sample axis.
- $R$: row axis. MLP에서는 neuron, image/event에서는 channel-spatial flatten row다.
- $T$: time axis.

MLP trace `B,T,F`는 `B,F,T`로 변환한다. image/event trace `B,T,C,H,W`는 `B,C*H*W,T`로 변환한다.

## dataset 입력과 model trace

prepared dataset은 manifest의 `psd_time_axis`, `psd_row_axes`, `psd_flatten_rule`, `psd_logical_shape`를 사용해 SignalMap으로 변환한다. 모델 trace는 모델 forward capture에서 나온 hidden/output record만 SignalMap으로 변환한다.

## metadata

SignalMap은 tensor만으로 식별되지 않는다. 최소 metadata는 다음이다.

```text
dataset, run_id, seed, split, scope, probe_family,
layer, layer_index, signal_kind, series,
checkpoint_path, checkpoint_epoch
```

Dataset 분석은 `signal_kind=input`을 사용할 수 있다. 모델 분석은 `signal_kind=input`을 사용하지 않는다.

# SpikGRU auxiliary diversity 구현 명세

## 1. 범위

이 문서는 `Spec/theory/models/spikegru.md` 를 구현으로 연결한다. `spikegru` 는 non-image time-series dataset 에 적용하는 fixed auxiliary diversity model 이며, dense SNN 주 분석을 대체하지 않는다.

공식 구현 token 은 아래 하나다.

```text
spikegru
```

`spikegru` 는 `Investigating Current-based and Gating Approaches for Accurate and Energy-efficient Spiking Recurrent Neural Networks` 의 1x128 SpikGRU network 를 뜻한다.

## 2. official source contract

공식 근거 source 는 아래다.

1. `spikegru.zip/paper/Investigating Current-based and Gating Approaches for Accurate and Energy-efficient Spiking Recurrent Neural Networks.pdf`
2. `spikegru.zip/code/SpikGRU-DVSLip-main.zip`
3. `SpikGRU-DVSLip-main/layers.py::GRUlayer`
4. `SpikGRU-DVSLip-main/layers.py::LiGRU`

구현자는 DVS-Lip code 의 CNN frontend, bidirectional 3-layer backend, two-gate SpikGRU2, signed activation ablation 을 official `spikegru` 로 가져오면 안 된다. 필요한 경우 `GRUlayer` 의 single-gate update 만 reference 로 사용하고, project wrapper 는 1 recurrent layer, hidden 128 로 고정한다.

## 3. fixed profile table

| field | value |
| --- | --- |
| `model_profile` | `spikegru` |
| `paper_model` | SpikGRU |
| `paper_topology` | `1x128` |
| `input_domain` | non-image time-series |
| `recurrent_layers` | 1 |
| `hidden_size` | 128 |
| `gate_count` | 1 |
| `state_family` | `i_current`, `z_gate`, `y_mem`, `y_spike` |
| `threshold` | 1 |
| `alpha_init` | 0.9 |
| `alpha_clamp` | `[0,1]` |
| `readout` | non-spiking self-recurrent integrating readout |
| `loss_reference` | max-over-time cross entropy |
| `optimizer` | Adam |
| `learning_rate_reference` | 0.001 |
| `epoch_reference` | 200 |
| `batch_reference` | 128, SSC reference 512 |
| `structure_variation` | `none` |

Optimizer policy:
- Official PSD project execution uses Adam.
- `spikegru` does not expose parameter decay or optimizer-family override.

## 4. 구현 요구사항

### IMPL-MODEL-SGRU-001

요구사항:
Model registry 는 `spikegru` 하나를 official SpikGRU diversity profile 로 제공해야 한다.

적용 대상:
- `src/model/model_registry.py`
- `src/model/spikegru.py` 또는 동등 wrapper
- `src/psd_analysis.py`

수용 기준:
- registry token 은 정확히 `spikegru` 다.
- hidden size 는 128 로 고정한다.
- recurrent layer 수는 1 로 고정한다.
- gate 수는 1 로 고정한다.
- token 이름에 `1x128`, dataset 이름, gate 수를 붙이지 않는다.

### IMPL-MODEL-SGRU-002

요구사항:
SpikGRU cell 은 current, update gate, membrane, spike update 를 paper profile 에 맞춰 구현해야 한다.

적용 대상:
- `src/model/spikegru.py`
- `src/model/arch_spec.py`

수용 기준:
- current state `i_current` 는 trainable per-neuron `alpha` 로 누적한다.
- update gate `z_gate` 는 sigmoid gate 이며 spike 로 이산화하지 않는다.
- membrane `y_mem` 은 gate 로 previous membrane 과 current 를 혼합한다.
- spike `y_spike` 는 threshold surrogate path 를 사용한다.
- previous spike reset term `-v_th * s_prev` 를 포함한다.
- `alpha` 는 0.9 로 초기화하고 training 중 `[0,1]` 로 clamp 한다.

### IMPL-MODEL-SGRU-003

요구사항:
Dataset adapter 는 non-image time-series 입력만 허용해야 한다.

적용 대상:
- `src/data/registry.py`
- `src/model/spikegru.py`
- `src/data_prep.py`

수용 기준:
- model input 은 logical `[B,T,C]` 또는 author-compatible equivalent 로 기록한다.
- PSD logical view 는 rows = input channels, time = sequence timestep 이다.
- image tensor 또는 CNN frontend 를 `spikegru` adapter 로 자동 연결하지 않는다.
- class 수 변경은 readout output dimension 변경으로만 처리한다.

### IMPL-MODEL-SGRU-004

요구사항:
SpikGRU observer 는 raw activation archive 없이 PSD accumulator 로 streaming 해야 한다.

적용 대상:
- `src/psd_analysis.py`
- `src/util/psd_analysis_driver.py`
- `src/signal/family_spectral_analysis.py`

수용 기준:
- `x_probe`, `x_layer`, `i_current`, `z_gate`, `y_mem`, `y_spike`, `readout_mem` 을 hook 가능한 범위에서 수집한다.
- `i_current`, `z_gate`, `readout_mem` 은 observe-only auxiliary family 로 기록한다.
- full raw activation archive 를 저장하지 않는다.
- 각 family 는 `representative(PSD(signal))` 순서로 처리된다.

### IMPL-MODEL-SGRU-005

요구사항:
Run metadata 는 fixed 1x128 SpikGRU profile 을 명시해야 한다.

적용 대상:
- `src/util/metadata.py`
- `src/psd_analysis.py`

수용 기준:
- `analysis_role = auxiliary_diversity` 를 기록한다.
- `model_token = spikegru` 를 기록한다.
- `paper_title`, `paper_topology`, `hidden_size`, `recurrent_layers`, `gate_count`, `readout`, `alpha_init`, `alpha_clamp`, `structure_variation = none` 을 기록한다.
- DVS-Lip frontend 또는 two-gate backend 를 사용하지 않았음을 metadata 에 남긴다.

### IMPL-MODEL-SGRU-006

요구사항:
`spikegru` output artifact 는 다른 PSD 분석과 동일하게 CSV-first output contract 를 따라야 한다.

적용 대상:
- `src/psd_analysis.py`
- `src/plot/*`
- `Spec/impl/spec/csv_schema.md`

수용 기준:
- 원래 plot 위치와 plot 이름마다 `<plot_name>.csv` 를 저장한다.
- training/analysis 단계에서는 PNG 를 생성하지 않는다.
- `plotting.py` 단계에서 같은 CSV 를 읽어 PNG 를 생성한다.
- `.npz` 에 SpikGRU 분석 결과를 묶지 않는다.

## 5. 금지 동작

1. DVS-Lip ResNet frontend 를 official `spikegru` 로 포함하지 않는다.
2. 2x128, 3-layer, bidirectional, SpikGRU2, signed activation profile 을 official `spikegru` 로 기록하지 않는다.
3. image dataset 을 자동 flatten 해서 `spikegru` 에 투입하지 않는다.
4. gate trace 를 spike trace 로 저장하지 않는다.
5. optimizer override 또는 parameter decay override CLI 를 만들지 않는다.
6. model checkpoint `.pt` 를 공식 PSD 산출물로 저장하지 않는다.

## 2026-05-02 수정 고정

- `spikegru` 는 바닐라 `2x128` SpikGRU 로 고정한다.
- class 이름은 `SpikGRUClassifier` 를 유지한다. 별도 legacy class 나 checkpoint profile 감지는 만들지 않는다.
- `hidden_spec`, neuron model suffix, reset suffix, threshold suffix 는 `spikegru` 구조에 반영하지 않는다.
- 두 recurrent block 의 기록 신호는 `membrane = v_t`, `spike = s_t`, auxiliary 로 `i_current`, `z_gate` 를 둔다.
- 출력층은 non-spiking readout membrane trace 를 만들고, loss 는 readout membrane trace 의 max-over-time CE 를 사용한다.

# PSD 프로젝트 대규모 리팩터링 변경점 점검 문서

작성일: 2026-05-23  
대상 입력: `psd.zip`, `psd_old.zip`, `crossspec 감사 프로젝트 zip`, 업로드된 SpikingJelly/API/RF/프로젝트 구조 개선 문서, 업로드된 PCA/FFT 분석 논문

이 문서는 현재 코드를 참고 자료로 두고, 모델 학습·trace 수집·신호처리·동역학 통계·시각화·실험 설정을 독립 실행 단위로 재구성하기 위한 최종 설계 기준을 정리한다.

---

## 1. 전체 설계 원칙

핵심 목표는 다음과 같다.

```text
모델 학습
  모델 구성, 학습, checkpoint 저장, metric 저장

trace 수집
  checkpoint 모델을 복원하고 probe 입력을 다시 forward해 layer 신호를 표준 record로 방출

신호 분석
  PSD, element PSD, 2D FFT, PCA-PSD를 독립 실행 단위로 계산

동역학 통계
  학습 파라미터와 SpikingJelly 내부 상태, probe 기반 상태 trace 통계를 같은 실행 단위에서 저장

artifact 저장
  CSV, tensor artifact, manifest, plot 입력 파일을 규격화

시각화
  분석 산출물만 읽고 그림을 생성
```

`psd_analysis.py` 하나가 checkpoint 복원, trace 수집, PSD 계산, CSV 저장, plot 입력 생성을 모두 담당하지 않는다. 새 구조에서는 공통 인프라만 공유하고, 분석 목적별 실행 단위를 분리한다.

공통 인프라는 다음으로 제한한다.

```text
CheckpointLoader
ModelRestoreService
ProbeSetBuilder
TraceAdapter
SignalMapEmitter
ArtifactWriter
ArtifactReader
CSV 규격 검증기
```

---

## 2. 실행 단위

새 실행 단위는 아래처럼 둔다.

| 실행 단위 | 책임 | forward 필요 여부 |
|---|---|---:|
| `train` | 모델 학습, checkpoint 저장, training/evaluation metric 저장 | 예 |
| `analyze_psd` | representative signal 또는 element power에서 PSD curve, dispersion, distance 계산 | 예 |
| `analyze_element_psd` | row, neuron, channel 단위 PSD matrix 계산 | 예 |
| `analyze_fft2d` | row-time map에서 2D FFT matrix 계산 | 예 |
| `analyze_pca_psd` | PCA mode 대표 신호를 만들고 mode별 PSD 또는 mode 간 spectral matrix 계산 | 예 |
| `analyze_dynamics` | 학습 파라미터, 내부 상태, probe 기반 상태 trace 통계 저장 | 선택 |
| `evaluate_checkpoint` | checkpoint별 accuracy, loss, confusion, prediction summary 계산 | 예 |
| `plot` | CSV/tensor artifact를 읽어 그림 생성 | 아니오 |

`analyze_dynamics`는 파라미터 통계와 내부 상태 통계를 같은 작업으로 묶는다. 파라미터만 읽는 실행은 forward가 필요하지 않지만, SpikingJelly memory state나 state trace 통계를 저장할 때는 probe forward가 필요하다.

---

## 3. 추천 프로젝트 구조

```text
src/
  psd_snn/
    core/
      config.py
      paths.py
      seed.py
      device.py
      checkpoint.py
      manifest.py

    data/
      registry.py
      storage.py
      preprocessing/
        common.py
        mnist.py
        sequential.py
        shd.py
        dvs.py
        uci.py

    models/
      build_spec.py
      topology.py
      constraints.py
      registry.py
      factory.py
      checkpoint_io.py
      fixed/
        vgg.py
        resnet.py
        spikegru.py
        spikformer.py
        spiking_ssm.py
      mlp/
        builder.py
        blocks.py
        readout_blocks.py
      trace_types.py

    sj_custom/
      base.py
      surrogate.py
      cells/
        if_cell.py
        lif.py
        rf.py
        tc_lif.py
        ts_lif.py
        dh_snn.py
        d_rf.py
      blocks/
        dense_temporal.py
        conv_temporal.py
        residual.py
        recurrent.py

    readout/
      readout.py
      final_if.py
      final_mem.py
      first_spike.py
      rate.py

    training/
      trainer.py
      losses.py
      regularizers.py
      schedules.py
      metrics.py

    analysis/
      common/
        checkpoint_loader.py
        model_restore.py
        probe.py
        trace_adapter.py
        signal_map.py
        artifact_paths.py
      spectral/
        psd.py
        accumulator.py
        representative.py
        element_psd.py
        fft2d.py
        pca_psd.py
        distance.py
      dynamics/
        parameter_vectors.py
        memory_states.py
        state_trace_stats.py
        summarize.py
      accuracy/
        checkpoint_eval.py

    artifacts/
      csv_columns.py
      validators.py
      writers/
        scalar.py
        curve.py
        matrix.py
        manifest.py
      tensor_store.py

    plotting/
      readers.py
      figures.py
      renderers.py

    cli/
      train.py
      analyze_psd.py
      analyze_element_psd.py
      analyze_fft2d.py
      analyze_pca_psd.py
      analyze_dynamics.py
      evaluate_checkpoint.py
      plot.py

configs/
  datasets/
  models/
  experiments/

scripts/
  launch.py

bash/
  train.sh
  analyze.sh
  plot.sh

third_party/
  origin_reference/

Spec/
  impl/
  theory/
  traceability.md
  conflict.md

tests/
  test_checkpoint_restore.py
  test_trace_layout.py
  test_signal_map.py
  test_psd_accumulator.py
  test_dynamics_stats.py
  test_csv_columns.py
```

`Origin` 계열 모델은 실행 경로에서 직접 import하지 않는다. 원본 구현은 `third_party/origin_reference/`에 보관하고, 실제 실행 모델은 `models/fixed/` 또는 `sj_custom/` 아래에서 SpikingJelly 기반 구현으로 제공한다.

---

## 4. 모델 구성 전략

### 4.1 4.1 고정 논문 모델과 가변 MLP 분리

VGG, ResNet, S4/SSM, GRU, Spike-Transformer 계열은 논문 구조를 기준으로 고정 factory로 둔다. MLP만 topology와 cell을 분리해 적층식으로 구성한다.

| 모델군 | 구성 방식 | 변경 가능한 축 |
|---|---|---|
| MLP stack | `MLPStackBuilder`가 `DenseTemporalBlock`을 적층 | hidden width, depth, cell, reset, threshold, readout, topology scenario |
| VGG | fixed model factory | cell family, readout, trace adapter 정도만 제한적으로 변경 |
| ResNet | fixed model factory | block trace 지점, cell family, readout |
| S4/SSM | fixed model factory | SSM block trace 지점, spiking readout, output cell |
| SpikeGRU | fixed model factory | gate/state trace 지점, readout |
| Spike-Transformer | fixed model factory | block trace 지점, attention/spike/state trace |

MLP는 특정 뉴런 모델을 뜻하지 않는다. 여기서 MLP는 dense temporal block을 순차 적층하는 topology다. 따라서 동일한 MLP topology에 LIF, IF, RF, TC-LIF, TS-LIF, DH-SNN, D-RF cell을 갈아 끼울 수 있어야 한다.

```text
MLP topology
  DenseTemporalBlock 1
  DenseTemporalBlock 2
  ...
  OutputTemporalBlock

각 block
  synapse: Linear 또는 masked Linear
  cell: IF, LIF, RF, TC-LIF, TS-LIF, DH-SNN, D-RF 등
```

### 4.2 4.2 BuildSpec 중심 설정

model token에 topology, reset, threshold, readout, scenario 정보를 모두 넣지 않는다. 실행 설정은 `BuildSpec`으로 관리한다.

```yaml
model:
  topology:
    kind: mlp_stack
    hidden_widths: [256, 256, 128]

  cell:
    kind: rf
    recurrent: false
    reset:
      mode: threshold_only
    threshold:
      trainable: true
      init: 1.0
    params:
      exact_zoh: true
      frequency_init: band_uniform
      damping_init: constant

  output_cell:
    kind: if
    emit_spike: false
    reset_enabled: false

  readout:
    mode: final_if
```

checkpoint에는 모델 객체를 저장하지 않고, 모델을 복원할 수 있는 build 설정과 `state_dict`를 저장한다.

### 4.3 MLP SRNN recurrent 연결

MLP stack은 hidden block 단위 SRNN recurrent 연결을 옵션으로 지원한다. 여기서 recurrent 연결은 **이전 시점의 출력 spike를 recurrent weight로 가중합해 현재 시점 hidden current에 더하는 연결**만 의미한다. recurrent source 선택 옵션은 두지 않는다.

```yaml
model:
  topology:
    kind: mlp_stack
    hidden_widths: [256, 256]

  cell:
    kind: lif
    recurrent: true
```

hidden block `l`의 recurrent current는 다음처럼 정의한다.

```text
I_t^(l) = W_ff^(l) x_t^(l) + R^(l) s_{t-1}^(l) + b^(l)
```

여기서 `s_{t-1}^(l)`는 같은 hidden block이 직전 시점에 방출한 spike tensor다. sequence 시작 시점의 `s_{-1}`은 zero tensor로 초기화한다. 이 recurrent state는 SpikingJelly reset과 같이 sample/batch 경계에서 초기화해야 한다.

`structure` 또는 `clipstructure` scenario에서는 feedforward mask와 별개로 recurrent mask를 생성한다. recurrent mask는 같은 group 내부 연결만 허용한다.

```text
feedforward mask:
  target_group == source_group

recurrent mask:
  target_group == source_group within same hidden block
```

출력층 recurrent 연결은 기본 비활성화한다. 출력층 recurrent가 필요한 경우에는 별도 ablation 설정으로 두되, 기본 MLP SRNN 정의에는 포함하지 않는다.


---

## 5. SpikingJelly 기반 custom node 구현

모든 모델은 SpikingJelly 기반으로 통합한다. 의미는 다음이다.

```text
SpikingJelly의 상태관리, reset, step mode, surrogate, monitor 계열 convention을 따른다.
각 뉴런 모델의 수식은 보존한다.
SpikingJelly 내장 LIF로 모든 모델을 단순 치환하지 않는다.
```

### 5.1 5.1 공통 cell 계약

모든 temporal cell은 아래 계약을 만족한다.

```python
@dataclass
class CellOutput:
    traces: dict[str, torch.Tensor]
    output: torch.Tensor
```

표준 raw trace layout은 `B,T,*`다.

```text
MLP trace          B,T,F
CNN trace          B,T,C,H,W
Transformer trace  B,T,N,D
SSM trace          B,T,D
```

SpikingJelly 내부 multi-step 처리에서 `T,B,*`가 필요하면 wrapper 내부에서 변환한다. 프로젝트 외부와 분석 모듈에는 항상 `B,T,*`로 내보낸다.

### 5.2 5.2 공통 trace series

| series | 의미 | 기본 저장 |
|---|---|---:|
| `input_current` | synapse를 지난 cell 입력 전류 | 선택 |
| `membrane_pre` | fire/reset 직전 raw 막전위 또는 RF real state | 예 |
| `decision` | spike 함수 입력값. 예: `$v_t^{pre}-v_{th}$` | 예 |
| `spike` | cell 출력 spike 또는 weighted spike | 예 |
| `membrane_post` | reset 적용 뒤 다음 step으로 carry되는 상태 | 선택 |
| `readout_membrane` | readout용 output membrane | 예 |

RF는 다음 trace를 추가한다.

| series | 의미 |
|---|---|
| `rf_real_pre` | reset 전 RF real state |
| `rf_imag_pre` | reset 전 RF imaginary state |
| `rf_real_post` | reset 후 RF real state |
| `rf_imag_post` | reset 후 RF imaginary state |
| `rf_state_amplitude_pre` | reset 전 oscillator amplitude |
| `rf_state_energy_pre` | reset 전 oscillator energy |
| `rf_phase_pre` | reset 전 phase |

TC-LIF, TS-LIF, DH-SNN, D-RF, GRU, SSM 계열은 `extras`가 아니라 명명된 trace series를 사용한다. PSD 분석 코드는 모델별 `getattr` 분기를 갖지 않는다.

### 5.3 5.3 cell별 구현 기준

| cell | 구현 단위 | 내부 상태 | 필수 trace |
|---|---|---|---|
| IF | `MemoryModule` 또는 `BaseNode` | `v` | `membrane_pre`, `decision`, `spike`, `membrane_post` |
| LIF | custom `BaseNode` | `v`, optional learnable decay | `input_current`, `membrane_pre`, `decision`, `spike`, `membrane_post` |
| RF | custom `MemoryModule` | `x`, `y` | `rf_real_pre`, `rf_imag_pre`, `decision`, `spike`, `rf_real_post`, `rf_imag_post` |
| TC-LIF | custom `MemoryModule` | coupled compartments | compartment traces, `decision`, `spike` |
| TS-LIF | custom `MemoryModule` | short/long states | short/long membrane, short/long spike, mixed output |
| DH-SNN | custom `MemoryModule` | branch state, soma state | dendrite state, soma membrane, `decision`, `spike` |
| D-RF | custom block | kernel/filter state | filtered signal, RF kernel summary, `decision`, `spike` |
| SpikeGRU | custom block | gate/current/hidden/spike | `z_gate`, `i_current`, hidden state, output spike |
| SpikingSSM | custom block | SSM state, spiking state | SSM state, filtered output, spike |

### 5.4 5.4 reset과 threshold 설정

IF/LIF 계열 reset은 아래 enum을 사용한다.

```text
none
hard
soft
```

RF 계열 reset은 별도 enum을 사용한다.

```text
none
threshold_only
hard_state
hard_real
soft_real
scale_state
smooth_damping
continuous_reset
```

threshold는 모든 vanilla 및 scenario cell에서 학습 가능 여부를 독립 설정한다.

```yaml
cell:
  kind: lif
  reset:
    mode: soft
  threshold:
    trainable: false
    init: 1.0
```

```yaml
cell:
  kind: rf
  reset:
    mode: threshold_only
  threshold:
    trainable: true
    init: 1.0
```

RF의 frequency, damping, decay radius는 파라미터 통계 대상이다. LIF의 decay coefficient, time constant도 파라미터 통계 대상이다.

---

## 6. MLP scenario: clip, structure, clipstructure

MLP topology에 대해 `none`, `clip`, `structure`, `clipstructure` scenario를 제공한다. scenario는 모델 token이 아니라 topology constraint로 관리한다.

```yaml
topology:
  scenario: clipstructure
  tear: 2
  group:
    mode: auto
  lif:
    alpha_edges: [0.0, 0.25, 0.5, 0.75, 1.0]
  rf:
    frequency_edges: [0.0, 0.05, 0.10, 0.20, 0.50]
    damping_bounds: [0.01, 2.0]
  apply_to_output_layer: false
```

### 6.1 6.1 structure

`structure`는 hidden neuron을 group으로 나누고, group 조건에 맞는 연결만 허용한다.

```text
source group과 target group이 같은 연결만 활성화
SRNN recurrent weight는 같은 group 내부 연결만 활성화
```

첫 hidden layer 입력에 group 정의가 없으면 hidden-to-hidden부터 structure를 적용한다. 입력 feature에 의미 있는 group을 정의할 수 있는 dataset에서는 첫 hidden layer에도 적용할 수 있다.

### 6.2 6.2 clip

`clip`은 뉴런 동역학 파라미터를 group별 범위 안으로 제한한다.

| cell | clip 대상 | 의미 |
|---|---|---|
| LIF | `membrane_decay_alpha` | time memory 강도 |
| LIF | `membrane_time_constant_tau_step` | step 단위 time constant |
| RF | `resonant_frequency_cyc_per_step` | 공진 주파수 |
| RF | `damping_magnitude` | 감쇠 강도 |
| RF | `decay_radius_rho` | discrete decay radius |

RF frequency는 group별 clip을 기본으로 둔다. RF damping은 전체 bound를 기본으로 두되, group별 damping clip을 실험 옵션으로 둔다.

### 6.3 6.3 clipstructure

`clipstructure`는 structure mask와 parameter clip을 동시에 적용한다. 연결 topology와 뉴런 파라미터의 band 구조가 같이 기록되어야 하므로, dynamics 통계 산출물에는 `group_id`, `lower_bound`, `upper_bound`가 포함된다.

---

## 7. readout 설계

rate, max-rate, first-spike, temporal membrane 계열 readout과 함께 `final_if`, `final_mem`을 제공한다.

| readout | output cell | output spike | logits |
|---|---|---:|---|
| `rate` | spiking cell | 있음 | time 평균 firing rate |
| `max_rate` | spiking cell | 있음 | class별 firing score |
| `first_spike` | spiking cell | 있음 | first spike timing 기반 score |
| `temporal_membrane` | no-spike output cell | 없음 | time 축 membrane 집계 |
| `final_if` | no-spike IF output cell | 없음 | 마지막 시점 IF membrane |
| `final_mem` | hidden cell과 같은 계열의 no-spike output cell | 없음 | 마지막 시점 output membrane |

`final_if`와 `final_mem`의 output cell은 fire와 reset을 끈다.

```text
emit_spike = false
reset_enabled = false
logits = output_membrane[:, -1, :]
```

`final_mem`은 hidden cell이 RF면 RF no-spike output state를 쓰고, hidden cell이 LIF면 LIF no-spike output membrane을 쓴다. `final_if`는 hidden cell과 무관하게 IF-style accumulator를 output cell로 사용한다.

output layer가 spike를 내지 않는 readout에서는 output `spike` artifact를 만들지 않는다. manifest에는 해당 series가 생성되지 않은 이유를 기록한다.

---

## 8. Trace와 Signal Map 표준화 순서

이 순서는 SpikingJelly 통합에 맞는 순서다.

```text
SpikingJelly model forward
  ↓
TraceAdapter
  ↓
LayerTraceRecord, raw layout B,T,*
  ↓
SignalMapEmitter
  ↓
LayerSignalMapRecord, PSD-ready layout S,R,T
  ↓
PSD, element PSD, 2D FFT, PCA-PSD, dynamics trace stats
```

중요한 점은 Signal Map이 SpikingJelly runtime 입력이 아니라 분석용 layout이라는 점이다. SpikingJelly multi-step node는 내부에서 `T,B,*`를 쓸 수 있지만, TraceAdapter가 이를 `B,T,*`로 통일한다. 그 다음 SignalMapEmitter가 분석 목적에 맞게 `S,R,T`로 바꾼다.

### 8.1 8.1 TraceAdapter 책임

TraceAdapter는 다음을 담당한다.

```text
functional.reset_net(model)을 forward 전후에 호출
SpikingJelly node output, v_seq, custom memory state를 수집
T,B,* convention을 B,T,* convention으로 변환
LayerTraceRecord를 생성
trace_level에 따라 필요한 series만 방출
```

TraceAdapter는 모델별로 존재할 수 있다.

```text
MLPStackTraceAdapter
FixedCNNTraceAdapter
SpikformerTraceAdapter
SpikingSSMTraceAdapter
SpikeGRUTraceAdapter
```

분석 코드는 모델 family별 분기를 갖지 않고 TraceAdapter가 방출한 record만 처리한다.

### 8.2 8.2 SignalMapEmitter 책임

SignalMapEmitter는 raw trace를 분석 map으로 바꾼다.

| raw trace layout | signal map layout |
|---|---|
| `B,T,F` | `S=B, R=F, T=T` |
| `B,T,C,H,W` | `S=B, R=C*H*W, T=T` |
| `B,T,N,D` | `S=B, R=N*D, T=T` |
| `B,T,D` | `S=B, R=D, T=T` |

raw trace는 시간영역 artifact 저장과 raster, spike rate, autocorrelation, state trace 통계에 재사용된다. Signal Map은 PSD, element PSD, 2D FFT, PCA-PSD에서 쓰는 분석 표현이다.

---

## 9. 시간영역 spike 출력 저장

시간영역 spike 출력은 PSD 계산용 accumulator와 별도 writer가 저장한다.

```text
TraceAdapter
  ├─ SignalMapEmitter → PSDAccumulator.update(...)
  └─ TraceArtifactWriter.write_chunk(...)
```

저장 원칙은 다음과 같다.

```text
메모리에는 전체 probe trace를 모으지 않는다.
disk에는 선택된 spike trace를 chunk 단위로 저장한다.
chunking은 sample/batch axis 기준으로 한다.
time axis는 chunk 안에 온전히 보존한다.
```

권장 저장 layout은 raw trace layout이다.

```text
MLP spike          B,T,F
CNN spike          B,T,C,H,W
Transformer spike  B,T,N,D
SSM spike          B,T,D
```

spike는 `uint8` 저장을 기본으로 하고, 필요하면 bitpack 압축을 선택한다. CSV는 spike 값 저장용이 아니라 tensor artifact manifest와 summary 저장용으로만 쓴다.

---


## 10. Probe set과 분석 scope

Probe set은 모든 신호분석 실행 단위가 공유하는 표준 입력 집합이다. PSD, element PSD, 2D FFT, PCA-PSD, 동역학 trace 통계는 같은 `ProbeSetBuilder` 규격을 사용한다. 특정 분석 전용 임시 scope를 만들지 않는다.

모든 분석 산출물에는 아래 metadata를 남긴다.

```text
split
scope
probe_family
label
sample_selection_seed
sample_index 또는 sample_manifest_ref
exclusion_family optional
```

### 10.1 probe family

| probe family | 목적 | sampling rule | 기본 사용처 |
|---|---|---|---|
| `balanced_global` | class-balanced 전역 비교 | 모든 label에서 같은 수의 sample 선택 후 결합 | 기본 PSD, 거리, epoch trend |
| `distributed_set` | 데이터 분포 반영 | split의 empirical class count 비율을 보존하도록 sample 수 배정 | 분포 민감도, 실제 split 성향 반영 분석 |
| `label_set` | label-conditioned 분석 | 지정 label 또는 전체 label에 대해 label별 sample set 구성 | label별 PSD, label별 state 통계, label별 dynamics |
| `label_single` | label별 최소 대표 sample inspection | 기준 global probe set에 포함되지 않은 sample 중 label별 1개 선택 | 모든 신호분석의 inspection scope |

`label_set`은 label-conditioned multi-sample probe 역할을 담당한다. artifact, config, CSV에는 `label_set`만 기록한다.

`label_single`은 2D FFT나 matrix 분석 전용 scope가 아니다. `balanced_global`, `distributed_set`과 같은 ProbeSetBuilder family로 관리하고, 모든 신호분석 실행에서 선택적으로 사용할 수 있게 한다.

### 10.2 distributed_set quota

`distributed_set`은 class-balanced set이 아니라 split의 class distribution을 반영하는 set이다. class count를 `n_c`, 최소 class count를 `n_min`, 기준 최소 sample 수를 `m`이라고 하면 label별 ideal quota는 다음이다.

```text
q_c = (n_c / n_min) * m
```

quota rounding은 아래처럼 고정한다.

```text
floor quota 계산
남은 remainder는 fractional part가 큰 label부터 배정
fractional part tie는 label id 오름차순으로 해결
rounded quota가 실제 class count를 넘으면 오류 처리
```

manifest에는 아래 값을 기록한다.

```text
class_counts
ideal_quota
floor_quota
fractional_part
rounded_quota
total_quota
rounding_rule
sample_selection_seed
```

### 10.3 label_single sampling

`label_single`은 기준 global probe set을 제외한 나머지 split 집합에서 label별 1개 sample을 선택한다. 기준 global probe set은 기본적으로 `balanced_global`이며, 설정으로 바꿀 수 있다.

```text
candidate set = split 전체 sample - exclusion_family sample
각 label마다 candidate set에서 deterministic hash rank가 가장 작은 sample 1개 선택
```

candidate가 비어 있으면 기본 동작은 실패 처리다. 재사용을 허용하는 옵션을 두더라도 manifest에 `reused_from_exclusion_family=true`, `exclusion_family`, `reason`을 명시해야 한다.

### 10.4 probe 설정 예시

```yaml
probe:
  families:
    - balanced_global
    - distributed_set
    - label_set
    - label_single

  balanced_global:
    samples_per_label: 16

  distributed_set:
    min_class_n: 8

  label_set:
    labels: all
    samples_per_label: 16

  label_single:
    exclusion_family: balanced_global
    samples_per_label: 1
```

## 11. PSD accumulator

PSD accumulator는 time axis를 streaming하지 않는다. streaming 대상은 probe set의 sample axis다.

```text
각 batch trace: B,R,T
  ↓
T 전체에 대해 FFT/PSD 계산
  ↓
batch raw power summary 누적
  ↓
trace tensor 폐기
```

조건은 다음과 같다.

```text
각 sample의 전체 T는 보존한다.
FFT는 batch 안에서 T 전체에 대해 계산한다.
probe sample 방향 평균과 분산만 batch-wise로 누적한다.
```

기본 accumulator 상태는 다음이다.

```text
count
sum_power_curve 또는 mean_power_curve
M2_power_curve
min_power_curve
max_power_curve
optional band_energy_state
```

정확히 streaming 가능한 통계는 mean, variance, std, min, max, band energy다. 정확한 median, MAD, quantile은 sample curve 저장 또는 quantile sketch가 필요하므로 별도 옵션으로 둔다.

---

## 12. dB 변환 규칙

**dB 변환은 finalize 단계에서만 수행한다.**

분석 중간에서는 raw power만 누적한다.

```text
batch trace
  ↓
raw power PSD
  ↓
raw power sum/mean/variance 누적
  ↓
finalize
  ↓
raw mean PSD 생성
  ↓
필요한 경우 dB curve 생성
```

batch별 dB curve를 만든 뒤 평균내지 않는다.

수학적으로 다음 둘은 다르다.

```text
10 * log10(mean(raw_power))
mean(10 * log10(raw_power))
```

기본 PSD summary는 raw power curve와 dB curve를 둘 다 만들 수 있지만, dB curve는 항상 raw aggregate가 끝난 뒤 생성한다. distance 계산이 dB curve를 쓰도록 설정되어도, 그 dB curve는 finalize 이후 생성된 curve여야 한다.

---

## 13. PSD 대표화 방법

`analyze_psd`는 layer signal을 PSD curve로 만들 때 대표화 방법을 설정으로 받는다.

```yaml
analysis:
  psd:
    representative:
      method: pca
      pca:
        n_components: 3
        fit_scope: reference_checkpoint
        basis_policy: fixed_per_layer_series
```

대표화 방법은 다음과 같다.

| method | 처리 방식 | 산출물 | 주의점 |
|---|---|---|---|
| `mean` | row 축을 time-domain 평균 signal로 축소한 뒤 PSD 계산 | representative PSD curve | anti-phase row가 상쇄될 수 있음 |
| `median` | row 축을 time-domain median signal로 축소한 뒤 PSD 계산 | robust representative PSD curve | median 연산 비용과 비연속성 주의 |
| `element_psd` | row별 PSD를 먼저 계산하고 matrix 또는 power summary로 저장 | element PSD matrix, summary curve | layer 전체 spectral energy 보존에 유리 |
| `pca` | row 축에 PCA basis를 맞추고 mode signal로 투영한 뒤 mode별 PSD 계산 | PCA mode PSD, basis, explained variance | basis 고정 정책이 중요 |

`element_psd`는 대표 신호 하나로 축소하지 않는 방식이다. row별 power를 보존하고 이후 summary를 계산한다. `mean`과 `median`은 time-domain 대표 신호를 만든 뒤 PSD를 계산하므로 `element_psd`와 통계 의미가 다르다.

### 13.1 신호분석 spectral axis 정책

PSD, element PSD, PCA-PSD, 2D FFT는 분석 실행마다 `exact` 또는 `userbin` 중 하나의 spectral axis 정책만 사용한다. 같은 실행 안에서 두 축을 동시에 계산하지 않는다.

```yaml
analysis:
  spectral_axis:
    mode: userbin      # exact | userbin
    bin_reducer: mean  # mean | median
```

`exact`는 FFT가 만든 native frequency bin을 그대로 사용하는 정책이다. `userbin`은 exact frequency bin들을 사용자가 지정한 구간으로 묶은 뒤, bin 안의 raw power를 대표값으로 바꾸는 정책이다.

`userbin`의 bin 대표화는 다음 규칙을 따른다.

```text
bin membership은 frequency axis에서 결정한다.
bin 안의 raw power 값만 mean 또는 median으로 대표화한다.
dB 변환은 bin 대표화와 aggregate finalize가 끝난 뒤에만 수행한다.
```

거리 계산도 선택한 spectral axis에서만 수행한다. `exact` curve와 `userbin` curve는 직접 비교하지 않는다. artifact manifest에는 `spectral_axis.mode`, `bin_edges`, `bin_reducer`, `empty_bin_policy`를 기록한다.

empty bin 처리 기본값은 실패다. 빈 bin을 0 또는 NaN으로 채우는 정책은 별도 설정으로만 허용하고, manifest에 이유를 남긴다.


---

## 14. PCA-PSD 방법론

업로드된 논문은 LSTM hidden state activation에 PCA를 적용하고, 첫 번째 principal component에 FFT를 적용해 oscillation pattern을 분석한다. 본 프로젝트에서는 이 방법을 SNN layer signal에 맞게 일반화한다.

### 14.1 13.1 PC1 기반 기본 경로

기본 PCA 대표화는 `n_components=1`이다.

```text
Layer signal map X: S,R,T
  ↓
observation matrix: S*T,R
  ↓
PCA basis W: R,K
  ↓
mode trace Z: S,K,T
  ↓
mode별 PSD
```

`K=1`이면 PC1 대표 신호의 PSD를 계산한다. 이는 업로드 논문의 Figure 4가 설명한 FFT of first principal components와 같은 분석 흐름이다.

### 14.2 13.2 여러 component 사용

`n_components > 1`이면 PC1부터 PCK까지 mode별 PSD를 저장한다.

저장 대상은 다음이다.

```text
pca_basis
pca_explained_variance
pca_mode_trace optional
pca_mode_psd
pca_mode_band_energy
pca_cross_spectral_matrix optional
```

basis는 layer, series, scope 단위로 고정한다. epoch trend를 비교할 때 basis를 매 checkpoint마다 새로 fit하면 mode 의미가 바뀌므로 기본 정책은 reference checkpoint에서 basis를 fit하고 나머지 checkpoint에 같은 basis를 적용하는 것이다.

### 14.3 13.3 PCA 입력 series

PCA는 다음 series에 적용 가능하다.

```text
membrane_pre
decision
spike
membrane_post
rf_state_energy_pre
```

기본은 `membrane_pre` 또는 `decision`이다. spike에 PCA를 적용할 수는 있으나 binary sparsity가 강하면 PC mode 해석이 달라질 수 있으므로 별도 실험 설정으로 둔다.

---

## 15. element PSD와 2D FFT 독립 기능

### 15.1 14.1 element PSD

`element_psd`는 row, neuron, channel, spatial position별 PSD를 계산한다.

```text
Signal map: S,R,T
  ↓
PSD over T
  ↓
Element PSD: R,F 또는 S,R,F summary
```

기본 산출물은 다음이다.

```text
element PSD matrix
row frequency axis
layer summary curve
band energy per row
optional top-k rows by band energy
```

### 15.2 14.2 row-time 2D FFT

`analyze_fft2d`는 막전위 map과 spike map에 대해 row-time 2D FFT를 계산하는 독립 기능이다.

```text
Signal map: S,R,T
  ↓
aggregate over S 또는 sample별 처리
  ↓
2D FFT over R,T
  ↓
row-frequency by time-frequency matrix
```

입력 series는 다음을 기본으로 한다.

```text
membrane_pre
decision
spike
rf_state_energy_pre optional
```

2D FFT는 PSD의 보조 옵션이 아니라 독립 실행 단위다. matrix artifact는 CSV wide matrix와 axis sidecar, 또는 tensor artifact로 저장한다.

#### 2D FFT spectral axis

2D FFT도 `exact` 또는 `userbin` 중 하나만 사용한다.

```yaml
analysis:
  fft2d:
    spectral_axis:
      mode: userbin
      userbin_axes: time_frequency      # time_frequency | row_frequency | both_frequency_axes
      bin_reducer: mean                 # mean | median
      row_axis_semantics: unordered     # unordered | group_ordered | spatial | frequency_ordered | pca_mode
```

`exact`는 2D FFT 결과 matrix의 native `F_row x F_time` bin을 그대로 저장한다.

`userbin`은 원본 `R,T` map을 pooling하는 기능이 아니다. 처리 순서는 항상 다음이다.

```text
Signal map S,R,T
  ↓
2D FFT over R,T
  ↓
exact power matrix F_row,F_time
  ↓
선택한 frequency axis의 bin grouping
```

`userbin_axes`의 의미는 다음과 같다.

| userbin axis | 의미 | 기본 허용 여부 |
|---|---|---:|
| `time_frequency` | 2D FFT 결과의 time-frequency 축만 사용자 bin으로 묶음 | 예 |
| `row_frequency` | 2D FFT 결과의 row-frequency 축만 사용자 bin으로 묶음 | 조건부 |
| `both_frequency_axes` | row-frequency와 time-frequency 축을 모두 사용자 bin으로 묶음 | 조건부 |

`row_frequency`와 `both_frequency_axes`는 row axis의 의미가 명시된 경우에만 허용한다. 일반 MLP neuron index처럼 row order가 임의적인 경우에는 기본적으로 비활성화하거나 강한 경고를 낸다. 허용 가능한 row axis 의미는 `group_ordered`, `spatial`, `frequency_ordered`처럼 row 순서가 해석 가능한 경우다.

2D FFT `userbin` 대표화도 raw power에서만 수행한다. bin 대표값은 `mean` 또는 `median`이다. dB 변환은 matrix aggregate가 끝난 뒤에만 수행한다.


---

## 16. 거리 지표

거리 지표는 두 가지만 유지한다.

| metric | 정의 | 용도 |
|---|---|---|
| `centered_l2` | curve 평균을 제거한 뒤 L2 거리 계산. `$d(a,b)=\| (a-\bar a)-(b-\bar b) \|_2$` | spectral shape 비교 |
| `diff_l2` | first difference curve 간 L2 거리 계산. `$d(a,b)=\| \Delta a-\Delta b \|_2$` | peak 위치와 slope 변화 비교 |

거리 계산 대상은 finalized curve다. PCA mode가 여러 개면 mode별 distance를 저장하고, 필요하면 mode 평균 또는 Frobenius summary를 추가한다. matrix artifact의 직접 distance는 기본 기능으로 두지 않고, summary curve나 band energy를 통해 비교한다.

---

## 17. 동역학 통계 분석

파라미터 통계와 내부 상태 통계는 `analyze_dynamics` 실행 단위에서 함께 처리한다.

```text
analyze_dynamics
  ├─ checkpoint parameter vector 통계
  ├─ SpikingJelly MemoryModule state snapshot
  ├─ probe forward 기반 state trace 통계
  └─ scenario group별 통계
```

### 17.1 16.1 파라미터 벡터

각 custom cell은 `analysis_parameter_vectors()`를 제공한다.

| cell | parameter name | 의미 |
|---|---|---|
| LIF | `membrane_decay_alpha` | discrete membrane retention |
| LIF | `membrane_time_constant_tau_step` | step 단위 time constant |
| IF | `threshold` | fire threshold |
| RF | `resonant_frequency_cyc_per_step` | 공진 frequency |
| RF | `angular_frequency_rad_per_step` | angular frequency |
| RF | `damping_magnitude` | damping magnitude |
| RF | `decay_radius_rho` | discrete pole radius |
| TC-LIF | model-specific decay/coupling parameters | compartment coupling |
| DH-SNN | branch time constants | dendritic dynamics |
| D-RF | kernel pole parameters | SSM/RF filter dynamics |

저장 통계는 다음이다.

```text
count
mean
std
min
q05
q25
q50
q75
q95
max
```

scenario group이 있으면 `group_id`별 통계도 저장한다.

### 17.2 16.2 내부 상태 통계

SpikingJelly custom node는 `register_memory()`로 상태를 등록한다. 내부 상태 통계는 `base.named_memories(model)` 또는 custom state registry에서 가져온다.

| state | cell | 의미 |
|---|---|---|
| `v` | IF/LIF | membrane state |
| `x`, `y` | RF | oscillator real/imag state |
| `branch_state` | DH-SNN | dendrite branch state |
| `short_state`, `long_state` | TS-LIF | multi-timescale state |
| `ssm_state` | SSM | state-space hidden state |
| `gate_state` | SpikeGRU | recurrent gate/state |

내부 상태 snapshot은 probe forward 후 특정 시점에서 저장하거나, time trace 통계로 저장한다.

```yaml
analysis:
  dynamics:
    parameter_stats: true
    memory_state_snapshot: true
    state_trace_stats: true
    snapshot_time: final
    trace_series:
      - membrane_pre
      - membrane_post
      - rf_state_energy_pre
```

### 17.3 16.3 state trace 통계

probe forward 기반 trace 통계는 raw trace 또는 signal map에서 계산한다.

```text
mean over sample,row,time
std over sample,row,time
mean over time by row
active ratio
spike rate
silent row ratio
state energy
phase concentration for RF
```

CSV category는 모두 dynamics 계열로 묶는다.

```text
dynamics_parameter_vector
dynamics_parameter_snapshot
dynamics_parameter_trend
dynamics_memory_snapshot
dynamics_trace_stat
```

---

## 18. CSV와 tensor artifact 규격

CSV는 raw tensor 값을 직접 담는 용도가 아니라 summary, curve, distance, matrix metadata, manifest를 저장하는 용도다.

### 18.1 17.1 공통 column block

모든 CSV는 아래 block을 사용한다.

```text
artifact_id
category
source_program
status
message
created_at

dataset
run_id
seed
model_id
model_family
topology_kind
topology_scenario
cell_kind
readout_mode
checkpoint_epoch
checkpoint_path

split
scope
probe_family
label
layer_index
layer_name
signal_kind
series

method
representative
reducer
centering
scale
statistic
distance_metric

axis_name
axis_index
axis_value
axis_unit
value_name
value
value_unit
```

### 18.2 17.2 category 목록

| category | 의미 |
|---|---|
| `training_metric` | 학습 중 metric |
| `evaluation_metric` | checkpoint 평가 metric |
| `spectral_curve` | PSD curve |
| `spectral_dispersion` | PSD dispersion |
| `spectral_distance` | curve distance |
| `spectral_matrix_1d` | element PSD matrix |
| `spectral_matrix_2d` | row-time 2D FFT matrix |
| `pca_basis` | PCA basis와 explained variance |
| `pca_spectral_curve` | PCA mode PSD curve |
| `pca_spectral_matrix` | PCA mode spectral matrix |
| `dynamics_parameter_vector` | 뉴런 파라미터 원본 벡터 |
| `dynamics_parameter_snapshot` | checkpoint별 파라미터 통계 |
| `dynamics_parameter_trend` | epoch별 파라미터 변화 |
| `dynamics_memory_snapshot` | 내부 memory state snapshot |
| `dynamics_trace_stat` | probe trace 기반 내부 상태 통계 |
| `trace_manifest` | tensor trace artifact manifest |
| `artifact_manifest` | 분석 산출물 manifest |

### 18.3 17.3 matrix 저장

PSD curve, dispersion, distance, dynamics 통계는 long format을 쓴다. 2D FFT matrix는 wide matrix와 axis sidecar를 기본으로 한다.

```text
matrix.csv
matrix_column_axis.csv
matrix_row_axis.csv optional
```

`matrix_column_axis.csv`는 각 wide column의 의미를 기록한다.

```csv
artifact_id,column_name,column_index,column_value,column_unit
fft_001,time_freq_000000,0,-0.500,normalized_frequency
fft_001,time_freq_000001,1,-0.492,normalized_frequency
```

대규모 matrix는 tensor artifact로 저장하고 CSV에는 manifest만 기록할 수 있다.

---

## 19. 실험 설정 방식

실험 설정은 YAML 파일로 관리한다. bash는 실행 wrapper만 담당한다.

```yaml
experiment:
  run_id: smnist_rf_clipstructure_seed0
  seed: 0

dataset:
  name: s-mnist
  prepared_root: /path/to/prepared

model:
  topology:
    kind: mlp_stack
    hidden_widths: [256, 256, 128]
  cell:
    kind: rf
    reset:
      mode: threshold_only
    threshold:
      trainable: true
  output_cell:
    kind: if
    emit_spike: false
    reset_enabled: false
  readout:
    mode: final_if

topology:
  scenario: clipstructure
  tear: 2
  rf:
    frequency_edges: [0.0, 0.05, 0.10, 0.20, 0.50]
    damping_bounds: [0.01, 2.0]

training:
  epochs: 50
  batch_size: 256
  optimizer: adam
  lr: 0.001
  checkpoint_epochs: [1, 5, 10, 20, 30, 40, 50]

analysis:
  psd:
    enabled: true
    series: [membrane_pre, decision, spike]
    representative:
      method: pca
      pca:
        n_components: 3
    accumulator:
      stream_over_probe_batches: true
  dynamics:
    enabled: true
    parameter_stats: true
    memory_state_snapshot: true
    state_trace_stats: true
  trace_save:
    enabled: true
    series: [spike]
    dtype: uint8
    chunk_axis: sample
```

---

## 20. checkpoint 복원 정책

checkpoint는 다음 정보를 포함한다.

```text
build_spec
state_dict
optimizer_state optional
training_config
dataset_ref
readout_config
topology_config
trace_config optional
created_at
```

모델 복원은 `ModelRestoreService`가 담당한다.

```text
checkpoint load
  ↓
build_spec parse
  ↓
ModelFactory build
  ↓
state_dict load
  ↓
TraceAdapter selection
```

분석 코드는 checkpoint 내부의 model class를 직접 import하지 않는다. 모델 복원에 실패하면 분석은 중단하고 manifest에 실패 이유를 남긴다.

---

## 21. 삭제 대상

`reinterpret` 관련 기능은 실행 경로에서 제거한다.

삭제 대상은 다음이다.

```text
reinterpret 실행 스크립트
reinterpret metric category
reinterpretation 전용 config
reinterpretation plot 입력
reinterpretation 관련 private 함수 import
```

Origin 모델은 reinterpret 기능으로 재해석하지 않는다. 필요한 논문 모델은 fixed model factory로 흡수한다.

---

## 22. 구현 순서

1. `BuildSpec`, `TopologySpec`, `CellSpec`, `ReadoutSpec`, `TraceSpec`를 정의한다.
2. `LayerTraceRecord`, `LayerSignalMapRecord`, `ForwardResult`를 모델 빌더 밖으로 이동한다.
3. SpikingJelly custom IF/LIF/RF cell을 먼저 구현하고 동일 입력에 대한 trace equivalence test를 만든다.
4. MLPStackBuilder를 dense topology와 cell factory 분리 구조로 다시 작성한다.
5. MLP SRNN recurrent 연결을 hidden block 단위로 구현하고, recurrent input은 이전 시점 output spike로 고정한다.
6. feedforward mask와 recurrent mask를 모두 `structure` constraint에서 생성한다.
7. `final_if`, `final_mem` readout을 구현한다.
8. TraceAdapter와 SignalMapEmitter를 분리한다.
9. `ProbeSetBuilder`에 `balanced_global`, `distributed_set`, `label_set`, `label_single`을 구현한다.
10. PSDAccumulator를 probe batch streaming 방식으로 구현한다.
11. dB 변환을 finalize 전용 처리로 고정한다.
12. `exact`, `userbin` spectral axis와 `mean`, `median` bin reducer를 구현한다.
13. `mean`, `median`, `element_psd`, `pca` representative를 구현한다.
14. `analyze_element_psd`, `analyze_fft2d`, `analyze_pca_psd`를 독립 실행 단위로 구현한다.
15. `analyze_dynamics`에서 파라미터 통계와 내부 상태 통계를 함께 처리한다.
16. time-domain spike trace artifact writer를 추가한다.
17. CSV writer와 manifest writer를 artifact category 기준으로 정리한다.
18. VGG, ResNet, S4/SSM, GRU, Spike-Transformer fixed factory와 trace adapter를 구현한다.
19. plot은 CSV/tensor artifact reader만 사용하게 재구성한다.

---

## 23. 성공 기준

리팩터링 성공 기준은 다음이다.

```text
MLP hidden topology를 바꿔도 fixed model 코드가 바뀌지 않는다.
새 cell을 MLP에 추가할 때는 cell factory와 custom cell만 추가한다.
MLP SRNN recurrent는 이전 시점 output spike만 recurrent input으로 사용한다.
VGG, ResNet, S4/SSM, GRU, Spike-Transformer를 추가해도 PSD 분석 코드는 바뀌지 않는다.
SpikingJelly state reset이 train/eval/analysis에서 일관되게 수행된다.
TraceAdapter가 모든 raw trace를 B,T,* layout으로 정규화한다.
SignalMapEmitter가 모든 PSD 입력을 S,R,T layout으로 정규화한다.
ProbeSetBuilder가 balanced_global, distributed_set, label_set, label_single manifest를 생성한다.
PSD는 probe sample axis에서만 streaming하고 time axis를 자르지 않는다.
신호분석은 exact 또는 userbin 중 선택한 spectral axis에서만 수행된다.
userbin bin 대표화는 raw power에서 mean 또는 median으로 수행된다.
dB 변환은 finalize에서만 수행된다.
시간영역 spike output은 sample-axis chunk로 저장된다.
파라미터 통계와 내부 상태 통계는 같은 실행 단위에서 관리된다.
거리 지표는 centered_l2와 diff_l2만 사용한다.
reinterpret 기능은 실행 경로에 남지 않는다.
```

---

## 24. 용어 고정

| 용어 | 정의 |
|---|---|
| raw trace | 모델 forward에서 나온 `B,T,*` 시계열 tensor |
| signal map | 분석용 `S,R,T` tensor |
| series | `membrane_pre`, `decision`, `spike` 같은 trace 종류 |
| representative | PSD 계산 전에 layer row 축을 처리하는 방식 |
| element PSD | row별 PSD를 보존하는 matrix 분석 |
| dynamics stats | 뉴런 파라미터와 내부 상태 통계를 함께 다루는 분석 |
| no-spike output cell | fire와 reset을 끈 output cell |
| final readout | 마지막 시점 membrane을 logits로 쓰는 readout |

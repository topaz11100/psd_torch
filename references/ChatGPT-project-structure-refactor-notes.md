# 프로젝트 구조 개선 방안

**User:** Anonymous  
**Created:** 5/22/2026 21:04:24  
**Updated:** 5/23/2026 2:22:33  
**Exported:** 5/23/2026 2:29:56  
**Link:** [https://chatgpt.com/c/6a104647-2e38-8322-8a99-06e5343e4ea5](https://chatgpt.com/c/6a104647-2e38-8322-8a99-06e5343e4ea5)  

## Prompt:
2026. 5. 22. 오후 9:04:22

프로젝트 구조 개선 방안이 있을까?
일단 프로젝트 모델 학습에
- 예전 프로젝트의 clip, structure, clipstructure 넣어서 학습하기
- mlp 토폴로지 모델에 대해 막전위(스파이크 결정변수, 특히 lif 에선 막전위 감쇠계수, rf 에선 공진주파수, 감쇠계수) 통계 분석 기능 추가
- psd csv 규격이 너무 난잡한거같은데 어떻게 정리하지?
어떻게 고칠지(대규모 리팩터링할거임) 설명해줘

[Attachment: psd(57).zip]

[Attachment: psd_old(1).zip]

## Response:
2026. 5. 22. 오후 9:04:25

먼저 올려준 두 ZIP의 실제 폴더 구조와 CSV 산출물 형식을 확인한 뒤, 모델·실험·분석·산출물 경계를 나누는 대규모 리팩터링안으로 정리하겠습니다.

## Response:
2026. 5. 22. 오후 9:04:28

현재 구조는 큰 단계 분리는 되어 있지만, 모델 변형이 registry/builder 밖에 흩어지고 CSV가 scalar·matrix·manifest를 섞어 관리 부담이 커진 상태입니다.

## Response:
2026. 5. 22. 오후 9:15:13

## 결론

대규모 리팩터링 방향은 이렇게 잡는 게 맞습니다.

**"학습", "모델 구성", "체크포인트 복원", "probe 수집", "분석", "CSV 저장", "plotting", "job launch"를 완전히 분리해야 합니다.** 

지금 프로젝트는 예전보다 많이 정리되어 있지만, 아직 핵심 로직이 `src/psd_analysis.py`, `src/model_training.py`, `src/model/snn_builder.py`, `src/util/csv_schema.py`, `bash/*.sh`에 너무 많이 몰려 있습니다. 특히 현재 구조에서 가장 위험한 부분은 다음입니다.

| 문제 | 현재 상태 | 리팩터링 방향 |
|---|---|---|
| old project의 `clip`, `structure`, `clipstructure` | 현재 `LIFLayer`, `RFLayer`에는 mask/bound 인자가 이미 있지만 builder가 거의 사용하지 않음 | old code를 통째로 복사하지 말고 `TopologySpec`, `ConstraintSpec`으로 흡수 |
| MLP topology 모델 분석 | `filter_snapshot`, `filter_trend`가 있으나 이름과 의미가 불명확 | `neuron_parameter_snapshot`, `neuron_parameter_trend`, `membrane_trace_stat`로 분리 |
| PSD CSV 규격 | category 기반이지만 여전히 category 수가 많고 matrix CSV가 난잡함 | artifact type 기준으로 `scalar`, `curve`, `matrix`, `manifest`를 분리 |
| 분석 공통 로직 | `analysis_matrix_common.py`가 `src.psd_analysis`의 private 함수 `_...`를 가져다 씀 | checkpoint/probe/model-restore 공통 모듈로 빼야 함 |
| bash | dataset별 hardcoded script가 많음 | bash는 얇은 wrapper, 실험 설정은 YAML/JSON config로 이동 |
| Spec과 실제 구조 | `structure.txt`에는 `Spec/theory`, `Spec/README.md` 등이 있지만 실제 zip에는 `Spec/impl/spec`만 있음 | Spec 디렉터리도 실제 구조와 맞게 정리 |

---

# 1. 추천 최종 구조

지금처럼 `src/` 아래에 바로 `model_training.py`, `psd_analysis.py`가 있는 구조는 유지보수에는 불리합니다. 패키지 이름을 명확히 주는 게 좋습니다.

현재:

```text
src/
  model_training.py
  psd_analysis.py
  element_psd.py
  2d_fft_analysis.py
  model/
  neurons/
  signal/
  stat/
  util/
```

추천:

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
        mnist.py
        s_mnist.py
        shd.py
        dvs.py
        uci.py

    models/
      tokens.py
      topology.py
      constraints.py
      builder.py
      checkpoint_io.py
      introspection.py
      training.py
      regularizers.py
      schedules.py

    neurons/
      lif.py
      rf.py
      tc_lif.py
      ts_lif.py
      dh_snn.py
      d_rf.py
      common.py

    readout/
      readout.py
      first_spike.py

    analysis/
      common/
        checkpoint_loader.py
        model_restore.py
        probe.py
        capture.py
        artifact_paths.py

      spectral/
        psd_curve.py
        psd_dispersion.py
        element_psd.py
        fft2d.py
        distance.py

      neuron_stats/
        parameter_vectors.py
        parameter_stats.py
        membrane_trace_stats.py

      accuracy/
        checkpoint_accuracy.py

    artifacts/
      schema/
        base.py
        registry.py
        categories.py
        validate.py
      writers/
        scalar_csv.py
        curve_csv.py
        matrix_csv.py
        manifest_csv.py

    plotting/
      readers.py
      renderers.py
      figures.py

    cli/
      data_prep.py
      train.py
      analyze_psd.py
      analyze_element_psd.py
      analyze_fft2d.py
      analyze_neuron_stats.py
      analyze_accuracy.py
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
  author_code/

Spec/
  impl/
  theory/
  traceability.md
  conflict.md

tests/
  test_model_tokens.py
  test_topology_constraints.py
  test_checkpoint_roundtrip.py
  test_csv_schema.py
  test_neuron_stats.py
```

핵심은 `cli/`는 입구만 담당하고, 실제 로직은 `models/`, `analysis/`, `artifacts/` 아래로 내려야 한다는 것입니다.

---

# 2. old project의 `clip`, `structure`, `clipstructure` 넣는 방식

## 2.1 old code를 그대로 복사하면 안 됨

예전 프로젝트에는 대략 이런 variant가 있었습니다.

```text
rf
rf_struct
rf_clip
rf_structclip

lif
lif_struct
lif_clip
lif_structclip
```

이걸 현재 프로젝트에 그대로 model token으로 박아 넣으면 다시 난잡해집니다. 이유는 간단합니다.

현재 model token은 이미 다음 정보를 담고 있습니다.

```text
lif_soft_fixed
lif_R_soft_fixed
rf_soft_fixed
rf_R_soft_fixed
vgg11_lif_soft_fixed
...
```

여기에 또 `struct`, `clip`, `structclip`, `tear`, `band_neuron_ends`, `alpha_clip_edges`, `w_clip_edges`까지 넣으면 token이 실험 설정 파일처럼 변합니다.

따라서 **model family와 topology constraint를 분리해야 합니다.** 

---

## 2.2 추천 개념 분리

### 모델 자체

```text
model_family = lif
reset_mode = soft_reset
threshold = fixed
recurrent = false
hidden = [128,64,64]
```

### topology 제약

```text
topology_variant = none | structure | clip | structclip
tear = 2
band_neuron_ends = ...
lif_alpha_edges = ...
rf_frequency_edges = ...
rf_damping_edges = ...
```

즉, 모델은 "무슨 뉴런인가"를 말하고, topology는 "연결과 결정변수 범위를 어떻게 제한할 것인가"를 말해야 합니다.

---

## 2.3 새 dataclass 설계

```python
@dataclass(frozen=True)
class TopologySpec:
    variant: str
    tear: int = 1
    band_neuron_ends: tuple[tuple[int, ...], ...] | None = None

    lif_alpha_edges: tuple[float, ...] | None = None
    rf_frequency_edges: tuple[float, ...] | None = None
    rf_damping_edges: tuple[float, ...] | None = None

    apply_to_output_layer: bool = False
```

그리고 layer 생성 직전에 이것으로부터 mask와 bound를 만듭니다.

```python
@dataclass(frozen=True)
class LayerConstraint:
    input_mask: torch.Tensor | None
    recurrent_mask: torch.Tensor | None
    alpha_bounds: tuple[torch.Tensor, torch.Tensor] | None
    frequency_bounds: tuple[torch.Tensor, torch.Tensor] | None
    damping_bounds: tuple[torch.Tensor, torch.Tensor] | None
    group_ids: torch.Tensor | None
```

---

## 2.4 `structure`의 의미

`structure`는 뉴런을 band/group으로 나누고, 서로 같은 group끼리만 연결하게 만드는 것입니다.

예를 들어 hidden layer 1과 hidden layer 2가 있을 때,

```text
hidden_1 neuron group: [0,0,0,1,1,1,2,2,2]
hidden_2 neuron group: [0,0,1,1,2,2]
```

이면 hidden_1에서 hidden_2로 가는 weight mask는 다음 조건으로 만듭니다.

$$
M_{ij}=1
$$

단,

$$
g_i^{target}=g_j^{source}
$$

일 때만 1이고, 아니면 0입니다.

코드 개념은 이렇습니다.

```python
input_mask = curr_groups[:, None] == prev_groups[None, :]
```

recurrent LIF/RF이면 recurrent mask도 만들 수 있습니다.

```python
recurrent_mask = curr_groups[:, None] == curr_groups[None, :]
```

주의할 점은 **첫 번째 hidden layer** 입니다. 첫 hidden layer의 입력은 원본 input입니다. MNIST flatten input이나 SHD input neuron에 group을 명확히 주지 않으면 첫 hidden layer에는 input mask를 걸 수 없습니다. 그래서 old project처럼 보통 hidden-to-hidden부터 structure를 적용하는 게 안전합니다.

---

## 2.5 `clip`의 의미

`clip`은 뉴런별 spike 결정변수 관련 파라미터의 허용 범위를 band/group별로 제한하는 것입니다.

### LIF

LIF에서 핵심 파라미터는 membrane decay coefficient입니다.

$$
v_t=\alpha v_{t-1}+I_t
$$

여기서 $\alpha$가 클수록 과거 막전위가 오래 남습니다. 즉, 장기의존성(long-term dependency)에 직접 관련됩니다.

따라서 LIF clip은 group별로 $\alpha$ 범위를 다르게 줍니다.

예:

```text
lif_alpha_edges = [0.0,0.25,0.5,0.75,1.0]
```

이면 group 0은 $[0.0,0.25]$, group 1은 $[0.25,0.5]$, group 2는 $[0.5,0.75]$, group 3은 $[0.75,1.0]$ 범위를 갖습니다.

현재 `LIFLayer`에는 이미 `alpha_bounds` 인자가 있습니다. 그러므로 새로 뉴런을 뜯어고칠 필요는 적고, builder에서 bound를 넘기면 됩니다.

```python
LIFLayer(
    ...,
    alpha_bounds=(alpha_lower, alpha_upper),
    input_mask=input_mask,
    recurrent_mask=recurrent_mask,
)
```

### RF

RF에서는 핵심 파라미터가 두 개입니다.

| 파라미터 | 의미 |
|---|---|
| `center_frequency` | 공진주파수 |
| `damping` | 감쇠계수 |
| `rho = exp(-damping)` | 실제 시간 방향 decay radius |

현재 `RFLayer`에는 이미 `frequency_bounds`와 `damping_magnitude_bounds`가 있습니다. old project는 주로 frequency clip을 썼지만, 지금 연구 목적상 damping도 통계 분석 대상에 넣는 게 맞습니다.

RF clip은 기본적으로 이렇게 갑니다.

```python
RFLayer(
    ...,
    frequency_bounds=(freq_lower, freq_upper),
    damping_magnitude_bounds=(damping_lower, damping_upper),
    input_mask=input_mask,
    recurrent_mask=recurrent_mask,
)
```

다만 damping을 group별로 다르게 clip할지, global bound만 줄지는 연구 가설에 따라 다릅니다.

제 추천은 다음입니다.

| 항목 | 추천 |
|---|---|
| RF frequency | group별 clip 적용 |
| RF damping | 처음에는 global bound만 적용하고 통계 분석 |
| RF damping group clip | 별도 ablation으로 추가 |

---

## 2.6 `structclip`

`structclip`은 단순합니다.

```text
structure mask + parameter clip
```

즉, 같은 group끼리만 연결하고, 동시에 group별로 LIF alpha 또는 RF frequency 범위를 제한합니다.

---

## 2.7 CLI는 이렇게 바꾸는 게 좋음

현재처럼 model token에 모든 걸 넣기보다, 아래처럼 분리하는 게 좋습니다.

```bash
python -m psd_snn.cli.train \
  --dataset s-mnist \
  --model lif_soft_fixed \
  --hidden_spec 128,64,64 \
  --topology_variant structclip \
  --tear 2 \
  --lif_alpha_edges 0.0 0.25 0.5 0.75 1.0 \
  --band_neuron_ends auto \
  ...
```

RF는 이렇게 갑니다.

```bash
python -m psd_snn.cli.train \
  --dataset shd \
  --model rf_soft_fixed \
  --hidden_spec 128,64,64 \
  --topology_variant structclip \
  --tear 2 \
  --rf_frequency_edges 0.0 0.05 0.10 0.20 0.50 \
  --rf_damping_bounds 0.1 1.0 \
  ...
```

더 좋은 방식은 CLI가 아니라 config 파일입니다.

```yaml
dataset: s-mnist

model:
  token: lif_soft_fixed
  hidden: [128, 64, 64]
  readout_mode: temporal_membrane

topology:
  variant: structclip
  tear: 2
  band_neuron_ends: auto
  lif_alpha_edges: [0.0, 0.25, 0.5, 0.75, 1.0]
  apply_to_output_layer: false

training:
  epochs: 50
  batch_size: 256
  lr: 0.005
  seed: 0
  checkpoint_epochs: [1, 5, 10, 20, 30, 40, 50]
```

---

# 3. MLP topology 모델의 막전위/뉴런 파라미터 통계 분석

여기서 반드시 구분해야 할 것이 있습니다.

## 3.1 "막전위"와 "spike 결정변수"는 같은 말이 아님

현재 `LIFLayer`와 `RFLayer` 코드를 보면 일반적인 spiking hidden layer에서는 저장되는 `membrane`이 실제 raw membrane일 수도 있고, threshold를 뺀 spike decision signal일 수도 있습니다.

LIF 쪽 현재 forward 개념은 이렇습니다.

$$
v_t^{pre}=\alpha v_{t-1}+I_t
$$

spike decision은 다음입니다.

$$
d_t=v_t^{pre}-v_{th}
$$

현재 코드에서는 대체로 `membrane_signal = membrane_pre - threshold`를 기록합니다. 그러면 이름은 `membrane`이지만 실제 의미는 "threshold를 뺀 spike 결정변수"입니다.

이건 논문/분석에서 혼동을 만듭니다. 반드시 이름을 분리해야 합니다.

추천 trace 이름은 다음입니다.

| 이름 | 의미 |
|---|---|
| `input_current` | layer에 들어간 current |
| `membrane_pre` | reset/spike 전 막전위 |
| `decision` | spike 결정변수, 보통 $v_t-v_{th}$ |
| `spike` | spike output |
| `membrane_post` | reset 적용 후 막전위 |
| `readout_membrane` | readout용 누적 membrane |

현재 `series=membrane`은 너무 애매합니다. 리팩터링 후에는 `decision`으로 바꾸는 게 맞습니다. 기존 호환성을 위해 v1 CSV에서는 `membrane`, v2 CSV에서는 `decision`을 쓰면 됩니다.

---

## 3.2 분석 category를 세 개로 나누기

현재 `filter_snapshot`, `filter_trend`는 이름이 부정확합니다. LIF alpha, RF frequency, RF damping은 filter라기보다 **neuron dynamic parameter**입니다.

따라서 다음 category로 바꾸는 게 좋습니다.

| 새 category | 의미 | forward 필요 여부 |
|---|---|---|
| `neuron_parameter_vector` | 뉴런별 파라미터 원본값 | 필요 없음 |
| `neuron_parameter_snapshot` | checkpoint 하나에서 layer별 파라미터 통계 | 필요 없음 |
| `neuron_parameter_trend` | epoch에 따른 파라미터 통계 변화 | 필요 없음 |
| `membrane_trace_stat` | probe forward로 얻은 membrane/decision/spike trace 통계 | 필요 |
| `membrane_trace_distribution` | 선택적, per-neuron 또는 per-time 분포 | 필요 |

---

## 3.3 뉴런 parameter API 추가

각 뉴런 layer가 자기 파라미터를 표준 형식으로 노출하게 해야 합니다.

예시:

```python
@dataclass(frozen=True)
class NeuronParameterVector:
    name: str
    role: str
    unit: str
    values: torch.Tensor
    lower_bound: torch.Tensor | None = None
    upper_bound: torch.Tensor | None = None
    group_ids: torch.Tensor | None = None
```

LIF는 이렇게 반환합니다.

```python
def analysis_parameter_vectors(self):
    alpha = self.effective_alpha().detach()
    tau = -1.0 / torch.log(torch.clamp(alpha, min=1e-6, max=1.0 - 1e-6))

    return [
        NeuronParameterVector(
            name="membrane_decay_alpha",
            role="lif_membrane_decay",
            unit="dimensionless",
            values=alpha,
            lower_bound=self.alpha_lower,
            upper_bound=self.alpha_upper,
            group_ids=getattr(self, "group_ids", None),
        ),
        NeuronParameterVector(
            name="membrane_time_constant_tau_step",
            role="lif_time_constant",
            unit="step",
            values=tau,
            group_ids=getattr(self, "group_ids", None),
        ),
    ]
```

RF는 이렇게 반환합니다.

```python
def analysis_parameter_vectors(self):
    frequency = self.effective_frequency().detach()
    damping = self.effective_damping_magnitude().detach()
    rho = torch.exp(-damping)

    return [
        NeuronParameterVector(
            name="resonant_frequency_cyc_per_step",
            role="rf_resonant_frequency",
            unit="cycle_per_step",
            values=frequency,
            lower_bound=self.freq_lower,
            upper_bound=self.freq_upper,
            group_ids=getattr(self, "group_ids", None),
        ),
        NeuronParameterVector(
            name="damping_magnitude",
            role="rf_damping",
            unit="inverse_step",
            values=damping,
            group_ids=getattr(self, "group_ids", None),
        ),
        NeuronParameterVector(
            name="decay_radius_rho",
            role="rf_decay_radius",
            unit="dimensionless",
            values=rho,
            group_ids=getattr(self, "group_ids", None),
        ),
    ]
```

기존 `filter_stats_vectors()`는 당장 지우지 말고 alias로 남기면 됩니다.

```python
def filter_stats_vectors(self):
    return {
        item.name: item.values
        for item in self.analysis_parameter_vectors()
    }
```

---

## 3.4 저장할 통계

기본 통계는 다음이면 충분합니다.

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

topology 모델이면 group별 통계도 저장해야 합니다.

예:

```text
layer_1 group_0 membrane_decay_alpha mean
layer_1 group_1 membrane_decay_alpha mean
layer_1 group_2 membrane_decay_alpha mean
layer_1 group_3 membrane_decay_alpha mean
```

이렇게 해야 "low-frequency band 뉴런이 진짜 높은 alpha를 학습했는가?" 같은 질문에 답할 수 있습니다.

---

## 3.5 `neuron_parameter_snapshot.csv` 예시

```csv
schema_version,category,source_program,status,message,dataset,run_id,created_at,model_token,topology_variant,checkpoint_epoch,layer_index,layer_name,group_id,parameter_name,parameter_role,statistic,value,value_unit,lower_bound,upper_bound
psd_csv_v2,neuron_parameter_snapshot,neuron_stats,ok,,s-mnist,run_001,2026-05-22T00:00:00Z,lif_soft_fixed,structclip,50,1,layer_01,0,membrane_decay_alpha,lif_membrane_decay,mean,0.181,dimensionless,0.0,0.25
```

---

## 3.6 `membrane_trace_stat.csv` 예시

```csv
schema_version,category,source_program,status,message,dataset,run_id,created_at,model_token,topology_variant,checkpoint_epoch,split,scope,layer_index,layer_name,signal_name,series,statistic,value,value_unit
psd_csv_v2,membrane_trace_stat,neuron_stats,ok,,s-mnist,run_001,2026-05-22T00:00:00Z,lif_soft_fixed,structclip,50,test,test_balanced_global,1,layer_01,hidden,decision,mean,-0.132,voltage_relative
```

여기서 `series=decision`은 $v_t-v_{th}$입니다. 이걸 `membrane`이라고 부르면 나중에 해석이 꼬입니다.

---

# 4. PSD CSV 규격 정리

현재 `src/util/csv_schema.py`는 한 곳에서 category를 관리한다는 점은 좋습니다. 하지만 category가 너무 많고, 일부 category 이름이 producer 중심입니다.

예:

```text
analysis_curve
analysis_dispersion
analysis_2d_fft
element_psd
filter_snapshot
filter_trend
checkpoint_accuracy
pair_distance
layer_distance_profile
layer_distance_trend
...
```

문제는 이겁니다.

1. `analysis_2d_fft`는 "무엇을 저장하는가"보다 "어느 프로그램이 만들었는가"에 가깝습니다.
2. `filter_snapshot`은 LIF alpha/RF frequency/damping을 설명하기에 부정확합니다.
3. matrix 계열은 `freq_000000`, `time_freq_000000` 같은 동적 column을 만들어서 parser가 까다로워집니다.
4. 파일명이 너무 많은 정보를 담고 있습니다. 과학적 metadata는 파일명이 아니라 CSV column과 manifest에 있어야 합니다.
5. `write_common_csv(..., extra_columns=...)`가 너무 관대하면 schema가 다시 흐트러집니다.

---

## 4.1 category를 artifact 의미 기준으로 줄이기

추천 category는 다음 정도입니다.

| 새 category | 기존 category |
|---|---|
| `training_metric` | 유지 |
| `evaluation_metric` | `checkpoint_accuracy`, `accuracy_loss_join` |
| `spectral_curve` | `dataset_curve`, `analysis_curve` |
| `spectral_dispersion` | `dataset_dispersion`, `analysis_dispersion` |
| `spectral_distance` | `pair_distance`, `layer_distance_profile`, `layer_distance_trend`, `drift_distance` |
| `spectral_matrix_1d` | `element_psd` |
| `spectral_matrix_2d` | `analysis_2d_fft` |
| `neuron_parameter_vector` | 신규 |
| `neuron_parameter_snapshot` | 기존 `filter_snapshot` 대체 |
| `neuron_parameter_trend` | 기존 `filter_trend` 대체 |
| `membrane_trace_stat` | 신규 |
| `manifest` | `analysis_manifest`, `dataset_psd_manifest`, `plotting_manifest` 통합 |

이렇게 하면 "어느 프로그램이 만들었나"는 `source_program` column으로 남기고, category는 artifact의 의미만 나타냅니다.

---

## 4.2 CSV column을 5개 block으로 고정

모든 CSV는 다음 block 구조를 따르게 하는 게 좋습니다.

```text
[공통 block]
[실험 block]
[probe/layer block]
[analysis block]
[value/coordinate block]
```

### 공통 block

```text
schema_version
category
artifact_id
source_program
status
message
created_at
```

### 실험 block

```text
dataset
run_id
seed
model_token
model_family
readout_mode
topology_variant
topology_id
checkpoint_epoch
checkpoint_path
```

### probe/layer block

```text
split
scope
probe_family
label
sample_role
sample_index
layer_index
layer_name
signal_kind
series
```

### analysis block

```text
method
extractor
reducer
variant
scale
statistic
distance_metric
comparison_type
```

### value/coordinate block

curve/stat 계열:

```text
frequency_index
frequency
frequency_unit
value_name
value
value_unit
```

matrix 계열:

```text
row_index
row_value
row_unit
column_axis_name
column_axis_unit
value_unit
matrix columns...
```

---

## 4.3 matrix CSV는 두 가지 중 하나를 선택

### 선택 A: 완전 long format

2D matrix도 이렇게 저장합니다.

```csv
row_index,column_index,row_frequency,time_frequency,value
```

장점:

- parser가 제일 단순함
- schema가 깔끔함
- plotting이 쉬움

단점:

- 파일 크기가 커짐
- 2D FFT matrix가 크면 row 수가 폭발함

### 선택 B: matrix는 wide 유지, 단 axis sidecar 추가

현재처럼:

```csv
row_frequency_index,row_frequency,time_freq_000000,time_freq_000001,...
```

형식을 유지하되, 반드시 sidecar CSV를 둡니다.

```text
matrix.csv
matrix_column_axis.csv
```

`matrix_column_axis.csv`:

```csv
artifact_id,column_name,column_index,column_value,column_unit
abc123,time_freq_000000,0,-0.5,normalized_frequency
abc123,time_freq_000001,1,-0.49,normalized_frequency
```

이러면 wide matrix의 효율은 유지하면서도 column 의미가 명확해집니다.

제 추천은 다음입니다.

| artifact | 추천 |
|---|---|
| PSD curve | long format |
| dispersion | long format |
| distance | long format |
| neuron parameter stats | long format |
| element PSD | long format 또는 partitioned long |
| 2D FFT | wide matrix + axis sidecar |

---

## 4.4 파일명은 짧게, 의미는 CSV에 넣기

현재는 파일명에 너무 많은 정보가 들어갑니다.

예:

```text
analysis_curve__epoch_1__layer_1__train_balanced_global__hidden__membrane__psd_exact__mean__raw__db.csv
```

이 방식은 처음에는 편하지만, 나중에는 유지보수가 매우 어렵습니다.

추천:

```text
outputs/
  runs/
    run_20260522_001/
      checkpoints/
      metrics/
        training_metric.csv

      analysis/
        psd/
          manifest.csv
          spectral_curve/
            checkpoint_epoch=000001/
              layer=001/
                part.csv
          spectral_dispersion/
            checkpoint_epoch=000001/
              layer=001/
                part.csv

        neuron_stats/
          neuron_parameter_snapshot.csv
          neuron_parameter_trend.csv
          membrane_trace_stat.csv

        matrices/
          spectral_matrix_2d/
            checkpoint_epoch=000001/
              layer=001/
                matrix_000001.csv
                matrix_000001_column_axis.csv
```

파일명은 짧게 두고, 실제 의미는 CSV column에 넣는 게 맞습니다.

---

# 5. checkpoint metadata도 정리해야 함

topology와 CSV 분석을 제대로 하려면 checkpoint 안에 다음 정보가 반드시 들어가야 합니다.

```python
checkpoint = {
    "schema_version": "checkpoint_v2",
    "epoch": epoch,
    "model_token": model_spec.canonical_token,
    "model_config": {
        "input_dim": ...,
        "sequence_length": ...,
        "hidden_spec": ...,
        "v_th": ...,
        "topology_config": {
            "variant": "structclip",
            "tear": 2,
            "band_neuron_ends": ...,
            "lif_alpha_edges": ...,
            "rf_frequency_edges": ...,
            "rf_damping_edges": ...,
            "topology_id": "...",
        },
    },
    "readout_config": ...,
    "dataset_token": ...,
    "prepared_data_ref": ...,
    "axis_metadata_ref": ...,
    "state_dict": ...,
}
```

특히 `topology_id`를 넣는 게 좋습니다. topology config를 canonical JSON으로 만든 뒤 hash를 뜨면 됩니다.

```text
topology_id = sha1(canonical_topology_json)[:12]
```

그러면 CSV, plot, checkpoint를 서로 연결하기 쉬워집니다.

---

# 6. 현재 코드에서 바로 고쳐야 할 구체적 지점

## 6.1 `analysis_matrix_common.py`가 `src.psd_analysis` private 함수를 쓰는 문제

현재 구조:

```python
import src.psd_analysis as psd_common

psd_common._load_checkpoint(...)
psd_common._resolve_checkpoint_files(...)
psd_common._build_model_from_checkpoint(...)
psd_common._checkpoint_common_base(...)
```

이건 좋지 않습니다. `_`로 시작하는 함수는 그 모듈 내부용이라는 뜻인데, 다른 분석 모듈이 의존하고 있습니다.

고쳐야 할 구조:

```text
analysis/common/checkpoint_loader.py
  load_checkpoint()
  resolve_checkpoint_files()

analysis/common/model_restore.py
  build_model_from_checkpoint()

analysis/common/probe.py
  build_probe_subsets()
  collect_probe_maps()

analysis/common/artifact_paths.py
  safe_token()
  layer_folder()
  checkpoint_output_dir()
```

그러면 `psd_analysis.py`, `element_psd.py`, `2d_fft_analysis.py`, `neuron_stats.py`, `checkpoint_accuracy.py`가 같은 공통 모듈을 씁니다.

---

## 6.2 `src/model/snn_builder.py`가 너무 큼

현재 `snn_builder.py`가 1000줄 이상입니다. 여기에 topology까지 넣으면 더 망가집니다.

분리 추천:

```text
models/
  builder.py
  dense_builder.py
  cnn_builder.py
  author_backbone_builder.py
  topology.py
  constraints.py
  metadata.py
```

특히 topology 관련은 `dense_builder.py` 안에 넣지 말고 `constraints.py`에서 만들어 넘겨야 합니다.

```python
constraints = build_dense_layer_constraints(
    topology_spec=topology_spec,
    hidden_sizes=hidden_sizes,
    family=model_spec.family,
    recurrent=model_spec.recurrent,
)

for layer_index in range(num_layers):
    layer_constraint = constraints[layer_index]
    layer = build_dense_layer(..., constraint=layer_constraint)
```

---

## 6.3 `data/preprocessing.py`도 너무 큼

현재 `data/preprocessing.py`가 약 1800줄입니다. dataset별 preprocessing은 분리하는 게 좋습니다.

추천:

```text
data/preprocessing/
  common.py
  mnist.py
  s_mnist.py
  ps_mnist.py
  shd.py
  ssc.py
  dvs128_gesture.py
  cifar10_dvs.py
  deap.py
  uci.py
```

`data_prep.py`는 registry만 보고 dispatch해야 합니다.

---

## 6.4 bash 정리

현재 bash에는 dataset별 hardcoded script가 많습니다.

예:

```text
model_training_s-mnist.sh
model_training_shd.sh
element_psd_s-mnist.sh
element_psd_shd.sh
2d_fft_analysis_s-mnist.sh
2d_fft_analysis_shd.sh
run_plot1.sh
run_plot2.sh
...
```

이건 오래 가면 반드시 터집니다.

추천:

```text
bash/
  train.sh
  analyze.sh
  plot.sh
```

그리고 실험은 config로 둡니다.

```text
configs/experiments/
  smnist_lif_baseline.yaml
  smnist_lif_structclip.yaml
  shd_rf_structclip.yaml
```

실행:

```bash
python -m psd_snn.scripts.launch \
  --config configs/experiments/smnist_lif_structclip.yaml \
  --stage train
```

또는 얇은 bash:

```bash
bash bash/train.sh configs/experiments/smnist_lif_structclip.yaml
```

bash가 실험 조합을 직접 만들지 않게 해야 합니다.

---

## 6.5 현재 bash에서 보이는 즉시 수정 사항

현재 zip 기준으로 눈에 띄는 부분입니다.

```bash
REGULARIZATION_SET_RAW="0|0.001|y_spike|exact|db|raw|mean 0|0|y_spike|exact|db|raw|mean 0|-0.001|y_spike|exact|db|raw|mean}"
```

마지막에 `}`가 붙어 있습니다. 이건 파싱 오류나 이상한 reducer 이름으로 이어질 수 있습니다.

또한 `Spec`은 generic launcher를 말하지만 실제 파일은 다음처럼 dataset-specific입니다.

```text
element_psd_mnist.sh
element_psd_s-mnist.sh
element_psd_shd.sh
2d_fft_analysis_s-mnist.sh
2d_fft_analysis_shd.sh
```

Spec과 실제 파일명이 맞지 않습니다. 리팩터링할 때 generic launcher 하나로 줄이는 게 맞습니다.

---

# 7. 리팩터링 순서

한 번에 다 갈아엎으면 위험합니다. 다음 순서가 안전합니다.

## Phase 1. 공통 분석 모듈 분리

먼저 동작을 바꾸지 말고 이동만 합니다.

```text
src/psd_analysis.py 내부 private 함수
  -> analysis/common/checkpoint_loader.py
  -> analysis/common/model_restore.py
  -> analysis/common/probe.py
  -> analysis/common/artifact_paths.py
```

완료 기준:

- `psd_analysis.py` 결과가 이전과 같음
- `element_psd.py` 결과가 이전과 같음
- `2d_fft_analysis.py` 결과가 이전과 같음
- private import 제거

---

## Phase 2. CSV schema v2 설계

`src/util/csv_schema.py`를 다음처럼 분리합니다.

```text
artifacts/schema/base.py
artifacts/schema/categories.py
artifacts/schema/registry.py
artifacts/schema/validate.py
artifacts/writers/scalar_csv.py
artifacts/writers/curve_csv.py
artifacts/writers/matrix_csv.py
artifacts/writers/manifest_csv.py
```

완료 기준:

- v1 CSV writer는 유지
- v2 CSV writer 추가
- `--schema_version v1|v2`로 전환 가능
- v2 validation test 통과

---

## Phase 3. topology spec 추가

old project의 `psd_model_variants.py`에서 필요한 개념만 가져옵니다.

가져올 것:

```text
validate_rf_clip_edges
validate_lif_clip_edges
default_band_neuron_ends
parse_band_neuron_ends
group_ids_from_ends
groups_from_cli
validate_tear
```

하지만 위치는 새로 둡니다.

```text
models/topology.py
models/constraints.py
```

완료 기준:

- `lif_soft_fixed + topology_variant=none`은 기존과 완전히 동일
- `lif_soft_fixed + topology_variant=clip`은 alpha bound 적용
- `rf_soft_fixed + topology_variant=clip`은 frequency bound 적용
- `structure`는 hidden-to-hidden mask 적용
- `structclip`은 둘 다 적용

---

## Phase 4. checkpoint에 topology metadata 저장

`model_training.py`의 checkpoint payload에 다음 추가:

```python
"topology_config": topology_spec.to_dict()
"topology_id": topology_spec.id
```

완료 기준:

- checkpoint restore 시 같은 topology로 model 재구성
- 잘못된 topology config면 load 단계에서 fail
- old checkpoint는 `topology_variant=none`으로 fallback

---

## Phase 5. neuron stats 분석 추가

새 entrypoint:

```text
psd_snn/cli/analyze_neuron_stats.py
```

기능:

```bash
python -m psd_snn.cli.analyze_neuron_stats \
  --checkpoint /path/to/checkpoints \
  --dataset s-mnist \
  --prep_root /path/to/prep \
  --output_root /path/to/out \
  --include_parameter_stats 1 \
  --include_trace_stats 1 \
  --anal_batch 128 \
  --gpu_index 0
```

산출물:

```text
neuron_parameter_vector.csv
neuron_parameter_snapshot.csv
neuron_parameter_trend.csv
membrane_trace_stat.csv
```

완료 기준:

- LIF: `membrane_decay_alpha`, `membrane_time_constant_tau_step` 저장
- RF: `resonant_frequency_cyc_per_step`, `damping_magnitude`, `decay_radius_rho` 저장
- topology model이면 `group_id`, `lower_bound`, `upper_bound` 저장
- epoch별 trend plotting 가능

---

## Phase 6. bash/config 정리

마지막에 bash를 줄입니다.

```text
bash/train.sh
bash/analyze.sh
bash/plot.sh
```

dataset-specific script는 삭제하거나 `configs/examples/`로 이동합니다.

---

# 8. 추천 CSV v2 category 예시

## 8.1 `spectral_curve.csv`

```csv
schema_version,category,artifact_id,source_program,status,message,created_at,dataset,run_id,seed,model_token,model_family,readout_mode,topology_variant,checkpoint_epoch,split,scope,probe_family,label,layer_index,layer_name,signal_kind,series,method,extractor,reducer,variant,scale,frequency_index,frequency,frequency_unit,value_name,value,value_unit
```

## 8.2 `spectral_matrix_2d.csv`

```csv
schema_version,category,artifact_id,source_program,status,message,created_at,dataset,run_id,seed,model_token,model_family,readout_mode,topology_variant,checkpoint_epoch,split,scope,label,layer_index,layer_name,signal_kind,series,method,variant,scale,row_index,row_frequency,row_frequency_unit,column_axis_id,value_unit,time_freq_000000,time_freq_000001,...
```

그리고 sidecar:

```csv
schema_version,category,artifact_id,column_axis_id,column_name,column_index,column_value,column_unit
```

## 8.3 `neuron_parameter_snapshot.csv`

```csv
schema_version,category,artifact_id,source_program,status,message,created_at,dataset,run_id,seed,model_token,model_family,readout_mode,topology_variant,topology_id,checkpoint_epoch,layer_index,layer_name,group_id,parameter_name,parameter_role,statistic,value,value_unit,lower_bound,upper_bound
```

## 8.4 `membrane_trace_stat.csv`

```csv
schema_version,category,artifact_id,source_program,status,message,created_at,dataset,run_id,seed,model_token,model_family,readout_mode,topology_variant,checkpoint_epoch,split,scope,probe_family,label,layer_index,layer_name,signal_kind,series,statistic,value,value_unit
```

---

# 9. 지금 연구 목적 기준으로 가장 중요한 분석 추가

사용자 연구가 "layer output을 PSD로 바꾸고 layer의 주파수 특성을 분석"하는 쪽이면, topology model에서 봐야 할 건 단순 정확도보다 이것입니다.

## 9.1 LIF topology model

반드시 봐야 할 것:

| 분석 | 질문 |
|---|---|
| layer별 $\alpha$ 분포 | 깊은 layer로 갈수록 장기 memory가 커지는가? |
| group별 $\alpha$ 분포 | low/high frequency band와 $\alpha$가 맞게 분리되는가? |
| $\alpha$ trend | 학습 중 특정 band로 collapse되는가? |
| decision trace PSD | $\alpha$가 큰 뉴런이 실제 low frequency PSD를 만드는가? |
| spike rate | 특정 group이 죽는가? |

## 9.2 RF topology model

반드시 봐야 할 것:

| 분석 | 질문 |
|---|---|
| resonant frequency 분포 | clip band 안에 잘 유지되는가? |
| damping 분포 | frequency와 damping이 함께 변하는가? |
| $\rho=\exp(-damping)$ | 실제 memory radius가 어느 정도인가? |
| group별 output PSD | group frequency band와 output PSD peak가 맞는가? |
| spike rate | 특정 frequency band 뉴런이 silent 되는가? |

---

# 10. 최종 추천

현재 프로젝트는 완전히 버릴 구조는 아닙니다. 하지만 대규모 리팩터링을 한다면 다음 원칙으로 가야 합니다.

1. **old project의 `clip`, `structure`, `structclip`은 model token에 억지로 넣지 말고 `TopologySpec`으로 분리** 해야 합니다.
2. **현재 LIF/RF layer의 `alpha_bounds`, `frequency_bounds`, `input_mask`, `recurrent_mask`를 활용** 해야 합니다. old model을 통째로 복사할 필요는 없습니다.
3. **막전위 분석은 `membrane`이라는 애매한 이름을 버리고 `membrane_pre`, `decision`, `membrane_post`, `spike`로 나눠야 합니다.**
4. **`filter_snapshot`/`filter_trend`는 `neuron_parameter_snapshot`/`neuron_parameter_trend`로 이름과 schema를 바꿔야 합니다.**
5. **CSV category는 producer 기준이 아니라 artifact 의미 기준으로 줄여야 합니다.**
6. **bash는 실험 설정 파일 역할을 하면 안 됩니다. 실험 설정은 YAML/JSON, bash는 실행 wrapper만 담당해야 합니다.**
7. **`psd_analysis.py`의 private 함수들을 다른 분석 모듈이 가져다 쓰는 구조는 반드시 없애야 합니다.**

가장 먼저 할 작업은 이것입니다.

```text
1. analysis/common/ 분리
2. models/topology.py, models/constraints.py 추가
3. LIF/RF builder에 constraint 전달
4. neuron_parameter_snapshot/trend 추가
5. CSV schema v2 추가
6. bash를 config 기반 launcher로 축소
```

이 순서로 하면 기존 결과를 최대한 유지하면서, topology model과 PSD 분석을 깨끗하게 붙일 수 있습니다.

## Prompt:
2026. 5. 22. 오후 9:20:40

막전위 기록말인데 지금은 특정시점에서 리셋 전/후 뭘 기록하고있어? 너는 어떻게 고쳐야한다고 생각해?

## Response:
2026. 5. 22. 오후 9:20:45

현재 코드의 LIF/RF forward와 PSD probe 수집 경로를 기준으로, `membrane`이 실제 막전위인지 decision 값인지와 reset 전후 위치를 먼저 구분하겠습니다.

## Response:
2026. 5. 22. 오후 9:20:49

확인 결과, 현재 `membrane`은 hidden layer에서는 대체로 리셋 전 전위에서 threshold를 뺀 spike decision이고, temporal membrane readout 출력층만 raw membrane입니다.

## Response:
2026. 5. 22. 오후 9:24:53

> **검토할 필요가 있는 부분**
>
> 먼저 `snnbench_backbones`를 살펴볼 필요가 있어 보여요. 실제로 VGG/ResNet 경로가 SNNBench를 사용한다고 하네요. 코드에서 `build_snnbench_cnn_classifier`가 반환되는데, 이는 고정된 분류기가 아니라는 점도 언급해야 할 것 같아요. 그리고 MLP 구조는 넓은 수준에서 언급할 때만 다루면 될 것 같고, LIF/RF에 대해 현재 구조를 설명할 수 있습니다.
>
> 7m 29s 동안 생각함

## Response:
2026. 5. 22. 오후 9:28:14

## 결론부터

현재 코드는 대부분의 hidden layer에서 **리셋 전 raw 막전위** 를 기록하지 않습니다.

현재 `record.membrane`에 들어가는 것은 보통 이것입니다.

$$
d_t=v_t^{pre}-v_{th}
$$

즉, **스파이크 결정변수(decision variable)** 입니다.

다만 output layer가 `temporal_membrane` readout을 쓸 때는 예외입니다. 이때는 output layer가 `emit_spike=False`, `reset_enabled=False`로 만들어지므로, output의 `record.membrane`은 threshold를 뺀 값이 아니라 raw membrane에 가깝습니다.

그래서 현재 가장 큰 문제는 이겁니다.

```text
hidden layer의 record.membrane = decision, 즉 v_pre - v_th
output layer의 record.membrane = 경우에 따라 raw membrane
CSV series 이름 = membrane
```

이름은 `membrane`인데 실제 의미가 layer/readout에 따라 달라집니다. 리팩터링에서는 반드시 고쳐야 합니다.

---

# 1. 현재 LIF는 뭘 기록하나?

현재 `src/neurons/LIF_neuron.py` 기준으로 forward 흐름은 이렇습니다.

```python
membrane_pre = alpha.unsqueeze(0) * membrane + current
membrane_signal = membrane_pre - threshold.unsqueeze(0)

spike = surrogate_spike(membrane_signal)

if self.reset_enabled:
    if self.reset_mode == 'soft_reset':
        membrane = membrane_pre - threshold.unsqueeze(0) * spike
    else:
        membrane = membrane_pre * (1.0 - spike)
else:
    membrane = membrane_pre

membrane_steps.append(membrane_pre if record_raw_membrane else membrane_signal)
```

수식으로 쓰면, 시간 $t$에서 다음 순서입니다.

$$
I_t=W x_t+R s_{t-1}
$$

$$
v_t^{pre}=\alpha v_{t-1}^{post}+I_t
$$

$$
d_t=v_t^{pre}-v_{th}
$$

$$
s_t=H(d_t)
$$

soft reset이면,

$$
v_t^{post}=v_t^{pre}-v_{th}s_t
$$

hard reset이면,

$$
v_t^{post}=v_t^{pre}(1-s_t)
$$

그런데 저장되는 값은 보통 `membrane_signal`입니다.

$$
record.membrane_t=d_t=v_t^{pre}-v_{th}
$$

즉, **현재 hidden LIF에서 `record.membrane`은 리셋 전 raw 막전위도 아니고, 리셋 후 막전위도 아닙니다. 리셋 전 spike 판정값입니다.**

중요한 점은 코드상으로 `membrane_steps.append(...)`가 reset 코드 뒤에 있지만, 저장하는 텐서 자체는 reset 전에 계산된 `membrane_signal`입니다. 따라서 저장 시점은 reset 뒤처럼 보이지만, 저장 내용은 reset 전 값입니다.

---

# 2. 현재 RF는 뭘 기록하나?

현재 `src/neurons/RF_neuron.py`도 구조는 거의 같습니다.

RF는 내부 상태가 실수부 $x$, 허수부 $y$로 있습니다.

$$
x_t^{pre}=\rho(\cos(\omega)x_{t-1}^{post}-\sin(\omega)y_{t-1}^{post})+\beta_x I_t
$$

$$
y_t^{pre}=\rho(\sin(\omega)x_{t-1}^{post}+\cos(\omega)y_{t-1}^{post})+\beta_y I_t
$$

스파이크 판정은 실수부만 씁니다.

$$
d_t=x_t^{pre}-v_{th}
$$

현재 기록은 보통 이것입니다.

$$
record.membrane_t=d_t=x_t^{pre}-v_{th}
$$

RF도 마찬가지로 **리셋 후 상태 $x_t^{post}$, $y_t^{post}$는 기록하지 않습니다.** 그리고 더 중요한 문제는 **허수부 $y_t$도 기록하지 않습니다.**

현재 RF에서 저장되는 정보는 사실상 다음뿐입니다.

| 현재 이름 | 실제 의미 |
|---|---|
| `record.membrane` | $x_t^{pre}-v_{th}$ |
| `record.spike` | $H(x_t^{pre}-v_{th})$ |
| `record.layer_input` | $I_t$, 즉 synaptic current |

RF의 공진 특성까지 제대로 분석하려면 $x$만 보면 부족합니다. RF는 2차원 상태계라서 $y_t$, amplitude, phase도 분석 후보가 됩니다.

---

# 3. output layer는 예외가 있음

`temporal_membrane` readout은 output layer를 이렇게 만듭니다.

```python
return {'emit_spike': False, 'reset_enabled': False}
```

이 조건에서는 LIF/RF forward 내부에서 다음 값이 True가 됩니다.

```python
record_raw_membrane = (not self.emit_spike) and (not self.reset_enabled)
```

그래서 output layer에서는 다음이 저장됩니다.

```python
membrane_steps.append(membrane_pre)
```

즉, `temporal_membrane` readout의 output `record.membrane`은 대체로 raw pre-reset membrane입니다.

따라서 현재 구조는 다음처럼 불일치합니다.

| 위치 | `record.membrane` 실제 의미 |
|---|---|
| hidden LIF | $v_t^{pre}-v_{th}$ |
| hidden RF | $x_t^{pre}-v_{th}$ |
| output, temporal membrane readout | $v_t^{pre}$ 또는 $x_t^{pre}$ |
| output, max-rate/first-spike 계열 | 대체로 hidden과 비슷하게 decision |

이건 분석 CSV에서 매우 위험합니다. 같은 `series=membrane`인데 어떤 layer에서는 raw membrane이고, 어떤 layer에서는 threshold-shifted decision입니다.

---

# 4. `layer_input`도 이름이 부정확함

현재 `record.layer_input`도 실제로는 "layer input spike"가 아닙니다.

LIF에서는 이렇게 저장합니다.

```python
current = input_current_sequence[:, time_index, :]
if recurrent_weight is not None:
    current = current + prev_spike @ recurrent_weight.t()

layer_input_steps.append(current)
```

즉,

$$
record.layer_input_t=I_t
$$

입니다.

여기서 $I_t$는 이미 weight를 지난 synaptic current입니다.

$$
I_t=W x_t+R s_{t-1}
$$

따라서 `layer_input`이라는 이름은 헷갈립니다. 더 정확한 이름은 다음 중 하나입니다.

```text
input_current
synaptic_current
pre_activation_current
```

저라면 `input_current`로 바꿉니다.

---

# 5. 현재 코드의 가장 큰 해석 문제

## 5.1 `membrane`이라는 이름이 틀림

현재 hidden layer의 `membrane`은 진짜 막전위가 아니라 spike 판정값입니다.

정확히는:

```text
현재 membrane = decision = membrane_pre - threshold
```

그러므로 논문이나 분석 결과에서 "membrane PSD"라고 쓰면 부정확합니다.

더 정확한 표현은 다음입니다.

```text
decision PSD
threshold-centered membrane PSD
pre-reset spike decision signal PSD
```

## 5.2 리셋 후 상태를 분석할 수 없음

현재는 $v_t^{post}$가 저장되지 않습니다.

하지만 장기의존성(long-term dependency)을 보려면 사실 $v_t^{post}$도 중요합니다. 이유는 다음 time step으로 넘어가는 상태가 $v_t^{post}$이기 때문입니다.

즉, memory를 실제로 carry하는 변수는 다음입니다.

$$
v_t^{post}
$$

그런데 현재 PSD 분석은 이걸 못 봅니다.

## 5.3 raw PSD일 때 threshold shift가 DC 성분을 건드림

현재 decision은 raw membrane에서 threshold를 뺀 값입니다.

$$
d_t=v_t^{pre}-v_{th}
$$

PSD에서 `centering=cen`이면 시간 평균을 빼기 때문에 raw membrane과 decision의 차이가 대부분 사라집니다.

하지만 `centering=raw`이면 threshold shift가 DC 성분에 영향을 줍니다. 따라서 raw PSD에서 low-frequency/DC 해석이 꼬일 수 있습니다.

---

# 6. 내가 고친다면 이렇게 고침

## 6.1 `membrane` 하나로 뭉개지 말고 trace를 분리

리팩터링 후에는 최소한 아래 신호를 명시적으로 분리해야 합니다.

| 새 series 이름 | 의미 | 필요성 |
|---|---|---|
| `input_current` | $I_t$ | layer에 들어온 synaptic current |
| `membrane_pre` | $v_t^{pre}$ 또는 RF $x_t^{pre}$ | 리셋 전 raw 막전위 |
| `decision` | $v_t^{pre}-v_{th}$ 또는 $x_t^{pre}-v_{th}$ | spike 결정변수 |
| `spike` | $s_t$ | 실제 layer output |
| `membrane_post` | $v_t^{post}$ 또는 RF $x_t^{post}$ | 다음 step으로 carry되는 상태 |

RF는 추가로 다음도 넣는 게 좋습니다.

| 새 series 이름 | 의미 |
|---|---|
| `rf_real_pre` | $x_t^{pre}$ |
| `rf_imag_pre` | $y_t^{pre}$ |
| `rf_real_post` | $x_t^{post}$ |
| `rf_imag_post` | $y_t^{post}$ |
| `rf_state_amplitude_pre` | $\sqrt{(x_t^{pre})^2+(y_t^{pre})^2}$ |
| `rf_state_energy_pre` | $(x_t^{pre})^2+(y_t^{pre})^2$ |

RF에서 공진주파수와 감쇠계수까지 분석하려면 `decision`만으로는 부족합니다. spike 생성에는 $x_t$만 쓰지만, RF dynamics 자체는 $(x_t,y_t)$ 상태공간에서 움직입니다.

---

## 6.2 `LayerRecord` 구조를 바꾸는 게 맞음

현재는 이렇습니다.

```python
@dataclass
class LayerRecord:
    layer_name: str
    membrane: torch.Tensor
    spike: torch.Tensor
    layer_input: torch.Tensor | None = None
```

이 구조가 문제입니다. `membrane` 하나로 너무 많은 의미를 담고 있습니다.

추천 구조는 이렇습니다.

```python
@dataclass
class LayerRecord:
    layer_name: str
    traces: dict[str, torch.Tensor]
    layer_index: int | None = None
    signal_kind: str = "hidden"

    @property
    def spike(self) -> torch.Tensor:
        return self.traces["spike"]

    @property
    def membrane(self) -> torch.Tensor:
        # legacy compatibility only
        if "decision" in self.traces:
            return self.traces["decision"]
        if "membrane_pre" in self.traces:
            return self.traces["membrane_pre"]
        raise KeyError("No legacy membrane-compatible trace exists.")
```

즉, 내부적으로는 `traces`를 쓰고, 기존 코드 호환을 위해서만 `record.membrane` property를 남깁니다.

---

## 6.3 LIF forward는 이렇게 바꾸는 게 좋음

현재처럼 `membrane_steps` 하나만 두지 말고, trace 이름별로 저장해야 합니다.

개념 코드는 이렇습니다.

```python
trace_steps = {
    "input_current": [],
    "membrane_pre": [],
    "decision": [],
    "spike": [],
    "membrane_post": [],
}

for time_index in range(time_steps):
    current = input_current_sequence[:, time_index, :]
    if recurrent_weight is not None:
        current = current + prev_spike @ recurrent_weight.t()

    membrane_pre = alpha.unsqueeze(0) * membrane_post_prev + current
    decision = membrane_pre - threshold.unsqueeze(0)

    if self.emit_spike:
        spike = surrogate_spike(decision)
    else:
        spike = torch.zeros_like(decision)

    if self.reset_enabled:
        if self.reset_mode == "soft_reset":
            membrane_post = membrane_pre - threshold.unsqueeze(0) * spike
        else:
            membrane_post = membrane_pre * (1.0 - spike)
    else:
        membrane_post = membrane_pre

    trace_steps["input_current"].append(current)
    trace_steps["membrane_pre"].append(membrane_pre)
    trace_steps["decision"].append(decision)
    trace_steps["spike"].append(spike)
    trace_steps["membrane_post"].append(membrane_post)

    membrane_post_prev = membrane_post
    prev_spike = spike
```

이렇게 하면 reset 전/후가 명확해집니다.

---

## 6.4 RF forward는 이렇게 바꾸는 게 좋음

RF는 $x$, $y$를 모두 남겨야 합니다.

```python
trace_steps = {
    "input_current": [],
    "rf_real_pre": [],
    "rf_imag_pre": [],
    "decision": [],
    "spike": [],
    "rf_real_post": [],
    "rf_imag_post": [],
    "rf_state_energy_pre": [],
}
```

그리고 시간 $t$에서:

```python
x_pre = ...
y_pre = ...

decision = x_pre - threshold.unsqueeze(0)
spike = surrogate_spike(decision)

if self.reset_enabled:
    if self.reset_mode == "soft_reset":
        x_post = x_pre - threshold.unsqueeze(0) * spike
        y_post = y_pre
    else:
        keep = 1.0 - spike
        x_post = x_pre * keep
        y_post = y_pre * keep
else:
    x_post = x_pre
    y_post = y_pre

energy_pre = x_pre.square() + y_pre.square()
```

저장:

```python
trace_steps["input_current"].append(current)
trace_steps["rf_real_pre"].append(x_pre)
trace_steps["rf_imag_pre"].append(y_pre)
trace_steps["decision"].append(decision)
trace_steps["spike"].append(spike)
trace_steps["rf_real_post"].append(x_post)
trace_steps["rf_imag_post"].append(y_post)
trace_steps["rf_state_energy_pre"].append(energy_pre)
```

RF에서 `membrane_pre`라는 이름을 쓸 수도 있지만, 나는 RF에는 `rf_real_pre`를 더 추천합니다. 왜냐하면 RF의 "막전위"가 정확히 $x$인지, 복소 상태 전체인지 해석이 애매해질 수 있기 때문입니다.

---

# 7. PSD 분석에서는 어떤 trace를 기본으로 써야 하나?

연구 목적에 따라 다릅니다.

## 7.1 layer가 실제로 내보낸 출력을 보고 싶다

그러면 기본은 `spike`입니다.

```text
series=spike
```

이건 진짜 layer output입니다. 다음 layer로 넘어가는 값도 hidden layer에서는 spike입니다.

## 7.2 뉴런 내부의 시간 필터링 특성을 보고 싶다

그러면 기본은 `membrane_pre`입니다.

```text
series=membrane_pre
```

이 값은 threshold와 reset을 적용하기 직전의 내부 전위입니다. LIF의 $\alpha$, RF의 공진주파수/감쇠계수가 만든 시간적 필터링이 가장 직접적으로 반영됩니다.

## 7.3 spike가 왜 발생했는지 보고 싶다

그러면 `decision`입니다.

```text
series=decision
```

이 값은 spike function 바로 앞입니다.

$$
decision_t=v_t^{pre}-v_{th}
$$

또는 RF에서는,

$$
decision_t=x_t^{pre}-v_{th}
$$

학습 중 PSD regularization을 `y_mem`에 걸고 있었다면, 현재 실제로는 `decision`에 걸고 있었다고 보는 게 맞습니다.

## 7.4 reset 이후 장기 상태 유지가 궁금하다

그러면 `membrane_post`입니다.

```text
series=membrane_post
```

장기의존성 연구에서는 이 값이 중요합니다. 왜냐하면 다음 step으로 전달되는 내부 상태가 post-reset state이기 때문입니다.

---

# 8. 내 추천 기본 설정

대규모 리팩터링 후 기본 분석 series는 이렇게 가는 게 좋습니다.

```text
input:
  x_probe

hidden:
  input_current
  membrane_pre
  decision
  spike
  membrane_post

output:
  input_current
  readout_membrane 또는 membrane_pre
  decision
  spike
```

다만 모든 trace를 항상 저장하면 메모리가 커집니다. 그래서 CLI에서 선택 가능하게 만들어야 합니다.

예:

```bash
python -m psd_snn.cli.analyze_psd \
  --trace_series input_current,membrane_pre,decision,spike
```

상세 분석할 때만:

```bash
python -m psd_snn.cli.analyze_psd \
  --trace_series input_current,membrane_pre,decision,spike,membrane_post
```

RF 상세 분석:

```bash
python -m psd_snn.cli.analyze_psd \
  --trace_series input_current,rf_real_pre,rf_imag_pre,decision,spike,rf_real_post,rf_imag_post,rf_state_energy_pre
```

---

# 9. CSV 이름도 바꿔야 함

현재는 이런 식입니다.

```text
signal_kind=hidden
series=membrane
```

이건 버리는 게 좋습니다.

새 CSV에서는 이렇게 가야 합니다.

```text
signal_kind=hidden
series=decision
```

또는:

```text
signal_kind=hidden
series=membrane_pre
```

legacy 호환을 위해 당분간은 다음 mapping을 두면 됩니다.

| v1 series | v2 series |
|---|---|
| `membrane` | `decision` for normal hidden spiking layers |
| `membrane` | `membrane_pre` for non-spiking non-reset output |
| `layer_input` | `input_current` |
| `spike` | `spike` |

하지만 v2부터는 `membrane`이라는 애매한 series 이름을 쓰지 않는 게 좋습니다.

---

# 10. 지금 당장 고칠 우선순위

내 기준으로는 다음 순서가 맞습니다.

## 1순위: 이름부터 고치기

현재 hidden layer의 `membrane`은 실제로 `decision`입니다.

그래서 먼저 PSD CSV에서 다음 alias를 추가합니다.

```text
legacy membrane -> decision
legacy layer_input -> input_current
```

기존 결과를 완전히 버리지 않으려면 v1은 유지하고, v2에서 새 이름을 쓰면 됩니다.

## 2순위: `membrane_pre`와 `membrane_post` 추가

LIF/RF forward에서 `return_traces`가 켜졌을 때 다음을 저장하게 합니다.

```text
membrane_pre
decision
membrane_post
spike
input_current
```

RF는 여기에 `rf_imag_pre`, `rf_imag_post`도 추가합니다.

## 3순위: regularization signal 이름 바꾸기

현재 `regularization_signal=y_mem`은 너무 애매합니다.

새 이름은 이렇게 가는 게 맞습니다.

```text
y_decision
y_membrane_pre
y_membrane_post
y_spike
input_current
```

기존 `y_mem`은 당분간 alias로 남기되, 내부적으로는 `y_decision`으로 매핑합니다.

```python
if signal_name == "y_mem":
    signal_name = "decision"
```

## 4순위: output readout trace 분리

output layer는 hidden과 다르게 readout 목적이 있습니다.

따라서 output에서는 `membrane` 대신 다음 이름을 쓰는 게 좋습니다.

```text
readout_membrane
readout_decision
readout_spike
```

특히 `temporal_membrane` readout의 output trace는 classifier score에 직접 쓰이는 값이므로 `readout_membrane`이라는 이름이 더 정확합니다.

---

# 최종 판단

현재 코드는 다음 상태입니다.

```text
hidden LIF/RF의 record.membrane
= reset 전 raw membrane이 아님
= reset 후 membrane도 아님
= spike 판정 직전 decision variable
= v_pre - v_th 또는 x_pre - v_th
```

따라서 내가 고친다면:

```text
1. record.membrane 이름을 분석용으로 더 이상 쓰지 않음
2. hidden series를 decision, membrane_pre, membrane_post, spike, input_current로 분리
3. RF는 rf_real_pre, rf_imag_pre까지 기록
4. CSV v2에서 series=membrane 제거
5. 기존 y_mem은 decision alias로만 유지
```

연구 해석 기준으로는 이렇게 보면 됩니다.

```text
layer output PSD를 보려면 spike
뉴런 내부 시간 필터링을 보려면 membrane_pre
스파이크 결정 과정을 보려면 decision
장기 상태 유지와 reset 효과를 보려면 membrane_post
RF 공진 상태를 보려면 rf_real/rf_imag/state_energy
```

현재의 `membrane`은 논문/분석에서는 "막전위"라고 부르면 안 됩니다. 정확히는 **threshold-centered pre-reset decision signal** 입니다.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
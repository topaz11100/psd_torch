# dense SNN 구현 명세

## 1. 범위

이 문서는 `Spec/theory/models/dense.md` 를 구현 계약으로 연결한다. Dense SNN 은 프로젝트의 주 분석 대상이다. 이 문서는 hidden depth, hidden width, neuron family 를 고정하지 않는다. 구조는 bash scenario slot 과 model slot 에서 해석한다.

## 2. slot contract

Dense SNN builder 는 아래 정보를 받아 model 을 생성해야 한다.

| field | source | 의미 |
| --- | --- | --- |
| `hidden_spec` | scenario slot | comma-separated hidden width list, 예: `64,32,20` |
| `neuron_token` | model slot | `lif_soft_fixed`, `rf_soft_fixed`, `lif_R_soft_fixed`, `rf_R_soft_fixed`, 기타 등록 full neuron token |
| `gpu_index` | model slot | 해당 model process 가 사용할 GPU |
| `readout` | scenario 또는 default | 기본값 `max_rate` |
| `reset_profile` | scenario 또는 neuron config | 같은 comparison row 안에서 고정 |
| `threshold_profile` | scenario 또는 neuron config | 같은 comparison row 안에서 고정 |

`hidden_spec` 이 `64,32,20` 이면 builder 는 hidden layer 3개를 만든다. `neuron_token` 의 `_R_` marker 또는 동등한 recurrent flag 가 있으면 hidden layer 에 same-layer recurrent spike connection 을 추가한다.

## 3. 구현 요구사항

### IMPL-MODEL-DENSE-001

요구사항:
Dense SNN registry 는 fixed token list 대신 `hidden_spec` 과 `neuron_token` 을 해석해 feedforward dense SNN 또는 recurrent dense SNN 을 생성해야 한다.

적용 대상:
- `src/model/model_registry.py`
- `src/model/snn_builder.py`
- `src/model/arch_spec.py`
- `bash/model_training.sh`, `bash/psd_analysis.sh`

수용 기준:
- `hidden_spec = 64,32,20` 이면 hidden widths `[64,32,20]` 로 구성한다.
- `lif_soft_fixed` 는 recurrent connection 없는 dense SNN 으로 해석한다.
- `lif_R_soft_fixed` 또는 recurrent flag 는 같은 hidden layout 에 recurrent current 를 추가한다.
- `rf_soft_fixed`, `rf_R_soft_fixed` 및 다른 등록 full neuron token 도 같은 skeleton 에서 동작할 수 있어야 한다.
- bare family-only shorthand 는 공식 실행 token 으로 받지 않는다.
- `2x256` 은 예시 hidden layout 일 수 있지만 주 분석 구조로 강제하지 않는다.

### IMPL-MODEL-DENSE-002

요구사항:
Dense builder 는 hidden width list 에 따라 adapter, hidden spiking layers, output layer 를 순서대로 생성해야 한다.

적용 대상:
- `src/model/snn_builder.py`
- `src/model/arch_spec.py`

수용 기준:
- 첫 hidden layer 는 `Linear(d_in -> hidden_widths[0]) + neuron` 이다.
- 중간 hidden layer 는 `Linear(hidden_widths[i-1] -> hidden_widths[i]) + neuron` 이다.
- output layer 는 `Linear(hidden_widths[-1] -> num_classes)` 뒤 readout policy 에 맞는 output neuron 또는 membrane path 를 가진다.
- hidden width 가 1개 이상이어야 한다.
- readout 기본값은 `max_rate` 다.

### IMPL-MODEL-DENSE-003

요구사항:
Recurrent dense SNN builder 는 hidden layer 에 same-layer recurrent spike current 를 추가해야 한다.

적용 대상:
- `src/model/snn_builder.py`
- `src/model/recurrent_snn.py` 또는 동등 wrapper

수용 기준:
- recurrent current 는 `R_l s_l[t-1]` 의미론을 가진다.
- feedforward current 와 recurrent current 를 더한 뒤 동일 neuron update 에 넣는다.
- recurrent connection 은 hidden layer 에만 둔다.
- output layer recurrence 는 기본 경로에 포함하지 않는다.
- recurrent state 는 batch 또는 probe sequence 시작 시 명시적으로 reset 된다.

### IMPL-MODEL-DENSE-004

요구사항:
Neuron family 비교 중 dense skeleton 과 non-neuron setting 을 고정해야 한다.

적용 대상:
- `src/model/snn_builder.py`
- `bash/model_training.sh`, `bash/psd_analysis.sh`
- `src/util/metadata.py`

수용 기준:
- 같은 scenario row 에서 비교하는 model 들은 hidden layout 이 같다.
- neuron family 또는 recurrent flag 외의 optimizer, readout, reset, threshold, analysis checkpoint epoch, probe manifest, userbin edges 를 동시에 바꾸지 않는다.
- neuron family 가 바뀌면 metadata 에 `neuron_family` 와 `neuron_spec_doc` 을 기록한다.

### IMPL-MODEL-DENSE-005

요구사항:
Dense SNN observer 는 main PSD family 를 raw archive 없이 accumulator 로 streaming 해야 한다.

적용 대상:
- `src/psd_analysis.py`
- `src/signal/family_spectral_analysis.py`
- `src/psd_analysis.py`

수용 기준:
- `x_probe`, `z_front`, `x_layer`, `i_ff`, `y_mem`, `y_spike` 를 hook 가능한 범위에서 수집한다.
- recurrent dense SNN 에서는 `i_rec` 를 optional observe-only family 로 수집한다.
- branch/dendritic/filter neuron 은 `z_branch` 또는 family-specific name 으로 observe-only accumulator 를 둘 수 있다.
- analysis checkpoint epoch 에서 full raw activation archive 를 저장하지 않는다.
- 각 family 는 `representative(PSD(signal))` 순서로 처리된다.

### IMPL-MODEL-DENSE-006

요구사항:
Run metadata 는 dense SNN 이 main analysis target 임을 명시해야 한다.

적용 대상:
- `src/util/metadata.py`
- `src/psd_analysis.py`

수용 기준:
- main dense run 은 `analysis_role = main` 을 기록한다.
- `main_analysis_target = dense_snn` 을 기록한다.
- `topology`, `neuron_family`, `recurrent_enabled`, `hidden_layout`, `readout`, `reset_profile`, `threshold_profile` 을 기록한다.
- State-space 또는 Spikformer run 은 `analysis_role = auxiliary_diversity` 로 구분한다.

### IMPL-MODEL-DENSE-007

요구사항:
Bash launcher 는 model slot 에 neuron token 과 GPU index 를 함께 묶고, hidden layout 은 scenario slot 에 둔다.

적용 대상:
- `bash/model_training.sh`, `bash/psd_analysis.sh`
- `src/util/cli_common.py`
- `src/psd_analysis.py`

수용 기준:
- GPU index 를 scenario-level BASE_SET 에 두지 않는다.
- 예시 model slot 은 `lif_soft_fixed|0`, `lif_R_soft_fixed|1`, `rf_soft_fixed|0`, `rf_R_soft_fixed|1` 처럼 full token 으로 해석 가능해야 한다.
- 예시 scenario slot 은 `dataset|64,32,20` 처럼 hidden layout 을 포함할 수 있다.
- output metadata 에 final interpreted hidden layout 을 list 로 저장한다.

## 4. 금지 동작

1. fixed depth-width-neuron 조합 token 을 주 분석 구조의 유일한 official path 로 강제하지 않는다.
2. hidden layout 을 code default 로 조용히 덮어쓰지 않는다.
3. recurrent suffix 가 있는 token 을 feedforward dense SNN 으로 저장하지 않는다.
4. neuron family 를 바꾸면서 hidden layout, readout, optimizer 를 동시에 바꾸고 같은 comparison row 로 합치지 않는다.
5. dense 구조 설명을 CNN, Spikformer, state-space 문서 안에 중복 서술하지 않는다. 해당 문서들은 `dense.md` 와 이 구현 명세를 참조한다.

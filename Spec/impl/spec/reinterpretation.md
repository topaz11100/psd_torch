# Reinterpretation 구현 계약

## 1. 범위

이 문서는 `Spec/theory/reinterprete/reinterpretation_experiments.md` 를 구현과 연결한다. Independent reinterpretation experiment 는 observation hook 과 CSV-compatible spectral output 을 추가하되, author-code raw data reading, preprocessing, training, evaluation setting 을 보존한다.

Reinterpretation 은 main `model_training.py -> psd_analysis.py -> plotting.py` pipeline 과 독립적이지만, numeric output 은 가능한 경우 `Spec/impl/spec/csv_schema.md` 의 category-based CSV 규칙을 따른다.

## 2. 허용 CLI 범위

| argument | 허용 역할 |
| --- | --- |
| `--run_need_high` | Need High paper experiment 활성화 |
| `--run_drf` | D-RF paper experiment 활성화 |
| `--run_dh_snn` | DH-SNN paper experiment 활성화 |
| `--gpu_map` | `experiment_id:gpu_index` comma list |
| `--output_root` | reinterpretation result root |
| `--log_root` | stdout/stderr log root |
| `--userbin_edges` | 사용자 지정 주파수 구간 경계 |
| `--seed_bundle` | probe/sample/reproducibility seed bundle |

Driver 는 dataset, model depth, timestep, optimizer, parameter-decay change, output value-scale selection 같은 paper-structure 또는 artifact-filter override 를 노출하지 않는다. Numeric curve hook output 이 채워지는 경우 exact curve, userbin curve, raw scale, db scale 을 모두 보존해야 한다.

## 3. experiment-gpu mapping

### IMPL-REINT-001

요구사항:
활성화된 reinterpretation experiment 는 `--gpu_map` 을 통해 GPU 에 mapping 된다.

수용 기준:
- `need_high:<gpu>` 는 Need High 를 선택된 GPU 에 mapping 한다.
- `drf:<gpu>` 는 D-RF 를 선택된 GPU 에 mapping 한다.
- `dh_snn:<gpu>` 는 DH-SNN 을 선택된 GPU 에 mapping 한다.
- GPU mapping 이 없는 enabled experiment 는 launch 전에 fail 한다.
- Disabled experiment 는 launch 하지 않는다.

## 4. paper setting preservation

### IMPL-REINT-002

요구사항:
구현은 각 source paper experiment setting 을 보존한다.

수용 기준:
- Need High 는 지정된 CIFAR10-DVS Max-Former vs MS-QKFormer comparison 을 사용한다.
- D-RF 는 지정된 SHD D-RF vs BRF comparison 을 사용한다.
- DH-SNN 은 지정된 SHD vanilla SFNN vs DH-SFNN comparison 을 사용한다.
- driver 는 model, dataset, timestep, epoch, optimizer, decay override 를 노출하지 않는다.

## 5. observation hook

### IMPL-REINT-003

요구사항:
Observation hook 은 author-code learning semantics 를 변경하지 않고 signal 을 수집한다.

수용 기준:
- hook 은 loss 에 사용되는 forward output 을 수정하지 않는다.
- hook 은 gradient 를 수정하지 않는다.
- hook 은 author-code preprocessing 을 대체하지 않는다.
- hook 은 category-based spectral CSV 와 case metric 을 생성한다.

## 6. output layout

### IMPL-REINT-004

요구사항:
output 은 reinterpretation root 아래에 격리한다.

권장 layout:

```text
<output_root>/
  reinterpretation/
    need_high/
      dataset_psd/
      psd_analysis/
      metrics/
    drf/
      dataset_psd/
      psd_analysis/
      metrics/
    dh_snn/
      dataset_psd/
      psd_analysis/
      metrics/
```

수용 기준:
- reinterpretation output 은 main pipeline checkpoint-analysis output 과 섞이지 않는다.
- numeric output 은 가능한 경우 category-based CSV schema 를 사용한다.
- figure 가 필요하면 생성된 CSV tree 에 `plotting.py` 를 실행해 만든다.

## 7. data preparation profile

### IMPL-REINT-005

요구사항:
Author-code preparation profile 은 project-standard prepared data profile 과 분리한다.

수용 기준:
- 논문 저자 코드가 요구하는 raw data reader 와 preprocessing 을 유지한다.
- main project 의 `data_prep.py` 결과를 강제로 주입하지 않는다.
- output metadata 에 author-code profile 을 기록한다.

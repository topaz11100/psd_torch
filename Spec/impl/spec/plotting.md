# plotting 구현 계약

## 1. 문서 역할

이 문서는 `src/plotting.py` 의 implementation contract 를 정의한다. 이 program 은 CSV artifact 에서 figure 를 render 한다. train 하지 않고, `.pt` checkpoint 를 load 하지 않고, PSD value 를 recompute 하지 않는다.

## 2. official CLI

| argument | required | 의미 |
| --- | --- | --- |
| `--input` | yes | 단일 `.csv` file 또는 directory 하나 |
| `--output_root` | optional | figure 용 root, 기본값은 각 input CSV 옆 |
| `--format` | optional | figure format. official 값은 `png` |
| `--overwrite` | optional | existing figure file overwrite 여부 |
| `--manifest_name` | optional | rendering manifest file name. 기본값 `plotting_manifest.csv` |

checkpoint, GPU, training, model-construction argument 는 허용하지 않는다.

## 3. input mode

### 3.1 file mode

`--input` 이 file 을 가리키는 경우:

1. file suffix 는 `.csv` 여야 한다.
2. category-based CSV schema 로 file 을 parse 한다.
3. `category` 가 지원되면 file 을 render 한다.
4. unsupported category 는 manifest 에 skipped 로 기록한다.

### 3.2 recursive directory mode

`--input` 이 directory 를 가리키는 경우:

1. program 은 모든 child directory 를 재귀적으로 순회한다.
2. suffix 가 `.csv` 인 모든 file 을 고려한다.
3. traversal order 는 worker dispatch 전에 lexical path order 로 둔다.
4. 동일 plotting invocation 이 생성한 file 은 manifest-name matching 으로 rendering input 에서 제외한다.
5. non-CSV file 은 ignore 한다.

## 4. CPU worker 의미론

`--cpu_cores` 는 제거한다. rendering 은 single-process 로 수행한다.

규칙:

1. `cpu_cores >= 1` 이 필요하다.
2. `cpu_cores = 1` 은 single worker 로 실행한다.
3. `cpu_cores > 1` 은 process 또는 thread worker 를 사용할 수 있다.
4. worker failure 는 manifest 에 기록해야 한다.
5. program 은 mandatory rendering 에 GPU-only dependency 를 사용하면 안 된다.

## 5. 지원 category

필요 column 이 충분히 있으면 renderer 는 최소한 아래 category 를 지원해야 한다.

| category | expected visualization |
| --- | --- |
| `dataset_curve` | input PSD curve |
| `dataset_dispersion` | input PSD dispersion curve |
| `analysis_curve` | layer/family PSD curve |
| `analysis_dispersion` | layer/family dispersion curve |
| `pair_distance` | 기본 plotting 대상에서 제외한다. 수치 table 로만 유지한다. |
| `layer_distance_profile` | 특정 checkpoint 의 input-reference 및 adjacent layer curve distance |
| `layer_distance_trend` | epoch 축 input-reference 및 adjacent layer curve distance trend |
| `layer_dispersion_profile` | 특정 checkpoint 의 layer-wise variance/MAD scalar profile |
| `layer_dispersion_trend` | epoch 축 layer-wise variance/MAD scalar trend |
| `filter_snapshot` | histogram 또는 summary plot |
| `filter_trend` | checkpoint-axis filter trend |
| `training_metric` | epoch 와 scope 별 metric curve |
| `accuracy_loss_join` | 새 산출물에서는 render 하지 않음 |
| `pairwise_dependency_appendix` | appendix-specific dependency plot |

Unsupported `category` value 는 error 가 아니다. manifest status `skipped_unsupported_category` 로 기록한다.

## 6. PSD curve 시각화 양식

PSD curve 계열 category 는 아래 양식을 반드시 따른다.

대상 category:

1. `dataset_curve`
2. `analysis_curve`
3. PSD curve 로 해석되는 `dataset_dispersion`
4. PSD curve 로 해석되는 `analysis_dispersion`

### 6.1 x축

1. x축은 정규화 주파수다.
2. 축 라벨은 `Normalized Frequency` 로 표기한다.
3. x축 범위는 `0.0` 부터 `0.5` 까지로 고정한다.
4. x축 tick 은 `0.0`, `0.1`, `0.2`, `0.3`, `0.4`, `0.5` 를 표시한다.
5. x축 tick 값은 반드시 그림에 보이게 한다.

### 6.2 y축

1. y축 라벨은 curve category 에 맞게 설정한다.
2. PSD raw scale 의 y축 라벨은 `Power` 로 표기한다.
3. PSD dB scale 의 y축 라벨은 `Power (dB)` 로 표기한다.
4. distance category 의 y축 라벨은 `Distance` 로 표기한다.
5. dispersion category 의 y축 라벨은 statistic 이름을 사람이 읽기 쉬운 형태로 표기한다.

### 6.3 제목

1. 파일명이나 underscore 가 포함된 이름을 제목으로 쓰지 않는다.
2. 사람이 읽기 쉬운 제목을 사용한다.
3. 입력 데이터는 `PSD of Input` 으로 표기한다.
4. n번 레이어는 `PSD of Layer n` 로 표기한다.
5. 2번 레이어는 `PSD of Layer 2` 로 표기한다.
6. 3번 레이어는 `PSD of Layer 3` 로 표기한다.

### 6.4 figure size 와 선 스타일

1. 가로:세로 비율은 `3.5:1` 로 설정한다.
2. matplotlib `figsize` 는 `(14, 4)` 를 사용한다.
3. 꺾은선 그래프로 시각화한다.
4. 선 굵기는 `3.5` 로 설정한다.

### 6.5 글자 크기

1. 제목 글자 크기는 `25` 로 설정한다.
2. x축 및 y축 라벨 글자 크기는 `20` 으로 설정한다.
3. x축 및 y축 tick 글자 크기는 `18` 로 설정한다.

### 6.6 배경, 격자, 테두리

1. 그림 배경은 흰색으로 설정한다.
2. 축 내부 배경도 흰색으로 설정한다.
3. 내부 격자선은 표시하지 않는다.
4. `grid` 는 사용하지 않는다.
5. 축 테두리선 두께는 `1.2` 로 설정한다.
6. tick 선 두께는 `1.2` 로 설정한다.
7. tick 길이는 `6` 으로 설정한다.

### 6.7 여백과 저장

1. 제목과 그래프 사이 간격은 `pad=10` 으로 설정한다.
2. x축 및 y축 라벨과 축 사이 간격은 `labelpad=8` 로 설정한다.
3. 저장 전 `tight_layout` 을 적용한다.
4. PNG 형식으로 저장한다.
5. 저장 해상도는 `300 dpi` 로 설정한다.
6. 저장 시 `bbox_inches="tight"` 를 사용한다.
7. 저장 시 `facecolor="white"` 를 사용한다.

## 7. output contract

기본 file-mode output:

```text
<input_csv_parent>/<input_csv_stem>.png
```

`--output_root` 없는 기본 directory-mode output:

```text
same directory as each input CSV
```

`--output_root` 가 있는 directory-mode output:

```text
<output_root>/<relative_path_from_input_root>/<input_csv_stem>.png
```

program 은 category-based schema 를 사용해 rendering manifest CSV 를 쓴다.

Manifest semantic field:

| schema field | value |
| --- | --- |
| `category` | `plotting_manifest` |
| `source_program` | `plotting` |
| `input_csv_path` | rendering 대상으로 고려한 CSV |
| `output_figure_path` | generated 된 경우 figure path |
| `render_seconds` | timing row 의 numeric value |
| `status` | `rendered`, `skipped_unsupported_category`, `failed`, or `skipped_existing` |
| `message` | error 또는 skip reason |

## 8. renderer input validation

rendering 전에 program 은 다음을 validate 한다.

1. required common prefix column 이 존재한다.
2. `schema_version` 이 지원된다.
3. `category` 가 known 이거나 safely skippable 이다.
4. renderer 에 필요한 category-specific numeric column 을 parse 할 수 있다.
5. `--overwrite` 가 enabled 가 아니면 output path 가 existing file 을 overwrite 하지 않는다.

한 CSV 의 validation failure 는 directory-mode rendering 에서 unrelated CSV file 처리를 중단하면 안 된다. 반드시 manifest failure row 를 생성해야 한다.

## 9. 금지 operation

`src/plotting.py` 는 다음을 수행하면 안 된다.

1. `.pt` file load.
2. project model build.
3. model forward pass 실행.
4. raw signal 에서 PSD curve recompute.
5. analysis CSV value 수정.
6. input CSV file 삭제.
7. training 또는 analysis entrypoint 호출.


## Patched drift plot split

새 layer distance 및 layer dispersion rendering 은 category folder 구조를 유지한다.

```text
traces/layer_distance_trend/input_reference/
traces/layer_distance_trend/adjacent/
traces/layer_dispersion_trend/variance/
traces/layer_dispersion_trend/mad/
```

`curve_distance` 는 input PSD 대비 대상 layer/series representative PSD curve 의 centered shape distance 를 그린다. `dispersion_variance` 와 `dispersion_mad` 는 representative reducer 와 독립이므로 파일명과 grouping 에서 reducer 를 사용하지 않는다. x축은 checkpoint epoch 를 사용한다. `checkpoint_epoch_a == checkpoint_epoch_b` 이면 tick label 은 단일 epoch 로 표시한다.

## 2026-05-02 수정 고정

- `--cpu_cores` 인수는 제거한다. 병렬 rendering 은 구현하지 않는다.
- `bash/plotting.sh` 는 각 input 마다 `OUTPUT_ROOT/<case_id>/` 아래로 산출물을 분리한다.
- plotter 는 `layer_distance_profile`, `layer_distance_trend`, `layer_dispersion_profile`, `layer_dispersion_trend`, `filter_trend` category 를 처리한다.
- 새 산출물 기준으로 `drift_distance` 와 `accuracy_loss_join` plot 은 생성하지 않는다.

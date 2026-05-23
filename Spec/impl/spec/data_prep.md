# data_prep 구현 명세서

이 문서는 `Spec/theory/data_prep/data_prep.md` 의 dataset 별 전처리 기준선을 실제 프로그램 입출력 계약으로 연결하는 공식 구현 명세다. 단일파일 mmap 저장 계약은 이 문서로 통합한다.

## 1. 핵심 원칙

1. `src/data_prep.py` 만 raw 데이터를 직접 읽는다.
2. dataset 별 전처리 규칙은 `Spec/theory/data_prep/data_prep.md` 의 공식 기준선을 그대로 따른다.
3. split shard size-limit parameter 는 공식 인터페이스에 포함하지 않는다.
4. split shard 는 사용하지 않는다.
5. 각 dataset 은 train split 과 test split 을 각각 단일 physical payload 파일 하나로 저장한다. 단, 정적 이미지 dataset 은 CNN용 view 와 flatten용 view 를 분리 저장할 수 있다.
6. 각 split payload 는 numpy mmap 이 가능한 `.npy` 파일이어야 한다.
7. downstream 실험은 raw 데이터를 직접 읽지 않고 prepared split `.npy` 만 읽는다.
8. downstream 단계에서 dataset 별 전처리 규칙을 다시 적용하면 안 된다.
9. 같은 prepared split 파일은 여러 모델, 여러 seed, 여러 실험이 공통 참조해야 하며 per-run 복사본을 만들면 안 된다.
10. `data_prep` 가 새로 생성하는 공식 산출물은 기본적으로 dataset 당 `train.npy`, `test.npy`, `manifest.json` 이다. 단, 정적 이미지 dataset 은 `train_cnn.npy`, `test_cnn.npy`, `train_flatten.npy`, `test_flatten.npy`, `manifest.json` 을 공식 산출물로 둔다.

## 2. 공식 엔트리와 CLI

공식 전처리 프로그램 엔트리는 `src/data_prep.py` 다. 호출 형식은 `python -m src.data_prep` 를 기준으로 한다.

공식 CLI 인수는 아래만 둔다.

1. `--dataset`
2. `--raw_data_root`
3. `--prep_root`
4. `--seed`
5. `--force_overwrite`


## 3. 출력 디렉터리 구조

전처리 결과 루트의 최소 구조는 아래 하나뿐이다.

```text
<prep_root>/
  <dataset>/
    train.npy
    test.npy
    manifest.json
```

정적 이미지 dataset 의 구조는 아래를 허용한다.

```text
<prep_root>/
  <dataset>/
    train_cnn.npy
    test_cnn.npy
    train_flatten.npy
    test_flatten.npy
    manifest.json
```

설명은 아래와 같다.

1. `train.npy` 는 train split 전체 sample 을 담는 단일 mmap 가능 payload 파일이다.
2. `test.npy` 는 test split 전체 sample 을 담는 단일 mmap 가능 payload 파일이다.
3. 정적 이미지 dataset 에서는 `train_cnn.npy`, `test_cnn.npy` 가 `(T,C,H,W)` CNN view 를 담고, `train_flatten.npy`, `test_flatten.npy` 가 `(T,C*H*W)` flatten view 를 담는다.
4. `manifest.json` 은 dataset 단위의 생성 metadata 와 runtime view 해석 규칙을 담는 소형 요약 파일이다.
4. split 별 `labels.npy`, `sample_indices.npy`, shard json, sidecar base npy 파일은 더 이상 두지 않는다.
5. `train/`, `test/` 하위 shard 디렉터리는 사용하지 않는다.

## 4. split payload 형식

각 split payload 는 shape 가 `(N,)` 인 structured numpy array 를 담는 단일 `.npy` 파일이어야 한다. 여기서 $N$ 은 split sample 수다.

권장 structured dtype 개념은 아래와 같다.

```text
record_dtype = [
  ("sample_index", int64),
  ("label", int64),
  ("input", stored_dtype, stored_shape)
]
```

의미는 아래와 같다.

1. `sample_index` 는 raw split 내부의 고정 sample 순서를 보존하는 `int64` 값이다.
2. `label` 은 class label 또는 동등한 supervised target 을 담는 `int64` 값이다.
3. `input` 은 해당 dataset 에 대해 정한 단일 stored input view 를 담는 고정 shape 수치 배열이다.
4. 모든 sample 은 같은 `stored_shape` 와 `stored_dtype` 를 가져야 한다.
5. object dtype, pickle 기반 payload, 가변 길이 payload 는 허용하지 않는다.
6. 모든 값은 numpy mmap 으로 직접 접근 가능해야 한다.

## 5. stored input view 와 PSD logical view 규칙

단일 physical file 저장을 위해 split 당 하나의 stored input view 만 실제로 저장한다. stored input view 는 원칙적으로 supervised model 이 직접 소비하는 입력 기대 차원과 같아야 한다. 가능한 경우에는 이 model input layout 과 PSD 분석의 time/row 축 해석을 통일한다.

이 문서에서 `training_view_name` 은 supervised training 과 평가가 실제로 소비하는 입력 view 를 뜻한다. `psd_view_name` 은 `dataset_psd` 와 `psd_analysis` 의 probe reference 또는 input family 가 소비하는 logical PSD view 를 뜻한다. 두 view 는 수치적으로 같은 tensor 를 가리킬 수 있지만, 역할은 항상 metadata 로 구분한다.

PSD 분석 모듈은 `(R, T)` 같은 고정 physical shape 를 prepared input 에 요구하지 않는다. 대신 `manifest.json` 과 model profile 은 아래 canonical PSD axis metadata 를 제공해야 한다.

1. `psd_sample_axis`: split payload 또는 per-sample stored tensor 에서 sample 축을 가리킨다. structured array record 내부 input 에 sample 축이 없으면 `null` 로 둔다.
2. `psd_batch_axis`: runtime batch tensor 에서 batch 축을 가리킨다. manifest 에서는 loader 가 만드는 batch layout 기준으로 기록한다.
3. `psd_time_axis`: 막전위 또는 neuron state update 가 진행되는 timestep 축을 가리킨다.
4. `psd_row_axes`: PSD row 로 flatten 되는 모든 non-time signal axes 를 가리킨다.
5. `psd_feature_axes`, 필요한 경우
6. `psd_token_axes`, transformer 계열인 경우
7. `psd_flatten_rule`
8. `psd_logical_shape`

PSD logical view 는 아래 계약을 따른다.

$$
X_{\mathrm{model}} \rightarrow X_{\mathrm{psd-view}} \in \mathbb{R}^{N \times R \times T}
$$

여기서 $X_{\mathrm{psd-view}}$ 는 PSD accumulator 에 전달되는 logical view 이며, 저장 tensor 의 physical shape 를 뜻하지 않는다. runtime transpose, permute, flatten 은 필요한 경우 view 해석 또는 accumulator 전달을 위한 최소 변환으로만 허용한다. 특정 dataset 에 대해 별도 PSD용 physical layout 으로 되돌리는 필수 변환을 명세에 두지 않는다.

### 5.1 시퀀셜 데이터

시퀀셜 데이터의 공통 규칙은 `training_view_name = model_input`, `psd_view_name = model_input_psd_view`, `psd_axis_kind = temporal` 이다.

| dataset | stored_view_name | stored_shape | logical PSD shape | axis metadata 요약 |
| --- | --- | --- | --- | --- |
| `s-mnist` | `model_input` | `(784, 1)` | `(1, 784)` | time axis는 pixel sequence, row axis는 scalar channel |
| `ps-mnist` | `model_input` | `(784, 1)` | `(1, 784)` | time axis는 permuted pixel sequence, row axis는 scalar channel |
| `s-cifar10` | `model_input` | `(1024, 3)` | `(3, 1024)` | time axis는 raster index, row axis는 RGB channel |
| `shd` | `model_input` | `(1200, 700)` | `(700, 1200)` | time axis는 bin index, row axis는 input neuron |
| `ssc` | `model_input` | `(1000, 700)` | `(700, 1000)` | time axis는 bin index, row axis는 input neuron |
| `deap` | `model_input` | `(384, 32)` | `(32, 384)` | time axis는 EEG sample index, row axis는 EEG channel |
| `uci-har` | `model_input` | `(128, 6)` | `(6, 128)` | time axis는 sensor sample index, row axis는 selected sensor channel |

위 표의 `logical PSD shape` 는 분석 의미론이다. 구현은 stored tensor 를 반드시 해당 shape 로 materialize 하거나 transpose 기반으로 materialize 해야 한다고 가정하지 않는다.

### 5.2 정적 이미지 데이터

정적 이미지 데이터 `mnist`, `cifar-10`, `cifar-100` 은 PLIF식 direct input으로 처리한다. 같은 image frame을 `data_prep` 저장 단계에서 4 timestep으로 반복한다. CNN용 physical storage 는 `(T,C,H,W)` 로 저장하고, Dense/MLP용 physical storage 는 `(T,C*H*W)` 로 따로 저장한다. 모델 코드와 runtime view는 반복 frame을 새로 만들지 않는다.

| dataset | CNN stored view | flatten stored view | logical PSD shape | axis metadata 요약 |
| --- | --- | --- | --- | --- |
| `mnist` | `(4, 1, 28, 28)` | `(4, 784)` | `(784, 4)` | `T=4`, row axis는 channel-height-width flatten, time axis는 static repeat timestep |
| `cifar-10` | `(4, 3, 32, 32)` | `(4, 3072)` | `(3072, 4)` | `T=4`, row axis는 channel-height-width flatten, time axis는 static repeat timestep |
| `cifar-100` | `(4, 3, 32, 32)` | `(4, 3072)` | `(3072, 4)` | `T=4`, row axis는 channel-height-width flatten, time axis는 static repeat timestep |

Manifest 기준값은 다음과 같다.

```text
static_repeat_T = 4
sequence_length = 4
input_dim = C*H*W
cnn_input_shape = [4, C, H, W]
flatten_input_shape = [4, C*H*W]
model_input_axis_order = [time, channel, height, width]
```

`model_input_cnn`, `cnn_input`, `psd_input`, `image_psd_view` 는 `(T,C,H,W)` 를 반환한다. `model_input_flatten`, `flatten_input`, `sequence_input` 은 `(T,C*H*W)` 를 반환한다. CNN 계열 모델은 dataloader batch인 `(B,T,C,H,W)` 를 그대로 받고, Dense 계열 모델은 dataloader batch인 `(B,T,C*H*W)` 를 받는다. PSD 분석은 `tensor_to_channel_major_maps` 단계에서 CNN view 를 `(B,T,C,H,W) -> (B,C*H*W,T)` 로 해석한다.

### 5.3 DVS 데이터셋

DVS 데이터셋은 physical event frame time 을 가진 image-origin 입력이다. 공통 규칙은 `training_view_name = model_input`, `psd_view_name = event_frame_psd_view`, `psd_axis_kind = image_temporal` 이다.

| dataset | stored_view_name | stored_shape | logical PSD shape | axis metadata 요약 |
| --- | --- | --- | --- | --- |
| `n-mnist` | `model_input` | `(10, 2, 34, 34)` | `(2312, 10)` | time axis는 frame index, row axes는 polarity와 spatial axes |
| `cifar10-dvs` | `model_input` | `(20, 2, 128, 128)` | `(32768, 20)` | time axis는 frame index, row axes는 polarity와 spatial axes |
| `dvs128-gesture` | `model_input` | `(20, 2, 128, 128)` | `(32768, 20)` | time axis는 frame index, row axes는 polarity와 spatial axes |

원 논문 또는 author code 기반 profile 이 위 stored shape 와 다른 input order 를 요구하면 stored view 또는 wrapper 는 author-code input order 를 유지한다. PSD analysis 는 `psd_time_axis`, `psd_row_axes`, `psd_flatten_rule`, `psd_logical_shape` metadata 를 사용해 logical PSD view 를 정의한다.

### 5.4 DVS128 Gesture loader 계약

SpikingJelly 기반 DVS128 Gesture 준비 경로는 frame dataset API 를 사용한다. 구현은 아래 의미를 만족해야 한다.

```text
DVS128Gesture(
  root=<raw_or_cache_root>,
  train=<bool>,
  data_type="frame",
  frames_number=<num_frames>,
  split_by="number"
)
```

구버전 인수명이나 임의 normalize 인수를 공식 계약으로 두지 않는다. raw root 에 단순히 `DVS Gesture dataset.zip` 파일만 놓는 구조는 충분하지 않다. loader 가 기대하는 다운로드 파일 구조는 아래를 만족해야 한다.

```text
<raw_root>/download/DvsGesture.tar.gz
<raw_root>/download/gesture_mapping.csv
<raw_root>/download/LICENSE.txt
```

frame count 는 manifest 의 `num_frames` 와 stored shape 의 time axis 길이에 일치해야 한다.

단일 stored input view 로부터 공식 training view 와 PSD logical view 를 deterministic 하게 해석할 수 없는 dataset 이 추가되면 이 문서를 먼저 개정해야 한다.

## 6. dtype 규칙

1. 실수 입력은 `float32` 저장을 기본으로 한다.
2. binary occupancy 입력은 `uint8` 저장을 기본으로 한다. 값은 0 또는 1 이어야 한다.
3. event count frame 입력은 overflow 가 없도록 unsigned integer 계열을 사용한다. 구체 dtype 은 `manifest.json` 에 기록한다.
4. `label` 과 `sample_index` 는 `int64` 로 고정한다.
5. 저장 dtype 선택은 runtime 에 추가 normalize 나 값의 의미 변경을 유발하면 안 된다.

## 7. `manifest.json` 규칙

`manifest.json` 은 dataset 당 하나만 둔다. 최소 아래 필드를 기록해야 한다.

1. `dataset_name`
2. `files.train = "train.npy"` 또는 정적 이미지 dataset 의 경우 `"train_cnn.npy"`
3. `files.test = "test.npy"` 또는 정적 이미지 dataset 의 경우 `"test_cnn.npy"`
4. `storage_format = "single_structured_npy_v1"`
5. `preprocessing_spec_doc = "Spec/theory/data_prep/data_prep.md"`
6. `preprocessing_impl_spec_doc = "Spec/impl/spec/data_prep.md"`
7. `raw_data_root`
8. `seed`
9. `split_internal_order_preserved = true`
10. `stored_view_name_by_split` 또는 dataset category 별 `stored_view_name`
11. `training_view_name`
12. `psd_view_name`
13. `psd_axis_kind` (`"temporal"`, `"image_temporal"`, `"repeated_static"`, `"raster_spatial"` 중 하나)
14. `stored_shape`
15. `stored_dtype`
16. `model_input_axis_order`
17. `psd_sample_axis`
18. `psd_batch_axis`
19. `psd_time_axis`
20. `psd_row_axes`
21. `psd_feature_axes`, 필요한 경우
22. `psd_token_axes`, 필요한 경우
23. `psd_flatten_rule`
24. `psd_logical_shape`
25. `stored_order_is_model_input_order = true`
26. `label_dtype = "int64"`
27. `sample_index_dtype = "int64"`
28. dataset 별 전처리 고정값 요약 예시 `dt`, `max_time`, `segment_length`, `num_frames`, `normalization_rule`, `permutation_seed`, `flatten_order`
29. `progress_logger = "tqdm"`
30. 정적 이미지 dataset 은 `files_by_view`, `cnn_training_view_name`, `flatten_training_view_name`, `stored_shape_by_view` 를 기록한다.

`manifest.json` 은 사람이 읽는 요약이자 runtime 해석 규칙의 단일 출처다. downstream 실험은 split `.npy` 만 직접 열지 말고 먼저 `manifest.json` 을 읽어 저장 형식과 view 해석 규칙을 확인해야 한다.

## 8. writer 규칙

writer 는 `numpy.lib.format.open_memmap` 또는 동등한 `.npy` mmap writer 를 사용해 split payload 를 생성한다. 권장 write 순서는 아래와 같다.

1. raw split 의 sample 수 $N$ 확정
2. record dtype 확정
3. 임시 경로 `train.tmp.npy` 또는 `test.tmp.npy` 생성
4. `open_memmap` 으로 shape `(N,)` structured array 생성
5. sample 순회하며 `record["sample_index"]`, `record["label"]`, `record["input"]` 채움
6. flush 수행
7. file descriptor `fsync` 수행
8. 동일 경로를 다시 mmap read 모드로 열어 shape, dtype, 첫 record, 마지막 record, sample index 연속성 검증
9. 검증 성공 시 atomic rename 으로 `train.npy` 또는 `test.npy` 로 교체
10. 마지막에 `manifest.json` 기록

원칙은 아래와 같다.

1. split 쓰기 중간 실패가 최종 파일명을 오염시키면 안 된다.
2. `manifest.json` 은 필요한 모든 split payload 검증이 끝난 뒤에만 기록한다.
3. `force_overwrite` 가 false 인 상태에서 기존 payload 가 있으면 덮어쓰지 않는다.

## 9. loader 규칙

runtime loader 는 manifest 의 `files` 또는 `files_by_view` 가 가리키는 split payload 를 `np.load(path, mmap_mode="r")` 로 연다. loader 는 split 전체를 RAM 으로 materialize 하면 안 된다.

기본 흐름은 아래와 같다.

1. `manifest.json` 으로 dataset 저장 형식과 stored input view 규칙을 읽는다.
2. split `.npy` 를 mmap 으로 등록한다.
3. batch sampler 가 batch 인덱스를 선택한다.
4. 해당 인덱스의 record 만 실제 접근한다.
5. `record["input"]` 으로부터 training view 또는 PSD logical view 를 runtime 에서 해석한다.
6. 필요 시 numpy 를 torch tensor 로 변환한 뒤 GPU 로 전송한다.
7. batch 객체 참조가 사라지면 파이썬 객체는 정리되며, mmap handle 과 OS page cache 는 재사용될 수 있다.

loader 는 매 sample 마다 파일을 다시 열고 닫는 구조를 사용하면 안 된다. 권장 구조는 worker 당 split mmap handle 을 한 번 열고 반복 재사용하는 방식이다.

## 10. downstream 입력 계약

`dataset_psd`, `psd_analysis`, supervised training, 기타 후속 실험은 공식적으로 `--prep_root` 만 받는다.

아래 규칙을 따라야 한다.

1. raw 데이터 경로를 직접 입력으로 받지 않는다.
2. `manifest.json` 을 먼저 읽는다.
3. `manifest.json` 이 가리키는 공식 split payload 만 사용한다.
4. split 전체를 `torch.cat` 해 메모리에 쌓으면 안 된다.
5. batch streaming 으로 집계해야 한다.
6. stored input 또는 model input 에서 manifest 의 official logical view 를 해석하는 것 외에, `data_prep.md` 에 없는 추가 normalize, permutation 재생성, event re-binning, 의미 변경을 수행하면 안 된다.
7. deterministic probe selection 은 `sample_index` 가 보존한 split 내부 고정 순서를 기준으로만 수행한다.
8. 여러 모델과 여러 seed 는 같은 prepared payload 를 공통 참조해야 하며 per-run 복사본을 만들면 안 된다.
9. supervised training 과 inference 는 항상 `manifest.json` 의 view metadata 를 사용해야 한다. 정적 이미지 dataset 에서 CNN 계열은 `cnn_training_view_name`, Dense 계열은 `flatten_training_view_name` 을 우선 사용한다.
10. `dataset_psd` 와 probe reference 재생성 경로는 항상 `manifest.json` 의 `psd_view_name`, `psd_sample_axis`, `psd_batch_axis`, `psd_time_axis`, `psd_row_axes`, `psd_flatten_rule`, `psd_logical_shape` 을 사용해야 한다.
11. `training_view_name` 과 `psd_view_name` 이 같은 physical tensor 를 가리키더라도, 후속 문서에서는 실제 학습 입력과 probe 기준 입력을 개념적으로 분리해 기록해야 한다.

## 11. 전체 재생성이 필요한 경우

다음 경우에는 split 전체를 다시 생성해야 한다.

1. sample 수 변경
2. `stored_shape` 변경
3. `stored_dtype` 변경
4. stored input view 변경
5. label 정의 변경
6. `sample_index` 규약 변경

## 11A. 독립 재해석 data flow 예외

독립 재해석 실험에서는 이 `src/data_prep.py` 계약을 강제로 사용하지 않는다. 해당 실험은 저자 코드의 raw data read, preprocessing, training, evaluation 흐름을 보존하고, `src/reinterpretation/` 쪽 wrapper 또는 hook 에서 PSD accumulator 를 삽입한다. 다만 output schema 와 디렉터리 이름은 각 case root 아래 `dataset_psd/` 와 `psd_analysis/` 를 유지한다.

## 12. 금지 사항

1. split shard size-limit parameter 를 다시 도입하면 안 된다.
2. shard 디렉터리, `part-xxxxx` 파일, sidecar `labels.npy`, sidecar `sample_indices.npy`, sidecar base npy 파일을 생성하면 안 된다.
3. object dtype, pickle 기반 payload, `npz` archive, torch `.pt` 를 새 공식 산출물로 사용하면 안 된다.
4. downstream 실험이 raw dataset 파일을 직접 열면 안 된다.
5. downstream 단계에서 dataset 별 전처리 규칙을 다시 적용하면 안 된다.
6. official view 를 해석할 수 없는 형태로 stored input view 를 저장하면 안 된다.
7. 공식 split payload 경로를 timestamp 나 실험 시나리오 의존 이름으로 만들어 split 내부 고정 순서 재현성을 해치면 안 된다.


## 13. one-sample streaming writer

`data_prep` writer 는 split 전체를 RAM 에 쌓지 않는다. writer 의 real memory upper bound 는 원칙적으로 현재 처리 중인 raw sample 1개, 해당 sample 의 전처리 중간 tensor, mmap write buffer, manifest metadata 로 제한한다.

공식 pseudo flow 는 아래다.

```text
for split in splits:
  N = count_samples_without_materializing_payload(split)
  dst = open_memmap(tmp_path, dtype=record_dtype, shape=(N,))
  for j, raw_ref in enumerate(iter_raw_sample_refs(split)):
    raw = read_one_sample(raw_ref)
    x, y, sample_index = preprocess_one_sample(raw)
    dst[j]["sample_index"] = sample_index
    dst[j]["label"] = y
    dst[j]["input"] = x
    release(raw)
    release(x)
  flush_and_fsync(dst)
  validate_with_mmap_read(tmp_path)
  atomic_rename(tmp_path, final_path)
```

아래 구현은 금지한다.

1. `inputs = []` 에 split 전체 sample 을 append 한 뒤 `np.stack` 하는 방식
2. split 전체를 torch tensor 로 만든 뒤 `.numpy()` 로 저장하는 방식
3. preprocessing 결과를 `.pt`, `.npz`, pickle, object dtype 으로 임시 저장하는 방식
4. DVS 또는 SHD event 를 split 전체 dense tensor 로 먼저 변환하는 방식
5. CPU 에서 저장 dtype 을 학습 dtype 으로 미리 확장하는 방식

## 14. SNN 입력 축 순서와 PSD axis metadata

stored input view 는 supervised model 이 기대하는 입력 순서와 같아야 한다. 시간축이 있는 project-side SNN 입력은 가능한 한 time-major 로 통일한다. 그러나 원 논문 또는 author code 를 그대로 사용하는 모델에서는 author-code layout 을 우선하고, PSD 분석은 axis metadata 를 사용한다.

| dataset | stored/model input shape | logical PSD shape | note |
| --- | --- | --- | --- |
| `s-mnist` | `(784, 1)` | `(1, 784)` | scalar sequence |
| `ps-mnist` | `(784, 1)` | `(1, 784)` | permuted scalar sequence |
| `s-cifar10` | `(1024, 3)` | `(3, 1024)` | RGB triplet sequence |
| `shd` | `(1200, 700)` | `(700, 1200)` | project-side time-major model input |
| `ssc` | `(1000, 700)` | `(700, 1000)` | project-side time-major model input |
| `deap` | `(384, 32)` | `(32, 384)` | EEG sequence |
| `uci-har` | `(128, 6)` | `(6, 128)` | inertial sequence |
| `mnist` | `(4, 1, 28, 28)` | `(784, 4)` | data_prep 저장 단계에서 4-frame static repeat |
| `cifar-10` | `(4, 3, 32, 32)` | `(3072, 4)` | data_prep 저장 단계에서 4-frame static repeat |
| `n-mnist` | `(10, 2, 34, 34)` | `(2312, 10)` | DVS frame input |
| `cifar10-dvs` | `(20, 2, 128, 128)` | `(32768, 20)` | DVS frame input |
| `dvs128-gesture` | `(20, 2, 128, 128)` | `(32768, 20)` | DVS frame input |

`manifest.json` 은 아래 필드를 추가로 기록한다.

```text
model_input_axis_order
psd_sample_axis
psd_batch_axis
psd_time_axis
psd_row_axes
psd_feature_axes
psd_token_axes
psd_flatten_rule
psd_logical_shape
stored_order_is_model_input_order = true
```

`psd_flatten_rule` 은 transpose, permute, flatten, reshape, channel-preserving raster flatten, token flatten, token pooling 중 필요한 해석 규칙을 표현한다. 이 규칙은 의미 변경이나 재전처리가 아니라 PSD logical view 정의다. 명세는 특정 dataset 에 대해 runtime transpose 를 필수 요구하지 않는다.

## 15. dtype 과 GPU-side cast

저장 dtype 은 다음과 같이 고정한다.

| data kind | stored dtype | GPU 변환 |
| --- | --- | --- |
| real-valued sequence | `float32` | 필요 시 `float16`, `bfloat16`, `float32` 로 GPU 에서 cast |
| binary spike occupancy | `uint8` | GPU 전송 후 floating tensor 로 cast |
| event count frame | unsigned integer | GPU 전송 후 학습 dtype 으로 cast |
| label | `int64` | loss 함수 입력 직전 device 이동 |
| sample index | `int64` | metadata 용도 |

CPU DataLoader 는 binary 또는 count 입력을 미리 `float32` batch 로 확장하지 않는다. batch collation 은 저장 dtype 을 유지하고, GPU transfer 이후 model wrapper 가 필요한 dtype 변환을 수행한다.

## 16. bash script 와 로그

`data_prep` 실행은 `Spec/impl/bash_execution.md` 를 따른다. `bash/data_prep.sh` 또는 후속 bash launcher 는 각 dataset 또는 split 작업을 `nohup` background 로 실행하고, stdout 과 stderr 를 외부 지정 `LOG_ROOT` 아래로 redirect 해야 한다.

## 17. 독립 재해석 data_prep profile 구현 계약

독립 재해석 실험은 `Spec/theory/data_prep/data_prep.md` 의 author-code profile 을 구현에서 분리해 다룬다. 저자 코드의 raw data read, preprocessing, training, evaluation 흐름을 유지해야 하므로 project-standard `src/data_prep.py` 를 강제하지 않는다. 구현은 project-standard prepared bundle 과 author-code profile 을 같은 dataset name 만으로 합치면 안 된다.

### IMPL-DATA-REINT-001

요구사항:
재해석 profile 은 project-standard `src/data_prep.py` 실행과 분리하고, `src/reinterpretation/` driver 또는 wrapper 가 author-code data flow 안에서 profile metadata 를 기록해야 한다.

적용 대상:
- `src/reinterpretation/driver.py`
- `src/reinterpretation/`
- `src/data/specs.py`

수용 기준:
- `cifar10-dvs` 의 project-standard $T=20$ bundle 과 `need_high_cifar10_dvs_t16` author-code flow 가 같은 output directory 를 공유하지 않는다.
- `shd` 의 project-standard $T=1200$ bundle, `drf_shd_t250` flow, `dh_snn_shd_t1000` flow 가 metadata 로 구분된다.
- 재해석 flow 는 core `data_prep`, `dataset_psd`, `psd_analysis` 호출 순서를 전제로 하지 않는다.
- 산출물은 각 reinterpretation case root 아래 `dataset_psd/` 와 `psd_analysis/` 로 저장한다.

### IMPL-DATA-REINT-002

요구사항:
reinterpretation profile 의 `manifest.json` 은 최소 아래 필드를 포함해야 한다.

```text
prep_profile
origin_code_root
origin_paper
origin_config_path
training_view_name
psd_view_name
stored_shape
psd_sample_axis
psd_batch_axis
psd_time_axis
psd_row_axes
psd_flatten_rule
psd_logical_shape
layout_source
```

수용 기준:
- wrapper 또는 observer 가 author code tensor 를 PSD accumulator 로 넘길 때 axis metadata 로 logical PSD view 를 구성할 수 있다.
- author code 의 physical axis order 와 PSD logical view metadata 가 동시에 남는다.

### IMPL-DATA-REINT-003

요구사항:
Need High CIFAR10-DVS profile 은 author-code frame length T=16 을 사용하고, PSD logical view 는 rows = 32768, time = 16 으로 해석해야 한다.

수용 기준:
- `manifest.json` 의 `prep_profile` 은 `need_high_cifar10_dvs_t16` 이다.
- `stored_shape` 은 `(16, 2, 128, 128)` 이다.
- `psd_logical_shape` 은 `(32768, 16)` 이다.

### IMPL-DATA-REINT-004

요구사항:
D-RF SHD profile 은 SHD 250 timestep condition 을 사용하고, PSD logical view 는 rows = 700, time = 250 으로 해석해야 한다.

수용 기준:
- `manifest.json` 의 `prep_profile` 은 `drf_shd_t250` 이다.
- `stored_shape` 은 `(250, 700)` 이다.
- `psd_logical_shape` 은 `(700, 250)` 이다.

### IMPL-DATA-REINT-005

요구사항:
DH-SNN SHD profile 은 author setting 의 sequence length 1000, input dimension 700 을 사용하고, PSD logical view 는 rows = 700, time = 1000 으로 해석해야 한다.

수용 기준:
- `manifest.json` 의 `prep_profile` 은 `dh_snn_shd_t1000` 이다.
- `stored_shape` 은 `(1000, 700)` 이다.
- `psd_logical_shape` 은 `(700, 1000)` 이다.

## 18. static repeat schedule metadata

### IMPL-DATA-STATIC-001

요구사항:
정적 image 를 SNN time 으로 반복 입력하는 prepared sample 은 반복 schedule 을 manifest 와 sample metadata 에 기록해야 한다.

Required metadata fields:
- `is_static_repeat`: bool
- `static_repeat_T`: int 또는 null
- `repeat_schedule`: string 또는 structured object
- `time_axis_semantics`: `repeated_static`, `event_time`, `sensor_time`, `sequence_time` 중 하나

수용 기준:
- 반복 입력을 PSD 분석에서 제외하지 않는다.
- 반복 schedule signature 가 PSD 결과에 나타났을 때, metadata 로 해당 결과가 repeated static input 에서 나온 것임을 추적할 수 있어야 한다.

## 2026-05-01 구현 보정: 정적 이미지 반복 주입

정적 이미지 데이터셋 `mnist`, `cifar-10` 은 모델 입력으로 한 번만 주입하지 않고 같은 이미지를 4 time step 동안 반복 주입한다. 반복은 runtime/model code에서 만들지 않고 data_prep 저장 단계에서 만든다.

```text
static_repeat_T = 4
mnist cnn view shape = (4, 1, 28, 28)
mnist flatten view shape = (4, 784)
cifar-10/cifar-100 cnn view shape = (4, 3, 32, 32)
cifar-10/cifar-100 flatten view shape = (4, 3072)
sequence_length = 4
input_dim = C*H*W
cnn_input_shape = [4, C, H, W]
flatten_input_shape = [4, C*H*W]
```

`model_input_cnn`, `cnn_input`, `psd_input`, `image_psd_view` 는 `(T,C,H,W)` 를 반환한다. `model_input_flatten`, `flatten_input`, `sequence_input` 은 `(T,C*H*W)` 를 반환한다. CNN 계열 모델은 dataloader batch인 `(B,T,C,H,W)` 를 그대로 받고, Dense 계열 모델은 dataloader batch인 `(B,T,C*H*W)` 를 받는다. PSD 분석은 `tensor_to_channel_major_maps` 단계에서 CNN view 를 `(B,T,C,H,W) -> (B,C*H*W,T)` 로 해석한다.

## 2026-05-02 수정 고정

- 새 실험 launcher 기본 preset 에서 MNIST 를 제외한다.
- 정적 이미지 반복 입력은 `static_repeat` profile 로 취급한다.
- manifest 에 PSD axis metadata 를 명시하고, 이후 dataset PSD 및 analysis CSV 에 전달한다.

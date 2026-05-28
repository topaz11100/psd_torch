# 체크포인트 오류 원인

**User:** Anonymous  
**Created:** 5/27/2026 17:32:18  
**Updated:** 5/27/2026 18:12:51  
**Exported:** 5/28/2026 19:12:43  
**Link:** [https://chatgpt.com/g/g-p-6945792ff4ac8191a44ea5c89eaac2bf-hanyongog/c/6a16ac0e-d644-83ab-be1c-49a249478296](https://chatgpt.com/g/g-p-6945792ff4ac8191a44ea5c89eaac2bf-hanyongog/c/6a16ac0e-d644-83ab-be1c-49a249478296)  

## Prompt:
2026. 5. 27. 오후 5:32:15

topaz11100/psd 레포에서 psd_analysis 실행시

```  
payload = torch.load(path, map_location=map_location)

psd_analysis:checkpoints:   0%|          | 0/30 [00:00<?, ?it/s]
                                                                
Traceback (most recent call last):
  File "/home/yongokhan/바탕화면/psd/src/psd_analysis.py", line 1689, in <module>
    raise SystemExit(main())
  File "/home/yongokhan/바탕화면/psd/src/patch_overlays/runtime_patch.py", line 484, in main_patched
    model, _readout, model_spec, readout_mode = g['_build_model_from_checkpoint'](payload, device=device)
  File "/home/yongokhan/바탕화면/psd/src/psd_analysis.py", line 285, in _build_model_from_checkpoint
    raise ValueError('Checkpoint model_config must be a mapping.')
ValueError: Checkpoint model_config must be a mapping.
```

이거 무슨 오류야

## Response:
2026. 5. 27. 오후 5:32:18

레포의 현재 `main` 기준으로 `psd_analysis.py`의 체크포인트 로딩 경로와 학습 체크포인트 저장 형식을 대조해서 원인을 좁혀보겠습니다. 우선 traceback만 보면 핵심은 체크포인트 안의 `model_config`가 dict가 아니라는 점입니다.

현재 `main` 코드 기준으로 불일치가 보입니다. `psd_analysis.py`는 체크포인트에 `model_config`와 `readout_config`라는 dict가 있다고 가정하는데, `model_training.py`의 저장 payload는 현재 그대로 보면 그 두 필드를 직접 넣지 않습니다.

오류의 직접 원인은 **체크포인트 `.pt` 안에 들어있는 `model_config` 필드가 dict 형태가 아니기 때문** 입니다.

## 핵심 원인

`psd_analysis.py`의 `_build_model_from_checkpoint()`는 체크포인트 payload에서 다음 구조를 기대합니다.

```python
model_config = payload.get('model_config')
readout_config = payload.get('readout_config')

if not isinstance(model_config, Mapping):
    raise ValueError('Checkpoint model_config must be a mapping.')
```

즉 체크포인트 내부가 대략 아래처럼 생겨야 합니다.

```python
{
    "model_token": "...",
    "model_config": {
        "input_dim": ...,
        "sequence_length": ...,
        "num_classes": ...,
        "hidden_spec": ...,
        "v_th": ...
    },
    "readout_config": {
        "mode": "final_membrane"
    },
    "state_dict": ...
}
```

그런데 현재 로드한 체크포인트에는 `model_config`가 없거나, dict가 아니라서 바로 죽은 것입니다. `psd_analysis.py`는 `model_config`와 `readout_config`가 모두 `Mapping`, 즉 dict류 객체여야 한다고 강제합니다. `psd_analysis.py`

## 현재 레포 기준으로 보면 코드 간 schema mismatch 가능성이 큼

현재 `main`의 `model_training.py`에서 체크포인트 저장 payload를 보면 `_checkpoint_payload()`는 기본적으로 전달받은 값들에 `schema_version`과 `state_dict`만 추가합니다. `model_training.py`

그리고 실제 저장 시점의 payload에는 `model_token`, `readout_mode`, `dataset_token`, `prep_root`, `prepared_dataset_path`, `seed`, `training_args`, `metric_snapshot` 등은 들어가지만, `psd_analysis.py`가 요구하는 **`model_config`와 `readout_config`는 직접 저장되지 않습니다.** `model_training.py`

따라서 이건 단순히 사용자가 파일을 잘못 고른 문제일 수도 있지만, 현재 코드만 보면 더 근본적으로는:

> **`model_training.py`가 저장하는 checkpoint schema와 `psd_analysis.py`가 읽는 checkpoint schema가 서로 안 맞는 상태**

입니다.

## 왜 `runtime_patch.py`가 traceback에 보이나?

`psd_analysis.py` 하단에서 runtime patch를 적용하고, 실제 실행은 `runtime_patch.py`의 `main_patched()`로 바뀐 상태입니다. 이 patched main은 체크포인트를 하나씩 로드한 뒤 그대로 기존 `_build_model_from_checkpoint()`를 호출합니다. `runtime_patch.py`

즉 `runtime_patch.py`가 직접 원인이라기보다, patch된 실행 경로에서도 결국 기존 `psd_analysis._build_model_from_checkpoint()`가 호출되고, 거기서 `model_config` 검증에 실패한 것입니다.

## 먼저 확인할 것

아래를 체크포인트 파일 하나에 대해 실행해보면 바로 확인됩니다.

```bash
python - <<'PY'
import torch
from pathlib import Path

path = Path("/home/yongokhan/바탕화면/psd/여기에/체크포인트.pt")
payload = torch.load(path, map_location="cpu")

print("keys:")
for k in sorted(payload.keys()):
    print(" ", k, type(payload[k]))

print("\nmodel_config =", payload.get("model_config"))
print("type(model_config) =", type(payload.get("model_config")))

print("\nreadout_config =", payload.get("readout_config"))
print("type(readout_config) =", type(payload.get("readout_config")))

print("\nreadout_mode =", payload.get("readout_mode"))
print("model_token =", payload.get("model_token"))
print("training_args =", payload.get("training_args"))
PY
```

아마 다음 중 하나일 가능성이 큽니다.

```text
model_config = None
readout_config = None
readout_mode = final_membrane
model_token = ...
```

## 임시 해결 방향

### 1. 체크포인트가 진짜 모델 학습 체크포인트인지 확인

`psd_analysis --checkpoint`에는 다음 같은 파일/폴더를 넣어야 합니다.

```text
checkpoint_epoch_000001.pt
checkpoint_epoch_000002.pt
...
```

그리고 `psd_analysis.py`의 checkpoint resolver는 디렉터리 안에 `.pt` 일반 파일만 있어야 하고, 하위 폴더가 있으면 에러를 내는 strict 구조입니다. `psd_analysis.py`

잘못해서 `pca_basis.pt`, `best_model.pt`, 다른 실험 산출물, 또는 예전 schema의 checkpoint를 넣어도 비슷하게 터질 수 있습니다.

### 2. 근본 수정은 둘 중 하나

#### 선택 A: `model_training.py` 저장 payload에 `model_config`, `readout_config` 추가

이게 정석입니다. 저장 시점에 아래 metadata를 넣어야 합니다.

```python
model_config={
    'input_dim': int(bundle.input_dim),
    'sequence_length': int(bundle.sequence_length),
    'num_classes': int(bundle.num_classes),
    'input_shape': None,
    'hidden_spec': str(args.hidden_spec).strip(),
    'arch_spec': str(args.hidden_spec).strip(),
    'v_th': float(args.v_th),
},
readout_config={
    'mode': canonical_readout_mode,
    'readout_mode': canonical_readout_mode,
},
```

즉 `payload=_checkpoint_payload(...)` 호출부에 위 두 항목을 추가하면 됩니다.

#### 선택 B: `psd_analysis.py`가 legacy checkpoint도 읽도록 fallback 추가

예를 들어 `model_config`가 없으면 `payload`의 top-level metadata와 CLI/bundle 정보에서 복원하도록 바꾸는 방식입니다. 다만 현재 `_build_model_from_checkpoint()`는 bundle을 resolve하기 전에 model을 먼저 만들기 때문에, 이 방식은 함수 구조를 좀 더 바꿔야 합니다.

## 결론

이 에러는 CUDA나 PSD 계산 문제가 아니라 **checkpoint metadata schema 문제**입니다.

현재 `psd_analysis.py`는 `model_config`/`readout_config` 기반 체크포인트를 기대하는데, 현재 `model_training.py` 저장 payload는 그 구조를 넣지 않습니다. 그래서 `payload.get("model_config")`가 `None`이 되고, 곧바로:

```text
ValueError: Checkpoint model_config must be a mapping.
```

이 발생한 것입니다.

## Prompt:
2026. 5. 27. 오후 5:37:14

이게 대상 파일인데 너가 진단해봐

[Attachment: checkpoint_epoch_000030.pt]

## Response:
2026. 5. 27. 오후 5:37:18

> **검사 위한 체크포인트 파일 크기 확인**
>
>
> **로딩된 데이터 타입 검사**
>
>
> **Keys 검사하기**
>
>
> **진단 및 체크포인트 수정 고려 중**
>
> 모델 구성을 유추하려면 state_dict의 형태를 살펴보아야 할 것 같아요. 첫 번째 레이어의 `input_weight` 형태로 input_dim을 추론하고, 가중치에서 hidden size를, output_layer에서 클래스 수를 유추할 수 있습니다. 하지만 sequence_length는 manifest에서만 확인 가능해서 이 부분이 문제네요.
>
> **Evaluating model configuration from state_dict shapes**
>
>
> **제약 조건 살펴보기**
>
> 이 작업을 수행하는 데 있어 제약 조건을 고려해야 할 것 같아요. 어떤 제한 사항이나 조건이 결과에 영향을 미칠지 점검해봐야 할 것 같아요. 필요할 수 있는 정보를 발견하고, 그에 맞춰 해결할 수 있도록 해야겠네요.
>
> **검사할 제약 조건 출력**
>
>
> **진단 및 수정 제안**
>
> 사용자가 진단을 요청했으니, 먼저 업로드된 체크포인트를 확인한 결과 `model_config/readout_config`가 없고, `training_args`에 데이터셋 정보는 포함되지 않았어요. sequence_length는 누락되고, 이를 해결하려면 관련 매니페스트를 확인해야 합니다. 수정은 필요할 수 있지만, 요청 없이는 파일 수정은 최소화해야겠어요.
>
> 1m 17s 동안 생각함

올린 `.pt` 파일 자체를 직접 열어서 payload key/type, `model_config`, `readout_config`, `state_dict` prefix를 확인하겠습니다. 텐서 전체는 출력하지 않고 metadata와 shape만 보겠습니다.

진단 결과, **파일 자체가 깨진 것은 아닙니다.** 정상적으로 `torch.load()` 됩니다. 문제는 checkpoint metadata schema가 `psd_analysis.py`가 기대하는 형태와 다릅니다.

## 1. 업로드한 체크포인트의 실제 payload 구조

대상 파일:

```text
checkpoint_epoch_000030.pt
```

로드 결과 key는 다음과 같습니다.

```text
compile_model_applied
compile_model_policy
compile_model_requested
constraint_metadata
dataset_token
epoch
metric_snapshot
model_token
prep_root
prepared_dataset_path
psd_regularization_metadata
readout_mode
schema_version
seed
state_dict
training_args
```

중요한 부분은 아래입니다.

```python
model_config = None
readout_config = None
readout_mode = "temporal_membrane"
model_token = "lif_soft_fixed"
dataset_token = "s-mnist"
epoch = 30
schema_version = "psd_checkpoint_v1"
```

즉, 이 checkpoint에는 `psd_analysis.py`가 요구하는 `model_config`와 `readout_config`가 **아예 없습니다.**

## 2. 그래서 traceback이 정확히 맞음

`psd_analysis.py`는 checkpoint를 읽은 뒤 아래처럼 검사합니다.

```python
model_config = payload.get('model_config')
readout_config = payload.get('readout_config')

if not isinstance(model_config, Mapping):
    raise ValueError('Checkpoint model_config must be a mapping.')
```

현재 파일의 `model_config`는 `None`이므로:

```text
ValueError: Checkpoint model_config must be a mapping.
```

이 발생하는 게 맞습니다. `readout_config`도 없기 때문에 `model_config`를 통과하더라도 다음 단계에서 또 죽습니다. 해당 검증은 `psd_analysis.py`의 `_build_model_from_checkpoint()`에 있습니다. `psd_analysis.py`

## 3. state_dict 자체는 정상적으로 보임

`state_dict`는 있고, shape도 일관됩니다.

```text
hidden_layers.0.input_weight   (32, 1)
hidden_layers.1.input_weight   (32, 32)
hidden_layers.2.input_weight   (32, 32)
output_layer.input_weight      (10, 32)
```

따라서 모델 구조는 거의 다음으로 추정됩니다.

```python
model_token = "lif_soft_fixed"
input_dim = 1
hidden_spec = "32-32-32"
num_classes = 10
readout_mode = "temporal_membrane"
dataset = "s-mnist"
```

다만 `sequence_length`는 `state_dict`만으로는 안 나옵니다. S-MNIST라면 보통 `784`일 가능성이 높지만, 코드상으로는 prepared dataset manifest/bundle에서 읽어야 안전합니다.

## 4. 현재 checkpoint는 `model_training.py`가 저장한 schema 그대로임

현재 `model_training.py`의 저장부를 보면 `_checkpoint_payload()`는 `schema_version`과 `state_dict`를 붙이고, 호출부에서는 `model_token`, `readout_mode`, `dataset_token`, `training_args`, `metric_snapshot` 등을 저장합니다. `model_training.py` `model_training.py`

그런데 `psd_analysis.py`가 요구하는 `model_config`, `readout_config`는 저장되지 않습니다.

정리하면:

> **체크포인트는 정상 파일이지만, `model_training.py`가 저장한 checkpoint schema와 `psd_analysis.py`가 읽으려는 checkpoint schema가 불일치합니다.**

## 5. 바로 고치려면

### 권장 수정: `model_training.py` 저장 payload에 metadata 추가

`model_training.py`의 payload 생성부에 아래 두 항목을 넣어야 합니다.

```python
model_config={
    'input_dim': int(bundle.input_dim),
    'sequence_length': int(bundle.sequence_length),
    'num_classes': int(bundle.num_classes),
    'input_shape': None,
    'hidden_spec': str(args.hidden_spec).strip(),
    'arch_spec': str(args.hidden_spec).strip(),
    'v_th': float(args.v_th),
},
readout_config={
    'mode': canonical_readout_mode,
    'readout_mode': canonical_readout_mode,
},
```

대략 이 위치입니다.

```python
payload = _checkpoint_payload(
    epoch=epoch,
    model=model,
    model_token=model_spec.canonical_token,
    readout_mode=canonical_readout_mode,

    # 추가
    model_config={
        'input_dim': int(bundle.input_dim),
        'sequence_length': int(bundle.sequence_length),
        'num_classes': int(bundle.num_classes),
        'input_shape': None,
        'hidden_spec': str(args.hidden_spec).strip(),
        'arch_spec': str(args.hidden_spec).strip(),
        'v_th': float(args.v_th),
    },
    readout_config={
        'mode': canonical_readout_mode,
        'readout_mode': canonical_readout_mode,
    },

    dataset_token=dataset_token,
    prep_root=str(prep_root),
    prepared_dataset_path=str(prepared_dataset_path),
    seed=int(args.seed),
    ...
)
```

## 6. 이미 만들어진 이 checkpoint만 임시 복구하려면

S-MNIST 설정이 확실히 `sequence_length=784`라면, 이 파일은 아래처럼 metadata를 주입해서 복구할 수 있습니다.

```python
import torch
from pathlib import Path

src = Path("checkpoint_epoch_000030.pt")
dst = Path("checkpoint_epoch_000030.fixed.pt")

payload = torch.load(src, map_location="cpu")

payload["model_config"] = {
    "input_dim": 1,
    "sequence_length": 784,
    "num_classes": 10,
    "input_shape": None,
    "hidden_spec": "32-32-32",
    "arch_spec": "32-32-32",
    "v_th": 1.0,
}

payload["readout_config"] = {
    "mode": payload.get("readout_mode", "temporal_membrane"),
    "readout_mode": payload.get("readout_mode", "temporal_membrane"),
}

torch.save(payload, dst)
print(f"saved: {dst}")
```

단, 이건 **임시 처방**입니다. 근본적으로는 앞으로 생성되는 checkpoint에 `model_config`와 `readout_config`가 저장되도록 `model_training.py`를 고치는 게 맞습니다.

## 결론

이 파일의 문제는 다음 하나입니다.

```text
model_config / readout_config 누락
```

파일 손상, CUDA 문제, PSD 계산 문제, state_dict 문제는 아닙니다.  
현재 checkpoint는 학습 결과로는 정상에 가까우나, `psd_analysis.py`가 요구하는 분석용 metadata가 빠져 있어서 분석 단계에서 모델 재구성이 불가능한 상태입니다.

## Prompt:
2026. 5. 27. 오후 5:56:20

![image.png](https://chatgpt.com/backend-api/estuary/content?id=file_00000000d168720b9fe88609ab4a8ffe&ts=494434&p=fs&cid=1&sig=da8f86d64a829f96eda53dd9380f2082c1b88e68265749af4e11845ea85f4874&v=0)

아하 그럼 지금 빠진 메타데이터가 뭔지 설명해주고
일단 만들어진 .pt 는 살려야하거든? 그래서 저 상위경로에서 재귀적으로 그 하위의 모든 .pt 에 대해 빠진 메타데이터를 재워넣는 코드를 작성해줘
s-mnist, shd 만썼고 각각타임스텝 784, 1200 이야

문제되지 않는 부분은 절대로 건드리지 않고 빠진 메타데이터만 채워넣는식으로만 해줘

## Response:
2026. 5. 27. 오후 5:56:27

> **검토 후 코드 생성 준비 중**
>
> 사용자의 요청에 맞춰 코드 생성을 준비하고 있어요. GitHub에서 관련 파일들을 이미 참조한 적이 있기 때문에 추가로 웹 검색은 필요 없어 보입니다. 코드에서는 주어진 내용을 기반으로 필요한 메타데이터만 추가하고, 손대지 말아야 할 부분은 그대로 두 있도록 할 거예요.
>
> **검토 후 SNN 모델 설정**
>
> 이 스크립트는 Dense SNN만 지원하는 것 같네요. 특히 `v_th`와 관련해서는, 만약 모델이 이를 저장하지 않으면 기본값인 1.0을 사용해야 할 듯합니다. `state_dict`에서 `v_threshold_param` 키가 있으면 이를 참조할 수 있을 거예요. 필요한 키를 찾아볼 필요가 있습니다.
>
> 57s 동안 생각함

맞습니다. 현재 만들어진 `.pt`는 **학습 가중치 자체가 문제인 게 아니라, `psd_analysis.py`가 모델을 재구성할 때 필요한 metadata가 빠진 상태** 입니다.

## 1. 빠진 metadata

현재 `psd_analysis.py`는 checkpoint에서 최소한 아래 두 dict를 기대합니다.

```python
payload["model_config"]
payload["readout_config"]
```

구체적으로 필요한 값은 다음입니다.

```python
model_config = {
    "input_dim": ...,
    "sequence_length": ...,
    "num_classes": ...,
    "input_shape": None,
    "hidden_spec": ...,
    "arch_spec": ...,
    "v_th": ...
}

readout_config = {
    "mode": ...,
    "readout_mode": ...
}
```

`psd_analysis.py`는 `model_config`와 `readout_config`가 dict류 객체인지 검사하고, 아니면 바로 에러를 냅니다. `psd_analysis.py`  
반면 현재 `model_training.py`의 checkpoint 저장부는 `model_token`, `readout_mode`, `dataset_token`, `training_args`, `metric_snapshot`, `state_dict` 등은 저장하지만, `model_config`/`readout_config`를 직접 넣지 않는 구조입니다. `model_training.py`

즉, 지금 `.pt`에 추가로 채워야 하는 건 **가중치가 아니라 분석용 모델 재구성 metadata** 입니다.

---

## 2. 재귀적으로 모든 `.pt`에 누락 metadata만 채우는 코드

아래 파일을 예를 들어 프로젝트 루트에 `fill_checkpoint_metadata.py`로 저장하세요.

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

DATASET_SEQUENCE_LENGTH = {
    "s-mnist": 784,
    "s_mnist": 784,
    "smnist": 784,
    "shd": 1200,
}

def torch_load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict):
        raise ValueError(f"not a dict payload: {path}")

    return payload

def infer_dataset_token(payload: Mapping[str, Any], path: Path) -> str:
    raw = str(
        payload.get("dataset_token")
        or payload.get("training_args", {}).get("dataset")
        or ""
    ).strip().lower()

    path_lower = str(path).lower()

    candidates = [raw, path_lower]
    for item in candidates:
        if "s-mnist" in item or "s_mnist" in item or "smnist" in item:
            return "s-mnist"
        if re.search(r"(^|[^a-z0-9])shd([^a-z0-9]|$)", item):
            return "shd"

    raise ValueError("cannot infer dataset token")

def sequence_length_for_dataset(dataset_token: str) -> int:
    key = dataset_token.strip().lower()
    if key not in DATASET_SEQUENCE_LENGTH:
        raise ValueError(f"unsupported dataset token for sequence_length: {dataset_token!r}")
    return int(DATASET_SEQUENCE_LENGTH[key])

def strip_state_prefix(key: str) -> str:
    prefixes = (
        "module._orig_mod.",
        "_orig_mod.",
        "module.",
    )
    for prefix in prefixes:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key

def infer_arch_from_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    hidden_by_index: dict[int, tuple[int, int]] = {}
    output_shape: tuple[int, int] | None = None

    for raw_key, value in state_dict.items():
        key = strip_state_prefix(str(raw_key))

        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim != 2:
            continue

        m = re.fullmatch(r"hidden_layers\.(\d+)\.input_weight", key)
        if m:
            idx = int(m.group(1))
            out_dim = int(value.shape[0])
            in_dim = int(value.shape[1])
            hidden_by_index[idx] = (out_dim, in_dim)
            continue

        if key == "output_layer.input_weight":
            output_shape = (int(value.shape[0]), int(value.shape[1]))

    if not hidden_by_index:
        raise ValueError("cannot infer hidden layers from state_dict")

    hidden_sizes = [hidden_by_index[i][0] for i in sorted(hidden_by_index)]
    first_idx = min(hidden_by_index)
    input_dim = int(hidden_by_index[first_idx][1])

    if output_shape is None:
        raise ValueError("cannot infer num_classes from output_layer.input_weight")

    num_classes = int(output_shape[0])
    hidden_spec = "-".join(str(v) for v in hidden_sizes)

    return {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "hidden_spec": hidden_spec,
    }

def infer_readout_mode(payload: Mapping[str, Any]) -> str:
    mode = str(
        payload.get("readout_mode")
        or payload.get("training_args", {}).get("readout_mode")
        or ""
    ).strip()

    if not mode:
        raise ValueError("cannot infer readout_mode")

    return mode

def infer_v_th(payload: Mapping[str, Any]) -> float:
    training_args = payload.get("training_args")
    if isinstance(training_args, Mapping) and "v_th" in training_args:
        return float(training_args["v_th"])

    # 현재 model_training.py 기본값.
    # 기존 checkpoint에 v_th가 저장되지 않았으면 완전 복원은 불가능하므로 기본값만 사용.
    return 1.0

def is_probably_training_checkpoint(payload: Mapping[str, Any]) -> bool:
    if "state_dict" not in payload:
        return False
    if "model_token" not in payload and "training_args" not in payload:
        return False
    if "dataset_token" not in payload and "training_args" not in payload:
        return False
    return True

def build_missing_metadata(payload: Mapping[str, Any], path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("state_dict is missing or not a mapping")

    dataset_token = infer_dataset_token(payload, path)
    sequence_length = sequence_length_for_dataset(dataset_token)
    arch = infer_arch_from_state_dict(state_dict)
    readout_mode = infer_readout_mode(payload)
    v_th = infer_v_th(payload)

    model_config = {
        "input_dim": int(arch["input_dim"]),
        "sequence_length": int(sequence_length),
        "num_classes": int(arch["num_classes"]),
        "input_shape": None,
        "hidden_spec": str(arch["hidden_spec"]),
        "arch_spec": str(arch["hidden_spec"]),
        "v_th": float(v_th),
    }

    readout_config = {
        "mode": str(readout_mode),
        "readout_mode": str(readout_mode),
    }

    return model_config, readout_config

def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)

    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

def patch_checkpoint(
    path: Path,
    *,
    write: bool,
    backup: bool,
    repair_non_mapping: bool,
) -> tuple[str, str]:
    payload = torch_load_checkpoint(path)

    if not is_probably_training_checkpoint(payload):
        return "skip", "not a training checkpoint-like payload"

    model_config_existing = payload.get("model_config")
    readout_config_existing = payload.get("readout_config")

    model_config_ok = isinstance(model_config_existing, Mapping)
    readout_config_ok = isinstance(readout_config_existing, Mapping)

    if model_config_ok and readout_config_ok:
        return "skip", "already has model_config/readout_config"

    if (
        ("model_config" in payload and model_config_existing is not None and not model_config_ok)
        or ("readout_config" in payload and readout_config_existing is not None and not readout_config_ok)
    ):
        if not repair_non_mapping:
            return (
                "skip",
                "model_config/readout_config exists but is non-mapping; "
                "use --repair-non-mapping to replace only those problematic fields",
            )

    model_config_new, readout_config_new = build_missing_metadata(payload, path)

    changed_fields: list[str] = []

    if not model_config_ok:
        payload["model_config"] = model_config_new
        changed_fields.append("model_config")

    if not readout_config_ok:
        payload["readout_config"] = readout_config_new
        changed_fields.append("readout_config")

    if not changed_fields:
        return "skip", "nothing to add"

    detail = (
        f"add {','.join(changed_fields)} "
        f"dataset={infer_dataset_token(payload, path)} "
        f"T={payload['model_config']['sequence_length']} "
        f"input_dim={payload['model_config']['input_dim']} "
        f"hidden_spec={payload['model_config']['hidden_spec']} "
        f"num_classes={payload['model_config']['num_classes']} "
        f"readout={payload['readout_config']['mode']}"
    )

    if not write:
        return "dry-run", detail

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    atomic_torch_save(payload, path)
    return "patched", detail

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recursively fill missing psd_analysis checkpoint metadata only."
    )
    parser.add_argument(
        "--root",
        required=True,
        help="Upper directory containing scenario folders. All *.pt files below it are scanned recursively.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Actually modify checkpoint files. Without this, only dry-run is performed.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create *.pt.bak backup files before overwriting.",
    )
    parser.add_argument(
        "--repair-non-mapping",
        action="store_true",
        help="Replace model_config/readout_config when they exist but are not dict-like.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    pt_files = sorted(root.rglob("*.pt"), key=lambda p: str(p))
    if not pt_files:
        print(f"[info] no .pt files found under {root}")
        return 0

    counts = {
        "patched": 0,
        "dry-run": 0,
        "skip": 0,
        "error": 0,
    }

    for path in pt_files:
        try:
            status, detail = patch_checkpoint(
                path,
                write=bool(args.write),
                backup=not bool(args.no_backup),
                repair_non_mapping=bool(args.repair_non_mapping),
            )
            counts[status] = counts.get(status, 0) + 1
            rel = path.relative_to(root)
            print(f"[{status}] {rel} :: {detail}")
        except Exception as exc:
            counts["error"] += 1
            rel = path.relative_to(root)
            print(f"[error] {rel} :: {type(exc).__name__}: {exc}")

    print(
        "[summary] "
        + " ".join(f"{key}={value}" for key, value in sorted(counts.items()))
    )

    if counts["error"] > 0:
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

---

## 3. 실행 방법

먼저 dry-run으로 확인합니다. 이 단계에서는 `.pt`를 수정하지 않습니다.

```bash
cd /home/yongokhan/바탕화면/psd

python fill_checkpoint_metadata.py \
  --root "/home/yongokhan/바탕화면/psd/ddp_train_scenario"
```

출력이 정상적으로 보이면 실제 적용합니다.

```bash
python fill_checkpoint_metadata.py \
  --root "/home/yongokhan/바탕화면/psd/ddp_train_scenario" \
  --write
```

기본적으로 각 원본 옆에 백업이 생깁니다.

```text
checkpoint_epoch_000030.pt.bak
```

백업 없이 바로 덮어쓰려면 다음처럼 실행합니다.

```bash
python fill_checkpoint_metadata.py \
  --root "/home/yongokhan/바탕화면/psd/ddp_train_scenario" \
  --write \
  --no-backup
```

---

## 4. 이 코드가 건드리는 것

수정 대상은 `.pt` payload의 아래 두 key뿐입니다.

```python
payload["model_config"]
payload["readout_config"]
```

이미 정상 dict로 들어있으면 건드리지 않습니다.

기존의 아래 항목들은 그대로 둡니다.

```text
state_dict
model_token
readout_mode
dataset_token
prep_root
prepared_dataset_path
seed
epoch
metric_snapshot
constraint_metadata
psd_regularization_metadata
compile_model_*
training_args
schema_version
```

단, `.pt` 파일은 `torch.save()`로 다시 저장되기 때문에 **바이너리 파일 바이트 자체는 다시 직렬화됩니다.** 하지만 payload 내부에서 기존 tensor와 기존 metadata 값은 그대로 유지하고, 누락된 두 dict만 추가합니다.

---

## 5. 추가되는 metadata 예시

S-MNIST LIF 32-32-32 checkpoint라면 대략 이렇게 들어갑니다.

```python
"model_config": {
    "input_dim": 1,
    "sequence_length": 784,
    "num_classes": 10,
    "input_shape": None,
    "hidden_spec": "32-32-32",
    "arch_spec": "32-32-32",
    "v_th": 1.0,
},
"readout_config": {
    "mode": "temporal_membrane",
    "readout_mode": "temporal_membrane",
}
```

SHD는 `sequence_length`만 다음처럼 들어갑니다.

```python
"sequence_length": 1200
```

`input_dim`, `hidden_spec`, `num_classes`는 `state_dict`의 `hidden_layers.*.input_weight`와 `output_layer.input_weight` shape에서 추론합니다.

## Prompt:
2026. 5. 27. 오후 6:01:05

다른거 안건드리지? 아까는 32-32 형식으로 강제저장되어서 오류낫었어 32,32 로 줘야하는데

## Response:
2026. 5. 27. 오후 6:01:10

> **코드 수정 확인 및 설명**
>
> 사용자가 "다른거 안건드리지?"라고 물었으니, 네, 수정사항은 "-".join을 ",".join으로 바꾸는 것뿐입니다. 아마 코드에서 hidden_spec을 처리할 때 하이픈 대신 쉼표를 사용해야 할 필요가 있을 수 있고, 이전 코드도 잘못된 방식으로 arch_spec을 작성했을 것입니다.
>
> 40s 동안 생각함

맞습니다. 현재 레포 기준으로는 `hidden_spec` / `arch_spec`가 **`32,32,32` 형식** 이어야 합니다. 제가 직전 코드에서 `32-32-32`로 만든 부분은 이 레포의 parser 기준으로 틀립니다.

현재 `src/model/arch_spec.py`는 `hidden_spec` 문자열을 `,`로 split하고, 각 token이 양의 정수인지 검사합니다. 즉 `64,32,20` 같은 형식을 요구하고, `32-32-32`는 숫자 token이 아니어서 오류가 납니다. `arch_spec.py`

또 `psd_analysis.py`는 `model_config["hidden_spec"]` 또는 `model_config["arch_spec"]`를 그대로 읽어서 모델 재구성에 사용합니다. `psd_analysis.py`

아래처럼 고쳐야 합니다.

```python
hidden_spec = ",".join(str(v) for v in hidden_sizes)
```

---

## 안전 수정 버전

아래 코드는 다음만 건드립니다.

1. `model_config`가 없으면 추가
2. `readout_config`가 없으면 추가
3. 이미 `model_config`가 있는데 `hidden_spec` 또는 `arch_spec`가 `32-32-32` 같은 하이픈 형식이면 **그 두 필드만** `32,32,32`로 교정
4. `state_dict`, `epoch`, `metric_snapshot`, `constraint_metadata`, `psd_regularization_metadata`, `compile_model_*`, 기존 weight tensor는 건드리지 않음

```python id="dquwbl"
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

DATASET_SEQUENCE_LENGTH = {
    "s-mnist": 784,
    "s_mnist": 784,
    "smnist": 784,
    "shd": 1200,
}

def torch_load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")

    if not isinstance(payload, dict):
        raise ValueError(f"not a dict payload: {path}")

    return payload

def infer_dataset_token(payload: Mapping[str, Any], path: Path) -> str:
    raw = str(
        payload.get("dataset_token")
        or payload.get("training_args", {}).get("dataset")
        or ""
    ).strip().lower()

    path_lower = str(path).lower()

    for item in (raw, path_lower):
        if "s-mnist" in item or "s_mnist" in item or "smnist" in item:
            return "s-mnist"
        if re.search(r"(^|[^a-z0-9])shd([^a-z0-9]|$)", item):
            return "shd"

    raise ValueError("cannot infer dataset token")

def sequence_length_for_dataset(dataset_token: str) -> int:
    key = dataset_token.strip().lower()
    if key not in DATASET_SEQUENCE_LENGTH:
        raise ValueError(f"unsupported dataset token: {dataset_token!r}")
    return int(DATASET_SEQUENCE_LENGTH[key])

def strip_state_prefix(key: str) -> str:
    for prefix in ("module._orig_mod.", "_orig_mod.", "module."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key

def infer_arch_from_state_dict(state_dict: Mapping[str, Any]) -> dict[str, Any]:
    hidden_by_index: dict[int, tuple[int, int]] = {}
    output_shape: tuple[int, int] | None = None

    for raw_key, value in state_dict.items():
        key = strip_state_prefix(str(raw_key))

        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim != 2:
            continue

        m = re.fullmatch(r"hidden_layers\.(\d+)\.input_weight", key)
        if m:
            idx = int(m.group(1))
            hidden_by_index[idx] = (int(value.shape[0]), int(value.shape[1]))
            continue

        if key == "output_layer.input_weight":
            output_shape = (int(value.shape[0]), int(value.shape[1]))

    if not hidden_by_index:
        raise ValueError("cannot infer hidden layers from state_dict")

    if output_shape is None:
        raise ValueError("cannot infer num_classes from output_layer.input_weight")

    hidden_sizes = [hidden_by_index[i][0] for i in sorted(hidden_by_index)]
    first_idx = min(hidden_by_index)
    input_dim = int(hidden_by_index[first_idx][1])
    num_classes = int(output_shape[0])

    # 중요: 이 레포는 comma-delimited hidden_spec를 요구함.
    hidden_spec = ",".join(str(v) for v in hidden_sizes)

    return {
        "input_dim": input_dim,
        "num_classes": num_classes,
        "hidden_spec": hidden_spec,
    }

def infer_readout_mode(payload: Mapping[str, Any]) -> str:
    mode = str(
        payload.get("readout_mode")
        or payload.get("training_args", {}).get("readout_mode")
        or ""
    ).strip()

    if not mode:
        raise ValueError("cannot infer readout_mode")

    return mode

def infer_v_th(payload: Mapping[str, Any]) -> float:
    training_args = payload.get("training_args")
    if isinstance(training_args, Mapping) and "v_th" in training_args:
        return float(training_args["v_th"])

    return 1.0

def is_hyphen_integer_spec(value: Any) -> bool:
    text = str(value).strip()
    return bool(re.fullmatch(r"\d+(?:-\d+)+", text))

def hyphen_to_comma_spec(value: Any) -> str:
    return str(value).strip().replace("-", ",")

def is_probably_training_checkpoint(payload: Mapping[str, Any]) -> bool:
    if "state_dict" not in payload:
        return False
    if "model_token" not in payload and "training_args" not in payload:
        return False
    if "dataset_token" not in payload and "training_args" not in payload:
        return False
    return True

def build_reference_metadata(payload: Mapping[str, Any], path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, Mapping):
        raise ValueError("state_dict is missing or not a mapping")

    dataset_token = infer_dataset_token(payload, path)
    arch = infer_arch_from_state_dict(state_dict)
    readout_mode = infer_readout_mode(payload)

    model_config = {
        "input_dim": int(arch["input_dim"]),
        "sequence_length": int(sequence_length_for_dataset(dataset_token)),
        "num_classes": int(arch["num_classes"]),
        "input_shape": None,
        "hidden_spec": str(arch["hidden_spec"]),
        "arch_spec": str(arch["hidden_spec"]),
        "v_th": float(infer_v_th(payload)),
    }

    readout_config = {
        "mode": str(readout_mode),
        "readout_mode": str(readout_mode),
    }

    return model_config, readout_config

def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path = path.resolve()
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    os.close(fd)

    tmp_path = Path(tmp_name)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

def patch_checkpoint(path: Path, *, write: bool, backup: bool) -> tuple[str, str]:
    payload = torch_load_checkpoint(path)

    if not is_probably_training_checkpoint(payload):
        return "skip", "not a training checkpoint-like payload"

    ref_model_config, ref_readout_config = build_reference_metadata(payload, path)

    changed: list[str] = []

    model_config = payload.get("model_config")
    if not isinstance(model_config, Mapping):
        payload["model_config"] = dict(ref_model_config)
        changed.append("add model_config")
    else:
        new_model_config = dict(model_config)

        # 누락된 key만 채움. 이미 있는 정상 key는 덮어쓰지 않음.
        for key, value in ref_model_config.items():
            if key not in new_model_config or new_model_config[key] in (None, ""):
                new_model_config[key] = value
                changed.append(f"fill model_config.{key}")

        # 이전에 잘못 들어간 32-32-32 형식만 교정.
        for key in ("hidden_spec", "arch_spec"):
            if key in new_model_config and is_hyphen_integer_spec(new_model_config[key]):
                old = str(new_model_config[key])
                new_model_config[key] = hyphen_to_comma_spec(new_model_config[key])
                changed.append(f"fix model_config.{key}: {old} -> {new_model_config[key]}")

        if new_model_config != dict(model_config):
            payload["model_config"] = new_model_config

    readout_config = payload.get("readout_config")
    if not isinstance(readout_config, Mapping):
        payload["readout_config"] = dict(ref_readout_config)
        changed.append("add readout_config")
    else:
        new_readout_config = dict(readout_config)

        # 누락된 key만 채움. 기존 정상 값은 덮어쓰지 않음.
        for key, value in ref_readout_config.items():
            if key not in new_readout_config or new_readout_config[key] in (None, ""):
                new_readout_config[key] = value
                changed.append(f"fill readout_config.{key}")

        if new_readout_config != dict(readout_config):
            payload["readout_config"] = new_readout_config

    if not changed:
        return "skip", "nothing to change"

    final_mc = payload["model_config"]
    final_rc = payload["readout_config"]
    detail = (
        "; ".join(changed)
        + f" | hidden_spec={final_mc.get('hidden_spec')}"
        + f" | arch_spec={final_mc.get('arch_spec')}"
        + f" | T={final_mc.get('sequence_length')}"
        + f" | readout={final_rc.get('mode')}"
    )

    if not write:
        return "dry-run", detail

    if backup:
        backup_path = path.with_suffix(path.suffix + ".bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)

    atomic_torch_save(payload, path)
    return "patched", detail

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely fill missing checkpoint metadata and fix only invalid hyphen hidden_spec."
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(root)

    pt_files = sorted(root.rglob("*.pt"), key=lambda p: str(p))
    counts = {"patched": 0, "dry-run": 0, "skip": 0, "error": 0}

    for path in pt_files:
        try:
            status, detail = patch_checkpoint(
                path,
                write=bool(args.write),
                backup=not bool(args.no_backup),
            )
            counts[status] += 1
            print(f"[{status}] {path.relative_to(root)} :: {detail}")
        except Exception as exc:
            counts["error"] += 1
            print(f"[error] {path.relative_to(root)} :: {type(exc).__name__}: {exc}")

    print("[summary] " + " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 1 if counts["error"] else 0

if __name__ == "__main__":
    raise SystemExit(main())
```

## 실행

먼저 dry-run:

```bash
python fill_checkpoint_metadata_safe.py \
  --root "/home/yongokhan/바탕화면/psd/ddp_train_scenario"
```

실제 적용:

```bash
python fill_checkpoint_metadata_safe.py \
  --root "/home/yongokhan/바탕화면/psd/ddp_train_scenario" \
  --write
```

이전에 `32-32-32`로 이미 잘못 주입된 `.pt`도 이 코드가 `model_config.hidden_spec`와 `model_config.arch_spec`만 `32,32,32`로 바꿉니다. 다른 payload key와 `state_dict`는 수정하지 않습니다.

## Prompt:
2026. 5. 27. 오후 6:07:41

이거 psd_analysis의 "num_workers": 0 는 무슨 역할이야

## Response:
2026. 5. 27. 오후 6:07:47

`psd_analysis`의 `"num_workers": 0`은 **분석용 DataLoader가 데이터를 읽을 때 사용할 별도 worker 프로세스 수** 입니다.

현재 `psd_analysis.py` parser에서 기본값이 `0`으로 정의되어 있고, `anal_batch`와 별개 인자입니다. `anal_batch`는 한 번의 분석 forward에 넣을 sample 수이고, `num_workers`는 그 batch를 만들기 위해 데이터를 로드하는 CPU-side worker 수입니다. `psd_analysis.py`

`runtime_patch.py`의 분석 루프에서도 이 값은 probe subset용 loader를 만들 때 그대로 들어갑니다.

```python id="q5jc0u"
loader = make_loader(
    subset,
    batch_size=int(anal_batch),
    shuffle=False,
    num_workers=int(num_workers),
    ...
)
```

즉 `num_workers`는 모델 forward, PSD 계산, GPU 연산을 병렬화하는 옵션이 아니라 **DataLoader 데이터 공급 병렬성** 옵션입니다. `runtime_patch.py`

## `"num_workers": 0`의 의미

```json id="cb2k4q"
"num_workers": 0
```

이면:

- 별도 worker process를 만들지 않음
- 메인 Python 프로세스가 직접 데이터를 읽고 batch를 구성
- 가장 안정적임
- worker spawn/fork overhead 없음
- mmap / numpy / torch tensor loader에서 예측 가능성이 높음
- 대신 데이터 로딩이 느리면 GPU가 기다릴 수 있음

## `num_workers > 0`이면?

예를 들어:

```json id="roea2h"
"num_workers": 4
```

이면 DataLoader가 worker process 4개를 띄워서 데이터를 미리 읽고 batch를 준비합니다.

장점:

- CPU 데이터 로딩이 병목이면 빨라질 수 있음
- GPU utilization이 올라갈 수 있음

단점:

- RAM 사용량 증가
- worker 생성 overhead 증가
- probe scope마다 loader가 반복 생성되는 분석 구조에서는 오히려 overhead가 커질 수 있음
- `.npy`/mmap 파일 핸들, pickle, fork 관련 문제가 생길 수 있음
- 재현성과 디버깅이 약간 불편해짐

## 이 프로젝트에서는 0이 꽤 합리적임

PSD analysis는 보통 학습처럼 매 epoch 수천 batch를 계속 도는 구조가 아니라, checkpoint별로 probe subset을 만들고 hidden trace를 수집합니다. 그래서 `num_workers > 0`이 항상 빠르지는 않습니다. 특히 `anal_batch=128`, prepared `.npy`/mmap 기반이면 데이터 로딩보다 모델 forward/trace 수집/PSD 계산 쪽이 더 클 가능성이 큽니다.

따라서 지금 설정:

```json id="7g3gvs"
"num_workers": 0
```

은 **안전 우선 설정** 입니다.

속도 테스트를 해보고 싶으면 다음 정도만 실험하면 됩니다.

```json id="sngqeg"
"num_workers": 2
```

또는

```json id="302iqk"
"num_workers": 4
```

다만 오류가 나거나 속도가 비슷하면 다시 `0`으로 두는 게 낫습니다.

## Prompt:
2026. 5. 27. 오후 6:09:42

csv 저장은 병렬처리안되나

## Response:
2026. 5. 27. 오후 6:09:47

> **Parallelizing CSV saving and performance considerations**
>
> The current code writes CSV files sequentially. Although parallelizing using ThreadPoolExecutor or ProcessPoolExecutor could improve speed, it comes with the risk of creating tiny files and issues with thread safety. The bottleneck is likely in the I/O, not the CSV generation itself. Performance is mainly constrained by the sequential writing rather than CSV handling.
>
> **병렬 CSV 저장에 대한 제안**
>
> 현재 코드에서는 병렬 저장이 지원되지 않습니다. 가능은 하지만 `num_workers`와 관련된 부분이 아닙니다. 각 파일에 대해 병렬 쓰기를 안전하게 하려면 각각 다른 경로에 저장하고, manifest는 정렬 후 모은 뒤 적용하는 방식이 필요합니다. 간단한 코드 수정 예시를 제공할 수 있습니다.
>
> 29s 동안 생각함

현재 `topaz11100/psd` 기준으로는 **CSV 저장은 병렬처리 안 됩니다.**  
`"num_workers": 0`은 DataLoader worker 수라서 **데이터 로딩만 영향** 있고, CSV 저장에는 영향이 없습니다. `psd_analysis.py`에서 loader를 만들 때 `num_workers=int(num_workers)`로 들어가는 구조입니다. `psd_analysis.py`

## 현재 CSV 저장 구조

현재 CSV 저장은 대략 이런 식입니다.

```python
for out_path_text, group_rows in sorted(groups.items()):
    out_path = Path(out_path_text)
    write_common_csv(out_path, group_rows)
    manifest_rows.append(...)
```

즉 파일 그룹을 만든 뒤, 각 CSV를 **하나씩 순차 저장** 합니다. 이 구조는 `_write_artifact()`와 `_write_rows_to_dir()` 둘 다 동일합니다. `psd_analysis.py`

그리고 `write_common_csv()` 자체도 단일 프로세스에서 파일을 열고, header를 쓰고, row를 하나씩 `writer.writerow()`로 저장합니다. `csv_schema.py`

그래서 현재는:

```json
"num_workers": 4
```

로 바꿔도 **CSV 저장 속도는 안 빨라집니다.**

---

## 병렬처리는 가능함

가능합니다. 다만 안전하게 하려면 **row 단위 병렬화가 아니라 CSV 파일 단위 병렬화** 로 해야 합니다.

안전한 구조는 이겁니다.

```text
1. rows를 output CSV path별로 group
2. 각 CSV 파일을 서로 다른 worker/thread가 write_common_csv로 저장
3. manifest_rows는 worker 안에서 직접 append하지 않음
4. 저장 결과 path 목록을 main thread에서 회수
5. main thread에서 manifest_rows를 순서대로 append
```

즉, 병렬화 대상은 이 부분입니다.

```python
write_common_csv(out_path, group_rows)
```

단, 이건 **서로 다른 파일에 쓸 때만 안전** 합니다. 같은 CSV 파일을 여러 worker가 동시에 append하는 식은 하면 안 됩니다.

---

## 추천 방식

이 작업은 대부분 **파일 I/O bound** 이기 때문에 `multiprocessing`보다 `ThreadPoolExecutor`가 더 적절합니다. 예:

```python
from concurrent.futures import ThreadPoolExecutor
```

그리고 worker 수는 처음부터 크게 주지 말고:

```json
"csv_num_workers": 4
```

정도만 추천합니다.

너무 크게 주면 오히려 느려질 수 있습니다. 특히 지금 `psd_analysis`는 작은 CSV 파일을 많이 만드는 구조라서, worker를 과하게 늘리면 파일 생성/metadata flush 비용 때문에 디스크가 병목이 됩니다.

---

## 결론

- 현재 `psd_analysis`의 CSV 저장은 **직렬 저장** 입니다.
- `"num_workers"`는 **CSV 저장 병렬화 옵션이 아님**.
- CSV 저장 병렬화는 가능하지만, **파일 단위 병렬 저장 + manifest는 main thread에서 회수** 하는 방식으로 해야 안전합니다.
- 속도 개선 효과는 CSV 파일 수가 많고 디스크가 버티는 경우에만 의미가 있습니다. GPU forward/PSD 계산이 병목이면 체감은 작을 수 있습니다.

## Prompt:
2026. 5. 27. 오후 6:12:13

if wait "$PID"; then
        STATUS=0
    else
        STATUS="$?"
    fi
    echo "STATUS=${STATUS}"
    if [ "$STATUS" -ne 0 ]; then
        echo "FAILED_CONFIG=${CONFIG_PATH}"
        exit "$STATUS"
    fi

psd_analysis.sh 에서 저거 없애면 병렬이지?

## Response:
2026. 5. 27. 오후 6:12:25

> **선택적 병렬 처리에 대한 설명**
>
> 사용자가 "psd_analysis.sh에서 wait을 없애면 병렬인가?"라고 묻는다면, 이건 명령에 "&"가 포함된 경우와 루프에서 기다리는 방식에 따라 달라집니다. "wait"은 PID를 기다리므로, 루프에서 즉시 실행 후 기다리지 않으면 병렬 실행이 가능하지만, 실패 감지 및 VRAM 부족 같은 문제가 발생할 수 있습니다.
>
> **병렬 처리 및 GPU 제한 고려한 코드 제안**
>
> 명령이 백그라운드에서 실행될 때, 즉 `command &`로 시작하고, 루프에서 기다리는 구문을 제거하면 병렬로 실행됩니다. 하지만 배치가 끝날 때를 위한 최종 `wait`이 필요하며, GPU 인덱스에 따라 여러 작업이 같은 GPU에서 실행되면 충돌할 수 있습니다.
>
> 11s 동안 생각함

조건부로 **맞습니다.**  

현재 구조가 대략 아래라면:

```bash
python src/psd_analysis.py --config "$CONFIG_PATH" &
PID=$!

if wait "$PID"; then
    STATUS=0
else
    STATUS="$?"
fi
...
```

이 `wait "$PID"`가 **각 config 실행 직후에 바로 기다리는 역할** 을 하므로, 사실상 순차 실행입니다.

즉:

```text
config 1 실행 -> 끝날 때까지 wait
config 2 실행 -> 끝날 때까지 wait
config 3 실행 -> 끝날 때까지 wait
```

## 그런데 그냥 저 블록만 없애면 문제 있음

`wait "$PID"` 블록을 없애면 loop가 다음 config로 바로 넘어가므로 병렬 실행이 되긴 합니다. 단, 아래 문제가 생깁니다.

1. **스크립트가 background job 완료를 기다리지 않을 수 있음**
2. **실패한 config를 감지하지 못함**
3. **GPU 하나에 여러 psd_analysis가 동시에 올라가서 VRAM 터질 수 있음**
4. **어떤 config가 실패했는지 추적이 어려움**

따라서 그냥 삭제하는 것보다 **즉시 wait는 없애고, 마지막에 전체 wait를 두는 방식** 이 맞습니다.

---

## 무제한 병렬 실행 형태

모든 config를 동시에 돌려도 되는 경우:

```bash
PIDS=()
PID_CONFIGS=()

for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
    echo "RUN_CONFIG=${CONFIG_PATH}"

    python src/psd_analysis.py --config "$CONFIG_PATH" &
    PID=$!

    PIDS+=("$PID")
    PID_CONFIGS+=("$CONFIG_PATH")
done

FAILED=0

for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    CONFIG_PATH="${PID_CONFIGS[$i]}"

    if wait "$PID"; then
        echo "DONE_CONFIG=${CONFIG_PATH}"
    else
        STATUS="$?"
        echo "FAILED_CONFIG=${CONFIG_PATH}"
        echo "STATUS=${STATUS}"
        FAILED="$STATUS"
    fi
done

exit "$FAILED"
```

이러면 실행은 병렬이고, 마지막에 전부 기다리면서 실패 여부도 확인합니다.

---

## 더 안전한 방식: 동시 실행 개수 제한

PSD analysis는 GPU/VRAM을 쓰므로 보통 이쪽이 더 안전합니다.

예를 들어 동시에 2개만 실행:

```bash
MAX_JOBS=2
PIDS=()
PID_CONFIGS=()

wait_one() {
    local PID="$1"
    local CONFIG_PATH="$2"

    if wait "$PID"; then
        echo "DONE_CONFIG=${CONFIG_PATH}"
        return 0
    else
        local STATUS="$?"
        echo "FAILED_CONFIG=${CONFIG_PATH}"
        echo "STATUS=${STATUS}"
        return "$STATUS"
    fi
}

for CONFIG_PATH in "${CONFIG_PATHS[@]}"; do
    echo "RUN_CONFIG=${CONFIG_PATH}"

    python src/psd_analysis.py --config "$CONFIG_PATH" &
    PID=$!

    PIDS+=("$PID")
    PID_CONFIGS+=("$CONFIG_PATH")

    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 2
    done
done

FAILED=0

for i in "${!PIDS[@]}"; do
    PID="${PIDS[$i]}"
    CONFIG_PATH="${PID_CONFIGS[$i]}"

    if ! wait_one "$PID" "$CONFIG_PATH"; then
        FAILED=1
    fi
done

exit "$FAILED"
```

---

## 결론

질문한 블록은 **병렬 실행을 막는 wait 블록** 이 맞습니다.

하지만 그냥 제거만 하면 안 되고, 구조는 이렇게 바꾸는 게 맞습니다.

```text
각 config 실행 직후 wait 제거
대신 PID 목록 저장
마지막에 전체 wait
필요하면 MAX_JOBS로 동시 실행 수 제한
```

특히 `psd_analysis`는 GPU를 쓰므로, 같은 GPU에 여러 개를 동시에 올릴 거면 `MAX_JOBS=1` 또는 GPU별 config 분리가 안전합니다.



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)
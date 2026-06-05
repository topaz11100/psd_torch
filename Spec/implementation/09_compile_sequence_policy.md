# Sequence-level compile policy

이 프로젝트의 `torch.compile` 적용 단위는 **단일 timestep**이 아니라 **layer sequence region**이다. SNN의 timestep loop는 호출 횟수가 많고 per-step 연산이 작기 때문에 `_step_impl` 하나만 compile하면 Python loop와 compiled callable 호출 오버헤드가 남는다. 따라서 IF/LIF/RF/TC-LIF/TS-LIF/DH-SNN/CNN2D/SpikGRU는 sequence 전체를 처리하는 함수로 분리하고, 그 함수만 regional compile한다.

## 고정 정책

- 공개 compile on/off key는 `compile: true|false`이다. CPU thread 수는 `compile_cpu_threads`로 지정할 수 있다.
- CUDA compile kwargs는 `backend=inductor`, `fullgraph=true`, `dynamic=false`를 사용한다.
- CPU는 codegen 지연을 피하기 위해 `backend=eager`를 사용한다.
- DDP에서 compile CPU thread 기본값은 rank당 2이다. `compile_cpu_threads` 인자 또는 `PSD_TORCH_COMPILE_CPU_THREADS` 환경변수로 override할 수 있다. 명시값은 자동 8-thread cap에 의해 잘리지 않는다.
- DDP/NCCL collective timeout은 `ddp_timeout_minutes`로 설정하며 기본값은 120분이다. 긴 첫 compile 또는 eval compile 동안 rank 간 대기 시간이 길어져도 watchdog timeout으로 중단되지 않게 한다.
- 프로젝트 모델에 `enable_compiled_forward()` hook이 있으면 top-level model compile fallback을 사용하지 않는다.
- compile hook이 sequence region을 설치하지 못하면 eager sequence path로 fallback한다.
- `sequence_backend` metadata 값은 `compiled_sequence_prealloc`이다.
- `sequence_buffer_mode` metadata 값은 `prealloc`이다.

## 구현 계약

각 sequence-capable layer는 다음 구조를 따른다.

```text
forward(input_sequence)
  1. time-independent projection을 loop 밖에서 계산한다.
  2. recurrent state 초기값과 per-layer scalar/vector parameter view를 만든다.
  3. _run_sequence(...)를 한 번 호출한다.
  4. return_traces=True일 때만 membrane/layer_input trace를 반환한다.
```

compile 대상 함수는 다음 조건을 만족해야 한다.

- Python list append를 사용하지 않는다.
- output tensor는 loop 진입 전에 `new_empty(...)`로 preallocate한다.
- loop 내부에는 tensor 연산과 static Python flag branch만 둔다.
- module state를 loop 안에서 mutate하지 않는다.
- runtime compile 실패 시 `_sequence_compiled_runtime_disabled`를 설정하고 eager sequence function으로 fallback한다.

## fast path와 trace path

일반 학습/추론 hidden layer는 `return_traces=False`를 사용한다. 이 경로는 spike sequence만 preallocate한다. PSD regularization, hidden capture, output layer trace는 `return_traces=True`를 사용하며 membrane/input trace도 함께 preallocate한다.

## 제외 대상

D-RF는 origin `BiRFModel.forward` 계약을 보존한다. 별도의 project-owned timestep loop가 없으므로 이 sequence compile 정책의 필수 대상이 아니다.

## DDP evaluation contract

DDP evaluation은 rank0-only로 수행하지 않는다. 모든 rank가 test split의 strided subset을 평가한다. subset은 padding 없이 `rank, rank + world_size, ...` index를 사용하므로 중복 sample을 만들지 않는다. 각 rank의 local `EpochMetrics`는 `(loss * total, correct, total)` 형태로 `all_reduce(SUM)`한 뒤 global loss/accuracy로 복원한다.

Evaluation DataLoader의 nominal `batch_size`는 training per-rank batch와 동일하게 둔다. 마지막 partial batch는 정확한 metric 계산을 위해 drop하지 않는다. rank별 evaluation forward는 DDP wrapper를 벗긴 local module로 실행하여 rank별 subset 길이 차이 때문에 per-forward buffer sync collective가 발생하지 않도록 한다. checkpoint metadata에는 `eval_batch_size`, `eval_dataset_policy`, `ddp_eval_policy`, `ddp_timeout_minutes`를 기록한다.

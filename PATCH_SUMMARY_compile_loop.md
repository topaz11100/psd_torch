# Compile/precision cleanup patch

## 변경 요약

- timestep step-region compile을 제거하고 layer sequence 전체를 regional compile 단위로 전환했다.
- sequence output buffer 정책은 항상 `prealloc`으로 고정하고, preallocated tensor write를 compiled sequence 함수 내부로 이동했다.
- 공개 AMP 인자는 `--amp off` / `--amp on` 두 개만 허용한다. `on`은 내부적으로 `bf16_safe` 정책으로만 실행한다.
- checkpoint 및 startup metadata에는 `sequence_backend=compiled_sequence_prealloc`, `sequence_buffer_mode=prealloc`, `amp=off|on`, `amp_internal_policy=off|bf16_safe`, `amp_active=off|bf16_safe`를 기록한다.
- DH-SNN branch mask sparsity 검증 테스트를 추가했다. branch 수가 늘면 각 dendritic branch의 active connection density가 `1/branch`가 되는지 확인한다.
- SpikeGRU는 single-step compile 대신 block-level sequence compile 경로를 사용한다.
- 결과 저장 entrypoint는 기본적으로 `timestamped_output=true`이며, 실제 산출물을 `run_<timestamp>` 또는 `<stage>_<timestamp>` 폴더 아래에 저장한다.

## 실행 형태

```bash
python src/model_training.py --config config/model_training.yaml --amp off
python src/model_training.py --config config/model_training.yaml --amp on
```

`--amp on`은 CUDA BF16 지원 장치에서만 활성화되며, 지원되지 않으면 `amp_active=off`로 기록된다.


## 검증

```text
python -m py_compile 핵심 compile/neuron/model/output files: 통과
focused regression tests: 통과
```

## 점검 메모

- DH-SNN은 origin layer의 `mask`를 forward graph에서 직접 곱해 masked weight의 gradient도 차단한다. Dense/RNN 모두 branch별 active connection density가 `1/branch`가 되도록 테스트했다.
- SpikeGRU는 2-layer 128 hidden, single update gate, max-over-time readout trace 계약을 유지한다.
- Spikformer는 checked-in author source를 우선 사용하고, optional dependency가 없을 때만 최소 torch fallback stub으로 import/build 가능성을 보장한다.

## 추가 DDP 평가/컴파일 스레드 패치

- DDP evaluation을 rank0-only에서 all-rank strided-subset 평가로 변경했다. 각 rank는 padding 없는 test subset을 평가하고 `(loss*total, correct, total)`을 all-reduce하여 global metric을 복원한다.
- DDP evaluation DataLoader의 nominal batch size를 train per-rank batch와 동일하게 맞췄다.
- DDP compile CPU thread 기본값을 rank당 2로 설정했다. `--compile_cpu_threads` 또는 config key `compile_cpu_threads`로 override할 수 있고, `PSD_TORCH_COMPILE_CPU_THREADS` 환경변수도 지원한다.

- DDP/NCCL timeout을 `--ddp_timeout_minutes` / config key `ddp_timeout_minutes`로 지정할 수 있게 했고, 기본값은 120분이다.


## First-spike / regularizer compile policy patch

- `first_spike` readout은 origin module loading 계약을 유지하되, 기본 runtime을 compile-friendly tensor adapter로 전환했다.
- first-spike analyze/train-loss/eval-loss는 `enable_compiled_forward()`에서 model compile kwargs와 동일하게 별도 compile할 수 있다.
- origin custom autograd의 dynamic boolean indexing은 straight-through tensor surrogate로 대체했다. forward first-time semantics와 Gaussian surrogate gradient assignment는 유지한다.
- 모든 signal/PSD regularizer는 `torch.compiler.disable(recursive=True)` eager-GPU boundary 뒤에서 실행한다. regularizer는 compile graph에 섞이지 않으며, GPU tensor autograd는 유지한다.
- PCA reference bank는 epoch loop 전에 active device로 1회 이동하여 minibatch마다 CPU -> GPU transfer가 발생하지 않게 했다.
- checkpoint/metadata에는 readout compile policy와 `regularizer_backend=eager_gpu` 계열 정보를 기록한다.

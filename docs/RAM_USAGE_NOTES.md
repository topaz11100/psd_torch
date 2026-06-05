# System RAM Usage Notes

This project is designed to keep prepared split payloads lazy by using NumPy memmap-backed structured `.npy` files, but several runtime stages can still consume significant host RAM.

## Major host-RAM consumers

1. **DDP PCA reference-bank construction**
   - Triggered only when `lambda_psd_pca_input != 0` or `lambda_psd_pca_adjacent != 0`.
   - Each rank runs one `capture_hidden=True` reference forward, moves the input and hidden traces to CPU, then uses `torch.distributed.all_gather_object`.
   - `all_gather_object` serializes/pickles tensors, so peak RAM is larger than the raw tensor size.
   - Rank 0 then concatenates per-rank tensors to build the global PCA bank. With both `input` and `adjacent` PCA relations enabled, only one gathered payload is collected, but two relation-specific PCA banks are fitted from that payload.
   - Approximate payload size per rank is:

     ```text
     input_bytes + sum_hidden_layers(spike_bytes) + optional sum_hidden_layers(membrane_bytes)
     ```

     For `output_family="spike"`, membrane is not included in the gathered payload. For `output_family="membrane"`, both spike and membrane can be materialized.

2. **DataLoader worker/prefetch memory**
   - Each train/eval process has its own DataLoader.
   - DDP multiplies this by `world_size`.
   - With `num_workers > 0`, each worker can prefetch batches. Effective queued batch memory is roughly:

     ```text
     world_size * num_workers * prefetch_factor * batch_bytes
     ```

   - `pin_memory=true` also keeps pinned host buffers for CUDA transfers.
   - The project includes a `/dev/shm`-aware worker policy, but this protects shared memory crashes more than total RSS.

3. **torch.compile / Inductor process memory**
   - The first compiled forward can allocate substantial Python/CPU memory while Dynamo/Inductor traces, lowers, optimizes, and compiles kernels.
   - This is independent of CUDA VRAM. Even when CUDA memory is stable, host RAM can rise during compilation.
   - Large SNN sequence graphs and multiple parallel DDP jobs can multiply this cost.

4. **Prepared dataset metadata copies**
   - The main split tensors are memmap-backed, but labels and sample indices are copied into memory for indexing:

     ```python
     self.labels = torch.as_tensor(np.array(records['label'], copy=True))
     self.sample_indices = [int(v) for v in sample_index_array.reshape(-1).tolist()]
     ```

   - This is usually small relative to input traces, but can matter for very large datasets.

5. **Analysis/DI outputs**
   - `psd_analysis`, `dataset_psd`, and `DI.py` may accumulate curves, per-layer summaries, CSV rows, and plots in memory before writing.
   - Large all-epoch/all-layer analysis runs can therefore use more RAM than a single training step.

## Quick diagnosis commands

Per process RSS:

```bash
ps -o pid,ppid,stat,etime,rss,vsz,pcpu,pmem,args -p <PID>
```

DDP rank/environment view:

```bash
for p in <PID0> <PID1>; do
  echo "===== $p ====="
  tr '\0' '\n' < /proc/$p/environ | egrep 'RANK|LOCAL_RANK|WORLD_SIZE|CUDA_VISIBLE|TORCHINDUCTOR|OMP|MKL'
  ps -o pid,rss,vsz,pcpu,pmem,args -p "$p"
done
```

Top RAM consumers under the current user:

```bash
ps -u "$USER" -o pid,ppid,rss,vsz,pcpu,pmem,args --sort=-rss | head -30
```

Per-rank DataLoader policy appears in logs through `loader_runtime_policy` when emitted by entrypoints. If RSS is high and PCA regularization is off, first check `num_workers`, `prefetch_factor`, `pin_memory`, and parallel job count.

## Practical reductions

- For debugging, set `num_workers: 0`.
- Avoid launching multiple 2-rank DDP jobs on the same two GPUs when `compile=true`.
- Keep `compile_cpu_threads` small, e.g. `2`.
- For PCA-regularized experiments, start with smaller batch size or `pca_dim_per_layer: [1]` before increasing to MIMO.
- Prefer `psd_reg_output_family: "spike"` over `"membrane"` when the PCA reference bank does not need membrane traces.

# Acceptance audit

최종 검증은 다음을 확인한다.

- `src/psd_snn` 기반 CLI와 tests 통과.
- raw trace CSV 없음.
- exact/userbin 직접 distance 차단.
- PCA distance는 같은 basis id만 허용.
- 2D FFT artifact는 `spectral_matrix_2d`.
- `label_single`은 top-level probe family.
- archive/reference 디렉터리는 current runtime으로 안내하지 않음.

권장 명령:

```bash
PYTHONPATH=src pytest -q tests
find examples/bash -name "*.sh" -print0 | xargs -0 -I{} bash -n {}
```

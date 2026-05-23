# 구현 구조

현재 공식 구현은 `src/psd_snn` 패키지다. root-level 과거 entrypoint는 현재 실행 계약이 아니다.

## 책임 분리

- `config`: dataclass 설정과 validation.
- `models`: MLP, fixed topology, cell, readout, checkpoint metadata.
- `analysis`: probe, trace, signal map, PSD/PCA/FFT2D, distance.
- `artifacts`: CSV writer, trace writer, reader, plotting.
- `cli`: 사용자 실행 entrypoint.
- `training`: 최소 synthetic training/checkpoint smoke.

## 공식 CLI

```text
psd_snn.cli.train
psd_snn.cli.analyze_signal
psd_snn.cli.analyze_fft2d
psd_snn.cli.analyze_dynamics
psd_snn.cli.plot_artifacts
```

Archive 디렉터리는 구현 참고 자료일 뿐 current runtime layer가 아니다.

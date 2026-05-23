# CLI contract

공식 CLI는 package module로 실행한다.

```bash
PYTHONPATH=src python -m psd_snn.cli.train --help
PYTHONPATH=src python -m psd_snn.cli.analyze_signal --help
PYTHONPATH=src python -m psd_snn.cli.analyze_fft2d --help
PYTHONPATH=src python -m psd_snn.cli.analyze_dynamics --help
PYTHONPATH=src python -m psd_snn.cli.plot_artifacts --help
```

`analyze_signal`은 PSD/PCA/element PSD를 다룬다. `analyze_fft2d`는 독립 2D FFT 분석이다. `plot_artifacts`는 artifact reader 기반 rendering만 수행한다.

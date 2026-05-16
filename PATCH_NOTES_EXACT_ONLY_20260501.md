# Exact-only PSD patch - 2026-05-01

## 목적

`psd_analysis.py` 와 `dataset_psd.py` 의 official 실행 경로에서 userbin 분석, 저장, 산출을 비활성화하고 `psd_exact` 만 산출한다.

## 코드 정책

- `ALL_CURVE_EXTRACTORS = ('psd_exact',)` 유지
- `compute_family_spectral_summary(..., include_userbin=False)` 호출 유지
- `dataset_psd.py` streaming summary 는 `psd_exact` 만 누적
- `--userbin_edges` 는 Python parser 호환성 인수로만 남기고 실행 경로에서는 무시
- `bash/psd_analysis.sh`, `bash/dataset_psd.sh` 는 `USERBIN_EDGES` 를 Python 에 전달하지 않음

## 영향

생성되지 않는 산출물:

```text
extractor=psd_userbin
userbin_edges / userbin_centers 기반 CSV
userbin 기반 analysis_curve / analysis_dispersion / pair_distance / drift_distance
```

생성되는 산출물:

```text
extractor=psd_exact
raw / db scale
mean / median representative curve
variance / MAD dispersion
```

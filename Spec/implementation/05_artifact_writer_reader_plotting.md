# Artifact writer, reader, plotting

CSV row는 `src/util/csv_schema.py`의 category별 column을 따른다. 동적 matrix column은 `write_common_csv(..., extra_columns=...)`로 추가한다.

각 분석 stage는 manifest CSV를 작성한다.

- dataset PSD: `dataset_psd_manifest.csv`
- dataset FFT: `dataset_fft_manifest.csv`
- model 분석: `analysis_manifest.csv`
- plotting: `recursive_plot_manifest.csv` 또는 설정한 manifest 이름

Plotting은 CSV를 읽어 PNG만 생성한다. 학습이나 분석 계산을 다시 수행하지 않는다.

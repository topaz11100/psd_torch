# Artifact writer, reader, plotting

CSV는 summary와 manifest 용도다. raw trace 값은 CSV로 쓰지 않는다. trace tensor는 `.pt` chunk로 저장한다.

주요 파일은 `spectral_curve.csv`, `spectral_matrix_1d.csv`, `spectral_matrix_2d.csv`, `pca_basis.csv`, `spectral_distance.csv`, `trace_manifest.csv`, `analysis_manifest.csv`다.

ArtifactReader와 plotting은 이 파일들을 읽어 그림을 만든다.

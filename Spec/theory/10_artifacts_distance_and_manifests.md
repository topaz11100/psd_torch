# Artifact, distance, manifest 이론

## 1. 목적

분석 결과는 재현 가능하고 비교 가능해야 한다. 이를 위해 value artifact와 manifest를 분리한다. raw trace 값은 CSV로 저장하지 않고 tensor chunk로 저장한다.

## 2. 주요 artifact type

- `spectral_curve`: PSD curve.
- `spectral_matrix_1d`: row-preserving PSD 또는 PCA component matrix.
- `spectral_matrix_2d`: 2D FFT matrix.
- `pca_basis`: PCA basis metadata.
- `spectral_distance`: compatible artifacts 사이의 거리.
- `trace_manifest`: trace tensor chunk inventory.
- `analysis_manifest`: analysis run output/status inventory.

## 3. trace tensor chunk

raw trace는

$$
B \times T \times *
$$

layout을 보존한다. chunking은 sample axis에서만 수행한다.

$$
B = B_1 + B_2 + \cdots + B_m
$$

각 chunk는 `.pt` artifact로 저장하고 manifest에는 path, shape, dtype, sample_start, sample_count, time_length를 남긴다.

## 4. distance identity

distance는 숫자 shape만 같다고 계산하지 않는다. 두 artifact의 identity가 compatible해야 한다.

공통 compatibility 조건은 다음이다.

```text
artifact_type
spectral_axis
scale
centering/window policy
representative or analysis_method
frequency/bin policy
layer/series identity
```

PCA는 추가로 같은 non-empty basis id가 필요하다. 2D matrix는 row_axis_semantics, shift policy, userbin axes, row/column bin identity가 같아야 한다.

## 5. 거리 수식

centered L2는

$$
d(A,B)=\| (A-\bar{A})-(B-\bar{B}) \|_2
$$

이다. matrix에서는 global matrix mean을 뺀 뒤 flatten한다.

diff L2는 지정한 frequency axis에서 first difference를 취한 뒤 L2를 계산한다.

$$
d_\Delta(A,B)=\|\Delta A-\Delta B\|_2
$$

## 6. manifest의 역할

`analysis_manifest`는 성공과 실패를 모두 기록한다. checkpoint load failure, unsupported topology, missing PCA basis, unavailable series, distance incompatibility 같은 상태는 silent skip이 아니라 status/reason으로 남아야 한다.

## 7. 금지 출력

- raw trace value CSV.
- PCA basis tensor를 CSV wide row로 저장하는 방식.
- 서로 다른 axis space artifact의 직접 distance.
- 이름만 같고 metadata가 다른 artifact 비교.

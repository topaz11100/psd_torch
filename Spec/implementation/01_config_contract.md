# 설정 contract

현재 설정은 dataclass 기반이다. 핵심 객체는 `ExperimentConfig`, `ModelSpec`, `TopologySpec`, `CellSpec`, `ReadoutSpec`, `ConstraintSpec`, `ProbeSpec`, `SignalAnalysisSpec`이다.

## validation 원칙

- `mlp_stack`은 hidden widths가 필요하다.
- fixed topology는 MLP cell replacement 대상이 아니다.
- canonical scenario는 `none`, `clip`, `structure`, `clipstructure`다.
- probe family는 `balanced_global`, `distributed_set`, `label_set`, `label_single`만 허용한다.
- spectral axis는 `exact` 또는 `userbin`이다.
- distance metric은 `centered_l2`, `diff_l2`만 허용한다.

잘못된 cell/bounds 조합은 validator에서 차단한다.

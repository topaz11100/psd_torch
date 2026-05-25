# CLI 계약

각 stage는 다음 방식으로 실행한다.

```bash
python src/model_training.py --config config/model_training.json
```

Bash wrapper는 같은 config를 기본값으로 사용한다.

```bash
bash/model_training.sh config/model_training.json
```

`--help`는 데이터 경로 검증과 heavy dependency import 없이 출력되어야 한다.

## PCA 관련 CLI/Config 계약 (현재 구현)

### `psd_analysis`
- `enable_pca_1d` (bool/string, 기본 `true`)
- `enable_pca_mimo` (bool/string, 기본 `true`)
- `pca_ref_epoch` (int, 기본 `1`, 양수, checkpoint 목록에 반드시 존재)
- `pca_min_train_accuracy` (float, 기본 `0.0`, \([0,1]\))
- `pca_dim_per_layer` (int list, 양의 정수, layer index 초과 시 tail broadcast)

### `model_training`
- 기존 `regularization_lambda1/2` 계약은 그대로 유지한다.
- PCA PSD 규제는 별도 옵션으로 추가한다:
  - `lambda_psd_rep_1d`
  - `lambda_psd_pca_1d`
  - `lambda_psd_pca_mimo`
  - `psd_reg_variant` (`raw|centered`)
  - `psd_reg_output_family` (`spike|membrane`)
  - `pca_dim_per_layer` (int list)

학습 손실은
\[
\mathcal{L}_{\text{total}}
=\mathcal{L}_{\text{task}}
+\mathcal{L}_{\text{legacy-reg}}
+\mathcal{L}_{\text{psd-reg}}
\]
이며, \(\mathcal{L}_{\text{legacy-reg}}\)는 기존 `lambda1/lambda2` 경로를 보존한다.

### DDP 정책

현재 PCA PSD penalty의 rank 간 reference basis 동기화(broadcast)는 미구현이다. 따라서 DDP에서 PCA penalty를 켜는 조합은 명시적 `ValueError`로 fail-fast 한다.

# Example Execution Contract

## Clean base 실행

```bash
python src/data_prep.py --config config/base/data_prep.clean.yaml
python src/model_training.py --config config/base/model_training.clean.yaml
python src/psd_analysis.py --config config/base/psd_analysis.clean.yaml
```

## DDP 학습

```bash
NPROC_PER_NODE=2 bash bash/model_training_ddp.sh config/base/model_training_ddp.clean.yaml
```

## Scenario 전체 분석

`config/ddp_train_scenario/<group>/<case>.yaml`으로 학습한 뒤, 같은 `<group>/<case>`의 분석 config를 실행한다.

```bash
python src/psd_analysis.py --config config/psd_analysis_scenario/00_soft_fixed/s-mnist_lif_soft_fixed.yaml
python src/2d_fft_analysis.py --config config/fft2d_analysis_scenario/00_soft_fixed/s-mnist_lif_soft_fixed.yaml
python src/element_psd.py --config config/element_psd_scenario/00_soft_fixed/s-mnist_lif_soft_fixed.yaml
python src/element_fft.py --config config/element_fft_scenario/00_soft_fixed/s-mnist_lif_soft_fixed.yaml
python src/dataset_fft.py --config config/dataset_fft_scenario/00_soft_fixed/s-mnist_lif_soft_fixed.yaml
```

## 2D bash 그룹

```bash
# group0의 두 config를 병렬 실행하고, 모두 성공하면 group1 실행
# 스크립트 내부 CONFIG_GROUP_* 배열을 편집한다.
bash bash/psd_analysis.sh
```

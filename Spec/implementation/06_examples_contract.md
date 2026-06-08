# Example Execution Contract

## Clean template 실행 방식

`config/`의 YAML은 모든 인자를 공란으로 둔 template다. 실행 시 필요한 값은 YAML에 채우거나 CLI로 넘긴다.

```bash
bash bash/data_prep.sh config/data_prep.yaml   --dataset mnist   --raw_data_root /home/yongokhan/workspace/data/raw_data   --prep_root /home/yongokhan/workspace/data/prep_data
```

```bash
bash bash/model_training.sh config/model_training.yaml   --dataset mnist   --prep_root /home/yongokhan/workspace/data/prep_data   --neuron_type my_dh_snn   --branch 8   --reset soft   --hidden_spec 256,128   --readout_mode temporal_membrane   --v_th train 1.0   --epochs 100   --batch_size 128   --lr 0.001   --seed 0   --checkpoint_root /home/yongokhan/workspace/result/checkpoints   --metric_root /home/yongokhan/workspace/result/metrics
```

## DDP 학습

```bash
bash bash/model_training.sh config/model_training.yaml   --dataset mnist   --prep_root /home/yongokhan/workspace/data/prep_data   --neuron_type my_d_rf   --branch 8   --reset hard   --hidden_spec 256,128   --readout_mode temporal_membrane   --v_th train 1.0   --epochs 100   --batch_size 256   --lr 0.001   --seed 0   --gpu_index 0 1   --checkpoint_root /home/yongokhan/workspace/result/checkpoints   --metric_root /home/yongokhan/workspace/result/metrics
```

`--gpu_index 0 1`은 physical cuda:0,1을 선택하고 자동으로 `torchrun` DDP child process를 띄운다.

## Checkpoint 분석

```bash
bash bash/psd_analysis.sh config/psd_analysis.yaml   --checkpoint /home/yongokhan/workspace/result/checkpoints/epoch0100.pt   --dataset mnist   --prep_root /home/yongokhan/workspace/data/prep_data   --output_root /home/yongokhan/workspace/result/psd_analysis   --anal_batch 128   --gpu_index 0
```

## `my_*` branch regularization 예

```bash
bash bash/model_training.sh config/model_training.yaml   --dataset shd   --prep_root /home/yongokhan/workspace/data/prep_data   --neuron_type my_r_dh_snn   --branch 8   --reset soft   --hidden_spec 512,256   --readout_mode temporal_membrane   --v_th train 1.0   --lambda_branch_ortho 0.001   --lambda_branch_s 0.0001   --soft_mask_epochs 80   --ste_epochs 5   --harden_epoch 90   --epochs 120   --batch_size 64   --lr 0.001   --seed 0   --checkpoint_root /home/yongokhan/workspace/result/checkpoints   --metric_root /home/yongokhan/workspace/result/metrics
```

## Vanilla clip/structure 예

`clip`, `structure`, `clipstructure`는 vanilla `if/lif/rf` dense family에만 사용한다.

```bash
bash bash/model_training.sh config/model_training.yaml   --dataset mnist   --prep_root /home/yongokhan/workspace/data/prep_data   --neuron_type lif   --hidden_spec 256,128   --readout_mode temporal_membrane   --v_th fixed 1.0   --scenario_mode clipstructure   --alpha_clip_edges 0.2 0.98   --epochs 100   --batch_size 128   --lr 0.001   --seed 0   --checkpoint_root /home/yongokhan/workspace/result/checkpoints   --metric_root /home/yongokhan/workspace/result/metrics
```

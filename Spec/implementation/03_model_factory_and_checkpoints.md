# Model factory와 checkpoint

`src/model/model_registry.py`가 model token을 canonical spec으로 바꾼다. `src/model/snn_builder.py`가 모델을 생성한다.

Checkpoint는 `.pt` 파일이며 최소 다음 metadata를 포함해야 한다.

- model token/config
- readout config
- dataset token
- prepared data reference
- axis metadata reference
- seed
- epoch
- state_dict

분석 entrypoint는 checkpoint metadata와 CLI/config dataset 값이 일치하는지 확인한다.

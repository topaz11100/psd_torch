# Model factory와 checkpoint

`build_model`은 topology kind를 기준으로 MLP와 fixed topology를 분리한다. MLP는 `CellSpec`을 사용하고 fixed topology는 독립 factory를 사용한다.

Checkpoint payload는 model object pickle이 아니라 state dict와 metadata 중심이다. metadata에는 topology, cell, readout, constraint, checkpoint epoch, hash가 포함된다.

Restore 실패나 unsupported topology는 status/reason으로 표현되어야 한다.

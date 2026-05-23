# Model Factory and Checkpoints

## Model factory

`src/psd_snn/models/factory.py` dispatches model creation.

- `mlp_stack`: MLP builder with IF/LIF/RF cell selection.
- fixed topology kinds: fixed factory path.

## Checkpoint payload

Checkpoint payloads contain:

- `state_dict`
- sanitized config or model metadata
- checkpoint epoch
- run/checkpoint metadata where available

Model object pickle storage is not the current contract.

## Restore policy

Restore loads the state dictionary, rebuilds the model from metadata, loads weights, applies evaluation mode, and moves the model to the requested device.

Unsupported topology is a status/reason outcome, not a silent skip.

## Hashes

`config_hash` and `constraint_hash` are permitted deterministic identifiers. Current artifacts do not require a schema key.

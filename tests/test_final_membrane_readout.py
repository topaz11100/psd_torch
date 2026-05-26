import pytest
import torch

from src.readout.readout import FinalMembraneReadout, build_readout


def test_final_membrane_readout_uses_last_timestep_logits() -> None:
    readout = build_readout('final_membrane', num_classes=3, sequence_length=4, device='cpu')

    assert isinstance(readout, FinalMembraneReadout)
    assert readout.output_layer_overrides() == {'emit_spike': False, 'reset_enabled': False}

    output_membrane = torch.tensor(
        [
            [
                [1.0, 0.0, -1.0],
                [0.0, 2.0, -2.0],
                [0.1, 0.2, 3.0],
            ],
            [
                [0.0, 1.0, 2.0],
                [1.0, 3.0, 0.0],
                [4.0, 0.0, 1.0],
            ],
        ],
        dtype=torch.float32,
    )
    output_spike = torch.ones_like(output_membrane)

    analysis = readout.analyze_output_record(output_membrane, output_spike)

    torch.testing.assert_close(analysis.scores, output_membrane[:, -1, :])
    assert readout.predictions_from_analysis(analysis).tolist() == [2, 0]
    loss = readout.loss_from_analysis(analysis, torch.tensor([2, 0]), training=True)
    assert torch.isfinite(loss)


def test_final_membrane_readout_rejects_invalid_output_shapes() -> None:
    readout = build_readout('final_membrane', num_classes=3, sequence_length=4, device='cpu')

    with pytest.raises(ValueError, match='final_membrane requires output membrane shape'):
        readout.analyze_output_record(torch.zeros(2, 3), torch.zeros(2, 3))

    with pytest.raises(ValueError, match='final_membrane requires at least one timestep'):
        readout.analyze_output_record(torch.zeros(2, 0, 3), torch.zeros(2, 0, 3))

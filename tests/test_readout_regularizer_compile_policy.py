import torch

from src.model.first_spike_loss import FirstSpikeLossAdapter
from src.model.psd_minibatch_regularizer import FixedPCALayerReference, move_fixed_pca_reference_bank_to_device
from src.model.training import regularizer_compile_metadata
from src.readout.readout import build_readout


def test_first_spike_tensor_readout_compiles_with_eager_backend_and_backprops():
    readout = build_readout('first_spike', num_classes=3, sequence_length=8, device='cpu')
    applied, policy = readout.enable_compiled_forward(backend='eager', fullgraph=True, dynamic=False)
    assert applied
    assert 'first_spike' in policy
    membrane = torch.randn(2, 8, 3, requires_grad=True)
    spike = torch.rand(2, 8, 3, requires_grad=True)
    analysis = readout.analyze_output_record(membrane, spike)
    loss = readout.loss_from_analysis(analysis, torch.tensor([0, 2]), training=True)
    loss.backward()
    assert loss.ndim == 0
    # Released first-spike surrogate assigns gradients to spike traces, not membrane potentials.
    assert spike.grad is not None
    assert torch.isfinite(spike.grad).all()
    meta = readout.compile_metadata()
    assert meta['readout_compile_applied'] is True
    assert meta['readout_backend'] == 'compiled_friendly_tensor_first_spike'


def test_first_spike_origin_objects_remain_available_for_reference_contract():
    adapter = FirstSpikeLossAdapter(num_classes=3, sequence_length=6, device='cpu')
    assert adapter.train_loss.__class__.__module__ == 'origin_first_spike_loss'
    assert adapter.eval_loss.__class__.__module__ == 'origin_first_spike_loss'


def test_regularizer_policy_is_eager_gpu_and_compiler_disabled():
    meta = regularizer_compile_metadata()
    assert meta['regularizer_backend'] == 'eager_gpu'
    assert meta['regularizer_compile_applied'] is False
    assert 'disable' in meta['regularizer_compile_policy'] or meta['regularizer_compile_policy'] == 'eager_gpu_no_disable_api'
    assert meta['regularizer_gradient_policy'] == 'autograd_preserved_no_detach_no_cpu'


def test_pca_reference_bank_moves_to_target_device_once():
    bank = {
        'layer': FixedPCALayerReference(
            layer_name='layer',
            layer_index=0,
            dim=2,
            x_basis=torch.eye(3, 2),
            x_centroid=torch.zeros(3),
            y_basis=torch.eye(4, 2),
            y_centroid=torch.zeros(4),
            output_family='spike',
        )
    }
    moved = move_fixed_pca_reference_bank_to_device(bank, device=torch.device('cpu'), dtype=torch.float64)
    assert moved is not None
    ref = moved['layer']
    assert ref.x_basis.device.type == 'cpu'
    assert ref.x_basis.dtype == torch.float64
    assert ref.x_basis.requires_grad is False
    assert ref.metadata['device_resident'] == 'cpu'

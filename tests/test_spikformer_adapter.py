import torch

from src.model.author_adapter_spikformer import _adapt_to_cifar10dvs_frames


def test_spikformer_adapter_rank3_sequence_to_cifar10dvs_frames():
    x = torch.randn(2, 7, 32)
    frames = _adapt_to_cifar10dvs_frames(x)
    assert frames.shape == (2, 16, 2, 128, 128)
    assert frames.dtype == torch.float32
    assert frames.is_contiguous()


def test_spikformer_adapter_image_and_channel_policy():
    gray = torch.randn(2, 1, 32, 32)
    gray_frames = _adapt_to_cifar10dvs_frames(gray)
    assert gray_frames.shape == (2, 16, 2, 128, 128)
    assert torch.allclose(gray_frames[:, :, 0], gray_frames[:, :, 1])

    rgb = torch.randn(2, 6, 3, 32, 32)
    rgb_frames = _adapt_to_cifar10dvs_frames(rgb)
    assert rgb_frames.shape == (2, 16, 2, 128, 128)


def test_spikformer_author_classifier_builds_with_checked_in_source():
    from src.model.snn_builder import build_snn_classifier

    model = build_snn_classifier(
        model_token='spikformer',
        input_dim=32,
        sequence_length=7,
        num_classes=3,
        hidden_sizes=[8],
        v_th=1.0,
    )
    meta = model.model_metadata()
    assert meta['model_profile'] == 'spikformer'
    assert meta['source_code_path'] == 'Origin/spikformer/cifar10dvs/model.py'
    assert meta['dependency_backend'] in {'author_dependencies', 'fallback_stubs'}
    out = model(torch.randn(1, 7, 32), capture_hidden=True)
    assert out.output_record.membrane.shape == (1, 1, 3)
    assert len(out.hidden_records) == 2

import pytest

torch = pytest.importorskip("torch")

from src.model.model_registry import canonicalize_model_token
from src.model.snnbench_adapter import build_snnbench_cnn_classifier
from src.model.snnbench_backbones import spiking_resnet18
from src.model.snn_builder import ForwardResult, LayerRecord, build_snn_classifier


CONTRACT = "topology_only_from_reference_SNNs; input geometry comes from prep_data"


def test_sew_resnet18_is_reference_input_frontend_free():
    model = spiking_resnet18(in_channels=2, num_classes=11)
    assert not hasattr(model, "reference_input_frontend")
    assert not hasattr(model, "maxpool")
    assert not hasattr(model, "conv1")
    assert model.input_channels == 2
    first_block = model.layer1[0]
    assert first_block.conv1.in_channels == 2
    assert first_block.downsample is not None


def test_resnet18_accepts_dvs_gesture_128_prepared_frames():
    model = build_snn_classifier(
        model_token="resnet18_lif_soft_fixed",
        input_dim=2 * 128 * 128,
        sequence_length=1,
        num_classes=11,
        input_shape=(1, 2, 128, 128),
        arch_spec="-",
    )
    assert type(model).__name__ == "FixedCNN2DClassifier"
    first_block = model.hidden_layers[0]
    assert first_block.input_size == 2
    assert first_block.output_size == 64

    small = build_snn_classifier(
        model_token="resnet18_lif_soft_fixed",
        input_dim=2 * 16 * 16,
        sequence_length=1,
        num_classes=11,
        input_shape=(1, 2, 16, 16),
        arch_spec="-",
    )
    with torch.no_grad():
        result = small(torch.randn(1, 1, 2, 16, 16), capture_hidden=False)
    assert result.output_record.membrane.shape == (1, 1, 11)

    md = model.model_metadata()
    assert md["reference_backbone_contract"] == CONTRACT
    assert md["resnet_input_projection"] == "none_first_basicblock_consumes_prepared_frame_channels"
    assert md["sew_resnet_connect_function"] == "ADD"
    assert md["spiking_backend"] == "project_native_torch"


def test_vgg11_accepts_project_image_channels_and_resolution():
    model = build_snn_classifier(
        model_token="vgg11_lif_soft_fixed",
        input_dim=2 * 128 * 128,
        sequence_length=1,
        num_classes=11,
        input_shape=(1, 2, 128, 128),
        arch_spec="-",
    )
    assert type(model).__name__ == "FixedCNN2DClassifier"

    small = build_snn_classifier(
        model_token="vgg11_lif_soft_fixed",
        input_dim=2 * 16 * 16,
        sequence_length=1,
        num_classes=11,
        input_shape=(1, 2, 16, 16),
        arch_spec="-",
    )
    with torch.no_grad():
        result = small(torch.randn(1, 1, 2, 16, 16), capture_hidden=False)
    assert result.output_record.spike.shape == (1, 1, 11)

    md = model.model_metadata()
    assert md["reference_backbone_contract"] == CONTRACT
    assert md["input_policy"] == "prepared_data_shape_driven_no_reference_input_frontend"


def test_snnbench_adapter_is_topology_only_compatibility_path():
    spec = canonicalize_model_token("resnet18_lif_soft_fixed")
    model = build_snnbench_cnn_classifier(
        spec=spec,
        input_dim=2 * 32 * 32,
        sequence_length=1,
        num_classes=5,
        input_shape=(1, 2, 32, 32),
        v_th=1.0,
        layer_record_cls=LayerRecord,
        forward_result_cls=ForwardResult,
    )
    with torch.no_grad():
        result = model(torch.randn(1, 1, 2, 32, 32), capture_hidden=False)
    assert result.output_record.membrane.shape == (1, 1, 5)
    md = model.model_metadata()
    assert md["reference_backbone_contract"] == CONTRACT
    assert md["resnet_input_projection"] == "none_first_basicblock_consumes_prepared_frame_channels"

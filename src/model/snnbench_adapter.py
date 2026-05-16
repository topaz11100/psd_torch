from __future__ import annotations

from typing import Any, Iterable, Sequence

import torch
import torch.nn as nn

from spikingjelly.activation_based import encoding, functional

from src.model.snnbench_backbones import (
    TDBatchNorm,
    VGG_SNN,
    make_snnbench_lif_node,
    make_snnbench_rf_node,
    spiking_resnet18,
)


class SNNBenchCNNAdapter(nn.Module):
    def __init__(
        self,
        *,
        spec,
        input_dim: int,
        sequence_length: int,
        input_shape: Sequence[int],
        num_classes: int,
        v_th: float,
        layer_record_cls,
        forward_result_cls,
    ):
        super().__init__()

        self.spec = spec
        self.input_dim = int(input_dim)
        self.sequence_length = int(sequence_length)
        self.input_shape = tuple(int(v) for v in input_shape)
        self.num_classes = int(num_classes)
        self.output_sequence_length = int(sequence_length)

        self.layer_record_cls = layer_record_cls
        self.forward_result_cls = forward_result_cls

        if len(self.input_shape) != 4:
            raise ValueError(f"snn-bench CNN adapter expects input_shape=(T,C,H,W), got {self.input_shape}")

        time_steps, in_channels, height, width = self.input_shape

        self.use_poisson_encoder = int(in_channels) == 3 and int(height) == 32 and int(width) == 32
        self.encoder = encoding.PoissonEncoder()

        self.output_sn = None

        if spec.family == "cnn_lif":
            self.neuron_tag = "lif"
            node_factory = make_snnbench_lif_node
        elif spec.family == "cnn_rf":
            self.neuron_tag = "rf"
            node_factory = make_snnbench_rf_node
        else:
            raise ValueError(f"snn-bench CNN adapter supports cnn_lif/cnn_rf, got {spec.family!r}")

        node_kwargs = {
            "v_threshold": float(v_th),
            "reset_mode": spec.reset_mode,
            "trainable_threshold": bool(getattr(spec, "trainable_threshold", False)),
        }

        if spec.backbone == "vgg11":
            self.core = VGG_SNN(
                in_channels=int(in_channels),
                num_classes=int(num_classes),
                depth=11,
                bias=True,
                single_step_neuron=node_factory,
                **node_kwargs,
            )

        elif spec.backbone == "resnet18":
            self.core = spiking_resnet18(
                pretrained=False,
                progress=True,
                norm_layer=TDBatchNorm,
                single_step_neuron=node_factory,
                num_classes=int(num_classes),
                **node_kwargs,
            )

            self.output_sn = node_factory(
                channels=int(num_classes),
                **node_kwargs,
            )

            # snn-bench 원본 ResNet은 CIFAR 전용이라 conv1 입력 채널이 3으로 고정되어 있다.
            # DVS128Gesture, N-MNIST류를 위해 여기서만 얇게 교체한다.
            if int(in_channels) != 3:
                self.core.conv1 = nn.Conv2d(
                    int(in_channels),
                    64,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
                nn.init.kaiming_normal_(self.core.conv1.weight, mode="fan_out", nonlinearity="relu")

        else:
            raise ValueError(f"Unsupported snn-bench backbone: {spec.backbone!r}")

    def reset_state(self) -> None:
        functional.reset_net(self.core)
        if self.output_sn is not None:
            functional.reset_net(self.output_sn)

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(x)

        if tensor.ndim != 5:
            raise ValueError(f"snn-bench CNN expects input shape (B,T,C,H,W), got {tuple(tensor.shape)}")

        expected = self.input_shape
        actual = tuple(int(v) for v in tensor.shape[1:])

        if actual != expected:
            raise ValueError(f"snn-bench CNN expected input_shape={expected}, got {actual}")

        return tensor.contiguous()

    def _maybe_encode_frame(self, frame: torch.Tensor) -> torch.Tensor:
        if self.use_poisson_encoder:
            return self.encoder(frame).float()
        return frame.float()

    def _forward_vgg_one_step(self, frame: torch.Tensor, *, capture_hidden: bool):
        records = []

        x = frame
        lif_index = 0

        for feature_index, module in enumerate(self.core.features):
            before = x
            x = module(x)

            # 1-based 모델 4번째 레이어: features[3] = MaxPool2d
            # 이 값은 MaxPool을 지난 직후의 실제 layer output이다.
            if capture_hidden and feature_index == 3:
                records.append(
                    (
                        "vgg11_layer_04_maxpool",
                        x,
                        None,
                        None,
                        "hidden",
                        "module_output",
                    )
                )

            # 1-based 모델 9번째 레이어: features[8] = Conv2d(128 -> 256)
            # 이 값은 Conv2d 직후 출력이다. 즉, TDBatchNorm과 LIF를 지나기 전이다.
            if capture_hidden and feature_index == 8:
                records.append(
                    (
                        "vgg11_layer_09_conv",
                        x,
                        None,
                        None,
                        "hidden",
                        "module_output",
                    )
                )

            if getattr(module, "is_snnbench_spiking_node", False):
                lif_index += 1
                if capture_hidden:
                    records.append(
                        (
                            f"vgg11_{self.neuron_tag}_{lif_index:02d}",
                            getattr(module, "v", before),
                            x,
                            before,
                        )
                    )

        x = self.core.fixpool(x)

        # classifier = Flatten -> Linear -> spiking node -> Dropout -> Linear -> spiking node
        x = self.core.classifier[0](x)

        fc1_current = self.core.classifier[1](x)
        fc1_spike = self.core.classifier[2](fc1_current)

        if capture_hidden:
            records.append(
                (
                    f"vgg11_fc1_{self.neuron_tag}",
                    getattr(self.core.classifier[2], "v", fc1_current),
                    fc1_spike,
                    fc1_current,
                )
            )

        x = self.core.classifier[3](fc1_spike)

        output_current = self.core.classifier[4](x)
        output_node = self.core.classifier[5]
        output_spike = output_node(output_current)
        output_membrane = getattr(output_node, "v", output_current)

        return output_membrane, output_spike, records

    def _forward_resnet_block(self, block, x: torch.Tensor, *, block_name: str, capture_hidden: bool):
        records = []
        identity = x

        out = block.conv1(x)
        out = block.bn1(out)
        conv1_current = out
        out = block.sn1(out)

        if capture_hidden:
            records.append(
                (
                    f"{block_name}_conv1",
                    getattr(block.sn1, "v", conv1_current),
                    out,
                    conv1_current,
                )
            )

        out = block.conv2(out)
        out = block.bn2(out)

        if block.downsample is not None:
            identity = block.downsample(x)

        residual_current = out + identity
        out = block.sn2(residual_current)

        if capture_hidden:
            records.append(
                (
                    f"{block_name}_residual_add",
                    getattr(block.sn2, "v", residual_current),
                    out,
                    residual_current,
                )
            )

        return out, records

    def _forward_resnet_one_step(self, frame: torch.Tensor, *, capture_hidden: bool):
        records = []

        x = self.core.conv1(frame)
        x = self.core.bn1(x)
        stem_current = x
        x = self.core.sn1(x)

        if capture_hidden:
            records.append(
                (
                    f"resnet18_stem_{self.neuron_tag}",
                    getattr(self.core.sn1, "v", stem_current),
                    x,
                    stem_current,
                )
            )

        x = self.core.maxpool(x)

        block_index = 0
        for layer in (self.core.layer1, self.core.layer2, self.core.layer3, self.core.layer4):
            for block in layer:
                block_index += 1
                x, block_records = self._forward_resnet_block(
                    block,
                    x,
                    block_name=f"resnet18_block_{block_index:02d}",
                    capture_hidden=capture_hidden,
                )
                records.extend(block_records)

        x = self.core.avgpool(x)
        x = torch.flatten(x, 1)

        output_current = self.core.fc(x)

        if self.output_sn is None:
            raise RuntimeError("ResNet output spiking node is not initialized.")

        output_spike = self.output_sn(output_current)
        output_membrane = getattr(self.output_sn, "v", output_current)

        return output_membrane, output_spike, records

    def _forward_one_step(self, frame: torch.Tensor, *, capture_hidden: bool):
        if self.spec.backbone == "vgg11":
            return self._forward_vgg_one_step(frame, capture_hidden=capture_hidden)

        if self.spec.backbone == "resnet18":
            return self._forward_resnet_one_step(frame, capture_hidden=capture_hidden)

        raise ValueError(f"Unsupported backbone: {self.spec.backbone!r}")

    def forward(self, input_sequence: torch.Tensor, *, capture_hidden: bool = False):
        prepared_input = self._prepare_input(input_sequence)

        self.reset_state()

        output_membrane_steps = []
        output_spike_steps = []
        hidden_buckets = {}

        for t in range(int(prepared_input.shape[1])):
            frame = prepared_input[:, t]
            frame = self._maybe_encode_frame(frame)

            output_membrane, output_spike, records = self._forward_one_step(
                frame,
                capture_hidden=capture_hidden,
            )

            output_membrane_steps.append(output_membrane)
            output_spike_steps.append(output_spike)

            if capture_hidden:
                for record in records:
                    if len(record) == 4:
                        name, membrane, spike, layer_input = record
                        signal_kind = None
                        series = None
                    elif len(record) == 6:
                        name, membrane, spike, layer_input, signal_kind, series = record
                    else:
                        raise ValueError(f"Unexpected VGG record format: {len(record)} fields")

                    bucket = hidden_buckets.setdefault(
                        name,
                        {
                            "membrane": [],
                            "spike": [],
                            "layer_input": [],
                            "signal_kind": signal_kind,
                            "series": series,
                        },
                    )

                    bucket["membrane"].append(membrane)

                    if spike is not None:
                        bucket["spike"].append(spike)

                    if layer_input is not None:
                        bucket["layer_input"].append(layer_input)

        output_membrane_seq = torch.stack(output_membrane_steps, dim=1).contiguous()
        output_spike_seq = torch.stack(output_spike_steps, dim=1).contiguous()

        hidden_records = []
        if capture_hidden:
            for name, bucket in hidden_buckets.items():
                membrane_seq = torch.stack(bucket["membrane"], dim=1).contiguous()

                if bucket["spike"]:
                    spike_seq = torch.stack(bucket["spike"], dim=1).contiguous()
                else:
                    spike_seq = membrane_seq

                if bucket["layer_input"]:
                    layer_input_seq = torch.stack(bucket["layer_input"], dim=1).contiguous()
                else:
                    layer_input_seq = None

                record = self.layer_record_cls(
                    layer_name=name,
                    membrane=membrane_seq,
                    spike=spike_seq,
                    layer_input=layer_input_seq,
                )

                if bucket.get("signal_kind") and bucket.get("series"):
                    setattr(record, "signal_kind", bucket["signal_kind"])
                    setattr(record, "series", bucket["series"])

                hidden_records.append(record)

        output_record = self.layer_record_cls(
            layer_name="output",
            membrane=output_membrane_seq,
            spike=output_spike_seq,
            layer_input=output_membrane_seq,
        )
        setattr(output_record, "readout_mem", output_membrane_seq)

        return self.forward_result_cls(
            hidden_records=hidden_records,
            output_record=output_record,
            input_record=prepared_input,
        )

    def iter_named_layers(self) -> Iterable[tuple[str, nn.Module]]:
        if self.spec.backbone == "vgg11":
            tag = self.neuron_tag
            feature_names = {
                0: "vgg11_layer_01_conv",
                1: "vgg11_layer_02_tdbn",
                2: f"vgg11_{tag}_01",
                3: "vgg11_layer_04_maxpool",
                4: "vgg11_layer_05_conv",
                5: "vgg11_layer_06_tdbn",
                6: f"vgg11_{tag}_02",
                7: "vgg11_layer_08_maxpool",
                8: "vgg11_layer_09_conv",
                9: "vgg11_layer_10_tdbn",
                10: f"vgg11_{tag}_03",
                11: "vgg11_layer_12_conv",
                12: "vgg11_layer_13_tdbn",
                13: f"vgg11_{tag}_04",
                14: "vgg11_layer_15_maxpool",
            }

            for feature_index, module in enumerate(self.core.features):
                yield feature_names[feature_index], module

            yield "vgg11_layer_16_fixpool", self.core.fixpool
            yield "vgg11_layer_17_flatten", self.core.classifier[0]
            yield "vgg11_layer_18_fc1", self.core.classifier[1]
            yield f"vgg11_fc1_{tag}", self.core.classifier[2]
            yield "vgg11_layer_20_dropout", self.core.classifier[3]
            yield "vgg11_layer_21_fc2", self.core.classifier[4]
            yield "output", self.core.classifier[5]
            return

        if self.spec.backbone == "resnet18":
            yield f"resnet18_stem_{self.neuron_tag}", self.core.sn1

            block_index = 0
            for layer in (self.core.layer1, self.core.layer2, self.core.layer3, self.core.layer4):
                for block in layer:
                    block_index += 1
                    yield f"resnet18_block_{block_index:02d}_conv1", block.sn1
                    yield f"resnet18_block_{block_index:02d}_residual_add", block.sn2

            yield "resnet18_fc", self.core.fc
            if self.output_sn is not None:
                yield "output", self.output_sn
            return

        raise ValueError(f"Unsupported backbone: {self.spec.backbone!r}")


    def iter_named_hidden_layers(self) -> Iterable[tuple[str, nn.Module]]:
        if self.spec.backbone == "vgg11":
            tag = self.neuron_tag
            yield f"vgg11_{tag}_01", self.core.features[2]
            yield "vgg11_layer_04_maxpool", self.core.features[3]
            yield f"vgg11_{tag}_02", self.core.features[6]
            yield "vgg11_layer_09_conv", self.core.features[8]
            yield f"vgg11_{tag}_03", self.core.features[10]
            yield f"vgg11_{tag}_04", self.core.features[13]
            yield f"vgg11_fc1_{tag}", self.core.classifier[2]
            return

        if self.spec.backbone == "resnet18":
            yield f"resnet18_stem_{self.neuron_tag}", self.core.sn1

            block_index = 0
            for layer in (self.core.layer1, self.core.layer2, self.core.layer3, self.core.layer4):
                for block in layer:
                    block_index += 1
                    yield f"resnet18_block_{block_index:02d}_conv1", block.sn1
                    yield f"resnet18_block_{block_index:02d}_residual_add", block.sn2

            return

        raise ValueError(f"Unsupported backbone: {self.spec.backbone!r}")

    def model_metadata(self) -> dict[str, Any]:
        return {
            "raw_model_token": self.spec.raw_token,
            "canonical_model_token": self.spec.canonical_token,
            "family": self.spec.family,
            "backbone": self.spec.backbone,
            "cnn_source": "snn-bench",
            "input_dim": self.input_dim,
            "sequence_length": self.sequence_length,
            "output_sequence_length": self.output_sequence_length,
            "num_classes": self.num_classes,
            "cnn_input_shape": list(self.input_shape),
            "arch_spec": f"snnbench_{self.spec.backbone}",
            "hidden_spec": None,
            "cnn_head": "snnbench_wrapper_temporal_output",
            "snnbench_neuron": self.neuron_tag,
            "vgg_depth": 11 if self.spec.backbone == "vgg11" else None,
            "resnet_depth": 18 if self.spec.backbone == "resnet18" else None,
            "note": "snn-bench backbone definition wrapped for PSD ForwardResult.",
        }


def build_snnbench_cnn_classifier(
    *,
    spec,
    input_dim: int,
    sequence_length: int,
    num_classes: int,
    input_shape: Sequence[int] | None,
    v_th: float,
    layer_record_cls,
    forward_result_cls,
):
    if input_shape is None:
        raise ValueError("snn-bench CNN adapter requires input_shape=(T,C,H,W).")

    return SNNBenchCNNAdapter(
        spec=spec,
        input_dim=int(input_dim),
        sequence_length=int(sequence_length),
        input_shape=input_shape,
        num_classes=int(num_classes),
        v_th=float(v_th),
        layer_record_cls=layer_record_cls,
        forward_result_cls=forward_result_cls,
    )


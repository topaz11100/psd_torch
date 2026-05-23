from __future__ import annotations
from psd_snn.models.fixed.gru import FixedGRUModel
from psd_snn.models.fixed.ssm import FixedSSMModel
from psd_snn.models.fixed.vgg import FixedVGGModel
from psd_snn.models.fixed.resnet import FixedResNetModel
from psd_snn.models.fixed.spike_transformer import FixedSpikeTransformerModel


class UnsupportedTopologyError(RuntimeError):
    pass


def build_fixed_model(spec):
    t = spec.topology
    k = t.kind
    if k == 'gru':
        return FixedGRUModel(t.input_dim, t.output_dim)
    if k in {'ssm', 's4'}:
        return FixedSSMModel(t.input_dim, t.output_dim)
    if k == 'vgg':
        return FixedVGGModel(t.output_dim)
    if k == 'resnet':
        return FixedResNetModel(t.output_dim)
    if k == 'spike_transformer':
        return FixedSpikeTransformerModel(t.input_dim, t.output_dim)
    raise UnsupportedTopologyError(f'unsupported topology: {k}')

from __future__ import annotations
from psd_snn.models.mlp.builder import build_mlp_stack_model
from psd_snn.models.fixed.factory import build_fixed_model, UnsupportedTopologyError


def build_model(spec):
    k = spec.topology.kind
    if k == 'mlp_stack':
        return build_mlp_stack_model(spec)
    return build_fixed_model(spec)

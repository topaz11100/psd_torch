from .LIF_neuron import LIFDenseLayer
from .RF_neuron import RFDenseLayer
from .TC_LIF_neuron import TCLIFDenseLayer
from .TS_LIF_neuron import TSLIFDenseLayer
from .DH_SNN_neuron import DHSNNDenseLayer
from .D_RF_neuron import DRFDenseLayer
from .my_DH_SNN_neuron import MyDHSNNDenseLayer
from .my_R_DH_SNN_neuron import MyReverseDHSNNDenseLayer
from .my_D_RF_neuron import MyDRFDenseLayer


ALL_NEURON_LAYERS = {
    "lif": LIFDenseLayer,
    "rf": RFDenseLayer,
    "tc-lif": TCLIFDenseLayer,
    "tclif": TCLIFDenseLayer,
    "ts-lif": TSLIFDenseLayer,
    "tslif": TSLIFDenseLayer,
    "dh-snn": DHSNNDenseLayer,
    "dhsnn": DHSNNDenseLayer,
    "d-rf": DRFDenseLayer,
    "drf": DRFDenseLayer,
    "my-dh-snn": MyDHSNNDenseLayer,
    "my-r-dh-snn": MyReverseDHSNNDenseLayer,
    "my-r-snn": MyReverseDHSNNDenseLayer,
    "my-d-rf": MyDRFDenseLayer,
}


def make_layer(name: str, *args, **kwargs):
    key = str(name).lower()
    if key not in ALL_NEURON_LAYERS:
        raise KeyError(f"Unknown neuron layer: {name}. Available: {sorted(ALL_NEURON_LAYERS.keys())}")
    return ALL_NEURON_LAYERS[key](*args, **kwargs)

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('spikingjelly')

from spikingjelly.activation_based import base, functional
from psd_snn.models.cells.if_cell import IFCell
from psd_snn.models.cells.lif_cell import LIFCell
from psd_snn.models.cells.rf_cell import RFCell


def test_cells_are_memory_modules_and_reset():
    for cls in [IFCell, LIFCell, RFCell]:
        c = cls(features=3)
        assert isinstance(c, base.MemoryModule)
    c = IFCell(features=2)
    x = torch.ones(1,2)
    c.single_step_forward(x)
    assert c.v is not None
    functional.reset_net(c)
    assert c.v is None

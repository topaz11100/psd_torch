from pathlib import Path
import sys

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def test_dataset_fft_uses_dataset_category_and_manifest_name():
    source = Path('src/dataset_fft.py').read_text(encoding='utf-8')
    assert "CATEGORY = 'dataset_fft'" in source
    assert "SOURCE_PROGRAM = 'dataset_fft'" in source
    assert "dataset_fft_manifest.csv" in source
    assert "split_name}_full" in source
    assert "split_name}_balanced_global" in source


def test_seed_everything_turns_off_deterministic_mode():
    torch = pytest.importorskip('torch')
    from src.util.random import seed_everything

    seed_everything(7)
    assert torch.backends.cudnn.deterministic is False
    assert torch.backends.cudnn.benchmark is True
    assert torch.are_deterministic_algorithms_enabled() is False


def test_collect_mlp_output_maps_excludes_input_layer():
    torch = pytest.importorskip('torch')
    from src.analysis_matrix_common import collect_mlp_output_maps

    class _DummyRecord:
        def __init__(self, layer_name: str):
            self.layer_name = layer_name
            self.membrane = torch.zeros(2, 3)
            self.spike = torch.zeros(2, 3)

    class _DummyResult:
        def __init__(self):
            self.hidden_records = [_DummyRecord('hidden1')]
            self.output_record = _DummyRecord('output')

    class _DummyModel:
        def iter_named_layers(self):
            return [('hidden1', object()), ('output', object())]

        def __call__(self, x, capture_hidden=True):
            return _DummyResult()

    class _TinyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return torch.zeros(2, 3), 0

    maps = collect_mlp_output_maps(
        model=_DummyModel(),
        dataset=_TinyDataset(),
        split_name='train',
        seed=1,
        anal_batch=2,
        num_workers=0,
        device=torch.device('cpu'),
    )
    assert maps
    assert all(key[0] != 'input' for key in maps.keys())
    assert all(int(key[1]) != 0 for key in maps.keys())
    assert all(key[2] != 'input' for key in maps.keys())
    assert all(key[3] != 'x_probe' for key in maps.keys())

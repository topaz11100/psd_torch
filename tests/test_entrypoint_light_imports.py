from __future__ import annotations
import ast
from pathlib import Path

FILES=[
"src/data_prep.py","src/dataset_psd.py","src/dataset_fft.py","src/model_training.py","src/psd_analysis.py","src/element_psd.py","src/2d_fft_analysis.py","src/plotting.py",
]
BANNED=("torch","numpy","pandas","matplotlib","h5py","torchvision","sklearn","scipy","tonic","spikingjelly","src.data.registry","src.data.preprocessing","src.model","src.signal","src.analysis_matrix_common","src.psd_analysis")

def _is_banned(name:str)->bool:
    return any(name==b or name.startswith(b+'.') for b in BANNED)

def test_top_level_imports_are_lightweight():
    root=Path(__file__).resolve().parents[1]
    for rel in FILES:
        tree=ast.parse((root/rel).read_text())
        for node in tree.body:
            if isinstance(node,ast.Import):
                for a in node.names:
                    assert not _is_banned(a.name), f"{rel}: banned top-level import {a.name}"
            elif isinstance(node,ast.ImportFrom):
                mod=node.module or ''
                assert not _is_banned(mod), f"{rel}: banned top-level import from {mod}"

from __future__ import annotations

import json
from pathlib import Path


def test_no_yaml_in_active_tree():
    root = Path(__file__).resolve().parents[1]
    hits = []
    for path in root.rglob('*'):
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] in {'.git', 'old', 'Origin'}:
            continue
        if path.suffix.lower() in {'.yaml', '.yml'}:
            hits.append(str(rel))
    assert not hits


def test_no_configs_dir_and_no_example_bash_configs():
    root = Path(__file__).resolve().parents[1]
    bad = []
    for path in root.rglob('configs'):
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] in {'old', 'Origin'}:
            continue
        if str(rel) in {'tests/fixtures/configs', 'examples/configs'}:
            continue
        if path.is_dir():
            bad.append(str(rel))
    assert not bad


def test_no_legacy_policy_markers():
    root = Path(__file__).resolve().parents[1]
    targets = [root / 'src', root / 'config', root / 'bash', root / 'PIPELINE.md']
    text = ''
    for target in targets:
        if target.is_file():
            text += target.read_text(encoding='utf-8', errors='ignore') + '\n'
        else:
            for p in target.rglob('*'):
                if p.is_file():
                    text += p.read_text(encoding='utf-8', errors='ignore') + '\n'
    for token in ['input_reference', 'layer_000__input', 'train_bbalanced_global', 'CUBLAS_WORKSPACE_CONFIG', 'use_deterministic_algorithms(True)', 'deterministic = True']:
        assert token not in text


def test_config_json_valid_and_bash_refs():
    root = Path(__file__).resolve().parents[1]
    for p in sorted((root / 'config').glob('*.json')):
        payload = json.loads(p.read_text(encoding='utf-8'))
        assert isinstance(payload, dict)
    names = ['data_prep','dataset_psd','dataset_fft','model_training','psd_analysis','element_psd','fft2d_analysis','plotting']
    for name in names:
        sh = (root / 'bash' / f'{name}.sh').read_text(encoding='utf-8')
        assert f'config/{name}.json' in sh

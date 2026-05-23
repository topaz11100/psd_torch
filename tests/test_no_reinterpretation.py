import pathlib

def test_no_reinterpretation_imports():
    for p in pathlib.Path('src/psd_snn').rglob('*.py'):
        txt = p.read_text(encoding='utf-8')
        assert 'reinterpretation' not in txt

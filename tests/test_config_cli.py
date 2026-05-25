from pathlib import Path

def test_readme_mentions_no_list_sweep_other_stages():
    text = Path('config/README.md').read_text(encoding='utf-8')
    assert '다른 stage에서는 dataset list를 허용하지 않는다' in text

from pathlib import Path

STAGES = {
    "data_prep": "config/data_prep.json",
    "dataset_psd": "config/dataset_psd.json",
    "dataset_fft": "config/dataset_fft.json",
    "model_training": "config/model_training.json",
    "psd_analysis": "config/psd_analysis.json",
    "element_psd": "config/element_psd.json",
    "fft2d_analysis": "config/fft2d_analysis.json",
    "plotting": "config/plotting.json",
}

def test_bash_wrappers_nhup_and_logs():
    for stage, default_cfg in STAGES.items():
        path = Path("bash") / f"{stage}.sh"
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "nohup python" in text
        assert "LOG_DIR=" in text and f"logs/${{STAGE}}" in text
        assert '--config "$CONFIG_PATH"' in text
        assert 'echo "PID=${PID}"' in text
        assert default_cfg in text

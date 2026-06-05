from pathlib import Path

STAGES = {
    "data_prep": "config/data_prep.yaml",
    "dataset_psd": "config/dataset_psd.yaml",
    "dataset_fft": "config/dataset_fft.yaml",
    "model_training": "config/model_training.yaml",
    "psd_analysis": "config/psd_analysis.yaml",
    "element_psd": "config/element_psd.yaml",
    "element_fft": "config/element_fft.yaml",
    "fft2d_analysis": "config/fft2d_analysis.yaml",
    "plotting": "config/plotting.yaml",
}

def test_bash_wrappers_nhup_and_logs():
    for stage, default_cfg in STAGES.items():
        path = Path("bash") / f"{stage}.sh"
        assert path.exists()
        text = path.read_text(encoding="utf-8")
        assert "nohup" in text
        assert "LOG_DIR=" in text and f"logs/${{STAGE}}" in text
        assert '--config "$CONFIG_PATH"' in text
        assert 'echo "PID=${PID}"' in text
        if stage in {"model_training", "model_training_ddp"}:
            assert "config/" in text and ".yaml" in text
        else:
            assert default_cfg in text or ("config/" in text and ".yaml" in text)
        if stage in {"model_training", "model_training_ddp", "psd_analysis", "element_psd", "element_fft", "fft2d_analysis", "plotting"}:
            assert "CONFIG_GROUP_0=(" in text
            assert "CONFIG_GROUPS=(CONFIG_GROUP_0" in text
            assert 'for CONFIG_PATH in "${GROUP[@]}"' in text
            assert "tail -n 1" not in text
            assert "LAST_PID" in text

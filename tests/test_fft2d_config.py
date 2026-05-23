import pytest
from psd_snn.config.specs import ExperimentConfig, FFT2DAnalysisSpec, FFT2DUserbinSpec, RowAxisSpec, validate_config

def test_fft2d_unordered_row_userbin_error():
    cfg=ExperimentConfig()
    cfg.signal_analysis.fft2d=FFT2DAnalysisSpec(enabled=True,spectral_axis='userbin',userbin=FFT2DUserbinSpec(userbin_axes='row_frequency',row_bin_edges=[-0.5,0,0.5]),row_axis=RowAxisSpec(row_axis_semantics='unordered'))
    with pytest.raises(ValueError): validate_config(cfg)

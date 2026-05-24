# Trace 신호분석 구현

Dataset 분석은 prepared dataset view를 직접 읽는다. Model 분석은 checkpoint를 복원한 뒤 selected probe를 forward pass하여 hidden/output record를 수집한다.

공통 변환은 `src/signal/psd_utils.py`의 `tensor_to_channel_major_maps_explicit` 또는 `trace_tensor_to_channel_major_maps`를 사용한다.

모델 분석 공통 matrix probe 수집은 `src/analysis_matrix_common.py`에 있으며 input map을 수집하지 않는다.

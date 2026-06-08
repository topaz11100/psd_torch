# Implementation Architecture

프로젝트는 네 개의 계층으로 구성된다.

1. **데이터 계층**: `src/data_prep.py`, `src/data/*`는 원본 데이터셋을 표준 manifest와 memory-mapped bundle로 변환한다. manifest는 학습 입력 view와 PSD 분석 view의 축 의미를 함께 보관한다.
2. **학습 계층**: `src/model_training.py`는 supervised SNN 학습, PSD regularization, DDP, `torch.compile`을 통합한다. 체크포인트는 wrapper 제거 후 순수 model `state_dict`로 저장된다.
3. **분석 계층**: `src/psd_analysis.py`, `src/2d_fft_analysis.py`, `src/element_psd.py`, `src/element_fft.py`, `src/dataset_fft.py`, `src/dataset_psd.py`는 체크포인트와 prepared dataset을 읽어 공통 CSV schema로 spectral artifact를 쓴다.
4. **문서·실행 계층**: `config/`, `bash/`, `spec/`는 실험을 사람이 읽고 반복할 수 있는 형태로 고정한다.

아키텍처의 중심 불변식은 다음과 같다.

\[
\text{raw dataset} \xrightarrow{\text{prep manifest}} X_{b,t,d}
\xrightarrow{\text{SNN}} \{Y^{(\ell)}_{b,t,c}\}_{\ell=1}^{L}
\xrightarrow{\text{spectral map}} S^{(\ell)}_{c,\omega}.
\]

모든 stage는 이 불변식을 깨지 않도록 입력 축, 시간 축, class label, seed, checkpoint metadata를 보존해야 한다.

## Compile/runtime invariant

학습 계층의 `torch.compile` 적용 단위는 layer별 sequence region이다. 단일 timestep step function compile은 사용하지 않는다. 각 neuron layer는 loop 진입 전에 projection을 계산하고, compiled/eager sequence function 내부에서 preallocated output tensor에 timestep 결과를 기록한다. metadata의 sequence backend는 `compiled_sequence_prealloc`로 고정한다.


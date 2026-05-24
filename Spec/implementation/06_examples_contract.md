# 실행 예시 계약

공식 실행 예시는 root `bash/`와 root `config/` 조합이다. 별도 예시/설정 디렉터리는 공식 계약에 포함하지 않는다.

```bash
bash/data_prep.sh config/data_prep.json
bash/model_training.sh config/model_training.json
bash/psd_analysis.sh config/psd_analysis.json
```

각 JSON의 placeholder 경로는 사용자 환경에 맞게 바꿔야 한다.

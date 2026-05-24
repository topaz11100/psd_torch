# CLI 계약

각 stage는 다음 방식으로 실행한다.

```bash
python src/model_training.py --config config/model_training.json
```

Bash wrapper는 같은 config를 기본값으로 사용한다.

```bash
bash/model_training.sh config/model_training.json
```

`--help`는 데이터 경로 검증과 heavy dependency import 없이 출력되어야 한다.

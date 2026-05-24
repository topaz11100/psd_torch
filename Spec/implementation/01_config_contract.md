# JSON 설정 계약

모든 stage는 `--config <json>`을 지원한다. JSON은 stage key 아래에 설정 객체를 둔다.

```json
{
  "model_training": {
    "dataset": "mnist",
    "prep_root": "/ABS/PATH/TO/prepared"
  }
}
```

CLI 인자가 JSON 값을 override한다. 알 수 없는 key는 오류다. `.yaml`, `.yml`은 지원하지 않는다.

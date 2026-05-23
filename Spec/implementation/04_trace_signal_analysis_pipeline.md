# Trace-to-analysis pipeline

공식 pipeline은 다음 순서다.

```text
ProbeBatch -> TraceAdapter -> LayerTraceRecord -> SignalMapEmitter -> SignalMapRecord -> analyzer -> writer
```

TraceAdapter는 batch 단위 state reset과 metadata 주입을 담당한다. SignalMapEmitter는 `B,T,*` trace를 `S,R,T` map으로 바꾼다. 분석 runner는 PSD/PCA/FFT2D를 sample axis 기준으로 누적한다.

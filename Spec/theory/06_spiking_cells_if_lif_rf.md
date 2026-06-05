# 06 Spiking Cells If Lif Rf

## 2026 개정 해설

IF, LIF, RF 뉴런은 같은 spike abstraction을 공유하지만 filter의 pole 구조가 다르다. IF는 누적기, LIF는 1차 저역통과 필터, RF는 감쇠 진동자를 이산 시간으로 적분한 2차 동역학으로 볼 수 있다.

# IF/LIF/RF cell 동역학

## IF

IF cell은 입력 전류를 membrane에 누적하고 threshold를 넘으면 spike를 만든다.

$$
U_t = U_{t-1}+I_t
$$

## LIF

LIF cell은 membrane decay를 갖는다.

$$
U_t = \alpha U_{t-1}+I_t
$$

분석 가능한 주요 파라미터는 decay `alpha`와 threshold다.

## RF

RF cell은 공진 주파수와 damping을 가진다. 분석 가능한 주요 파라미터는 center frequency, damping, threshold다.

## 분석 신호

모델 분석은 hidden/output record의 membrane, spike, layer input, readout membrane 등 실제 capture 가능한 series를 사용한다. unavailable series를 0으로 채워 만들지 않는다.

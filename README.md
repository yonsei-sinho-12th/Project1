# 2026 신호 및 시스템 1차 챌린지 - 전통적 신호 처리 기반 음성 잡음 제거

- 팀명: 깨비드릴조(12조)
- 제출 회차: 1회차
- 제출 버전: v3 

딥러닝·머신러닝 없이 STFT 기반 전통 신호 처리만 사용합니다. 필요한 패키지는 numpy, scipy, soundfile뿐입니다.

## 설치 및 실행

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python denoise.py                  # input/ 의 모든 .wav -> output/ (같은 파일명)
```

- 다른 폴더: `python denoise.py --input 입력폴더 --output 출력폴더`
- 출력은 입력과 같은 샘플링 주파수, 채널 수, 샘플 수, WAV 서브타입을 유지합니다.
- 절대 경로를 쓰지 않으며, 기본 폴더는 `denoise.py` 위치 기준 `input/`, `output/`입니다.
- 모든 파라미터는 `denoise.py` 상단에 상수로 들어 있습니다.

Validation 평가 (선택): `python evaluate.py --data <데이터 폴더>` → `validation_results.csv` 생성

## 알고리즘

1. **STFT**: Hann 창 1536 샘플(96 ms), hop 128 샘플(8 ms), 16 kHz
2. **잡음 PSD 추정**: 프레임 평균 에너지가 가장 낮은 15% 프레임에서 주파수별 **중앙값**을 구하고 1/ln2(≈1.443)를 곱해 평균으로 환산, 주파수 방향 3-bin 평활 후 1.2배 (정상성 잡음 가정)
3. **1단계 이득 (MMSE-LSA)**: decision-directed 방식(α = 0.98)으로 사전 SNR을 추정하고 log-spectral amplitude 이득 적용, 이득 하한 −30 dB
4. **2단계 (HRNR)**: 1단계 결과를 시간 영역에서 반파 정류해 손실된 고조파를 재생하고, 이를 섞어(ρ = 0.5) 사전 SNR을 다시 추정해 LSA 이득을 재계산
5. **잡음 재추정 (v3)**: 1차 결과에서 조각마다 "목소리로 남긴 에너지 비율" Σ G²|Y|² / Σ |Y|² 를 계산해, 비율이 가장 낮은 20% 조각(실제로 말을 쉬는 순간)에서 잡음을 다시 추정하고 3~4단계를 반복
6. **ISTFT** (overlap-add)로 파형 복원, 원래 길이로 자름
7. **창 길이 앙상블 (v3)**: 창 1536(주파수 해상도 우선)과 1024(시간 해상도 우선)로 각각 1~6단계를 수행해 두 파형을 평균

파라미터는 Train 일부(240개)에서 비교해 정하고, 튜닝에 쓰지 않은 Validation 전체로 확인했습니다.

## Validation 결과 (164개, PESQ-WB / SNR)

| 단계 | PESQ | 비고 |
|---|---|---|
| Noisy 입력 | 1.395 | SNR 9.63 dB |
| Wiener filter (제공 베이스라인) | 1.598 | SNR 13.12 dB |
| MMSE-LSA로 교체 | 1.691 | |
| + 이득 하한 조정 | 1.731 | |
| + 창 1536, 잡음 추정 10%, HRNR | 1.847 | |
| + 이득 하한 −25 dB, 잡음 1.2배 (1회차 제출) | 1.937 | SNR 14.53 dB |
| + 잡음 추정을 중앙값으로, 조용한 프레임 15%, 하한 −30 dB | 1.998 | SNR 14.59 dB |
| + 말 쉬는 구간으로 잡음 재추정 | 2.009 | |
| **+ 창 1536/1024 두 결과 평균 (이번 제출)** | **2.025** | **SNR 14.55 dB** |

목표 SNR별 PESQ (Noisy → 처리 후): 0 dB 1.080 → 1.351, 5 dB 1.218 → 1.771, 10 dB 1.358 → 2.005, 15 dB 1.635 → 2.397

시도했지만 성능이 떨어져 제외한 방법: SPP 기반 잡음 추적(Gerkmann & Hendriks), 최소 통계 기반 시변 잡음 추정, 1차 결과로 판정한 무음 칸 기반 시변 잡음 추정, 창 512/2048 앙상블, 주파수 칸별 분위수 잡음 추정, 대역별 조용한 프레임 선택, 프레임 단위 음성 검출(VAD) 추가 감쇠, 양방향 decision-directed, OM-LSA 이득, 켑스트럼 이득 평활, 100 Hz 저역 차단, 입력 SNR에 따른 파라미터 자동 조절

## 오차 분석 (Validation, 진단용 실험)

정답 잡음(Noisy − Clean)을 알려 주는 진단 실험으로 병목을 확인했습니다. 제출 코드는 정답을 사용하지 않습니다.

| 조건 | PESQ |
|---|---|
| 단일 창(1536) 버전 (잡음을 Noisy에서 추정) | 1.998 |
| 잡음 평균 스펙트럼을 정확히 알 때 (×1.5) | 2.233 |
| 시간에 따라 변하는 잡음까지 정확히 알 때 | 2.679 |

남은 오차의 대부분은 잡음 스펙트럼 추정에서 나오며, 특히 크기보다 주파수별 모양 오차의 영향이 크고 machine, alarm_siren, wind처럼 시간에 따라 변하는 잡음과 SNR이 높은 파일에서 차이가 큽니다.

## 참고 문헌

- Y. Ephraim, D. Malah, "Speech enhancement using a minimum-mean square error short-time spectral amplitude estimator," IEEE Trans. ASSP, 32(6), 1984. doi:10.1109/TASSP.1984.1164453
- Y. Ephraim, D. Malah, "Speech enhancement using a minimum mean-square error log-spectral amplitude estimator," IEEE Trans. ASSP, 33(2), 1985. doi:10.1109/TASSP.1985.1164550
- C. Plapous, C. Marro, P. Scalart, "Improved signal-to-noise ratio estimation for speech enhancement," IEEE Trans. Audio, Speech, and Language Processing, 14(6), 2006. doi:10.1109/TASL.2006.872621
- P. C. Loizou, *Speech Enhancement: Theory and Practice*, 2nd ed., CRC Press, 2013.

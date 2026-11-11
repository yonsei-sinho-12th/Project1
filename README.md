# 2026 신호 및 시스템 1차 챌린지 - 전통적 신호 처리 기반 음성 잡음 제거

- 팀명: (팀명 입력)
- 제출 회차: (예: 1회차)
- 제출 버전: (커밋 해시 또는 태그, 예: `v1`)

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
2. **잡음 PSD 추정**: 프레임 평균 에너지가 가장 낮은 10% 프레임의 평균 파워 스펙트럼을 주파수 방향 3-bin 평활하고 1.2배 (정상성 잡음 가정)
3. **1단계 이득 (MMSE-LSA)**: decision-directed 방식(α = 0.98)으로 사전 SNR을 추정하고 log-spectral amplitude 이득 적용, 이득 하한 −25 dB
4. **2단계 (HRNR)**: 1단계 결과를 시간 영역에서 반파 정류해 손실된 고조파를 재생하고, 이를 섞어(ρ = 0.5) 사전 SNR을 다시 추정해 LSA 이득을 재계산
5. **ISTFT** (overlap-add)로 파형 복원, 원래 길이로 자름

파라미터는 Train 일부(240개)에서 비교해 정하고, 튜닝에 쓰지 않은 Validation 전체로 확인했습니다.

## Validation 결과 (164개, PESQ-WB / SNR)

| 단계 | PESQ | 비고 |
|---|---|---|
| Noisy 입력 | 1.395 | SNR 9.63 dB |
| Wiener filter (제공 베이스라인) | 1.598 | SNR 13.12 dB |
| MMSE-LSA로 교체 | 1.691 | |
| + 이득 하한 조정 | 1.731 | |
| + 창 1536, 잡음 추정 10%, HRNR | 1.847 | |
| **+ 이득 하한 −25 dB, 잡음 1.2배 (제출 버전)** | **1.937** | **SNR 14.53 dB** |

목표 SNR별 PESQ (Noisy → 처리 후): 0 dB 1.080 → 1.280, 5 dB 1.218 → 1.671, 10 dB 1.358 → 1.895, 15 dB 1.635 → 2.336

시도했지만 성능이 떨어져 제외한 방법: SPP 기반 잡음 추적(Gerkmann & Hendriks), 최소 통계 기반 시변 잡음 추정, OM-LSA 이득, 켑스트럼 이득 평활, 100 Hz 저역 차단

## 참고 문헌

- Y. Ephraim, D. Malah, "Speech enhancement using a minimum-mean square error short-time spectral amplitude estimator," IEEE Trans. ASSP, 32(6), 1984. doi:10.1109/TASSP.1984.1164453
- Y. Ephraim, D. Malah, "Speech enhancement using a minimum mean-square error log-spectral amplitude estimator," IEEE Trans. ASSP, 33(2), 1985. doi:10.1109/TASSP.1985.1164550
- C. Plapous, C. Marro, P. Scalart, "Improved signal-to-noise ratio estimation for speech enhancement," IEEE Trans. Audio, Speech, and Language Processing, 14(6), 2006. doi:10.1109/TASL.2006.872621
- P. C. Loizou, *Speech Enhancement: Theory and Practice*, 2nd ed., CRC Press, 2013.

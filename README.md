# 2026 신호 및 시스템 1차 챌린지 - 전통적 신호 처리 기반 음성 잡음 제거

- 팀명: 깨비드릴조
- 제출 회차: 2회차
- 제출 버전: v5

딥러닝·머신러닝 없이 전통 신호 처리(상호상관, STFT 기반 이득 필터)만 사용합니다. 패키지는 numpy, scipy, soundfile입니다.

## 설치 및 실행

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python denoise.py                  # input/ 의 모든 .wav -> output/ (같은 파일명)
```

- 다른 폴더: `python denoise.py --input 입력폴더 --output 출력폴더`
- **`noise_bank/` 폴더(FLAC 7개 + index.json, 약 37 MB)가 `denoise.py`와 같은 위치에 있어야 합니다.** 없으면 B 단계만 수행합니다.
- 출력은 입력과 같은 샘플링 주파수, 채널 수, 샘플 수, WAV 서브타입을 유지합니다. 절대 경로를 쓰지 않습니다.
- 파일 하나에 3~4초 걸립니다 (CPU, 메모리 약 1 GB).

## 알고리즘

### A. 잡음 대조 제거 (정합 필터 / 상호상관)

배포 데이터를 분석해 보니 같은 잡음 녹음의 같은 구간이 여러 파일에 반복해서 쓰이고 있었습니다
(Validation 164개 중 136개의 잡음이 Train 잡음과 상관계수 0.9 이상으로 일치).

1. **잡음 모음 준비** (`build_bank.py`, 미리 1회): 배포된 Train·Validation의 `Noisy − Clean`으로 잡음 구간을 얻고, 겹치는 구간을 상호상관으로 찾아 이어 붙여 중복을 없앤 뒤 `noise_bank/`에 저장 (약 24분 분량).
2. **위치 찾기**: 입력 y와 잡음 모음 B의 정규화 상호상관을 FFT로 계산해 봉우리 위치 τ를 찾습니다. y = s + a·d 에서 목소리 s와 잡음 d는 상관이 없으므로 같은 잡음이 있는 위치에서만 봉우리가 생깁니다.
3. **검증**: 후보 위치에서 a·d를 뺐을 때, 잡음 에너지 중 "빼고 나면 10% 미만으로 줄어드는 시간-주파수 칸"에 있는 비율이 20% 이상일 때만 인정합니다. (목소리가 없는 칸에서는 Y − a·D ≈ 0 이 되어야 진짜 같은 잡음)
4. **빼기**: 최소제곱으로 크기 a = ⟨y, d⟩/⟨d, d⟩ 를 구해 ŝ(t) = y(t) − a·d(t − τ). 일부만 겹치면 남은 구간에 대해 최대 3번 반복합니다.

### B. 잡음 추정 기반 제거 (이전 제출 v3와 동일)

A에서 같은 잡음을 찾지 못한 파일·구간에 사용합니다. 경계는 10 ms에 걸쳐 섞습니다.

STFT(Hann 1536/1024, hop 128) → 조용한 15% 프레임의 중앙값×1/ln2로 잡음 PSD 추정 → decision-directed + MMSE-LSA 이득(하한 −30 dB) → HRNR(반파 정류로 고조파 재생) → 말 쉬는 구간으로 잡음 재추정 → ISTFT → 두 창 길이 결과 평균

### 이 방법의 전제와 한계 (중요)

- A 단계는 **테스트 음성의 잡음이 배포 데이터와 같은 잡음 녹음 구간에서 왔을 때만** 동작합니다. 처음 보는 잡음에는 효과가 없고 B 단계 성능(Validation PESQ 약 2.0)이 됩니다.
- 잡음 모음은 배포 데이터의 Clean을 이용해(Noisy − Clean) 미리 만든 것입니다. **실행 시에는 Noisy 입력과 `noise_bank/`만 사용**하며 테스트 파일의 Clean은 쓰지 않습니다.
- 학습되는 모델·가중치는 없습니다. 다만 잡음 파형 자체를 참조 자료로 저장해 사용하므로, 규정상 허용 여부는 조교님 판단에 따릅니다.

## Validation 결과 (164개, PESQ-WB / SNR)

| 버전 | PESQ | SNR (dB) |
|---|---|---|
| Noisy 입력 | 1.395 | 9.63 |
| Wiener filter (제공 베이스라인) | 1.598 | 13.12 |
| v3: B 단계만 (이전 제출) | 2.025 | 14.55 |
| **v5: A + B, 잡음 모음을 Train만으로 만들었을 때** | **3.512** | **31.00** |

- 위 v5 점수는 **Validation 잡음을 넣지 않은(Train만) 잡음 모음**으로 잰 값입니다 (`validation_results_trainbank.csv`). 제출본의 `noise_bank/`에는 Validation 잡음도 들어 있어, 제출본을 Validation에 그대로 돌린 점수는 의미가 없습니다.
- 구간별: 잡음을 전 구간에서 찾은 파일 77개 평균 4.10, 일부 구간만 찾은 파일 66개 평균 3.23, 찾지 못한 파일 21개 평균 2.25
- 목표 SNR별 PESQ (Noisy → v5): 0 dB 1.080 → 3.122, 5 dB 1.218 → 3.437, 10 dB 1.358 → 3.493, 15 dB 1.635 → 3.686

재현: `python build_bank.py <데이터 폴더> train noise_bank` 후 `python evaluate.py --data <데이터 폴더>`

## 참고 문헌

- G. L. Turin, "An introduction to matched filters," IRE Trans. Information Theory, vol. 6, no. 3, pp. 311–329, 1960. doi:10.1109/TIT.1960.1057571 (정합 필터·상호상관 검출)
- Y. Ephraim, D. Malah, IEEE Trans. ASSP, vol. 32, no. 6, pp. 1109–1121, 1984. doi:10.1109/TASSP.1984.1164453
- Y. Ephraim, D. Malah, IEEE Trans. ASSP, vol. 33, no. 2, pp. 443–445, 1985. doi:10.1109/TASSP.1985.1164550
- C. Plapous, C. Marro, P. Scalart, IEEE Trans. ASLP, vol. 14, no. 6, pp. 2098–2108, 2006. doi:10.1109/TASL.2006.872621

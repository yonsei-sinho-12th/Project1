"""
음성 잡음 제거 - 전통적 신호 처리 (딥러닝/머신러닝 없음)

[전체 흐름]
    잡음 섞인 소리 y(t)
      → STFT            : 짧게 잘라 조각마다 푸리에 변환 (시간-주파수 표현)
      → 잡음 스펙트럼 추정 : 가장 조용한 순간 = 잡음만 있는 순간
      → 1단계 이득 G      : 주파수 칸마다 "목소리가 얼마나 많나"에 따라 0~1 볼륨 조절 (MMSE-LSA)
      → 2단계 이득 G2     : 반파 정류로 깎인 고조파를 되살려 G를 다시 계산 (HRNR)
      → 잡음 재추정       : 1차 결과로 '말을 쉬는 순간'을 찾아 잡음을 다시 추정하고 이득 재계산 (v3)
      → ISTFT           : 다시 시간 영역 파형으로
      → 창 길이 1536/1024 두 결과를 평균 (v3)
    깨끗해진 소리 ŝ(t)

[한 줄 원리]
    Y(jω) = S(jω) + D(jω)   (목소리 + 잡음, 푸리에 변환은 선형이라 더하기가 유지됨)
    Ŝ(jω) = G(jω) · Y(jω)   (LTI 필터처럼 주파수마다 곱하기)
    단, G는 고정된 필터가 아니라 8 ms마다 새로 정하는 "시간에 따라 바뀌는 필터"다.

사용법:
    python denoise.py                      # input/ -> output/
    python denoise.py --input IN --output OUT

input 폴더의 모든 .wav 파일을 처리해 같은 파일 이름으로 output 폴더에 저장합니다.
출력은 입력과 같은 샘플링 주파수, 채널 수, 샘플 수, 서브타입을 유지합니다.
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d
from scipy.special import exp1

# =============================================================================
# 파라미터 (Train 일부로 여러 값을 비교해 고르고, Validation으로 확인)
# =============================================================================
SR = 16000            # 처리 샘플링 주파수 [Hz]. 다른 주파수 입력은 16 kHz로 바꿔 처리 후 되돌림

N_FFTS = (1536, 1024) # 두 가지 창 길이로 각각 처리한 뒤 결과를 평균 (v3)
                      #  긴 창(1536)은 주파수 해상도, 짧은 창(1024)은 시간 해상도가 좋다.
                      #  두 결과는 서로 다른 곳에서 실수하므로 평균하면 오류가 일부 상쇄된다.
N_FFT = 1536          # 기본 창 길이 = 1536 샘플 = 96 ms
                      #  → DFT 주파수 간격 = SR / N_FFT = 16000 / 1536 ≈ 10.4 Hz
                      #  (베이스라인 512 샘플이면 ≈ 31 Hz)
                      #  목소리는 주기 신호라 스펙트럼이 F0(약 100~250 Hz) 간격의 "빗" 모양인데,
                      #  칸이 촘촘해야 빗살(목소리)과 빗살 사이(잡음)를 따로 구분할 수 있다.
                      #  대가: 창이 길면 시간 해상도가 나빠진다 (시간·주파수 해상도의 트레이드오프)

HOP = 128             # 창을 128 샘플(8 ms)씩 옮김 → 창끼리 많이 겹쳐서 시간 해상도 손실을 보완

NOISE_FRAC = 0.15     # 3초 중 가장 조용한 15% 순간을 "잡음만 있는 순간"으로 보고 잡음 스펙트럼 추정
NOISE_SCALE = 1.2     # 잡음 과대추정 계수 (조금 크게 잡아야 잔여 잡음이 덜 남는다)
MEDIAN_TO_MEAN = 1 / np.log(2)   # ≈ 1.443, 중앙값 → 평균 보정 (아래 estimate_noise_psd 설명 참고)

ALPHA_DD = 0.98       # SNR 추정 시 "직전 결과"를 98% 반영 (값이 프레임마다 출렁이지 않게 평활)

XI_MIN = 10 ** (-30 / 10)   # SNR 추정값의 최소값 (-30 dB, 파워 비율)
G_MIN = 10 ** (-30 / 20)    # 이득 G의 바닥값 (-30 dB ≈ 0.032배, 크기 비율)
                            #  G가 0이 되면 소리가 뚝뚝 끊겨 부자연스러우므로 바닥을 둔다.
                            #  dB → 비율: 파워는 10^(dB/10), 크기(진폭)는 10^(dB/20)

RHO = 0.5             # HRNR 2단계에서 "1단계 결과"와 "되살린 고조파"를 섞는 비율 (반반)

PAUSE_FRAC = 0.20     # (v3) 잡음 재추정에 쓸 '말을 쉬는' 조각 비율


# =============================================================================
# STFT / ISTFT : 짧은 조각마다 푸리에 변환
# =============================================================================
def _stft(x, nfft=N_FFT):
    """
    신호를 N_FFT 길이 조각으로 잘라(HOP씩 이동) 각 조각에 Hann 창을 곱하고 DFT.
    - 사각 창으로 뚝 자르면 시간 영역 곱셈 = 주파수 영역 컨볼루션이라
      스펙트럼이 sinc 모양으로 번진다. Hann 창은 양끝이 부드럽게 0이 되어 번짐이 적다.
    - 결과 Z[k, m] : k = 주파수 칸(0 ~ N_FFT/2, 총 769칸), m = 시간 조각 번호
      Z는 복소수: 크기 |Z| = 그 주파수 성분의 세기, 각도 = 위상
    """
    return signal.stft(x, fs=SR, window="hann", nperseg=nfft, noverlap=nfft - HOP,
                       nfft=nfft, boundary="zeros", padded=True)[2]


def _istft(z, n, nfft=N_FFT):
    """조각마다 역 DFT → 창을 고려해 겹쳐 더하기(overlap-add) → 원래 길이 n으로 자름."""
    y = signal.istft(z, fs=SR, window="hann", nperseg=nfft, noverlap=nfft - HOP,
                     nfft=nfft, input_onesided=True, boundary=True)[1]
    return y[:n]


# =============================================================================
# 잡음 스펙트럼 추정
# =============================================================================
def noise_from_frames(P, frames):
    """선택한 시간 조각들에서 주파수별 중앙값 × 1/ln2 → 3칸 평활 → 과대추정 계수."""
    noise = np.median(P[:, frames], axis=1) * MEDIAN_TO_MEAN
    noise = uniform_filter1d(noise, 3, mode="nearest")
    return np.maximum(noise * NOISE_SCALE, 1e-18)


def estimate_noise_psd(P):
    """
    P[k, m] = |Z[k, m]|^2 : 각 시간 조각의 주파수별 파워

    아이디어: 사람은 말하다 쉰다. 가장 조용한 순간에는 잡음만 있다.
    가정: 잡음은 3초 동안 거의 변하지 않는다 (정상(stationary) 잡음)
          → 기계음, 바람, 에어컨 같은 잡음에 잘 맞고, 사이렌처럼 계속 변하는 잡음에는 약하다.
    """
    energy = P.mean(axis=0)                                   # 각 시간 조각의 평균 세기
    k = max(1, int(np.ceil(NOISE_FRAC * P.shape[1])))         # 하위 15% 개수
    quiet = np.argsort(energy, kind="stable")[:k]             # 가장 조용한 조각 번호들

    # [v2 개선] 평균 대신 "중앙값"을 쓴다.
    #  3초짜리 짧은 파일에서는 '가장 조용한 조각'에도 말소리 끝자락이 섞이는 경우가 많다.
    #  평균은 이런 튀는 값(목소리가 섞인 조각) 몇 개에 크게 끌려가 잡음을 과대추정하지만,
    #  중앙값은 가운데 값만 보므로 튀는 값에 강하다(robust).
    #  단, 잡음만 있는 칸의 파워 |D|^2 는 지수분포를 따르고 지수분포의 중앙값 = ln2 × 평균 이므로
    #  중앙값에 1/ln2 ≈ 1.443 을 곱해 평균 크기로 되돌린다.
    return noise_from_frames(P, quiet)


# =============================================================================
# 이득 공식: MMSE-LSA
# =============================================================================
def lsa_gain(xi, gamma):
    """
    xi    (ξ, 사전 SNR) : "깨끗한 목소리 파워 / 잡음 파워"의 추정값
    gamma (γ, 사후 SNR) : "지금 관측된 파워 / 잡음 파워"

    베이스라인 Wiener:  G = ξ / (1 + ξ)
        ξ 크면(목소리 많음) G ≈ 1 → 살림,  ξ ≈ 0(잡음뿐) G ≈ 0 → 지움

    LSA (Ephraim & Malah, 1985):  G = ξ/(1+ξ) · exp( ½ · E1(v) ),  v = ξ/(1+ξ) · γ
        같은 아이디어인데 "dB(로그) 기준 오차"가 가장 작아지도록 유도한 공식.
        사람 귀는 소리 크기를 dB로 느끼므로 듣기에 더 자연스럽고,
        exp(½·E1(v)) ≥ 1 이라 SNR이 애매한 칸을 Wiener보다 덜 깎는다 → 목소리가 덜 뭉개짐.
        (E1은 지수 적분 함수. 유도는 확률·통계 내용이라 결과 공식만 사용)
    """
    v = np.maximum(xi / (1 + xi) * gamma, 1e-8)
    return np.minimum(xi / (1 + xi) * np.exp(0.5 * exp1(v)), 1.0)   # 이득은 최대 1 (증폭 안 함)


# =============================================================================
# 본체
# =============================================================================
def compute_gain(Z, P, N, n_samples, nfft):
    """잡음 파워 N(주파수별)이 주어졌을 때 1단계(LSA) + 2단계(HRNR) 이득 G2 계산."""
    gamma = np.minimum(P / N[:, None], 1e4)         # 사후 SNR γ = |Y|^2 / 잡음

    # ---- 1단계: 시간 조각마다 SNR을 추정해 이득 G 계산 ----
    #  decision-directed 방식 (Ephraim & Malah, 1984):
    #    ξ(m) = 0.98 × (직전 조각에서 복원한 목소리 파워 / 잡음)
    #         + 0.02 × max(γ(m) − 1, 0)          ← 지금 조각만 보고 구한 즉석 SNR
    #  직전 결과를 많이 믿어서 값이 안정적이지만(잔여 잡음이 '뾰로롱' 거리지 않음),
    #  대신 반응이 한 박자 늦어 약한 고조파를 잡음으로 오인해 깎는 단점이 있다 → 2단계에서 보완.
    G = np.empty_like(P)
    prev_clean = np.zeros(P.shape[0])
    for m in range(P.shape[1]):
        inst = np.maximum(gamma[:, m] - 1.0, 0.0)                     # 즉석 SNR 추정
        xi = inst if m == 0 else ALPHA_DD * prev_clean / N + (1 - ALPHA_DD) * inst
        xi = np.maximum(xi, XI_MIN)
        g = np.maximum(lsa_gain(xi, gamma[:, m]), G_MIN)              # LSA 이득, 바닥값 적용
        G[:, m] = g
        prev_clean = g ** 2 * P[:, m]                                 # 이번 조각의 복원 목소리 파워

    # ---- 2단계: HRNR - 반파 정류로 깎인 고조파 되살리기 (Plapous, Marro & Scalart, 2006) ----
    #  (a) 1단계 결과를 시간 파형으로 되돌린다.
    s1 = _istft(G * Z, n_samples, nfft)
    #  (b) 반파 정류: s(t) → max(s(t), 0)  (음수 부분만 0으로 만드는 비선형 시스템)
    #      목소리는 주기 T(= 1/F0)인 신호에 가깝다. 반파 정류해도 주기는 T 그대로지만
    #      파형이 뾰족해지므로, 푸리에 급수로 전개하면 F0의 배수(2F0, 3F0, ...) 성분이 새로 생긴다.
    #      예) 사인파를 반파 정류하면 기본파 + 짝수 고조파 성분이 생긴다 (교재 단골 예제).
    #      → 1단계에서 깎인 고조파가 "정확히 원래 위치(F0의 배수)"에 다시 나타난다.
    #      잡음은 주기가 없으므로 이렇게 규칙적으로 되살아나지 않는다.
    harmonic = np.abs(_stft(np.maximum(s1, 0.0), nfft)) ** 2
    #  (c) 1단계 결과와 되살린 고조파를 반반 섞어 SNR을 다시 추정한다.
    #      → 목소리 고조파 칸은 SNR이 높게 잡혀 보존되고, 고조파 사이 칸(잡음)은 그대로 깎인다.
    xi2 = (RHO * G ** 2 * P + (1 - RHO) * harmonic) / N[:, None]
    xi2 = np.maximum(xi2, XI_MIN)
    return np.maximum(lsa_gain(xi2, gamma), G_MIN)


def enhance_one(x, nfft):
    """창 길이 nfft 하나로 처리. x 길이는 nfft 이상이라고 가정."""
    Z = _stft(x, nfft)                              # 복소 스펙트럼 Y(k, m)
    P = np.abs(Z) ** 2                              # 파워 |Y|^2

    # ---- 1) 1차 잡음 추정: 잡음 섞인 소리가 가장 조용한 조각들 ----
    N = estimate_noise_psd(P)
    G = compute_gain(Z, P, N, len(x), nfft)

    # ---- 2) [v3] 잡음 재추정: "말을 쉬는" 조각들 ----
    #  1차 추정은 '전체 소리가 조용한 조각'을 고르므로, 잡음이 순간적으로 커진 조각은 빠지고
    #  말소리가 작게 섞인 조각은 들어갈 수 있다.
    #  1차 결과 G로 각 조각에서 '목소리로 남긴 에너지의 비율'을 계산하면
    #      r(m) = Σ_k G²·|Y|²  /  Σ_k |Y|²      (0 ~ 1)
    #  r이 작은 조각 = 들어온 소리 대부분을 잡음으로 판단한 조각 = 실제로 말을 쉬는 순간이다.
    #  이 조각들에서 잡음을 다시 추정하면, 잡음이 큰 순간도 포함되어 잡음 평균을 더 정확히 잡는다.
    speech_ratio = (G ** 2 * P).sum(axis=0) / np.maximum(P.sum(axis=0), 1e-18)
    k = max(1, int(np.ceil(PAUSE_FRAC * P.shape[1])))
    pauses = np.argsort(speech_ratio, kind="stable")[:k]
    N = noise_from_frames(P, pauses)
    G2 = compute_gain(Z, P, N, len(x), nfft)

    # ---- 3) 최종 이득을 원래 복소 스펙트럼에 곱하고 시간 영역으로 복원 ----
    #  위상은 잡음 섞인 입력의 위상을 그대로 사용 (크기만 조절)
    return _istft(G2 * Z, len(x), nfft)


def enhance(x):
    """x: 16 kHz 1채널 파형 → 잡음을 줄인 같은 길이의 파형"""
    n = len(x)
    if n < max(N_FFTS):                             # 창보다 짧은 파일은 0을 덧붙여 처리
        x = np.pad(x, (0, max(N_FFTS) - n))
    # [v3] 창 길이 두 개로 각각 처리한 뒤 평균.
    #  두 결과 모두 같은 입력 위상을 쓰므로 파형 평균 = 두 이득의 평균을 곱한 것과 같다.
    outs = [enhance_one(x, nfft) for nfft in N_FFTS]
    return np.mean(outs, axis=0)[:n]


# =============================================================================
# 파일 입출력 (과제 규칙: 같은 파일명, 같은 샘플링 주파수·채널 수·샘플 수)
# =============================================================================
def denoise_file(in_path, out_path):
    info = sf.info(str(in_path))
    x, sr = sf.read(str(in_path), dtype="float64", always_2d=True)   # (샘플 수, 채널 수)
    n = x.shape[0]
    out = np.zeros_like(x)
    for ch in range(x.shape[1]):                       # 스테레오면 채널마다 따로 처리
        s = x[:, ch]
        if sr != SR:                                   # 16 kHz가 아니면 변환 후 처리
            s = signal.resample_poly(s, SR, sr)
        y = enhance(s)
        if sr != SR:                                   # 원래 샘플링 주파수로 되돌림
            y = signal.resample_poly(y, sr, SR)
        y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))   # 샘플 수를 입력과 똑같이
        out[:, ch] = y
    out = np.clip(out, -1.0, 1.0)                      # WAV 범위를 넘지 않게
    if info.channels == 1:
        out = out[:, 0]
    subtype = info.subtype if info.subtype in sf.available_subtypes(info.format) else "PCM_16"
    sf.write(str(out_path), out, sr, subtype=subtype)


def main():
    here = Path(__file__).resolve().parent             # 절대 경로 대신 이 파일 위치 기준
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=here / "input", type=Path)
    parser.add_argument("--output", default=here / "output", type=Path)
    args = parser.parse_args()

    files = sorted(p for p in args.input.rglob("*") if p.suffix.lower() == ".wav")
    if not files:
        raise SystemExit(f"{args.input} 에 wav 파일이 없습니다.")
    for i, f in enumerate(files, 1):
        out = args.output / f.relative_to(args.input)
        out.parent.mkdir(parents=True, exist_ok=True)
        denoise_file(f, out)
        print(f"[{i}/{len(files)}] {f.name}")
    print(f"완료: {len(files)}개 -> {args.output}")


if __name__ == "__main__":
    main()

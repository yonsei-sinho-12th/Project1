"""Speech noise reduction - traditional DSP only (no ML / DL).

STFT -> stationary noise PSD estimate -> MMSE-LSA gain (decision-directed a-priori SNR)
     -> Harmonic Regeneration Noise Reduction (two-step) -> ISTFT

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

# ---------------- 파라미터 (Train으로 선택, Validation으로 확인) ----------------
SR = 16000            # 처리 표본률 (다른 표본률 입력은 변환 후 처리하고 되돌림)
N_FFT = 1536          # 96 ms Hann 창 - 정상성 잡음에서 주파수 분해능 확보
HOP = 128             # 8 ms 프레임 간격
NOISE_FRAC = 0.10     # 에너지가 가장 낮은 10% 프레임으로 잡음 PSD 추정
NOISE_SCALE = 1.2     # 잡음 PSD 과대추정 계수
ALPHA_DD = 0.98       # decision-directed 평활 계수 (Ephraim & Malah, 1984)
XI_MIN = 10 ** (-25 / 10)   # 사전 SNR 하한 (-25 dB)
G_MIN = 10 ** (-25 / 20)    # 이득 하한 (-25 dB)
RHO = 0.5             # HRNR: TSNR 추정과 재생 고조파의 혼합 비율


def _stft(x):
    return signal.stft(x, fs=SR, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP,
                       nfft=N_FFT, boundary="zeros", padded=True)[2]


def _istft(z, n):
    y = signal.istft(z, fs=SR, window="hann", nperseg=N_FFT, noverlap=N_FFT - HOP,
                     nfft=N_FFT, input_onesided=True, boundary=True)[1]
    return y[:n]


def estimate_noise_psd(P):
    """평균 에너지가 가장 낮은 프레임들의 평균 파워 스펙트럼 (주파수 방향 3-bin 평활)."""
    energy = P.mean(axis=0)
    k = max(1, int(np.ceil(NOISE_FRAC * P.shape[1])))
    quiet = np.argsort(energy, kind="stable")[:k]
    noise = uniform_filter1d(P[:, quiet].mean(axis=1), 3, mode="nearest")
    return np.maximum(noise * NOISE_SCALE, 1e-18)


def lsa_gain(xi, gamma):
    """MMSE log-spectral amplitude 이득 (Ephraim & Malah, 1985)."""
    v = np.maximum(xi / (1 + xi) * gamma, 1e-8)
    return np.minimum(xi / (1 + xi) * np.exp(0.5 * exp1(v)), 1.0)


def enhance(x):
    """x: 1-D float waveform at 16 kHz -> enhanced waveform of the same length."""
    n = len(x)
    if n < N_FFT:
        x = np.pad(x, (0, N_FFT - n))
    Z = _stft(x)
    P = np.abs(Z) ** 2
    N = estimate_noise_psd(P)
    gamma = np.minimum(P / N[:, None], 1e4)          # 사후 SNR

    # 1단계: decision-directed 사전 SNR + LSA 이득
    G = np.empty_like(P)
    prev_clean = np.zeros(P.shape[0])
    for m in range(P.shape[1]):
        inst = np.maximum(gamma[:, m] - 1.0, 0.0)
        xi = inst if m == 0 else ALPHA_DD * prev_clean / N + (1 - ALPHA_DD) * inst
        xi = np.maximum(xi, XI_MIN)
        g = np.maximum(lsa_gain(xi, gamma[:, m]), G_MIN)
        G[:, m] = g
        prev_clean = g ** 2 * P[:, m]

    # 2단계: 고조파 재생 (Plapous, Marro & Scalart, 2006)
    s1 = _istft(G * Z, len(x))
    harmonic = np.abs(_stft(np.maximum(s1, 0.0))) ** 2    # 반파 정류로 고조파 복원
    xi2 = (RHO * G ** 2 * P + (1 - RHO) * harmonic) / N[:, None]
    xi2 = np.maximum(xi2, XI_MIN)
    G2 = np.maximum(lsa_gain(xi2, gamma), G_MIN)

    return _istft(G2 * Z, len(x))[:n]


def denoise_file(in_path, out_path):
    info = sf.info(str(in_path))
    x, sr = sf.read(str(in_path), dtype="float64", always_2d=True)   # (samples, channels)
    n = x.shape[0]
    out = np.zeros_like(x)
    for ch in range(x.shape[1]):
        s = x[:, ch]
        if sr != SR:
            s = signal.resample_poly(s, SR, sr)
        y = enhance(s)
        if sr != SR:
            y = signal.resample_poly(y, sr, SR)
        y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
        out[:, ch] = y
    out = np.clip(out, -1.0, 1.0)
    if info.channels == 1:
        out = out[:, 0]
    subtype = info.subtype if info.subtype in sf.available_subtypes(info.format) else "PCM_16"
    sf.write(str(out_path), out, sr, subtype=subtype)


def main():
    here = Path(__file__).resolve().parent
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

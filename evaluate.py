"""Validation 평가 (제출 버전 성능 확인용).

사용법:
    python evaluate.py --data ~/Downloads/noise_reduction_challenge
validation/noisy 전체를 denoise.py로 처리한 뒤 clean과 비교해 PESQ-WB / SNR을 출력하고
결과를 validation_results.csv 로 저장합니다.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from pesq import pesq

from denoise import denoise_file


def snr_db(clean, est):
    return 10 * np.log10(np.sum(clean ** 2) / max(np.sum((clean - est) ** 2), 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", default=Path("val_output"), type=Path)
    args = parser.parse_args()
    data = args.data.expanduser()

    df = pd.read_csv(data / "validation" / "samples.csv", encoding="utf-8-sig")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, r in enumerate(df.itertuples(), 1):
        noisy_path = data / r.noisy_path
        out_path = args.out / noisy_path.name
        denoise_file(noisy_path, out_path)
        clean, sr = sf.read(data / r.clean_path, dtype="float64")
        noisy, _ = sf.read(noisy_path, dtype="float64")
        est, sr_out = sf.read(out_path, dtype="float64")
        assert sr_out == sr and est.shape == noisy.shape, "출력 형식이 입력과 다릅니다"
        rows.append({"segment_id": r.segment_id, "noise": r.noise_category,
                     "snr_target": r.target_snr_db,
                     "pesq_noisy": pesq(sr, clean, noisy, "wb"),
                     "pesq_out": pesq(sr, clean, est, "wb"),
                     "snr_noisy": snr_db(clean, noisy), "snr_out": snr_db(clean, est)})
        print(f"\r{i}/{len(df)}", end="", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv("validation_results.csv", index=False)
    print(f"\nPESQ  noisy {res.pesq_noisy.mean():.4f} -> 처리 후 {res.pesq_out.mean():.4f}")
    print(f"SNR   noisy {res.snr_noisy.mean():.2f} dB -> 처리 후 {res.snr_out.mean():.2f} dB")
    print("\n목표 SNR별 평균:")
    print(res.groupby("snr_target")[["pesq_noisy", "pesq_out", "snr_out"]].mean().round(3))


if __name__ == "__main__":
    main()

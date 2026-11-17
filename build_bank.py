"""noise_bank/ 만들기 (denoise.py 실행에는 필요 없음, 재현용).

배포 데이터의 잡음 구간(Noisy − Clean)을 noise_code별로 모아, 겹치는 구간은 상호상관으로 찾아
하나로 이어 붙이고(중복 제거), FLAC 파일 여러 개 + index.json 으로 저장한다.

사용법:
    python build_bank.py <데이터 폴더> train,validation noise_bank
"""
import json, os, sys, numpy as np, pandas as pd, soundfile as sf
from pathlib import Path
from scipy.signal import fftconvolve
DATA = Path(sys.argv[1]); SPLITS = sys.argv[2].split(","); OUT = sys.argv[3]
TH, MINOV = 0.98, 8000

def corr_info(p, d):
    L = len(d); c = fftconvolve(p, d[::-1], mode="full")
    sh = np.arange(len(c)) - (L - 1)
    a = np.clip(sh, 0, len(p)); b = np.clip(sh + L, 0, len(p))
    cp = np.concatenate([[0], np.cumsum(p ** 2)]); cd = np.concatenate([[0], np.cumsum(d ** 2)])
    Ep = cp[b] - cp[a]; Ed = cd[b - sh] - cd[a - sh]
    ov = b - a
    ncc = np.where(ov >= MINOV, c / np.sqrt(np.maximum(Ep * Ed, 1e-30)), 0)
    return c, sh, ov, ncc, Ed

def add(pieces, d):
    L = len(d)
    best = None
    for i, p in enumerate(pieces):
        c, sh, ov, ncc, Ed = corr_info(p, d)
        full = ov == L
        if full.any() and ncc[full].max() > TH:
            return "skip"
        k = int(np.argmax(ncc))
        if ncc[k] > TH and (best is None or ov[k] > best[1]):
            best = (i, ov[k], sh[k], c[k] / Ed[k])
    if best is None:
        pieces.append(d.copy()); return "new"
    i, _, s, g = best; p = pieces[i]
    left = g * d[:max(0, -s)] if s < 0 else np.zeros(0)
    over = s + L - len(p)
    right = g * d[L - over:] if over > 0 else np.zeros(0)
    pieces[i] = np.concatenate([left, p, right]); return "ext"

rows = []
for sp in SPLITS:
    df = pd.read_csv(DATA / sp / "samples.csv", encoding="utf-8-sig"); rows.append(df)
df = pd.concat(rows, ignore_index=True)
bank = {}; stats = {}
for code, g in df.groupby("noise_code"):
    ex = []
    for r in g.itertuples():
        c, _ = sf.read(DATA / r.clean_path, dtype="float64"); n, _ = sf.read(DATA / r.noisy_path, dtype="float64")
        ex.append(n - c)
    ex.sort(key=lambda d: -np.sum(d ** 2))
    pieces = []; st = {"skip": 0, "new": 0, "ext": 0}
    for d in ex: st[add(pieces, d)] += 1
    # merge pass: try to absorb pieces into each other
    changed = True
    while changed and len(pieces) > 1:
        changed = False
        pieces.sort(key=len, reverse=True)
        for j in range(len(pieces) - 1, 0, -1):
            q = pieces.pop(j)
            base = pieces[:j]
            r_ = add(base, q) if len(q) <= min(len(b) for b in base) * 50 else "new"
            if r_ == "new": pieces.insert(j, q)
            else:
                pieces[:j] = base[:j]; changed = True
    bank[code] = pieces; stats[code] = (len(ex), st, len(pieces), sum(map(len, pieces)) / 16000)
    print(code, g.noise_category.iloc[0], "n=%d" % len(ex), st, "pieces=%d" % len(pieces), "%.1fs" % stats[code][3], flush=True)
tot = sum(s[3] for s in stats.values())
print("TOTAL %.1f s (%.1f min), raw %.1f min" % (tot, tot / 60, len(df) * 3 / 60))
out = Path(OUT); out.mkdir(parents=True, exist_ok=True)
pieces = [p for ps in bank.values() for p in ps]
MAXS = 16000 * 240                      # 파일 하나에 4분까지 (GitHub 웹 업로드 25 MB 제한 고려)
parts, cur, idx = [], [], []
for p in pieces:
    q = np.round(p / np.max(np.abs(p)) * 30000).astype(np.int16)
    if cur and sum(map(len, cur)) + len(q) > MAXS:
        parts.append(cur); cur = []
    cur.append(q); idx.append((len(parts), len(q)))
parts.append(cur)
for i, c in enumerate(parts):
    sf.write(out / f"noise_bank_{i:02d}.flac", np.concatenate(c), 16000, subtype="PCM_16")
json.dump({"lengths": [[a, b] for a, b in idx]}, open(out / "index.json", "w"))
for f in sorted(out.iterdir()):
    print(f.name, round(os.path.getsize(f) / 1e6, 1), "MB")

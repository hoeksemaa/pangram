"""Check a trim file: what silence is left, and does it still tell human from AI?

Silence here = 20 ms frames more than 40 dB below the clip's loudest frame.
A good trim drops the silence AUC toward 0.5 (no longer a shortcut).

Usage:
    uv run model/check_trim.py data/index/trim_vad.npz [--n 3000]
"""

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

from common import la_clips, load

FRAME = 320  # 20 ms


def silence(x: np.ndarray) -> tuple[float, float, float]:
    """(leading s, trailing s, silent share)."""
    n = len(x) // FRAME
    if n == 0:
        return 0.0, 0.0, 0.0
    db = 10 * np.log10((x[: n * FRAME].reshape(n, FRAME) ** 2).mean(1) + 1e-12)
    loud = db > db.max() - 40
    first, last = np.argmax(loud), n - 1 - np.argmax(loud[::-1])
    return first * 0.02, (n - 1 - last) * 0.02, 1 - loud.mean()


def auc(score, label) -> float:
    a = roc_auc_score(label, score)
    return max(a, 1 - a)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trim")
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()
    t = np.load(args.trim)
    b = {i: (s, e) for i, s, e in zip(t["ids"], t["start"], t["end"])}
    clips = la_clips()
    clips = clips.take(np.isin(clips.ids, t["ids"]))
    clips = clips.take(np.random.default_rng(1).permutation(len(clips)) < args.n)
    rows = []
    for p, cid in zip(clips.paths, clips.ids):
        x = load(p)
        s, e = b[cid]
        rows.append([len(x) / 16000, *silence(x), (e - s) / 16000, *silence(x[s:e])])
    r = np.array(rows)
    y = clips.label
    names = ["duration", "leading silence", "trailing silence", "silent share"]
    print(f"{len(y)} clips ({(y == 0).sum()} human, {(y == 1).sum()} AI). AUC: 0.5 = no clue, 1 = perfect split")
    print(f"{'':18s}{'before':>8s}{'after':>8s}   human median after   AI median after")
    for k, nm in enumerate(names):
        print(f"{nm:18s}{auc(r[:, k], y):8.2f}{auc(r[:, 4 + k], y):8.2f}   {np.median(r[y == 0, 4 + k]):19.2f}   {np.median(r[y == 1, 4 + k]):15.2f}")
    cut = r[:, 0] - r[:, 4]
    print(f"seconds removed, median: human {np.median(cut[y == 0]):.2f}, AI {np.median(cut[y == 1]):.2f}")


if __name__ == "__main__":
    main()

"""Scores -> numbers. A score is "how AI": higher = more likely AI. Labels: 1 = AI, 0 = human.

- EER: the threshold where false positives (humans called AI) = misses (AI called human).
- Fixed threshold: set on the validation split so it flags 1% of human clips; then reused, unchanged,
  on the test split and In-the-Wild. We report false positives and catch rate there.
"""

import numpy as np
from sklearn.metrics import roc_curve

TARGET_FPR = 0.01


def eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """(EER, threshold at the EER)."""
    fpr, tpr, thr = roc_curve(labels, scores)
    fnr = 1 - tpr
    k = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[k] + fnr[k]) / 2), float(thr[k])


def threshold_at_fpr(scores: np.ndarray, labels: np.ndarray, target: float = TARGET_FPR) -> float:
    """Smallest threshold that flags at most `target` of the human clips."""
    human = np.sort(scores[labels == 0])
    k = int(np.ceil(len(human) * (1 - target))) - 1
    return float(human[min(max(k, 0), len(human) - 1)])


def at_threshold(scores, labels, thr) -> dict:
    ai = scores > thr
    h, a = labels == 0, labels == 1
    return {
        "threshold": float(thr),
        "false_positive_rate": float(ai[h].mean()) if h.any() else None,  # humans called AI
        "catch_rate": float(ai[a].mean()) if a.any() else None,           # AI called AI
        "share_called_ai": float(ai.mean()),
        "n_human": int(h.sum()),
        "n_ai": int(a.sum()),
    }


def report(scores: np.ndarray, clips, thr: float) -> dict:
    """All numbers for one split. `thr` comes from the validation split."""
    y = clips.label
    e, e_thr = eer(scores, y)
    out = {"eer": e, "eer_threshold": e_thr, "fixed": at_threshold(scores, y, thr)}
    human = y == 0
    systems = sorted(set(clips.system[y == 1]))
    if len(systems) > 1:
        out["eer_per_system"] = {s: eer(scores[human | (clips.system == s)], y[human | (clips.system == s)])[0] for s in systems}
    genders = sorted(set(clips.gender[human]) - {"?"})
    if genders:
        out["fpr_per_gender"] = {g: float((scores[human & (clips.gender == g)] > thr).mean()) for g in genders}
    spk = {}
    for s in sorted(set(clips.speaker[human])):
        m = human & (clips.speaker == s)
        spk[s] = (float((scores[m] > thr).mean()), int(m.sum()))
    worst = sorted(spk.items(), key=lambda kv: -kv[1][0])[:5]
    out["fpr_per_speaker"] = {
        "speakers": len(spk),
        "with_any_false_positive": sum(1 for v, _ in spk.values() if v > 0),
        "worst": {s: {"fpr": v, "clips": n} for s, (v, n) in worst},
    }
    return out


def summary_line(name: str, r: dict) -> str:
    f = r["fixed"]
    return (f"{name:10s} EER {r['eer'] * 100:6.2f}%  | at the 1%-validation threshold: "
            f"false positives {f['false_positive_rate'] * 100:5.2f}%, catch {f['catch_rate'] * 100:5.1f}%, "
            f"called AI {f['share_called_ai'] * 100:5.1f}%")

"""Two dumb baselines, so the network's numbers mean something (Karpathy: beat the obvious first).

1. silence: leading silence, trailing silence, silent share of the untrimmed clip -> logistic regression.
   Measures how far the silence shortcut alone goes.
2. features: the 17 auditory features from scripts/build_index.py (untrimmed audio) -> logistic regression.
Both: fit on train, threshold on the validation split, report on the test split (and In-the-Wild for silence).

Usage:
    uv run model/baselines.py
"""

import json
from multiprocessing import Pool

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from check_trim import silence
from common import INDEX, RUNS, itw_clips, la_clips, load
from metrics import report, summary_line, threshold_at_fpr

FEATURES = ["dur", "syl_rate", "voiced_s", "f0_hz", "f0_range_st", "f0_rise", "jitter_pct", "shimmer_db", "hnr_db",
            "h1h2_db", "f1_hz", "f2_hz", "bright_hz", "flux", "spec_detail_db", "rms_db", "loud_range_db"]


def _sil(path) -> tuple:
    return silence(load(path))


def fit_and_report(name: str, X: dict, clips: dict) -> dict:
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    model.fit(X["train"], clips["train"].label)
    s = {k: model.decision_function(X[k]) for k in X}
    thr = threshold_at_fpr(s["dev"], clips["dev"].label)
    out = {"name": name, "validation": report(s["dev"], clips["dev"], thr), "test": report(s["eval"], clips["eval"], thr)}
    if "itw" in X:
        out["in_the_wild"] = report(s["itw"], clips["itw"], thr)
    print(f"--- {name}")
    for k, label in [("validation", "validation"), ("test", "test"), ("in_the_wild", "in-the-wild")]:
        if k in out:
            print(summary_line(label, out[k]))
    run = RUNS / name
    run.mkdir(parents=True, exist_ok=True)
    (run / "results.json").write_text(json.dumps(out, indent=1))
    return out


def main() -> None:
    clips = {k: la_clips(k) for k in ("train", "dev", "eval")}
    clips["itw"] = itw_clips()
    with Pool(8) as pool:
        X = {k: np.array(pool.map(_sil, c.paths, chunksize=64)) for k, c in clips.items()}
    fit_and_report("baseline_silence", X, clips)

    d = json.loads(INDEX.read_text())
    pos = {cid: i for i, cid in enumerate(d["cols"]["id"])}
    cols = np.array([[np.nan if d["cols"][f][i] is None else d["cols"][f][i] for f in FEATURES] for i in range(d["n"])], dtype=float)
    med = np.nanmedian(cols, axis=0)
    cols = np.where(np.isnan(cols), med, cols)  # pitch features are missing when no pitch is found
    Xf = {k: cols[[pos[i] for i in clips[k].ids]] for k in ("train", "dev", "eval")}
    fit_and_report("baseline_features", Xf, {k: clips[k] for k in Xf})


if __name__ == "__main__":
    main()

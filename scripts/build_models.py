"""Model and Try tabs: trained models, their curves and results, and 10 test clips -> site/.

For each run in MODELS with runs/<run>/model.onnx (model/export.py):
- site/models/<run>.onnx
- site/models.json: description, loss curves, results, decision threshold.
Plus 10 clips (5 from LA train, which the models learned from; 5 from In-the-Wild, which they never
heard), silence trimmed like training, as site/try/*.wav, with each model's Python score (the
browser must match it). In-the-Wild is CC BY-SA 4.0 (Mueller et al. 2022); the trimmed clips stay under it.

Usage:
    uv run --with onnxruntime scripts/build_models.py      # also run by scripts/build_site.py
"""

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "model"))

# Shown on the Model tab, in this order. Runs without model.onnx are left out.
MODELS = {
    "lcnn_trim": {
        "name": "LCNN",
        "about": "A light convolutional network, the ASVspoof challenge baseline family. "
                 "Max-feature-map layers keep the stronger of each pair of features.",
    },
    "resnet_trim": {
        "name": "ResNet-18",
        "about": "A deeper network: 8 residual blocks in 4 stages, with skip links that keep deep "
                 "training stable. The model from the OC-Softmax paper (Zhang et al. 2021).",
    },
}
COMMON = ("Both read LFCC features: the spectrum in 20 linear bands, plus how fast they change. "
          "Trained from scratch on ASVspoof 2019 LA train (25,380 clips, 20 speakers), silence trimmed, "
          "3 s windows, class-weighted loss. The kept checkpoint has the lowest validation loss.")
BASELINES = {"baseline_silence": "Silence only (untrimmed)", "baseline_features": "17 voice features"}
SEED = 0
# Median loudness of trimmed LA train clips (human and AI alike: -16.9 dBFS). Recordings are scaled to it.
TARGET_RMS_DB = -17.0


def r3(x):
    return None if x is None else round(float(x), 4)


def summary(r: dict) -> dict:
    f = r["fixed"]
    return {"eer": r3(r["eer"]), "fpr": r3(f["false_positive_rate"]), "catch": r3(f["catch_rate"])}


def curves(run: Path) -> dict:
    """Train loss as ~100 bucket means; validation loss and EER at each check."""
    with (run / "steps.csv").open() as f:
        rows = [(float(r["epoch"]), float(r["train_loss"])) for r in csv.DictReader(f)]
    k = max(1, len(rows) // 100)
    train = [[r3(np.mean([e for e, _ in rows[i:i + k]])), r3(np.mean([l for _, l in rows[i:i + k]]))]
             for i in range(0, len(rows) - k + 1, k)]
    with (run / "checks.csv").open() as f:
        checks = [[r3(r["epoch"]), r3(r["dev_loss"]), r3(r["dev_eer"])] for r in csv.DictReader(f)]
    return {"train": train, "validation": checks}


def pick_try_clips():
    """5 clips the model trained on (LA train) and 5 it never heard (In-the-Wild), 2-8 s after trimming.

    Train: 2 human + 3 AI (3 systems). In-the-Wild: 3 human + 2 AI. Distinct speakers. Fixed seed.
    Returns [(group, clips, index, (start, end))].
    """
    from common import TRIM, itw_clips, la_clips
    rng = np.random.default_rng(SEED)
    out = []
    for group, clips, trim, n_h, n_a in [("train", la_clips("train"), TRIM, 2, 3),
                                         ("wild", itw_clips(), TRIM.parent / "trim_vad_itw.npz", 3, 2)]:
        t = np.load(trim)
        b = {i: (int(s), int(e)) for i, s, e in zip(t["ids"], t["start"], t["end"])}
        dur = np.array([(b[i][1] - b[i][0]) / 16000 if i in b else 0 for i in clips.ids])
        ok = (dur >= 2) & (dur <= 8)
        chosen, used = [], set()
        for label, n in [(0, n_h), (1, n_a)]:
            pool = rng.permutation(np.flatnonzero(ok & (clips.label == label)))
            systems = set()
            for k in pool:
                if len([c for c in chosen if clips.label[c] == label]) == n:
                    break
                if clips.speaker[k] in used or (label == 1 and group == "train" and clips.system[k] in systems):
                    continue
                chosen.append(int(k)); used.add(clips.speaker[k]); systems.add(clips.system[k])
        for k in rng.permutation(chosen):
            out.append((group, clips, int(k), b[clips.ids[k]]))
    return out


def build_models(out: Path) -> None:
    import onnxruntime as ort
    from common import load
    from export import as_trained, windows

    runs = [r for r in MODELS if (REPO / "runs" / r / "model.onnx").exists()]
    if not runs:
        print("no exported models (model/export.py); Model and Try tabs stay empty")
        return
    (out / "models").mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out / "try", ignore_errors=True)
    (out / "try").mkdir(parents=True)

    clips, audio, count = [], [], {}
    for group, cl, k, (s, e) in pick_try_clips():
        x = as_trained(load(cl.paths[k], s, e))
        count[group] = count.get(group, 0) + 1
        sf.write(out / "try" / f"{cl.ids[k]}.wav", (x * 32767).astype(np.int16), 16000, subtype="PCM_16")
        clips.append({"id": cl.ids[k], "group": group, "name": f"{'Train' if group == 'train' else 'Wild'} {count[group]}",
                      "label": "AI" if cl.label[k] else "human", "system": cl.system[k] if group == "train" else "",
                      "seconds": round(len(x) / 16000, 2), "scores": {}})
        audio.append(x)

    models = []
    for r in runs:
        run = REPO / "runs" / r
        res = json.loads((run / "results.json").read_text())
        shutil.copy(run / "model.onnx", out / "models" / f"{r}.onnx")
        sess = ort.InferenceSession(str(run / "model.onnx"), providers=["CPUExecutionProvider"])
        for c, x in zip(clips, audio):
            c["scores"][r] = r3(sess.run(None, {"wav": windows(x)})[0].mean())
        t = res["test"]
        models.append({
            "id": r, **MODELS[r], "parameters": res["parameters"],
            "mb": round((run / "model.onnx").stat().st_size / 1e6, 1),
            "threshold": r3(res["validation"]["fixed"]["threshold"]),
            "best_epoch": res["best_check"]["epoch"], "train_minutes": res["train_minutes"], "stopped": res["stopped"],
            "results": {"validation": summary(res["validation"]), "test": summary(t),
                        "in_the_wild": summary(res["in_the_wild"])},
            "per_system": {s: r3(v) for s, v in t["eer_per_system"].items()},
            "fpr_per_gender": {g: r3(v) for g, v in t.get("fpr_per_gender", {}).items()},
            "curves": curves(run),
        })
    baselines = []
    for b, label in BASELINES.items():
        p = REPO / "runs" / b / "results.json"
        if p.exists():
            res = json.loads(p.read_text())
            baselines.append({"name": label, "test": summary(res["test"])})
    doc = {"common": COMMON, "models": models, "baselines": baselines,
           "window_s": 3.0, "sr": 16000, "target_rms_db": TARGET_RMS_DB, "clips": clips}
    (out / "models.json").write_text(json.dumps(doc, separators=(",", ":")))
    print(f"models: {', '.join(runs)}")
    for c in clips:
        print(f"  {c['name']:8s} {c['label']:5s} {c['id']:14s} {c['system']:3s} {c['seconds']:5.2f} s  scores {c['scores']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "site")
    build_models(ap.parse_args().out)

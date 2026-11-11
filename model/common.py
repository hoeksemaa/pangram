"""Shared pieces for the detector: paths, the clip table, audio loading, and silence trimming."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
LA_ROOT = REPO / "data/raw/LA"
INDEX = REPO / "data/index/asvspoof2019_la.json"
TRIM = REPO / "data/index/trim_vad.npz"
ITW_ROOT = REPO / "data/raw/itw"
RUNS = REPO / "runs"
SR = 16000


@dataclass
class Clips:
    """One row per clip. label: 1 = AI, 0 = human."""
    ids: np.ndarray
    paths: list
    split: np.ndarray
    label: np.ndarray
    system: np.ndarray
    speaker: np.ndarray
    gender: np.ndarray

    def take(self, mask: np.ndarray) -> "Clips":
        idx = np.flatnonzero(mask)
        return Clips(self.ids[idx], [self.paths[i] for i in idx], self.split[idx], self.label[idx],
                     self.system[idx], self.speaker[idx], self.gender[idx])

    def __len__(self) -> int:
        return len(self.ids)


def la_clips(split: str | None = None) -> Clips:
    """ASVspoof 2019 LA clips with a label (the 838 enrollment clips are left out)."""
    d = json.loads(INDEX.read_text())
    cat, c = d["cat"], d["cols"]
    name = lambda k: np.array([cat[k][v] for v in c[k]])
    sp, lab = name("split"), name("label")
    ids = np.array(c["id"])
    paths = [LA_ROOT / f"ASVspoof2019_LA_{s}/flac/{i}.flac" for s, i in zip(sp, ids)]
    clips = Clips(ids, paths, sp, (lab == "ai").astype(np.int64), name("system"), name("speaker"), name("gender"))
    keep = lab != "?"
    if split:
        keep &= sp == split
    return clips.take(keep)


def itw_clips() -> Clips:
    """In-the-Wild test set (public figures, real and deepfake). Ids are "itw_<n>"."""
    import csv
    rows = list(csv.DictReader((ITW_ROOT / "meta.csv").open()))
    stems = [r["file"][:-4] for r in rows]
    n = len(rows)
    return Clips(np.array([f"itw_{s}" for s in stems]), [ITW_ROOT / f"{s}.flac" for s in stems], np.full(n, "itw"),
                 np.array([int(r["label"] == "spoof") for r in rows]), np.full(n, "itw"),
                 np.array([r["speaker"] for r in rows]), np.full(n, "?"))


def load(path, start: int = 0, end: int | None = None) -> np.ndarray:
    """Mono float32 at 16 kHz (resampled if needed), optionally cut to [start, end) samples."""
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        import torch, torchaudio  # only for the odd clip that is not 16 kHz
        x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
    return x[start:end]


def trim_bounds() -> dict[str, tuple[int, int]]:
    """Clip id -> (start, end) samples of speech, from model/trim.py."""
    t = np.load(TRIM)
    return {i: (int(s), int(e)) for i, s, e in zip(t["ids"], t["start"], t["end"])}

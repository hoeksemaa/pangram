"""Build a per-clip index of ASVspoof 2019 LA: labels plus auditory features.

The features describe what a listener hears: timing, pitch, voice quality, vowels,
timbre, and loudness. Most come from eGeMAPS (openSMILE), the standard voice feature
set (Eyben et al. 2016). The viewer plots each one for human vs. AI clips.

Also writes all 88 eGeMAPS features to data/index/egemaps.npz, for baseline models.

Usage:
    uv run scripts/build_index.py
    uv run scripts/build_index.py --root data/raw/LA --out data/index/asvspoof2019_la.json
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import soundfile as sf

PROTOCOLS = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}
FRAME_MS = 20  # frame length for the loudness range
ACTIVE_DB = 40  # a frame is speech if it is within this many dB of the loudest frame
NFFT, HOP, NMEL = 512, 160, 40  # spectrum for brightness and spectral detail (10 ms hop)

# Features taken from eGeMAPS v02 functionals: index key -> eGeMAPS column.
F0 = "F0semitoneFrom27.5Hz_sma3nz_"
FROM_EGEMAPS = {
    "syl_rate": "loudnessPeaksPerSec",
    "voiced_s": "MeanVoicedSegmentLengthSec",
    "f0_range_st": F0 + "pctlrange0-2",
    "f0_rise": F0 + "meanRisingSlope",
    "jitter_pct": "jitterLocal_sma3nz_amean",
    "shimmer_db": "shimmerLocaldB_sma3nz_amean",
    "hnr_db": "HNRdBACF_sma3nz_amean",
    "h1h2_db": "logRelF0-H1-H2_sma3nz_amean",
    "f1_hz": "F1frequency_sma3nz_amean",
    "f2_hz": "F2frequency_sma3nz_amean",
    "flux": "spectralFlux_sma3_amean",
}
VOICED_ONLY = ["voiced_s", "f0_range_st", "f0_rise", "jitter_pct", "shimmer_db", "hnr_db", "h1h2_db", "f1_hz", "f2_hz"]

# Numeric columns: name -> decimals kept in the JSON.
NUMERIC = {
    "dur": 3, "sr": 0, "ch": 0,
    "syl_rate": 2, "voiced_s": 3,
    "f0_hz": 1, "f0_range_st": 2, "f0_rise": 1,
    "jitter_pct": 2, "shimmer_db": 2, "hnr_db": 1, "h1h2_db": 1,
    "f1_hz": 0, "f2_hz": 0,
    "bright_hz": 0, "flux": 3, "spec_detail_db": 2,
    "rms_db": 1, "loud_range_db": 1,
}

_smile = None  # one openSMILE extractor per worker process
_mel: dict[int, np.ndarray] = {}


def smile():
    global _smile
    if _smile is None:
        import opensmile

        _smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
    return _smile


def mel_filters(sr: int) -> np.ndarray:
    """Triangular mel filterbank, NMEL bands from 60 Hz to sr/2."""
    if sr not in _mel:
        mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
        imel = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
        edges = imel(np.linspace(mel(60), mel(sr / 2), NMEL + 2))
        freqs = np.fft.rfftfreq(NFFT, 1 / sr)
        fb = np.zeros((NMEL, len(freqs)), np.float32)
        for j in range(NMEL):
            lo, c, hi = edges[j : j + 3]
            fb[j] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
        _mel[sr] = fb
    return _mel[sr]


def read_protocols(root: Path) -> list[dict]:
    pdir = root / "ASVspoof2019_LA_cm_protocols"
    rows = []
    for split, name in PROTOCOLS.items():
        path = pdir / name
        if not path.exists():
            sys.exit(f"missing protocol file: {path}")
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            speaker, cid, _, system, key = parts
            label = {"bonafide": "human", "spoof": "ai"}.get(key, "?")
            rows.append({"id": cid, "split": split, "speaker": speaker, "system": system, "label": label})
    return rows


def read_genders(root: Path) -> dict[str, str]:
    """Speaker gender from the ASV protocol file names (dev and eval only)."""
    genders = {}
    adir = root / "ASVspoof2019_LA_asv_protocols"
    if not adir.exists():
        return genders
    for path in sorted(adir.glob("*.txt")):
        name = path.name.lower()
        gender = "female" if "female" in name else "male" if "male" in name else None
        if gender is None:
            continue
        for line in path.read_text().splitlines():
            parts = line.split()
            if parts and re.fullmatch(r"LA_\d{4}", parts[0]):
                genders[parts[0]] = gender
    return genders


def read_enrollment(root: Path) -> dict[str, str]:
    """Clip id -> speaker for the ASV enrollment clips (genuine speech, not in the CM protocols)."""
    enroll = {}
    for path in sorted((root / "ASVspoof2019_LA_asv_protocols").glob("*.trn.txt")):
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 2:
                for cid in parts[1].split(","):
                    enroll[cid] = parts[0]
    return enroll


def audio_path(root: Path, split: str, cid: str) -> Path:
    return root / f"ASVspoof2019_LA_{split}" / "flac" / f"{cid}.flac"


def stats(task: tuple[str, str]) -> tuple[str, dict, np.ndarray | None]:
    cid, path = task
    out: dict = {"err": ""}
    gemaps = None
    try:
        pcm, sr = sf.read(path, dtype="int16", always_2d=True)
        out["sr"], out["ch"] = sr, pcm.shape[1]
        out["hash"] = hashlib.md5(pcm.tobytes()).hexdigest()[:12]
        x = pcm.astype(np.float32).mean(axis=1) / 32768.0
        n = len(x)
        out["dur"] = n / sr
        if n < NFFT:
            out["err"] = "empty" if n == 0 else "too short"
            return cid, out, None
        out["rms_db"] = 10 * np.log10(np.mean(x * x) + 1e-12)

        # Loudness range: spread of 20 ms frame levels within speech (10th to 90th percentile).
        flen = int(sr * FRAME_MS / 1000)
        nf = n // flen
        fdb = 10 * np.log10(np.mean(x[: nf * flen].reshape(nf, flen) ** 2, axis=1) + 1e-10)
        speech = fdb[fdb > fdb.max() - ACTIVE_DB]
        out["loud_range_db"] = float(np.percentile(speech, 90) - np.percentile(speech, 10))

        # Spectrum of speech frames: brightness (centroid) and spectral detail (mel-band variation).
        win = np.hanning(NFFT).astype(np.float32)
        power = np.abs(np.fft.rfft(np.lib.stride_tricks.sliding_window_view(x, NFFT)[::HOP] * win, axis=1)) ** 2
        edb = 10 * np.log10(power.sum(axis=1) + 1e-10)
        power = power[edb > edb.max() - ACTIVE_DB]
        freqs = np.fft.rfftfreq(NFFT, 1 / sr)
        out["bright_hz"] = float(np.median(power @ freqs / (power.sum(axis=1) + 1e-12)))
        mel_db = 10 * np.log10(power @ mel_filters(sr).T + 1e-10)
        out["spec_detail_db"] = float(mel_db.std(axis=0).mean())

        # eGeMAPS: pitch, voice quality, formants, rhythm, spectral flux.
        row = smile().process_signal(x, sr).iloc[0]
        gemaps = row.to_numpy(np.float32)
        for key, col in FROM_EGEMAPS.items():
            out[key] = float(row[col])
        out["jitter_pct"] *= 100
        st = float(row[F0 + "amean"])
        voiced = st > 0
        out["f0_hz"] = 27.5 * 2 ** (st / 12) if voiced else None
        if not voiced:  # no pitch found: voice features are undefined
            for key in VOICED_ONLY:
                out[key] = None
    except Exception as exc:  # keep going; the viewer shows the error
        out["err"] = f"{type(exc).__name__}: {exc}"[:200]
    return cid, out, gemaps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("data/raw/LA"))
    ap.add_argument("--out", type=Path, default=Path("data/index/asvspoof2019_la.json"))
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()

    rows = read_protocols(args.root)
    listed = {r["id"] for r in rows}
    enroll = read_enrollment(args.root)
    for split in PROTOCOLS:  # audio files that no CM protocol lists: ASV enrollment clips, or unknown
        for path in sorted((args.root / f"ASVspoof2019_LA_{split}" / "flac").glob("*.flac")):
            if path.stem not in listed:
                spk = enroll.get(path.stem)
                rows.append({"id": path.stem, "split": split, "speaker": spk or "?",
                             "system": "enroll" if spk else "?", "label": "?"})
    genders = read_genders(args.root)
    for r in rows:
        r["gender"] = genders.get(r["speaker"], "?")

    tasks = [(r["id"], str(audio_path(args.root, r["split"], r["id"]))) for r in rows]
    results: dict[str, dict] = {}
    gemaps: dict[str, np.ndarray] = {}
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (cid, out, vec) in enumerate(pool.imap_unordered(stats, tasks, chunksize=32), 1):
            results[cid] = out
            if vec is not None:
                gemaps[cid] = vec
            if i % 5000 == 0 or i == len(tasks):
                print(f"{i:>7,}/{len(tasks):,}  {time.time() - t0:5.0f}s", flush=True)
    for r in rows:
        r.update(results[r["id"]])

    hashes = Counter(r.get("hash", "") for r in rows)
    for r in rows:
        r["dup"] = hashes[r["hash"]] if r.get("hash") else 0

    rows.sort(key=lambda r: r["id"])
    cats = {k: sorted({r[k] for r in rows}) for k in ("split", "label", "system", "speaker", "gender")}
    cols: dict[str, list] = {"id": [r["id"] for r in rows], "hash": [r.get("hash", "") for r in rows]}
    for k, vocab in cats.items():
        code = {v: i for i, v in enumerate(vocab)}
        cols[k] = [code[r[k]] for r in rows]
    for k, dec in NUMERIC.items():
        cols[k] = [None if r.get(k) is None or not np.isfinite(r[k]) else round(float(r[k]), dec) for r in rows]
    cols["dup"] = [r["dup"] for r in rows]
    cols["err"] = [r["err"] for r in rows]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"dataset": "ASVspoof 2019 LA", "n": len(rows), "audio": {"base": "audio", "ext": "flac"}, "cat": cats, "cols": cols}
    args.out.write_text(json.dumps(doc, separators=(",", ":")))

    gids = [r["id"] for r in rows if r["id"] in gemaps]
    names = list(smile().feature_names)
    np.savez_compressed(args.out.parent / "egemaps.npz", ids=np.array(gids), names=np.array(names),
                        X=np.stack([gemaps[c] for c in gids]) if gids else np.zeros((0, len(names)), np.float32))

    # Short summary for the terminal.
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB), {len(rows):,} clips; egemaps.npz: {len(gids):,} rows")
    print("split x label:", dict(Counter((r["split"], r["label"]) for r in rows)))
    print("sample rates:", dict(Counter(r.get("sr") for r in rows)))
    print("errors:", Counter(r["err"] for r in rows if r["err"]).most_common(5) or 0)
    print("no pitch found:", sum(1 for r in rows if not r["err"] and r.get("f0_hz") is None))
    print("clips in duplicate groups:", sum(1 for r in rows if r["dup"] > 1))
    print("not in CM protocols:", dict(Counter(r["system"] for r in rows if r["label"] == "?")))


if __name__ == "__main__":
    main()

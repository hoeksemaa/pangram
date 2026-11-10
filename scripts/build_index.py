"""Build a per-clip index of ASVspoof 2019 LA: labels plus simple audio statistics.

The statistics are cheap clues that a model could use as shortcuts (silence, loudness,
bandwidth, ...). The viewer plots them for human vs. AI clips.

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
FRAME_MS = 20  # frame length for silence statistics
SIL_REL_DB = 40  # a frame is silent if it is this far below the loudest frame
NFFT, HOP = 512, 256  # for the bandwidth statistic
ROLLOFF = 0.99

# Numeric columns: name -> decimals kept in the JSON.
NUMERIC = {
    "dur": 3, "sr": 0, "ch": 0, "lead_ms": 0, "trail_ms": 0, "sil": 3,
    "rms_db": 1, "peak_db": 1, "noise_db": 1, "zero": 4, "bw_khz": 2, "clip": 5,
}


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


def stats(task: tuple[str, str]) -> tuple[str, dict]:
    cid, path = task
    out: dict = {"err": ""}
    try:
        pcm, sr = sf.read(path, dtype="int16", always_2d=True)
        out["sr"], out["ch"] = sr, pcm.shape[1]
        out["hash"] = hashlib.md5(pcm.tobytes()).hexdigest()[:12]
        x = pcm.astype(np.float32).mean(axis=1) / 32768.0
        n = len(x)
        out["dur"] = n / sr
        if n == 0:
            out["err"] = "empty"
            return cid, out
        absx = np.abs(x)
        out["peak_db"] = 20 * np.log10(absx.max() + 1e-12)
        out["rms_db"] = 10 * np.log10(np.mean(x * x) + 1e-12)
        out["zero"] = float(np.mean(pcm[:, 0] == 0))
        out["clip"] = float(np.mean(absx >= 32767 / 32768))

        flen = max(1, int(sr * FRAME_MS / 1000))
        nf = n // flen
        if nf > 0:
            frames = x[: nf * flen].reshape(nf, flen)
            fdb = 10 * np.log10(np.mean(frames * frames, axis=1) + 1e-10)
            active = np.flatnonzero(fdb > fdb.max() - SIL_REL_DB)
            out["lead_ms"] = int(active[0]) * FRAME_MS
            out["trail_ms"] = int(nf - 1 - active[-1]) * FRAME_MS
            out["sil"] = 1 - len(active) / nf
            out["noise_db"] = float(np.percentile(fdb, 10))

        if n >= NFFT:
            win = np.hanning(NFFT).astype(np.float32)
            frames = np.lib.stride_tricks.sliding_window_view(x, NFFT)[::HOP] * win
            power = np.mean(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=0)
            cum = np.cumsum(power)
            k = int(np.searchsorted(cum, ROLLOFF * cum[-1]))
            out["bw_khz"] = k * sr / NFFT / 1000
    except Exception as exc:  # keep going; the viewer shows the error
        out["err"] = f"{type(exc).__name__}: {exc}"[:200]
    return cid, out


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
    t0 = time.time()
    with Pool(args.workers) as pool:
        for i, (cid, out) in enumerate(pool.imap_unordered(stats, tasks, chunksize=128), 1):
            results[cid] = out
            if i % 10000 == 0 or i == len(tasks):
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
        cols[k] = [None if r.get(k) is None else round(float(r[k]), dec) for r in rows]
    cols["dup"] = [r["dup"] for r in rows]
    cols["err"] = [r["err"] for r in rows]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    doc = {"dataset": "ASVspoof 2019 LA", "n": len(rows), "audio": {"base": "audio", "ext": "flac"}, "cat": cats, "cols": cols}
    args.out.write_text(json.dumps(doc, separators=(",", ":")))

    # Short summary for the terminal.
    print(f"\nwrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB), {len(rows):,} clips")
    print("split x label:", dict(Counter((r["split"], r["label"]) for r in rows)))
    print("sample rates:", dict(Counter(r.get("sr") for r in rows)))
    print("channels:", dict(Counter(r.get("ch") for r in rows)))
    print("errors:", sum(1 for r in rows if r["err"]))
    print("clips in duplicate groups:", sum(1 for r in rows if r["dup"] > 1))
    print("not in CM protocols:", dict(Counter(r["system"] for r in rows if r["label"] == "?")))


if __name__ == "__main__":
    main()

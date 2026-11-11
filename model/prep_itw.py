"""Unpack In-the-Wild (Müller et al. 2022, CC BY-SA 4.0) into 16 kHz mono FLAC. Test set only.

Reads each WAV straight from the zip (no full extract), writes data/raw/itw/<n>.flac + meta.csv.
Get the zip first:
    hf download mueller91/In-The-Wild release_in_the_wild.zip --repo-type dataset --local-dir data/raw/itw_zip
Usage:
    uv run model/prep_itw.py [--delete-zip]
    uv run model/prep_itw.py --keep 0.1   # then keep a 10% sample (same mix of speakers and labels); delete the rest
"""

import argparse
import io
import zipfile
from multiprocessing import Pool

import soundfile as sf

from common import ITW_ROOT, REPO, SR

ZIP = REPO / "data/raw/itw_zip/release_in_the_wild.zip"


def convert(names: list[str]) -> list[tuple[str, int]]:
    import numpy as np
    import torch
    import torchaudio
    torch.set_num_threads(1)
    out = []
    with zipfile.ZipFile(ZIP) as z:
        for name in names:
            x, sr = sf.read(io.BytesIO(z.read(name)), dtype="float32", always_2d=True)
            x = x.mean(axis=1)
            if sr != SR:
                x = torchaudio.functional.resample(torch.from_numpy(x), sr, SR).numpy()
            stem = name.rsplit("/", 1)[-1][:-4]
            sf.write(ITW_ROOT / f"{stem}.flac", np.clip(x, -1, 1), SR, subtype="PCM_16")
            out.append((stem, sr))
    return out


def keep_sample(frac: float, seed: int = 0) -> None:
    """Keep `frac` of the clips in each (speaker, label) group; delete the other FLACs. Full list -> meta_full.csv."""
    import csv
    import random
    full = ITW_ROOT / "meta_full.csv"
    if not full.exists():
        full.write_bytes((ITW_ROOT / "meta.csv").read_bytes())
    rows = list(csv.DictReader(full.open()))
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["speaker"], r["label"]), []).append(r)
    rng = random.Random(seed)
    kept = []
    for key in sorted(groups):
        g = groups[key]
        kept += rng.sample(g, max(1, round(len(g) * frac)))
    keep = {r["file"] for r in kept}
    for r in rows:
        if r["file"] not in keep:
            (ITW_ROOT / f"{r['file'][:-4]}.flac").unlink(missing_ok=True)
    kept.sort(key=lambda r: int(r["file"][:-4]))
    with (ITW_ROOT / "meta.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "speaker", "label"])
        w.writeheader()
        w.writerows(kept)
    n_ai = sum(r["label"] == "spoof" for r in kept)
    print(f"kept {len(kept)} of {len(rows)} ({len(kept) - n_ai} human, {n_ai} AI, {len(groups)} speaker-label groups)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete-zip", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--keep", type=float, default=0, help="keep this share of the unpacked clips; delete the rest")
    args = ap.parse_args()
    if args.keep:
        keep_sample(args.keep)
        return
    ITW_ROOT.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP) as z:
        names = [n for n in z.namelist() if n.endswith(".wav")]
        (ITW_ROOT / "meta.csv").write_bytes(z.read("release_in_the_wild/meta.csv"))
    chunks = [names[i::args.workers * 8] for i in range(args.workers * 8)]
    with Pool(args.workers) as pool:
        done = [r for part in pool.map(convert, chunks) for r in part]
    rates = {}
    for _, sr in done:
        rates[sr] = rates.get(sr, 0) + 1
    print(f"{len(done)} clips -> {ITW_ROOT}; source sample rates: {rates}")
    if args.delete_zip and len(done) == len(names):
        ZIP.unlink()
        print(f"deleted {ZIP}")


if __name__ == "__main__":
    main()

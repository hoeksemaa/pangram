"""Build the static site for Vercel: the web app plus a stratified sample of clips.

The sample takes the same number of clips from each split x system group (fixed seed),
so the human class and every AI system appear in each split. Audio is written as
16-bit WAV, which every browser can play and decode (the full data stays local).

Usage:
    uv run scripts/build_site.py                       # -> site/
    uv run scripts/build_site.py --per-group 20 --seed 0
Deploy:
    cd site && vercel deploy --prod
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
WEB_FILES = ["index.html", "app.js", "explorer.js", "style.css"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=REPO / "data/index/asvspoof2019_la.json")
    ap.add_argument("--root", type=Path, default=REPO / "data/raw/LA")
    ap.add_argument("--out", type=Path, default=REPO / "site")
    ap.add_argument("--per-group", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    doc = json.loads(args.index.read_text())
    cols, cat, n = doc["cols"], doc["cat"], doc["n"]

    groups: dict[tuple[int, int], list[int]] = {}
    for i in range(n):
        if cols["err"][i] or cat["label"][cols["label"][i]] == "?":
            continue
        groups.setdefault((cols["split"][i], cols["system"][i]), []).append(i)
    rng = np.random.default_rng(args.seed)
    pick: list[int] = []
    for key in sorted(groups):
        idx = groups[key]
        pick.extend(rng.choice(idx, size=min(args.per_group, len(idx)), replace=False).tolist())
    pick.sort()

    # Replace only what this script makes; keep site/.vercel (the project link).
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(args.out / "audio", ignore_errors=True)
    audio_bytes = 0
    for i in pick:
        split, cid = cat["split"][cols["split"][i]], cols["id"][i]
        pcm, sr = sf.read(args.root / f"ASVspoof2019_LA_{split}" / "flac" / f"{cid}.flac", dtype="int16")
        dst = args.out / "audio" / split / f"{cid}.wav"
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, pcm, sr, subtype="PCM_16")
        audio_bytes += dst.stat().st_size

    site = {
        **doc,
        "n": len(pick),
        "audio": {"base": "audio", "ext": "wav"},
        "sample": {"n_total": n, "per_group": args.per_group, "seed": args.seed, "groups": len(groups)},
        "cols": {k: [v[i] for i in pick] for k, v in cols.items()},
    }
    (args.out / "index.json").write_text(json.dumps(site, separators=(",", ":")))
    # Whisper's guess of what each clip says (scripts/transcribe_sample.py): display only, optional.
    guesses = REPO / "data/transcripts/whisper-sample.json"
    (args.out / "transcripts.json").unlink(missing_ok=True)
    if guesses.exists():
        shutil.copy(guesses, args.out / "transcripts.json")
    for name in WEB_FILES:
        shutil.copy(REPO / "web" / name, args.out / name)

    labels = Counter(cat["label"][cols["label"][i]] for i in pick)
    print(f"{len(pick)} clips from {len(groups)} split x system groups ({dict(labels)})")
    print(f"audio {audio_bytes / 1e6:.1f} MB, index {(args.out / 'index.json').stat().st_size / 1e3:.0f} KB -> {args.out}")


if __name__ == "__main__":
    main()

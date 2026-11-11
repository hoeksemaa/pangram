# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx-whisper", "soundfile", "numpy"]
# ///
"""Guess what each site clip says, with Whisper. Display only: nothing else reads this file.

ASVspoof 2019 has no transcripts, so this is Whisper's guess, not the script the speaker read.
The inline dependencies run in their own environment (the project environment does not change).

Usage:
    uv run scripts/transcribe_sample.py            # site/ clips -> data/transcripts/whisper-sample.json
    uv run scripts/transcribe_sample.py --limit 5  # quick test
"""

import argparse
import json
import time
from datetime import date
from pathlib import Path

import mlx_whisper
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
MODEL = "mlx-community/whisper-large-v3-turbo"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=Path, default=REPO / "site")
    ap.add_argument("--out", type=Path, default=REPO / "data/transcripts/whisper-sample.json")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    doc = json.loads((args.site / "index.json").read_text())
    ids, splits = doc["cols"]["id"], [doc["cat"]["split"][s] for s in doc["cols"]["split"]]
    ext = doc.get("audio", {}).get("ext", "wav")
    todo = list(zip(ids, splits))[: args.limit or None]

    text: dict[str, str] = {}
    t0 = time.time()
    for k, (cid, split) in enumerate(todo, 1):
        x, sr = sf.read(args.site / "audio" / split / f"{cid}.{ext}", dtype="float32")
        if sr != 16000:
            raise SystemExit(f"{cid}: {sr} Hz; Whisper needs 16 kHz")
        r = mlx_whisper.transcribe(x, path_or_hf_repo=MODEL, language="en", condition_on_previous_text=False)
        text[cid] = r["text"].strip()
        if k % 50 == 0 or k == len(todo):
            print(f"{k}/{len(todo)}  {time.time() - t0:.0f} s  {cid}: {text[cid]}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"model": MODEL, "made": date.today().isoformat(), "text": text}, indent=1))
    print(f"{len(text)} guesses -> {args.out}")


if __name__ == "__main__":
    main()

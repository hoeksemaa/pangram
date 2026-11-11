# /// script
# requires-python = ">=3.11"
# dependencies = ["mlx-whisper", "soundfile", "numpy"]
# ///
"""Does the trim cut words? Whisper transcribes each clip before and after trimming; we compare words.

Inline dependencies: runs in its own environment (Apple GPU via MLX).

Usage:
    uv run model/check_trim_words.py data/index/trim_vad.npz [--n 200]
"""

import argparse
import re
import sys
from pathlib import Path

import mlx_whisper
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import la_clips, load  # noqa: E402

MODEL = "mlx-community/whisper-large-v3-turbo"


def words(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trim")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    t = np.load(args.trim)
    b = {i: (s, e) for i, s, e in zip(t["ids"], t["start"], t["end"])}
    clips = la_clips()
    clips = clips.take(np.isin(clips.ids, t["ids"]))
    rng = np.random.default_rng(2)
    pick = np.concatenate([rng.choice(np.flatnonzero(clips.label == k), args.n // 2, replace=False) for k in (0, 1)])
    same = lost = 0
    for k in pick:
        x = load(clips.paths[k])
        s, e = b[clips.ids[k]]
        full = words(mlx_whisper.transcribe(x, path_or_hf_repo=MODEL, language="en", condition_on_previous_text=False)["text"])
        cut = words(mlx_whisper.transcribe(x[s:e], path_or_hf_repo=MODEL, language="en", condition_on_previous_text=False)["text"])
        if full == cut:
            same += 1
            continue
        if full[:1] != cut[:1] or full[-1:] != cut[-1:]:
            lost += 1  # first or last word changed: the cut can be the cause
        who = "human" if clips.label[k] == 0 else "AI"
        print(f"{clips.ids[k]} ({who}, cut {s / 16000:.2f} s / {(len(x) - e) / 16000:.2f} s)\n  full: {' '.join(full)}\n  trim: {' '.join(cut)}")
    print(f"\n{len(pick)} clips: {same} identical, {len(pick) - same} differ, {lost} differ in the first or last word")


if __name__ == "__main__":
    main()

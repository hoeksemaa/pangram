"""Find where speech starts and ends in each clip, with Silero VAD (a small neural speech detector).

We cut only before the first speech and after the last speech, plus a 30 ms margin. model/check_trim.py and model/check_trim_words.py chose it:
30 ms removes the silence shortcut (AUC 0.83 -> 0.56); 100 ms leaves part of it (0.65).
An energy threshold is not safe here: AI clips have near-digital silence and human clips
have room noise, so a loudness rule would trim the two classes differently.
No audio is rewritten; the bounds go to data/index/trim_vad.npz (or --out).

Usage:
    uv run model/trim.py                 # all labeled LA clips -> data/index/trim_vad.npz
    uv run model/trim.py --set itw       # In-the-Wild -> data/index/trim_vad_itw.npz
    uv run model/trim.py --limit 200     # timing test
"""

import argparse
import time
from multiprocessing import Pool

import numpy as np

from common import REPO, TRIM, itw_clips, la_clips, load

_model = None


def _init_pad(pad_ms: int) -> None:
    global PAD_MS
    PAD_MS = pad_ms
    _init()


def _init() -> None:
    global _model
    import torch
    from silero_vad import load_silero_vad
    torch.set_num_threads(1)
    _model = load_silero_vad()


PAD_MS = 30  # margin kept around speech; set by --pad-ms


def bounds(path) -> tuple[int, int, int, int]:
    """(start, end, n_samples, n_speech_segments). No speech found -> the whole clip, 0 segments."""
    import torch
    from silero_vad import get_speech_timestamps
    x = load(path)
    ts = get_speech_timestamps(torch.from_numpy(x), _model, sampling_rate=16000, speech_pad_ms=PAD_MS)
    if not ts:
        return 0, len(x), len(x), 0
    return max(0, ts[0]["start"]), min(len(x), ts[-1]["end"]), len(x), len(ts)


def run(paths, workers: int = 8, pad_ms: int = 30) -> np.ndarray:
    with Pool(workers, initializer=_init_pad, initargs=(pad_ms,)) as pool:
        return np.array(pool.map(bounds, paths, chunksize=64), dtype=np.int64)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=str(TRIM))
    ap.add_argument("--pad-ms", type=int, default=30)
    ap.add_argument("--set", choices=["la", "itw"], default="la")
    args = ap.parse_args()
    clips = la_clips() if args.set == "la" else itw_clips()
    if args.set == "itw" and args.out == str(TRIM):
        args.out = str(REPO / "data/index/trim_vad_itw.npz")
    if args.limit:
        clips = clips.take(np.random.default_rng(0).permutation(len(clips)) < args.limit)
    t0 = time.time()
    b = run(clips.paths, args.workers, args.pad_ms)
    print(f"{len(clips)} clips in {time.time() - t0:.0f} s")
    np.savez(args.out, ids=clips.ids, start=b[:, 0], end=b[:, 1], n=b[:, 2], segs=b[:, 3])
    removed = 1 - (b[:, 1] - b[:, 0]) / b[:, 2]
    for name, m in [("human", clips.label == 0), ("AI", clips.label == 1)]:
        print(f"{name}: removed {np.median(removed[m]) * 100:.0f}% median, {(b[m, 3] == 0).sum()} with no speech found")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

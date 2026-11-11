"""Train the detector on ASVspoof 2019 LA (train split), choose on the validation split (dev), test once.

Logs train loss every step and validation loss + EER every half epoch, on one chart.
Training uses random 3 s crops; scoring uses fixed 3 s windows (mean over the windows of a clip).
Stops when validation loss has not improved for --patience checks (or at --max-min minutes)
and keeps the checkpoint with the lowest validation loss.

Usage:
    uv run model/train.py --name lcnn_trim --trim vad --loss ce
    uv run model/train.py --name smoke --limit 2000 --max-epochs 1     # quick test
    uv run model/train.py --name overfit --overfit                     # one batch must reach ~0 loss
Outputs in runs/<name>/: steps.csv, checks.csv, curve.png, best.pt, results.json.
"""

import argparse
import csv
import json
import time
from multiprocessing import Pool

import numpy as np
import torch

from common import RUNS, REPO, itw_clips, la_clips, load
from metrics import TARGET_FPR, eer, report, summary_line, threshold_at_fpr
from nets import LCNN, loss_fn

SR = 16000


def bounds_for(clips, trim: str, file: str) -> np.ndarray:
    """(start, end) per clip; end = -1 means "to the end"."""
    if trim == "none":
        return np.tile([0, -1], (len(clips), 1))
    t = np.load(REPO / "data/index" / file)
    b = {i: (s, e) for i, s, e in zip(t["ids"], t["start"], t["end"])}
    return np.array([b[i] for i in clips.ids])


def _load16(job) -> np.ndarray:
    path, s, e = job
    x = load(path, s, None if e < 0 else e)
    return (np.clip(x, -1, 1) * 32767).astype(np.int16)


def decode(pool, clips, bounds) -> list:
    return pool.map(_load16, [(p, int(s), int(e)) for p, (s, e) in zip(clips.paths, bounds)], chunksize=64)


def tile(x: np.ndarray, n: int, offset: int = 0) -> np.ndarray:
    """Repeat x until it has n samples (never pad with silence), starting at offset."""
    reps = int(np.ceil((n + offset) / len(x)))
    return np.tile(x, reps)[offset:offset + n]


def crop(x: np.ndarray, n: int, rng) -> np.ndarray:
    if len(x) >= n:
        s = rng.integers(0, len(x) - n + 1)
        return x[s:s + n]
    return tile(x, n, int(rng.integers(0, len(x))))


def windows(x: np.ndarray, n: int) -> list:
    """Fixed n-sample windows that cover the clip: a short clip is repeated to n; a long one is cut
    into back-to-back windows (the last one ends at the clip end). Same input length as training."""
    if len(x) <= n:
        return [tile(x, n)]
    starts = list(range(0, len(x) - n + 1, n))
    if starts[-1] != len(x) - n:
        starts.append(len(x) - n)
    return [x[s:s + n] for s in starts]


@torch.no_grad()
def score_audio(model, audio: list, y: np.ndarray, w, dev, n: int, batch: int = 256) -> tuple[np.ndarray, float]:
    """Clip scores and the class-weighted loss. A clip's output = the mean over its windows,
    so a score does not depend on which other clips share the batch."""
    model.eval()
    owner, wins = [], []
    for i, a in enumerate(audio):
        for win in windows(a, n):
            owner.append(i)
            wins.append(win)
    owner = np.array(owner)
    outs = []
    for k in range(0, len(wins), batch):
        x = torch.from_numpy(np.stack(wins[k:k + batch]).astype(np.float32) / 32767).to(dev)
        outs.append(model(x).float().cpu())
    out = torch.cat(outs)
    counts = torch.from_numpy(np.bincount(owner, minlength=len(audio))).float()
    shape = (len(audio),) + tuple(out.shape[1:])
    clip = torch.zeros(shape).index_add_(0, torch.from_numpy(owner), out) / counts.view(-1, *([1] * (out.dim() - 1)))
    yy = torch.from_numpy(y)
    loss = loss_fn(model, clip, yy, w.cpu()).item()
    scores = (clip[:, 1] - clip[:, 0]) if model.head_type == "ce" else -clip
    model.train()
    return scores.numpy().astype(np.float64), loss


def score_clips(model, pool, clips, bounds, w, dev, n: int, chunk: int = 4096) -> np.ndarray:
    """Score a big split without holding all its audio in memory."""
    scores = np.zeros(len(clips))
    lengths = np.where(bounds[:, 1] < 0, 0, bounds[:, 1] - bounds[:, 0])
    order = np.argsort(lengths)
    for k in range(0, len(order), chunk):
        part = order[k:k + chunk]
        sub = clips.take(np.isin(np.arange(len(clips)), part))
        idx = np.flatnonzero(np.isin(np.arange(len(clips)), part))
        audio = decode(pool, sub, bounds[idx])
        scores[idx], _ = score_audio(model, audio, sub.label, w, dev, n)
    return scores


def plot(run, name: str, steps: list, checks: list) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    st = np.array(steps)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.3]})
    a1.plot(st[:, 1], st[:, 2], color="#4f8bff", alpha=0.18, lw=0.6)
    ema, v = [], 0.0
    for t, x in enumerate(st[:, 2], 1):
        v = 0.95 * v + 0.05 * x
        ema.append(v / (1 - 0.95 ** t))  # bias-corrected: no lag at the start
    a1.plot(st[:, 1], ema, color="#4f8bff", lw=2, label="train loss (every step, smoothed)")
    if checks:
        c = np.array([[r["epoch"], r["dev_loss"], r["dev_eer"]] for r in checks])
        a1.plot(c[:, 0], c[:, 1], "o-", color="#ffb000", lw=2, label="validation loss")
        best = min(checks, key=lambda r: r["dev_loss"])
        a1.axvline(best["epoch"], color="#888", ls="--", lw=1, label=f"best (epoch {best['epoch']:.1f})")
        a2.plot(c[:, 0], c[:, 2] * 100, "o-", color="#ffb000", lw=2)
    a1.set_ylabel("loss")
    a1.set_title(name)
    a1.legend()
    a1.grid(alpha=0.3)
    a2.set_ylabel("validation EER %")
    a2.set_xlabel("epoch")
    a2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run / "curve.png", dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--trim", choices=["vad", "none"], default="vad")
    ap.add_argument("--loss", choices=["ce", "oc"], default="ce")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--crop", type=float, default=3.0, help="training crop, seconds")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--max-min", type=float, default=60, help="stop training after this many minutes")
    ap.add_argument("--patience", type=int, default=6, help="checks without a better validation loss")
    ap.add_argument("--checks-per-epoch", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="use only this many train and dev clips (tests)")
    ap.add_argument("--overfit", action="store_true", help="train on one batch only (sanity check)")
    ap.add_argument("--no-test", action="store_true", help="skip the test split and In-the-Wild")
    args = ap.parse_args()

    run = RUNS / args.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "config.json").write_text(json.dumps(vars(args), indent=1))
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    tr, va = la_clips("train"), la_clips("dev")
    if args.limit:
        tr = tr.take(rng.permutation(len(tr)) < args.limit)
        va = va.take(rng.permutation(len(va)) < args.limit)
    pool = Pool(8)
    t0 = time.time()
    tr_audio = decode(pool, tr, bounds_for(tr, args.trim, "trim_vad.npz"))
    va_audio = decode(pool, va, bounds_for(va, args.trim, "trim_vad.npz"))
    print(f"decoded {len(tr)} train + {len(va)} validation clips in {time.time() - t0:.0f} s", flush=True)

    # Class weights: each class counts the same in the loss, so the 9:1 AI share is not learned as a prior.
    n_h, n_a = int((tr.label == 0).sum()), int((tr.label == 1).sum())
    w = torch.tensor([len(tr) / (2 * n_h), len(tr) / (2 * n_a)], dtype=torch.float32, device=dev)

    model = LCNN(head=args.loss).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{args.name}: {n_params:,} parameters, trim={args.trim}, loss={args.loss}, seed={args.seed}, "
          f"weights human {w[0].item():.2f} / AI {w[1].item():.2f}", flush=True)

    L = int(args.crop * SR)
    per_epoch = len(tr) // args.batch
    check_every = max(1, per_epoch // args.checks_per_epoch)
    steps, checks = [], []
    best, bad, step = float("inf"), 0, 0
    t_start = time.time()
    fixed = rng.permutation(len(tr))[: args.batch] if args.overfit else None
    sf_steps = (run / "steps.csv").open("w", newline="")
    sw = csv.writer(sf_steps)
    sw.writerow(["step", "epoch", "train_loss"])
    stop = None
    for epoch in range(args.max_epochs if not args.overfit else 1000):
        order = rng.permutation(len(tr))
        for k in range(per_epoch if not args.overfit else 1):
            idx = fixed if args.overfit else order[k * args.batch:(k + 1) * args.batch]
            x = torch.from_numpy(np.stack([crop(tr_audio[i], L, rng) for i in idx]).astype(np.float32) / 32767).to(dev)
            y = torch.from_numpy(tr.label[idx]).to(dev)
            loss = loss_fn(model, model(x), y, w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
            ep = step / per_epoch
            steps.append((step, ep, loss.item()))
            sw.writerow([step, f"{ep:.4f}", f"{loss.item():.5f}"])
            if args.overfit:
                if step % 25 == 0:
                    print(f"step {step}: loss {loss.item():.4f}", flush=True)
                if step >= 300:
                    stop = "overfit test done"
                break
            if step % check_every == 0:
                va_scores, va_loss = score_audio(model, va_audio, va.label, w, dev, L)
                va_eer = eer(va_scores, va.label)[0]
                train_recent = float(np.mean([s[2] for s in steps[-check_every:]]))
                checks.append({"step": step, "epoch": round(ep, 3), "train_loss": train_recent,
                               "dev_loss": va_loss, "dev_eer": va_eer, "minutes": (time.time() - t_start) / 60})
                if va_loss < best:
                    best, bad = va_loss, 0
                    torch.save(model.state_dict(), run / "best.pt")
                else:
                    bad += 1
                print(f"epoch {ep:5.2f}  train {train_recent:.4f}  validation {va_loss:.4f}  "
                      f"validation EER {va_eer * 100:5.2f}%  {'*' if bad == 0 else ' '}  "
                      f"{(time.time() - t_start) / 60:4.1f} min", flush=True)
                with (run / "checks.csv").open("w", newline="") as f:
                    cw = csv.DictWriter(f, fieldnames=list(checks[0]))
                    cw.writeheader()
                    cw.writerows(checks)
                sf_steps.flush()
                plot(run, args.name, steps, checks)
                if bad >= args.patience:
                    stop = f"validation loss did not improve for {args.patience} checks"
                elif (time.time() - t_start) / 60 > args.max_min:
                    stop = f"time limit ({args.max_min:.0f} min)"
                if stop:
                    break
        if stop:
            break
    sf_steps.close()
    stop = stop or f"max epochs ({args.max_epochs})"
    print(f"stopped: {stop}", flush=True)
    if args.overfit:
        plot(run, args.name, steps, [])
        pool.close()
        return

    # Best checkpoint -> threshold on validation -> test split and In-the-Wild, once.
    model.load_state_dict(torch.load(run / "best.pt", map_location=dev))
    va_scores, _ = score_audio(model, va_audio, va.label, w, dev, L)
    thr = threshold_at_fpr(va_scores, va.label, TARGET_FPR)
    results = {"name": args.name, "config": vars(args), "parameters": n_params, "stopped": stop,
               "train_minutes": round((time.time() - t_start) / 60, 1), "best_check": min(checks, key=lambda r: r["dev_loss"]),
               "threshold_rule": f"flags {TARGET_FPR:.0%} of validation humans", "validation": report(va_scores, va, thr)}
    print(summary_line("validation", results["validation"]), flush=True)
    del tr_audio, va_audio
    if not args.no_test:
        te = la_clips("eval")
        te_scores = score_clips(model, pool, te, bounds_for(te, args.trim, "trim_vad.npz"), w, dev, L)
        results["test"] = report(te_scores, te, thr)
        print(summary_line("test", results["test"]), flush=True)
        itw = itw_clips()
        itw_scores = score_clips(model, pool, itw, bounds_for(itw, args.trim, "trim_vad_itw.npz"), w, dev, L)
        results["in_the_wild"] = report(itw_scores, itw, thr)
        print(summary_line("in-the-wild", results["in_the_wild"]), flush=True)
        np.savez(run / "scores.npz", test_ids=te.ids, test=te_scores, itw_ids=itw.ids, itw=itw_scores,
                 dev_ids=va.ids, dev=va_scores)
    (run / "results.json").write_text(json.dumps(results, indent=1))
    pool.close()


if __name__ == "__main__":
    main()

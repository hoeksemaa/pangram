"""Export a trained run to ONNX for the browser (Try tab): waveform windows in, AI scores out.

The browser runtime has no reliable STFT, so the LFCC step is rebuilt from plain ops:
the power spectrum is a fixed convolution (Hann window x cos/sin), then the same filter bank,
log, and DCT matrices as torchaudio, then deltas by slicing. The script checks that the export
gives the same scores as the trained model (PyTorch, then ONNX Runtime) before it writes the file.

Input:  wav, float32 (batch, 48000): 3 s windows at 16 kHz, scaled to [-1, 1].
Output: score, float32 (batch): AI log-odds (higher = more AI). A clip's score = mean over its windows.

Usage:
    uv run --with onnx --with onnxruntime model/export.py --run lcnn_trim
Writes runs/<run>/model.onnx (not in git).
"""

import argparse
import json
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import RUNS, la_clips, load, trim_bounds
from nets import ARCHS, LFCC

N = 48000  # 3 s at 16 kHz, the training window


class ConvLFCC(nn.Module):
    """Same numbers as nets.LFCC, built from conv1d + matmul (ONNX-friendly)."""

    def __init__(self, ref: LFCC):
        super().__init__()
        spec = ref.lfcc.Spectrogram
        n_fft, win, hop = spec.n_fft, spec.win_length, spec.hop_length
        w = torch.zeros(n_fft)
        left = (n_fft - win) // 2
        w[left:left + win] = spec.window  # torch.stft centers a short window inside n_fft
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
        t = torch.arange(n_fft, dtype=torch.float64)[None, :]
        ang = 2 * math.pi * k * t / n_fft
        self.register_buffer("cos", (torch.cos(ang) * w.double()).float()[:, None, :])
        self.register_buffer("sin", (torch.sin(ang) * w.double()).float()[:, None, :])
        self.hop = hop
        self.register_buffer("fb", ref.lfcc.filter_mat.clone())
        self.register_buffer("dct", ref.lfcc.dct_mat.clone())

    @staticmethod
    def deltas(c: torch.Tensor) -> torch.Tensor:
        """torchaudio.compute_deltas (window 5, edge padding) by slicing."""
        p = torch.cat([c[..., :1], c[..., :1], c, c[..., -1:], c[..., -1:]], -1)
        T = c.shape[-1]
        return (2 * (p[..., 4:4 + T] - p[..., 0:T]) + (p[..., 3:3 + T] - p[..., 1:1 + T])) / 10

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        re = F.conv1d(x, self.cos, stride=self.hop)
        im = F.conv1d(x, self.sin, stride=self.hop)
        power = re * re + im * im                       # B, 257, frames
        lf = torch.log(torch.matmul(power.transpose(1, 2), self.fb) + 1e-6)
        c = torch.matmul(lf, self.dct).transpose(1, 2)  # B, 20, frames
        d1 = self.deltas(c)
        d2 = self.deltas(d1)
        return torch.cat([c, d1, d2], 1).unsqueeze(1)


class Export(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.model.front = ConvLFCC(model.front)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.model.score(wav)


def as_trained(x: np.ndarray) -> np.ndarray:
    """Training stored audio as int16 and divided by 32767; do the same."""
    return (np.clip(x, -1, 1) * 32767).astype(np.int16).astype(np.float32) / 32767


def windows(x: np.ndarray, n: int = N) -> np.ndarray:
    """Same as train.windows: repeat a short clip to n; cut a long one into back-to-back windows."""
    if len(x) <= n:
        return np.tile(x, int(np.ceil(n / len(x))))[None, :n]
    starts = list(range(0, len(x) - n + 1, n))
    if starts[-1] != len(x) - n:
        starts.append(len(x) - n)
    return np.stack([x[s:s + n] for s in starts])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    args = ap.parse_args()
    run = RUNS / args.run
    cfg = json.loads((run / "config.json").read_text())
    model = ARCHS[cfg.get("arch", "lcnn")](head=cfg["loss"])
    model.load_state_dict(torch.load(run / "best.pt", map_location="cpu"))
    model.eval()
    ref = ARCHS[cfg.get("arch", "lcnn")](head=cfg["loss"])
    ref.load_state_dict(model.state_dict())
    ref.eval()
    exp = Export(model).eval()

    # Check 1: rebuilt LFCC = torchaudio LFCC, and scores match, on real trimmed test clips.
    te = la_clips("eval")
    b = trim_bounds()
    rng = np.random.default_rng(0)
    pick = rng.choice(len(te), 64, replace=False)
    xs = []
    for k in pick:
        s, e = b[te.ids[k]]
        xs.append(windows(as_trained(load(te.paths[k], s, e))))
    x = torch.from_numpy(np.concatenate(xs))
    with torch.no_grad():
        f_ref, f_new = ref.front(x), exp.model.front(x)
        s_ref, s_new = ref.score(x), exp(x)
    print(f"LFCC max diff {(f_ref - f_new).abs().max():.2e} (feature scale {f_ref.abs().max():.1f}); "
          f"score max diff {(s_ref - s_new).abs().max():.2e} over {len(x)} windows")
    assert (s_ref - s_new).abs().max() < 1e-3

    out = run / "model.onnx"
    torch.onnx.export(exp, torch.zeros(2, N), out, input_names=["wav"], output_names=["score"],
                      dynamic_axes={"wav": {0: "batch"}, "score": {0: "batch"}}, opset_version=17, dynamo=False)

    # Check 2: ONNX Runtime = PyTorch.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    s_onnx = sess.run(None, {"wav": x.numpy()})[0]
    print(f"ONNX Runtime max diff {np.abs(s_onnx - s_ref.numpy()).max():.2e}; file {out} ({out.stat().st_size / 1e6:.2f} MB)")
    assert np.abs(s_onnx - s_ref.numpy()).max() < 1e-3


if __name__ == "__main__":
    main()

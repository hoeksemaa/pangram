"""The detector network: waveform -> LFCC (inside the model, so one file runs anywhere) -> LCNN -> score.

LFCC: linear-frequency cepstral coefficients. Linear (not mel) frequency keeps detail in the high
frequencies, where vocoder artifacts live. 20 coefficients + deltas + delta-deltas = 60 rows.
Two bodies on the same LFCC input:
- LCNN: light CNN with max-feature-map units, the ASVspoof 2021 baseline family (~164k parameters).
- ResNet18: 4 stages x 2 residual blocks (He et al. 2016), as in the OC-Softmax paper (~2.8M parameters).

Two heads:
- "ce": two-class output, trained with weighted cross-entropy.
- "oc": one-class (OC-Softmax, Zhang et al. 2021): pulls human clips close to one direction,
  pushes AI clips away. Score = how far a clip is from "human".
Every head gives score(): higher = more AI.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

SR = 16000


class LFCC(nn.Module):
    """(B, samples) -> (B, 1, 60, frames). 20 ms windows, 10 ms hop."""

    def __init__(self, n: int = 20):
        super().__init__()
        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=SR, n_filter=n, n_lfcc=n, log_lf=True,
            speckwargs={"n_fft": 512, "win_length": 320, "hop_length": 160, "center": False})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.lfcc(x)
        d1 = torchaudio.functional.compute_deltas(c)
        d2 = torchaudio.functional.compute_deltas(d1)
        return torch.cat([c, d1, d2], 1).unsqueeze(1)


class MFM(nn.Module):
    """Max-feature-map: split the channels in two halves, keep the larger value."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, 1)
        return torch.max(a, b)


def conv(cin: int, cout: int, k: int) -> list:
    return [nn.Conv2d(cin, 2 * cout, k, padding=k // 2), MFM()]


class OCHead(nn.Module):
    """Cosine similarity to one learned "human" direction."""

    def __init__(self, dim: int):
        super().__init__()
        self.center = nn.Parameter(torch.randn(1, dim))

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return (F.normalize(e, dim=1) @ F.normalize(self.center, dim=1).T).squeeze(1)


class Detector(nn.Module):
    """Shared head and score(). Subclasses set front, body, emb and call _head()."""

    def _head(self, head: str, emb: int) -> None:
        self.head_type = head
        self.head = nn.Linear(emb, 2) if head == "ce" else OCHead(emb)

    def score(self, wav: torch.Tensor) -> torch.Tensor:
        out = self(wav)
        if self.head_type == "ce":
            return out[:, 1] - out[:, 0]  # log-odds of AI (as if human and AI were equally common)
        return -out                       # far from "human" = AI


class LCNN(Detector):
    def __init__(self, head: str = "ce", emb: int = 64, dropout: float = 0.5):
        super().__init__()
        self.front = LFCC()
        self.body = nn.Sequential(
            *conv(1, 32, 5), nn.MaxPool2d(2),
            *conv(32, 32, 1), nn.BatchNorm2d(32),
            *conv(32, 48, 3), nn.MaxPool2d(2), nn.BatchNorm2d(48),
            *conv(48, 48, 1), nn.BatchNorm2d(48),
            *conv(48, 64, 3), nn.MaxPool2d(2),
            *conv(64, 64, 1), nn.BatchNorm2d(64),
            *conv(64, 32, 3), nn.BatchNorm2d(32),
            *conv(32, 32, 1), nn.BatchNorm2d(32),
            *conv(32, 32, 3), nn.MaxPool2d(2),
        )  # 60 rows -> 3 after four 2x pools
        self.emb = nn.Sequential(nn.Dropout(dropout), nn.Linear(32 * 3, emb))
        self._head(head, emb)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        h = self.body(self.front(wav))      # B, 32, 3, frames/16
        e = self.emb(h.mean(-1).flatten(1))  # average over time
        return self.head(e)


class Block(nn.Module):
    """Two 3x3 convs plus a skip link. The skip link keeps deep networks easy to train."""

    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = nn.Identity() if stride == 1 and cin == cout else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x))))) + self.skip(x))


class ResNet18(Detector):
    def __init__(self, head: str = "ce", emb: int = 64, dropout: float = 0.5, widths=(32, 64, 128, 256)):
        super().__init__()
        self.front = LFCC()
        layers = [nn.Conv2d(1, widths[0], 3, 1, 1, bias=False), nn.BatchNorm2d(widths[0]), nn.ReLU()]
        cin = widths[0]
        for i, c in enumerate(widths):
            layers += [Block(cin, c, 1 if i == 0 else 2), Block(c, c, 1)]  # stages 2-4 halve the grid
            cin = c
        self.body = nn.Sequential(*layers)
        self.emb = nn.Sequential(nn.Dropout(dropout), nn.Linear(cin, emb))
        self._head(head, emb)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        h = self.body(self.front(wav))     # B, 256, 8, frames/8
        e = self.emb(h.mean((2, 3)))       # average over frequency and time
        return self.head(e)


ARCHS = {"lcnn": LCNN, "resnet": ResNet18}


# OC-Softmax margins and scale from Zhang, Jiang, Duan (2021).
OC_M_HUMAN, OC_M_AI, OC_ALPHA = 0.9, 0.2, 20.0


def loss_fn(model: Detector, out: torch.Tensor, y: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """Class-weighted mean loss. w = (weight for human, weight for AI)."""
    if model.head_type == "ce":
        return F.cross_entropy(out, y, weight=w)
    margin = torch.where(y == 0, OC_ALPHA * (OC_M_HUMAN - out), OC_ALPHA * (out - OC_M_AI))
    per = F.softplus(margin)
    wy = w[y]
    return (wy * per).sum() / wy.sum()

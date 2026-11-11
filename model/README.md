# Model

A small network, trained from scratch on a Mac, that scores a voice clip: human or AI.

- **Data:** ASVspoof 2019 LA. Fit on train, choose on validation (dev), report on test (eval). In-the-Wild (10% sample, 3,185 clips) is a second test set.
- **Silence:** human LA clips have long silence at the start and end; that alone gives 18.6% error. We cut it with a speech detector (Silero VAD, 30 ms margin).
- **Network:** LFCC features (inside the model) → LCNN, ~164k parameters. ~3 min per epoch on an M1 Pro.
- **Neutral:** class weights make human and AI count the same in the loss, so the 9:1 AI share is not learned.
- **Threshold:** set on validation to flag 1% of human clips; reused unchanged on the test sets.

```
uv run model/trim.py && uv run model/trim.py --set itw   # speech bounds (5 min)
uv run model/baselines.py                                 # silence-only and 17-feature baselines
uv run model/train.py --name lcnn_trim --trim vad --loss ce
```

Each run writes `runs/<name>/`: `curve.png` (train + validation loss), `checks.csv`, `steps.csv`, `results.json`.

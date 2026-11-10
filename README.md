# Real or fake audio benchmark

A standard ML benchmark for AI audio detection. It is a labeled set of real and AI-generated audio clips, plus one fixed way to score any detector on it.

## Why

AI audio models now make sound that people cannot tell from a real recording. Detectors claim to catch it, but an independent test must check those claims.

In October 2026, [Pangram's CEO asked for third-party benchmarks of AI detectors](https://x.com/max_spero_/status/2115167297937559667). This repo is one such benchmark, for audio.

## What is in it

| Part | What it is |
|---|---|
| Dataset | Audio clips, each labeled real or AI, with its source and the AI model that made it |
| Test split | One fixed set of clips, so each detector gets the same test |
| Metrics | Standard scores for this task: accuracy, AUROC, equal error rate |
| Baselines | Scores for existing detectors |
| Leaderboard | All results in one table |

## How scoring works

1. A detector gives each test clip a score: how likely it is that AI made the clip.
2. The scorer compares the scores with the labels and reports the metrics.

## Status

Work in progress.

## Author

[John Hoeksema](https://johnhoeksema.com). Independent project; not affiliated with Pangram Labs.

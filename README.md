# Real or fake audio benchmark

A benchmark that tests how well detectors tell real audio from AI-generated audio.

## Why

AI audio models now make sound that people cannot tell from a real recording. A detector is only as good as the test that checks it.

In October 2026, [Pangram's CEO asked for third-party AI detection benchmarks](https://x.com/max_spero_/status/2115167297937559667). He wants three things: very clean data, output from the newest models, and a test that is hard enough that most detectors do not max it out. This benchmark applies those three tests to audio.

## How it works

1. **Real clips:** recordings made by people, each with a known source and license.
2. **Fake clips:** audio from current AI models, matched to the real clips where possible.
3. **Score:** each detector labels each clip real or fake. The benchmark reports how often each detector is right, for each AI model and each condition.

## Status

Work in progress.

## Author

[John Hoeksema](https://johnhoeksema.com). Independent project; not affiliated with Pangram Labs.

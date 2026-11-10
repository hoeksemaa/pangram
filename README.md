# AI speech detector

A model that listens to a voice clip and tells whether a person or an AI made it.

## Why

AI voice models now sound like real people. Detectors must keep up with each new model, and must not flag real people as AI.

## How it works

1. **Data:** human speech from open corpora, and AI speech from many voice models, including clips made with current models.
2. **Model:** a pretrained speech model turns each clip into features. A small classifier on top gives a score for "AI made this."
3. **Test:** scores on datasets and voice models that the model never saw in training.

## Status

Work in progress.

## Author

[John Hoeksema](https://johnhoeksema.com). Independent project; not affiliated with Pangram Labs.

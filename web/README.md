# Web app

**Audio Detector**, a phone-first research site: home screen and four tabs. **Data** (clips and player) and **Stats** (feature histograms and counts) explore ASVspoof 2019 LA. **Model** (each trained model: what it is, loss curves, results) and **Try it** (run a model in the browser on 10 clips or a mic recording; % AI) use `model.js`. Home links: **See data** (dataset download page), **See source** (this repo on GitHub), and **John**.

```
uv run scripts/build_index.py   # once: labels + auditory features -> data/index/
uv run web/serve.py             # all clips: open http://127.0.0.1:8019/#/data
```

The audio goes in `data/raw/LA/` (from [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336), `LA.zip`).

**Live:** https://pangram-demo-hoeksema.vercel.app — 560 clips (20 from each split × system), built into `site/` and deployed by CLI. Not in git.

```
uv run --with onnxruntime scripts/build_site.py && cd site && vercel deploy --prod
```

`build_site.py` also runs `scripts/build_models.py`: every run in its `MODELS` list with `runs/<run>/model.onnx` (from `model/export.py`) goes to `site/models/`, with `site/models.json` and 10 clips in `site/try/` (5 from LA train, 5 from In-the-Wild).

- **Features:** time, pitch, voice quality, vowels, timbre, loudness. Most come from eGeMAPS ([openSMILE](https://audeering.github.io/opensmile-python/)). Tap a feature name for what it means.
- **Colors:** blue = human, red = AI, everywhere; nothing else uses them.
- **Text:** easy to read on a phone. Nothing under 15px; body 17px.
- **Histograms:** drag to filter, tap a bar for one bin, × or double-click to clear.
- **AUC:** pick one human and one AI clip at random; AUC is the chance that one feature alone tells which is which (0.5 = no clue, 1 = always). Tap it in the app. Amber (≥ 0.75) = a possible shortcut.
- **Clips:** a number (001, 002, … in index order; display only, the dataset id is the hover title) and the label. Tap a clip to see its waveform and play it.
- **Whisper's guess:** ASVspoof 2019 has no transcripts. `uv run scripts/transcribe_sample.py` guesses the text of each site clip (Whisper large-v3-turbo, MLX) into `data/transcripts/` (not in git); `build_site.py` copies it to `site/transcripts.json`. Display only: no feature or model reads it.
- **Try it:** ONNX Runtime Web (jsDelivr) runs the model: 3 s windows, mean score. % AI = sigmoid(score − line); the line flags 1% of validation humans. Recordings: silence cut at the ends, scaled to the training clips' median loudness (−17 dBFS).
- **Keys (Data tab):** `j`/`k` next/previous · `space` play · `r` random · `/` search · `c` chain · `i` clip details

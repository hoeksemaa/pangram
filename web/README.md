# Web app

**Audio Detector**, a phone-first research site: home screen and three tabs. **Data** explores ASVspoof 2019 LA. **Model** and **Try** wait for a model.

```
uv run scripts/build_index.py   # once: labels + auditory features -> data/index/
uv run web/serve.py             # all clips: open http://127.0.0.1:8019/#/data
```

The audio goes in `data/raw/LA/` (from [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336), `LA.zip`).

**Live:** https://pangram-demo-hoeksema.vercel.app — 560 clips (20 from each split × system), built into `site/` and deployed by CLI. Not in git.

```
uv run scripts/build_site.py && cd site && vercel deploy --prod
```

- **Features:** time, pitch, voice quality, vowels, timbre, loudness. Most come from eGeMAPS ([openSMILE](https://audeering.github.io/opensmile-python/)). Tap a feature name for what it means.
- **Colors:** blue = human, red = AI, everywhere; nothing else uses them.
- **Histograms:** drag to filter, tap a bar for one bin, × or double-click to clear.
- **AUC:** pick one human and one AI clip at random; AUC is the chance that one feature alone tells which is which (0.5 = no clue, 1 = always). Tap it in the app. Amber (≥ 0.75) = a possible shortcut.
- **Clips:** id and label only. Tap a clip to see its waveform and play it. **See source** opens the dataset download page.
- **Transcripts:** ASVspoof 2019 has none, so the player shows text only when the index has a `text` column.
- **Keys (Data tab):** `j`/`k` next/previous · `space` play · `r` random · `/` search · `a` autoplay · `c` chain

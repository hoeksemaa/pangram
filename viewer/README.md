# Data viewer

See every clip of ASVspoof 2019 LA, listen to it, and flag odd ones. Works on a phone.

```
uv run scripts/build_index.py   # once: labels + audio statistics -> data/index/
uv run viewer/serve.py          # all clips: open http://127.0.0.1:8019
```

The audio goes in `data/raw/LA/` (from [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336), `LA.zip`).

**Web version:** a sample of clips (same count from each split × system), built into `site/` and deployed by CLI. Not in git.

```
uv run scripts/build_site.py && cd site && vercel deploy --prod
```

- **Keys:** `j`/`k` next/previous · `space` play · `r` random · `f` note · `/` search · `a` autoplay · `c` chain
- **Histograms:** blue = human, orange = AI. Drag to filter, tap a bar for one bin, × or double-click to clear.
- **AUC:** how well one statistic alone separates human from AI. Red (≥ 0.75) = a possible shortcut.
- **Flags:** local server saves to `notes/flags.jsonl`; web version keeps them in the browser (export button).

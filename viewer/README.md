# Data viewer

See every clip of ASVspoof 2019 LA, listen to it, and flag odd ones.

```
uv run scripts/build_index.py   # once: labels + audio statistics -> data/index/
uv run viewer/serve.py          # then open http://127.0.0.1:8019
```

The audio goes in `data/raw/LA/` (from [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3336), `LA.zip`).

- **Keys:** `j`/`k` next/previous · `space` play · `r` random · `f` note · `/` search · `a` autoplay · `c` chain
- **Histograms:** blue = human, orange = AI. Drag to filter, click a bar for one bin, double-click to clear.
- **AUC:** how well one statistic alone separates human from AI. Red (≥ 0.75) = a possible shortcut.
- **Flags:** saved to `notes/flags.jsonl`.

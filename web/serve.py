"""Local server for the site, with all clips. Serves the page, the clip index, and audio.

Usage:
    uv run web/serve.py               # then open http://127.0.0.1:8019/#/data
"""

import argparse
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
STATIC = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/explorer.js": ("explorer.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
AUDIO_RE = re.compile(r"^/audio/(train|dev|eval)/(LA_[TDE]_A?\d+)\.flac$")
TRANSCRIPTS = REPO / "data/transcripts/whisper-sample.json"  # optional, display only
RANGE_RE = re.compile(r"^bytes=(\d*)-(\d*)$")


def make_handler(root: Path, index: Path):
    class Handler(BaseHTTPRequestHandler):
        def send(self, code: int, body: bytes, ctype: str, cache: str = "no-cache", extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache)
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def send_file(self, path: Path, ctype: str, cache: str = "no-cache") -> None:
            if not path.is_file():
                self.send(404, f"not found: {path.name}".encode(), "text/plain")
                return
            data = path.read_bytes()
            m = RANGE_RE.match(self.headers.get("Range", ""))
            if not m or not (m.group(1) or m.group(2)):
                self.send(200, data, ctype, cache, {"Accept-Ranges": "bytes"})
                return
            size = len(data)  # byte ranges: Safari needs them to play audio
            if m.group(1):
                start, end = int(m.group(1)), int(m.group(2)) if m.group(2) else size - 1
            else:
                start, end = max(0, size - int(m.group(2))), size - 1
            end = min(end, size - 1)
            if start > end:
                self.send(416, b"", "text/plain", cache, {"Content-Range": f"bytes */{size}"})
                return
            self.send(206, data[start : end + 1], ctype, cache,
                      {"Accept-Ranges": "bytes", "Content-Range": f"bytes {start}-{end}/{size}"})

        def do_GET(self) -> None:
            path = self.path.split("?", 1)[0]
            if path in STATIC:
                name, ctype = STATIC[path]
                self.send_file(HERE / name, ctype)
            elif path == "/index.json":
                self.send_file(index, "application/json")
            elif path == "/transcripts.json":
                self.send_file(TRANSCRIPTS, "application/json")
            elif m := AUDIO_RE.match(path):
                split, cid = m.groups()
                self.send_file(root / f"ASVspoof2019_LA_{split}" / "flac" / f"{cid}.flac", "audio/flac", "max-age=86400")
            else:
                self.send(404, b"not found", "text/plain")

        def log_message(self, *args) -> None:  # quiet
            pass

    return Handler


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=REPO / "data/raw/LA")
    ap.add_argument("--index", type=Path, default=REPO / "data/index/asvspoof2019_la.json")
    ap.add_argument("--port", type=int, default=8019)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.root, args.index))
    print(f"viewer: http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audio sheets: the sound of each stretch of a video drawn under its shots, to look at.

One picture per --window seconds, time running left to right over the full width. Rows,
top to bottom (sheets.json repeats this with each row's pixel range):
  time ruler    a tick every second, a taller one with MM:SS every 5 s
  shots         each shot's sharpest frame, its left edge on the shot's first moment
  cuts          red ticks; each cut also runs as a red line down through the sound rows
  hits          yellow ticks at the sound hits (index.json `onsets`)
  beats         cyan ticks at the beats, when the music has a clear one
  spectrogram   40 Hz (bottom) to 16 kHz (top), log scale, brighter = louder
  waveform      with a faint yellow line at each hit
  loudness      momentary loudness as green columns, -60 LUFS (empty) to 0 (full);
                silences have a blue background

The numbers behind every mark are in index.md and index.json; the sheet is for seeing
how they sit together. Marks are drawn by this script as one transparent image laid over
ffmpeg's spectrogram and waveform (ffmpeg's drawtext is often missing, and hundreds of
drawbox filters would not fit a Windows command line).

Usage:
  python3 sheet.py VIDEO --out-dir DIR [--index DIR/index.json] [--window 30] [--start T] [--end T]

Without --index it runs index.py into DIR first. Writes DIR/sheet_000.png ... and
DIR/sheets.json; prints `SHEETS: <path to sheets.json>` last. ffmpeg and Python stdlib only.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import frames  # noqa: E402

HERE = Path(__file__).resolve().parent
WIDTH = frames.MODEL_LONG_EDGE      # the long edge the model sees, so nothing is shrunk
ROWS = (("time ruler", 24), ("shots", 96), ("cuts", 12), ("hits", 12), ("beats", 12),
        ("spectrogram", 256), ("waveform", 96), ("loudness", 64))
TOP = {name: sum(h for _, h in ROWS[:i]) for i, (name, _) in enumerate(ROWS)}
TALL = dict(ROWS)
HEIGHT = sum(TALL.values())
CUT, HIT, BEAT = (255, 70, 70), (255, 220, 0), (0, 220, 255)
LOUD, SILENT, LANE, EDGE, TICK = (90, 220, 120), (30, 45, 110), (34, 34, 34), (70, 70, 70), (200, 200, 200)
LOUD_FLOOR = -60.0                  # LUFS drawn as an empty column
# ffmpeg sizes the spectrogram's frequency analysis by the picture's height. Drawn straight
# at 256 px, everything under 300 Hz came from three frequency bins and smeared into
# columns; drawn at 1024 px and shrunk, the partials of a chime are separate lines (2048 px
# blurred two pops 130 ms apart into one).
SPEC_RENDER = 1024
THUMB_GAP = 6                       # a shot starting closer than this (px) to the last thumbnail gets none
MAX_THUMBS = 80


def draw_marks(idx: dict, start: float, window: float, labels: Path | None = None) -> bytearray:
    """The transparent RGBA layer for one sheet: ruler, lanes, marker lines, loudness."""
    img = bytearray(WIDTH * HEIGHT * 4)

    def box(x: int, y: int, w: int, h: int, rgb: tuple, alpha: int = 255) -> None:
        x, w = max(0, x), min(w, WIDTH - max(0, x))
        if w > 0:
            px = bytes((*rgb, alpha)) * w
            for row in range(y, y + h):
                img[(row * WIDTH + x) * 4:(row * WIDTH + x + w) * 4] = px

    def at(t: float) -> int:
        return int(round((t - start) / window * WIDTH))

    def inside(times: list[float]) -> list[int]:
        return [at(t) for t in times if start <= t < start + window]

    for name in ("cuts", "hits", "beats"):
        box(0, TOP[name], WIDTH, TALL[name], LANE)
    for name, _ in ROWS[1:]:
        box(0, TOP[name], WIDTH, 1, EDGE)

    for sec in range(math.ceil(start), math.floor(start + window) + 1):
        tall = sec % 5 == 0
        box(at(sec), TALL["time ruler"] - (10 if tall else 4), 1, 10 if tall else 4, TICK)
        if tall and labels is not None and at(sec) + 60 < WIDTH:
            frames.label_pgm(common.clock(sec * 1000)[:-2], 2, labels)       # MM:SS
            head, _, data = labels.read_bytes().partition(b"\n")
            w, h = int(head.split()[1]), int(head.split()[2])
            for row in range(min(h, TALL["time ruler"])):
                for col in range(w):
                    if data[row * w + col]:
                        box(at(sec) + 3 + col, row, 1, 1, TICK)

    loud = idx.get("loudness")
    y0, tall = TOP["loudness"], TALL["loudness"]
    for s in idx.get("silences") or []:
        a, b = max(start, s["start_s"]), min(start + window, s["end_s"])
        if b > a:
            box(at(a), y0 + 1, at(b) - at(a), tall - 1, SILENT)
    if loud:
        for x in range(WIDTH):
            i = int((start + (x + 0.5) * window / WIDTH - loud["start_s"]) / loud["step_s"])
            if 0 <= i < len(loud["momentary"]):
                h = int(min(1.0, max(0.0, 1 - loud["momentary"][i] / LOUD_FLOOR)) * (tall - 1))
                box(x, y0 + tall - h, 1, h, LOUD)

    sound_rows = TOP["spectrogram"], HEIGHT - TOP["spectrogram"]
    for x in inside([s["start_s"] for s in idx.get("shots", [])[1:]]):
        box(x, TOP["cuts"], 3, TALL["cuts"], CUT)
        box(x, sound_rows[0], 1, sound_rows[1], CUT, 170)
    for x in inside([o["t_s"] for o in idx.get("onsets") or []]):
        box(x, TOP["hits"], 2, TALL["hits"], HIT)
        box(x, TOP["waveform"] + 1, 1, TALL["waveform"] - 1, HIT, 120)
    for x in inside((idx.get("tempo") or {}).get("beats", [])):
        box(x, TOP["beats"], 2, TALL["beats"], BEAT)
    return img


def write_pam(img: bytearray, dest: Path) -> None:
    dest.write_bytes(f"P7\nWIDTH {WIDTH}\nHEIGHT {HEIGHT}\nDEPTH 4\nMAXVAL 255\nTUPLTYPE RGB_ALPHA\nENDHDR\n".encode()
                     + bytes(img))


def place_thumbs(shots: list[dict], start: float, window: float) -> list[tuple[int, dict]]:
    """(x, shot) for the shots showing in this window; a shot already running starts at x 0."""
    out: list[tuple[int, dict]] = []
    for s in shots:
        if s["end_s"] <= start or s["start_s"] >= start + window:
            continue
        x = int(round((max(start, s["start_s"]) - start) / window * WIDTH))
        if not out or x - out[-1][0] >= THUMB_GAP:
            out.append((x, s))
    return out[:MAX_THUMBS]


def compose(video: str, start: float, window: float, thumbs: list[tuple[int, str]], png: Path, stage: Path) -> bool:
    """ffmpeg lays spectrogram, waveform and thumbnails on a black canvas, then marks.pam over it.
    Runs inside `stage` with bare file names for everything but the video and the output."""
    seek, cut = common.seek_input(video, start, start + window)
    graph = [f"color=black:s={WIDTH}x{HEIGHT},format=rgb24[bg]",
             f"[0:a]{common.trim(cut, audio=True)}aresample=48000,apad=whole_dur={window},atrim=0:{window},asplit[a1][a2]",
             f"[a1]showspectrumpic=s={WIDTH}x{SPEC_RENDER}:legend=0:fscale=log:saturation=0:start=40:stop=16000,"
             f"scale={WIDTH}:{TALL['spectrogram']}:flags=area[sp]",
             f"[a2]aformat=channel_layouts=mono,showwavespic=s={WIDTH}x{TALL['waveform']}:colors=0xd0d0d0:filter=peak[wv]",
             f"[bg][sp]overlay=0:{TOP['spectrogram']}:format=rgb[c0]",
             f"[c0][wv]overlay=0:{TOP['waveform']}:format=rgb[c1]"]
    last = "c1"
    for k, (x, _) in enumerate(thumbs):
        graph += [f"[{k + 1}:v]scale=-2:{TALL['shots'] - 1}[t{k}]",
                  f"[{last}][t{k}]overlay={x}:{TOP['shots'] + 1}:format=rgb[d{k}]"]
        last = f"d{k}"
    graph.append(f"[{last}][{len(thumbs) + 1}:v]overlay=0:0:format=rgb")
    inputs = [arg for _, name in thumbs for arg in ("-i", name)]
    r = frames.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *seek, "-i", video, *inputs,
                    "-i", "marks.pam", "-filter_complex", ";".join(graph), "-frames:v", "1", "-update", "1",
                    str(png.resolve())], cwd=stage)
    if r.returncode:
        print(f"[sheet] ffmpeg failed: {r.stderr.strip()[-300:]}", file=sys.stderr)
    return r.returncode == 0 and png.exists()


def main() -> None:
    ap = argparse.ArgumentParser(description="Audio sheets: spectrogram, waveform, loudness and marks under the shots.")
    ap.add_argument("video")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--index", help="index.json from index.py (default: build one in --out-dir).")
    ap.add_argument("--window", type=float, default=30.0, help="Seconds per sheet (default 30; 60 for an overview).")
    ap.add_argument("--start")
    ap.add_argument("--end")
    args = ap.parse_args()
    common.need("ffmpeg")
    common.need("ffprobe")
    video = str(Path(args.video).expanduser().resolve())
    if not Path(video).exists():
        sys.exit(f"no such file: {video}")
    meta = common.probe(video)
    if not meta["has_audio"]:
        sys.exit(f"no audio in {video}: nothing to draw")
    if not 5 <= args.window <= 300:
        sys.exit("--window must be 5 to 300 seconds")
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    index_path = Path(args.index).expanduser() if args.index else out / "index.json"
    if not args.index:
        r = frames.run([sys.executable, "-I", str(HERE / "index.py"), video, "--out-dir", str(out)])
        if r.returncode:
            sys.exit(f"index.py failed: {r.stderr[-300:]}")
    idx = json.loads(index_path.read_text(encoding="utf-8"))

    lo = common.parse_time(args.start) or 0.0
    hi = min(common.parse_time(args.end) or meta["duration_s"], meta["duration_s"])
    if hi <= lo:
        sys.exit("empty time range")
    starts = [lo + k * args.window for k in range(math.ceil((hi - lo) / args.window))]
    plans = [place_thumbs(idx.get("shots", []), a, args.window) for a in starts]

    work = common.make_tmp("sheet")
    sheets = []
    try:
        wanted = sorted({s["sharpest_s"] for plan in plans for _, s in plan})
        got = {f["t_ms"]: f["path"] for f in frames.grab(video, work, 240, wanted, meta["start_s"])} if wanted else {}
        for n, (a, plan) in enumerate(zip(starts, plans)):
            thumbs = [(x, Path(got[int(round(s["sharpest_s"] * 1000))]).name) for x, s in plan
                      if int(round(s["sharpest_s"] * 1000)) in got]
            write_pam(draw_marks(idx, a, args.window, work / "label.pgm"), work / "marks.pam")
            png = out / f"sheet_{n:03d}.png"
            if compose(video, a, args.window, thumbs, png, work):
                sheets.append({"path": str(png), "start_s": round(a, 3), "end_s": round(min(a + args.window, hi), 3)})
    finally:
        common.safe_rmtree(work)
    if not sheets:
        sys.exit(3)

    rows = [{"name": name, "y": [TOP[name], TOP[name] + h]} for name, h in ROWS]
    manifest = {"video": video, "window_s": args.window, "width": WIDTH, "height": HEIGHT,
                "x_of_time": "x = (t - sheet start_s) / window_s * width", "rows": rows,
                "colours": {"cuts": "red", "hits": "yellow", "beats": "cyan", "loudness": "green", "silence": "blue"},
                "index": str(index_path), "sheets": sheets}
    path = out / "sheets.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[sheet] {len(sheets)} sheets of {args.window:g} s, {WIDTH}x{HEIGHT}", file=sys.stderr)
    print(f"SHEETS: {path.resolve()}")


if __name__ == "__main__":
    main()

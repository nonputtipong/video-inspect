#!/usr/bin/env python3
"""Pick the frames worth looking at and tile them into timestamped grid images.

Default ("shots"): one frame per shot at its sharpest moment, then extra frames spread
across the range (denser on short clips, and in high-motion shots: still shots get half
the base rate, the busiest quarter twice it) so long takes aren't a single still.
Near-identical neighbours are dropped, except a shot's own frame, and on clips up to
3 minutes no gap between kept frames grows past --max-gap. The shot list comes from
index.py (pass --index to reuse one) and the frame budget grows to one frame per shot.

Other modes: --force keyframe (encoder I-frames), --force uniform (fixed fps), and
--at 4:32,7:10 for exactly one frame at each named moment (no dropping).

Grids read left to right, top to bottom, and every cell has its time stamped in the
corner. ALWAYS read the manifest with the grids: it holds each cell's exact time.

Usage:
  python3 frames.py VIDEO --out-dir DIR [--index DIR/index.json] [--cells 64]
                          [--start T] [--end T] [--max-grids N] [--budget N] [--no-dedup]
  python3 frames.py VIDEO --out-dir DIR --at 4:32,7:10 [--cells 16]

Writes DIR/grid_000.png ... and DIR/manifest.json
({grids: [{path, cols, rows, cells: [{index, t_ms, t_label, shot?}]}], ...});
prints `MANIFEST: <path>` last. Exit 3 if no frames came out.
Requires ffmpeg 5.1+ and ffprobe on PATH. Python stdlib only.
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import index as shot_index  # noqa: E402

DEFAULT_BUDGET = 512      # frames for a whole run: 8 grids of 64, whatever the length
SHOT_BUDGET_CAP = 2048    # the budget rises to one frame per shot, up to this many
SAME_PICTURE = 6.0        # largest 4x4-block change (0-255, 32px thumbs) that counts as no change
MAX_GAP_S = 0.5           # longest gap between kept frames on clips up to 3 minutes
MODEL_LONG_EDGE = 1568    # images are shrunk to this long edge before the model sees them

# Frames are grabbed by decoding short stretches of the video, in parallel: requests
# closer together than JOIN_GAP_S share one decode (cheaper than seeking for each),
# farther apart they start their own (cheaper than decoding the gap). A stretch holding
# a single request is exactly a seek. Measured on a 1080p60 clip: 228 frames took 13.9 s
# seeking one by one, 1.6 s decoding once.
JOIN_GAP_S = 2.0
MAX_PER_STRETCH = 64
PTS_RE = re.compile(r"pts_time:\s*(-?[\d.]+)")


def run(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def grab(video: str, folder: Path, width: int, times: list[float], origin: float) -> list[dict]:
    """The first frame at or after each time (seconds on the video's clock from 0), like an
    accurate seek would give. `origin` is the file's start timestamp (usually 0)."""
    want = sorted(set(times))
    early = common.seek_margin(video)
    stretches: list[list[float]] = []
    for t in want:
        if stretches and t - stretches[-1][-1] <= JOIN_GAP_S and len(stretches[-1]) < MAX_PER_STRETCH:
            stretches[-1].append(t)
        else:
            stretches.append([t])

    def decode(k: int) -> list[tuple[float, str]]:
        part = stretches[k]
        lead = min(max(0.05, early), part[0])   # select= then keeps exactly the wanted frames
        tests = "+".join(f"gte(t,{t + origin:.6f})*(isnan(prev_t)+lt(prev_t,{t + origin:.6f}))" for t in part)
        r = run(["ffmpeg", "-hide_banner", "-loglevel", "info", "-copyts",
                 "-ss", f"{part[0] - lead:.6f}", "-t", f"{part[-1] - part[0] + lead + 1:.6f}", "-i", video,
                 "-vf", f"select='{tests}',scale={width}:-2,showinfo", "-fps_mode", "passthrough",
                 "-q:v", "4", f"s{k:04d}_%04d.jpg"], cwd=folder)
        stamps = [float(x) - origin for x in PTS_RE.findall(r.stderr)]
        files = sorted(folder.glob(f"s{k:04d}_*.jpg"))
        return list(zip(stamps, map(str, files)))

    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
        got = sorted(x for part in pool.map(decode, range(len(stretches))) for x in part)
    stamps = [t for t, _ in got]
    out = []
    for t in times:
        i = bisect.bisect_left(stamps, t - 1e-6)
        if i < len(got):
            out.append({"path": got[i][1], "t_ms": int(round(t * 1000))})
        else:
            print(f"[frames] no frame at {common.clock(int(t * 1000))} (past the end?)", file=sys.stderr)
    return out


def keyframes(video: str, folder: Path, width: int, start: float | None, end: float | None) -> list[dict]:
    seek, cut = common.seek_input(video, start, end)
    r = run(["ffmpeg", "-hide_banner", "-loglevel", "info", "-skip_frame", "nokey", *seek, "-i", video,
             "-vf", f"{common.trim(cut)}scale={width}:-2,showinfo", "-fps_mode", "vfr", "-q:v", "4", "k_%05d.jpg"], cwd=folder)
    stamps = [float(x) for x in PTS_RE.findall(r.stderr)]
    files = sorted(folder.glob("k_*.jpg"))
    return [{"path": str(f), "t_ms": int(round(((stamps[i] if i < len(stamps) else 0.0) + (start or 0.0)) * 1000))}
            for i, f in enumerate(files)]


def uniform(video: str, folder: Path, width: int, fps: float, start: float | None, end: float | None) -> list[dict]:
    seek, cut = common.seek_input(video, start, end)
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", *seek, "-i", video,
         "-vf", f"{common.trim(cut)}fps={fps},scale={width}:-2", "-q:v", "4", "u_%05d.jpg"], cwd=folder)
    return [{"path": str(f), "t_ms": int(round(((start or 0.0) + i / fps) * 1000))}
            for i, f in enumerate(sorted(folder.glob("u_*.jpg")))]


def drop_repeats(frames: list[dict], max_gap_s: float = 0.0) -> tuple[list[dict], int]:
    """Keep a frame only if it differs from the last kept one, is a shot's own frame, or
    would otherwise leave a gap of max_gap_s or more. If the thumbnails can't be made,
    keep everything rather than drop frames unchecked."""
    thumbs = common.gray_thumbs([Path(f["path"]) for f in frames])
    if len(frames) < 2 or len(thumbs) != len(frames):
        return frames, 0
    kept, ref = [frames[0]], thumbs[0]
    for f, th in zip(frames[1:], thumbs[1:]):
        gap = (f["t_ms"] - kept[-1]["t_ms"]) / 1000
        if (f.get("anchor") or (max_gap_s and gap >= max_gap_s)
                or common.region_delta(th, ref) > SAME_PICTURE):
            kept.append(f)
            ref = th
    return kept, len(frames) - len(kept)


def spread(items: list, n: int) -> list:
    """n items evenly spaced through the list, first and last included."""
    if n >= len(items):
        return items
    if n <= 1:
        return items[:1]
    return [items[round(k * (len(items) - 1) / (n - 1))] for k in range(n)]


def plan_times(shots: list[dict], budget: int, lo: float, hi: float, fill_cap: float) -> list[dict]:
    """Each shot's sharpest moment, then the rest of the budget as fill weighted by motion:
    a still shot gets half the base rate, a high-motion shot twice it (base at most
    fill_cap a second). Fill within 0.3 s of a shot's frame is skipped."""
    anchors = spread([{"t": s["sharpest_s"], "anchor": True, "shot": s["index"]} for s in shots], budget)
    marks = sorted(a["t"] for a in anchors)
    plan = list(anchors)
    room = budget - len(anchors)
    still_below, busy_from = shot_index.motion_bands(shots)

    def weight(s: dict) -> float:
        m = s.get("motion")
        if m is None:
            return 1.0
        return 0.5 if m < still_below else 2.0 if m >= busy_from > still_below else 1.0

    segs = [(max(lo, s["start_s"]), min(hi, s["end_s"]), weight(s)) for s in shots]
    segs = [g for g in segs if g[1] > g[0]] or [(lo, hi, 1.0)]
    total = sum((b - a) * w for a, b, w in segs)   # motion-weighted seconds
    if room > 0 and total > 0:
        rate = min(fill_cap, room / total)
        n, k, u0 = int(total * rate), 0, 0.0
        for a, b, w in segs:   # evenly spaced on the weighted clock, so busy shots get more
            u1 = u0 + (b - a) * w
            while k < n and (k + 0.5) / rate < u1:
                t = a + ((k + 0.5) / rate - u0) / w
                k += 1
                i = bisect.bisect_left(marks, t)
                if all(abs(marks[j] - t) >= 0.3 for j in (i - 1, i) if 0 <= j < len(marks)):
                    plan.append({"t": t, "anchor": False})
            u0 = u1
    return sorted(plan, key=lambda p: p["t"])


def layout(cells: int, cell_aspect: float) -> tuple[int, int]:
    """Columns and rows for a near-square grid. Portrait cells get more columns so a phone
    video's grid isn't a tall strip that the long-edge shrink crushes."""
    if cell_aspect >= 1:
        cols = int(cells ** 0.5)
        cols += cols * cols < cells
    else:
        cols = min(cells, max(1, round((cells / cell_aspect) ** 0.5)))
    return cols, -(-cells // cols)


# Seven-segment digits for the time stamps, drawn as a binary PGM (stdlib only; ffmpeg's
# drawtext needs a freetype build many installs lack). Segments: a top, b upper right,
# c lower right, d bottom, e lower left, f upper left, g middle.
SEGMENTS = {"0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg",
            "5": "acdfg", "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg"}
SEG_BOXES = {"a": (0, 0, 4, 1), "b": (3, 0, 1, 4), "c": (3, 3, 1, 4), "d": (0, 6, 4, 1),
             "e": (0, 3, 1, 4), "f": (0, 0, 1, 4), "g": (0, 3, 4, 1)}   # x, y, w, h in units
DOTS = {":": [(0, 2), (0, 4)], ".": [(0, 6)]}                         # one-unit dots at x, y


def label_pgm(text: str, unit: int, dest: Path) -> None:
    def advance(ch: str) -> int:
        return 2 if ch in DOTS else 5
    pad = unit
    w = sum(advance(c) for c in text) * unit + 2 * pad
    h = 7 * unit + 2 * pad
    img = bytearray(w * h)

    def fill(x: int, y: int, bw: int, bh: int) -> None:
        for row in range(pad + y * unit, pad + (y + bh) * unit):
            start = row * w + pad + x
            img[start:start + bw * unit] = b"\xff" * (bw * unit)

    x = 0
    for ch in text:
        if ch == "1":   # a seven-segment 1 hugs the right edge and reads as " 1"; centre it
            fill(x + 2 * unit, 0, 1, 7)
        for seg in SEGMENTS.get(ch, "") if ch != "1" else "":
            sx, sy, sw, sh = SEG_BOXES[seg]
            fill(x + sx * unit, sy, sw, sh)
        for dx, dy in DOTS.get(ch, []):
            fill(x + dx * unit, dy, 1, 1)
        x += advance(ch) * unit
    dest.write_bytes(f"P5 {w} {h} 255\n".encode() + bytes(img))


def make_grid(frames: list[dict], cols: int, rows: int, cell_w: int, cell_h: int,
              png: Path, work: Path, labels: bool) -> bool:
    stage = work / f"grid_{png.stem}"
    stage.mkdir()
    texts = [f.get("label") or common.clock(f["t_ms"]) for f in frames]
    width_chars = max(map(len, texts))
    unit = max(1, min(cell_h // 50, int(cell_w * 0.5) // (5 * width_chars)))
    for i, (f, text) in enumerate(zip(frames, texts)):
        shutil.copy(f["path"], stage / f"c_{i:03d}.jpg")
        if labels:   # same label size across a grid, or ffmpeg restarts the tile mid-grid
            label_pgm(text.rjust(width_chars), unit, stage / f"t_{i:03d}.pgm")
    tile = f"tile={cols}x{rows}:margin=6:padding=4:color=white"
    if labels:
        graph = ["-i", "t_%03d.pgm", "-filter_complex",
                 f"[0:v]scale={cell_w}:{cell_h},setsar=1[c];[c][1:v]overlay=0:0,{tile}"]
    else:
        graph = ["-vf", f"scale={cell_w}:{cell_h},setsar=1,{tile}"]
    r = run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", "c_%03d.jpg", *graph,
             "-frames:v", "1", str(png.resolve())], cwd=stage)
    common.safe_rmtree(stage)
    return r.returncode == 0 and png.exists()


def picture_aspect(path: str) -> float:
    """Width/height of an extracted frame, so rotation metadata is already applied."""
    r = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-of", "csv=p=0", path])
    try:
        w, h = map(int, r.stdout.strip().split(",")[:2])
        return w / h
    except (ValueError, ZeroDivisionError):
        return 16 / 9


def main() -> None:
    ap = argparse.ArgumentParser(description="Pick, de-duplicate and grid-pack video frames.")
    ap.add_argument("video")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cells", type=int, default=64,
                    help="Cells per grid: 16 to read on-screen text, 36 for faces and fine "
                         "action, 64 for plot and scenes (default).")
    ap.add_argument("--fps", type=float, default=0.0,
                    help="Sampling rate for --force uniform (default: from --budget).")
    ap.add_argument("--budget", type=int, default=None,
                    help=f"Most frames for the whole run (default {DEFAULT_BUDGET}, raised to one "
                         f"per shot up to {SHOT_BUDGET_CAP}). Thinned evenly, never cut at the end. 0 = no cap.")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--max-grids", type=int, default=0,
                    help="Most grids (0 = no cap). Frames are thinned evenly across the range.")
    ap.add_argument("--cell-width", type=int, default=480)
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--max-gap", type=float, default=None,
                    help="Longest gap between kept frames in seconds (default 0.5 up to 3 min, off beyond).")
    ap.add_argument("--no-labels", action="store_true", help="Leave the time stamps off the cells.")
    ap.add_argument("--force", choices=["shots", "keyframe", "uniform"],
                    help="Frame picker (default shots).")
    ap.add_argument("--index", help="index.json from index.py, to reuse its shots instead of decoding again.")
    ap.add_argument("--at", help="Comma-separated moments (SS, MM:SS or HH:MM:SS): exactly one frame "
                                 "each, e.g. where the transcript says 'look at this'. No dropping.")
    args = ap.parse_args()
    common.need("ffmpeg")
    common.need("ffprobe")
    video = str(Path(args.video).expanduser().resolve())
    if not Path(video).exists():
        sys.exit(f"no such file: {video}")
    if args.cells < 1:
        sys.exit("--cells must be >= 1")
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = common.probe(video)
    if not meta["width"]:
        sys.exit(f"no video stream in {video}")
    start, end = common.parse_time(args.start), common.parse_time(args.end)
    lo = start or 0.0
    hi = end if end is not None else meta["duration_s"]
    if hi - lo <= 0:
        sys.exit("empty time range")
    max_gap = args.max_gap if args.max_gap is not None else (MAX_GAP_S if hi - lo <= 180 else 0.0)
    budget = DEFAULT_BUDGET if args.budget is None else args.budget
    engine = "cues" if args.at else (args.force or "shots")
    fps, dropped, shot_count = 0.0, 0, 0

    work = common.make_tmp("frames")
    raw = work / "raw"
    raw.mkdir()
    try:
        if engine == "cues":
            cues = sorted(common.parse_time(t) for t in args.at.split(",") if t.strip())
            frames = grab(video, raw, args.cell_width, cues, meta["start_s"])
            print(f"[frames] {len(frames)}/{len(cues)} cue frames", file=sys.stderr)
        elif engine == "shots":
            if args.index:
                idx = json.loads(Path(args.index).expanduser().read_text(encoding="utf-8"))
                shots = [s for s in idx["shots"] if s["end_s"] > lo and s["start_s"] < hi]
            else:
                (work / "idx").mkdir()
                shots = shot_index.shot_list(video, start, end, work / "idx")
            shot_count = len(shots)
            if args.budget is None:
                budget = min(max(DEFAULT_BUDGET, len(shots)), SHOT_BUDGET_CAP)
            span = hi - lo
            fill_cap = 4.0 if span <= 60 else 2.0 if span <= 300 else 1.0
            plan = plan_times(shots, budget if budget > 0 else max(len(shots), 1) * 4, lo, hi, fill_cap)
            frames = grab(video, raw, args.cell_width, [p["t"] for p in plan], meta["start_s"])
            by_ms = {int(round(p["t"] * 1000)): p for p in plan}
            for f in frames:
                p = by_ms.get(f["t_ms"], {})
                f["anchor"] = p.get("anchor", False)
                if "shot" in p:
                    f["shot"] = p["shot"]
            if not args.no_dedup:
                frames, dropped = drop_repeats(frames, max_gap)
            anchors = sum(1 for f in frames if f.get("anchor"))
            print(f"[frames] {len(shots)} shots -> {anchors} shot anchors + {len(frames) - anchors} fill frames",
                  file=sys.stderr)
        else:
            if engine == "uniform":
                fps = args.fps if args.fps > 0 else max(min(4.0, budget / (hi - lo)) if budget > 0 else 4.0, 0.05)
                frames = uniform(video, raw, args.cell_width, fps, start, end)
            else:
                frames = keyframes(video, raw, args.cell_width, start, end)
            if not args.no_dedup:
                frames, dropped = drop_repeats(frames, max_gap)

        if not frames:
            sys.exit(3)
        extracted = len(frames) + dropped
        thinned = 0
        if 0 < budget < len(frames):
            thinned = len(frames) - budget
            frames = spread(frames, budget)
        if args.max_grids > 0:
            frames = spread(frames, args.max_grids * args.cells)

        aspect = picture_aspect(frames[0]["path"])
        cell_w = args.cell_width if aspect >= 1 else int(round(args.cell_width * aspect / 2)) * 2
        cell_h = int(round(cell_w / aspect / 2)) * 2
        cols, rows = layout(args.cells, cell_w / cell_h)
        seen_w = min(cell_w, cell_w * MODEL_LONG_EDGE / max(cols * cell_w, rows * cell_h))
        note = f", {thinned} thinned to fit --budget {budget}" if thinned else ""
        print(f"[frames] {extracted} extracted, {dropped} near-duplicates dropped{note}, {len(frames)} kept "
              f"-> {cols}x{rows} grids (~{seen_w:.0f}px wide per cell after downscale)", file=sys.stderr)

        grids = []
        for g in range(0, len(frames), args.cells):
            chunk = frames[g:g + args.cells]
            png = out_dir / f"grid_{g // args.cells:03d}.png"
            c, r = (cols, rows) if len(chunk) == args.cells else layout(len(chunk), cell_w / cell_h)
            if make_grid(chunk, c, r, cell_w, cell_h, png, work, not args.no_labels):
                grids.append({"path": str(png), "cols": c, "rows": r,
                              "cells": [{"index": i, "t_ms": f["t_ms"], "t_label": common.clock(f["t_ms"]),
                                         **({"shot": f["shot"]} if "shot" in f else {})}
                                        for i, f in enumerate(chunk)]})
        if not grids:
            sys.exit(3)

        manifest = {
            "video": video, "duration_s": round(meta["duration_s"], 2), "has_audio": meta["has_audio"],
            "engine": engine, "shots": shot_count or None, "cells_per_grid": args.cells,
            "grid_cols": cols, "grid_rows": rows,   # a short last grid has its own cols/rows
            "labels": not args.no_labels, "frames_extracted": extracted, "frames_deduped": dropped,
            "frames_thinned_for_budget": thinned, "budget": budget,
            "sampling_fps": round(fps, 3) if engine == "uniform" else None,
            "frames_used": len(frames), "range": {"start_s": start, "end_s": end},
            "reading_order": "left-to-right, top-to-bottom", "grids": grids,
        }
        path = out_dir / "manifest.json"
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"MANIFEST: {path.resolve()}")
    finally:
        common.safe_rmtree(work)


if __name__ == "__main__":
    main()

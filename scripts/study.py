#!/usr/bin/env python3
"""Study reference videos: cut rhythm, sound hits and how they sit on the cuts, speaking pace,
on-screen label timeline.

Usage:
  python3 study.py VIDEO [VIDEO ...] --out-dir DIR [--no-transcript] [--no-ocr] [--lang en]
Output: DIR/study.md (one table row per video, then per-video timelines), DIR/<video>/index.*,
        DIR/<video>/transcript.json; prints `STUDY: <path>`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description="Reference-video study (local, free).")
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--no-transcript", action="store_true")
    ap.add_argument("--no-ocr", action="store_true")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    rows, details = [], []
    for v in args.videos:
        v = str(Path(v).expanduser().resolve())
        d = out / Path(v).stem
        cmd = [sys.executable, "-I", str(HERE / "index.py"), v, "--out-dir", str(d), "--min-silence", "0.3",
               "--ocr-lang", args.lang] + ([] if args.no_ocr else ["--ocr"])
        subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        idx = json.loads((d / "index.json").read_text(encoding="utf-8"))
        dur = idx["duration_s"] or 1
        wpm, nwords = "–", 0
        if not args.no_transcript:
            t = d / "transcript.json"
            subprocess.run([sys.executable, "-I", str(HERE / "transcript.py"), v, "--out", str(t),
                            "--lang", args.lang], capture_output=True, text=True, encoding="utf-8", errors="replace")
            if t.exists():
                tx = json.loads(t.read_text(encoding="utf-8"))
                ws = tx["words"]
                nwords = len(ws)
                if ws:
                    wpm = f"{nwords / max(0.1, ws[-1]['end_s'] - ws[0]['start_s']) * 60:.0f}"
        shots = idx["shot_count"]
        precise = idx.get("onsets") is not None      # 5 ms hit onsets; else the coarse loudness jumps
        hit_times = [e["t_s"] for e in (idx["onsets"] if precise else idx["audio_events"])]
        hits = len(hit_times)
        sil = sum(s["end_s"] - s["start_s"] for s in idx["silences"])
        sync = idx.get("sync") or {}
        with_hit = "–" if sync.get("pct_with_hit") is None else f"{sync['pct_with_hit']}%"
        offset = "–" if sync.get("median_offset_ms") is None else f"{sync['median_offset_ms']:+d} ms"
        rows.append(f"| {Path(v).name} | {dur:.1f}s | {shots} | {shots / dur * 60:.0f} | {idx['median_shot_s']}s "
                    f"| {hits / dur * 60:.0f} | {with_hit} | {offset} | {sil / dur * 100:.0f}% | {nwords} | {wpm} |")
        labels = [o for o in idx.get("onscreen_text", []) if o.get("box") and o["box"][1] + o["box"][3] / 2 < 0.42]
        details += ["", f"## {Path(v).name}", "",
                    "Cuts at: " + ", ".join(common.fmt_ts(s["start_s"]) for s in idx["shots"][1:60]),
                    "", "Sound hits at: " + ", ".join((common.clock_ms if precise else common.fmt_ts)(t) for t in hit_times[:60]),
                    "", "Labels in the upper part of the frame (term cards, stamps, numbers):", ""]
        details += [f"- {common.fmt_ts(o['start_s'])}–{common.fmt_ts(o['end_s'])} {o['text']}" for o in labels[:80]] or ["- none found"]
    md = ["# Reference study", "",
          "| video | length | cuts | cuts/min | median shot | sound hits/min | cuts with a hit | median hit offset "
          "| silence | words | wpm |",
          "|---|---|---|---|---|---|---|---|---|---|---|"] + rows + [
          "", "Cuts with a hit: cuts whose nearest sound hit starts within 250 ms. Median hit offset: + means the "
          "hits land after their cuts. Under dense music the nearest hit can be a note of the music."] + details
    (out / "study.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"STUDY: {(out / 'study.md').resolve()}")


if __name__ == "__main__":
    main()

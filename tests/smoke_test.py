#!/usr/bin/env python3
"""End-to-end smoke test: every script on a generated clip, on any OS. Python stdlib only.

Builds a 6 s clip (a test pattern with "VIDEO INSPECT 2026" printed near the top, a hard
cut to colour bars at 3 s, a tone) inside
a folder with a Thai name and a space, plus an MPEG-TS copy, then runs index, frames,
check, study and compare on them and checks the outputs make sense.

Usage:  python -I tests/smoke_test.py
Env:    VI_SMOKE_TRANSCRIPT=1  also run faster-whisper (downloads it and the `tiny` model
                               once via uv; needs network) and check.py's script check.
Needs ffmpeg 5.1+ and ffprobe; OCR (Vision or Tesseract) is used when present.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Windows pipes default to the ANSI code page and `python -I` ignores PYTHONUTF8, so printing
# the Thai folder name would crash (the scripts get the same fix from common.py).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
TEXT_PNG = Path(__file__).resolve().parent / "fixtures" / "text.png"   # printed text for OCR
CUT_AT = 3.0
failures: list[str] = []


def check(ok: bool, what: str) -> None:
    print(("ok    " if ok else "FAIL  ") + what, flush=True)
    if not ok:
        failures.append(what)


def run(script: str, *args: str, env: dict | None = None) -> subprocess.CompletedProcess:
    r = subprocess.run([sys.executable, "-I", str(SCRIPTS / script), *args], capture_output=True,
                       text=True, encoding="utf-8", errors="replace", env=env)
    if r.returncode:
        print(f"--- {script} exited {r.returncode}:\n{r.stderr[-1500:]}", flush=True)
    return r


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], check=True)


def main() -> None:
    root = Path(tempfile.mkdtemp(prefix="vi_smoke_"))
    try:
        folder = root / "วิดีโอ ทดสอบ"          # non-ASCII name with a space
        folder.mkdir()
        mp4, ts = folder / "คลิป.mp4", folder / "clip.ts"
        ffmpeg("-f", "lavfi", "-i", f"testsrc=s=640x360:r=30:d={CUT_AT}",
               "-f", "lavfi", "-i", f"smptebars=s=640x360:r=30:d={CUT_AT}",
               "-f", "lavfi", "-i", f"sine=frequency=440:duration={2 * CUT_AT}", "-i", str(TEXT_PNG),
               "-filter_complex", "[0:v][3:v]overlay=0:20,format=yuv420p[a];[a][1:v]concat=n=2:v=1:a=0[v]",
               "-map", "[v]", "-map", "2:a",
               "-c:v", "libx264", "-g", "60", "-pix_fmt", "yuv420p", "-c:a", "aac", str(mp4))
        ffmpeg("-i", str(mp4), "-c", "copy", "-f", "mpegts", str(ts))
        print(f"clip: {mp4}  (temp dir: {tempfile.gettempdir()})", flush=True)

        # index.py, with OCR when an engine is installed
        r = run("index.py", str(mp4), "--out-dir", str(root / "idx"), "--ocr")
        check(r.returncode == 0 and "INDEX:" in r.stdout, "index.py runs")
        idx = json.loads((root / "idx" / "index.json").read_text(encoding="utf-8"))
        cuts = [s["start_s"] for s in idx["shots"][1:]]
        check(len(cuts) == 1 and abs(cuts[0] - CUT_AT) < 0.1, f"one cut found at {CUT_AT}s (got {cuts})")
        check(idx["has_audio"] and idx["silence_s"] == 0, "audio seen, no silence")
        if idx["ocr_backend"]:
            texts = idx["onscreen_text"]
            hits = [o for o in texts if "INSPECT" in o["text"].upper()]
            check(bool(hits), f"{idx['ocr_backend']} OCR read the printed text ({[o['text'] for o in texts][:6]})")
            check(bool(hits) and hits[0]["start_s"] < 0.5 and abs(hits[0]["end_s"] - CUT_AT) < 0.6,
                  "printed text spans the first shot (0-3 s)")
            check(bool(hits) and 0.05 < hits[0]["box"][1] < 0.2, "its box sits near the top of the frame")
            check(all(o["box"] and len(o["box"]) == 4 for o in texts), "every OCR span has a screen box")
        else:
            print("skip  OCR (no Vision or Tesseract here)", flush=True)

        # index.py on the .ts with a range: cut time must match the mp4
        r = run("index.py", str(ts), "--out-dir", str(root / "idx_ts"), "--start", "1", "--end", "5")
        ts_idx = json.loads((root / "idx_ts" / "index.json").read_text(encoding="utf-8"))
        ts_cuts = [s["start_s"] for s in ts_idx["shots"][1:]]
        check(len(ts_cuts) == 1 and abs(ts_cuts[0] - CUT_AT) < 0.1, f".ts --start 1 --end 5: cut at {CUT_AT}s (got {ts_cuts})")
        check(abs(ts_idx["shots"][0]["start_s"] - 1) < 0.01 and abs(ts_idx["shots"][-1]["end_s"] - 5) < 0.01,
              ".ts range is exactly 1-5 s")

        # frames.py: default mode from the index, and exact moments on mp4 vs .ts
        r = run("frames.py", str(mp4), "--out-dir", str(root / "fr"), "--index", str(root / "idx" / "index.json"))
        man = json.loads((root / "fr" / "manifest.json").read_text(encoding="utf-8"))
        check(r.returncode == 0 and man["grids"] and all(Path(g["path"]).exists() for g in man["grids"]),
              f"frames.py writes grids ({man['frames_used']} frames)")
        check({c.get("shot") for g in man["grids"] for c in g["cells"]} >= {0, 1}, "both shots appear in the grids")
        at = "0.5,2.9,3.1,5.9"
        for name, clip in (("mp4", mp4), ("ts", ts)):
            run("frames.py", str(clip), "--out-dir", str(root / f"at_{name}"), "--at", at, "--no-labels")
        cells = [[c["t_ms"] for g in json.loads((root / f"at_{n}" / "manifest.json").read_text(encoding="utf-8"))["grids"]
                  for c in g["cells"]] for n in ("mp4", "ts")]
        check(cells[0] == cells[1] == [500, 2900, 3100, 5900], f"--at gives the same 4 moments on mp4 and .ts ({cells})")
        p = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(root / "at_mp4" / "grid_000.png"),
                            "-i", str(root / "at_ts" / "grid_000.png"), "-lavfi", "psnr", "-f", "null", "-"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        check("average:inf" in p.stderr, ".ts frames are pixel-identical to the mp4's")

        # check.py
        terms = root / "terms.txt"
        terms.write_text("0\n", encoding="utf-8")
        r = run("check.py", str(mp4), "--out-dir", str(root / "ck"), "--expect-size", "640x360",
                "--expect-fps", "30", "--terms", str(terms))
        ck = json.loads((root / "ck" / "check.json").read_text(encoding="utf-8"))
        byname = {c["name"]: c for c in ck["checks"]}
        check(r.returncode == 0 and byname["size"]["pass"] and byname["fps"]["pass"], "check.py: size and fps PASS")
        check(byname["black frames"]["pass"] is True, "check.py: no black frames")
        check((byname["overlay zones"]["pass"] is None) == (idx["ocr_backend"] is None),
              "check.py: overlay zones SKIPPED only without OCR")

        # study.py and compare.py
        r = run("study.py", str(mp4), "--out-dir", str(root / "st"), "--no-transcript")
        check(r.returncode == 0 and (root / "st" / "study.md").exists(), "study.py runs")
        a, b = root / "a.png", root / "b.png"
        ffmpeg("-ss", "1", "-i", str(mp4), "-frames:v", "1", str(a))
        ffmpeg("-ss", "4", "-i", str(mp4), "-frames:v", "1", str(b))
        r = run("compare.py", "--stills", str(a), "--refs", str(b), "--out-dir", str(root / "cmp"))
        check(r.returncode == 0 and (root / "cmp" / "compare.md").exists(), "compare.py runs")

        # transcript.py via uv + faster-whisper, then check.py's script check offline from cache
        if os.environ.get("VI_SMOKE_TRANSCRIPT") == "1":
            env = {**os.environ, "WHISPER_MODEL": "tiny"}
            tx = root / "tx.json"
            r = run("transcript.py", str(mp4), "--out", str(tx), "--lang", "en", "--source", "faster",
                    "--allow-network", env=env)
            check(r.returncode == 0 and json.loads(tx.read_text(encoding="utf-8"))["source"] == "faster",
                  "transcript.py: faster-whisper via uv")
            script = root / "script.txt"
            script.write_text("hello\n", encoding="utf-8")
            run("check.py", str(mp4), "--out-dir", str(root / "ck2"), "--script", str(script), env=env)
            ck2 = {c["name"]: c for c in json.loads((root / "ck2" / "check.json").read_text(encoding="utf-8"))["checks"]}
            check(ck2["script vs speech"].get("skipped") is not True, "check.py: script check ran offline from the cache")
    finally:
        shutil.rmtree(root, ignore_errors=True)
    print(f"\n{'FAILED: ' + str(len(failures)) if failures else 'ALL OK'}", flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Final quality check of a rendered video before posting. One report, PASS or FIX per item.

Checks (each optional input switches its check on):
  specs     size, fps, duration, integrated loudness, true peak
  silence   gaps longer than --min-silence
  picture   black frames, frozen stretches
  shots     cut count vs --expect-shots
  text      every --terms entry appears on screen (OCR); near-misses flagged as typos
  zones     on-screen text inside app-overlay zones (TikTok/Reels/Shorts buttons and bars)
  script    spoken words vs --script, word by word (local transcript; never paid)

Usage:
  python3 check.py VIDEO --out-dir DIR [--expect-size 1080x1920] [--expect-fps 60]
      [--expect-lufs -14] [--max-peak -1] [--min-silence 0.3] [--expect-duration S]
      [--expect-shots N] [--terms terms.txt] [--script script.txt] [--transcript t.json]
Output: DIR/check.md, DIR/check.json; prints `CHECK: <path>`.
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

HERE = Path(__file__).resolve().parent
# Approximate overlay danger zones on a 1080x1920 frame (fractions of width/height).
ZONES = {"right buttons": (1 - 140 / 1080, 0.0, 1.0, 1.0),
         "bottom bar": (0.0, 1 - 380 / 1920, 1.0, 1.0),
         "top bar": (0.0, 0.0, 1.0, 140 / 1920)}


def run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace").stderr


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9$][a-z0-9$,'.%]*", text.lower().replace("’", "'"))


def norm(w: str) -> str:
    return w.strip(".,'").replace(",", "")


FREEZE_DIFF = 0.05     # mean |ΔY| (0–255, 180x320) below this = no visible change; encoder noise on a still is ~0.003–0.02


def picture_scan(video: str, fps: float, out: Path, args, shots: list[dict], res: list) -> tuple[list[str], list[str]]:
    """Frame-by-frame picture checks in one decode:
    frozen stretches, one/two-frame glitches, unplanned jumps, per-shot luma/saturation, frame strips."""
    tmp = common.make_tmp("pic")
    try:
        y0, y1 = (float(x) for x in args.mask_band.split("-")) if args.mask_band != "none" else (0.0, 0.0)
        mask = f"drawbox=x=0:y=ih*{y0:.4f}:w=iw:h=ih*{y1 - y0:.4f}:color=black:t=fill," if y1 > y0 else ""
        sf, df, raw = tmp / "s.txt", tmp / "d.txt", tmp / "tiny.raw"
        subprocess.run(["ffmpeg", "-v", "error", "-i", video, "-filter_complex",
                        f"[0:v]scale=180:320:flags=area,signalstats,metadata=print:file=s.txt,{mask}split[a][b];"
                        f"[a]tblend=all_mode=difference,signalstats,metadata=print:file=d.txt[d];"
                        "[b]scale=45:80:flags=area,format=gray[r]",
                        "-map", "[d]", "-f", "null", "-", "-map", "[r]", "-f", "rawvideo", raw.name], cwd=tmp, check=True)
        stxt, dtxt = sf.read_text(encoding="utf-8"), df.read_text(encoding="utf-8")
        ts = [float(x) for x in re.findall(r"pts_time:([\d.]+)", stxt)]
        luma = [(float(x) - 16) / 219 for x in re.findall(r"signalstats\.YAVG=([\d.]+)", stxt)]
        sat = [float(x) for x in re.findall(r"signalstats\.SATAVG=([\d.]+)", stxt)]
        d = [float(x) for x in re.findall(r"signalstats\.YAVG=([\d.]+)", dtxt)]   # d[k]: change from frame k to k+1
        tiny = raw.read_bytes()
        fs = 45 * 80
        frame = lambda i: tiny[i * fs:(i + 1) * fs]
        tdiff = lambda i, j: sum(abs(a - b) for a, b in zip(frame(i), frame(j))) / fs
        n = len(tiny) // fs
        rows, strips = [], []

        # frozen: no visible change for >= freeze_s (captions masked, so word swaps don't hide a dead hold)
        need, run_start, freezes = max(2, round(args.freeze_s * fps)), None, []
        still = lambda k: 0 <= k < len(d) and d[k] < FREEZE_DIFF
        for k, x in enumerate(d + [1e9]):
            if x < FREEZE_DIFF or (still(k - 1) and still(k + 1)):  # a lone encoder keyframe blip doesn't end a hold
                run_start = k if run_start is None else run_start
            else:
                if run_start is not None and k - run_start >= need:
                    freezes.append((ts[run_start], (k - run_start) / fps))
                run_start = None
        for t, dur in freezes:
            rows.append(f"- {common.fmt_ts(t)} frozen for {dur:.2f}s")
            strips.append(t)
        res.append(("frozen picture", not freezes, f"{len(freezes)} stretches of {args.freeze_s:g}s+ with no visible change"
                    + ("" if not freezes else " at " + ", ".join(common.fmt_ts(t) for t, _ in freezes[:10]))))

        # spikes: change far above the local median. A spike that snaps back within 2 frames is a glitch (FIX);
        # one that stays is a cut, a pop-on or a jump, fine if planned.
        cuts = [s["start_s"] for s in shots[1:]]
        if args.cuts:
            txt = Path(args.cuts).read_text()
            cuts = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", txt)]
        glitches, jumps = [], []
        for k, x in enumerate(d):
            win = sorted(d[max(0, k - 8):k + 9])
            if x <= 3 * win[len(win) // 2] + 1.0 or k + 2 >= n:
                continue
            into = tdiff(k, k + 1)
            back = min(tdiff(k, k + 2), tdiff(k, k + 3) if k + 3 < n else 1e9)
            t = ts[k + 1]
            if glitches and t - glitches[-1] <= 3 / fps:
                continue  # the snap back out of a glitch already reported
            if into > 1.0 and back < 0.5 * into:
                glitches.append(t)
            elif any(f0 <= t <= f0 + dur for f0, dur in freezes):
                continue  # encoder blip inside a frozen hold
            elif not any(abs(t - c) <= args.cut_tol for c in cuts):
                if not jumps or t - jumps[-1] > 3 / fps:
                    jumps.append(t)
        for t in glitches:
            rows.append(f"- {common.fmt_ts(t)} glitch: the picture jumps and snaps back within 2 frames")
            strips.append(t)
        res.append(("glitch frames", not glitches, f"{len(glitches)} one/two-frame glitches"
                    + ("" if not glitches else " at " + ", ".join(common.fmt_ts(t) for t in glitches[:10]))))
        if jumps:
            rows.append(f"- jumps that aren't cuts ({len(jumps)}; usually pop-ons or snaps, check the planned ones): "
                        + ", ".join(common.fmt_ts(t) for t in jumps[:40]))

        # frame strips (6 frames from 2 before the moment) for each FIX item
        sdir = out / "strips"
        for t in strips[:20]:
            sdir.mkdir(exist_ok=True)
            p = sdir / f"strip_{t:07.3f}.jpg"
            first = max(0.0, t - 2 / fps)
            lead = min(common.seek_margin(video), first)   # 0 except on .ts-like files
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{first - lead:.4f}", "-i", video, "-vf",
                            f"{common.trim((lead, None))}scale=180:-1,tile=6x1:padding=4", "-frames:v", "1", "-update", "1", str(p)])
            rows.append(f"  - strip ({common.fmt_ts(t)}, 6 frames from 2 frames before): `{p}`")

        shot_rows = []
        for i, s in enumerate(shots or [{"start_s": 0.0, "end_s": ts[-1] + 1}]):
            ks = [k for k, t in enumerate(ts) if s["start_s"] <= t < s["end_s"]]
            if not ks:
                continue
            L = sorted(luma[k] for k in ks)[len(ks) // 2]
            S_ = sorted(sat[k] for k in ks)[len(ks) // 2]
            note = "dark?" if L < 0.12 else ("blown out?" if L > 0.92 else "")
            shot_rows.append(f"| {i + 1} | {common.fmt_ts(s['start_s'])}–{common.fmt_ts(s['end_s'])} | {L:.2f} | {S_:.1f} | {note} |")
        return rows, shot_rows
    finally:
        common.safe_rmtree(tmp)


def main() -> None:
    ap = argparse.ArgumentParser(description="Final video quality check (local, free).")
    ap.add_argument("video")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--expect-size", default="1080x1920")
    ap.add_argument("--expect-fps", type=float, default=60)
    ap.add_argument("--expect-lufs", type=float, default=-14)
    ap.add_argument("--lufs-tol", type=float, default=1.0)
    ap.add_argument("--max-peak", type=float, default=-1.0)
    ap.add_argument("--min-silence", type=float, default=0.3)
    ap.add_argument("--expect-duration", type=float)
    ap.add_argument("--expect-shots", type=int)
    ap.add_argument("--freeze-s", type=float, default=0.5, help="flag stretches this long with no visible change")
    ap.add_argument("--mask-band", default="0.46-0.56",
                    help="frame-height band ignored by the freeze/glitch checks (one-word captions), or 'none'")
    ap.add_argument("--cuts", help="file with planned cut times in seconds; jumps near them aren't listed")
    ap.add_argument("--cut-tol", type=float, default=0.05)
    ap.add_argument("--terms", help="text file, one expected on-screen term per line")
    ap.add_argument("--script", help="text file with the exact spoken script (plain words)")
    ap.add_argument("--transcript", help="existing transcript.json to use instead of transcribing")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()
    common.need("ffmpeg")
    video = str(Path(args.video).expanduser().resolve())
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    res: list[tuple[str, bool, str]] = []

    m = common.probe(video)
    w, h = (int(x) for x in args.expect_size.lower().split("x"))
    res.append(("size", (m["width"], m["height"]) == (w, h), f"{m['width']}x{m['height']} (want {w}x{h})"))
    res.append(("fps", abs(m["fps"] - args.expect_fps) < 0.01, f"{m['fps']:g} (want {args.expect_fps:g})"))
    if args.expect_duration:
        ok = abs(m["duration_s"] - args.expect_duration) <= 0.5
        res.append(("duration", ok, f"{m['duration_s']:.2f}s (want {args.expect_duration:.2f}s ±0.5)"))

    if m["has_audio"]:
        e = run(["ffmpeg", "-hide_banner", "-nostats", "-i", video, "-af", "ebur128=peak=true", "-f", "null", "-"])
        summ = e[e.rfind("Summary:"):]
        I = float(re.search(r"I:\s+(-?[\d.]+) LUFS", summ).group(1))
        tp = re.search(r"Peak:\s+(-?[\d.]+) dBFS", summ)
        peak = float(tp.group(1)) if tp else 0.0
        res.append(("loudness", abs(I - args.expect_lufs) <= args.lufs_tol, f"{I} LUFS (want {args.expect_lufs:g} ±{args.lufs_tol:g})"))
        res.append(("true peak", peak <= args.max_peak, f"{peak} dBFS (want ≤ {args.max_peak:g})"))
        s = run(["ffmpeg", "-hide_banner", "-nostats", "-i", video, "-af",
                 f"silencedetect=noise=-45dB:d={args.min_silence}", "-f", "null", "-"])
        starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", s)]
        ends = [float(x) for x in re.findall(r"silence_end: (-?[\d.]+)", s)]
        gaps = [(a, b) for a, b in zip(starts, ends) if a > 0.05 and b < m["duration_s"] - 0.05]
        res.append(("silences", not gaps, f"{len(gaps)} over {args.min_silence}s"
                    + ("" if not gaps else ": " + ", ".join(f"{common.fmt_ts(a)}–{common.fmt_ts(b)}" for a, b in gaps[:10]))))

    b = run(["ffmpeg", "-hide_banner", "-nostats", "-i", video, "-vf", "blackdetect=d=0.1:pix_th=0.10", "-an", "-f", "null", "-"])
    blacks = re.findall(r"black_start:([\d.]+) black_end:([\d.]+)", b)
    res.append(("black frames", not blacks, f"{len(blacks)} black stretches"
                + ("" if not blacks else ": " + ", ".join(f"{common.fmt_ts(float(a))}–{common.fmt_ts(float(z))}" for a, z in blacks[:10]))))
    # Shots + on-screen text, via the index (local OCR).
    work = out / "index"
    r = subprocess.run([sys.executable, "-I", str(HERE / "index.py"), video, "--out-dir", str(work),
                        "--ocr", "--ocr-lang", args.lang, "--min-silence", str(args.min_silence)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    idx = json.loads((work / "index.json").read_text(encoding="utf-8")) if (work / "index.json").exists() else {}
    if r.returncode != 0 or not idx:
        res.append(("index", False, f"index.py failed: {r.stderr[-200:]}"))
    shots = idx.get("shot_count", 0)
    if args.expect_shots:
        res.append(("shots", abs(shots - args.expect_shots) <= max(2, args.expect_shots // 10),
                    f"{shots} cuts found (storyboard {args.expect_shots}; locked-camera moves aren't cuts)"))
    onscreen = idx.get("onscreen_text", [])

    pic_rows, shot_rows = picture_scan(video, m["fps"], out, args, idx.get("shots", []), res)

    term_rows = []
    if args.terms:
        terms = [t.strip() for t in Path(args.terms).read_text(encoding="utf-8").splitlines() if t.strip()]
        seen = [o["text"] for o in onscreen]
        missing = typos = 0
        for t in terms:
            best, bt = 0.0, ""
            for s in seen:
                cand = [s] + s.split()
                for c in cand:
                    rt = difflib.SequenceMatcher(None, t.lower(), c.lower()).ratio()
                    if rt > best:
                        best, bt = rt, c
            if best >= 0.97:
                term_rows.append(f"- ✅ {t}")
            elif best >= 0.75:
                typos += 1
                term_rows.append(f"- ⚠️ {t}: closest on screen is \"{bt}\" ({best:.0%} match), check spelling")
            else:
                missing += 1
                term_rows.append(f"- ❌ {t}: not found on screen")
        res.append(("on-screen terms", missing == 0 and typos == 0,
                    f"{len(terms) - missing - typos}/{len(terms)} exact, {typos} possible typos, {missing} missing"))

    zone_hits = []
    for o in onscreen:
        bx = o.get("box")
        if not bx:
            continue
        x0, y0, x1, y1 = bx[0], bx[1], bx[0] + bx[2], bx[1] + bx[3]
        for name, (zx0, zy0, zx1, zy1) in ZONES.items():
            if x0 < zx1 and x1 > zx0 and y0 < zy1 and y1 > zy0:
                zone_hits.append(f"- {common.fmt_ts(o['start_s'])}–{common.fmt_ts(o['end_s'])} \"{o['text']}\" in {name}")
    res.append(("overlay zones", not zone_hits, f"{len(zone_hits)} text items inside app-overlay zones"))

    diff_rows = []
    if args.script:
        tpath = Path(args.transcript).expanduser() if args.transcript else out / "transcript.json"
        if not args.transcript:
            subprocess.run([sys.executable, "-I", str(HERE / "transcript.py"), video, "--out", str(tpath),
                            "--lang", args.lang], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if tpath.exists():
            tx = json.loads(tpath.read_text(encoding="utf-8"))
            heard = [(norm(x), w_["start_s"]) for w_ in tx["words"] for x in words(w_["text"])]
            want = [norm(x) for x in words(Path(args.script).read_text(encoding="utf-8"))]
            sm = difflib.SequenceMatcher(None, want, [x for x, _ in heard], autojunk=False)
            numbers = []
            for op, a0, a1, b0, b1 in sm.get_opcodes():
                if op == "equal":
                    continue
                at = common.fmt_ts(heard[min(b0, len(heard) - 1)][1]) if heard else "?"
                said, got = " ".join(want[a0:a1]), " ".join(x for x, _ in heard[b0:b1])
                row = f"- {at} {op}: script \"{said}\" → heard \"{got}\""
                # Whisper writes numbers as digits ("$2 000" for "two thousand dollars"): note it, don't fail it.
                (numbers if re.search(r"\d", said + got) else diff_rows).append(row)
            res.append(("script vs speech", not diff_rows,
                        f"{sm.ratio():.0%} word match, {len(diff_rows)} word differences"
                        f"{f', {len(numbers)} number-format notes' if numbers else ''} ({tx['source']} transcript)"))
            diff_rows += [r + " (number format, likely fine)" for r in numbers]
        else:
            res.append(("script vs speech", False, "no transcript (local Whisper unavailable); not checked"))

    verdict = "PASS" if all(ok for _, ok, _ in res) else "FIX"
    md = [f"# Check: {Path(video).name}", "", f"VERDICT: {verdict}", "", "| check | result | detail |", "|---|---|---|"]
    md += [f"| {n} | {'PASS' if ok else 'FIX'} | {d} |" for n, ok, d in res]
    if pic_rows:
        md += ["", "## Picture timeline", ""] + pic_rows
    md += ["", "## Per-shot picture", "", "luma 0–1 and mean saturation, medians over each shot (caption band included).", "",
           "| shot | time | luma | saturation | |", "|---|---|---|---|---|"] + shot_rows
    if term_rows:
        md += ["", "## On-screen terms", ""] + term_rows
    if zone_hits:
        md += ["", "## Text in overlay zones", ""] + zone_hits
    if diff_rows:
        md += ["", "## Script vs speech differences", ""] + diff_rows[:60]
    md += ["", f"Shot list and on-screen text timeline: `{work / 'index.md'}`."]
    (out / "check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "check.json").write_text(json.dumps({"verdict": verdict, "checks": [
        {"name": n, "pass": ok, "detail": d} for n, ok, d in res]}, indent=1), encoding="utf-8")
    print(f"CHECK: {(out / 'check.md').resolve()}")


if __name__ == "__main__":
    main()

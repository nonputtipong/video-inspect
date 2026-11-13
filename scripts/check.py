#!/usr/bin/env python3
"""Final quality check of a rendered video before posting. One report, PASS or FIX per item
(SKIPPED when a check couldn't run here, e.g. no OCR engine or no local Whisper).

Checks (each optional input switches its check on):
  specs     size, fps, duration, integrated loudness and its range (LRA), true peak
  silence   gaps longer than --min-silence
  mix       clipped samples, sudden loudness jumps away from cuts, and two rules of thumb:
            music too close to the voice (needs a transcript) and --cues effects that are buried
  picture   black frames, frozen stretches
  shots     cut count vs --expect-shots
  sync      each cut's nearest sound hit and how many ms it is off (--sync-tol makes it PASS/FIX)
  text      every --terms entry appears on screen (OCR); near-misses flagged as typos
  zones     on-screen text inside app-overlay zones (TikTok/Reels/Shorts buttons and bars)
  script    spoken words vs --script, word by word (local transcript; never paid)
  profile   with --profile (from study.py --profile): where this video's cuts per minute, hits on
            cuts, loudness, beat alignment, silence and speaking pace fall against your reference
            videos, with the moments behind each difference. A report section, not a PASS/FIX row.

Usage:
  python3 check.py VIDEO --out-dir DIR [--expect-size 1080x1920] [--expect-fps 60]
      [--expect-lufs -14] [--max-peak -1] [--min-silence 0.3] [--expect-duration S]
      [--expect-shots N] [--terms terms.txt] [--script script.txt] [--transcript t.json]
      [--cuts cuts.txt] [--sync-tol MS] [--cues cues.txt] [--profile p.json]
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
import audio  # noqa: E402
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
            cuts = common.read_times(args.cuts)
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
    ap.add_argument("--cuts", help="file with planned cut times in seconds; jumps near them aren't listed, "
                                   "and hit sync is measured against them")
    ap.add_argument("--cut-tol", type=float, default=0.05)
    ap.add_argument("--sync-tol", type=float,
                    help="ms: FIX when a cut's nearest sound hit is further from the cut than this")
    ap.add_argument("--cues", help="file with planned sound-effect times in seconds; ones that don't rise "
                                   "out of the sound before them are flagged (rule of thumb)")
    ap.add_argument("--profile", help="profile JSON from study.py --profile: report where this video falls "
                                      "against those reference videos")
    ap.add_argument("--terms", help="text file, one expected on-screen term per line")
    ap.add_argument("--script", help="text file with the exact spoken script (plain words)")
    ap.add_argument("--transcript", help="existing transcript.json to use instead of transcribing")
    ap.add_argument("--lang", default="en")
    args = ap.parse_args()
    common.need("ffmpeg")
    video = str(Path(args.video).expanduser().resolve())
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    res: list[tuple[str, bool | None, str]] = []   # None = couldn't run here: SKIPPED, not FIX
    measures: dict = {}                            # plain numbers, for check.json

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
        lra = re.search(r"LRA:\s+(-?[\d.]+) LU", summ)
        measures.update(lufs=I, true_peak_db=peak, lra_lu=float(lra.group(1)) if lra else None)
        res.append(("loudness", abs(I - args.expect_lufs) <= args.lufs_tol,
                    f"{I} LUFS (want {args.expect_lufs:g} ±{args.lufs_tol:g})"
                    + (f"; loudness range (LRA) {lra.group(1)} LU" if lra else "")))
        res.append(("true peak", peak <= args.max_peak, f"{peak} dBFS (want ≤ {args.max_peak:g})"))
        clip = audio.clipping(video)
        measures["clipped_s"] = clip["seconds"]
        res.append(("clipping", not clip["spans"],
                    f"{clip['seconds']:g} s pinned at full scale ({audio.CLIP_SHARE:.0%}+ of samples within 0.5 dB of it: "
                    "clipped, or limited with no headroom)"
                    + ("" if not clip["spans"] else ": " + ", ".join(
                        f"{common.fmt_ts(a)}–{common.fmt_ts(b)}" for a, b in clip["spans"][:10]))))
        s = run(["ffmpeg", "-hide_banner", "-nostats", "-i", video, "-af",
                 f"silencedetect=noise={common.SILENCE_DEAD_DB}dB:d={args.min_silence}", "-f", "null", "-"])
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
                        "--ocr", "--ocr-lang", args.lang, "--min-silence", str(args.min_silence)]
                       + (["--cues", args.cues] if args.cues else []),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    idx = json.loads((work / "index.json").read_text(encoding="utf-8")) if (work / "index.json").exists() else {}
    if r.returncode != 0 or not idx:
        res.append(("index", False, f"index.py failed: {r.stderr[-200:]}"))
    shots = idx.get("shot_count", 0)
    if args.expect_shots:
        res.append(("shots", abs(shots - args.expect_shots) <= max(2, args.expect_shots // 10),
                    f"{shots} cuts found (storyboard {args.expect_shots}; locked-camera moves aren't cuts)"))
    onscreen = idx.get("onscreen_text", [])
    # Without an OCR engine (or with index.py failing) no text was read at all, so "term
    # missing" or "nothing in the overlay zones" would be a guess, not a finding.
    no_ocr = ("" if idx.get("ocr_backend") else
              "index.py failed" if not idx else "no OCR engine (needs macOS Vision or Tesseract)")

    pic_rows, shot_rows = picture_scan(video, m["fps"], out, args, idx.get("shots", []), res)

    # Hit-to-cut sync, against the planned cuts when given. A cut with no hit in reach is
    # listed but never a FIX: under dense music a hit can be there and not be measurable.
    sync_rows = []
    cut_times = common.read_times(args.cuts) if args.cuts else [x["start_s"] for x in idx.get("shots", [])[1:]]
    if idx.get("onsets") is not None:
        pairs = audio.pair_cuts(cut_times, [o["t_s"] for o in idx["onsets"]])
        sync_rows = [f"- {common.clock_ms(p['cut_s'])} cut: {audio.offset_words(p['offset_ms'])}" for p in pairs]
        if args.sync_tol is not None:
            off = [p for p in pairs if p["offset_ms"] is not None and abs(p["offset_ms"]) > args.sync_tol]
            none = sum(p["offset_ms"] is None for p in pairs)
            res.append(("cut sync", not off,
                        f"{len(off)} of {len(pairs)} cuts have their nearest hit more than {args.sync_tol:g} ms away"
                        + ("" if not off else ": " + ", ".join(
                            f"{common.clock_ms(p['cut_s'])} ({p['offset_ms']:+d} ms)" for p in off[:10]))
                        + (f"; {none} with no hit within {audio.SYNC_WINDOW_S * 1000:.0f} ms (not counted)" if none else "")))
    elif args.sync_tol is not None:
        res.append(("cut sync", None, "skipped: " + ("hit timing didn't run" if m["has_audio"] else "no audio")))

    # Mix: loudness jumps that no cut explains, and planned effects that don't stand out.
    loud = idx.get("loudness")
    if loud:
        jumps = audio.loud_jumps(loud, cut_times)
        res.append(("loudness jumps", not jumps,
                    f"{len(jumps)} jumps of {audio.JUMP_LU:g}+ LU within {audio.JUMP_OVER_S:g} s away from cuts"
                    + ("" if not jumps else ": " + ", ".join(
                        f"{common.fmt_ts(j['t_s'])} (+{j['rise_lu']} LU)" for j in jumps[:10]))
                    + f" (not judged in the first {audio.SHORT_TERM_S + audio.JUMP_OVER_S:g} s)"))
    elif m["has_audio"]:
        res.append(("loudness jumps", None, "skipped: index.py gave no loudness"))
    cue_rows = []
    if args.cues and idx.get("cues") is not None:
        buried = [c for c in idx["cues"] if c["rise_db"] is not None and c["rise_db"] < audio.BURIED_DB]
        cue_rows = [f"- {common.clock_ms(c['cue_s'])} " + ("past the end of the sound" if c["rise_db"] is None else
                    f"rises {c['rise_db']:+.1f} dB over the sound before it"
                    + (" (buried or missing?)" if c["rise_db"] < audio.BURIED_DB else "")) for c in idx["cues"]]
        res.append(("buried effects (rule of thumb)", not buried,
                    f"{len(buried)} of {len(idx['cues'])} planned effects rise less than {audio.BURIED_DB:g} dB "
                    "over the 300 ms before them"
                    + ("" if not buried else ": " + ", ".join(common.clock_ms(c["cue_s"]) for c in buried[:10]))))
    elif args.cues:
        res.append(("buried effects (rule of thumb)", None, "skipped: " + ("no hit timing" if m["has_audio"] else "no audio")))

    term_rows = []
    if args.terms and no_ocr:
        res.append(("on-screen terms", None, f"skipped: {no_ocr}"))
    elif args.terms:
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
    res.append(("overlay zones", None, f"skipped: {no_ocr}") if no_ocr else
               ("overlay zones", not zone_hits, f"{len(zone_hits)} text items inside app-overlay zones"))

    tx = None
    if args.script or args.transcript:
        tpath = Path(args.transcript).expanduser() if args.transcript else out / "transcript.json"
        if not args.transcript:
            subprocess.run([sys.executable, "-I", str(HERE / "transcript.py"), video, "--out", str(tpath),
                            "--lang", args.lang], capture_output=True, text=True, encoding="utf-8", errors="replace")
        if tpath.exists():
            tx = json.loads(tpath.read_text(encoding="utf-8"))

    # Music over the voice needs word times, so it runs only when there is a transcript.
    voice_rows = []
    if loud:
        name = "music over voice (rule of thumb)"
        vb = audio.voice_vs_bed(loud, tx["words"]) if tx else None
        if vb:
            voice_rows = [f"- {common.fmt_ts(g['start_s'])}–{common.fmt_ts(g['end_s'])} "
                          f"the bed is within {g['margin_lu']:.1f} LU of the voice" for g in vb["close"]]
            res.append((name, vb["margin_lu"] >= audio.VOICE_BED_LU,
                        f"speech {vb['voice_lufs']} LUFS, pauses between words {vb['bed_lufs']} LUFS: the bed sits "
                        f"{vb['margin_lu']:.1f} LU under the voice (want {audio.VOICE_BED_LU:g}+); "
                        f"{len(vb['close'])} of {vb['pauses']} pauses closer than that"))
        else:
            res.append((name, None, "skipped: " + (
                "needs a transcript (--transcript FILE, or --script to make one)" if not tx else
                f"no pause of {audio.PAUSE_MIN_S:g}-{audio.PAUSE_MAX_S:g} s between words to measure the bed in")))

    diff_rows = []
    if args.script:
        if tx:
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
            res.append(("script vs speech", None, "skipped: no transcript (no local Whisper worked offline; "
                                                  "run transcript.py to see why)"))

    # Against the user's own references: a distance, with the moments behind it. Not a verdict.
    prof_rows, prof_notes, prof = [], [], None
    if args.profile and idx:
        profile = json.loads(Path(args.profile).expanduser().read_text(encoding="utf-8"))
        prof = audio.against_profile(audio.metrics(idx, tx["words"] if tx else None), profile)
        for r in prof:
            if "low" in r:
                show = r["fmt"].format
                prof_rows.append(f"| {r['name']} | {show(r['value']) if r['value'] is not None else 'not measured'} "
                                 f"| {show(r['low'])} to {show(r['high'])}, median {show(r['median'])} "
                                 f"| {r['where'] or '–'} |")
        where = {r["key"]: r for r in prof if r["where"] in ("below", "above")}
        pairs = idx.get("cut_sync") or []
        if where.get("pct_cuts_with_hit", {}).get("where") == "below":
            prof_notes += [f"- {common.clock_ms(p['cut_s'])} cut has no sound hit within "
                           f"{audio.SYNC_WINDOW_S * 1000:.0f} ms" for p in pairs if p["offset_ms"] is None]
        r = where.get("median_hit_offset_ms")
        if r:
            prof_notes += [f"- {common.clock_ms(p['cut_s'])} cut: {audio.offset_words(p['offset_ms'])} "
                           f"(references' median {r['median']:+.0f} ms)" for p in pairs if p["offset_ms"] is not None
                           and (p["offset_ms"] > r["high"] if r["where"] == "above" else p["offset_ms"] < r["low"])]
        if where.get("pct_cuts_on_beat", {}).get("where") == "below":
            prof_notes += [f"- {common.clock_ms(p['cut_s'])} cut is {abs(p['beat_offset_ms'])} ms "
                           f"{'after' if p['beat_offset_ms'] > 0 else 'before'} the beat" for p in pairs
                           if p.get("beat_offset_ms") is not None and abs(p["beat_offset_ms"]) > audio.BEAT_TOL_S * 1000]
        prof_notes.sort()

    verdict = "FIX" if any(ok is False for _, ok, _ in res) else "PASS"
    skipped = [n for n, ok, _ in res if ok is None]
    status = {True: "PASS", False: "FIX", None: "SKIPPED"}
    md = [f"# Check: {Path(video).name}", "",
          f"VERDICT: {verdict}" + (f" ({len(skipped)} not checked: {', '.join(skipped)})" if skipped else ""),
          "", "| check | result | detail |", "|---|---|---|"]
    md += [f"| {n} | {status[ok]} | {d} |" for n, ok, d in res]
    if pic_rows:
        md += ["", "## Picture timeline", ""] + pic_rows
    md += ["", "## Per-shot picture", "", "luma 0–1 and mean saturation, medians over each shot (caption band included).", "",
           "| shot | time | luma | saturation | |", "|---|---|---|---|---|"] + shot_rows
    if sync_rows:
        md += ["", "## Cut sync", "",
               f"Nearest sound hit to each {'planned' if args.cuts else 'detected'} cut (+ late, - early; hits are timed "
               "to about 5-10 ms). Under dense music a hit can be hidden, and the nearest one can be a note of the music.",
               ""] + sync_rows[:80]
    if cue_rows:
        md += ["", "## Planned effects (rule of thumb)", "",
               "Loudest 20 ms around each `--cues` time against the 300 ms before it, above 150 Hz. A busy bed "
               "reads a few dB with nothing there, so a low number means buried or missing; listen before judging.",
               ""] + cue_rows[:80]
    if voice_rows:
        md += ["", "## Music over voice (rule of thumb)", "",
               "Pauses between words where the loudness stays close to the speech level. A sound effect in a "
               "pause reads the same as loud music, so listen before judging.", ""] + voice_rows[:60]
    if prof is not None:
        judged = [r for r in prof if r["where"]]
        md += ["", "## Against your references", "",
               f"`{args.profile}` ({len(profile['videos'])} videos): {sum(r['where'] == 'inside' for r in judged)} of "
               f"{len(judged)} measures fall inside the references' range. The range is their lowest to highest "
               f"(their middle half once there are {audio.QUARTILES_FROM} or more). Outside it is a difference to look "
               "at, not a fault: the references are a taste, and taste is yours to judge.", "",
               "| measure | this video | references | |", "|---|---|---|---|"] + prof_rows
        if prof_notes:
            md += ["", "Moments behind the differences:", ""] + prof_notes[:60]
    if term_rows:
        md += ["", "## On-screen terms", ""] + term_rows
    if zone_hits:
        md += ["", "## Text in overlay zones", ""] + zone_hits
    if diff_rows:
        md += ["", "## Script vs speech differences", ""] + diff_rows[:60]
    md += ["", f"Shot list and on-screen text timeline: `{work / 'index.md'}`."]
    (out / "check.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    report = {"verdict": verdict, "measures": measures, "checks": [
        {"name": n, "pass": ok, "detail": d, **({"skipped": True} if ok is None else {})} for n, ok, d in res]}
    if skipped:
        report["skipped"] = skipped
    if prof is not None:
        report["profile"] = {"file": args.profile, "inside": sum(r["where"] == "inside" for r in prof),
                             "measured": sum(r["where"] is not None for r in prof),
                             "metrics": [{k: v for k, v in r.items() if k != "fmt"} for r in prof],
                             "notes": [n[2:] for n in prof_notes]}
    (out / "check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"CHECK: {(out / 'check.md').resolve()}")


if __name__ == "__main__":
    main()

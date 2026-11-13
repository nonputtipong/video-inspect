#!/usr/bin/env python3
"""Shot index of a video: cuts, how each shot moves, looks and sounds, plus on-screen text.

Frames show what is in a video and the transcript what is said. This file is the map
between them: where the cuts fall, which shots are dark or busy, where the sound jumps
or stops, and (with --ocr) what text appears where on screen. Read index.md first,
then look closer with frames.py.

How it measures (ffmpeg only, Python stdlib):
  video   every frame scaled to 160px wide gets a scene-change score, because a cut
          is one frame wide and sampling would step over it. Four samples a second
          also get brightness, motion and sharpness.
  audio   momentary and short-term loudness every 100 ms, and silence spans (in parallel).
  hits    level of every 5 ms of high-passed audio; a hit is where it rises out of its
          background (audio.py). Each cut is paired with the nearest hit within 250 ms, and
          each hit gets its attack, length, brightness and tonality, and a shape label.
  music   tempo and a beat grid from the same 5 ms levels, each cut's distance to the nearest
          beat, and swells and drops in short-term loudness.
  text    (--ocr) frames at a few per second, read only when the picture changed.

Usage:
  python3 index.py VIDEO --out-dir DIR [--start T] [--end T] [--ocr] [--ocr-lang th]
                         [--ocr-fps 8] [--min-silence 0.3] [--cues cues.txt] [--transcript t.json]

Writes DIR/index.json and DIR/index.md; prints `INDEX: <path to index.md>` last.
"""
from __future__ import annotations

import argparse
import bisect
import difflib
import json
import math
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio  # noqa: E402
import common  # noqa: E402

# A scene score at or above HARD_CUT is a cut. Between SOFT_CUT and HARD_CUT it is a cut
# only if it is SOFT_RATIO times the average of the NEIGHBOURS frames either side: in dark
# or low-contrast footage real cuts score low, but they still stand out from their
# surroundings, while a fast pan scores high on every frame and stands out from nothing.
HARD_CUT = 0.30
SOFT_CUT = 0.12
SOFT_RATIO = 3.0
NEIGHBOURS = 4
MIN_SHOT_S = 0.4          # a "shot" shorter than this is a flash frame or flicker

SAMPLES_PER_S = 4.0       # brightness / motion / sharpness samples
HIT_JUMP_DB = 12.0        # loudness rise over the previous ~0.4 s that counts as a sound hit
SAME_FRAME = 3.0          # OCR: skip a frame whose thumbnail barely changed since the last read
OCR_BATCH = 40

SILENCE_RE = re.compile(r"silence_(start|end): (-?[\d.]+)")


def read_metadata(path: Path) -> list[dict]:
    """Parse an ffmpeg `metadata=print:file=` log into [{"t": seconds, key: number}].
    Values that aren't finite numbers (blurdetect gives nan on flat frames) are left out."""
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("frame:"):
            t = line.split("pts_time:", 1)[1].split()[0] if "pts_time:" in line else "0"
            rows.append({"t": float(t)})
        elif rows and line.startswith("lavfi.") and "=" in line:
            key, _, val = line[len("lavfi."):].partition("=")
            try:
                num = float(val)
            except ValueError:
                continue
            if math.isfinite(num):
                rows[-1][key] = num
    return rows


def measure(video: str, start: float | None, end: float | None, with_audio: bool,
            tmp: Path, min_silence: float = 0.6) -> dict:
    """Run the video and audio passes side by side. Returns scene, stats, loud, silences, env.

    ffmpeg runs inside `tmp` and writes its logs under bare file names: a full Windows
    path (C:\\...) inside a filter option would be misread as more filter syntax."""
    seek, cut = common.seek_input(video, start, end)
    video_graph = (
        f"[0:v]{common.trim(cut)}scale=160:-2,split[cut][look];"
        "[cut]select='gte(scene,0)',metadata=print:key=lavfi.scene_score:file=scene.txt[c];"
        f"[look]fps={SAMPLES_PER_S},signalstats,blurdetect=block_width=16:block_height=16,"
        "metadata=print:file=stats.txt[l]")
    jobs = [["ffmpeg", "-hide_banner", "-loglevel", "error", *seek, "-i", video,
             "-filter_complex", video_graph,
             "-map", "[c]", "-f", "null", "-", "-map", "[l]", "-f", "null", "-"]]
    if with_audio:
        jobs.append(["ffmpeg", "-hide_banner", "-loglevel", "info", *seek,
                     "-i", video, "-vn", "-af",
                     f"{common.trim(cut, audio=True)}silencedetect=noise={common.SILENCE_PAUSE_DB}dB:d={min_silence},ebur128=metadata=1,"
                     "ametadata=print:file=loud.txt", "-f", "null", "-"])
        jobs.append(audio.hit_cmd(video, seek, cut))
    procs = [subprocess.Popen(cmd, cwd=tmp, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              text=True, encoding="utf-8", errors="replace") for cmd in jobs]
    logs = [p.communicate()[1] or "" for p in procs]
    if procs[0].returncode:
        sys.exit(f"ffmpeg video pass failed: {logs[0][-400:]}")

    silences: list[tuple[float, float]] = []
    began = None
    for kind, value in SILENCE_RE.findall(logs[1] if with_audio else ""):
        if kind == "start":
            began = float(value)
        elif began is not None:
            silences.append((began, float(value)))
            began = None
    return {"scene": read_metadata(tmp / "scene.txt"), "stats": read_metadata(tmp / "stats.txt"),
            "loud": read_metadata(tmp / "loud.txt") if with_audio else [], "silences": silences,
            "env": audio.read_env(tmp),   # (levels, first frame's time)
            "full": audio.read_env(tmp, "full.txt")[0], "spectrum": audio.read_spectrum(tmp)}


def find_cuts(scene: list[dict]) -> list[float]:
    times = [r["t"] for r in scene]
    scores = [r.get("scene_score", 0.0) for r in scene]
    cuts: list[float] = []
    last = times[0] if times else 0.0
    for i in range(1, len(scores)):
        s = scores[i]
        if s < SOFT_CUT:
            continue
        around = scores[max(0, i - NEIGHBOURS):i] + scores[i + 1:i + 1 + NEIGHBOURS]
        local = sum(around) / len(around) if around else 0.0
        if (s >= HARD_CUT or s >= SOFT_RATIO * max(local, 0.01)) and times[i] - last >= MIN_SHOT_S:
            cuts.append(times[i])
            last = times[i]
    return cuts


def usable(row: dict) -> bool:
    """Not black, not blown out, some contrast: a frame worth showing for its shot."""
    return (25 <= row.get("signalstats.YAVG", 128) <= 235
            and row.get("signalstats.YHIGH", 255) - row.get("signalstats.YLOW", 0) >= 60)


def make_shots(cuts: list[float], span: float, stats: list[dict], loud: list[dict]) -> list[dict]:
    edges = [0.0] + [c for c in cuts if 0 < c < span] + [span]
    stat_t = [r["t"] for r in stats]
    loud_t = [r["t"] for r in loud]
    shots: list[dict] = []
    for a, b in zip(edges, edges[1:]):
        if b <= a:
            continue
        shot = {"index": len(shots), "start_s": round(a, 3), "end_s": round(b, 3), "dur_s": round(b - a, 3)}
        rows = stats[bisect.bisect_left(stat_t, a):bisect.bisect_left(stat_t, b)]
        if rows:
            blur = [r.get("blur", 0.0) for r in rows]
            pool = [i for i, r in enumerate(rows) if usable(r)] or list(range(len(rows)))
            pick = min(pool, key=lambda i: blur[i])   # least blur = sharpest
            shot["motion"] = round(sum(r.get("signalstats.YDIF", 0) for r in rows) / len(rows), 2)
            shot["luma"] = round(sum(r.get("signalstats.YAVG", 0) for r in rows) / len(rows), 1)
            shot["blur"] = round(sum(blur) / len(blur), 2)
            shot["sharpest_s"] = round(rows[pick]["t"], 3)
        else:
            shot["sharpest_s"] = round((a + b) / 2, 3)
        levels = [r.get("r128.M", -70.0) for r in loud[bisect.bisect_left(loud_t, a):bisect.bisect_left(loud_t, b)]]
        levels = [v for v in levels if v > -120]
        if levels:
            shot["loud_lufs"] = round(sum(levels) / len(levels), 1)
            shot["loud_peak"] = round(max(levels), 1)
        shots.append(shot)
    return shots


def shot_list(video: str, start: float | None, end: float | None, tmp: Path) -> list[dict]:
    """Shots only (no audio, no text), on the video's own clock. Used by frames.py."""
    m = measure(video, start, end, False, tmp)
    span = (end if end is not None else common.probe(video)["duration_s"]) - (start or 0.0)
    return shift(make_shots(find_cuts(m["scene"]), span, m["stats"], []), start or 0.0)


def shift(shots: list[dict], by: float) -> list[dict]:
    for s in shots:
        for key in ("start_s", "end_s", "sharpest_s"):
            s[key] = round(s[key] + by, 3)
    return shots


def sound_hits(loud: list[dict]) -> list[dict]:
    """Moments the loudness jumps: a slam, a laugh, a music sting."""
    series = [(r["t"], r.get("r128.M", -70.0)) for r in loud]
    hits: list[dict] = []
    for i in range(5, len(series)):
        t, level = series[i]
        before = sum(v for _, v in series[i - 5:i - 1]) / 4
        if level - before < HIT_JUMP_DB or level <= -30:
            continue
        if hits and t - hits[-1]["t_s"] < 1.0:
            continue
        hits.append({"t_s": round(t, 2), "lufs": round(level, 1), "jump": round(level - before, 1)})
    return hits


def read_onscreen(video: str, start: float | None, end: float | None, span: float, tmp: Path,
                  langs: str = "en", fps: float | None = None) -> tuple[str | None, list[dict]]:
    """Text on screen over time: one span per text item, with its box (0-1, top-left).

    Reads 4 frames a second on clips up to 5 minutes (2 up to 30, then 1), so one-word
    captions that change several times a second are still caught, and runs OCR only on
    frames whose picture changed since the last one read."""
    backend, _ = common.ocr_backend()
    if backend is None:
        print("[index] --ocr skipped: needs macOS + swiftc, or tesseract", file=sys.stderr)
        return None, []
    rate = fps or (4.0 if span <= 300 else 2.0 if span <= 1800 else 1.0)
    width = "min(1280,iw)" + (f"*{common.TESS_UPSCALE}" if backend == "tesseract" else "")
    folder = tmp / "ocr"
    folder.mkdir()
    seek, cut = common.seek_input(video, start, end)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", *seek, "-i", video,
                    "-vf", f"{common.trim(cut)}fps={rate},scale='{width}':-2", "-q:v", "3", "o_%06d.jpg"],
                   cwd=folder, capture_output=True)
    frames = sorted(folder.glob("o_*.jpg"))

    thumbs = common.gray_thumbs(frames)
    source: list[int] = []        # frame i shows the same text as frame source[i]
    for i in range(len(frames)):
        last = source[-1] if source else None
        if thumbs and last is not None and common.region_delta(thumbs[i], thumbs[last]) < SAME_FRAME:
            source.append(last)
        else:
            source.append(i)
    to_read = sorted(set(source))
    batches = [[frames[i] for i in to_read[k:k + OCR_BATCH]] for k in range(0, len(to_read), OCR_BATCH)]
    found: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for part in pool.map(lambda batch: common.ocr_images(batch, langs), batches):
            found.update(part)

    step = 1.0 / rate
    origin = start or 0.0
    live: list[dict] = []
    spans: list[dict] = []
    for i in range(len(frames)):
        t = origin + i * step
        now: list[dict] = []
        for item in found.get(str(frames[source[i]]), []):
            if common.is_noise(item["text"]):
                continue
            text = item["text"].lower()
            same = next((s for s in live
                         if difflib.SequenceMatcher(None, s["text"].lower(), text).ratio() >= 0.8), None)
            if same:
                live.remove(same)
                same["end_s"] = round(t + step, 2)
                now.append(same)
            else:
                now.append({"start_s": round(t, 2), "end_s": round(t + step, 2),
                            "text": item["text"], "box": item.get("box")})
        spans += live            # anything not seen in this frame has ended
        live = now
    spans += live
    spans.sort(key=lambda s: (s["start_s"], (s["box"] or [0, 0])[1]))
    print(f"[index] OCR: {len(frames)} frames at {rate:g}/s, {len(to_read)} read, "
          f"{len(frames) - len(to_read)} unchanged and skipped", file=sys.stderr)
    return backend, spans


def zone(box: list | None) -> str:
    if not box:
        return ""
    middle = box[1] + box[3] / 2
    for limit, name in ((0.2, "top"), (0.42, "upper"), (0.62, "middle"), (0.8, "lower")):
        if middle < limit:
            return name
    return "bottom"


def ts(sec: float) -> str:
    return common.clock(int(round(sec * 1000)))


def motion_bands(shots: list[dict]) -> tuple[float, float]:
    """Motion words are relative to this video: the quietest quarter of its shots reads
    "still", the busiest quarter "high motion", whatever kind of footage it is."""
    motion = sorted(s["motion"] for s in shots if "motion" in s)
    quantile = (lambda q: motion[min(len(motion) - 1, int(q * len(motion)))]) if motion else (lambda q: 0.0)
    return max(0.5, quantile(0.25)), quantile(0.75)


def describe(shot: dict, still_below: float, busy_from: float) -> str:
    words = []
    if "motion" in shot:
        m = shot["motion"]
        words.append("still" if m < still_below else "moving" if m < busy_from else "high motion")
        if shot["luma"] < 45:
            words.append("dark")
        elif shot["luma"] >= 200:
            words.append("bright")
    if "loud_lufs" in shot:
        lufs = shot["loud_lufs"]
        words.append("quiet" if lufs < -40 else "loud" if lufs > -18 else "sound")
    return ", ".join(words)


def main() -> None:
    ap = argparse.ArgumentParser(description="Shot, motion, sound and on-screen text index.")
    ap.add_argument("video")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--ocr", action="store_true",
                    help="Also read on-screen text over time (captions, labels, signs).")
    ap.add_argument("--ocr-fps", type=float, default=None,
                    help="OCR reads per second (default 4 up to 5 min). Use 8 for one-word "
                         "captions that change faster than 4 a second.")
    ap.add_argument("--min-silence", type=float, default=0.6,
                    help="Shortest silence to report, in seconds (0.3 for a tight-paced short).")
    ap.add_argument("--ocr-lang", default="en",
                    help="On-screen text languages: th, en, zh, ja, ko ... (default en).")
    ap.add_argument("--transcript", help="transcript.json from transcript.py: hits that fall inside a spoken "
                                         "word are labelled speech.")
    ap.add_argument("--cues", help="File of planned sound-effect times in seconds: index.json gets how far "
                                   "each rises over the sound before it (check.py --cues reads that).")
    args = ap.parse_args()
    common.need("ffmpeg")
    common.need("ffprobe")
    video = str(Path(args.video).expanduser().absolute())
    if not Path(video).exists():
        sys.exit(f"no such file: {video}")
    meta = common.probe(video)
    if not meta["width"]:
        sys.exit(f"no video stream in {video}")
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    start, end = common.parse_time(args.start), common.parse_time(args.end)
    origin = start or 0.0
    span = (end if end is not None else meta["duration_s"]) - origin
    if span <= 0:
        sys.exit("empty time range")

    tmp = common.make_tmp("index")
    backend, onscreen = None, []
    try:
        m = measure(video, start, end, meta["has_audio"], tmp, args.min_silence)
        if args.ocr:
            backend, onscreen = read_onscreen(video, start, end, span, tmp, args.ocr_lang, args.ocr_fps)
    finally:
        common.safe_rmtree(tmp)

    shots = shift(make_shots(find_cuts(m["scene"]), span, m["stats"], m["loud"]), origin)
    hits = [{**h, "t_s": round(h["t_s"] + origin, 2)} for h in sound_hits(m["loud"])]
    silences = [(round(a + origin, 2), round(b + origin, 2)) for a, b in m["silences"]]
    # Precise hits and how far each cut is from its nearest one. None (not an empty list)
    # when the audio couldn't be measured, so "no hits" is never claimed without looking.
    onsets = cut_sync = sync = cues = tempo = None
    env, env_t = m["env"]
    if env:
        env_t = origin + audio.env_start(video, start is not None, m["scene"][0]["t"] if m["scene"] else 0.0, env_t)
        if args.cues:
            cues = audio.cue_rises(env, env_t, common.read_times(args.cues))
        strength = audio.rise(env)
        onsets = audio.find_onsets(env, strength)
        audio.measure_shapes(env, m["full"], onsets, m["spectrum"])
        onsets = [{**o, "t_s": round(o["t_s"] + env_t, 3), "label": audio.shape_label(o)} for o in onsets]
        if args.transcript and Path(args.transcript).expanduser().exists():
            said = json.loads(Path(args.transcript).expanduser().read_text(encoding="utf-8"))["words"]
            for o in onsets:
                if any(w["start_s"] - 0.03 <= o["t_s"] <= w["end_s"] for w in said):
                    o["label"] = "speech"
        cut_sync = audio.pair_cuts([s["start_s"] for s in shots[1:]], [o["t_s"] for o in onsets])
        sync = audio.sync_summary(cut_sync)
        tempo = audio.tempo(strength)        # None: no clear beat
        if tempo:
            tempo["beats"] = [round(b + env_t, 3) for b in tempo["beats"]]
            offs = audio.beat_offsets([p["cut_s"] for p in cut_sync], tempo["beats"], tempo["bpm"])
            for p, off in zip(cut_sync, offs):
                p["beat_offset_ms"] = off
            tempo = {"bpm": tempo["bpm"], "corr": tempo["corr"], **audio.beat_summary(offs, tempo["bpm"]),
                     "beats": tempo["beats"]}
    elif meta["has_audio"]:
        print("[index] hit timing skipped: the 5 ms level pass gave nothing", file=sys.stderr)

    # The loudness series, for check.py's mix checks: a value at t covers the audio up to t + 0.1 s
    loud = [r for r in m["loud"] if "r128.M" in r and "r128.S" in r]
    loudness = None
    if loud:
        loudness = {"integrated_lufs": round(loud[-1].get("r128.I", -70.0), 1),
                    "lra_lu": round(loud[-1].get("r128.LRA", 0.0), 1),
                    "start_s": round(loud[0]["t"] + origin, 3), "step_s": 0.1,
                    "momentary": [round(r["r128.M"], 1) for r in loud],
                    "short_term": [round(r["r128.S"], 1) for r in loud]}

    swells = audio.swells(loudness) if loudness else None

    still_below, busy_from = motion_bands(shots)

    durations = sorted(s["dur_s"] for s in shots)
    index = {
        "video": video, "duration_s": round(meta["duration_s"], 2),
        "range": {"start_s": start, "end_s": end},
        "width": meta["width"], "height": meta["height"], "has_audio": meta["has_audio"],
        "shot_count": len(shots),
        "median_shot_s": round(durations[len(durations) // 2], 2) if durations else 0,
        "silence_s": round(sum(b - a for a, b in silences), 1),
        "shots": shots, "audio_events": hits,
        "onsets": onsets, "cut_sync": cut_sync, "sync": sync, "cues": cues, "tempo": tempo, "swells": swells,
        "loudness": loudness,
        "silences": [{"start_s": a, "end_s": b} for a, b in silences],
        "ocr_backend": backend,
        "onscreen_text": onscreen,
        "thresholds": {"cut_hard": HARD_CUT, "cut_floor": SOFT_CUT, "cut_ratio": SOFT_RATIO,
                       "min_shot_s": MIN_SHOT_S, "sample_fps": SAMPLES_PER_S},
    }
    (out / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")

    md = [f"# Index: {Path(video).name}", "",
          f"{ts(meta['duration_s'])} long, {len(shots)} shots (median {index['median_shot_s']}s), "
          f"{'audio' if meta['has_audio'] else 'NO audio'}, {index['silence_s']}s silent.", "",
          "Shot lines: `#n start-end (dur) | motion, light, sound`. "
          "Zoom any span with `frames.py --start --end`. On-screen text shows [zone]: top/upper/middle/lower/bottom.", ""]
    if sync:
        md[2] += (f" {len(onsets)} hit{'s' * (len(onsets) != 1)}; {sync['with_hit']} of {sync['cuts']} cuts have one within "
                  f"{sync['window_ms']} ms" + (f" (median offset {sync['median_offset_ms']:+d} ms)."
                                               if sync["with_hit"] else "."))
        md[4] += (" `cut: hit +80 ms late` is the nearest hit to the cut that starts the shot (hits are timed to "
                  "about 5-10 ms, cuts to one frame). Under dense music a hit can be hidden and the nearest one "
                  "can be a note of the music, so `no hit` means none was measured, not that none is there.")
    if tempo:
        md[2] += (f" Tempo about {tempo['bpm']:g} BPM (half or double time reads the same); "
                  f"{tempo['cuts_on_beat']} of {len(cut_sync)} cuts within {tempo['tol_ms']} ms of a beat "
                  f"({tempo['chance_pct']}% would be by chance).")
        md[4] += " `60 ms after the beat` is the cut's distance to the nearest beat."
    elif onsets is not None:
        md[2] += " No clear beat."
    hit_t = [h["t_s"] for h in hits] if onsets is None else [o["t_s"] for o in onsets]
    at_cut = {p["cut_s"]: p for p in cut_sync or []}
    for s in shots:
        inside = hit_t[bisect.bisect_left(hit_t, s["start_s"]):bisect.bisect_left(hit_t, s["end_s"])]
        extra = "; cut: " + audio.offset_words(at_cut[s["start_s"]]["offset_ms"]) if s["start_s"] in at_cut else ""
        beat = at_cut.get(s["start_s"], {}).get("beat_offset_ms")
        if beat is not None:
            extra += ", on the beat" if abs(beat) < 10 else f", {abs(beat)} ms {'after' if beat > 0 else 'before'} the beat"
        if inside and onsets is not None:
            extra += f"; {len(inside)} hit{'s' * (len(inside) != 1)}"
        elif inside:
            extra += "; " + ", ".join(f"sound hit {ts(t)}" for t in inside)
        md.append(f"#{s['index']} {ts(s['start_s'])}-{ts(s['end_s'])} ({s['dur_s']:.1f}s) | "
                  f"{describe(s, still_below, busy_from)}{extra}")
    if onscreen:
        md += ["", f"## On-screen text ({backend} OCR)", ""]
        md += [f"- {ts(o['start_s'])}-{ts(o['end_s'])} [{zone(o.get('box'))}] {o['text']}" for o in onscreen]
    if onsets:
        listed = sorted(sorted(onsets, key=lambda o: -o["rise_db"])[:300], key=lambda o: o["t_s"])
        by_hit = {p["hit_s"]: p for p in cut_sync}
        md += ["", "## Sound hits", "",
               "The words describe each hit's shape (attack, brightness, tone or noise, length), not what made "
               "it: `click-like` is any short bright sound. Under music the music is in the measures too."
               + (f" Showing the 300 that rise most, of {len(onsets)}." if len(onsets) > 300 else ""), ""]
        for o in listed:
            p = by_hit.get(o["t_s"])
            where = "" if not p else ("; on the cut" if p["offset_ms"] == 0 else
                                      f"; {abs(p['offset_ms'])} ms {'after' if p['offset_ms'] > 0 else 'before'} the cut "
                                      f"at {common.clock_ms(p['cut_s'])}")
            md.append(f"- {common.clock_ms(o['t_s'])} {o['label']}{where}")
    if swells:
        md += ["", "## Swells and drops", "",
               f"Short-term loudness moving {audio.SWELL_LU:g} LU or more over {audio.SWELL_OVER_S:g} s. "
               "It averages 3 s, so an abrupt step also shows here as a rise over about 3 s.", ""]
        md += [f"- {ts(w['start_s'])}-{ts(w['end_s'])} {w['kind']} {w['change_lu']:+.1f} LU" for w in swells[:200]]
    if silences:
        md += ["", f"## Silences (>= {args.min_silence}s)", ""]
        md += [f"- {ts(a)}-{ts(b)} ({b - a:.1f}s)" for a, b in silences[:400]]
    md_path = out / "index.md"
    md_path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[index] {len(shots)} shots, {len(hits if onsets is None else onsets)} sound hits, {len(silences)} silences"
          + (f", {len(onscreen)} on-screen text spans" if args.ocr else ""), file=sys.stderr)
    print(f"INDEX: {md_path.resolve()}")


if __name__ == "__main__":
    main()

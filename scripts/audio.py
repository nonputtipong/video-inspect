"""Audio measures shared by index.py, check.py and study.py. ffmpeg plus the standard library.

Hit onsets: ffmpeg writes the RMS level of every 5 ms of high-passed audio (`hit_cmd`), and
Python finds where that level rises out of its recent background, then walks back to the
start of the attack. Each cut is then paired with its nearest hit.

These are measurements, not verdicts. Under dense music a hit can be hidden (its rise over
the music is too small), and a cut's nearest "hit" can be one of the music's own notes.
"""
from __future__ import annotations

import bisect
import statistics
import subprocess
from pathlib import Path

import common

HOP_S = 0.005            # envelope step; an onset is reported at its frame's centre, so +-2.5 ms
RATE = 48000             # the envelope is measured at one rate so a frame is always 5 ms
# The envelope ignores bass (two high-passes in a row, 24 dB an octave), so a music bed's
# kick and bass line don't mask a hit placed on top: an 80 Hz note twice as loud as a click
# hid it with one filter and not with two. A sound with nothing above 150 Hz is not seen.
HIGHPASS_HZ = 150
FLOOR_DB = -120.0        # digital silence reads -inf; clamp it
# An onset is a level this far above the median of the previous BEFORE_S. Measured with the
# 14 macOS system sounds mixed over three beds (pink noise, a chord with tremolo, bass plus
# held notes): at 8 dB all 14 are found once they peak 9 dB over the bed (12-13 at 6 dB over)
# with no false hits. At 6 dB the chord's own wobble (up to 6.9 dB) gave 84 false hits; at
# 10 dB a sound 9 dB over its bed starts to be missed.
ONSET_RISE_DB = 8.0
BEFORE_S = 0.4
# The background never counts as quieter than this (dBFS, 5 ms RMS), so a sound must reach
# -52 dBFS to be a hit and codec noise in a silent stretch (around -80) is not one.
ONSET_FLOOR_DB = -60.0
ATTACK_DB = 3.0          # walking back: a frame this far over its background is part of the attack
ATTACK_BACK_S = 0.2      # ... but no further back than this (a slow swell has no sharp start)
ONSET_GAP_S = 0.08       # closest two onsets can be: 16th notes at 180 BPM are 83 ms apart
SYNC_WINDOW_S = 0.25     # a cut is paired with the nearest hit within this (the handoff's figure)

ENV_FILTERS = (f"aresample={RATE},highpass=f={HIGHPASS_HZ},highpass=f={HIGHPASS_HZ},asetnsamples=n={int(RATE * HOP_S)},"
               "astats=metadata=1:reset=1:measure_perchannel=none:measure_overall=RMS_level,"
               "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=onset.txt")


def hit_cmd(video: str, seek: list[str], cut: tuple) -> list[str]:
    """The ffmpeg pass behind the onsets. Run it with cwd=tmp; it writes tmp/onset.txt."""
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", *seek, "-i", video, "-vn",
            "-af", common.trim(cut, audio=True) + ENV_FILTERS, "-f", "null", "-"]


def read_env(tmp: Path) -> tuple[list[float], float]:
    """(level in dB of each 5 ms frame in order, the first frame's time on the pass's clock)."""
    path = tmp / "onset.txt"
    if not path.exists():
        return [], 0.0
    out, first = [], None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("lavfi."):
            try:
                out.append(max(FLOOR_DB, float(line.partition("=")[2])))   # max() also drops nan
            except ValueError:
                out.append(FLOOR_DB)
        elif first is None and "pts_time:" in line:
            first = float(line.split("pts_time:", 1)[1].split()[0])
    return out, first or 0.0


def envelope(video: str, tmp: Path) -> list[float]:
    """The whole file's level frames (sound-only use: no video clock to line up with)."""
    subprocess.run(hit_cmd(video, [], (0.0, None)), cwd=tmp, capture_output=True)
    return read_env(tmp)[0]


def stream_starts(video: str) -> tuple[float, float]:
    """Start timestamps of the first video and the first audio stream, in seconds."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,start_time",
                        "-of", "csv=p=0", video], capture_output=True, text=True, encoding="utf-8", errors="replace")
    found: dict[str, float] = {}
    for row in r.stdout.splitlines():
        kind, _, t = row.partition(",")
        try:
            found.setdefault(kind, float(t.split(",")[0]))
        except ValueError:      # N/A
            pass
    return found.get("video", 0.0), found.get("audio", 0.0)


def env_start(video: str, ranged: bool, first_video_t: float, first_env_t: float) -> float:
    """Time of the envelope's first frame on the video pass's clock, so hits and cuts compare.

    The two streams of a file rarely start on the same timestamp (always apart in .ts, where
    the video starts a frame or two after the audio; sometimes in .mkv and .mp4). ffmpeg
    counts a video-only pass and a sound-only pass of a .ts from each stream's own start, so
    without this a hit read 23 ms late on a .ts copy of a clip, and 100 ms early on an .mkv.
      whole file    the sound starts (audio start - video start) after the first video frame
      a range       indexed formats put both passes on one clock; .ts-like ones need the gap
    """
    v0, a0 = stream_starts(video)
    if not ranged:
        return first_video_t + a0 - v0
    return first_env_t + (a0 - v0 if common.seek_margin(video) else 0.0)


def rise(env: list[float]) -> list[float]:
    """How far each frame stands above the median of the BEFORE_S before it (dB, never negative)."""
    out = [0.0] * len(env)
    span = int(BEFORE_S / HOP_S)
    win: list[float] = []            # the previous `span` frames, kept sorted
    for i, v in enumerate(env):
        if win:
            out[i] = max(0.0, v - max(win[len(win) // 2], ONSET_FLOOR_DB))
        bisect.insort(win, v)
        if i >= span:
            del win[bisect.bisect_left(win, env[i - span])]
    return out


def find_onsets(env: list[float], strength: list[float] | None = None) -> list[dict]:
    """Hits: [{"t_s", "rise_db", "level_db"}], times from the start of the envelope."""
    s = rise(env) if strength is None else strength
    gap, back = int(ONSET_GAP_S / HOP_S), int(ATTACK_BACK_S / HOP_S)
    hits: list[dict] = []
    last = -gap                      # frame the last hit starts at
    for i in range(1, len(s)):
        if s[i] < ONSET_RISE_DB or s[i - 1] >= ONSET_RISE_DB:
            continue
        j = i
        while j > 0 and i - j < back and s[j - 1] >= ATTACK_DB:
            j -= 1
        if j - last < gap:           # still the last hit's attack, or too close to tell apart
            continue
        hits.append({"t_s": round((j + 0.5) * HOP_S, 3), "rise_db": round(max(s[i:i + 10]), 1),
                     "level_db": round(max(env[i:i + 20]), 1)})
        last = j
    return hits


def pair_cuts(cuts: list[float], hit_times: list[float], window: float = SYNC_WINDOW_S) -> list[dict]:
    """Each cut with its nearest hit: offset_ms > 0 means the hit lands after the cut (late).
    hit_s and offset_ms are None when no hit starts within `window` of the cut."""
    out = []
    for c in cuts:
        i = bisect.bisect_left(hit_times, c)
        near = [h for h in hit_times[max(0, i - 1):i + 1] if abs(h - c) <= window]
        h = min(near, key=lambda x: abs(x - c)) if near else None
        out.append({"cut_s": round(c, 3), "hit_s": h,
                    "offset_ms": None if h is None else int(round((h - c) * 1000))})
    return out


def sync_summary(pairs: list[dict]) -> dict:
    offs = [p["offset_ms"] for p in pairs if p["offset_ms"] is not None]
    return {"window_ms": int(SYNC_WINDOW_S * 1000), "cuts": len(pairs), "with_hit": len(offs),
            "pct_with_hit": round(100 * len(offs) / len(pairs)) if pairs else None,
            "median_offset_ms": int(statistics.median(offs)) if offs else None}


def offset_words(offset_ms: int | None) -> str:
    if offset_ms is None:
        return f"no hit within {int(SYNC_WINDOW_S * 1000)} ms"
    return f"hit {offset_ms:+d} ms" + (" late" if offset_ms > 0 else " early" if offset_ms < 0 else "")

"""Audio measures shared by index.py, check.py and study.py. ffmpeg plus the standard library.

Shape labels: each hit's attack time, length, brightness and tonality, put into words such
as "short bright transient (click-like)". They say what a sound is shaped like, never what
it is.

Music timing: tempo by autocorrelating the onset signal over 60-180 BPM, a beat grid
lined up against it block by block, each cut's distance to the nearest beat, and swells and
drops in short-term loudness.

Reference profile: the same few numbers for every video (cuts per minute, hits on cuts,
loudness, beat alignment, silence, speaking pace), the middle and spread of them across the
user's reference videos, and where a render falls against that. Distance from the
references is a measurement; whether it matters is for the reader to judge.

Mix checks (for check.py): samples at full scale, sudden loudness jumps, how far the bed
sits under the voice, and planned effects that don't rise out of the bed.

Hit onsets: ffmpeg writes the RMS level of every 5 ms of high-passed audio (`hit_cmd`), and
Python finds where that level rises out of its recent background, then walks back to the
start of the attack. Each cut is then paired with its nearest hit.

These are measurements, not verdicts. Under dense music a hit can be hidden (its rise over
the music is too small), and a cut's nearest "hit" can be one of the music's own notes.
"""
from __future__ import annotations

import bisect
import math
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

LEVELS = (f"asetnsamples=n={int(RATE * HOP_S)},"
          "astats=metadata=1:reset=1:measure_perchannel=none:measure_overall=RMS_level,"
          "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=")
# Spectrum of the unfiltered sound in mono, one 21 ms window every 10.7 ms: the centroid says
# how bright it is and the flatness whether it is a tone (near 0) or noise (near 1).
SPEC_FILTERS = ("aformat=channel_layouts=mono,aspectralstats=win_size=1024,"
                "ametadata=print:key=lavfi.aspectralstats.1.centroid:file=centroid.txt,"
                "ametadata=print:key=lavfi.aspectralstats.1.flatness:file=flatness.txt")


def hit_cmd(video: str, seek: list[str], cut: tuple) -> list[str]:
    """The ffmpeg pass behind the onsets and their shapes. Run it with cwd=tmp. It writes
    onset.txt (5 ms levels above 150 Hz), full.txt (5 ms levels with the bass in),
    centroid.txt and flatness.txt there."""
    graph = (f"[0:a]{common.trim(cut, audio=True)}aresample={RATE},asplit=3[hp][full][spec];"
             f"[hp]highpass=f={HIGHPASS_HZ},highpass=f={HIGHPASS_HZ},{LEVELS}onset.txt[a];"
             f"[full]{LEVELS}full.txt[b];[spec]{SPEC_FILTERS}[c]")
    return ["ffmpeg", "-hide_banner", "-loglevel", "error", *seek, "-i", video, "-filter_complex", graph,
            "-map", "[a]", "-f", "null", "-", "-map", "[b]", "-f", "null", "-", "-map", "[c]", "-f", "null", "-"]


def read_env(tmp: Path, name: str = "onset.txt") -> tuple[list[float], float]:
    """(level in dB of each 5 ms frame in order, the first frame's time on the pass's clock)."""
    path = tmp / name
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


def read_spectrum(tmp: Path) -> list[tuple[float, float, float]]:
    """[(seconds from the envelope's first frame to the window's centre, centroid Hz, flatness 0-1)]."""
    def column(name: str) -> list[tuple[float, float]]:
        path, rows, t = tmp / name, [], 0.0
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if path.exists() else []:
            if "pts_time:" in line:
                t = float(line.split("pts_time:", 1)[1].split()[0])
            elif line.startswith("lavfi."):
                try:
                    rows.append((t, float(line.partition("=")[2])))
                except ValueError:
                    rows.append((t, 0.0))
        return rows
    cen, flat = column("centroid.txt"), column("flatness.txt")
    first = cen[0][0] if cen else 0.0
    return [(t - first, c, f) for (t, c), (_, f) in zip(cen, flat)]


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


SHAPE_MAX_S = 2.0          # a sound still going after this is "sustained": its length isn't followed further


def measure_shapes(env: list[float], full: list[float], onsets: list[dict], spectrum: list[tuple]) -> None:
    """Add each hit's shape to its dict. Times are still from the envelope's start here.
      attack_ms    start to within 6 dB of its peak (levels above 150 Hz, where attacks are sharp)
      dur_ms       start until the full-band level has died away 20 dB, or is back within 3 dB
                   of what played before, or the next hit starts (at most SHAPE_MAX_S)
      bass_db      how much louder it is with its bass in than above 150 Hz, at its peak
      centroid_hz, flatness   over its first 0.3 s, louder moments counting more
    """
    spec_t = [row[0] for row in spectrum]
    for n, o in enumerate(onsets):
        j = int(o["t_s"] / HOP_S)
        end = min(len(full), j + int(SHAPE_MAX_S / HOP_S), int(onsets[n + 1]["t_s"] / HOP_S) if n + 1 < len(onsets) else len(full))
        head = env[j:min(end, j + 60)]
        o["attack_ms"] = next(k for k, v in enumerate(head) if v >= max(head) - 6) * int(HOP_S * 1000)
        o["dur_ms"] = o["bass_db"] = o["centroid_hz"] = o["flatness"] = None
        body = full[j:min(end, j + 60)]
        if not body:
            continue
        top = max(body)
        before = max(statistics.median(full[max(0, j - 80):j]) if j else FLOOR_DB, ONSET_FLOOR_DB)
        low = max(top - 20, before + 3)
        stop = next((k for k in range(j + body.index(top), end) if full[k] < low), end)
        o["dur_ms"] = (stop - j) * int(HOP_S * 1000)
        o["bass_db"] = round(top - max(head), 1)
        rows = spectrum[bisect.bisect_left(spec_t, j * HOP_S):bisect.bisect_right(
            spec_t, j * HOP_S + min(max(o["dur_ms"] / 1000, 0.02), 0.3))]
        weight = [10 ** ((full[min(len(full) - 1, int(r[0] / HOP_S))] - top) / 10) for r in rows]
        if rows and sum(weight) > 0:
            o["centroid_hz"] = int(sum(w * r[1] for w, r in zip(weight, rows)) / sum(weight))
            o["flatness"] = round(sum(w * r[2] for w, r in zip(weight, rows)) / sum(weight), 3)


# Shape rules. They describe a sound's envelope and spectrum, not what made it: "click-like"
# is a short bright sound, whether a click, a tick or a rimshot. Set on generated sounds
# (noise clicks and claps, impacts and a kick with a 50-120 Hz body, noise whooshes and a
# riser, decaying tones and a beep), which each get their own label, and checked on the 14
# macOS system sounds: Tink, Pop and Purr read click-like, Bottle, Glass and Submarine
# ding-like, and the rest get the plain measured words (Basso: "fast attack, bright, tonal,
# ~140 ms"). ffmpeg's centroid weighs every frequency bin alike, so a little noise pulls it
# up: an impact with a 70 Hz body read 5.8 kHz. That is why "low" also looks at bass_db.
SHORT_MS = 60              # as long as a click, tick or clap
FAST_ATTACK_MS = 40        # at its level within this: struck, not swelled
SLOW_ATTACK_MS = 100       # takes at least this long to arrive: a swell
RING_MS = 250              # a tone that lasts this long rings
HIT_MS = (80, 800)         # how long an impact's body lasts
LOW_HZ = 1000              # centroid under this: the weight is in the bass and low mids
BASS_DB = 6.0              # ... or this much louder with its bass in: three quarters of it is under 150 Hz
BRIGHT_HZ = 1500           # centroid over this: bright
TONAL = 0.25               # flatness under this: mostly a tone
NOISY = 0.40               # flatness over this: mostly noise


def shape_label(o: dict) -> str:
    """Words for a hit's measured shape. Under music the bed is in the measures too."""
    a, d, c, f = o["attack_ms"], o["dur_ms"], o["centroid_hz"], o["flatness"]
    if d is None or c is None:
        return "fast attack" if a <= FAST_ATTACK_MS else "slow attack"
    low = c < LOW_HZ or o["bass_db"] >= BASS_DB
    if a >= SLOW_ATTACK_MS and f >= NOISY:
        return "slow noisy swell (whoosh-like)"
    if d <= SHORT_MS and c >= BRIGHT_HZ:
        return "short bright transient (click-like)"
    if a <= FAST_ATTACK_MS and f < TONAL and d >= RING_MS and not low:
        return "ringing tone (ding-like)"
    if a <= FAST_ATTACK_MS and low and HIT_MS[0] <= d <= HIT_MS[1]:
        return f"fast attack, low, ~{d} ms (hit/impact-like)"
    return ", ".join([
        "fast attack" if a <= FAST_ATTACK_MS else "slow attack" if a >= SLOW_ATTACK_MS else "medium attack",
        "low" if low else "bright" if c >= BRIGHT_HZ else "mid",
        "tonal" if f < TONAL else "noisy" if f >= NOISY else "part tone part noise",
        f"{d / 1000:g} s or more" if d >= SHAPE_MAX_S * 1000 else f"~{d} ms"])


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


# ---------- music timing ----------

BPM_RANGE = (60, 180)
# The beat's autocorrelation, as a share of the signal's own energy, must reach this, or the
# answer is "no clear beat" rather than a guess. Measured on generated signals only (there
# was no real music to tune on): click tracks from 60 to 174 BPM 0.96-0.99, a kick, snare
# and pad pattern 0.92, clicks played 20 ms off the grid 0.47; notes at random times (the
# timing of speech) 0.05-0.11 over four seeds, steady noise 0. A busy mix whose melody
# outweighed its drums read 0.20 with the wrong tempo, so refusing it is right.
BEAT_MIN_CORR = 0.3
# Tempo an octave off (60 for 120) correlates as well as the real one, so each candidate is
# weighted toward 120 BPM, losing 25% an octave away: a 120 BPM click reads 120, not 60.
OCTAVE_PENALTY = 0.25
BEAT_BLOCK_S = 8.0         # the grid's phase is found again for every block, so it follows drift
BEAT_MIN_SHARE = 0.2       # ... and a block with under this share of its onsets on the grid gets no beats
BEAT_MIN_FILL = 0.5        # ... nor one where fewer than this share of its beats have a note on them: one
                           # sound alone in a block is 100% "on the grid", and drew beats across 8 silent seconds
BEAT_SNAP_S = 0.025        # a beat moves to the strongest onset this close to its grid point
BEAT_TOL_S = 0.05          # a cut this close to a beat is on the beat (the handoff's figure)


def novelty(strength: list[float]) -> list[float]:
    """Where the level starts rising: spikes at note starts, flat through held sound. Rises
    under ATTACK_DB are left out, so the wobble of steady noise isn't read as rhythm."""
    raw = [0.0] + [max(0.0, b - a) if b >= ATTACK_DB else 0.0 for a, b in zip(strength, strength[1:])]
    return [0.25 * a + 0.5 * b + 0.25 * c for a, b, c in zip([0.0] + raw, raw, raw[1:] + [0.0])]


def tempo(strength: list[float]) -> dict | None:
    """{"bpm", "corr", "beats": [seconds from the envelope's start]}, or None for no clear beat."""
    nov = novelty(strength)
    lo, hi = int(60 / BPM_RANGE[1] / HOP_S), int(60 / BPM_RANGE[0] / HOP_S)
    # Blurred over 45 ms for the tempo only, so notes played 10-20 ms off the grid still line
    # up (a click track with 20 ms of jitter correlated 0.20 unblurred), and moved to sit
    # around zero, so only a repeating pattern correlates and steady noise does not.
    wide = [sum(nov[max(0, i - 4):i + 5]) for i in range(len(nov))]
    mean = sum(wide) / len(wide) if wide else 0.0
    flat = [v - mean for v in wide]
    energy = sum(v * v for v in flat)
    if energy <= 0 or len(nov) < 4 * hi:
        return None
    corr = {lag: sum(a * b for a, b in zip(flat, flat[lag:])) / energy for lag in range(lo - 1, hi + 2)}
    best = max(range(lo, hi + 1),
               key=lambda lag: corr[lag] * (1 - OCTAVE_PENALTY * abs(math.log2(60 / (lag * HOP_S) / 120))))
    if corr[best] < BEAT_MIN_CORR:
        return None
    a, b, c = corr[best - 1], corr[best], corr[best + 1]
    period = best + (0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c else 0.0)   # frames, between-frame peak

    beats: list[float] = []
    block, snap, bins = int(BEAT_BLOCK_S / HOP_S), int(BEAT_SNAP_S / HOP_S), int(round(period))
    for start in range(0, len(nov), block):
        part = nov[start:start + block]
        fold = [0.0] * bins               # the block's onsets stacked on one beat period
        for i, v in enumerate(part):
            if v:
                fold[int(i % period) % bins] += v * v     # squared: note starts outweigh wobble
        phase = max(range(bins), key=fold.__getitem__)
        if sum(fold[(phase + d) % bins] for d in (-1, 0, 1)) < BEAT_MIN_SHARE * (sum(fold) or 1):
            continue
        k, found, played = 0, [], 0
        while phase + k * period < len(part):
            at = start + int(round(phase + k * period))
            near = range(max(0, at - snap), min(len(nov), at + snap + 1))
            if any(nov[j] for j in near):
                at, played = max(near, key=nov.__getitem__), played + 1
            found.append(round((at + 0.5) * HOP_S, 3))
            k += 1
        if played >= BEAT_MIN_FILL * len(found):
            beats += [t for t in found if not beats or t - beats[-1] > 0.5 * period * HOP_S]
    if not beats:
        return None
    return {"bpm": round(60 / (period * HOP_S), 1), "corr": round(corr[best], 2), "beats": beats}


def beat_summary(offsets: list[int | None], bpm: float) -> dict:
    """How many cuts sit on a beat, and how many would by chance (the tolerance's share of a beat)."""
    on = sum(1 for o in offsets if o is not None and abs(o) <= BEAT_TOL_S * 1000)
    return {"tol_ms": int(BEAT_TOL_S * 1000), "cuts_on_beat": on,
            "pct_cuts_on_beat": round(100 * on / len(offsets)) if offsets else None,
            "chance_pct": round(100 * 2 * BEAT_TOL_S * bpm / 60)}


def beat_offsets(cuts: list[float], beats: list[float], bpm: float) -> list[int | None]:
    """Each cut's distance to the nearest beat in ms (+ after the beat), None where there is no beat."""
    out: list[int | None] = []
    for c in cuts:
        i = bisect.bisect_left(beats, c)
        near = [b for b in beats[max(0, i - 1):i + 1] if abs(b - c) <= 30 / bpm]   # within half a beat
        out.append(int(round((c - min(near, key=lambda b: abs(b - c))) * 1000)) if near else None)
    return out


# A swell or drop: short-term loudness moving SWELL_LU over SWELL_OVER_S (the handoff's
# figures). Short-term loudness averages 3 s, so an abrupt step also shows as a rise over
# about 3 s; check.py's "loudness jumps" tells the two apart. Below SWELL_FLOOR_LUFS is
# silence: sound coming back from it counts as a rise from the floor, not from -120.
SWELL_LU = 6.0
SWELL_OVER_S = 2.0
SWELL_FLOOR_LUFS = -50.0


def swells(loud: dict) -> list[dict]:
    """[{"kind": "swell" | "drop", "start_s", "end_s", "change_lu"}] from index.json's "loudness"."""
    step, t0 = loud["step_s"], loud["start_s"]
    s = [max(SWELL_FLOOR_LUFS, v) for v in loud["short_term"]]
    k = int(round(SWELL_OVER_S / step))
    out: list[dict] = []
    i = int(round(SHORT_TERM_S / step)) + k
    while i < len(s):
        d = s[i] - s[i - k]
        if abs(d) < SWELL_LU:
            i += 1
            continue
        first = i
        while i + 1 < len(s) and (s[i + 1] - s[i + 1 - k]) * d > 0 and abs(s[i + 1] - s[i + 1 - k]) >= SWELL_LU:
            i += 1
        out.append({"kind": "swell" if d > 0 else "drop", "start_s": round(t0 + (first - k) * step, 1),
                    "end_s": round(t0 + i * step, 1), "change_lu": round(s[i] - s[first - k], 1)})
        i += 1
    return out


# ---------- mix checks ----------

# Clipping is measured as time pinned near full scale, not as samples exactly at it: AAC
# and MP3 smooth a clipped wave's flat tops, so after encoding almost none of it is exactly
# at full scale (1000 clipped stretches in a PCM tone left 0-15 such samples after AAC, and
# ffmpeg's astats flat factor read 0). What survives is how much of the wave sits within
# 0.5 dB of full scale. Measured on a four-tone mix, noise and a sine, as PCM, AAC 128k/192k
# and MP3: clipped by about 1 dB 2-15% of samples, by 3 dB 6-33%, by 6 dB 14-52%; the same
# mix unclipped and peaking at -0.1 dBFS 0.07-0.16%. A codec can lift a peak by about 0.5 dB,
# so a steady tone at -1 dBFS read clipped at its abrupt start after AAC; one at -3 dBFS never.
# A CLIP_WIN_S window is clipped at CLIP_SHARE or more. A mix limited with no headroom at
# all (or a full-scale test tone, 21%) reads the same, and the true-peak check fails it too.
CLIP_WIN_S = 0.05
CLIP_SHARE = 0.01
_NEAR_FULL = bytes(1 if 0x79 <= b <= 0x86 else 0 for b in range(256))   # 16-bit high bytes within 0.5 dB of full scale


def clipping(video: str) -> dict:
    """{"seconds": total clipped time, "spans": [[start_s, end_s]]} for the first audio stream."""
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries",
                        "stream=sample_rate,channels", "-of", "csv=p=0", video],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    rate, ch = (int(x) for x in r.stdout.split()[0].split(",")[:2])
    win = int(rate * CLIP_WIN_S)
    p = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video, "-map", "0:a:0",
                          "-f", "s16le", "-"], stdout=subprocess.PIPE)
    clipped: set[int] = set()        # window numbers
    base = 0
    while True:
        chunk = p.stdout.read(2 * ch * win * 64)
        if len(chunk) < 2 * ch:
            break
        for c in range(ch):
            near = chunk[2 * c + 1::2 * ch].translate(_NEAR_FULL)   # this channel's high bytes
            if near.count(1):
                clipped.update(base + k // win for k in range(0, len(near), win)
                               if near.count(1, k, k + win) >= CLIP_SHARE * win)
        base += 64
    p.wait()
    spans: list[list[float]] = []
    for w in sorted(clipped):
        if spans and abs(spans[-1][1] - w * CLIP_WIN_S) < 1e-6:
            spans[-1][1] = round((w + 1) * CLIP_WIN_S, 2)
        else:
            spans.append([round(w * CLIP_WIN_S, 2), round((w + 1) * CLIP_WIN_S, 2)])
    return {"seconds": round(len(clipped) * CLIP_WIN_S, 2), "spans": spans}


# A sudden jump: short-term loudness (a 3 s average) up JUMP_LU within JUMP_OVER_S. Because
# of the averaging that is a step of about 10 dB that stays for a second: on a generated
# 12 dB step the short-term value crossed +6 LU 0.6 s after it, and a 6 dB step never did.
# The moment is then read from momentary loudness, which was right to 0.1 s.
JUMP_LU = 6.0
JUMP_OVER_S = 1.0
JUMP_QUIET_LUFS = -40.0    # sound starting from near silence is a start, not a jump
JUMP_CUT_S = 0.5           # a jump this close to a cut is a scene change
SHORT_TERM_S = 3.0         # short-term loudness isn't full until this far in, so nothing is judged before it


def loud_jumps(loud: dict, cuts: list[float]) -> list[dict]:
    """Jumps away from cuts: [{"t_s", "rise_lu"}]. `loud` is index.json's "loudness"."""
    step, t0, m, s = loud["step_s"], loud["start_s"], loud["momentary"], loud["short_term"]
    k = int(round(JUMP_OVER_S / step))
    out = []
    i = int(round(SHORT_TERM_S / step)) + k
    while i < len(s):
        before = s[i - k]
        if before < JUMP_QUIET_LUFS or s[i] - before < JUMP_LU:
            i += 1
            continue
        j = next((u for u in range(max(0, i - k - 5), i + 1) if m[u] >= before + JUMP_LU), i - k)
        rise = s[i] - before
        while i + 1 < len(s) and s[i + 1] - s[i + 1 - k] >= JUMP_LU:   # still the same jump
            i += 1
            rise = max(rise, s[i] - s[i - k])
        t = t0 + j * step
        if not any(abs(t - c) <= JUMP_CUT_S for c in cuts):
            out.append({"t_s": round(t, 1), "rise_lu": round(rise, 1)})
        i += 1
    return out


# Rule of thumb, not a measurement of intelligibility: momentary loudness in the pauses
# between words stands in for the bed under the voice. A bed within VOICE_BED_LU of the
# speech level is probably too loud (the handoff's figure; common mix advice is 6-12 LU).
# Pauses longer than PAUSE_MAX_S are left out: music is often brought up where nobody talks.
VOICE_BED_LU = 6.0
PAUSE_MIN_S = 0.6          # shortest pause a 400 ms loudness window fits inside
PAUSE_MAX_S = 3.0
RUN_GAP_S = 0.3            # words closer than this are one run of speech


def voice_vs_bed(loud: dict, words: list[dict]) -> dict | None:
    """Speech level, bed level in the pauses, and the pauses where the bed comes within
    VOICE_BED_LU of the voice. None when there are no words or no pause of a usable length."""
    step, t0, m = loud["step_s"], loud["start_s"], loud["momentary"]

    def levels(a: float, b: float) -> list[float]:
        """Momentary values whose 400 ms window lies inside a..b (a frame at t covers t-0.3..t+0.1)."""
        lo, hi = math.ceil((a + 0.3 - t0) / step), math.floor((b - 0.1 - t0) / step)
        return [m[i] for i in range(max(0, lo), min(len(m) - 1, hi) + 1)]

    speech: list[float] = []
    pauses = []
    run_start = words[0]["start_s"] if words else 0.0
    for w, nxt in zip(words, words[1:] + [None]):
        gap = nxt["start_s"] - w["end_s"] if nxt else RUN_GAP_S
        if gap < RUN_GAP_S:
            continue
        speech += levels(run_start, w["end_s"])
        if nxt:
            run_start = nxt["start_s"]
            got = levels(w["end_s"], nxt["start_s"]) if PAUSE_MIN_S <= gap <= PAUSE_MAX_S else []
            if got:
                pauses.append((w["end_s"], nxt["start_s"], statistics.median(got)))
    if not speech or not pauses:
        return None
    voice = statistics.median(speech)
    return {"voice_lufs": round(voice, 1), "bed_lufs": round(statistics.median(p[2] for p in pauses), 1),
            "margin_lu": round(voice - statistics.median(p[2] for p in pauses), 1), "pauses": len(pauses),
            "close": [{"start_s": round(a, 2), "end_s": round(b, 2), "margin_lu": round(voice - lv, 1)}
                      for a, b, lv in pauses if voice - lv < VOICE_BED_LU]}


# Rule of thumb: a planned effect should stand out of what was playing just before it. Its
# level is the loudest 20 ms from 50 ms before the cue to 150 ms after, the bed the 300 ms
# before that, both above 150 Hz. With nothing at the cue steady noise still reads about
# +1 dB and a busy bed more, so under BURIED_DB the effect is buried or missing.
BURIED_DB = 3.0


def cue_rises(env: list[float], env_t: float, cues: list[float]) -> list[dict]:
    """[{"cue_s", "rise_db"}]: how far each planned effect rises over the bed (None off the ends)."""
    def power(part: list[float]) -> float:
        return 10 * math.log10(sum(10 ** (x / 10) for x in part) / len(part))

    out = []
    for c in cues:
        i = int(round((c - env_t) / HOP_S))
        bed, after = env[max(0, i - 70):max(0, i - 10)], env[max(0, i - 10):max(0, i + 30)]
        if not bed or len(after) < 4:
            out.append({"cue_s": round(c, 3), "rise_db": None})
            continue
        peak = max(power(after[k:k + 4]) for k in range(len(after) - 3))
        # out of silence the rise is counted from ONSET_FLOOR_DB, not from -120
        out.append({"cue_s": round(c, 3), "rise_db": round(max(peak, ONSET_FLOOR_DB) - max(power(bed), ONSET_FLOOR_DB), 1)})
    return out


# ---------- reference profile ----------

# key, name in reports, how to print a value
METRICS = (("cuts_per_min", "cuts/min", "{:.0f}"), ("pct_cuts_with_hit", "cuts with a hit", "{:.0f}%"),
           ("median_hit_offset_ms", "median hit offset", "{:+.0f} ms"), ("lufs", "loudness", "{:.1f} LUFS"),
           ("lra_lu", "loudness range", "{:.1f} LU"), ("pct_cuts_on_beat", "cuts on the beat", "{:.0f}%"),
           ("silence_pct", "silence", "{:.0f}%"), ("wpm", "words/min", "{:.0f}"))
# With fewer references than this the quartiles are nearly one point, so the range a render
# is held against is lowest-to-highest instead of the middle half.
QUARTILES_FROM = 5


def metrics(idx: dict, words: list[dict] | None = None) -> dict:
    """One video's profile numbers from its index.json (and transcript words). None = not measured."""
    span = idx.get("range") or {}
    dur = ((span.get("end_s") or idx["duration_s"]) - (span.get("start_s") or 0.0)) or 1.0
    sync, loud, beat = idx.get("sync") or {}, idx.get("loudness") or {}, idx.get("tempo") or {}
    talk = words[-1]["end_s"] - words[0]["start_s"] if words else 0.0
    return {"cuts_per_min": round(idx["shot_count"] / dur * 60, 1),
            "pct_cuts_with_hit": sync.get("pct_with_hit"), "median_hit_offset_ms": sync.get("median_offset_ms"),
            "lufs": loud.get("integrated_lufs"), "lra_lu": loud.get("lra_lu"),
            "pct_cuts_on_beat": beat.get("pct_cuts_on_beat"),
            "silence_pct": round(sum(s["end_s"] - s["start_s"] for s in idx.get("silences", [])) / dur * 100, 1),
            "wpm": round(len(words) / talk * 60) if talk > 0.1 else None}


def make_profile(videos: dict[str, dict]) -> dict:
    """{"videos": {name: metrics}, "metrics": {key: {median, q1, q3, min, max, n}}} for study.py --profile."""
    out: dict = {}
    for key, _, _ in METRICS:
        vals = sorted(m[key] for m in videos.values() if m.get(key) is not None)
        if vals:
            q = statistics.quantiles(vals, n=4, method="inclusive") if len(vals) > 1 else [vals[0]] * 3
            out[key] = {"median": statistics.median(vals), "q1": q[0], "q3": q[2],
                        "min": vals[0], "max": vals[-1], "n": len(vals)}
    return {"videos": videos, "metrics": out}


def against_profile(m: dict, profile: dict) -> list[dict]:
    """Each metric against the references: [{"key", "name", "value", "low", "high", "median", "n",
    "where"}], where = "inside" | "below" | "above", or None when either side has no number."""
    rows = []
    for key, name, fmt in METRICS:
        ref, value = profile["metrics"].get(key), m.get(key)
        row = {"key": key, "name": name, "fmt": fmt, "value": value, "where": None}
        if ref:
            wide = ref["n"] >= QUARTILES_FROM
            row.update(low=ref["q1"] if wide else ref["min"], high=ref["q3"] if wide else ref["max"],
                       median=ref["median"], n=ref["n"])
            if value is not None:
                row["where"] = "below" if value < row["low"] else "above" if value > row["high"] else "inside"
        rows.append(row)
    return rows

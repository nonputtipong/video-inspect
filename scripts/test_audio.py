"""Self-check for audio.py on generated sounds with known answers: hit onsets land within
10 ms of where the sounds were planted, loud bass under them doesn't hide them or add false
ones, and each cut is paired with the right hit and offset. Then the mix checks: a clipped
stretch, a loudness jump, a bed too close to the voice and a buried effect are each found,
and their clean twins are not.
Usage: python3 -I scripts/test_audio.py   (needs ffmpeg)
"""
import array
import math
import random
import subprocess
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import audio  # noqa: E402
import common  # noqa: E402

SR = 48000
rng = random.Random(1)


def add(buf, at, samples):
    i0 = round(at * SR)
    for k, v in enumerate(samples):
        buf[i0 + k] += v


def noise(dur, amp):
    return [amp * (rng.random() * 2 - 1) for _ in range(int(dur * SR))]


def tone(dur, freq, amp, decay=0.0):
    """A sine; with `decay` (seconds) it dies away like a struck object."""
    return [amp * math.sin(2 * math.pi * freq * k / SR) * (math.exp(-k / SR / decay) if decay else 1.0)
            for k in range(int(dur * SR))]


def save(buf, path):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(array.array("h", [max(-32767, min(32767, int(v * 32767))) for v in buf]).tobytes())


def onsets_of(buf, tmp, name):
    save(buf, tmp / name)
    return [o["t_s"] for o in audio.find_onsets(audio.envelope(str(tmp / name), tmp))]


def near(found, truth, tol=0.010):
    return len(found) == len(truth) and all(abs(a - b) <= tol for a, b in zip(found, truth))


tmp = common.make_tmp("test_audio")
try:
    # Clicks and a tone burst on silence, one of them off the 5 ms grid
    planted = [1.0, 2.0025, 3.37, 5.0]
    buf = [0.0] * (7 * SR)
    for t in planted[:3]:
        add(buf, t, noise(0.004, 0.5))
    add(buf, 5.0, tone(0.3, 1000, 0.4))
    found = onsets_of(buf, tmp, "clean.wav")
    assert near(found, planted), (found, planted)

    # Each cut gets its nearest hit: on the cut, 80 ms late, 80 ms early, and none in reach
    pairs = audio.pair_cuts([1.0, 1.9225, 3.45, 6.0], found)
    offsets = [p["offset_ms"] for p in pairs]
    assert offsets[3] is None and all(abs(a - b) <= 10 for a, b in zip(offsets, [0, 80, -80])), offsets
    summary = audio.sync_summary(pairs)
    assert (summary["cuts"], summary["with_hit"], summary["pct_with_hit"]) == (4, 3, 75), summary

    # The same clicks over a bed: steady noise and a bass note twice as loud as the clicks.
    # The bass must not hide them, and neither bed may add hits of its own.
    bed = [0.0] * (7 * SR)
    add(bed, 0, noise(7, 0.03))
    add(bed, 0, tone(7, 80, 0.8))
    for t in planted[:3]:
        add(bed, t, noise(0.004, 0.4))
    found = onsets_of(bed, tmp, "bed.wav")
    assert near(found, planted[:3]), (found, planted[:3])

    # Planned effects: the three clicks stand out of the bed, a cue with nothing at it doesn't
    rises = [c["rise_db"] for c in audio.cue_rises(audio.envelope(str(tmp / "bed.wav"), tmp), 0.0, planted[:3] + [4.5])]
    assert all(r >= audio.BURIED_DB for r in rises[:3]) and rises[3] < audio.BURIED_DB, rises

    # Clipping: a tone pushed 3 dB past full scale for half a second is found where it is,
    # as PCM and after AAC (which smooths the flat tops); the tone peaking at -3 dBFS is clean
    clean = tone(4, 1000, 0.7)
    save(clean, tmp / "headroom.wav")
    assert audio.clipping(str(tmp / "headroom.wav"))["spans"] == []
    add(clean, 2.0, tone(0.5, 1000, 0.7))
    save(clean, tmp / "clipped.wav")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", "clipped.wav", "-c:a", "aac",
                    "clipped.m4a"], cwd=tmp, check=True)
    for name in ("clipped.wav", "clipped.m4a"):
        spans = audio.clipping(str(tmp / name))["spans"]
        assert len(spans) == 1 and abs(spans[0][0] - 2.0) <= 0.05 and abs(spans[0][1] - 2.5) <= 0.05, (name, spans)
finally:
    common.safe_rmtree(tmp)


def loudness(level, seconds=20.0):
    """index.json's "loudness" for a level curve (LUFS at time t): 400 ms and 3 s averages."""
    def mean(i, frames):
        part = [10 ** (level(k / 10) / 10) for k in range(max(0, i - frames + 2), i + 2)]
        return round(10 * math.log10(sum(part) / len(part)), 1)
    n = int(seconds * 10)
    return {"step_s": 0.1, "start_s": 0.0, "momentary": [mean(i, 4) for i in range(n)],
            "short_term": [mean(i, 30) for i in range(n)]}


# A 12 dB step at 10 s is a jump, timed within 0.2 s. A cut beside it explains it away,
# and neither a 6 dB step nor the same 12 dB spread over 6 s is a jump.
step = loudness(lambda t: -30 if t < 10 else -18)
jumps = audio.loud_jumps(step, [])
assert len(jumps) == 1 and abs(jumps[0]["t_s"] - 10) <= 0.2 and jumps[0]["rise_lu"] >= 6, jumps
assert audio.loud_jumps(step, [10.2]) == []
assert audio.loud_jumps(loudness(lambda t: -30 if t < 10 else -24), []) == []
assert audio.loud_jumps(loudness(lambda t: -30 + 12 * min(1, max(0, (t - 8) / 6))), []) == []

# Speech at -16 LUFS in 2 s runs with 1 s pauses. A bed 4 LU under it is too close in every
# pause; a bed 14 LU under is fine. No words, or no pauses, gives nothing to judge.
words = [{"start_s": a + k * 0.5, "end_s": a + k * 0.5 + 0.45} for a in (1, 4, 7, 10) for k in range(4)]
talking = lambda t: any(a <= t < a + 2 for a in (1, 4, 7, 10))
loud_bed = audio.voice_vs_bed(loudness(lambda t: -16 if talking(t) else -20), words)
assert 3 <= loud_bed["margin_lu"] <= 5 and len(loud_bed["close"]) == loud_bed["pauses"] == 3, loud_bed
quiet_bed = audio.voice_vs_bed(loudness(lambda t: -16 if talking(t) else -30), words)
assert 13 <= quiet_bed["margin_lu"] <= 15 and not quiet_bed["close"], quiet_bed
assert audio.voice_vs_bed(step, []) is None and audio.voice_vs_bed(step, words[:4]) is None

print("ok: hit onsets, cut sync and mix checks")

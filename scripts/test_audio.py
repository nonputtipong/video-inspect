"""Self-check for audio.py on generated sounds with known answers: hit onsets land within
10 ms of where the sounds were planted, loud bass under them doesn't hide them or add false
ones, and each cut is paired with the right hit and offset.
Usage: python3 -I scripts/test_audio.py   (needs ffmpeg)
"""
import array
import math
import random
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
finally:
    common.safe_rmtree(tmp)

print("ok: hit onsets and cut sync")

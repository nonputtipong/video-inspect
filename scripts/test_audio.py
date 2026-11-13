"""Self-check for audio.py on generated sounds with known answers: hit onsets land within
10 ms of where the sounds were planted, loud bass under them doesn't hide them or add false
ones, and each cut is paired with the right hit and offset. Then the mix checks: a clipped
stretch, a loudness jump, a bed too close to the voice and a buried effect are each found,
and their clean twins are not. Then music timing: a 120 BPM click track reads 120 with its
beats in place, notes at random times read "no clear beat", and swells and drops are found.
Then shape labels: a click, an impact, a whoosh and a ding each get their own. Last, the
reference profile: medians and ranges across videos, and where another video falls.
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

    # Music timing: clicks every half second are 120 BPM, and the beats land on the clicks
    clicks = [0.0] * (20 * SR)
    for k in range(1, 39):
        add(clicks, k * 0.5, noise(0.004, 0.5))
    save(clicks, tmp / "clicks.wav")
    beat = audio.tempo(audio.rise(audio.envelope(str(tmp / "clicks.wav"), tmp)))
    assert beat and abs(beat["bpm"] - 120) <= 0.5 and len(beat["beats"]) >= 36, beat
    assert all(abs(b - round(b * 2) / 2) <= 0.010 for b in beat["beats"]), beat["beats"]
    # Cuts on a beat, 60 ms after one, 40 ms before one, and where there are no beats
    offsets = audio.beat_offsets([5.0, 7.06, 9.46, 25.0], beat["beats"], beat["bpm"])
    assert offsets[3] is None and all(abs(a - b) <= 10 for a, b in zip(offsets, [0, 60, -40])), offsets
    assert audio.beat_summary(offsets, 120) == {"tol_ms": 50, "cuts_on_beat": 2, "pct_cuts_on_beat": 50, "chance_pct": 20}

    # Notes at random times, like speech: no beat to report
    talk = [0.0] * (20 * SR)
    t = 0.5
    while t < 19:
        add(talk, t, tone(rng.uniform(0.06, 0.2), rng.randint(150, 900), 0.3))
        t += rng.uniform(0.12, 0.9)
    save(talk, tmp / "talk.wav")
    assert audio.tempo(audio.rise(audio.envelope(str(tmp / "talk.wav"), tmp))) is None

    # Shape labels: four sounds two seconds apart, each shaped like its name
    shapes = [0.0] * (9 * SR)
    add(shapes, 1.0, noise(0.004, 0.5))                                             # click
    crack = [v * math.exp(-k / SR / 0.025) for k, v in enumerate(noise(0.5, 0.25))]
    add(shapes, 3.0, [a + b for a, b in zip(tone(0.5, 70, 0.7, 0.11), crack)])        # impact: low body, short crack
    swell = noise(0.8, 0.4)
    add(shapes, 5.0, [v * math.sin(math.pi * k / len(swell)) ** 2 for k, v in enumerate(swell)])   # whoosh
    add(shapes, 7.0, [a + b for a, b in zip(tone(1.5, 1500, 0.4, 0.25), tone(1.5, 4050, 0.15, 0.25))])   # ding
    save(shapes, tmp / "shapes.wav")
    env = audio.envelope(str(tmp / "shapes.wav"), tmp)
    hits = audio.find_onsets(env)
    audio.measure_shapes(env, audio.read_env(tmp, "full.txt")[0], hits, audio.read_spectrum(tmp))
    labels = [audio.shape_label(h) for h in hits]
    assert len(labels) == 4 and all(word in label for word, label in zip(
        ("click-like", "impact-like", "whoosh-like", "ding-like"), labels)), list(zip(hits, labels))
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

# Swells and drops: 16 LU up over 4 s from 8 s, then 16 LU down over 4 s from 20 s. Flat is neither.
ramp = loudness(lambda t: -34 + 16 * min(1, max(0, (t - 8) / 4)) - 16 * min(1, max(0, (t - 20) / 4)), 30.0)
moves = audio.swells(ramp)
assert [w["kind"] for w in moves] == ["swell", "drop"], moves
assert abs(moves[0]["start_s"] - 8) <= 1.5 and moves[0]["change_lu"] >= 6 and moves[1]["change_lu"] <= -6, moves
assert audio.swells(loudness(lambda t: -20)) == []

# Reference profile. Three references: 20, 24 and 30 cuts a minute; one has no beat.
def index(shots, with_hit, on_beat, lufs):
    return {"duration_s": 60.0, "shot_count": shots, "silences": [{"start_s": 1.0, "end_s": 4.0}],
            "sync": {"pct_with_hit": with_hit, "median_offset_ms": -10},
            "loudness": {"integrated_lufs": lufs, "lra_lu": 5.0},
            "tempo": None if on_beat is None else {"pct_cuts_on_beat": on_beat}}


talk = [{"start_s": 0.0, "end_s": 0.4}, {"start_s": 29.6, "end_s": 30.0}]      # 2 words in 30 s: 4 a minute
refs = {"a": audio.metrics(index(20, 80, 60, -14.0), talk), "b": audio.metrics(index(24, 90, 70, -15.0)),
        "c": audio.metrics(index(30, 100, None, -13.0))}
assert refs["a"] == {"cuts_per_min": 20.0, "pct_cuts_with_hit": 80, "median_hit_offset_ms": -10, "lufs": -14.0,
                     "lra_lu": 5.0, "pct_cuts_on_beat": 60, "silence_pct": 5.0, "wpm": 4}, refs["a"]
profile = audio.make_profile(refs)
cuts = profile["metrics"]["cuts_per_min"]
assert (cuts["median"], cuts["min"], cuts["max"], cuts["n"]) == (24.0, 20.0, 30.0, 3), cuts
assert profile["metrics"]["pct_cuts_on_beat"]["n"] == 2 and profile["metrics"]["wpm"]["n"] == 1
# A render with 38 cuts a minute, half its cuts with a hit, in range on loudness, and no transcript
judged = {r["key"]: r for r in audio.against_profile(audio.metrics(index(38, 50, 65, -14.5)), profile)}
assert [judged[k]["where"] for k in ("cuts_per_min", "pct_cuts_with_hit", "lufs", "pct_cuts_on_beat", "wpm")] == [
    "above", "below", "inside", "inside", None], judged
assert (judged["cuts_per_min"]["low"], judged["cuts_per_min"]["high"]) == (20.0, 30.0)   # under 5 refs: lowest to highest

print("ok: hit onsets, cut sync, mix checks, music timing, shape labels and reference profile")

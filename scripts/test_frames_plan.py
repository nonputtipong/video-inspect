"""Self-check for frames.py's fill planning: busy shots get more fill per second than still ones,
the total stays inside the budget, and a lone shot isn't boosted.
Usage: python3 -I scripts/test_frames_plan.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frames  # noqa: E402


def shots(span, motions):
    d = span / len(motions)
    return [{"index": i, "start_s": i * d, "end_s": (i + 1) * d, "sharpest_s": (i + 0.5) * d, "motion": m}
            for i, m in enumerate(motions)]


def fill_per_shot(plan, ss):
    return [sum(1 for p in plan if not p["anchor"] and s["start_s"] <= p["t"] < s["end_s"]) for s in ss]


# 30 s, fill cap 4/s: still 2/s, moving 4/s, high motion 8/s (7.5 s each, minus fill next to the shot frame)
ss = shots(30, [0.1, 2, 3, 9])
plan = frames.plan_times(ss, 512, 0, 30, 4.0)
still, mov1, mov2, high = fill_per_shot(plan, ss)
assert 12 <= still <= 15 and 27 <= mov1 <= 30 and 27 <= mov2 <= 30 and 54 <= high <= 60, (still, mov1, mov2, high)

# Long video, tight budget: stays inside it and still leans toward the busy shot
ss = shots(600, [0.1, 2, 3, 9])
plan = frames.plan_times(ss, 100, 0, 600, 1.0)
still, _, _, high = fill_per_shot(plan, ss)
assert len(plan) <= 100 and high >= 3 * still, (len(plan), still, high)

# One moving shot (no other shots to compare with): plain 4/s, not boosted
ss = shots(30, [3])
assert 116 <= len(frames.plan_times(ss, 512, 0, 30, 4.0)) <= 121

# One still shot, or every shot equally still: weights are relative, so no halving
for motions in ([0.1], [0.1, 0.2, 0.1]):
    n = len(frames.plan_times(shots(30, motions), 512, 0, 30, 4.0))
    assert 112 <= n <= 121, (motions, n)

# No motion data (shots from an old index): even fill, as before
ss = [{k: v for k, v in s.items() if k != "motion"} for s in shots(30, [0, 0, 0])]
counts = fill_per_shot(frames.plan_times(ss, 512, 0, 30, 4.0), ss)
assert max(counts) - min(counts) <= 1, counts

print("ok: fill planning")

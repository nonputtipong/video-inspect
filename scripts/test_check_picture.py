"""Self-check for check.py's picture scan: builds a 4 s clip with one planted glitch frame (1.0 s), one real cut
(2.0 s) and a 41-frame freeze (2.5 s), then asserts check.py reports exactly the glitch and the freeze.
Usage: uv run --no-project --with numpy python -I scripts/test_check_picture.py
"""
import json, subprocess, sys, tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
H, W = 640, 360
y, x = np.mgrid[0:H, 0:W]
blocks = np.random.default_rng(1).integers(0, 255, (H // 40 + 2, W // 40 + 2, 3))
a = lambda i: np.stack([200 * (np.sin((x - 4 * i) / 18) * .5 + .5) + 40, 120 * (np.sin((x - 4 * i) / 18) * .5 + .5) + 30, 60 + 0 * x], -1)
b = lambda i: np.repeat(np.repeat(blocks, 40, 0), 40, 1)[(np.arange(H) + 3 * i) % (H + 40)][:, :W] * [0.3, 0.6, 1.0]
frames = [a(i) for i in range(120)] + [b(i) for i in range(120, 240)]
frames[60] = frames[200]
for i in range(151, 191):
    frames[i] = frames[150]

with tempfile.TemporaryDirectory() as d:
    vid = Path(d) / "t.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "60", "-i", "-",
                    "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p", str(vid)],
                   input=b"".join(np.clip(f, 0, 255).astype(np.uint8).tobytes() for f in frames), check=True)
    subprocess.run([sys.executable, "-I", str(HERE / "check.py"), str(vid), "--out-dir", d, "--expect-size", f"{W}x{H}"],
                   check=True, capture_output=True)
    c = {r["name"]: r for r in json.loads((Path(d) / "check.json").read_text())["checks"]}
    assert not c["frozen picture"]["pass"] and c["frozen picture"]["detail"].startswith("1 ") and "00:02.5" in c["frozen picture"]["detail"], c["frozen picture"]
    assert not c["glitch frames"]["pass"] and c["glitch frames"]["detail"].startswith("1 ") and "00:01.0" in c["glitch frames"]["detail"], c["glitch frames"]
    assert "jumps that aren't cuts" not in (Path(d) / "check.md").read_text(), "the real cut at 2.0 s was listed as a jump"
print("OK: glitch and freeze caught, real cut ignored")

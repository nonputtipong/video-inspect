"""Self-check for sheet.py's drawing: every mark lands on its own row, in its own colour, at
the pixel its time maps to, and shots get their thumbnails where they start.
Usage: python3 -I scripts/test_sheet.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sheet  # noqa: E402

W = sheet.WIDTH
idx = {"shots": [{"start_s": 0.0, "end_s": 40.0}, {"start_s": 40.0, "end_s": 40.05}, {"start_s": 40.05, "end_s": 70.0}],
       "onsets": [{"t_s": 45.0}], "tempo": {"beats": [50.0, 75.0]},
       "silences": [{"start_s": 55.0, "end_s": 58.0}],
       "loudness": {"start_s": 0.0, "step_s": 0.1, "momentary": [-60.0] * 400 + [-30.0] * 300}}
img = sheet.draw_marks(idx, 30.0, 30.0)          # the sheet for 30-60 s


def pixel(x, row, dy=2):
    y = sheet.TOP[row] + dy
    return tuple(img[(y * W + x) * 4:(y * W + x) * 4 + 4])


x = lambda t: round((t - 30.0) / 30.0 * W)
assert pixel(x(40.0), "cuts") == (*sheet.CUT, 255)
assert pixel(x(40.0), "spectrogram", 50) == (*sheet.CUT, 170) and pixel(x(40.0) + 5, "spectrogram", 50)[3] == 0
assert pixel(x(45.0), "hits") == (*sheet.HIT, 255) and pixel(x(45.0), "cuts") == (*sheet.LANE, 255)
assert pixel(x(50.0), "beats") == (*sheet.BEAT, 255)
assert sum(1 for px in range(W) if pixel(px, "beats") == (*sheet.BEAT, 255)) == 2      # one 2 px tick: the 75 s beat is off this sheet
# Loudness: -30 LUFS from 40 s fills the lower half, -60 before it draws nothing; silence is blue above the columns
assert pixel(x(50.0), "loudness", sheet.TALL["loudness"] - 3) == (*sheet.LOUD, 255)
assert pixel(x(50.0), "loudness", 5)[3] == 0 and pixel(x(35.0), "loudness", sheet.TALL["loudness"] - 3)[3] == 0
assert pixel(x(56.0), "loudness", 5) == (*sheet.SILENT, 255)

# Thumbnails: the shot already running starts at the left edge, a 50 ms shot is too close to get its own
places = sheet.place_thumbs(idx["shots"], 30.0, 30.0)
assert [p[0] for p in places] == [0, x(40.0)], places
print("ok: sheet marks")

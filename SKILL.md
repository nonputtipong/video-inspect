---
name: video-inspect
description: Free, local, offline video and image inspection. Watch a video through a shot index, timestamped frame grids, fast on-screen-text OCR with screen positions and a local Whisper transcript; check a finished render before posting (specs, loudness, silences, black or frozen frames, label spelling, text under TikTok/Reels/Shorts buttons, script vs speech); study reference videos (cut rhythm, sound hits, words per minute, label timeline); and compare rendered stills against reference images with side-by-side boards and measured colour, light and detail stats. Use it whenever the user wants a video or render watched, checked, QA'd, compared with references or studied for pacing, even if they don't name a tool. Never spends API credits unless the user explicitly turns paid transcription on.
---

# video-inspect

Read videos and images the way an editor would, using only local tools: ffmpeg, the macOS
Vision OCR (or tesseract) and a cached local Whisper. Scripts do the measuring; you do the judging.

## Safety defaults (why they exist)
- **No paid calls.** Transcripts use local Whisper. ElevenLabs runs only with `--source elevenlabs --allow-paid`
  and `VIDEO_INSPECT_ALLOW_PAID=1`. Set those only after the user approves that specific run, because each
  call spends their credits.
- **No network.** Whisper (mlx-whisper on Apple Silicon, faster-whisper elsewhere) runs through uv with `--offline`. If its package or model isn't cached on this machine,
  the script stops and says so. Ask the user before re-running with `--allow-network`.
- **No `.env` reading.** Keys come only from the shell environment.
- **Deletes are fenced.** Scripts delete only their own `vi_*` folders in the system temp dir.

Scripts live in `SKILL_DIR/scripts/` (the folder holding this file). Run them with `python3 -I` (on Windows: `python -I` or `py -3 -I`).

## Pick the mode by the job

| Job | Run |
|---|---|
| Understand or summarize a video | `transcript.py` (background) + `index.py --ocr` + `frames.py --index`, then read everything |
| Check a finished render before posting | `check.py` |
| Study reference videos (pacing, labels) | `study.py` |
| Compare renders/stills with reference images | `compare.py` |
| One moment | `frames.py --start --end --cells 16`, or `--at 4:32,7:10` for exact instants |

### Understand a video
1. `python3 -I SKILL_DIR/scripts/transcript.py VIDEO --out WORK/transcript.json --lang en`. Start it first, in the background; it's the slow part.
2. `python3 -I SKILL_DIR/scripts/index.py VIDEO --out-dir WORK --ocr [--ocr-lang th] [--ocr-fps 8] [--min-silence 0.3]`. OCR reads 4 frames a second by default; use `--ocr-fps 8` for one-word captions that change faster than that. Read `index.md` whole: one line per shot, sound hits, silences, and on-screen text with its `[zone]` (top/upper/middle/lower/bottom).
3. `python3 -I SKILL_DIR/scripts/frames.py VIDEO --out-dir WORK --index WORK/index.json`. Read every grid in the manifest in one message. Each cell has its time stamped top-left; cells aren't evenly spaced, so read the stamp.
4. Read the transcript whole, then answer, citing times.

Frames say what's shown, the transcript what's said, the index where the cuts, hits and words are. When the
pictures suggest a cause, look for the line that confirms it; if none does, say what you saw and that the
reason isn't stated. Pictures alone produce confident, plausible, wrong stories.

`--cells`: 64 (default) for plot and scene inventory, 36 for faces and fine action, 16 for small text.
On clips up to 3 minutes, frames.py keeps at least one frame every 0.5 s (`--max-gap`), so locked-camera
animation doesn't lose beats. Grids can't show easing or velocity; for animation quality, look at short zoomed ranges or ask the user to watch.

### Check a render before posting
```bash
python3 -I SKILL_DIR/scripts/check.py VIDEO --out-dir WORK \
  [--expect-size 1080x1920 --expect-fps 60 --expect-lufs -14 --max-peak -1 --min-silence 0.3] \
  [--expect-duration 113.5] [--expect-shots 66] [--terms terms.txt] [--script script.txt]
```
- `terms.txt`: one expected on-screen term per line (labels, numbers, stamps), with exact spelling.
- `script.txt`: the spoken words only, with no directions or speaker names.

Picture checks run on every frame, with the caption band (`--mask-band 0.46-0.56`, or `none`) ignored so word swaps don't hide a still picture:
- **frozen picture:** stretches of `--freeze-s` (default 0.5 s) or longer with no visible change.
- **glitch frames:** the picture jumps and snaps back within 2 frames.
- **jumps that aren't cuts:** information only (pop-ons, snaps). Pass `--cuts FILE` (planned cut times in seconds) to measure against the plan instead of the detected cuts.
- A 6-frame strip is saved under `WORK/strips/` for every FIX moment, plus a per-shot luma and saturation table.
- Self-check: `uv run --no-project --with numpy python -I SKILL_DIR/scripts/test_check_picture.py`.

Read `check.md`. FIX items need fixing before posting. Shot counts only see hard cuts, not camera moves inside one shot, so treat a mismatch as something to look into. Overlay-zone hits are text that the app's buttons or caption bar may cover; check the frames at those times. Then build `frames.py` grids around any FIX to confirm it by eye.

### Study reference videos
`python3 -I SKILL_DIR/scripts/study.py V1 V2 ... --out-dir WORK` writes a table with cuts per minute, median shot, sound hits per minute, silence %, words and words per minute, plus per-video cut times, hit times and the labels in the upper frame. Pair it with `frames.py` grids to see what each cut shows.

### Compare stills with reference images
`python3 -I SKILL_DIR/scripts/compare.py --stills a.png b.png --refs refs_dir --out-dir WORK`, then read
`compare.md` and look at each `board_<still>.jpg`.
- It measures brightness, contrast, saturation, warmth, shadows, highlights, fine detail and the top-6 palette. Each still gets flags like "cooler" or "less fine detail" where it falls outside the references' range.
- The numbers point at gaps; style, materials and composition still need your eye on the boards.
- Write the gaps as specific, fixable notes.

## Limits
- No continuous motion (easing, bounce), and no frame-exact cut times.
- OCR reads printed and stylized text well, but tiny or heavily outlined text can be missed. Off macOS it uses Tesseract, which finds roughly half the words Vision does and some junk; `index.md` names the engine, so weigh its text accordingly. `check.py` reports "missing", so confirm in a grid before calling a term absent.
- Nothing inaudible and off-screen is in any output.

Clean up `WORK` when the user is unlikely to follow up.

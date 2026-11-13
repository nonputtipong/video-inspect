---
name: video-inspect
description: Free, local, offline video and image inspection. Watch a video through a shot index, timestamped frame grids, fast on-screen-text OCR with screen positions and a local Whisper transcript; hear it through sound hits timed against the cuts, sound-shape labels, tempo and beats, and audio sheets (spectrogram, waveform); check a finished render before posting (specs, loudness, clipping, loudness jumps, music over the voice, sound effects off their cuts, silences, black or frozen frames, label spelling, text under TikTok/Reels/Shorts buttons, script vs speech); study reference videos (cut rhythm, sound hits, tempo, words per minute, labels) and hold a render against their profile; and compare stills against reference images with boards and measured colour, light and detail stats. Use it whenever the user wants a video or render watched, checked, QA'd, compared with references or studied for pacing, even if they don't name a tool. Never spends API credits unless the user explicitly turns paid transcription on.
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
| Hold a render against the user's reference videos | `study.py --profile p.json`, then `check.py --profile p.json` |
| Compare renders/stills with reference images | `compare.py` |
| See how sound and picture line up | `sheet.py` after `index.py` |
| One moment | `frames.py --start --end --cells 16`, `--force uniform --fps 15` for a close look (see below), or `--at 4:32,7:10` for exact instants |

### Understand a video
1. `python3 -I SKILL_DIR/scripts/transcript.py VIDEO --out WORK/transcript.json --lang en`. Start it first, in the background; it's the slow part.
2. `python3 -I SKILL_DIR/scripts/index.py VIDEO --out-dir WORK --ocr [--ocr-lang th] [--ocr-fps 8] [--min-silence 0.3] [--transcript WORK/transcript.json]`. OCR reads 4 frames a second by default; use `--ocr-fps 8` for one-word captions that change faster than that. Read `index.md` whole: one line per shot, on-screen text with its `[zone]` (top/upper/middle/lower/bottom), and the sound timeline (hits, swells and drops, silences, and speech when a transcript is given, in the order they happen).
   - Sound hits are timed to about 5-10 ms, and each shot line says how far its opening cut is from the nearest hit: `cut: hit +78 ms late`, `cut: hit -20 ms early`, or `cut: no hit within 250 ms`. `index.json` has them as `onsets` and `cut_sync`.
   - The sound timeline lists every hit with its time and a shape label: `short bright transient (click-like)`, `fast attack, low, ~300 ms (hit/impact-like)`, `slow noisy swell (whoosh-like)`, `ringing tone (ding-like)`, or plain measured words such as `fast attack, bright, tonal, ~140 ms`. The label is the shape of the sound (attack, length, brightness, tone or noise), never what made it: say "a click-like sound", not "a click". Under music, the music is part of what was measured. With `--transcript` (when the transcript is already written), hits inside a spoken word are labelled `speech`.
   - Music timing: the summary line gives the tempo ("about 120 BPM") and how many cuts fall within 50 ms of a beat, beside the share that would by chance. Shot lines add the cut's distance to the nearest beat (`60 ms after the beat`). "No clear beat" means the rhythm wasn't regular enough to trust; it is not a guess at a tempo. Half and double time read the same, so 87 can be 174. One tempo is found for the whole range, so a video with several songs often reads "no clear beat". `index.json` has `tempo` (with every beat time) and `swells`.
   - A `swell` or `drop` in the timeline is short-term loudness moving 6 LU or more over 2 s. An abrupt step shows there too, as a rise over about 3 s.
   - Under dense music a hit can be hidden, and the nearest "hit" can be a note of the music. So `no hit` means none was measured, not that none is there. Look at the frames around it, or ask the user to listen.
3. `python3 -I SKILL_DIR/scripts/frames.py VIDEO --out-dir WORK --index WORK/index.json`. Read every grid in the manifest in one message. Each cell has its time stamped top-left; cells aren't evenly spaced, so read the stamp.
4. Read the transcript whole, then answer, citing times.

Frames say what's shown, the transcript what's said, the index where the cuts, hits and words are. When the
pictures suggest a cause, look for the line that confirms it; if none does, say what you saw and that the
reason isn't stated. Pictures alone produce confident, plausible, wrong stories.

`--cells`: 64 (default) for plot and scene inventory, 36 for faces and fine action, 16 for small text.
On clips up to 3 minutes, frames.py keeps at least one frame every 0.5 s (`--max-gap`), so locked-camera
animation doesn't lose beats. Fill frames follow motion: still shots get half the base rate (up to 4 a second on
clips up to a minute, 2 up to 5 minutes, then 1), high-motion shots twice it. Grids can't show easing or velocity; for animation quality, look at short zoomed ranges or ask the user to watch.

Close look: the overview grids top out around 8 frames a second, and far less on long videos. When a moment
needs more (a high-motion shot in `index.md`, a sound hit, a caption change, a glitch from `check.py`), grid
just that stretch at a fixed rate:
`frames.py VIDEO --out-dir WORK/zoom --start 1:12 --end 1:16 --force uniform --fps 15 --no-dedup --cells 36`.
Use `--fps 30` (up to the video's own rate) for frame-by-frame, and keep the range short: the 512-frame budget
covers about 17 s at 30 fps and thins anything longer. `--at` grabs exact instants instead of a range.

Other `frames.py` options: `--budget N` caps frames for the whole run (default 512, raised to one per shot up to 2048; 0 = no cap), `--max-grids N` caps grids, and both thin frames evenly rather than cutting the end. `--force keyframe` uses the encoder's keyframes and `--force uniform` a fixed rate (`--fps`); on variable-frame-rate screen recordings uniform labels can be a few hundred ms off, so prefer the default. `--no-dedup` keeps near-duplicate frames, `--no-labels` leaves off the time stamps, `--cell-width` sets cell size (default 480 px).

### See the sound: audio sheets
`python3 -I SKILL_DIR/scripts/sheet.py VIDEO --out-dir WORK --index WORK/index.json [--window 30] [--start T --end T]`
writes `sheet_000.png`, `sheet_001.png`, ... (one per 30 s, 1568 px wide, so about 52 px per second) and `sheets.json`.
Without `--index` it builds the index itself. Read the sheets like the grids, all in one message. Rows, top to bottom:
- **time ruler:** a tick every second, MM:SS every 5 s.
- **shots:** each shot's sharpest frame; its left edge is where the shot starts. Short shots are partly covered by the next one.
- **three marker lanes:** cuts (red), hits (yellow), beats (cyan, only when the music has a clear beat). Each cut also runs as a red line down through the rows below.
- **spectrogram:** 40 Hz at the bottom to 16 kHz at the top, brighter = louder. A click is a thin full-height line, a tone a horizontal line, a whoosh a cloud.
- **waveform**, with a faint yellow line at each hit.
- **loudness:** green columns from -60 LUFS (empty) to 0 (full); blue background = silence.

Use a sheet to see how things sit together: whether hits fall on cuts, where the music swells, what a silence looks like. One pixel is about 19 ms at the default window, so read exact offsets from `index.md`, not from the picture; for a closer look use `--start`, `--end` and `--window 10`. The image has no words on it: the rows and colours above are the legend (`sheets.json` repeats them).

### Check a render before posting
```bash
python3 -I SKILL_DIR/scripts/check.py VIDEO --out-dir WORK \
  [--expect-size 1080x1920 --expect-fps 60 --expect-lufs -14 --max-peak -1 --min-silence 0.3] \
  [--expect-duration 113.5] [--expect-shots 66] [--terms terms.txt] [--script script.txt] \
  [--cuts cuts.txt] [--sync-tol 40] [--cues cues.txt] [--transcript transcript.json] [--profile p.json]
```
- `terms.txt`: one expected on-screen term per line (labels, numbers, stamps), with exact spelling.
- `script.txt`: the spoken words only, with no directions or speaker names.

Picture checks run on every frame, with the caption band (`--mask-band 0.46-0.56`, or `none`) ignored so word swaps don't hide a still picture:
- **frozen picture:** stretches of `--freeze-s` (default 0.5 s) or longer with no visible change.
- **glitch frames:** the picture jumps and snaps back within 2 frames.
- **jumps that aren't cuts:** information only (pop-ons, snaps). Pass `--cuts FILE` (planned cut times in seconds) to measure against the plan instead of the detected cuts.
- A 6-frame strip is saved under `WORK/strips/` for every FIX moment, plus a per-shot luma and saturation table.
- Self-check: `uv run --no-project --with numpy python -I SKILL_DIR/scripts/test_check_picture.py`.

Sound sync: `check.md` lists each cut's nearest sound hit and its offset in ms (+ late, - early) under "Cut sync". With `--sync-tol MS` it becomes a check: FIX when a hit is further than that from its cut. Offsets are measured against `--cuts` when given, else the detected cuts. A cut with no hit within 250 ms is listed but never a FIX, because dense music can hide a hit.

Mix checks (sound only, a few seconds):
- **clipping:** time the waveform is pinned at full scale (1% or more of the samples in a 50 ms window within 0.5 dB of it). It works on AAC and MP3 too, where a clipped wave no longer sits exactly at full scale. A mix limited with no headroom reads the same; light clipping of a few peaks can be missed.
- **loudness range (LRA):** reported beside the loudness, in LU. No pass or fail: a low number means even loudness, a high one big swings.
- **loudness jumps:** short-term loudness rising 6 LU within a second (about a 10 dB step that stays), with no cut within 0.5 s. Not judged in the first 4 s.
- **music over voice (rule of thumb):** needs word times, so it runs only with `--transcript` or `--script`, and is SKIPPED otherwise. It compares the loudness in the pauses between words (0.6 to 3 s long) with the speech; under 6 LU apart, the bed is probably too loud. A sound effect in a pause reads the same as loud music.
- **buried effects (rule of thumb):** with `--cues FILE` (planned effect times in seconds), flags the ones that rise less than 3 dB over the 300 ms before them.
- The two rules of thumb are named that way in `check.md`. Treat them as places to listen, not as verdicts.

Against references: `--profile p.json` (made by `study.py --profile`) adds an "Against your references" section: cuts per minute, share of cuts with a hit, median hit offset, loudness, loudness range, cuts on the beat, silence % and words per minute, each beside the references' range, and the moments behind a difference (`00:12.400 cut has no sound hit within 250 ms`). The range is the references' lowest to highest, or their middle half once there are 5 or more. It is never a PASS/FIX row: being outside the references is a difference to weigh, not a fault. There is no built-in profile; it is only ever the user's own picks. Words per minute needs a transcript on both sides.

Tolerances: loudness passes within `--lufs-tol` (default 1 LU) of `--expect-lufs`; a picture jump within `--cut-tol` (default 0.05 s) of a cut counts as that cut (planned cuts with `--cuts`, else detected ones). `--transcript FILE` reuses an existing transcript instead of making one, and `--lang` sets the speech and on-screen text language (default en).

Read `check.md`. FIX items need fixing before posting. SKIPPED items couldn't run on this machine (no OCR engine, or no local Whisper for the script check); tell the user they weren't checked rather than calling them passed. Shot counts only see hard cuts, not camera moves inside one shot, so treat a mismatch as something to look into. Overlay-zone hits are text that the app's buttons or caption bar may cover; check the frames at those times. Then build `frames.py` grids around any FIX to confirm it by eye.

### Study reference videos
`python3 -I SKILL_DIR/scripts/study.py V1 V2 ... --out-dir WORK` writes a table with cuts per minute, median shot, sound hits per minute, the share of cuts with a sound hit within 250 ms and the median hit offset (+ means hits land after their cuts), BPM and the share of cuts on the beat, silence %, words and words per minute, plus per-video cut times, hit times and the labels in the upper frame. Pair it with `frames.py` grids to see what each cut shows. `--profile p.json` also saves those numbers per video with their median and spread, for `check.py --profile`; study at least 3 references that share a style, since the profile is only as good as they are alike. `--no-transcript` and `--no-ocr` skip the slow parts (words then read 0, wpm `–`, labels "none found"); `--lang` sets the language (default en).

### Compare stills with reference images
`python3 -I SKILL_DIR/scripts/compare.py --stills a.png b.png --refs refs_dir --out-dir WORK`, then read
`compare.md` and look at each `board_<still>.jpg`.
- It measures brightness, contrast, saturation, warmth, shadows, highlights, fine detail and the top-6 palette. Each still gets flags like "cooler" or "less fine detail" where it falls outside the references' range.
- The numbers point at gaps; style, materials and composition still need your eye on the boards.
- Write the gaps as specific, fixable notes.

## Limits
- No continuous motion (easing, bounce). Cut times are accurate to about one frame (17 ms at 60 fps), sound hits to about 5-10 ms on a clean sound.
- Tempo works on music with a clear, steady beat and poorly on ambient beds, rubato or speech; it was tuned on generated click and drum patterns, not on a music library. Musical key is not measured.
- Shape labels come from four numbers per hit (attack, length, brightness, tonality) and simple rules, with no sound-recognition model: they can't tell a door from a drum.
- Sound hits are found by level, above 150 Hz. A hit less than about 9 dB over the music under it, or a sound that is all sub-bass, is missed.
- OCR reads printed and stylized text well, but tiny or heavily outlined text can be missed. Off macOS it uses Tesseract, which finds roughly half the words Vision does and some junk; `index.md` names the engine, so weigh its text accordingly. `check.py` reports "missing", so confirm in a grid before calling a term absent.
- "Silence" means two things: `index.md` lists pauses below -35 dB (pacing), `check.py` fails only dead air below -45 dB.
- Nothing inaudible and off-screen is in any output.

Clean up `WORK` when the user is unlikely to follow up.

# video-inspect

A [Claude Code](https://claude.com/claude-code) skill that lets Claude watch, check and study videos and images using only local tools: ffmpeg, OCR and a local Whisper transcript. It is free and offline by default, and it never spends API credits unless you turn paid transcription on.

The scripts do the measuring. Claude does the judging.

## What it does

- **Understand a video.** It builds a shot index (cuts, motion, light, silences, sound hits timed to about 5-10 ms with a shape label ("click-like", "whoosh-like") and each cut's offset to its nearest hit, tempo and beats with each cut's distance to the beat, swells and drops), on-screen text with screen positions, timestamped frame grids and a transcript, so Claude can summarize the video and cite times.
- **Check a render before posting.** It produces one PASS/FIX report (SKIPPED for checks a missing tool can't run) covering size, fps, duration, loudness (LUFS), loudness range and true peak, clipping, sudden loudness jumps, music too close to the voice, planned sound effects that are buried, silences, black, frozen or glitch frames, shot count, sound hits that land off their cuts (`--sync-tol`), label spelling, text hidden under TikTok/Reels/Shorts buttons, and script vs speech.
- **Study reference videos.** It measures cuts per minute, median shot length, sound hits and how they sit on the cuts, tempo and cuts on the beat, silence %, words per minute and an on-screen label timeline.
- **Compare stills with references.** It makes side-by-side boards plus measured brightness, contrast, saturation, warmth, shadows, highlights, fine detail and palette, and flags where a still falls outside the references' range.

## Requirements

- **Python 3.9+.** The scripts use only the standard library.
- **ffmpeg 5.1+ and ffprobe** on your `PATH` (`brew install ffmpeg`, `apt install ffmpeg`, or `winget install Gyan.FFmpeg`).
- **OCR** (optional, for on-screen text), either:
  - macOS with `swiftc` (Xcode Command Line Tools). It uses Apple Vision, which reads small and stylized text best. A small helper is compiled into `bin/` on first use.
  - or [Tesseract](https://github.com/tesseract-ocr/tesseract) on any platform (`brew install tesseract`, `apt install tesseract-ocr`, or the [UB Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki) on Windows; add its folder to `PATH`). Each language needs its data file, for example `tha.traineddata` for Thai. Tesseract also gives screen positions, so the overlay-zone checks work, but it finds fewer words than Vision.
- **Transcripts** (optional), either:
  - [`uv`](https://docs.astral.sh/uv/) with `mlx-whisper` on Apple Silicon. It runs `mlx-whisper` (`whisper-large-v3-turbo`) through `uvx --offline`. The package and model must already be cached. Run once with `--allow-network` to download them.
  - or [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) on any platform, also run through `uv` (`uv run --with faster-whisper`), so nothing is installed into your system Python. The first run needs `--allow-network` to download the package and model (`base` by default; set `WHISPER_MODEL`, for example `small`, for a different one). After that it runs offline.
  - or ElevenLabs (paid). It is off unless you pass `--source elevenlabs --allow-paid` **and** set `VIDEO_INSPECT_ALLOW_PAID=1`. The key comes from `ELEVENLABS_API_KEY` in your shell. `.env` files are never read.

It works best on macOS (Apple Silicon). On Linux and Windows, use Tesseract and faster-whisper. On Windows, run the scripts with `python` or `py -3` instead of `python3`. Every script is tested on Windows, Linux and macOS by GitHub Actions (see Usage), including file and temp-folder paths with Thai characters.

## Installation

Clone it into your Claude Code skills folder:

```bash
git clone https://github.com/nonputtipong/video-inspect.git ~/.claude/skills/video-inspect
```

On Windows (PowerShell), the skills folder is under your user profile:

```powershell
git clone https://github.com/nonputtipong/video-inspect.git "$env:USERPROFILE\.claude\skills\video-inspect"
```

Claude Code picks it up the next time a session starts. Then just ask, for example "check this render before I post it" or "watch this video and summarize it".

## Usage

Claude runs these for you, but you can also run them directly. Each script prints the path of its report on its last line.

```bash
# Understand a video
python3 -I scripts/transcript.py clip.mp4 --out work/transcript.json --lang en
python3 -I scripts/index.py clip.mp4 --out-dir work --ocr
python3 -I scripts/frames.py clip.mp4 --out-dir work --index work/index.json

# One moment, close up
python3 -I scripts/frames.py clip.mp4 --out-dir work --at 4:32,7:10 --cells 16

# Check a vertical render before posting
python3 -I scripts/check.py render.mp4 --out-dir work \
  --expect-size 1080x1920 --expect-fps 60 --expect-lufs -14 --max-peak -1 \
  --terms terms.txt --script script.txt

# Study reference videos
python3 -I scripts/study.py ref1.mp4 ref2.mp4 --out-dir work

# Compare renders with reference images
python3 -I scripts/compare.py --stills a.png b.png --refs refs/ --out-dir work
```

See [`SKILL.md`](SKILL.md) for every option and how Claude is told to read the output.

Self-tests for the picture checks, for how frames.py spreads frames by motion, and for the sound measures:

```bash
uv run --no-project --with numpy python -I scripts/test_check_picture.py
```

```bash
python3 -I scripts/test_frames_plan.py
```

```bash
python3 -I scripts/test_audio.py
```

End-to-end smoke test (every script on a generated clip; set `VI_SMOKE_TRANSCRIPT=1` to include faster-whisper). GitHub Actions runs it on Windows, Linux and macOS with Python 3.9 and 3.13:

```bash
python3 -I tests/smoke_test.py
```

## Project structure

```
SKILL.md                      Instructions Claude follows when the skill is used
scripts/
  common.py                   Shared helpers: fenced temp folders, OCR backends, ffprobe
  index.py                    Shot index: cuts, motion, light, sound hits, silences, OCR text
  audio.py                    Sound measures: hit onsets every 5 ms and their shape labels,
                              hit-to-cut offsets, tempo and beats, swells, mix checks
  frames.py                   Picks the frames worth seeing and tiles them into timestamped grids
  transcript.py               Local Whisper transcript (paid ElevenLabs only when unlocked)
  check.py                    Pre-posting QA report (PASS/FIX)
  study.py                    Pacing and label stats across reference videos
  compare.py                  Stills vs reference images: boards and look stats
  ocr.swift                   macOS Vision OCR helper (compiled to bin/ on first use)
  test_check_picture.py       Self-test for check.py's frozen and glitch frame detection
  test_frames_plan.py         Self-test for frames.py's motion-weighted fill frames
  test_audio.py               Self-test for audio.py on generated sounds with known answers
tests/
  smoke_test.py               End-to-end test of every script (run by GitHub Actions)
  fixtures/text.png           Printed text the smoke test overlays for OCR
.github/workflows/test.yml    CI on Windows, Linux and macOS
```

## Safety defaults

- No paid calls and no network by default (see Requirements).
- Scripts only delete their own `vi_*` folders inside the system temp directory.

## License

[MIT](LICENSE)

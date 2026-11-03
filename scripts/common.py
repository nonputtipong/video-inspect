"""Shared helpers for video-inspect: safe temp folders, local OCR, small media utils.

Safety rules kept in one place:
  - Temp folders are created by `make_tmp()` (prefix "vi_", inside the system temp dir),
    and `safe_rmtree()` refuses to delete anything that isn't one of those.
  - No .env files are read. No network calls happen in this module.
"""
from __future__ import annotations

import functools
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIN = HERE.parent / "bin"          # compiled OCR helper lives inside the skill folder
TMP_PREFIX = "vi_"

# Windows pipes default to the ANSI code page, and `python -I` ignores PYTHONUTF8, so Thai
# text or a non-ASCII file name would crash a print. Every script imports this module first.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def need(tool: str) -> None:
    if shutil.which(tool) is None:
        sys.exit(f"{tool} not found on PATH. Install ffmpeg (brew install ffmpeg).")


def make_tmp(tag: str) -> Path:
    return Path(tempfile.mkdtemp(prefix=f"{TMP_PREFIX}{tag}_"))


def safe_rmtree(p: Path | str) -> None:
    """Delete only folders that live under the system temp dir inside a vi_* folder."""
    p = Path(p).resolve()
    base = Path(tempfile.gettempdir()).resolve()
    try:
        rel = p.relative_to(base)
    except ValueError:
        print(f"[video-inspect] refused to delete {p}: not under {base}", file=sys.stderr)
        return
    if not rel.parts or not any(part.startswith(TMP_PREFIX) for part in rel.parts):
        print(f"[video-inspect] refused to delete {p}: not a video-inspect temp folder", file=sys.stderr)
        return
    shutil.rmtree(p, ignore_errors=True)


def fmt_ts(sec: float) -> str:
    sec = max(0.0, sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:04.1f}" if h >= 1 else f"{int(m):02d}:{s:04.1f}"


def clock(ms: int) -> str:
    """Grid and index label: MM:SS.t (H:MM:SS.t past an hour), tenths cut, not rounded."""
    sec, rest = divmod(int(ms), 1000)
    h, sec = divmod(sec, 3600)
    m, sec = divmod(sec, 60)
    return (f"{h}:" if h else "") + f"{m:02d}:{sec:02d}.{rest // 100}"


def parse_time(text: str | None) -> float | None:
    """SS, MM:SS or HH:MM:SS (seconds may have decimals) -> seconds."""
    if text is None or not str(text).strip():
        return None
    total = 0.0
    try:
        parts = [float(p) for p in str(text).strip().split(":")]
    except ValueError:
        parts = []
    if not 1 <= len(parts) <= 3:
        sys.exit(f"cannot parse time {text!r} (want SS, MM:SS or HH:MM:SS)")
    for p in parts:
        total = total * 60 + p
    return total


def probe(video: str) -> dict:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "stream=codec_type,width,height,r_frame_rate,duration:format=duration,start_time",
                        "-of", "json", video], capture_output=True, text=True, encoding="utf-8", errors="replace")
    d = json.loads(r.stdout or "{}")
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    num, _, den = (v.get("r_frame_rate") or "0/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 0.0
    return {"duration_s": float(d.get("format", {}).get("duration") or v.get("duration") or 0),
            "width": v.get("width", 0), "height": v.get("height", 0), "fps": round(fps, 3),
            # file start timestamp (usually 0): `-ss T` means start_s + T on the frames' own clock
            "start_s": float(d.get("format", {}).get("start_time") or 0),
            "has_audio": any(s.get("codec_type") == "audio" for s in d.get("streams", []))}


# ---------- seeking ----------

# Containers with no seek index. ffmpeg's -ss lands near the target, but decoding can only
# start at the next keyframe, so everything before it is lost: on a .ts with 2 s between
# keyframes, frames came out up to 1.6 s late and the last ones were missing. For these,
# seek one keyframe gap early and trim the lead-in exactly. Other formats seek exactly.
LOOSE_SEEK = {"mpegts", "mpeg", "mpegvideo", "h264", "hevc"}


@functools.lru_cache(maxsize=None)
def seek_margin(video: str) -> float:
    """Seconds to start early so a seek lands before the target: 0 for indexed formats, else
    the longest keyframe gap in the first 2 minutes (read from packets, no decoding) + 0.5 s."""
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=format_name", "-of", "csv=p=0", video],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if not set(r.stdout.strip().split(",")) & LOOSE_SEEK:
        return 0.0
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-read_intervals", "%+120",
                        "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", video],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    keys = []
    for row in r.stdout.splitlines():
        t, _, flags = row.partition(",")
        if "K" in flags and t not in ("", "N/A"):
            keys.append(float(t))
    keys.sort()
    gap = max((b - a for a, b in zip(keys, keys[1:])), default=10.0)
    return round(gap + 0.5, 3)


def seek_input(video: str, start: float | None, end: float | None) -> tuple[list[str], tuple]:
    """Input -ss/-to for a range, plus what trim() must cut so the range is exact.

    On loose-seek files -to overshoots by as much as the seek missed, so reading runs one
    margin past the end and trim() cuts the end too. Elsewhere the args are the plain
    -ss/-to and trim() is empty."""
    margin = seek_margin(video)
    lead = min(margin, start) if start is not None else 0.0
    args = ["-ss", f"{start - lead:.3f}"] if start is not None else []
    if end is not None:
        args += ["-to", f"{end + margin:.3f}"]
    length = end - (start or 0.0) if (end is not None and margin) else None
    return args, (lead, length)


def trim(cut: tuple, audio: bool = False) -> str:
    """Filter prefix for seek_input()'s cut: drop the lead-in, stop at the range end, and put
    the clock back on the range start. Empty when nothing needs cutting."""
    lead, length = cut
    if not lead and length is None:
        return ""
    a = "a" if audio else ""
    span = f"start={lead:.3f}" + (f":end={lead + length:.3f}" if length is not None else "")
    return f"{a}trim={span},{a}setpts=PTS-{lead:.3f}/TB,"


# ---------- OCR (macOS Vision, or tesseract elsewhere) ----------

# --ocr-lang code -> (Apple Vision code, Tesseract traineddata name). Unknown codes pass through.
OCR_LANGS = {
    "en": ("en-US", "eng"), "th": ("th-TH", "tha"), "ja": ("ja-JP", "jpn"), "ko": ("ko-KR", "kor"),
    "zh": ("zh-Hans", "chi_sim"), "zh-tw": ("zh-Hant", "chi_tra"), "vi": ("vi-VN", "vie"),
    "id": ("id-ID", "ind"), "ms": ("ms-MY", "msa"), "ar": ("ar-SA", "ara"), "ru": ("ru-RU", "rus"),
    "hi": ("hi-IN", "hin"),
}


def ocr_codes(langs: str, engine: int) -> list[str]:
    """Translate "th,en" into one engine's codes (0 = Vision, 1 = Tesseract)."""
    return [OCR_LANGS.get(x.strip().lower(), (x.strip(), x.strip()))[engine] for x in langs.split(",") if x.strip()]


def vision_binary() -> Path | None:
    """Compile ocr.swift into the skill's own bin/ once; rebuild when the source changes."""
    if platform.system() != "Darwin" or shutil.which("swiftc") is None:
        return None
    src = HERE / "ocr.swift"
    binary = BIN / "ocr"
    if binary.exists() and binary.stat().st_mtime >= src.stat().st_mtime:
        return binary
    BIN.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["swiftc", "-O", str(src), "-o", str(binary)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"[ocr] swiftc failed: {r.stderr[-300:]}", file=sys.stderr)
        return None
    return binary


def ocr_backend() -> tuple[str | None, Path | None]:
    b = vision_binary()
    if b:
        return "vision", b
    return ("tesseract", None) if shutil.which("tesseract") else (None, None)


def ocr_images(images: list[Path], langs: str = "en") -> dict[str, list[dict]]:
    """Return {path: [{"text", "conf", "box": [x, y, w, h]}]} with boxes 0-1, top-left origin.
    Tesseract has no boxes here, so its items get box None."""
    backend, binary = ocr_backend()
    out: dict[str, list[dict]] = {}
    if backend == "vision":
        env = {**os.environ, "OCR_LANGS": ",".join(ocr_codes(langs, 0))}
        r = subprocess.run([str(binary), *map(str, images)], env=env, capture_output=True,
                           text=True, encoding="utf-8", errors="replace")
        for line in r.stdout.splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[row["path"]] = row.get("items", [])
    elif backend == "tesseract":
        out = tesseract_lines(images, "+".join(ocr_codes(langs, 1)))
    return out


# Word confidence (0-100) below this is mostly texture read as letters. Measured on 7 video
# frames against Vision: at 50, 78% of kept words are real (30: 70%), recall barely drops.
TESS_MIN_CONF = 50
# Tesseract reads small screen text far better from upscaled frames: 2x found 45% of
# Vision's words vs 19% at 1x (3x was no better and slower). index.py extracts at this scale.
TESS_UPSCALE = 2


def _tight(ch: str) -> bool:
    """Thai or CJK: scripts written without spaces between words."""
    return "\u0e00" <= ch <= "\u0e7f" or ch >= "\u3000"


def _no_space(a: str, b: str) -> bool:
    """Thai and CJK write words without spaces; Tesseract still splits them into words."""
    return bool(a and b) and _tight(a[-1]) and _tight(b[0])


# A line with no word of 3+ letters or digits is mostly texture read as letters ("pe", "oS",
# "Fr") unless Tesseract is very sure of it. Numbers count as one word with their
# punctuation ("11.3s", "18:40", "kb/s"), and lines with 2+ digits are always kept, since
# times, sizes and counts are what checks look for. Measured on 153 frames against Vision:
# junk lines 277 -> 156 (much of the rest is real text Vision skips as too small), 99% of
# real lines kept, no line with a number lost. Thai/CJK lines are kept: no spaces between words.
SHORT_LINE_MIN_CONF = 0.9
WORD_RE = re.compile(r"[^\W_]+(?:[.,:/][^\W_]+)*")


def _looks_like_text(text: str, conf: float) -> bool:
    if conf >= SHORT_LINE_MIN_CONF or any(_tight(ch) for ch in text):
        return True
    if sum(ch.isdigit() for ch in text) >= 2:
        return True
    return max(map(len, WORD_RE.findall(text)), default=0) >= 3


def tesseract_lines(images: list[Path], lang: str) -> dict[str, list[dict]]:
    """Tesseract in one process per batch, read as TSV so every line keeps its box.

    The batch is a text file listing the images; TSV numbers them by page_num. Page
    segmentation 11 (sparse text) suits video frames: captions and labels scattered
    over a picture, not a page of paragraphs. Boxes come back 0-1, top-left origin.
    """
    if not images:
        return {}
    listing = images[0].parent / f"_tess_{os.getpid()}_{id(images)}.txt"
    listing.write_text("".join(f"{p.resolve()}\n" for p in images), encoding="utf-8")
    env = {**os.environ, "OMP_THREAD_LIMIT": "1"}   # callers already run batches in parallel
    r = subprocess.run(["tesseract", str(listing), "stdout", "-l", lang, "--psm", "11", "tsv"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    listing.unlink()
    pages: dict[int, tuple[int, int]] = {}
    lines: dict[tuple, list[tuple]] = {}
    for row in r.stdout.splitlines()[1:]:
        f = row.split("\t")
        if len(f) < 12:
            continue
        level, page = int(f[0]), int(f[1]) - 1
        left, top, w, h, conf = int(f[6]), int(f[7]), int(f[8]), int(f[9]), float(f[10])
        if level == 1:
            pages[page] = (w, h)
        elif level == 5 and f[11].strip() and conf >= TESS_MIN_CONF:
            lines.setdefault((page, f[2], f[3], f[4]), []).append((f[11].strip(), conf, left, top, w, h))
    out: dict[str, list[dict]] = {str(p): [] for p in images}
    for (page, *_), words in lines.items():
        if page >= len(images) or page not in pages:
            continue
        pw, ph = pages[page]
        text = ""
        for word, *_ in words:
            text += word if not text or _no_space(text, word) else " " + word
        x0 = min(w[2] for w in words)
        y0 = min(w[3] for w in words)
        x1 = max(w[2] + w[4] for w in words)
        y1 = max(w[3] + w[5] for w in words)
        conf = round(sum(w[1] for w in words) / len(words) / 100, 3)
        if not _looks_like_text(text, conf):
            continue
        out[str(images[page])].append({
            "text": text,
            "conf": conf,
            "box": [round(x0 / pw, 3), round(y0 / ph, 3), round((x1 - x0) / pw, 3), round((y1 - y0) / ph, 3)],
        })
    return out


def is_noise(text: str) -> bool:
    """One stray character (a shirt logo, a pixel blob) isn't on-screen text worth reporting."""
    core = "".join(ch for ch in text if ch.isalnum())
    return len(core) < 2


def gray_thumbs(paths: list[Path], size: int = 32) -> list[bytes]:
    """size x size grayscale thumbnails for images in ONE folder, decoded by a single ffmpeg.

    The images are fed through a concat list of bare file names, run from that folder, so
    gaps in numbering and Windows drive letters don't matter. Returns [] if the count is off.
    """
    if not paths:
        return []
    folder = paths[0].parent
    listing = folder / "_thumbs.txt"
    listing.write_text("".join(f"file '{p.name}'\n" for p in paths), encoding="utf-8")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "concat", "-safe", "0",
                        "-i", listing.name, "-vf", f"scale={size}:{size},format=gray",
                        "-f", "rawvideo", "-"], capture_output=True, cwd=folder)
    listing.unlink()
    n = size * size
    if r.returncode or len(r.stdout) != n * len(paths):
        return []
    return [r.stdout[i * n:(i + 1) * n] for i in range(len(paths))]


def region_delta(a: bytes, b: bytes, size: int = 32, tile: int = 4) -> float:
    """Largest mean abs difference over tile x tile blocks: a small character moving in front
    of a locked camera changes one region a lot and the frame average hardly at all."""
    if not a or not b:
        return 1e9
    worst = 0.0
    for ty in range(0, size, tile):
        for tx in range(0, size, tile):
            s = 0
            for y in range(ty, ty + tile):
                row = y * size
                for x in range(tx, tx + tile):
                    s += abs(a[row + x] - b[row + x])
            worst = max(worst, s / (tile * tile))
    return worst

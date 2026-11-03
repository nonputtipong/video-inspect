#!/usr/bin/env python3
"""Timestamped transcript, free and local by default.

Sources (in order, for --source auto):
  mlx      mlx-whisper via `uvx --offline` (Apple Silicon). Uses only the cached
           package and model, so no network. If either is missing it says so and stops.
  faster   faster-whisper (any OS), run through `uv run --offline --with faster-whisper`,
           so nothing is installed into the system Python. Offline too: the package and
           model (WHISPER_MODEL, default "base") must be cached. Tried first off Apple Silicon.
  elevenlabs  PAID. Never tried on auto. Runs only with --source elevenlabs AND
           --allow-paid AND VIDEO_INSPECT_ALLOW_PAID=1 in the environment, so a single
           slip can't spend credits. The key is read from the shell environment only
           (no .env files).

--allow-network lets mlx-whisper or faster-whisper download its package and model the
first time on a new machine. Ask the user before passing it.

Usage:
  python3 transcript.py VIDEO --out DIR/transcript.json [--lang en] [--source auto]

Output JSON: {source, language, duration_s, segments:[{t_ms,end_ms,t_label,text}],
              words:[{text,start_s,end_s}]}
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

MLX_MODEL = "mlx-community/whisper-large-v3-turbo"


def via_mlx(video: str, lang: str | None, tmp: Path, allow_network: bool) -> dict:
    if shutil.which("uvx") is None:
        raise RuntimeError("uvx not found (install uv)")
    cmd = ["uvx"] + ([] if allow_network else ["--offline"]) + [
        "--from", "mlx-whisper", "mlx_whisper", video, "--model", MLX_MODEL,
        "--word-timestamps", "True", "--output-format", "json", "--output-dir", str(tmp)]
    if lang:
        cmd += ["--language", lang]
    env = dict(os.environ)
    if not allow_network:
        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    files = list(tmp.glob("*.json"))
    if r.returncode != 0 or not files:
        hint = "" if allow_network else " (offline: package or model not cached; ask the user before --allow-network)"
        raise RuntimeError(f"mlx-whisper failed{hint}: {r.stderr.strip()[-300:]}")
    d = json.loads(files[0].read_text(encoding="utf-8"))
    words = [{"text": w["word"].strip(), "start_s": round(w["start"], 3), "end_s": round(w["end"], 3)}
             for s in d.get("segments", []) for w in s.get("words", [])]
    segs = [{"start": s["start"], "end": s["end"], "text": s["text"].strip()} for s in d.get("segments", [])]
    return {"language": d.get("language"), "segments": segs, "words": words}


# Audio is decoded by ffmpeg into 16 kHz mono floats, the input faster-whisper expects, instead
# of by its own PyAV reader: faster-whisper 1.2.1 passes an argument PyAV 19 no longer accepts.
FASTER_RUN = """
import json, subprocess, sys
import numpy as np
from faster_whisper import WhisperModel
video, lang, name, local = sys.argv[1], sys.argv[2] or None, sys.argv[3], sys.argv[4] == "1"
pcm = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video, "-vn", "-ac", "1",
                      "-ar", "16000", "-f", "f32le", "-"], capture_output=True, check=True).stdout
model = WhisperModel(name, local_files_only=local)
parts, info = model.transcribe(np.frombuffer(pcm, np.float32), language=lang, word_timestamps=True)
segs, words = [], []
for s in parts:
    segs.append({"start": s.start, "end": s.end, "text": s.text.strip()})
    words += [{"text": w.word.strip(), "start_s": round(w.start, 3), "end_s": round(w.end, 3)} for w in s.words or []]
print(json.dumps({"language": info.language, "segments": segs, "words": words}))
"""


def via_faster(video: str, lang: str | None, allow_network: bool) -> dict:
    """faster-whisper in a uv-managed environment. Installing it into the system Python is
    blocked on Homebrew and recent Debian/Ubuntu, and these scripts run with `python -I`,
    which can't see a venv or user site-packages anyway."""
    if shutil.which("uv") is None:
        raise RuntimeError("uv not found (install uv)")
    cmd = ["uv", "run"] + ([] if allow_network else ["--offline"]) + [
        "--no-project", "--with", "faster-whisper", "python", "-I", "-c", FASTER_RUN,
        video, lang or "", os.environ.get("WHISPER_MODEL", "base"), "0" if allow_network else "1"]
    env = dict(os.environ)
    if not allow_network:
        env.update(HF_HUB_OFFLINE="1")
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    lines = r.stdout.strip().splitlines()
    if r.returncode != 0 or not lines:
        hint = "" if allow_network else " (offline: package or model not cached; ask the user before --allow-network)"
        raise RuntimeError(f"faster-whisper failed{hint}: {r.stderr.strip()[-300:]}")
    return json.loads(lines[-1])


def via_elevenlabs(video: str, lang: str | None, tmp: Path) -> dict:
    import mimetypes
    import urllib.request
    import uuid
    key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
    if not key:
        raise RuntimeError("ELEVENLABS_API_KEY not set in the shell environment")
    audio = tmp / "audio.mp3"
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", video, "-vn",
                    "-ac", "1", "-ar", "16000", "-b:a", "64k", str(audio)], check=True)
    boundary = f"----vi{uuid.uuid4().hex}"
    fields = {"model_id": "scribe_v2", "timestamps_granularity": "word"}
    if lang:
        fields["language_code"] = lang
    pre = b"".join(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
                   for k, v in fields.items())
    pre += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="audio.mp3"\r\n'
            f"Content-Type: {mimetypes.guess_type('a.mp3')[0]}\r\n\r\n").encode()
    body = pre + audio.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    req = urllib.request.Request("https://api.elevenlabs.io/v1/speech-to-text", data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("xi-api-key", key)
    with urllib.request.urlopen(req, timeout=900) as resp:
        d = json.loads(resp.read())
    words = [{"text": w["text"], "start_s": w["start"], "end_s": w["end"]}
             for w in d.get("words", []) if w.get("type", "word") == "word"]
    return {"language": d.get("language_code"), "segments": group(words), "words": words}


def group(words: list[dict], max_gap: float = 0.7, max_chars: int = 90) -> list[dict]:
    out, cur = [], []
    for w in words:
        if cur and (w["start_s"] - cur[-1]["end_s"] > max_gap or sum(len(x["text"]) + 1 for x in cur) > max_chars):
            out.append({"start": cur[0]["start_s"], "end": cur[-1]["end_s"], "text": " ".join(x["text"] for x in cur)})
            cur = []
        cur.append(w)
    if cur:
        out.append({"start": cur[0]["start_s"], "end": cur[-1]["end_s"], "text": " ".join(x["text"] for x in cur)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Local-first transcript (no paid calls by default).")
    ap.add_argument("video")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lang")
    ap.add_argument("--source", default="auto", choices=["auto", "mlx", "faster", "elevenlabs"])
    ap.add_argument("--allow-network", action="store_true",
                    help="Let mlx-whisper or faster-whisper download its package and model on first use. Ask the user first.")
    ap.add_argument("--allow-paid", action="store_true",
                    help="Needed (with VIDEO_INSPECT_ALLOW_PAID=1) for --source elevenlabs.")
    args = ap.parse_args()
    common.need("ffmpeg")
    video = str(Path(args.video).expanduser().resolve())
    if not Path(video).exists():
        sys.exit(f"no such file: {video}")
    if args.source == "elevenlabs" and not (args.allow_paid and os.environ.get("VIDEO_INSPECT_ALLOW_PAID") == "1"):
        sys.exit("[transcript] ElevenLabs is paid and off. It needs --allow-paid and "
                 "VIDEO_INSPECT_ALLOW_PAID=1, set only after the user approves this run.")
    apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"
    order = [args.source] if args.source != "auto" else (["mlx", "faster"] if apple_silicon else ["faster"])
    tmp = common.make_tmp("tx")
    errors, result, used = [], None, None
    try:
        for src in order:
            try:
                result = (via_mlx(video, args.lang, tmp, args.allow_network) if src == "mlx"
                          else via_faster(video, args.lang, args.allow_network) if src == "faster"
                          else via_elevenlabs(video, args.lang, tmp))
                used = src
                break
            except Exception as e:  # report and try the next free source
                errors.append(f"{src}: {e}")
    finally:
        common.safe_rmtree(tmp)
    if not result:
        print("[transcript] no source worked:\n  - " + "\n  - ".join(errors), file=sys.stderr)
        sys.exit(2)
    segs = [{"t_ms": int(s["start"] * 1000), "end_ms": int(s["end"] * 1000),
             "t_label": common.fmt_ts(s["start"]), "text": s["text"]} for s in result["segments"]]
    out = {"source": used, "language": result["language"],
           "duration_s": common.probe(video)["duration_s"], "segments": segs, "words": result["words"]}
    p = Path(args.out).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[transcript] {used}: {len(segs)} segments, {len(result['words'])} words", file=sys.stderr)
    print(f"TRANSCRIPT: {p.resolve()}")


if __name__ == "__main__":
    main()

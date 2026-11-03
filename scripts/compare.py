#!/usr/bin/env python3
"""Compare stills (renders, frames) against reference images: boards + measured look stats.

Numbers are signals for a visual judgement, not a verdict: they make gaps like
"cooler than the references" or "less fine detail" objective, then you look.

Usage:
  python3 compare.py --stills a.png b.png ... --refs refdir_or_files ... --out-dir DIR

Output:
  DIR/compare.md         stats table (refs mean ± spread, each still, flags)
  DIR/refs_sheet.jpg     all references in one sheet
  DIR/board_<still>.jpg  each still beside the reference sheet
  DIR/palettes.png       top-6 colour swatches per image (refs first)
  prints `COMPARE: <path to compare.md>`
"""
from __future__ import annotations

import argparse
import colorsys
import json
import math
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
SW, SH = 96, 170          # stats sample size (9:16-ish; aspect differences don't matter for stats)
DW, DH = 128, 228         # detail (edge) sample size


def raw(path: Path, w: int, h: int, fmt: str) -> bytes:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path),
                        "-vf", f"scale={w}:{h},format={fmt}", "-frames:v", "1", "-f", "rawvideo", "-"],
                       capture_output=True)
    return r.stdout


def kmeans(px: list[tuple[int, int, int]], k: int = 6, iters: int = 10) -> list[tuple[tuple[int, int, int], float]]:
    pts = px[::3]
    step = max(1, len(pts) // k)
    cent = [pts[i * step] for i in range(k)]
    for _ in range(iters):
        groups = [[] for _ in range(k)]
        for p in pts:
            j = min(range(k), key=lambda c: (p[0] - cent[c][0]) ** 2 + (p[1] - cent[c][1]) ** 2 + (p[2] - cent[c][2]) ** 2)
            groups[j].append(p)
        cent = [tuple(sum(c[i] for c in g) // len(g) for i in range(3)) if g else cent[n]
                for n, g in enumerate(groups)]
    sizes = [len(g) / len(pts) for g in groups]
    return sorted(zip(cent, sizes), key=lambda cs: -cs[1])


def stats(path: Path) -> dict:
    rgb = raw(path, SW, SH, "rgb24")
    px = [(rgb[i], rgb[i + 1], rgb[i + 2]) for i in range(0, len(rgb) - 2, 3)]
    lum = [0.2126 * r + 0.7152 * g + 0.0722 * b for r, g, b in px]
    n = len(px)
    mean_l = sum(lum) / n
    sat = sum(colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)[1] for r, g, b in px) / n
    warmth = sum(r - b for r, _, b in px) / n
    g = raw(path, DW, DH, "gray")
    grad = sum(abs(g[y * DW + x] - g[y * DW + x + 1]) + abs(g[y * DW + x] - g[(y + 1) * DW + x])
               for y in range(DH - 1) for x in range(DW - 1)) / ((DW - 1) * (DH - 1))
    return {"brightness": round(mean_l, 1),
            "contrast": round(math.sqrt(sum((l - mean_l) ** 2 for l in lum) / n), 1),
            "saturation": round(sat * 100, 1),
            "warmth": round(warmth, 1),
            "shadows_pct": round(100 * sum(l < 50 for l in lum) / n, 1),
            "highlights_pct": round(100 * sum(l > 215 for l in lum) / n, 1),
            "detail": round(grad, 1),
            "palette": [("#%02x%02x%02x" % c, round(s * 100)) for c, s in kmeans(px)]}


# How far a still may sit from the reference mean before it's flagged (in addition to 1.5x the refs' spread).
TOL = {"brightness": 12, "contrast": 8, "saturation": 6, "warmth": 8,
       "shadows_pct": 6, "highlights_pct": 6, "detail": 2.0}
WORDS = {"brightness": ("darker", "brighter"), "contrast": ("flatter", "punchier"),
         "saturation": ("duller", "more saturated"), "warmth": ("cooler", "warmer"),
         "shadows_pct": ("fewer deep shadows", "more deep shadows"),
         "highlights_pct": ("fewer highlights", "more blown highlights"),
         "detail": ("less fine detail", "busier detail")}


def gather(items: list[str]) -> list[Path]:
    out: list[Path] = []
    for it in items:
        p = Path(it).expanduser()
        out += sorted(q for q in p.iterdir() if q.suffix.lower() in IMG_EXT) if p.is_dir() else [p]
    return [p for p in out if p.exists()]


def ffmpeg(*args: str) -> None:
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Stills vs reference images: boards + look stats.")
    ap.add_argument("--stills", nargs="+", required=True)
    ap.add_argument("--refs", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    common.need("ffmpeg")
    stills, refs = gather(args.stills), gather(args.refs)
    if not stills or not refs:
        sys.exit("need at least one still and one reference image")
    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    rs = {str(p): stats(p) for p in refs}
    ss = {str(p): stats(p) for p in stills}
    keys = list(TOL)
    agg = {}
    for k in keys:
        vals = [v[k] for v in rs.values()]
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals))
        agg[k] = (round(m, 1), round(sd, 1), min(vals), max(vals))

    # Boards: reference sheet (3 columns of 360x640), then each still beside it at the same height.
    cols = 3
    rows = math.ceil(len(refs) / cols)
    inputs = sum((["-i", str(p)] for p in refs), [])
    cells = "".join(f"[{i}:v]scale=360:640:force_original_aspect_ratio=increase,crop=360:640,setsar=1[r{i}];"
                    for i in range(len(refs)))
    pad = cols * rows - len(refs)
    cells += "".join(f"color=c=0x202020:s=360x640:d=1[p{i}];" for i in range(pad))
    labels = "".join(f"[r{i}]" for i in range(len(refs))) + "".join(f"[p{i}]" for i in range(pad))
    layout = "|".join(f"{(i % cols) * 360}_{(i // cols) * 640}" for i in range(cols * rows))
    sheet = out / "refs_sheet.jpg"
    ffmpeg(*inputs, "-filter_complex", f"{cells}{labels}xstack=inputs={cols * rows}:layout={layout}[o]",
           "-map", "[o]", "-frames:v", "1", "-q:v", "3", str(sheet))
    H = 640 * rows
    for p in stills:
        ffmpeg("-i", str(p), "-i", str(sheet), "-filter_complex",
               f"[0:v]scale=-2:{H},setsar=1[a];[1:v]setsar=1[b];[a][b]hstack=inputs=2[o]",
               "-map", "[o]", "-frames:v", "1", "-q:v", "3", str(out / f"board_{p.stem}.jpg"))

    # Palette strip: one row per image, 6 swatches each (written as a PPM, converted to PNG).
    rowsp = [("ref", p, rs[str(p)]) for p in refs] + [("still", p, ss[str(p)]) for p in stills]
    w, h = 6 * 80, 40 * len(rowsp)
    buf = bytearray()
    for _, _, st in rowsp:
        line = bytearray()
        for hexc, _ in st["palette"][:6]:
            c = bytes.fromhex(hexc[1:])
            line += c * 80
        line += b"\x00\x00\x00" * (w - len(line) // 3)
        buf += bytes(line) * 40
    ppm = out / "palettes.ppm"
    ppm.write_bytes(f"P6 {w} {h} 255\n".encode() + bytes(buf))
    ffmpeg("-i", str(ppm), str(out / "palettes.png"))
    ppm.unlink()

    md = ["# Stills vs references", "",
          f"References: {len(refs)} ({', '.join(p.name for p in refs)}). "
          "Flags show where a still sits outside the references' range (mean ± max(1.5×spread, tolerance)).", "",
          "| metric | refs mean ± spread (min–max) | " + " | ".join(p.name for p in stills) + " |",
          "|---|---|" + "---|" * len(stills)]
    flags: dict[str, list[str]] = {str(p): [] for p in stills}
    for k in keys:
        m, sd, lo, hi = agg[k]
        cells_ = []
        for p in stills:
            v = ss[str(p)][k]
            lim = max(1.5 * sd, TOL[k])
            mark = ""
            if v < m - lim:
                mark = " ▼"
                flags[str(p)].append(f"{WORDS[k][0]} ({k} {v} vs {m})")
            elif v > m + lim:
                mark = " ▲"
                flags[str(p)].append(f"{WORDS[k][1]} ({k} {v} vs {m})")
            cells_.append(f"{v}{mark}")
        md.append(f"| {k} | {m} ± {sd} ({lo}–{hi}) | " + " | ".join(cells_) + " |")
    md += ["", "## Flags per still", ""]
    for p in stills:
        md.append(f"- **{p.name}:** " + ("; ".join(flags[str(p)]) if flags[str(p)] else "inside the reference range on every metric"))
    md += ["", "## Palettes (top 6 colours, % of image)", ""]
    for kind, p, st in rowsp:
        md.append(f"- {kind} **{p.name}:** " + ", ".join(f"{c} {s}%" for c, s in st["palette"]))
    md += ["", "Boards: `board_<still>.jpg` (still left, references right). Palette rows in `palettes.png` follow the list above."]
    (out / "compare.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out / "compare.json").write_text(json.dumps({"refs": rs, "stills": ss, "refs_agg": agg}, indent=1), encoding="utf-8")
    print(f"COMPARE: {(out / 'compare.md').resolve()}")


if __name__ == "__main__":
    main()

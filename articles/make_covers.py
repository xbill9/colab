#!/usr/bin/env python3
"""Textless covers for the two Colab articles, in the publishing kit's palette.

gemma4-colab-tpu: a TPU die with a chat bubble rising from it -- a model you can talk to, on a chip.
colab-mcp:        a plug and cable running into a notebook page inside a cloud -- an agent plugged into
                  hosted notebooks.

Drawn at 2x and downsampled; written at dev.to's displayed ratio (1376x578) and Builder Center's
(1200x675), each named by a hash of its bytes so a regenerated cover is a new URL.

    python3 articles/make_covers.py
"""

from __future__ import annotations

import hashlib
import io
import pathlib

from PIL import Image, ImageDraw, ImageFilter

HERE = pathlib.Path(__file__).resolve().parent
SURFACE = (26, 26, 25)
LIFT = (34, 34, 33)
RULE = (58, 58, 55)
BLUE = (57, 135, 229)
ORANGE = (217, 89, 38)
PAPER = (236, 233, 224)
SIZES = {"devto": (1376, 578), "builder": (1200, 675)}
S = 2  # supersampling


def canvas(w, h):
    img = Image.new("RGB", (w * S, h * S), SURFACE)
    d = ImageDraw.Draw(img)
    step = 48 * S  # faint grid, so the surface reads as a board rather than a void
    for x in range(0, w * S, step):
        d.line([(x, 0), (x, h * S)], fill=(31, 31, 30), width=S)
    for y in range(0, h * S, step):
        d.line([(0, y), (w * S, y)], fill=(31, 31, 30), width=S)
    return img, d


def glow(img, box, colour, radius):
    layer = Image.new("RGB", img.size, (0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle(box, radius=radius, fill=colour)
    layer = layer.filter(ImageFilter.GaussianBlur(60 * S))
    return Image.blend(img, Image.composite(layer, img, layer.convert("L")), 0.35)


def tpu_cover(w, h):
    img, d = canvas(w, h)
    u = min(w, h) / 578 * S  # one unit at dev.to scale
    cx, cy = w * S * 0.40, h * S * 0.58
    half = 150 * u
    die = (cx - half, cy - half, cx + half, cy + half)
    img = glow(img, die, BLUE, 30 * u)
    d = ImageDraw.Draw(img)
    # pins on four sides
    for i in range(9):
        t = -half + (i + 0.5) * (2 * half / 9)
        for x0, y0, x1, y1 in (
            (cx + t - 6 * u, cy - half - 34 * u, cx + t + 6 * u, cy - half),
            (cx + t - 6 * u, cy + half, cx + t + 6 * u, cy + half + 34 * u),
            (cx - half - 34 * u, cy + t - 6 * u, cx - half, cy + t + 6 * u),
            (cx + half, cy + t - 6 * u, cx + half + 34 * u, cy + t + 6 * u),
        ):
            d.rectangle((x0, y0, x1, y1), fill=RULE)
    d.rounded_rectangle(die, radius=22 * u, fill=LIFT, outline=RULE, width=int(4 * u))
    # 4x4 compute tiles, one lit orange
    pad, gap = 34 * u, 14 * u
    tile = (2 * half - 2 * pad - 3 * gap) / 4
    for r in range(4):
        for c in range(4):
            x0 = cx - half + pad + c * (tile + gap)
            y0 = cy - half + pad + r * (tile + gap)
            fill = ORANGE if (r, c) == (1, 2) else BLUE if (r + c) % 3 == 0 else (48, 70, 104)
            d.rounded_rectangle((x0, y0, x0 + tile, y0 + tile), radius=8 * u, fill=fill)
    # chat bubble rising from the lit tile
    bx0, by0 = cx + half + 90 * u, cy - half - 150 * u
    bw, bh = 330 * u, 170 * u
    bubble = (bx0, by0, bx0 + bw, by0 + bh)
    img = glow(img, bubble, ORANGE, 40 * u)
    d = ImageDraw.Draw(img)
    tail = [(bx0 + 40 * u, by0 + bh - 2 * u), (bx0 + 110 * u, by0 + bh - 2 * u), (bx0 - 10 * u, by0 + bh + 80 * u)]
    d.polygon(tail, fill=PAPER)
    d.rounded_rectangle(bubble, radius=48 * u, fill=PAPER)
    for i in range(3):  # typing dots
        ox = bx0 + bw / 2 + (i - 1) * 70 * u
        oy = by0 + bh / 2
        d.ellipse((ox - 20 * u, oy - 20 * u, ox + 20 * u, oy + 20 * u), fill=ORANGE if i == 1 else SURFACE)
    # a trace from the lit tile to the bubble
    lx = cx - half + pad + 2 * (tile + gap) + tile / 2
    ly = cy - half + pad + 1 * (tile + gap)
    d.line([(lx, ly), (lx, cy - half - 70 * u), (bx0 - 10 * u, cy - half - 70 * u)], fill=ORANGE, width=int(6 * u))
    return img


def mcp_cover(w, h):
    img, d = canvas(w, h)
    u = min(w, h) / 578 * S
    cx, cy = w * S * 0.62, h * S * 0.52
    # cloud: overlapping circles behind the page
    cloud = [(-210, 40, 150), (-60, -60, 190), (120, -20, 170), (240, 70, 130), (20, 110, 160)]
    layer = Image.new("RGB", img.size, (0, 0, 0))
    ld = ImageDraw.Draw(layer)
    for ox, oy, r in cloud:
        ld.ellipse((cx + (ox - r) * u, cy + (oy - r) * u, cx + (ox + r) * u, cy + (oy + r) * u), fill=(40, 52, 72))
    img = Image.composite(layer, img, layer.convert("L").point(lambda v: 255 if v else 0))
    d = ImageDraw.Draw(img)
    # notebook page with cells
    pw, ph = 300 * u, 360 * u
    page = (cx - pw / 2, cy - ph / 2, cx + pw / 2, cy + ph / 2)
    img = glow(img, page, BLUE, 26 * u)
    d = ImageDraw.Draw(img)
    d.rounded_rectangle(page, radius=24 * u, fill=PAPER)
    y = page[1] + 40 * u
    for i, (frac, colour) in enumerate(((0.8, BLUE), (0.55, RULE), (0.9, BLUE), (0.45, ORANGE), (0.7, RULE))):
        x0 = page[0] + 36 * u
        d.rounded_rectangle((x0 - 14 * u, y, x0 - 4 * u, y + 40 * u), radius=4 * u, fill=colour)  # run marker
        d.rounded_rectangle((x0 + 10 * u, y + 6 * u, x0 + 10 * u + (pw - 90 * u) * frac, y + 34 * u),
                            radius=10 * u, fill=(208, 204, 192))  # fmt: skip
        y += 60 * u
    # cable from the plug into the page
    px, py = w * S * 0.16, cy + 40 * u
    pts = [(px + 120 * u, py), (px + 260 * u, py), (px + 300 * u, py - 60 * u), (page[0] - 60 * u, py - 60 * u),
           (page[0] - 60 * u, cy), (page[0], cy)]  # fmt: skip
    d.line(pts, fill=ORANGE, width=int(14 * u), joint="curve")
    # plug body and prongs
    d.rounded_rectangle((px - 40 * u, py - 70 * u, px + 120 * u, py + 70 * u), radius=28 * u, fill=ORANGE)
    for dy in (-34, 34):
        d.rounded_rectangle((px - 110 * u, py + (dy - 12) * u, px - 40 * u, py + (dy + 12) * u), radius=6 * u, fill=PAPER)
    d.rounded_rectangle((px + 10 * u, py - 30 * u, px + 70 * u, py + 30 * u), radius=12 * u, fill=(180, 72, 30))
    # socket on the page edge
    d.rounded_rectangle((page[0] - 18 * u, cy - 26 * u, page[0] + 18 * u, cy + 26 * u), radius=8 * u, fill=SURFACE)
    return img


def write(draw, slug):
    out = {}
    for mode, (w, h) in SIZES.items():
        img = draw(w, h).resize((w, h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=92, optimize=True)
        data = buf.getvalue()
        name = f"{mode}-cover.{hashlib.sha256(data).hexdigest()[:8]}.jpg"
        (HERE / slug / name).write_bytes(data)
        out[mode] = name
        print(f"wrote {slug}/{name}  {w}x{h}  {len(data) // 1024} KB")
    return out


if __name__ == "__main__":
    write(tpu_cover, "gemma4-colab-tpu")
    write(mcp_cover, "colab-mcp")

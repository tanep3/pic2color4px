#!/usr/bin/env python3
"""pic2color4px prototype: joint 3-channel 2x2-cell pattern dithering.
Output pixels are RGB on/off (8 colours). Per-cell counts per channel come
from cell-level error diffusion (FS weights transplanted to cells, as in
pic2gray4px); the arrangement of the three channels' patterns inside each
cell is then optimised JOINTLY against the perceived colour error
(residual after a Gaussian visual kernel), by coordinate descent.
"""
import numpy as np
from PIL import Image

# ---- patterns: bit i = pixel (x=i%2, y=i//2) of the 2x2 cell ----
PATTERNS = {
    0: [0b0000],
    1: [0b0001, 0b0010, 0b0100, 0b1000],
    2: [0b1001, 0b0110],                      # the two diagonals
    3: [0b1110, 0b1101, 0b1011, 0b0111],
    4: [0b1111],
}
POS = [(0, 0), (1, 0), (0, 1), (1, 1)]        # (x, y) per bit index

def cell_counts(img):
    """img: HxWx3 float 0..1 (H, W even). Returns counts Hc x Wc x 3."""
    H, W, _ = img.shape
    Hc, Wc = H // 2, W // 2
    means = img.reshape(Hc, 2, Wc, 2, 3).mean(axis=(1, 3))
    err = np.zeros((Hc, Wc, 3))
    counts = np.zeros((Hc, Wc, 3), dtype=int)
    WTS = [((1, 0), 7/16), ((-1, 1), 3/16), ((0, 1), 5/16), ((1, 1), 1/16)]
    for y in range(Hc):
        for x in range(Wc):
            t = means[y, x] + err[y, x]
            k = np.clip(np.floor(t * 4 + 0.5), 0, 4).astype(int)
            counts[y, x] = k
            e = (t * 4 - k) / 4.0
            for (dx, dy), w in WTS:
                nx, ny = x + dx, y + dy
                if 0 <= nx < Wc and 0 <= ny < Hc:
                    err[ny, nx] += e * w
    return counts

def render(counts, pats):
    Hc, Wc, _ = counts.shape
    out = np.zeros((Hc * 2, Wc * 2, 3))
    for c in range(3):
        for y in range(Hc):
            for x in range(Wc):
                m = pats[y, x, c]
                for i, (px, py) in enumerate(POS):
                    if m >> i & 1:
                        out[y * 2 + py, x * 2 + px, c] = 1.0
    return out

def gauss1d(sigma=1.1, radius=4):
    x = np.arange(-radius, radius + 1)
    g = np.exp(-x**2 / (2 * sigma**2))
    return g / g.sum()

def blur(img, g):
    r = len(g) // 2
    pad = np.pad(img, ((r, r), (r, r), (0, 0)), mode='reflect')
    tmp = np.apply_along_axis(lambda m: np.convolve(m, g, mode='valid'), 1, pad)
    tmp = np.apply_along_axis(lambda m: np.convolve(m, g, mode='valid'), 0, tmp)
    # apply_along_axis over 3D is awkward; do it separably with stride tricks instead
    return tmp

def blur2(img, g):
    r = len(g) // 2
    H, W, C = img.shape
    pad = np.pad(img, ((r, r), (r, r), (0, 0)), mode='reflect')
    out = np.zeros_like(img)
    for c in range(C):
        p = pad[:, :, c]
        tmp = np.zeros((H + 2 * r, W))
        for i, wgt in enumerate(g):
            tmp += wgt * p[:, i:i + W]
        res = np.zeros((H, W))
        for i, wgt in enumerate(g):
            res += wgt * tmp[i:i + H, :]
        out[:, :, c] = res
    return out

def optimise(img, counts, passes=6):
    H, W, _ = img.shape
    Hc, Wc = H // 2, W // 2
    g = gauss1d()
    r = len(g) // 2
    K = np.outer(g, g)
    pats = np.zeros((Hc, Wc, 3), dtype=int)
    for y in range(Hc):
        for x in range(Wc):
            for c in range(3):
                pats[y, x, c] = PATTERNS[counts[y, x, c]][0]
    comp = render(counts, pats)
    R = blur2(comp, g) - blur2(img, g)          # perceived residual (RGB)
    total_changes = 0
    for p in range(passes):
        changed = 0
        for y in range(Hc):
            for x in range(Wc):
                for c in range(3):
                    cur = pats[y, x, c]
                    k = counts[y, x, c]
                    best_m, best_gain = cur, 0.0
                    for m in PATTERNS[k]:
                        if m == cur:
                            continue
                        # pixel deltas in this cell/channel
                        deltas = []
                        for i, (px, py) in enumerate(POS):
                            d = ((m >> i) & 1) - ((cur >> i) & 1)
                            if d:
                                deltas.append((x * 2 + px, y * 2 + py, float(d)))
                        # footprint: union of kernel supports
                        xs = [d[0] for d in deltas]; ys = [d[1] for d in deltas]
                        x0, x1 = max(0, min(xs) - r), min(W - 1, max(xs) + r)
                        y0, y1 = max(0, min(ys) - r), min(H - 1, max(ys) + r)
                        dB = np.zeros((y1 - y0 + 1, x1 - x0 + 1))
                        for (pxx, pyy, d) in deltas:
                            for wy in range(y0, y1 + 1):
                                ky = wy - pyy + r
                                if not (0 <= ky < len(g)): continue
                                for wx in range(x0, x1 + 1):
                                    kx = wx - pxx + r
                                    if 0 <= kx < len(g):
                                        dB[wy - y0, wx - x0] += d * g[ky] * g[kx]
                        Rp = R[y0:y1 + 1, x0:x1 + 1, c]
                        gain = -float(np.sum(2 * Rp * dB + dB * dB))  # score decrease
                        if gain > best_gain + 1e-12:
                            best_gain, best_m = gain, m
                    if best_m != cur:
                        # apply: update comp + R footprint
                        for i, (px, py) in enumerate(POS):
                            d = ((best_m >> i) & 1) - ((cur >> i) & 1)
                            if d:
                                gx, gy = x * 2 + px, y * 2 + py
                                comp[gy, gx, c] += d
                                x0, x1 = max(0, gx - r), min(W - 1, gx + r)
                                y0, y1 = max(0, gy - r), min(H - 1, gy + r)
                                for wy in range(y0, y1 + 1):
                                    for wx in range(x0, x1 + 1):
                                        R[wy, wx, c] += d * g[wy - gy + r] * g[wx - gx + r]
                        pats[y, x, c] = best_m
                        changed += 1
        total_changes += changed
        print(f'  pass {p + 1}: {changed} pattern changes')
        if changed == 0:
            break
    return comp

def main():
    src = Image.open('/home/hatch/workspace/pic2gray4px/docs/images/taneko-before.png').convert('RGB')
    # MSX2 screen: 256x212. Fit height 212, blur-fill the sides to 256 wide.
    base = src.resize((212, 212), Image.LANCZOS)
    canvas = base.resize((256, 212), Image.BILINEAR).filter(ImageFilter := __import__('PIL.ImageFilter', fromlist=['GaussianBlur']).GaussianBlur(18))
    canvas.paste(base, (22, 0))
    canvas.save('/home/hatch/workspace/dither-eval/msx/color/input_256x212.png')
    img = np.asarray(canvas, dtype=float) / 255.0
    print('counts (cell diffusion, per channel)...')
    counts = cell_counts(img)
    print('joint arrangement optimisation...')
    comp = optimise(img, counts)
    out = Image.fromarray((comp * 255).astype(np.uint8))
    out.save('/home/hatch/workspace/dither-eval/msx/color/pic2color_raw.png')
    print('saved pic2color_raw.png; unique colours:', len(np.unique((comp * 255).astype(np.uint8).reshape(-1, 3), axis=0)))

if __name__ == '__main__':
    main()

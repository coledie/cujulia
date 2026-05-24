"""CUDA Mandelbrot / Julia renderer.

Quick start:
    python render.py                                   # default mandelbrot
    python render.py --mode julia --jc -0.8 0.156
    python render.py --width 16384 --height 16384 --iter 4000 -o out/huge.tif
    python render.py --center -0.743643887 0.131825904 --scale 0.0000035 --iter 8000

How to modify behavior:
  - Bailout (epsilon-ish):  --escape  (radius; squared internally)
  - Image size:             --width / --height
  - Zoom:                   --scale (vertical extent in complex plane) + --center
  - Iterations:             --iter
  - Julia constant:         --jc REAL IMAG
  - Palette:                --palette (any matplotlib cmap: twilight, magma, inferno, turbo, ...)
  - Bit depth:              --bits 8|16 (16 -> TIFF recommended)
  - Tile size (VRAM):       --tile (smaller if OOM, larger for speed)
  - Output format:          --format png|tiff (auto-detected from extension too)

The actual fractal formula lives in kernels.py (look for "INNER LOOP").
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import numpy as np
import cupy as cp

import kernels
import palette as pal


def parse_args():
    p = argparse.ArgumentParser(description="CUDA Mandelbrot/Julia renderer")
    p.add_argument("--mode", choices=["mandelbrot", "julia"], default="mandelbrot")
    p.add_argument("--width",  type=int, default=3840)
    p.add_argument("--height", type=int, default=2160)
    p.add_argument("--center", type=float, nargs=2, default=[-0.5, 0.0],
                   metavar=("RE", "IM"))
    p.add_argument("--scale",  type=float, default=3.0,
                   help="Vertical extent in the complex plane")
    p.add_argument("--iter",   type=int, default=1000, dest="max_iter")
    p.add_argument("--escape", type=float, default=2.0,
                   help="Bailout radius (squared internally)")
    p.add_argument("--jc", type=float, nargs=2, default=[-0.8, 0.156],
                   metavar=("RE", "IM"), help="Julia constant")
    p.add_argument("--epsilon", type=float, nargs="+", default=[0.0],
                   help="Random perturbation amplitude per iter, per pixel. "
                        "One value (same for re/im) or two (re im). "
                        "Each step adds uniform random in [-eps, +eps].")
    p.add_argument("--seed", type=int, default=1,
                   help="RNG seed for the --epsilon perturbation (reproducible).")
    p.add_argument("--blend-epsilon", type=float, default=None,
                   help="If set, render once with eps=0 and once with this "
                        "amplitude, then per-pixel average the smooth-iter "
                        "counts before coloring.")
    p.add_argument("--palette", default="twilight")
    p.add_argument("--bits", type=int, choices=[8, 16], default=16)
    p.add_argument("--tile", type=int, default=2048)
    p.add_argument("--format", choices=["auto", "png", "tiff"], default="auto")
    p.add_argument("--in-set", type=int, nargs=3, default=[0, 0, 0],
                   metavar=("R", "G", "B"))
    p.add_argument("-o", "--output", default="out/render.tif")
    return p.parse_args()


def render(args) -> np.ndarray:
    W, H = args.width, args.height
    tile = args.tile
    dtype = cp.uint16 if args.bits == 16 else cp.uint8
    np_dtype = np.uint16 if args.bits == 16 else np.uint8

    lut = pal.build_lut(args.palette, bit_depth=args.bits)
    full = np.empty((H, W, 3), dtype=np_dtype)

    smooth_buf = cp.empty(tile * tile, dtype=cp.float32)
    smooth_buf_b = (cp.empty(tile * tile, dtype=cp.float32)
                    if args.blend_epsilon is not None else None)
    mode_id = 0 if args.mode == "mandelbrot" else 1
    escape_r2 = float(args.escape) ** 2
    if len(args.epsilon) == 1:
        eps_x, eps_y = float(args.epsilon[0]), 0.0
    else:
        eps_x, eps_y = float(args.epsilon[0]), float(args.epsilon[1])

    n_tiles_x = (W + tile - 1) // tile
    n_tiles_y = (H + tile - 1) // tile
    total = n_tiles_x * n_tiles_y
    print(f"Rendering {W}x{H} {args.mode}  iter={args.max_iter}  "
          f"tiles={n_tiles_x}x{n_tiles_y}  bits={args.bits}")

    t0 = time.time()
    done = 0
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            tw = min(tile, W - tx * tile)
            th = min(tile, H - ty * tile)
            sub_flat = smooth_buf[: tw * th]
            kernels.launch(
                sub_flat,
                width=tw, height=th,
                cx=args.center[0], cy=args.center[1],
                scale=args.scale,
                max_iter=args.max_iter,
                escape_r2=escape_r2,
                mode=mode_id,
                jx=args.jc[0], jy=args.jc[1],
                tile_x=tx * tile, tile_y=ty * tile,
                full_width=W, full_height=H,
                eps_x=eps_x, eps_y=eps_y,
                seed=args.seed,
            )
            if args.blend_epsilon is not None:
                sub_b = smooth_buf_b[: tw * th]
                kernels.launch(
                    sub_b,
                    width=tw, height=th,
                    cx=args.center[0], cy=args.center[1],
                    scale=args.scale,
                    max_iter=args.max_iter,
                    escape_r2=escape_r2,
                    mode=mode_id,
                    jx=args.jc[0], jy=args.jc[1],
                    tile_x=tx * tile, tile_y=ty * tile,
                    full_width=W, full_height=H,
                    eps_x=float(args.blend_epsilon),
                    eps_y=float(args.blend_epsilon),
                    seed=args.seed,
                )
                # Average in smooth-iter space. Treat in-set (-1) as max_iter
                # for averaging; mark output in-set only if BOTH were in-set.
                a = cp.where(sub_flat < 0,
                             cp.float32(args.max_iter), sub_flat)
                b = cp.where(sub_b < 0,
                             cp.float32(args.max_iter), sub_b)
                avg = (a + b) * cp.float32(0.5)
                both_in = (sub_flat < 0) & (sub_b < 0)
                avg = cp.where(both_in, cp.float32(-1.0), avg)
                sub_flat = avg
            sub_smooth = sub_flat.reshape(th, tw)
            rgb = pal.apply(sub_smooth, lut, args.max_iter,
                            in_set=tuple(args.in_set))
            full[ty * tile: ty * tile + th,
                 tx * tile: tx * tile + tw, :] = cp.asnumpy(rgb)
            done += 1
            print(f"  tile {done}/{total}  ({tx},{ty}) {tw}x{th}", end="\r")
    cp.cuda.Stream.null.synchronize()
    print(f"\nGPU render done in {time.time() - t0:.2f}s")
    return full


def save(image: np.ndarray, path: str, fmt: str, bits: int):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()
    if fmt == "auto":
        fmt = "tiff" if ext in (".tif", ".tiff") else "png"

    if fmt == "tiff":
        import tifffile
        # Uncompressed TIFF for fastest viewing & lossless data.
        tifffile.imwrite(str(p), image, photometric="rgb", compression=None)
    else:
        from PIL import Image
        if bits == 16:
            # 16-bit RGB PNG via PIL (uses 'RGB;16' mode trick is unreliable;
            # fall back to TIFF if user wants 16-bit).
            raise SystemExit("16-bit PNG not supported; use --format tiff "
                             "or --bits 8.")
        Image.fromarray(image, mode="RGB").save(str(p), format="PNG",
                                                compress_level=1)
    print(f"Wrote {p}  ({image.shape[1]}x{image.shape[0]}, "
          f"{image.dtype}, {os.path.getsize(p) / 1e6:.1f} MB)")


def main():
    args = parse_args()
    t_total = time.time()
    img = render(args)
    t_render = time.time() - t_total
    t_save = time.time()
    save(img, args.output, args.format, args.bits)
    t_save = time.time() - t_save
    print(f"Render: {t_render:.2f}s   Save: {t_save:.2f}s   "
          f"Total: {t_render + t_save:.2f}s")


if __name__ == "__main__":
    main()

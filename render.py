"""CUDA fractal renderer.

Quick start:
    python render.py                                # default mandelbrot
    python render.py --formula multibrot --power 3
    python render.py --formula tricorn
    python render.py --formula burning-ship
    python render.py --formula phoenix --jc 0.56667 0 --phoenix-p -0.5 0
    python render.py --formula magnet
    python render.py --formula newton --power 3
    python render.py --formula nova   --power 3 --relax 1 0
    python render.py --formula lyapunov --lyap-seq BBBBBBAAAAAA
    python render.py --formula buddhabrot --buddha-samples 100000000

Resolution / iteration / palette / output flags work for every formula.
Bigger formulas:
    python render.py --width 16384 --height 16384 --tile 2048
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


# Formula -> (family, polynomial_flags, default_power, default_center,
#             default_scale, default_iter, default_palette)
_FORMULA_DEFAULTS = {
    "mandelbrot":   ("poly", 0,                                2, (-0.5, 0.0),  3.0,   1000, "twilight"),
    "multibrot":    ("poly", 0,                                3, ( 0.0, 0.0),  3.0,   1000, "twilight"),
    "tricorn":      ("poly", kernels.FLAG_CONJ,                2, ( 0.0, 0.0),  3.5,   1000, "twilight"),
    "burning-ship": ("poly", kernels.FLAG_ABS,                 2, (-0.5,-0.5),  3.5,   1000, "inferno"),
    "phoenix":      ("poly", kernels.FLAG_PHOENIX,             2, ( 0.0, 0.0),  2.6,    500, "twilight"),
    "magnet":       ("poly", kernels.FLAG_MAGNET,              2, ( 1.5, 0.0),  4.0,    500, "twilight"),
    "newton":       ("newton",   0, 3, (0.0, 0.0), 3.0,   60, None),
    "nova":         ("nova",     0, 3, (0.0, 0.0), 3.0,  200, None),
    "lyapunov":     ("lyapunov", 0, 0, (3.0, 3.0), 2.0,  200, None),
    "buddhabrot":   ("buddhabrot", 0, 0, (-0.5, 0.0), 3.0, 500, None),
}


def parse_args():
    p = argparse.ArgumentParser(description="CUDA fractal renderer")
    p.add_argument("--formula", default="mandelbrot",
                   choices=list(_FORMULA_DEFAULTS.keys()))
    # Legacy --mode: mandelbrot / julia (parameter vs dynamical view).
    p.add_argument("--mode", choices=["mandelbrot", "julia"], default="mandelbrot")
    p.add_argument("--width",  type=int, default=3840)
    p.add_argument("--height", type=int, default=2160)
    p.add_argument("--center", type=float, nargs=2, default=None,
                   metavar=("RE", "IM"))
    p.add_argument("--scale",  type=float, default=None,
                   help="Vertical extent in the complex (or parameter) plane")
    p.add_argument("--iter",   type=int,   default=None, dest="max_iter")
    p.add_argument("--escape", type=float, default=2.0)
    p.add_argument("--power",  type=int,   default=None,
                   help="Exponent d for multibrot/tricorn/ship/newton/nova")
    p.add_argument("--jc", type=float, nargs=2, default=[-0.8, 0.156],
                   metavar=("RE", "IM"), help="Julia constant / phoenix c")
    p.add_argument("--phoenix-p", type=float, nargs=2, default=[-0.5, 0.0],
                   metavar=("RE", "IM"), help="Phoenix p parameter (complex)")
    p.add_argument("--relax", type=float, nargs=2, default=[1.0, 0.0],
                   metavar=("RE", "IM"), help="Nova relaxation factor R (complex)")
    p.add_argument("--lyap-seq", default="AB",
                   help="Lyapunov A/B pattern, e.g. 'ABBAB'")
    p.add_argument("--lyap-warmup", type=int, default=200)
    p.add_argument("--buddha-samples", type=int, default=100_000_000)
    p.add_argument("--buddha-min-iter", type=int, default=20)
    p.add_argument("--buddha-batch", type=int, default=4_000_000,
                   help="Buddhabrot samples per kernel launch")
    p.add_argument("--buddha-formula", default="mandelbrot",
                   choices=["mandelbrot", "multibrot", "tricorn",
                            "burning-ship", "julia", "phoenix", "magnet"],
                   help="Which iteration the buddhabrot accumulator uses")
    p.add_argument("--buddha-power", type=int, default=2,
                   help="Power d for buddhabrot multibrot/julia formulas")
    p.add_argument("--epsilon", type=float, nargs="+", default=[0.0])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--palette", default=None)
    p.add_argument("--bits", type=int, choices=[8, 16], default=16)
    p.add_argument("--tile", type=int, default=2048)
    p.add_argument("--format", choices=["auto", "png", "tiff"], default="auto")
    p.add_argument("--in-set", type=int, nargs=3, default=[0, 0, 0],
                   metavar=("R", "G", "B"))
    p.add_argument("-o", "--output", default="out/render.tif")
    args = p.parse_args()

    # Fill formula-specific defaults.
    family, _, d_pow, d_center, d_scale, d_iter, d_palette = _FORMULA_DEFAULTS[args.formula]
    if args.center is None: args.center = list(d_center)
    if args.scale  is None: args.scale  = d_scale
    if args.max_iter is None: args.max_iter = d_iter
    if args.power is None:    args.power = d_pow
    if args.palette is None:  args.palette = d_palette or "twilight"
    args._family = family
    return args


# ---------------------------------------------------------------------------
# Polynomial family render path (mandelbrot, multibrot, tricorn, ship,
# phoenix, magnet) — uses tiled rendering + LUT palette like the original.
# ---------------------------------------------------------------------------

def _poly_flags(formula):
    return _FORMULA_DEFAULTS[formula][1]


def render_poly(args) -> np.ndarray:
    W, H = args.width, args.height
    tile = args.tile
    np_dtype = np.uint16 if args.bits == 16 else np.uint8

    lut = pal.build_lut(args.palette, bit_depth=args.bits)
    full = np.empty((H, W, 3), dtype=np_dtype)

    smooth_buf = cp.empty(tile * tile, dtype=cp.float32)
    mode_id = 0 if args.mode == "mandelbrot" else 1
    escape_r2 = float(args.escape) ** 2
    eps_x = float(args.epsilon[0])
    eps_y = float(args.epsilon[1]) if len(args.epsilon) > 1 else 0.0

    flags = _poly_flags(args.formula)
    # Phoenix is conventionally rendered as a Julia (z0 = pixel, c constant).
    if args.formula == "phoenix":
        mode_id = 1
        # If user did not set --jc explicitly, use a classic phoenix c.
        if args.jc == [-0.8, 0.156]:
            args.jc = [0.56667, 0.0]

    n_tiles_x = (W + tile - 1) // tile
    n_tiles_y = (H + tile - 1) // tile
    total = n_tiles_x * n_tiles_y
    print(f"Rendering {W}x{H} {args.formula}  iter={args.max_iter}  "
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
                eps_x=eps_x, eps_y=eps_y, seed=args.seed,
                power=args.power, flags=flags,
                phx=args.phoenix_p[0], phy=args.phoenix_p[1],
            )
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


# ---------------------------------------------------------------------------
# Newton / Nova render path.
# ---------------------------------------------------------------------------

# Distinct hues per root index (HSV evenly spaced) tinted by iteration speed.
def _newton_palette(d: int, bits: int) -> cp.ndarray:
    import colorsys
    hues = []
    for k in range(d):
        h = k / d
        r, g, b = colorsys.hsv_to_rgb(h, 0.95, 1.0)
        hues.append((r, g, b))
    # (d, 3) base colors in [0,1]
    arr = np.array(hues, dtype=np.float32)
    if bits == 16:
        return cp.asarray((arr * 65535.0 + 0.5).astype(np.uint16))
    return cp.asarray((arr * 255.0 + 0.5).astype(np.uint8))


def render_newton(args) -> np.ndarray:
    W, H = args.width, args.height
    tile = args.tile
    np_dtype = np.uint16 if args.bits == 16 else np.uint8
    full = np.empty((H, W, 3), dtype=np_dtype)

    d = max(2, args.power)
    base = _newton_palette(d, args.bits).astype(cp.float32)  # (d, 3)
    max_iter = args.max_iter
    is_nova = 1 if args._family == "nova" else 0

    smooth_buf = cp.empty(tile * tile, dtype=cp.float32)
    n_tiles_x = (W + tile - 1) // tile
    n_tiles_y = (H + tile - 1) // tile
    total = n_tiles_x * n_tiles_y
    print(f"Rendering {W}x{H} {args.formula}  d={d}  iter={max_iter}  "
          f"tiles={n_tiles_x}x{n_tiles_y}")

    t0 = time.time()
    done = 0
    in_set = cp.asarray(np.array(args.in_set, dtype=np.float32))
    max_val = 65535.0 if args.bits == 16 else 255.0
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            tw = min(tile, W - tx * tile)
            th = min(tile, H - ty * tile)
            sub_flat = smooth_buf[: tw * th]
            c_mode = 0 if args.mode == "mandelbrot" else 1
            kernels.launch_newton(
                sub_flat,
                width=tw, height=th,
                cx=args.center[0], cy=args.center[1],
                scale=args.scale, max_iter=max_iter,
                tile_x=tx * tile, tile_y=ty * tile,
                full_width=W, full_height=H,
                power=d,
                relax=(args.relax[0], args.relax[1]),
                nova_mode=is_nova,
                c_mode=c_mode,
                jx=args.jc[0], jy=args.jc[1],
            )
            s = sub_flat.reshape(th, tw)
            # Decode: <0 -> in-set; else root_idx = floor(s/10000), iter = s%10000
            in_set_mask = s < 0
            iters = cp.mod(s, 10000.0)
            root_idx = cp.clip(cp.floor(s / 10000.0).astype(cp.int32), 0, d - 1)
            # Brightness: faster convergence = brighter (lower iter)
            brightness = cp.clip(1.0 - iters / float(max_iter), 0.05, 1.0)
            # base[root_idx] gives an RGB; multiply by brightness.
            rgb_f = base[root_idx] * brightness[..., None]
            rgb_f = cp.where(in_set_mask[..., None], in_set[None, None, :], rgb_f)
            rgb = cp.clip(rgb_f + 0.5, 0, max_val).astype(
                cp.uint16 if args.bits == 16 else cp.uint8)
            full[ty * tile: ty * tile + th,
                 tx * tile: tx * tile + tw, :] = cp.asnumpy(rgb)
            done += 1
            print(f"  tile {done}/{total}", end="\r")
    cp.cuda.Stream.null.synchronize()
    print(f"\nGPU render done in {time.time() - t0:.2f}s")
    return full


# ---------------------------------------------------------------------------
# Lyapunov render path.
# ---------------------------------------------------------------------------

def render_lyapunov(args) -> np.ndarray:
    import matplotlib.pyplot as plt
    W, H = args.width, args.height
    tile = args.tile
    np_dtype = np.uint16 if args.bits == 16 else np.uint8
    full = np.empty((H, W, 3), dtype=np_dtype)

    pat = np.array([0 if ch.upper() == 'A' else 1 for ch in args.lyap_seq],
                   dtype=np.uint8)
    if pat.size == 0:
        pat = np.array([0, 1], dtype=np.uint8)
    pat_dev = cp.asarray(pat)

    smooth_buf = cp.empty(tile * tile, dtype=cp.float32)
    n_tiles_x = (W + tile - 1) // tile
    n_tiles_y = (H + tile - 1) // tile
    total = n_tiles_x * n_tiles_y
    print(f"Rendering {W}x{H} lyapunov seq='{args.lyap_seq}' "
          f"iter={args.max_iter} warmup={args.lyap_warmup}  "
          f"tiles={n_tiles_x}x{n_tiles_y}")

    t0 = time.time()
    done = 0
    # Lyapunov palette (Markus-style): chaotic L>0 -> black; stable L<0
    # ramps through deep blue -> teal -> gold -> pale yellow as L grows
    # more negative (more attractive / periodic). Hand-built ramp.
    n_lut = 4096
    xs = np.linspace(0.0, 1.0, n_lut)        # 0=most negative, 1=most positive
    # Map xs (0..1) over signed L range [-L_RANGE .. +L_RANGE].
    rgb_lut = np.zeros((n_lut, 3), dtype=np.float32)
    # Stable side: xs in [0, 0.5) -> L in [-L_RANGE, 0)
    # We want xs=0 (very stable) -> pale yellow; xs=0.5 (L=0) -> deep blue.
    # Chaotic side: xs in [0.5, 1] -> black ramping to near-black warm.
    for i, x in enumerate(xs):
        if x < 0.5:
            # 0..0.5 -> remap to t in [1..0] where 1 = very stable.
            t = 1.0 - 2.0 * x   # t=1 most stable, t=0 at L=0
            # Pale yellow (1, 1, 0.7) at t=1, deep blue (0.05, 0.1, 0.3) at t=0.
            # Pass through gold/orange/red as intermediate.
            if t > 0.6:
                u = (t - 0.6) / 0.4
                r, g, b = 1.0, 0.85 + 0.15 * u, 0.4 + 0.3 * u
            elif t > 0.3:
                u = (t - 0.3) / 0.3
                r, g, b = 0.9 + 0.1 * u, 0.5 + 0.35 * u, 0.05 + 0.35 * u
            else:
                u = t / 0.3
                r = 0.05 + 0.85 * u
                g = 0.1 + 0.4 * u
                b = 0.3 - 0.25 * u
        else:
            # Chaotic side: t in [0..1], t=0 just above L=0, t=1 max chaos.
            t = (x - 0.5) * 2.0
            # Dark indigo to black.
            r = 0.08 * (1 - t)
            g = 0.04 * (1 - t)
            b = 0.18 * (1 - t)
        rgb_lut[i] = (r, g, b)
    if args.bits == 16:
        lut = np.clip(rgb_lut * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    else:
        lut = np.clip(rgb_lut * 255.0 + 0.5, 0, 255).astype(np.uint8)
    lut_dev = cp.asarray(lut)

    L_RANGE = 1.5   # clamp signed exponent into [-L_RANGE, L_RANGE]
    for ty in range(n_tiles_y):
        for tx in range(n_tiles_x):
            tw = min(tile, W - tx * tile)
            th = min(tile, H - ty * tile)
            sub_flat = smooth_buf[: tw * th]
            kernels.launch_lyapunov(
                sub_flat,
                width=tw, height=th,
                cx=args.center[0], cy=args.center[1],
                scale=args.scale, max_iter=args.max_iter,
                tile_x=tx * tile, tile_y=ty * tile,
                full_width=W, full_height=H,
                pattern_bytes=pat_dev,
                warmup=args.lyap_warmup, x0=0.5,
            )
            s = sub_flat.reshape(th, tw)
            t = cp.clip((s + L_RANGE) / (2.0 * L_RANGE), 0.0, 1.0)
            idx = cp.clip((t * (lut_dev.shape[0] - 1)).astype(cp.int32),
                          0, lut_dev.shape[0] - 1)
            rgb = lut_dev[idx]
            full[ty * tile: ty * tile + th,
                 tx * tile: tx * tile + tw, :] = cp.asnumpy(rgb)
            done += 1
            print(f"  tile {done}/{total}", end="\r")
    cp.cuda.Stream.null.synchronize()
    print(f"\nGPU render done in {time.time() - t0:.2f}s")
    return full


# ---------------------------------------------------------------------------
# Buddhabrot render path. Not tiled — needs a single global histogram.
# ---------------------------------------------------------------------------

def render_buddhabrot(args) -> np.ndarray:
    import matplotlib.pyplot as plt
    W, H = args.width, args.height

    # Decode the variant: which iteration formula the accumulator uses.
    bf = args.buddha_formula
    if bf == "mandelbrot":
        power, flags, julia_mode = 2, 0, 0
    elif bf == "multibrot":
        power, flags, julia_mode = args.buddha_power, 0, 0
    elif bf == "tricorn":
        power, flags, julia_mode = 2, kernels.FLAG_CONJ, 0
    elif bf == "burning-ship":
        power, flags, julia_mode = 2, kernels.FLAG_ABS, 0
    elif bf == "julia":
        power, flags, julia_mode = args.buddha_power, 0, 1
    elif bf == "phoenix":
        # Phoenix is conventionally rendered as a Julia (fix c, sample z0)
        # with the z_prev feedback term p*z_prev.
        power, flags, julia_mode = 2, kernels.FLAG_PHOENIX, 1
    elif bf == "magnet":
        power, flags, julia_mode = 2, kernels.FLAG_MAGNET, 0
    else:
        raise ValueError(f"Unknown --buddha-formula {bf!r}")

    print(f"Rendering {W}x{H} buddhabrot[{bf}]  iter={args.max_iter}  "
          f"samples={args.buddha_samples:,}  power={power}  flags={flags}  "
          f"julia={julia_mode}")

    hist = cp.zeros(H * W, dtype=cp.uint32)
    aspect = W / H
    sample_half_h = args.scale * 0.5
    sample_half_w = sample_half_h * aspect
    # Widen the sampling rectangle so trajectories from just outside the
    # frame still deposit into it. For Julia mode we sample initial z in
    # the frame itself, so use a tighter widen.
    widen = 1.05 if julia_mode == 0 else 1.0
    sample_half_h *= widen
    sample_half_w *= widen
    sample_cx = args.center[0]
    sample_cy = args.center[1]

    t0 = time.time()
    remaining = args.buddha_samples
    batch = args.buddha_batch
    seed = int(args.seed)
    chunk_idx = 0
    while remaining > 0:
        n = min(batch, remaining)
        kernels.launch_buddhabrot(
            hist,
            width=W, height=H,
            cx=args.center[0], cy=args.center[1],
            scale=args.scale,
            max_iter=args.max_iter,
            min_iter=args.buddha_min_iter,
            sample_half_w=sample_half_w,
            sample_half_h=sample_half_h,
            sample_cx=sample_cx, sample_cy=sample_cy,
            seed=(seed * 0x9E3779B97F4A7C15 + chunk_idx * 0xD2B74407B1CE6E93) & ((1 << 64) - 1),
            n_samples=n,
            power=power, flags=flags,
            julia_mode=julia_mode,
            jx=float(args.jc[0]), jy=float(args.jc[1]),
            phx=float(args.phoenix_p[0]), phy=float(args.phoenix_p[1]),
        )
        remaining -= n
        chunk_idx += 1
        if chunk_idx % 5 == 0 or remaining == 0:
            done_p = (args.buddha_samples - remaining) / args.buddha_samples * 100.0
            print(f"  buddhabrot {done_p:5.1f}% ({args.buddha_samples - remaining:,} samples)",
                  end="\r")
    cp.cuda.Stream.null.synchronize()
    print(f"\nGPU accumulate done in {time.time() - t0:.2f}s")

    # Tone mapping: sqrt curve normalized to a percentile (not the raw max).
    # Magnet and other formulas with strong attractor spikes deposit
    # orders-of-magnitude more counts at a single pixel; using the raw
    # max would crush the rest of the image to black.
    h_f = hist.reshape(H, W).astype(cp.float32)
    nonzero = h_f[h_f > 0]
    if nonzero.size == 0:
        print("WARNING: buddhabrot histogram is empty (no escaping trajectories).")
        return np.zeros((H, W, 3), dtype=np.uint16 if args.bits == 16 else np.uint8)
    # Use 99.9th percentile as the normalization ceiling so single-pixel
    # spikes don't wash out the rest of the image.
    norm_v = float(cp.percentile(nonzero, 99.9).item())
    if norm_v <= 0:
        norm_v = float(h_f.max().item())
    h_n = cp.sqrt(cp.clip(h_f / norm_v, 0.0, 1.0))
    h_n = cp.clip(h_n * 1.4, 0.0, 1.0)

    cmap = plt.get_cmap(args.palette)
    xs = np.linspace(0.0, 1.0, 4096)
    rgba = cmap(xs)
    if args.bits == 16:
        lut = np.clip(rgba[:, :3] * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    else:
        lut = np.clip(rgba[:, :3] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    lut_dev = cp.asarray(lut)
    idx = cp.clip((h_n * (lut_dev.shape[0] - 1)).astype(cp.int32),
                  0, lut_dev.shape[0] - 1)
    rgb = lut_dev[idx]
    return cp.asnumpy(rgb)


# ---------------------------------------------------------------------------

def render(args) -> np.ndarray:
    fam = args._family
    if fam == "poly":
        return render_poly(args)
    if fam in ("newton", "nova"):
        return render_newton(args)
    if fam == "lyapunov":
        return render_lyapunov(args)
    if fam == "buddhabrot":
        return render_buddhabrot(args)
    raise ValueError(f"unknown family {fam}")


def save(image: np.ndarray, path: str, fmt: str, bits: int):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ext = p.suffix.lower()
    if fmt == "auto":
        fmt = "tiff" if ext in (".tif", ".tiff") else "png"

    if fmt == "tiff":
        import tifffile
        tifffile.imwrite(str(p), image, photometric="rgb", compression=None)
    else:
        from PIL import Image
        if bits == 16:
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

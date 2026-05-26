"""Render every fractal formula at a chosen resolution.

Usage:
    python render_all.py                          # 16k PNG (default)
    python render_all.py --width 2048 --bits 16   # quick test at 2k TIFF
    python render_all.py --skip buddhabrot        # skip a formula
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path


# Per-formula CLI args. Each entry is a list of args passed to render.py.
# Resolution / iter / format / output get appended by this script.
RECIPES = [
    # name           extra args
    ("mandelbrot",   ["--formula", "mandelbrot",   "--palette", "magma"]),
    ("multibrot_d3", ["--formula", "multibrot", "--power", "3", "--palette", "magma"]),
    ("multibrot_d4", ["--formula", "multibrot", "--power", "4", "--palette", "magma"]),
    ("multibrot_d5", ["--formula", "multibrot", "--power", "5", "--palette", "magma"]),
    ("tricorn",      ["--formula", "tricorn",      "--palette", "magma"]),
    ("burning_ship", ["--formula", "burning-ship", "--center", "-0.5", "-0.5",
                       "--scale", "2.5", "--palette", "magma"]),
    ("phoenix",      ["--formula", "phoenix",
                       "--jc", "0.56667", "0",
                       "--phoenix-p", "-0.5", "0",
                       "--center", "0", "0", "--scale", "2.6",
                       "--palette", "magma"]),
    ("magnet",       ["--formula", "magnet",
                       "--center", "1.5", "0", "--scale", "4.0",
                       "--palette", "magma"]),
    ("newton",       ["--formula", "newton", "--power", "3",
                       "--center", "0", "0", "--scale", "3.0",
                       "--iter", "60"]),
    ("nova",         ["--formula", "nova", "--power", "3",
                       "--relax", "1.0", "0",
                       "--center", "-0.4", "0", "--scale", "2.0",
                       "--iter", "200"]),
    ("lyapunov",     ["--formula", "lyapunov", "--lyap-seq", "BBBBBBAAAAAA",
                       "--center", "3.4", "3.4", "--scale", "1.6",
                       "--iter", "400", "--lyap-warmup", "300"]),
    ("buddhabrot",   ["--formula", "buddhabrot",
                       "--center", "-0.5", "0", "--scale", "3.0",
                       "--iter", "500",
                       "--buddha-samples", "200000000",
                       "--buddha-batch", "20000000",
                       "--buddha-min-iter", "20",
                       "--palette", "magma"]),
    # ------- Buddha-logic variants (trajectory accumulation) -------
    ("buddha_multibrot_d3", ["--formula", "buddhabrot",
                       "--buddha-formula", "multibrot", "--buddha-power", "3",
                       "--center", "0", "0", "--scale", "2.8",
                       "--iter", "500",
                       "--buddha-samples", "100000000",
                       "--buddha-batch", "20000000",
                       "--buddha-min-iter", "20",
                       "--palette", "magma"]),
    ("buddha_multibrot_d5", ["--formula", "buddhabrot",
                       "--buddha-formula", "multibrot", "--buddha-power", "5",
                       "--center", "0", "0", "--scale", "2.6",
                       "--iter", "500",
                       "--buddha-samples", "100000000",
                       "--buddha-batch", "20000000",
                       "--buddha-min-iter", "20",
                       "--palette", "magma"]),
    ("buddha_tricorn", ["--formula", "buddhabrot",
                       "--buddha-formula", "tricorn",
                       "--center", "0", "0", "--scale", "3.5",
                       "--iter", "500",
                       "--buddha-samples", "100000000",
                       "--buddha-batch", "20000000",
                       "--buddha-min-iter", "20",
                       "--palette", "magma"]),
    ("buddha_burning_ship", ["--formula", "buddhabrot",
                       "--buddha-formula", "burning-ship",
                       "--center", "-0.5", "-0.5", "--scale", "3.0",
                       "--iter", "500",
                       "--buddha-samples", "100000000",
                       "--buddha-batch", "20000000",
                       "--buddha-min-iter", "20",
                       "--palette", "magma"]),
    ("buddha_phoenix", ["--formula", "buddhabrot",
                       "--buddha-formula", "phoenix",
                       "--jc", "0.56667", "0",
                       "--phoenix-p", "-0.5", "0",
                       "--center", "0.8", "0", "--scale", "1.8",
                       "--iter", "500",
                       "--buddha-samples", "125000000",
                       "--buddha-batch", "20000000",
                       "--buddha-min-iter", "30",
                       "--palette", "magma"]),
    ("buddha_magnet", ["--formula", "buddhabrot",
                       "--buddha-formula", "magnet",
                       "--center", "0.5", "0", "--scale", "2.5",
                       "--iter", "500",
                       "--buddha-samples", "300000000",
                       "--buddha-batch", "20000000",
                       "--buddha-min-iter", "10",
                       "--palette", "magma"]),
    # ------- Buddha-Julia: trajectory accumulation with fixed c -------
    ("buddha_julia_dendrite", ["--formula", "buddhabrot",
                       "--buddha-formula", "julia", "--jc", "0", "1.0",
                       "--center", "0", "0", "--scale", "3.2",
                       "--iter", "500",
                       "--buddha-samples", "125000000",
                       "--buddha-batch", "20000000",
                       "--buddha-min-iter", "30",
                       "--palette", "magma"]),
    ("buddha_julia_douady", ["--formula", "buddhabrot",
                       "--buddha-formula", "julia",
                       "--jc", "-0.123", "0.745",
                       "--center", "0", "0", "--scale", "3.2",
                       "--iter", "500",
                       "--buddha-samples", "125000000",
                       "--buddha-batch", "20000000",
                       "--buddha-min-iter", "30",
                       "--palette", "magma"]),
    ("buddha_julia_seahorse", ["--formula", "buddhabrot",
                       "--buddha-formula", "julia",
                       "--jc", "-0.75", "0.11",
                       "--center", "0", "0", "--scale", "3.2",
                       "--iter", "500",
                       "--buddha-samples", "125000000",
                       "--buddha-batch", "20000000",
                       "--buddha-min-iter", "30",
                       "--palette", "magma"]),
    ("buddha_julia_dragon", ["--formula", "buddhabrot",
                       "--buddha-formula", "julia",
                       "--jc", "-0.8", "0.156",
                       "--center", "0", "0", "--scale", "3.2",
                       "--iter", "500",
                       "--buddha-samples", "125000000",
                       "--buddha-batch", "20000000",
                       "--buddha-min-iter", "30",
                       "--palette", "magma"]),
    ("buddha_julia_spiral", ["--formula", "buddhabrot",
                       "--buddha-formula", "julia",
                       "--jc", "0.285", "0.01",
                       "--center", "0", "0", "--scale", "3.2",
                       "--iter", "500",
                       "--buddha-samples", "125000000",
                       "--buddha-batch", "20000000",
                       "--buddha-min-iter", "30",
                       "--palette", "magma"]),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--width",  type=int, default=16384)
    p.add_argument("--height", type=int, default=16384)
    p.add_argument("--tile",   type=int, default=2048)
    p.add_argument("--bits",   type=int, choices=[8, 16], default=8)
    p.add_argument("--format", choices=["auto", "png", "tiff"], default="png")
    p.add_argument("--outdir", default="out/gallery")
    p.add_argument("--skip", nargs="+", default=[])
    p.add_argument("--only", nargs="+", default=None,
                   help="If set, only render these names.")
    args = p.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    ext = "tif" if args.format == "tiff" or (args.format == "auto" and args.bits == 16) else "png"
    if args.format == "auto":
        args.format = "tiff" if ext == "tif" else "png"

    results = []
    t_all = time.time()
    for name, extra in RECIPES:
        if name in args.skip:
            print(f"=== SKIP {name}")
            continue
        if args.only and name not in args.only:
            continue
        out = str(Path(args.outdir) / f"{name}_{args.width}x{args.height}.{ext}")
        cmd = [sys.executable, "render.py",
               "--width", str(args.width),
               "--height", str(args.height),
               "--tile", str(args.tile),
               "--bits", str(args.bits),
               "--format", args.format,
               "-o", out] + extra
        print(f"\n=== {name}\n    {' '.join(cmd)}")
        t0 = time.time()
        r = subprocess.run(cmd)
        wall = time.time() - t0
        if r.returncode != 0:
            print(f"FAILED: {name}  (exit {r.returncode})")
            results.append((name, "FAIL", wall, 0))
            continue
        size_mb = Path(out).stat().st_size / 1e6 if Path(out).exists() else 0
        print(f"OK  {name}  wall={wall:.1f}s  size={size_mb:.1f}MB")
        results.append((name, "OK", wall, size_mb))

    print("\n" + "=" * 60)
    print(f"Total wall: {time.time() - t_all:.1f}s")
    print(f"{'Name':<16} {'Status':<6} {'Wall':>8}  {'Size':>10}")
    for name, st, wall, sz in results:
        print(f"{name:<16} {st:<6} {wall:>7.1f}s  {sz:>8.1f}MB")


if __name__ == "__main__":
    main()

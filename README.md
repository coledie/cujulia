# cujulia — CUDA Mandelbrot / Julia renderer

GPU-accelerated, tiled, lossless (uncompressed TIFF) fractal renderer.
Built on CuPy `RawKernel` (real CUDA C kernels), driven from a tiny Python CLI.

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`cupy-cuda12x` works with NVIDIA driver supporting CUDA 12+ (including newer
CUDA 13 drivers, which are backward-compatible). If you have an older driver
that only supports CUDA 11, install `cupy-cuda11x` instead.

## Usage

```powershell
# 4K mandelbrot, 16-bit TIFF
python render.py

# Julia set
python render.py --mode julia --jc -0.8 0.156 -o out/julia.tif

# 16K x 16K deep zoom
python render.py --width 16384 --height 16384 --iter 4000 `
    --center -0.743643887 0.131825904 --scale 0.0000035 `
    --tile 2048 -o out/deep.tif

# 8-bit PNG (universally viewable, smaller)
python render.py --bits 8 --format png -o out/quick.png
```

Run `python render.py --help` for all flags.

## Customizing

| Want to change | How |
|---|---|
| Bailout radius | `--escape 2.0` (squared internally) |
| Random perturbation | `--epsilon AMP` — each iteration of each pixel adds an independent uniform random in `[-AMP, +AMP]` to z (re and im). `--epsilon 0` is the deterministic fractal. `--epsilon RE_AMP IM_AMP` for asymmetric amplitudes. |
| RNG seed | `--seed N` (reproducible random perturbation; default 1) |
| Resolution | `--width / --height` |
| Zoom | `--scale` (vertical extent) and `--center RE IM` |
| Iteration depth | `--iter` |
| Julia constant | `--jc RE IM` |
| Palette | `--palette twilight` (any matplotlib cmap: `magma`, `inferno`, `turbo`, `viridis`, ...) |
| Color of in-set pixels | `--in-set R G B` |
| Bit depth | `--bits 8` or `--bits 16` (16 -> use TIFF) |
| Tile size (VRAM tuning) | `--tile 2048` — smaller if OOM, larger for speed |

### Changing the fractal formula

Open `kernels.py` and edit the block marked `INNER LOOP`. The default is
`z = z*z + c + (rx, ry)`, where `(rx, ry)` is a fresh per-iteration random
perturbation in `[-eps, +eps]` (zero when `--epsilon 0`). You can swap in
e.g. `z = z*z*z + c`, burning ship (`zx = |zx|, zy = |zy|` before squaring),
etc.

### Random epsilon details

- The perturbation is **per-pixel, per-iteration** (a fresh random draw at
  every step), not a fixed offset. Visually this looks like noise on the
  chaotic boundary of the set; the deep interior and far exterior are
  unaffected.
- RNG is xorshift64 seeded by `splitmix64(seed ⊕ pixel_coords)`, so output
  is fully reproducible given the same `--seed`.
- At small amplitudes (`--epsilon 1e-4`) you get a subtle fuzz; at larger
  amplitudes (`--epsilon 0.05`+) the set dissolves into noise.

## Output

- Default: **uncompressed 16-bit RGB TIFF**, fully lossless, ~1.5 GB for 16K×16K.
- Switch to PNG for 8-bit work.

## How it works

1. Image split into `--tile` sized tiles (default 2048).
2. Per tile, a CUDA kernel computes a `float32` smooth-iteration value per pixel.
3. A second CUDA kernel maps those values through a 4096-entry GPU LUT
   (log-compressed) into 8- or 16-bit RGB.
4. Tile is copied to a host buffer; full image is written with `tifffile`
   (or Pillow for PNG).

## Benchmarks

Square images, 1000 iterations, `--tile 2048`, 16-bit uncompressed TIFF output.
Measured on: **NVIDIA GeForce RTX 4060 (8 GB)**, Windows, CUDA 13.2 driver,
`cupy-cuda12x` 14.1.

| Mode       | Size         |  GPU  | Render | Save  | Wall  | File size |
|------------|--------------|------:|-------:|------:|------:|----------:|
| Mandelbrot |  1024×1024   | 0.05s |  0.14s | 0.02s | 0.83s |    6.3 MB |
| Mandelbrot |  2048×2048   | 0.14s |  0.20s | 0.02s | 0.90s |   25.2 MB |
| Mandelbrot |  4096×4096   | 0.44s |  0.50s | 0.07s | 1.25s |  100.7 MB |
| Mandelbrot |  8192×8192   | 1.71s |  1.78s | 0.26s | 2.76s |  402.7 MB |
| Mandelbrot | 16384×16384  | 6.68s |  6.76s | 1.03s | 8.63s | 1610.6 MB |
| Julia      |  1024×1024   | 0.04s |  0.12s | 0.03s | 0.84s |    6.3 MB |
| Julia      |  2048×2048   | 0.09s |  0.17s | 0.03s | 0.90s |   25.2 MB |
| Julia      |  4096×4096   | 0.28s |  0.35s | 0.15s | 1.17s |  100.7 MB |
| Julia      |  8192×8192   | 1.11s |  1.19s | 0.26s | 2.21s |  402.7 MB |
| Julia      | 16384×16384  | 3.84s |  3.91s | 0.99s | 5.71s | 1610.6 MB |

Columns:
- **GPU** — pure CUDA kernel + palette time (`cp.cuda.Stream.null.synchronize()` after all tiles).
- **Render** — GPU work plus per-tile device→host copy and assembly into the host buffer.
- **Save** — writing the uncompressed TIFF to disk.
- **Wall** — total process wall time including Python startup, CUDA context init, kernel JIT, and module imports (~0.7s baseline).
- **File size** — 16-bit RGB uncompressed TIFF (= `W × H × 3 × 2` bytes + small header).

Notes:
- Mandelbrot is ~1.7× slower than Julia at 16K because the centered Mandelbrot view contains a much larger interior region that hits `--iter` (worst case per pixel).
- Render time scales roughly linearly with pixel count (4× area → ~4× GPU time).
- For 8-bit PNG output, save time is dominated by zlib compression — switching `--format png --bits 8` cuts file size ~10× but slows the save phase.
- 16K×16K peak host RAM: ~1.5 GB for the output buffer. GPU memory is bounded by `--tile` (a 2048-tile float32 smooth buffer is ~16 MB).

Reproduce with:

```powershell
python bench.py
```

Outputs go to `out/bench/`, plus `out/bench/results.json`.

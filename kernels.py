"""CUDA kernels for Mandelbrot and Julia set rendering.

Each kernel writes a float32 'smooth iteration count' per pixel.
Pixels that never escape get value = -1.0 (treated as 'in-set').

To modify the fractal formula, edit the INNER LOOP block below.
"""
import cupy as cp


_KERNEL_SRC = r"""
extern "C" __global__
void fractal(
    float* __restrict__ out,    // [height * width] smooth iter count
    const int   width,
    const int   height,
    const double cx,            // view center x (real)
    const double cy,            // view center y (imag)
    const double scale,         // vertical extent in complex plane
    const int   max_iter,
    const double escape_r2,     // bailout radius squared (epsilon-ish)
    const int   mode,           // 0 = mandelbrot, 1 = julia
    const double jx,            // julia constant real
    const double jy,            // julia constant imag
    const int   tile_x,         // pixel offset of this tile within full image
    const int   tile_y,
    const int   full_width,
    const int   full_height,
    const double eps_x,         // additive perturbation per iter (real)
    const double eps_y          // additive perturbation per iter (imag)
){
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    // Map pixel -> complex plane (square pixels; vertical extent = scale).
    const double dy = scale / (double)full_height;
    const double gx = (double)(px + tile_x) - 0.5 * (double)full_width;
    const double gy = (double)(py + tile_y) - 0.5 * (double)full_height;
    const double u = cx + gx * dy;
    const double v = cy + gy * dy;

    double zx, zy, kx, ky;
    if (mode == 0) {            // Mandelbrot: z0 = 0, c = pixel
        zx = 0.0; zy = 0.0;
        kx = u;   ky = v;
    } else {                    // Julia: z0 = pixel, c = constant
        zx = u;   zy = v;
        kx = jx;  ky = jy;
    }

    // -------- INNER LOOP (edit here to change the formula) --------
    int i = 0;
    double zx2 = zx*zx, zy2 = zy*zy;
    while (zx2 + zy2 < escape_r2 && i < max_iter) {
        double new_zy = 2.0 * zx * zy + ky + eps_y;
        double new_zx = zx2 - zy2     + kx + eps_x;
        zx = new_zx; zy = new_zy;
        zx2 = zx * zx;
        zy2 = zy * zy;
        ++i;
    }
    // --------------------------------------------------------------

    float result;
    if (i >= max_iter) {
        result = -1.0f;         // in-set marker
    } else {
        // Smooth coloring: continuous iteration count.
        double mag2 = zx2 + zy2;
        double nu = log(log(mag2) * 0.5) / log(2.0);
        result = (float)((double)i + 1.0 - nu);
    }
    out[py * width + px] = result;
}
"""


_kernel = cp.RawKernel(_KERNEL_SRC, "fractal")


def launch(out_tile, *, width, height, cx, cy, scale, max_iter, escape_r2,
           mode, jx, jy, tile_x, tile_y, full_width, full_height,
           eps_x=0.0, eps_y=0.0):
    """Launch the fractal kernel on a CuPy float32 buffer."""
    block = (16, 16, 1)
    grid = ((width + 15) // 16, (height + 15) // 16, 1)
    _kernel(
        grid, block,
        (out_tile, cp.int32(width), cp.int32(height),
         cp.float64(cx), cp.float64(cy), cp.float64(scale),
         cp.int32(max_iter), cp.float64(escape_r2),
         cp.int32(mode), cp.float64(jx), cp.float64(jy),
         cp.int32(tile_x), cp.int32(tile_y),
         cp.int32(full_width), cp.int32(full_height),
         cp.float64(eps_x), cp.float64(eps_y)),
    )

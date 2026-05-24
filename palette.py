"""Palette / colormap utilities.

Builds a GPU-resident lookup table from any matplotlib colormap and
applies it to a buffer of smooth iteration counts.
"""
import cupy as cp
import numpy as np
import matplotlib.pyplot as plt


LUT_SIZE = 4096


def build_lut(name: str, bit_depth: int = 8) -> cp.ndarray:
    """Return a CuPy LUT of shape (LUT_SIZE, 3), dtype matching bit depth."""
    cmap = plt.get_cmap(name)
    xs = np.linspace(0.0, 1.0, LUT_SIZE)
    rgba = cmap(xs)  # (LUT_SIZE, 4) float64 in [0,1]
    rgb = rgba[:, :3]
    if bit_depth == 16:
        arr = np.clip(rgb * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
    else:
        arr = np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cp.asarray(arr)


_APPLY_SRC = r"""
extern "C" __global__
void apply_lut(
    const float* __restrict__ smooth,   // [N], negative = in-set
    const T*     __restrict__ lut,      // [LUT_SIZE * 3]
    T*           __restrict__ rgb,      // [N * 3]
    const int N,
    const int lut_size,
    const float log_scale,              // 1.0 / log(max_iter)
    const T in_set_r, const T in_set_g, const T in_set_b
){
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float s = smooth[i];
    T r, g, b;
    if (s < 0.0f) {
        r = in_set_r; g = in_set_g; b = in_set_b;
    } else {
        // log-compress so deep zooms still have contrast
        float t = logf(s + 1.0f) * log_scale;
        if (t < 0.0f) t = 0.0f;
        if (t > 1.0f) t = 1.0f;
        int idx = (int)(t * (lut_size - 1));
        r = lut[idx * 3 + 0];
        g = lut[idx * 3 + 1];
        b = lut[idx * 3 + 2];
    }
    rgb[i * 3 + 0] = r;
    rgb[i * 3 + 1] = g;
    rgb[i * 3 + 2] = b;
}
"""


def _make_kernel(dtype):
    ctype = "unsigned short" if dtype == cp.uint16 else "unsigned char"
    src = _APPLY_SRC.replace("T", ctype)
    return cp.RawKernel(src, "apply_lut")


_kernel_cache: dict = {}


def apply(smooth: cp.ndarray, lut: cp.ndarray, max_iter: int,
          in_set=(0, 0, 0)) -> cp.ndarray:
    """Apply LUT to a (H, W) float32 smooth-iter buffer -> (H, W, 3) image."""
    dtype = lut.dtype
    if dtype not in _kernel_cache:
        _kernel_cache[dtype] = _make_kernel(dtype)
    kern = _kernel_cache[dtype]

    h, w = smooth.shape
    n = h * w
    rgb = cp.empty((h, w, 3), dtype=dtype)
    block = (256, 1, 1)
    grid = ((n + 255) // 256, 1, 1)
    log_scale = 1.0 / float(np.log(max(2, max_iter)))
    kern(
        grid, block,
        (smooth.ravel(), lut.ravel(), rgb.ravel(),
         cp.int32(n), cp.int32(LUT_SIZE), cp.float32(log_scale),
         dtype.type(in_set[0]), dtype.type(in_set[1]), dtype.type(in_set[2])),
    )
    return rgb

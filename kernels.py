"""CUDA kernels for fractal rendering.

Families supported (one CUDA kernel each, dispatched by `formula`):

  polynomial family  -> kernel `fractal`
    - mandelbrot  : z_{n+1} = z^2 + c              (d=2, no flags)
    - multibrot   : z_{n+1} = z^d + c              (any d>=2)
    - tricorn     : z_{n+1} = conj(z)^d + c        (FLAG_CONJ)
    - burning_ship: z_{n+1} = (|Re z| + i|Im z|)^d + c (FLAG_ABS)
    - phoenix     : z_{n+1} = z^2 + c + p*z_{n-1}  (FLAG_PHOENIX, d ignored)
    - magnet1     : z_{n+1} = ((z^2+c-1)/(2z+c-2))^2 (FLAG_MAGNET, d ignored)
    All of the above support --mode {mandelbrot,julia} (parameter vs dynamical).

  newton family      -> kernel `newton`
    polynomial is z^d - 1 (degree set via --power, default 3).
    Newton:  z' = z - f/f'
    Nova:    z' = z - R*f/f' + c       (c = pixel or constant)

  lyapunov           -> kernel `lyapunov`
    pixel = (rA, rB), iterate logistic x*r*(1-x) with r alternating per
    a repeating A/B sequence string. Output = Lyapunov exponent.

  buddhabrot         -> kernel `buddhabrot`
    Samples random c, iterates z^2+c. For escaping orbits, atomic-adds
    each visited point into a uint32 image-space histogram.
"""
import cupy as cp


# Flag bits for the polynomial-family kernel.
FLAG_CONJ    = 1 << 0   # tricorn / mandelbar
FLAG_ABS     = 1 << 1   # burning ship
FLAG_PHOENIX = 1 << 2   # phoenix
FLAG_MAGNET  = 1 << 3   # magnet 1


_POLY_SRC = r"""
__device__ inline unsigned long long splitmix64(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}
__device__ inline unsigned long long lcg64(unsigned long long *s) {
    *s = (*s) * 6364136223846793005ULL + 1442695040888963407ULL;
    return *s;
}
__device__ inline void rand2f_signed(unsigned long long *s,
                                     float *a, float *b) {
    unsigned long long r = lcg64(s);
    unsigned int hi = (unsigned int)(r >> 40);
    unsigned int md = (unsigned int)((r >> 16) & 0xFFFFFFu);
    const float inv = 1.0f / 8388608.0f;
    *a = (float)hi * inv - 1.0f;
    *b = (float)md * inv - 1.0f;
}

// w = z^d  (d >= 1) by repeated complex multiplication.
__device__ inline void cpow_int(double zx, double zy, int d,
                                double *wx, double *wy) {
    double rx = zx, ry = zy;
    for (int k = 1; k < d; ++k) {
        double nx = rx * zx - ry * zy;
        double ny = rx * zy + ry * zx;
        rx = nx; ry = ny;
    }
    *wx = rx; *wy = ry;
}

extern "C" __global__
void fractal(
    float* __restrict__ out,
    const int   width,
    const int   height,
    const double cx, const double cy,
    const double scale,
    const int   max_iter,
    const double escape_r2,
    const int   mode,
    const double jx, const double jy,
    const int   tile_x, const int   tile_y,
    const int   full_width, const int   full_height,
    const double eps_x, const double eps_y,
    const unsigned long long seed,
    const int   power,
    const int   flags,
    const double phx, const double phy
){
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    const double dy = scale / (double)full_height;
    const double gx = (double)(px + tile_x) - 0.5 * (double)full_width;
    const double gy = (double)(py + tile_y) - 0.5 * (double)full_height;
    const double u = cx + gx * dy;
    const double v = cy + gy * dy;

    double zx, zy, kx, ky;
    if (mode == 0) { zx = 0.0; zy = 0.0; kx = u; ky = v; }
    else           { zx = u;   zy = v;   kx = jx; ky = jy; }

    unsigned long long gx_i = (unsigned long long)(px + tile_x);
    unsigned long long gy_i = (unsigned long long)(py + tile_y);
    unsigned long long rng = splitmix64(seed
        ^ (gx_i * 0xD2B74407B1CE6E93ULL)
        ^ (gy_i * 0x9E3779B97F4A7C15ULL));
    const bool use_eps = (eps_x != 0.0) || (eps_y != 0.0);

    const bool f_conj    = (flags & 1) != 0;
    const bool f_abs     = (flags & 2) != 0;
    const bool f_phoenix = (flags & 4) != 0;
    const bool f_magnet  = (flags & 8) != 0;

    double zpx = 0.0, zpy = 0.0;
    int i = 0;
    double mag2 = zx*zx + zy*zy;
    bool converged = false;
    while (mag2 < escape_r2 && i < max_iter) {
        double rx = 0.0, ry = 0.0;
        if (use_eps) {
            float a, b; rand2f_signed(&rng, &a, &b);
            rx = (double)a * eps_x;
            ry = (double)b * eps_y;
        }
        double tx = zx, ty = zy;
        if (f_abs)  { tx = fabs(tx); ty = fabs(ty); }
        if (f_conj) { ty = -ty; }

        double nx, ny;
        if (f_magnet) {
            double n_re = tx*tx - ty*ty + kx - 1.0 + rx;
            double n_im = 2.0*tx*ty       + ky       + ry;
            double d_re = 2.0*tx + kx - 2.0 + rx;
            double d_im = 2.0*ty + ky       + ry;
            double denom = d_re*d_re + d_im*d_im;
            if (denom < 1e-300) { converged = true; break; }
            double qx = (n_re*d_re + n_im*d_im) / denom;
            double qy = (n_im*d_re - n_re*d_im) / denom;
            nx = qx*qx - qy*qy;
            ny = 2.0*qx*qy;
            double ddx = nx - 1.0, ddy = ny;
            if (ddx*ddx + ddy*ddy < 1e-6) {
                converged = true; zx = nx; zy = ny; ++i; break;
            }
        } else if (f_phoenix) {
            double sqx = tx*tx - ty*ty;
            double sqy = 2.0*tx*ty;
            double pzx = phx*zpx - phy*zpy;
            double pzy = phx*zpy + phy*zpx;
            nx = sqx + kx + rx + pzx;
            ny = sqy + ky + ry + pzy;
            zpx = zx; zpy = zy;
        } else {
            double wx, wy;
            cpow_int(tx, ty, power, &wx, &wy);
            nx = wx + kx + rx;
            ny = wy + ky + ry;
        }
        zx = nx; zy = ny;
        mag2 = zx*zx + zy*zy;
        ++i;
    }

    float result;
    if (converged || i >= max_iter) {
        result = -1.0f;
    } else {
        double pwr = (double)((power < 2) ? 2 : power);
        double lr = log(sqrt(mag2));
        double lR = log(sqrt(escape_r2));
        double nu = log(lr / lR) / log(pwr);
        result = (float)((double)i + 1.0 - nu);
        if (!isfinite(result)) result = (float)i;
    }
    out[py * width + px] = result;
}
"""

_poly_kernel = cp.RawKernel(_POLY_SRC, "fractal")


def launch(out_tile, *, width, height, cx, cy, scale, max_iter, escape_r2,
           mode, jx, jy, tile_x, tile_y, full_width, full_height,
           eps_x=0.0, eps_y=0.0, seed=0,
           power=2, flags=0, phx=0.0, phy=0.0):
    """Launch the polynomial-family fractal kernel."""
    block = (16, 16, 1)
    grid = ((width + 15) // 16, (height + 15) // 16, 1)
    _poly_kernel(
        grid, block,
        (out_tile, cp.int32(width), cp.int32(height),
         cp.float64(cx), cp.float64(cy), cp.float64(scale),
         cp.int32(max_iter), cp.float64(escape_r2),
         cp.int32(mode), cp.float64(jx), cp.float64(jy),
         cp.int32(tile_x), cp.int32(tile_y),
         cp.int32(full_width), cp.int32(full_height),
         cp.float64(eps_x), cp.float64(eps_y),
         cp.uint64(seed),
         cp.int32(power), cp.int32(flags),
         cp.float64(phx), cp.float64(phy)),
    )


# ---------------------------------------------------------------------------
# Newton / Nova kernel
# ---------------------------------------------------------------------------
# Output encoding: a single float per pixel.
#   -1.0           -> did not converge to any root (paint as "in-set")
#   root_idx * 10000 + smooth_iter
# The renderer decodes by dividing/modding when palettizing.

_NEWTON_SRC = r"""
__device__ inline void cpow_int_n(double zx, double zy, int d,
                                  double *wx, double *wy) {
    double rx = zx, ry = zy;
    for (int k = 1; k < d; ++k) {
        double nx = rx * zx - ry * zy;
        double ny = rx * zy + ry * zx;
        rx = nx; ry = ny;
    }
    *wx = rx; *wy = ry;
}

extern "C" __global__
void newton(
    float* __restrict__ out,
    const int   width,
    const int   height,
    const double cx, const double cy,
    const double scale,
    const int   max_iter,
    const int   tile_x, const int   tile_y,
    const int   full_width, const int   full_height,
    const int   power,
    const double relax_x, const double relax_y,
    const int   nova_mode,
    const int   c_mode,
    const double jx, const double jy
){
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    const double dy = scale / (double)full_height;
    const double gx = (double)(px + tile_x) - 0.5 * (double)full_width;
    const double gy = (double)(py + tile_y) - 0.5 * (double)full_height;
    const double u = cx + gx * dy;
    const double v = cy + gy * dy;

    double zx = u, zy = v;
    double kx = 0.0, ky = 0.0;
    if (nova_mode != 0) {
        if (c_mode == 0) { kx = u; ky = v; }
        else             { kx = jx; ky = jy; }
    }
    // Nova traditionally starts z at 1, not at the pixel. Pure Newton
    // starts z at the pixel.
    if (nova_mode != 0) { zx = 1.0; zy = 0.0; }

    const double TOL = 1e-6;
    int i;
    for (i = 0; i < max_iter; ++i) {
        double zd_x, zd_y;
        cpow_int_n(zx, zy, power, &zd_x, &zd_y);
        double f_re = zd_x - 1.0;
        double f_im = zd_y;
        double zdm_x, zdm_y;
        cpow_int_n(zx, zy, power - 1, &zdm_x, &zdm_y);
        double df_re = (double)power * zdm_x;
        double df_im = (double)power * zdm_y;
        double denom = df_re * df_re + df_im * df_im;
        if (denom < 1e-300) break;
        double q_re = (f_re * df_re + f_im * df_im) / denom;
        double q_im = (f_im * df_re - f_re * df_im) / denom;
        double rq_re = relax_x * q_re - relax_y * q_im;
        double rq_im = relax_x * q_im + relax_y * q_re;
        double nx = zx - rq_re;
        double ny = zy - rq_im;
        if (nova_mode != 0) { nx += kx; ny += ky; }
        double dxv = nx - zx, dyv = ny - zy;
        double step2 = dxv * dxv + dyv * dyv;
        zx = nx; zy = ny;
        if (step2 < TOL * TOL) break;
        if (zx*zx + zy*zy > 1e12) break;
    }

    double ang = atan2(zy, zx);
    if (ang < 0.0) ang += 6.283185307179586;
    double k_d = ang * (double)power / 6.283185307179586;
    int root_idx = ((int)(k_d + 0.5)) % power;
    if (root_idx < 0) root_idx += power;
    double r_x = cos(6.283185307179586 * (double)root_idx / (double)power);
    double r_y = sin(6.283185307179586 * (double)root_idx / (double)power);
    double dist2 = (zx - r_x)*(zx - r_x) + (zy - r_y)*(zy - r_y);

    float result;
    if (!isfinite(zx) || !isfinite(zy)) {
        result = -1.0f;
    } else if (nova_mode == 0) {
        // Pure Newton: strict basin classification. Off-basin = in-set black.
        if (dist2 > 0.25 || i >= max_iter) {
            result = -1.0f;
        } else {
            result = (float)(root_idx * 10000) + (float)i;
        }
    } else {
        // Nova: even pixels that don't settle near a root still get colored
        // by their nearest-root hue at low brightness (i = max_iter).
        int smooth_i = (dist2 > 0.25 || i >= max_iter) ? max_iter : i;
        result = (float)(root_idx * 10000) + (float)smooth_i;
    }
    out[py * width + px] = result;
}
"""

_newton_kernel = cp.RawKernel(_NEWTON_SRC, "newton")


def launch_newton(out_tile, *, width, height, cx, cy, scale, max_iter,
                  tile_x, tile_y, full_width, full_height,
                  power=3, relax=(1.0, 0.0),
                  nova_mode=0, c_mode=0, jx=0.0, jy=0.0):
    block = (16, 16, 1)
    grid = ((width + 15) // 16, (height + 15) // 16, 1)
    _newton_kernel(
        grid, block,
        (out_tile, cp.int32(width), cp.int32(height),
         cp.float64(cx), cp.float64(cy), cp.float64(scale),
         cp.int32(max_iter),
         cp.int32(tile_x), cp.int32(tile_y),
         cp.int32(full_width), cp.int32(full_height),
         cp.int32(power),
         cp.float64(relax[0]), cp.float64(relax[1]),
         cp.int32(nova_mode), cp.int32(c_mode),
         cp.float64(jx), cp.float64(jy)),
    )


# ---------------------------------------------------------------------------
# Lyapunov fractal
# ---------------------------------------------------------------------------

_LYAP_SRC = r"""
extern "C" __global__
void lyapunov(
    float* __restrict__ out,
    const int   width,
    const int   height,
    const double cx, const double cy,
    const double scale,
    const int   max_iter,
    const int   tile_x, const int   tile_y,
    const int   full_width, const int   full_height,
    const unsigned char* __restrict__ pattern,
    const int   pat_len,
    const int   warmup,
    const double x0
){
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= width || py >= height) return;

    const double dy = scale / (double)full_height;
    const double gx = (double)(px + tile_x) - 0.5 * (double)full_width;
    const double gy = (double)(py + tile_y) - 0.5 * (double)full_height;
    const double rA = cx + gx * dy;
    const double rB = cy + gy * dy;

    double x = x0;
    for (int i = 0; i < warmup; ++i) {
        double r = (pattern[i % pat_len] == 0) ? rA : rB;
        x = r * x * (1.0 - x);
    }
    double sum = 0.0;
    int count = 0;
    for (int i = 0; i < max_iter; ++i) {
        double r = (pattern[i % pat_len] == 0) ? rA : rB;
        double deriv = r * (1.0 - 2.0 * x);
        double a = fabs(deriv);
        if (a > 1e-300) {
            sum += log(a);
            count += 1;
        }
        x = r * x * (1.0 - x);
        if (!isfinite(x)) { count = 0; break; }
    }
    float L = (count == 0) ? 0.0f : (float)(sum / (double)count);
    out[py * width + px] = L;
}
"""

_lyap_kernel = cp.RawKernel(_LYAP_SRC, "lyapunov")


def launch_lyapunov(out_tile, *, width, height, cx, cy, scale, max_iter,
                    tile_x, tile_y, full_width, full_height,
                    pattern_bytes, warmup=200, x0=0.5):
    block = (16, 16, 1)
    grid = ((width + 15) // 16, (height + 15) // 16, 1)
    _lyap_kernel(
        grid, block,
        (out_tile, cp.int32(width), cp.int32(height),
         cp.float64(cx), cp.float64(cy), cp.float64(scale),
         cp.int32(max_iter),
         cp.int32(tile_x), cp.int32(tile_y),
         cp.int32(full_width), cp.int32(full_height),
         pattern_bytes, cp.int32(int(pattern_bytes.size)),
         cp.int32(warmup), cp.float64(x0)),
    )


# ---------------------------------------------------------------------------
# Buddhabrot accumulator
# ---------------------------------------------------------------------------

_BUDDHA_SRC = r"""
__device__ inline unsigned long long _bud_lcg(unsigned long long *s) {
    *s = (*s) * 6364136223846793005ULL + 1442695040888963407ULL;
    return *s;
}
__device__ inline double _bud_rand_d(unsigned long long *s) {
    return (double)(_bud_lcg(s) >> 11) * (1.0 / 9007199254740992.0);
}
__device__ inline unsigned long long _bud_smix(unsigned long long x) {
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

// Integer power for double-precision complex: (zx + i*zy)^p.
__device__ inline void _bud_cpow(double zx, double zy, int p,
                                 double *ox, double *oy) {
    double rx = 1.0, ry = 0.0;   // accumulate r = z^p
    double bx = zx, by = zy;
    int e = p;
    while (e > 0) {
        if (e & 1) {
            double nx = rx*bx - ry*by;
            double ny = rx*by + ry*bx;
            rx = nx; ry = ny;
        }
        e >>= 1;
        if (e) {
            double nx = bx*bx - by*by;
            double ny = 2.0*bx*by;
            bx = nx; by = ny;
        }
    }
    *ox = rx; *oy = ry;
}

// flags: bit0=CONJ, bit1=ABS, bit2=PHOENIX (z_new = z^2 + c + p*z_prev),
//        bit3=MAGNET  (z_new = ((z^2 + c - 1)/(2z + c - 2))^2)
// julia_mode: 0 = Mandelbrot-style (sample c, z0=0)
//             1 = Julia-style (fix c=(jx,jy), sample initial z0)
extern "C" __global__
void buddhabrot(
    unsigned int* __restrict__ hist,
    const int   width, const int   height,
    const double cx, const double cy,
    const double scale,
    const int   max_iter,
    const int   min_iter,
    const double sample_half_w,
    const double sample_half_h,
    const double sample_cx,
    const double sample_cy,
    const unsigned long long seed,
    const int   n_samples,
    const int   power,
    const int   flags,
    const int   julia_mode,
    const double jx,
    const double jy,
    const double bailout2,
    const double phx,
    const double phy
){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_samples) return;

    unsigned long long rng = _bud_smix(seed
        ^ ((unsigned long long)idx * 0xD2B74407B1CE6E93ULL));
    double rxs = sample_cx + (2.0 * _bud_rand_d(&rng) - 1.0) * sample_half_w;
    double rys = sample_cy + (2.0 * _bud_rand_d(&rng) - 1.0) * sample_half_h;

    double crx, cry, zx, zy;
    if (julia_mode) {
        crx = jx; cry = jy;
        zx = rxs; zy = rys;
    } else {
        crx = rxs; cry = rys;
        zx = 0.0; zy = 0.0;
    }
    double zprev_x = 0.0, zprev_y = 0.0;

    int i;
    for (i = 0; i < max_iter; ++i) {
        double tx = zx, ty = zy;
        if (flags & 2) { tx = fabs(tx); ty = fabs(ty); }
        if (flags & 1) { ty = -ty; }
        double nx, ny;
        if (flags & 8) {
            // Magnet: ((z^2 + c - 1) / (2z + c - 2))^2
            double a_re = tx*tx - ty*ty + crx - 1.0;
            double a_im = 2.0*tx*ty       + cry;
            double b_re = 2.0*tx + crx - 2.0;
            double b_im = 2.0*ty + cry;
            double denom = b_re*b_re + b_im*b_im + 1e-30;
            double q_re = (a_re*b_re + a_im*b_im) / denom;
            double q_im = (a_im*b_re - a_re*b_im) / denom;
            nx = q_re*q_re - q_im*q_im;
            ny = 2.0*q_re*q_im;
        } else if (power == 2) {
            nx = tx*tx - ty*ty + crx;
            ny = 2.0*tx*ty     + cry;
        } else {
            double px2, py2;
            _bud_cpow(tx, ty, power, &px2, &py2);
            nx = px2 + crx;
            ny = py2 + cry;
        }
        if (flags & 4) {
            // Phoenix: add p * z_prev
            nx += phx * zprev_x - phy * zprev_y;
            ny += phx * zprev_y + phy * zprev_x;
            zprev_x = zx; zprev_y = zy;
        }
        zx = nx; zy = ny;
        if (zx*zx + zy*zy > bailout2) break;
    }
    if (i >= max_iter || i < min_iter) return;

    const double dy = scale / (double)height;

    // Replay
    if (julia_mode) {
        zx = rxs; zy = rys;
    } else {
        zx = 0.0; zy = 0.0;
    }
    zprev_x = 0.0; zprev_y = 0.0;
    for (int k = 0; k < i; ++k) {
        double tx = zx, ty = zy;
        if (flags & 2) { tx = fabs(tx); ty = fabs(ty); }
        if (flags & 1) { ty = -ty; }
        double nx, ny;
        if (flags & 8) {
            double a_re = tx*tx - ty*ty + crx - 1.0;
            double a_im = 2.0*tx*ty       + cry;
            double b_re = 2.0*tx + crx - 2.0;
            double b_im = 2.0*ty + cry;
            double denom = b_re*b_re + b_im*b_im + 1e-30;
            double q_re = (a_re*b_re + a_im*b_im) / denom;
            double q_im = (a_im*b_re - a_re*b_im) / denom;
            nx = q_re*q_re - q_im*q_im;
            ny = 2.0*q_re*q_im;
        } else if (power == 2) {
            nx = tx*tx - ty*ty + crx;
            ny = 2.0*tx*ty     + cry;
        } else {
            double px2, py2;
            _bud_cpow(tx, ty, power, &px2, &py2);
            nx = px2 + crx;
            ny = py2 + cry;
        }
        if (flags & 4) {
            nx += phx * zprev_x - phy * zprev_y;
            ny += phx * zprev_y + phy * zprev_x;
            zprev_x = zx; zprev_y = zy;
        }
        zx = nx; zy = ny;
        double gxf = (zx - cx) / dy + 0.5 * (double)width;
        double gyf = (zy - cy) / dy + 0.5 * (double)height;
        int px2i = (int)gxf;
        int py2i = (int)gyf;
        if (px2i >= 0 && px2i < width && py2i >= 0 && py2i < height) {
            atomicAdd(&hist[py2i * width + px2i], 1u);
        }
    }
}
"""

_buddha_kernel = cp.RawKernel(_BUDDHA_SRC, "buddhabrot")


def launch_buddhabrot(hist, *, width, height, cx, cy, scale, max_iter,
                      min_iter, sample_half_w, sample_half_h,
                      sample_cx, sample_cy, seed, n_samples,
                      power=2, flags=0, julia_mode=0, jx=0.0, jy=0.0,
                      bailout2=None, phx=0.0, phy=0.0):
    block = (256, 1, 1)
    grid = ((n_samples + 255) // 256, 1, 1)
    if bailout2 is None:
        # Higher powers need a bigger bailout radius for the trajectory to
        # actually leave the captured region cleanly. Magnet escapes to
        # infinity slowly so it also benefits from a bigger bailout.
        if flags & 8:
            bailout2 = 100.0
        else:
            bailout2 = 4.0 if power <= 2 else 16.0 if power <= 4 else 64.0
    _buddha_kernel(
        grid, block,
        (hist, cp.int32(width), cp.int32(height),
         cp.float64(cx), cp.float64(cy), cp.float64(scale),
         cp.int32(max_iter), cp.int32(min_iter),
         cp.float64(sample_half_w), cp.float64(sample_half_h),
         cp.float64(sample_cx), cp.float64(sample_cy),
         cp.uint64(seed), cp.int32(n_samples),
         cp.int32(power), cp.int32(flags),
         cp.int32(julia_mode), cp.float64(jx), cp.float64(jy),
         cp.float64(bailout2),
         cp.float64(phx), cp.float64(phy)),
    )

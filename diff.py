"""Visualize the difference between two rendered TIFFs (or PNGs).

Usage:
    python diff.py A.tif B.tif out/diff.png [--gain 8]
"""
import sys
import numpy as np
import tifffile
from PIL import Image


def load(path):
    if path.lower().endswith((".tif", ".tiff")):
        return tifffile.imread(path)
    return np.array(Image.open(path))


def main():
    args = sys.argv[1:]
    gain = 8.0
    if "--gain" in args:
        i = args.index("--gain")
        gain = float(args[i + 1])
        del args[i:i + 2]
    a_path, b_path, out_path = args[:3]

    a = load(a_path).astype(np.int32)
    b = load(b_path).astype(np.int32)
    if a.shape != b.shape:
        raise SystemExit(f"shape mismatch {a.shape} vs {b.shape}")

    diff = np.abs(a - b)
    max_v = 65535 if a.dtype != np.uint8 else 255
    # Sum across channels for a single-channel magnitude
    mag = diff.sum(axis=-1).astype(np.float64) / (3.0 * max_v)
    mag = np.clip(mag * gain, 0.0, 1.0)
    img8 = (mag * 255 + 0.5).astype(np.uint8)
    nz = (diff.sum(axis=-1) > 0).sum()
    total = mag.size
    print(f"Differing pixels: {nz}/{total} ({100*nz/total:.2f}%)")
    print(f"Max abs diff: {diff.max()}  Mean abs diff: {diff.mean():.3f}")

    # Downscale for previewing if huge
    h, w = img8.shape
    if max(h, w) > 4096:
        from PIL import Image as I
        scale = 4096 / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        img8 = np.array(I.fromarray(img8).resize((nw, nh), I.LANCZOS))
        print(f"Resized preview to {nw}x{nh}")

    Image.fromarray(img8, mode="L").save(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

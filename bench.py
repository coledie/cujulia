"""Run a benchmark sweep across resolutions for both modes."""
import json
import re
import subprocess
import sys
import time
from pathlib import Path

SIZES = [1024, 2048, 4096, 8192, 16384]
MODES = ["mandelbrot", "julia"]
ITER = 1000
TILE = 2048

results = []
for mode in MODES:
    for n in SIZES:
        out = f"out/bench/{mode}_{n}.tif"
        Path("out/bench").mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "render.py",
            "--mode", mode,
            "--width", str(n), "--height", str(n),
            "--center", "-0.5" if mode == "mandelbrot" else "0", "0",
            "--scale", "3.0",
            "--iter", str(ITER),
            "--tile", str(TILE),
            "--bits", "16",
            "--format", "tiff",
            "-o", out,
        ]
        if mode == "julia":
            cmd += ["--jc", "-0.8", "0.156"]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        wall = time.time() - t0
        if r.returncode != 0:
            print(r.stdout); print(r.stderr); sys.exit(1)
        text = r.stdout
        # Parse "GPU render done in X.XXs" and "Render: X.XXs   Save: X.XXs"
        gpu = float(re.search(r"GPU render done in ([\d.]+)s", text).group(1))
        m = re.search(r"Render: ([\d.]+)s\s+Save: ([\d.]+)s", text)
        render_s = float(m.group(1)); save_s = float(m.group(2))
        size_mb = Path(out).stat().st_size / 1e6
        print(f"{mode:11s} {n:>6}x{n:<6} GPU={gpu:6.2f}s  render={render_s:6.2f}s  "
              f"save={save_s:6.2f}s  wall={wall:6.2f}s  size={size_mb:7.1f} MB")
        results.append(dict(mode=mode, size=n, gpu=gpu, render=render_s,
                            save=save_s, wall=wall, size_mb=size_mb))

Path("out/bench/results.json").write_text(json.dumps(results, indent=2))
print("\nWrote out/bench/results.json")

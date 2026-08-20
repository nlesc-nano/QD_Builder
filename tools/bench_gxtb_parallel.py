#!/usr/bin/env python3
"""Find the best (concurrent jobs x threads per job) for g-xTB on this machine.

Why a benchmark and not a rule of thumb
---------------------------------------

Two knobs compete for the same cores: how many g-xTB processes run at once,
and how many OpenMP threads each one gets.  Which combination wins depends on
the machine *and* on the size of the cluster being relaxed -- a 17-atom SCF is
too small to thread and is dominated by process startup and memory latency,
while a 35-atom one has more work to amortise those over.  Measured on a
16-core laptop, threading a single job gave 5.75 -> 4.25 s from 1 to 2 threads
and nothing after, while 8 concurrent single-threaded jobs gave 4.1x
throughput and 16 gave *less* than 8.  None of that transfers to an HPC node,
so measure it there.

The script relaxes real structures from a finished growth run, perturbed so
the optimiser actually does work (an already-converged input finishes in one
cycle and measures nothing), over a grid of (concurrency, threads) at several
system sizes, and reports the throughput-optimal and the core-efficient choice
for each size.

    python tools/bench_gxtb_parallel.py \\
      --pack-dir geometry_packs/cdse_cdcl2_zb \\
      --run      /scratch/.../growth_k1_to_k4_zb3 \\
      --bins 2,2 3,3 4,3 4,6 \\
      --cores 72 \\
      --concurrency 1,4,8,16,32,48,64,72 \\
      --threads 1,2,4 \\
      --output   /scratch/.../bench_gxtb

Only the g-xTB binary and PyYAML are needed; the builder package is not
imported, so this runs anywhere the binary does.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import yaml
except ImportError:  # pragma: no cover - yaml ships with the env
    yaml = None


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def read_xyz(path: Path) -> Tuple[List[str], List[List[float]]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    count = int(lines[0].split()[0])
    symbols: List[str] = []
    points: List[List[float]] = []
    for line in lines[2 : 2 + count]:
        parts = line.split()
        symbols.append(parts[0])
        points.append([float(x) for x in parts[1:4]])
    return symbols, points


def perturb(
    points: Sequence[Sequence[float]], amplitude: float, seed: int
) -> List[List[float]]:
    """Displace so the optimiser has real work to do.

    A relaxed input converges in one cycle and would measure process startup,
    not the optimisation the pipeline actually spends its time on.
    """

    rng = random.Random(seed)
    return [[x + rng.gauss(0.0, amplitude) for x in point] for point in points]


def write_case(directory: Path, symbols, points, max_steps: int) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    lines = [str(len(symbols)), "bench"]
    lines.extend(
        f"{s} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}" for s, p in zip(symbols, points)
    )
    (directory / "in.xyz").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if max_steps > 0:
        (directory / "xcontrol").write_text(
            f"$opt\n  maxcycle={max_steps}\n$end\n", encoding="utf-8"
        )


def pack_settings(pack_dir: Path) -> Dict[str, Any]:
    """Binary, max_steps and XTBPATH exactly as the pipeline would use them."""

    out = {"binary": "gxtb", "max_steps": 150, "charge": 0, "xtb_path": None}
    driver = pack_dir / "run_gxtb.yaml"
    if yaml is None or not driver.is_file():
        return out
    raw = yaml.safe_load(driver.read_text()) or {}
    relax = raw.get("relaxation") or {}
    out["binary"] = str(relax.get("binary", "gxtb"))
    out["max_steps"] = int(relax.get("max_steps", 150))
    out["charge"] = int(relax.get("charge", 0))
    if relax.get("xtb_path"):
        out["xtb_path"] = str(relax["xtb_path"])
    return out


def collect_structures(
    run_dir: Optional[Path], bins: Sequence[Tuple[int, int]], explicit: Sequence[Path]
) -> List[Tuple[str, int, List[str], List[List[float]]]]:
    """(label, n_atoms, symbols, coordinates) for each requested size."""

    cases: List[Tuple[str, int, List[str], List[List[float]]]] = []
    for path in explicit:
        symbols, points = read_xyz(path)
        cases.append((path.stem[:24], len(symbols), symbols, points))
    if run_dir is not None:
        for k, p in bins:
            directory = run_dir / f"k{k:03d}" / f"p{p:03d}"
            found = None
            for pattern in ("*_xtb.xyz", "*_offpath.xyz"):
                for candidate in sorted(directory.glob(pattern)):
                    if " 2." in candidate.name:  # rsync duplicate
                        continue
                    found = candidate
                    break
                if found is not None:
                    break
            if found is None:
                print(f"[bench] no structure for k{k}p{p} in {directory}", flush=True)
                continue
            symbols, points = read_xyz(found)
            cases.append((f"k{k}p{p}", len(symbols), symbols, points))
    cases.sort(key=lambda item: item[1])
    return cases


# ---------------------------------------------------------------------------
# the measurement
# ---------------------------------------------------------------------------


def run_batch(
    symbols,
    points,
    *,
    concurrency: int,
    threads: int,
    settings: Dict[str, Any],
    amplitude: float,
    tmp_root: Path,
) -> Tuple[float, int, float]:
    """Run ``concurrency`` optimisations at once; return (wall, ok, max_rss_mb)."""

    work = Path(tempfile.mkdtemp(prefix="bench_", dir=str(tmp_root)))
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    if settings.get("xtb_path"):
        env["XTBPATH"] = str(settings["xtb_path"])
    cmd = [str(settings["binary"]), "in.xyz", "--gxtb", "--opt"]
    if int(settings.get("charge", 0)):
        cmd += ["--chrg", str(int(settings["charge"]))]
    if int(settings["max_steps"]) > 0:
        cmd += ["--input", "xcontrol"]
    if threads > 1:
        cmd += ["-P", str(threads)]

    for index in range(concurrency):
        write_case(
            work / f"job{index}",
            symbols,
            # a different displacement per job, so they do not all take an
            # identical optimisation path and share cache in an unrealistic way
            perturb(points, amplitude, seed=1000 + index),
            int(settings["max_steps"]),
        )

    def one(index: int) -> int:
        directory = work / f"job{index}"
        subprocess.run(
            cmd,
            cwd=directory,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        # Success here means "the optimisation ran and produced a geometry",
        # which is what we are timing -- not "converged".  g-xTB writes
        # xtbopt.xyz on convergence and xtblast.xyz + a NOT_CONVERGED marker
        # when it exhausts maxcycle, exiting non-zero in the latter case.  A
        # non-converged run did *more* work, not less, so it is a valid timing
        # point; only a run that produced neither file is a real failure.
        return int(
            (directory / "xtbopt.xyz").is_file()
            or (directory / "xtblast.xyz").is_file()
        )

    before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        ok = sum(pool.map(one, range(concurrency)))
    wall = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # ru_maxrss is bytes on macOS and kilobytes on Linux
    scale = 1.0 / (1024.0 * 1024.0) if sys.platform == "darwin" else 1.0 / 1024.0
    shutil.rmtree(work, ignore_errors=True)
    return wall, ok, max(after, before) * scale


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--pack-dir", type=Path, required=True)
    ap.add_argument("--run", type=Path, default=None, help="finished growth run")
    ap.add_argument(
        "--bins",
        nargs="*",
        default=["2,2", "3,3", "4,3", "4,6"],
        help="k,p bins to take a reference structure from",
    )
    ap.add_argument("--structures", nargs="*", type=Path, default=[])
    ap.add_argument(
        "--cores",
        type=int,
        default=os.cpu_count() or 8,
        help="core budget; grid points with concurrency*threads above this are skipped",
    )
    ap.add_argument("--concurrency", default="1,2,4,8,16,32")
    ap.add_argument("--threads", default="1,2,4")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument(
        "--amplitude",
        type=float,
        default=0.25,
        help=(
            "displacement sigma in A.  0.25 gives a long optimisation that "
            "still converges within maxcycle; at 0.45 a 17-atom cluster "
            "exhausts 150 cycles and writes no xtbopt.xyz"
        ),
    )
    ap.add_argument("--tmpdir", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)

    settings = pack_settings(args.pack_dir)
    resolved = shutil.which(str(settings["binary"]))
    if resolved is None:
        raise SystemExit(f"g-xTB binary not on PATH: {settings['binary']}")
    bins = []
    for item in args.bins:
        try:
            k, p = item.split(",")
            bins.append((int(k), int(p)))
        except ValueError:
            raise SystemExit(f"--bins wants k,p pairs, got {item!r}")
    cases = collect_structures(args.run, bins, args.structures)
    if not cases:
        raise SystemExit("no reference structures found")

    concurrency = [int(x) for x in str(args.concurrency).split(",") if x.strip()]
    threads = [int(x) for x in str(args.threads).split(",") if x.strip()]
    tmp_root = args.tmpdir or Path(os.environ.get("TMPDIR", "/tmp"))
    tmp_root.mkdir(parents=True, exist_ok=True)

    print(f"[bench] binary   : {resolved}")
    print(f"[bench] max_steps: {settings['max_steps']}   charge: {settings['charge']}")
    print(f"[bench] cores    : {args.cores}   tmpdir: {tmp_root}")
    print(f"[bench] sizes    : " + ", ".join(f"{c[0]}({c[1]}at)" for c in cases))
    print(f"[bench] grid     : concurrency {concurrency} x threads {threads}\n")

    # Warm-up: first call pays for page cache and parameter loading.
    print("[bench] warm-up ...", flush=True)
    run_batch(
        cases[0][2], cases[0][3], concurrency=1, threads=1,
        settings=settings, amplitude=args.amplitude, tmp_root=tmp_root,
    )

    rows: List[Dict[str, Any]] = []
    for label, natoms, symbols, points in cases:
        print(f"\n=== {label}  ({natoms} atoms) ===")
        print(
            f"{'conc':>5} {'thr':>4} {'cores':>6} {'wall s':>8} "
            f"{'opt/s':>8} {'per-opt ms':>11} {'speedup':>8} {'per core':>9} {'RSS MB':>8}"
        )
        base: Optional[float] = None
        for thr in threads:
            for conc in concurrency:
                if conc * thr > args.cores:
                    continue
                walls = []
                rss = 0.0
                ok_total = 0
                for _ in range(max(1, args.repeats)):
                    wall, ok, mb = run_batch(
                        symbols, points, concurrency=conc, threads=thr,
                        settings=settings, amplitude=args.amplitude, tmp_root=tmp_root,
                    )
                    walls.append(wall)
                    rss = max(rss, mb)
                    ok_total += ok
                wall = statistics.median(walls)
                rate = conc / wall
                if base is None:
                    base = rate
                row = dict(
                    label=label, n_atoms=natoms, concurrency=conc, threads=thr,
                    cores=conc * thr, wall_s=wall, rate_per_s=rate,
                    per_opt_ms=1000.0 * wall / conc, speedup=rate / base,
                    per_core=rate / (conc * thr), rss_mb=rss,
                    ok=ok_total, attempted=conc * max(1, args.repeats),
                )
                rows.append(row)
                flag = "" if row["ok"] == row["attempted"] else "  <-- FAILURES"
                print(
                    f"{conc:5d} {thr:4d} {conc*thr:6d} {wall:8.2f} {rate:8.2f} "
                    f"{row['per_opt_ms']:11.0f} {row['speedup']:8.2f}x "
                    f"{row['per_core']:9.3f} {rss:8.0f}{flag}"
                )

    print("\n" + "=" * 78)
    print("RECOMMENDATION per system size")
    print(f"{'size':>8} {'atoms':>6} {'fastest (conc x thr)':>22} {'speedup':>9} "
          f"{'most core-efficient':>22}")
    best_overall: List[Tuple[int, int, int]] = []
    for label, natoms, _s, _p in cases:
        mine = [r for r in rows if r["label"] == label and r["ok"] == r["attempted"]]
        if not mine:
            continue
        fast = max(mine, key=lambda r: r["rate_per_s"])
        eff = max(mine, key=lambda r: r["per_core"])
        best_overall.append((natoms, fast["concurrency"], fast["threads"]))
        fast_txt = f"{fast['concurrency']} x {fast['threads']}"
        eff_txt = f"{eff['concurrency']} x {eff['threads']}"
        print(
            f"{label:>8} {natoms:6d} {fast_txt:>22} "
            f"{fast['speedup']:8.2f}x {eff_txt:>22}"
        )
    if best_overall:
        print(
            "\nIf the fastest concurrency rises with atom count, the plateau is "
            "memory/latency bound at small sizes and it is worth setting\n"
            "`relaxation.workers` per k-range rather than once for the whole run."
        )
        conc = statistics.median([b[1] for b in best_overall])
        thr = statistics.median([b[2] for b in best_overall])
        print(f"\nSuggested starting point:  workers = {int(conc)}, OMP_NUM_THREADS = {int(thr)}")

    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "bench.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )
        print(f"\n[bench] wrote {args.output / 'bench.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Estimate max shed / max redecoration p as a function of core size k.

This is a *map-design* calculator (chemical potentials come later).  It encodes
the ballpark rules discussed for continuous nucleation growth:

  * Surface scale for a compact / quasi-spherical particle:  ~ k^(2/3)
  * Stoichiometry: Cd_{k+p} Se_k Cl_{2p}  →  shed/add are excess-CdCl2 units (p)
  * Shed decreases (or stays mild) with k  (less complete shell memory loss)
  * Redecoration is NOT up to the CN capacity ceiling (3k for max_cn[Se]=4)
  * Local inventory: after shed s and monomer package p_m, the residual shell
    p-s stays bound and the local pool is M=s+p_m.  Therefore
        p_child ≤ (p-s) + M = p+p_m
    before the realistic surface-capacity bound p_surf(k).

Usage (from repo root)::

    python scripts/evaluate_shed_add_bounds.py
    python scripts/evaluate_shed_add_bounds.py --kmax 20 --alpha-surf 1.5
    python scripts/evaluate_shed_add_bounds.py --csv /tmp/shed_add.csv

No project import required — pure Python 3 / optional matplotlib.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Geometry / capacity
# ---------------------------------------------------------------------------


def k_to_the_two_thirds(k: float) -> float:
    return float(k) ** (2.0 / 3.0)


def p_surface_spherical(
    k: int,
    *,
    beta: float = 1.5,
    floor_min: int = 0,
) -> int:
    """Max realistic excess p on a compact quasi-spherical particle.

    N_surf ~ beta * k^(2/3).  beta ~ 1–3 is a tunable prefactor (facet packing,
    how many p units map onto one surface cation site).
    """

    if k <= 0:
        return 0
    return max(floor_min, int(math.floor(beta * k_to_the_two_thirds(k))))


def p_capacity_cn(k: int, max_cn_anion: int = 4) -> int:
    """Hard graph-style bound used in the nucleation code.

    capacity_cap = (max_cn[Se] - 1) * k  →  3k when max_cn[Se]=4.
    Safety ceiling only — not a physical decoration target.
    """

    return max(0, (max_cn_anion - 1) * int(k))


def p_max_equal_k(k: int) -> int:
    """Simple map window: p ≤ k."""

    return max(0, int(k))


# ---------------------------------------------------------------------------
# Shed laws  (should decrease or stay mild as k grows)
# ---------------------------------------------------------------------------


def s_max_power_decay(
    k: int,
    p_parent: int,
    *,
    s0: float = 8.0,
    nu: float = 0.5,
    s_floor: int = 1,
) -> int:
    """s_max = min(p, max(s_floor, floor(s0 / k^nu))).

    Early k: large shed (memory loss).  Large k: approaches s_floor.
    """

    if k <= 0 or p_parent <= 0:
        return 0
    raw = s0 / (float(k) ** nu)
    return int(min(p_parent, max(s_floor, math.floor(raw))))


def s_max_surface_fraction(
    k: int,
    p_parent: int,
    *,
    alpha: float = 1.0,
    beta_surf: float = 1.5,
) -> int:
    """s_max = min(p, floor(alpha * p_surf(k))).

    Shed at most a fraction/multiple of surface scale (not whole capacity).
    """

    if p_parent <= 0:
        return 0
    surf = p_surface_spherical(k, beta=beta_surf)
    return int(min(p_parent, max(0, math.floor(alpha * surf))))


def s_max_full(p_parent: int) -> int:
    """Map-complete: shed everything (s_max = p)."""

    return max(0, int(p_parent))


def s_max_fixed(p_parent: int, fixed: int) -> int:
    return int(min(p_parent, max(0, fixed)))


# ---------------------------------------------------------------------------
# Inventory redecoration
# ---------------------------------------------------------------------------


def inventory_pool(shed: int, p_m: int) -> int:
    """Local CdCl2 pool after shed s and monomer package p_m: M = s + p_m."""

    return max(0, int(shed) + int(p_m))


def p_child_max(
    *,
    parent_p: int,
    shed: int,
    p_m: int,
    k_child: int,
    beta_surf: float,
    use_surface_cap: bool = True,
    use_equal_k_cap: bool = False,
    use_cn_cap: bool = True,
    max_cn_anion: int = 4,
) -> int:
    """Max product p after redecoration for one (shed, p_m) channel.

    The unshed residual ``p-s`` remains attached and the local pool
    ``M=s+p_m`` may be re-adsorbed, hence ``p_child ≤ p+p_m`` before optional
    geometric caps.
    """

    m = inventory_pool(shed, p_m)
    residual = max(0, int(parent_p) - int(shed))
    caps = [residual + m]
    if use_surface_cap:
        caps.append(p_surface_spherical(k_child, beta=beta_surf))
    if use_equal_k_cap:
        caps.append(p_max_equal_k(k_child))
    if use_cn_cap:
        caps.append(p_capacity_cn(k_child, max_cn_anion=max_cn_anion))
    return int(min(caps)) if caps else 0


def p0_land(parent_p: int, shed: int, p_m: int) -> int:
    """Inject stoichiometry as in the code: p0 = p - s + p_m."""

    return max(0, int(parent_p) - int(shed) + int(p_m))


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str


SCENARIOS: Tuple[Scenario, ...] = (
    Scenario(
        "A_inventory_surf",
        "s_max = min(p, floor(α_shed·p_surf)); p_child≤min(p+p_m, p_surf)  [recommended ballpark]",
    ),
    Scenario(
        "B_decay_shed",
        "s_max = min(p, max(1, floor(s0/k^ν))); p_child≤min(p+p_m, p_surf)  [shed decreases with k]",
    ),
    Scenario(
        "C_full_shed_inventory",
        "s_max = p (full memory loss); p_child≤min(p+p_m, p_surf)  [aggressive map]",
    ),
    Scenario(
        "D_fixed_shed2",
        "s_max = min(p, 2); p_child≤min(p+p_m, p_surf)  [current-ish tight shed]",
    ),
    Scenario(
        "E_capacity_ceiling",
        "s_max = p; p_child≤3k  [UNPHYSICAL redecorate-to-ceiling; contrast only]",
    ),
)


def parent_p_default(k: int, beta_surf: float) -> int:
    """Assume parent is already surface-passivated: p ≈ p_surf(k)."""

    return max(0, p_surface_spherical(k, beta=beta_surf))


def evaluate_row(
    k: int,
    *,
    scenario: str,
    beta_surf: float,
    alpha_shed: float,
    s0: float,
    nu: float,
    p_m_list: Sequence[int],
    parent_p: Optional[int] = None,
) -> dict:
    k = int(k)
    k_child = k + 1
    p_par = int(parent_p) if parent_p is not None else parent_p_default(k, beta_surf)
    p_surf_k = p_surface_spherical(k, beta=beta_surf)
    p_surf_kp1 = p_surface_spherical(k_child, beta=beta_surf)
    p_cn = p_capacity_cn(k_child)
    p_eqk = p_max_equal_k(k_child)

    if scenario.startswith("A"):
        s_max = s_max_surface_fraction(
            k, p_par, alpha=alpha_shed, beta_surf=beta_surf
        )
        use_cn_for_child = False
    elif scenario.startswith("B"):
        s_max = s_max_power_decay(k, p_par, s0=s0, nu=nu, s_floor=1)
        use_cn_for_child = False
    elif scenario.startswith("C"):
        s_max = s_max_full(p_par)
        use_cn_for_child = False
    elif scenario.startswith("D"):
        s_max = s_max_fixed(p_par, 2)
        use_cn_for_child = False
    elif scenario.startswith("E"):
        s_max = s_max_full(p_par)
        use_cn_for_child = True
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    # Representative channels: max shed and mid shed, each with each p_m
    channels: List[dict] = []
    shed_values = sorted({0, max(0, s_max // 2), s_max})
    for s in shed_values:
        for p_m in p_m_list:
            p0 = p0_land(p_par, s, p_m)
            pool = inventory_pool(s, p_m)
            if scenario.startswith("E"):
                p_hi = p_cn
            else:
                p_hi = p_child_max(
                    parent_p=p_par,
                    shed=s,
                    p_m=p_m,
                    k_child=k_child,
                    beta_surf=beta_surf,
                    use_surface_cap=True,
                    use_equal_k_cap=False,
                    use_cn_cap=use_cn_for_child,
                )
            # Ladder "add" after inject: how many free-site +1 steps allowed
            delta_add = max(0, p_hi - p0)
            channels.append(
                {
                    "shed": s,
                    "p_m": p_m,
                    "p0": p0,
                    "pool_M": pool,
                    "p_child_max": p_hi,
                    "delta_add_max": delta_add,
                }
            )

    # Envelope over channels: max shed possible; max redecoration p possible
    max_p_child = max((c["p_child_max"] for c in channels), default=0)
    max_delta = max((c["delta_add_max"] for c in channels), default=0)

    return {
        "k": k,
        "k_child": k_child,
        "p_parent": p_par,
        "p_surf_k": p_surf_k,
        "p_surf_k+1": p_surf_kp1,
        "p_cap_cn_3k": p_cn,
        "p_cap_equal_k": p_eqk,
        "s_max": s_max,
        "p_child_max_envelope": max_p_child,
        "delta_add_max_envelope": max_delta,
        "channels": channels,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def print_scenario_table(
    rows: Sequence[dict],
    *,
    scenario_name: str,
    description: str,
    p_m_list: Sequence[int],
) -> None:
    print()
    print("=" * 88)
    print(f"Scenario {scenario_name}")
    print(description)
    print("=" * 88)
    hdr = (
        f"{'k':>4} {'p_par':>6} {'p_surf':>7} {'3k':>5} "
        f"{'s_max':>6} {'p_ch_max':>8} {'Δadd_max':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['k']:4d} {r['p_parent']:6d} {r['p_surf_k']:7d} "
            f"{r['p_cap_cn_3k']:5d} {r['s_max']:6d} "
            f"{r['p_child_max_envelope']:8d} {r['delta_add_max_envelope']:8d}"
        )

    # Detail for a few k values: all (shed, p_m) channels
    detail_ks = {rows[0]["k"], rows[len(rows) // 2]["k"], rows[-1]["k"]}
    print()
    print("  Channel detail: p0=p−s+p_m, p_child≤min((p−s)+M, caps)")
    for r in rows:
        if r["k"] not in detail_ks:
            continue
        print(
            f"  --- k={r['k']} → {r['k_child']}  "
            f"p_parent={r['p_parent']}  s_max={r['s_max']} ---"
        )
        print(
            f"      {'s':>3} {'p_m':>3} {'p0':>4} {'M':>4} "
            f"{'p_max':>5} {'Δadd':>5}"
        )
        for c in r["channels"]:
            print(
                f"      {c['shed']:3d} {c['p_m']:3d} {c['p0']:4d} "
                f"{c['pool_M']:4d} {c['p_child_max']:5d} {c['delta_add_max']:5d}"
            )


def write_csv(path: str, all_rows: Sequence[Tuple[str, dict]]) -> None:
    fieldnames = [
        "scenario",
        "k",
        "k_child",
        "p_parent",
        "p_surf_k",
        "p_surf_k+1",
        "p_cap_cn_3k",
        "p_cap_equal_k",
        "s_max",
        "p_child_max_envelope",
        "delta_add_max_envelope",
        "shed",
        "p_m",
        "p0",
        "pool_M",
        "p_child_max",
        "delta_add_max",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for scenario, row in all_rows:
            base = {k: row[k] for k in fieldnames if k in row and k != "scenario"}
            for ch in row["channels"]:
                rec = {"scenario": scenario, **base, **ch}
                w.writerow(rec)
    print(f"\nWrote {path}")


def try_plot(
    rows_by_scenario: dict,
    *,
    out_path: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skip plot", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    ax0, ax1 = axes
    for name, rows in rows_by_scenario.items():
        if name.startswith("E"):
            continue  # keep plot readable; capacity is huge
        ks = [r["k"] for r in rows]
        ax0.plot(ks, [r["s_max"] for r in rows], marker="o", label=name)
        ax1.plot(
            ks,
            [r["p_child_max_envelope"] for r in rows],
            marker="o",
            label=name,
        )
    # reference surface law on both
    if rows_by_scenario:
        any_rows = next(iter(rows_by_scenario.values()))
        ks = [r["k"] for r in any_rows]
        ax0.plot(
            ks,
            [r["p_surf_k"] for r in any_rows],
            "k--",
            alpha=0.5,
            label="p_surf(k)",
        )
        ax1.plot(
            ks,
            [r["p_surf_k+1"] for r in any_rows],
            "k--",
            alpha=0.5,
            label="p_surf(k+1)",
        )
        ax1.plot(
            ks,
            [r["p_cap_cn_3k"] for r in any_rows],
            "r:",
            alpha=0.4,
            label="CN cap 3k",
        )

    ax0.set_xlabel("k (parent)")
    ax0.set_ylabel("s_max (max shed)")
    ax0.set_title("Max shed vs k")
    ax0.legend(fontsize=8)
    ax0.grid(True, alpha=0.3)

    ax1.set_xlabel("k (parent)")
    ax1.set_ylabel("p_child max (envelope)")
    ax1.set_title("Max redecoration p vs k")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    fig.suptitle(
        "Nucleation map bounds (surface ~ k^{2/3}; residual shell + M=s+p_m)",
        fontsize=11,
    )
    fig.savefig(out_path, dpi=140)
    print(f"Wrote plot {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--kmin", type=int, default=1)
    p.add_argument("--kmax", type=int, default=16)
    p.add_argument(
        "--beta-surf",
        type=float,
        default=3.0,
        help="p_surf = floor(beta * k^(2/3))  [default 3.0 = scenario A]",
    )
    p.add_argument(
        "--alpha-shed",
        type=float,
        default=1.0,
        help="scenario A: s_max = min(p, floor(alpha * p_surf))  [default 1]",
    )
    p.add_argument(
        "--s0",
        type=float,
        default=8.0,
        help="scenario B: s_max ~ s0 / k^nu  [default 8]",
    )
    p.add_argument(
        "--nu",
        type=float,
        default=0.5,
        help="scenario B: decay exponent  [default 0.5]",
    )
    p.add_argument(
        "--p-m",
        type=str,
        default="0,1,2",
        help="monomer package p_m values, comma-separated  [default 0,1,2]",
    )
    p.add_argument(
        "--parent-p",
        type=str,
        default="surf",
        help="'surf' = p_parent=p_surf(k), or a fixed integer, or 'k' for p=k",
    )
    p.add_argument(
        "--scenarios",
        type=str,
        default="A,B,C,D,E",
        help="which scenarios to run (letters)  [default A,B,C,D,E]",
    )
    p.add_argument("--csv", type=str, default=None, help="write long-form CSV")
    p.add_argument(
        "--plot",
        type=str,
        default=None,
        help="write PNG comparison plot (needs matplotlib)",
    )
    args = p.parse_args(list(argv) if argv is not None else None)

    p_m_list = [int(x.strip()) for x in args.p_m.split(",") if x.strip()]
    want = {c.strip().upper() for c in args.scenarios.split(",") if c.strip()}
    ks = list(range(int(args.kmin), int(args.kmax) + 1))

    def resolve_parent_p(k: int) -> int:
        if args.parent_p == "surf":
            return parent_p_default(k, args.beta_surf)
        if args.parent_p in {"k", "equal_k"}:
            return p_max_equal_k(k)
        return int(args.parent_p)

    print("Assumptions")
    print(f"  particle: quasi-spherical  p_surf(k) = floor({args.beta_surf} * k^(2/3))")
    print(f"  parent p: {args.parent_p!r} (default surface-passivated)")
    print(f"  monomer packages p_m: {p_m_list}")
    print("  inventory: residual p−s remains; M=s+p_m can re-adsorb")
    print("             p_child ≤ min((p−s)+M, p_surf, …) = min(p+p_m, …)")
    print(f"  CN safety cap: p ≤ 3k (max_cn[Se]=4) — contrast only in scenario E")
    print()
    print("Note: chemical potentials are NOT applied; these are MAP BOUNDS only.")

    rows_by_scenario = {}
    csv_rows: List[Tuple[str, dict]] = []

    for sc in SCENARIOS:
        letter = sc.name[0]
        if letter not in want and sc.name not in want:
            continue
        rows = [
            evaluate_row(
                k,
                scenario=sc.name,
                beta_surf=args.beta_surf,
                alpha_shed=args.alpha_shed,
                s0=args.s0,
                nu=args.nu,
                p_m_list=p_m_list,
                parent_p=resolve_parent_p(k),
            )
            for k in ks
        ]
        rows_by_scenario[sc.name] = rows
        print_scenario_table(
            rows,
            scenario_name=sc.name,
            description=sc.description,
            p_m_list=p_m_list,
        )
        for r in rows:
            csv_rows.append((sc.name, r))

    # Side-by-side envelope summary
    if len(rows_by_scenario) > 1:
        print()
        print("=" * 88)
        print("Envelope summary: s_max / p_child_max by scenario")
        print("=" * 88)
        names = list(rows_by_scenario.keys())
        short = [n.split("_")[0] for n in names]
        hdr = f"{'k':>4}" + "".join(f" | {s:>10}" for s in short)
        print(hdr + "   (each cell: s_max / p_ch_max)")
        print("-" * len(hdr))
        for i, k in enumerate(ks):
            parts = [f"{k:4d}"]
            for name in names:
                r = rows_by_scenario[name][i]
                parts.append(f"{r['s_max']:3d}/{r['p_child_max_envelope']:<3d}")
            print(" | ".join(parts))

    if args.csv:
        write_csv(args.csv, csv_rows)
    if args.plot:
        try_plot(rows_by_scenario, out_path=args.plot)

    print()
    print("How to read")
    print("  s_max      — max p units you may shed at this k (map bound)")
    print("  p_ch_max   — max product p after redecoration (envelope over channels)")
    print("  Δadd_max   — max free-site ladder steps after inject (p_ch_max − p0)")
    print("  M = s+p_m  — re-adsorbable pool in addition to residual p−s")
    print("  Prefer scenarios A/B; E is the unphysical 'paint to 3k' contrast.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Read-only corpus normalization and descriptors shared with the pilot.

Construction graphs, relaxed connectivity and minimum geometries are separate
objects. Missing historical metadata is never interpreted as chemical failure.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import numpy as np

from .molecular_growth import (
    MinimumConsolidation,
    ParentStructure,
    parse_xyz,
    relaxed_minimum_similarity,
)
from .molecular_zb_growth import _occupation_shape_certificate
from .zb_motifs import _all_n_cycles


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str, allow_nan=False).encode()
    ).hexdigest()


def truth(value):
    return str(value).lower() in {"true", "1", "yes"}


def descriptors(symbols, edges, coordinates=None):
    graph = nx.Graph()
    graph.add_nodes_from((i, {"element": s}) for i, s in enumerate(symbols))
    graph.add_edges_from(
        (int(a), int(b))
        for a, b in edges
        if a != b and 0 <= a < len(symbols) and 0 <= b < len(symbols)
    )
    core = graph.copy()
    core.remove_nodes_from(i for i, s in enumerate(symbols) if s == "Cl")
    core.remove_edges_from(
        (a, b) for a, b in list(core.edges) if symbols[a] == symbols[b]
    )
    cd, se, cl = Counter(), Counter(), Counter()
    for i, s in enumerate(symbols):
        ns = Counter(symbols[j] for j in graph[i])
        if s == "Cd":
            cd[f"{ns['Se']},{ns['Cl']}"] += 1
        if s == "Se":
            se[str(ns["Cd"])] += 1
        if s == "Cl":
            cl[str(ns["Cd"])] += 1
    out = dict(
        cd_environments=dict(cd),
        se_cn=dict(se),
        cl_cn=dict(cl),
        terminal=cl["1"],
        mu2=cl["2"],
        mu3=cl["3"],
        bridge_deficit=symbols.count("Cl") - cl["2"],
        n_components=nx.number_connected_components(graph) if len(graph) else 0,
        core_components=nx.number_connected_components(core) if len(core) else 0,
        n4=len(_all_n_cycles(core, 4)),
        n6=len(_all_n_cycles(core, 6)),
        core_hash=nx.weisfeiler_lehman_graph_hash(core, node_attr="element"),
        graph_hash=nx.weisfeiler_lehman_graph_hash(graph, node_attr="element"),
    )
    if coordinates is not None and len(coordinates) == len(symbols):
        xyz = np.asarray(coordinates, float)
        pts = xyz[[i for i, s in enumerate(symbols) if s != "Cl"]]
        core_symbols = [s for s in symbols if s != "Cl"]
        out["core_geometry_hash"] = digest(
            sorted(
                (
                    tuple(sorted((core_symbols[a], core_symbols[b]))),
                    int(round(float(np.linalg.norm(pts[a] - pts[b])) / 0.05)),
                )
                for a in range(len(pts))
                for b in range(a)
            )
        )
        out["radius_A"] = float(
            np.sqrt(np.mean(np.sum((pts - pts.mean(0)) ** 2, axis=1)))
        )
        q4 = []
        for i in core:
            ns = list(core[i])
            if len(ns) != 4:
                continue
            vectors = xyz[ns] - xyz[i]
            lengths = np.linalg.norm(vectors, axis=1)
            if np.any(lengths < 1e-8):
                continue
            v = vectors / lengths[:, None]
            q4.append(
                float(
                    1
                    - 3
                    / 8
                    * sum(
                        (v[a] @ v[b] + 1 / 3) ** 2 for a in range(4) for b in range(a)
                    )
                )
            )
        out["tetrahedral_q4"] = q4
    return out


def corrected_occupation(record, tolerance=0.2):
    """New identity plus legacy alias; never alter the caller's record."""
    o = dict(record)
    cert = _occupation_shape_certificate(
        o["symbols"], np.asarray(o["lattice_coordinates"]), tolerance
    )
    new = f"zb_k{o['k']:03d}_p{o['p']:03d}_{cert}"
    aliases = set(o.get("identity_aliases", []))
    if o.get("occupation_id") and o["occupation_id"] != new:
        aliases.add(o["occupation_id"])
    o.update(occupation_id=new, identity_version=3, identity_aliases=sorted(aliases))
    return o


def iter_corpus(root):
    """Yield one normalized manifest row per run/id; exclude filesystem copies."""
    root = Path(root)
    runs = (
        [root]
        if (root / "zb_occupations.jsonl").is_file()
        else sorted(
            p
            for p in root.glob("growth_*")
            if p.is_dir() and not p.name.endswith((" 2", " 3"))
        )
    )
    for run in runs:
        manifest = run / "zb_occupations.jsonl"
        if not manifest.is_file():
            continue
        snapshot = run / "protocol.json"
        protocol = (
            json.loads(snapshot.read_text()).get("fingerprint")
            if snapshot.exists()
            else None
        )
        seen = set()
        with manifest.open() as fh:
            for line_number, line in enumerate(fh, 1):
                try:
                    r = json.loads(line)
                except ValueError:
                    continue  # interrupted trailing row
                o = r.get("occupation") or {}
                sid = r.get("structure_id")
                if not sid or sid in seen or not o.get("symbols"):
                    continue
                seen.add(sid)
                sy = list(o["symbols"]) + ["Cl"] * (2 * int(o["p"]))
                energy = r.get("energy_eV")
                if energy is not None and not np.isfinite(energy):
                    energy = None
                violations = list(r.get("violations") or [])
                converged = truth(r.get("xtb_converged", False))
                clean = (
                    truth(r["chemically_ok"])
                    if "chemically_ok" in r
                    else (not violations and "violations" in r)
                )
                xyz = run / str(r.get("xyz", "missing"))
                row = dict(
                    run=run.name,
                    structure_id=sid,
                    k=int(o["k"]),
                    p=int(o["p"]),
                    energy_eV=energy,
                    converged=converged,
                    chemical_clean=clean,
                    accepted=clean and converged and energy is not None,
                    protocol=protocol or f"unknown:{run.name}",
                    protocol_verified=protocol is not None,
                    violations=violations,
                    xyz=str(xyz),
                    symbols=sy,
                    edges=r.get("final_edges", []),
                    source_edges=r.get("source_edges", []),
                    occupation=corrected_occupation(o),
                    connectivity_preserved=r.get("topology_status") == "preserved",
                    lineage_parents=r.get("parent_structure_ids", []),
                    source=f"{manifest}:{line_number}",
                )
                row["final"] = descriptors(sy, row["edges"])
                row["constructed"] = (
                    descriptors(sy, row["source_edges"])
                    if row["source_edges"]
                    else None
                )
                yield row


def as_parent(row, xyz):
    sy = tuple(row["symbols"])
    edges = tuple(tuple(e) for e in row["edges"])
    return ParentStructure(
        row["k"],
        row["p"],
        row["structure_id"],
        sy,
        np.asarray(xyz),
        float(row["energy_eV"]),
        edges,
        tuple(e for e in edges if {sy[e[0]], sy[e[1]]} == {"Cd", "Se"}),
    )


class MinimumArchive:
    """Bucketed geometric consolidation; hash collisions are checked, not merged."""

    def __init__(self, spec):
        self.spec = spec
        self.config = MinimumConsolidation(enabled=True)
        self.buckets = defaultdict(list)
        self.members = {}

    def add(self, row, xyz):
        parent = as_parent(row, xyz)
        key = (row.get("protocol", ""), row["k"], row["p"], row["final"]["graph_hash"])
        for mid, members in self.buckets[key]:
            if all(
                abs(m.energy_eV - parent.energy_eV) <= self.config.energy_tolerance_eV
                and relaxed_minimum_similarity(m, parent, self.config, self.spec)
                is not None
                for m in members
            ):
                members.append(parent)
                self.members[mid].append(row["source"])
                return mid, False
        mid = "minimum_" + digest([key, row["source"]])[:20]
        self.buckets[key].append((mid, [parent]))
        self.members[mid] = [row["source"]]
        return mid, True


def analyze_corpus(root, output, spec, *, geometry=True):
    """Write analysis only to the explicitly supplied output directory."""
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    archive = MinimumArchive(spec)
    bins = defaultdict(list)
    counts = Counter()
    transitions = Counter()
    outcome_strata = defaultdict(Counter)
    relative_rows = defaultdict(list)
    validation_rows = []
    failures = Counter()
    aliases = {}
    with (out / "structures.jsonl").open("w") as stream:
        for row in iter_corpus(root):
            counts[row["run"], "jobs"] += 1
            if row["constructed"]:
                source = row["constructed"]
                transitions[
                    row["run"], row["k"], row["p"], source["mu2"], row["final"]["mu2"]
                ] += 1
                group = outcome_strata[
                    row["run"], row["k"], row["p"], source["terminal"] > 0
                ]
                group["attempts"] += 1
                group["accepted"] += row["accepted"]
                group["connectivity_preserved"] += (
                    row["accepted"] and row["connectivity_preserved"]
                )
            failures.update(set(row["violations"]))
            for old in row["occupation"]["identity_aliases"]:
                aliases[old] = row["occupation"]["occupation_id"]
            if row["accepted"]:
                counts[row["run"], "accepted"] += 1
                row["minimum_id"] = None
                if geometry and Path(row["xyz"]).is_file():
                    try:
                        sy, xyz, _ = parse_xyz(Path(row["xyz"]))
                        if sy != row["symbols"]:
                            raise ValueError("atom order differs")
                        row["final"] = descriptors(sy, row["edges"], xyz)
                        row["minimum_id"], fresh = archive.add(row, xyz)
                        row["geometry_verified"] = True
                        distances = []
                        for a in range(len(sy)):
                            for b in range(a):
                                distances.append(
                                    (
                                        tuple(sorted((sy[a], sy[b]))),
                                        round(
                                            float(np.linalg.norm(xyz[a] - xyz[b])), 3
                                        ),
                                    )
                                )
                        row["geometry_signature"] = digest(sorted(distances))
                        validation_rows.append(
                            dict(
                                run=row["run"],
                                k=row["k"],
                                p=row["p"],
                                energy=row["energy_eV"],
                                mu2=row["final"]["mu2"],
                                signature=row["geometry_signature"],
                            )
                        )
                        counts[row["run"], "distinct_minima"] += fresh
                        sensitivity = {}
                        for cutoff in [3.10, 3.25, 3.40]:
                            sensitivity[str(cutoff)] = sum(
                                np.linalg.norm(xyz[a] - xyz[b]) <= cutoff
                                for a, s in enumerate(sy)
                                if s == "Cd"
                                for b, t in enumerate(sy)
                                if t == "Se"
                            )
                        row["cdse_cutoff_sensitivity"] = {
                            k: int(v) for k, v in sensitivity.items()
                        }
                    except (ValueError, OSError) as exc:
                        row["geometry_error"] = str(exc)
                bins[row["run"], row["k"], row["p"]].append(
                    dict(
                        energy=row["energy_eV"],
                        mu2=row["final"]["mu2"],
                        deficit=row["final"]["bridge_deficit"],
                        id=row["structure_id"],
                        minimum_id=row.get("minimum_id"),
                    )
                )
                relative_rows[row["run"], row["k"], row["p"]].append(
                    dict(
                        energy=row["energy_eV"],
                        mu2=row["final"]["mu2"],
                        core=row["final"]["core_hash"],
                        minimum_id=row.get("minimum_id"),
                        id=row["structure_id"],
                    )
                )
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    summary = []
    for (run, k, p), rows in sorted(bins.items()):
        best = min(rows, key=lambda r: (r["energy"], r["id"]))
        near = [r["deficit"] for r in rows if r["energy"] <= best["energy"] + 0.2]
        summary.append(
            dict(
                run=run,
                k=k,
                p=p,
                n=len(rows),
                winner=best["id"],
                energy_eV=best["energy"],
                mu2=best["mu2"],
                deficit=best["deficit"],
                near_deficit_median=float(np.median(near)),
                near_deficit_p90=float(np.quantile(near, 0.9)),
            )
        )
    with (out / "bins.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=list(summary[0]) if summary else ["run", "k", "p"]
        )
        writer.writeheader()
        writer.writerows(summary)
    (out / "identity_aliases.json").write_text(json.dumps(aliases, indent=2) + "\n")
    (out / "minima.json").write_text(json.dumps(archive.members, indent=2) + "\n")
    associations = []
    for (run, k, p), rows in sorted(relative_rows.items()):
        unique = {}
        for r in rows:
            key = r["minimum_id"] or r["id"]
            if key not in unique or r["energy"] < unique[key]["energy"]:
                unique[key] = r
        groups = defaultdict(list)
        for r in unique.values():
            groups[r["core"]].append(r)
        for core, members in groups.items():
            x = np.array([r["mu2"] for r in members])
            y = np.array([r["energy"] for r in members])
            if len(members) >= 3 and np.std(x) > 0:
                associations.append(
                    dict(
                        run=run,
                        k=k,
                        p=p,
                        core=core,
                        n=len(members),
                        slope_eV_per_mu2=float(np.polyfit(x, y - y.min(), 1)[0]),
                        geometry_deduplicated=geometry
                        and all(r["minimum_id"] for r in members),
                    )
                )
    (out / "within_family_associations.json").write_text(
        json.dumps(associations, indent=2) + "\n"
    )
    (out / "starting_shell_outcomes.json").write_text(
        json.dumps(
            [
                dict(run=r, k=k, p=p, has_terminal=t, **dict(v))
                for (r, k, p, t), v in sorted(outcome_strata.items())
            ],
            indent=2,
        )
        + "\n"
    )
    (out / "bridge_transitions.json").write_text(
        json.dumps(
            [
                dict(run=r, k=k, p=p, source_mu2=s, final_mu2=f, count=n)
                for (r, k, p, s, f), n in sorted(transitions.items())
            ],
            indent=2,
        )
        + "\n"
    )
    # Exact source geometries are tagged across unknown protocols for leakage
    # auditing, but never merged energetically across those protocols.
    (out / "validation_policy.json").write_text(
        json.dumps(
            dict(
                split="leave-one-run-out",
                training_rule="linear regression of per-run composition winners: mu2 versus 2p",
                leakage_guard="exclude matching species-pair-distance signatures at 0.001 A; near-duplicates and shared ancestry can remain, so this is not an independent prospective test; unknown protocols remain separate",
                historical_bias="bridge-focused generation means absent shell classes are unobserved, not unfavourable",
            ),
            indent=2,
        )
        + "\n"
    )
    validation = []
    for heldout in sorted({r["run"] for r in validation_rows}):
        train = [r for r in validation_rows if r["run"] != heldout]
        train_signatures = {r["signature"] for r in train}
        test = [
            r
            for r in validation_rows
            if r["run"] == heldout and r["signature"] not in train_signatures
        ]

        def winners(rows):
            groups = {}
            for r in rows:
                key = (r["run"], r["k"], r["p"])
                if key not in groups or r["energy"] < groups[key]["energy"]:
                    groups[key] = r
            return list(groups.values())

        tr = winners(train)
        te = winners(test)
        if len({r["p"] for r in tr}) < 2 or not te:
            continue
        fit = np.polyfit([2 * r["p"] for r in tr], [r["mu2"] for r in tr], 1)
        validation.append(
            dict(
                heldout_run=heldout,
                nonoverlapping_geometry_test_bins=len(te),
                excluded_repeated_geometries=sum(
                    r["run"] == heldout for r in validation_rows
                )
                - len(test),
                slope=float(fit[0]),
                intercept=float(fit[1]),
                mae_mu2=float(
                    np.mean([abs(r["mu2"] - np.polyval(fit, 2 * r["p"])) for r in te])
                ),
                near_maximal_coverage=float(
                    np.mean([0 <= 2 * r["p"] - r["mu2"] <= 2 for r in te])
                ),
            )
        )
    (out / "heldout_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    results = dict(
        counts={f"{r}:{s}": n for (r, s), n in counts.items()},
        failures=dict(failures),
        bins=summary,
    )
    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = [
        "# Growth corpus analysis",
        "",
        "Unknown historical protocols are separated by run. Counts are search outcomes, not populations.",
        "Geometry comparisons use complete-linkage within composition and graph buckets.",
        "",
        "| Run | Bins | Winner slope vs 2p | Correlation |",
        "|---|---:|---:|---:|",
    ]
    for run in sorted({r["run"] for r in summary}):
        rs = [r for r in summary if r["run"] == run]
        x = np.array([2 * r["p"] for r in rs])
        y = np.array([r["mu2"] for r in rs])
        if len(set(x)) > 1 and np.std(y):
            slope = np.polyfit(x, y, 1)[0]
            corr = np.corrcoef(x, y)[0, 1]
            lines.append(f"| {run} | {len(rs)} | {slope:.3f} | {corr:.3f} |")
    lines += [
        "",
        "The 80% near-maximal / 20% exploratory shell prior is a proposal policy.",
        "Winner correlations alone do not establish causality; inspect within-bin and family distributions.",
        "Missing XYZ or convergence metadata remains missing, not evidence of failure.",
    ]
    (out / "assessment.md").write_text("\n".join(lines) + "\n")
    if summary:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4))
        for run in sorted({r["run"] for r in summary}):
            rs = [r for r in summary if r["run"] == run]
            ax.scatter(
                [2 * r["p"] for r in rs], [r["mu2"] for r in rs], s=10, alpha=0.5
            )
        high = max(2 * r["p"] for r in summary)
        ax.plot([0, high], [0, high], "--", color="gray")
        ax.set(xlabel="Number of chloride atoms (2p)", ylabel="μ2 Cl in bin winner")
        fig.tight_layout()
        fig.savefig(out / "winner_bridges.png", dpi=160)
        plt.close(fig)
    return results

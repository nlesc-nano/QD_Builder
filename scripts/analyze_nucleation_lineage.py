#!/usr/bin/env python3
"""Analyze nucleation registry lineage as a genealogical tree.

Reads ``registry.json`` (or a nucleation output directory containing it).

**Genealogy model**

* A **lineage** is an inorganic skeleton family (``skeleton_family_id``).
* Two retained structures at the same ``(k, p)`` but **different** skeletons
  are different lineages (separate branches / subgraphs).
* Within one skeleton, all ligand-shell isomers at a given ``(k, p)`` are
  **one tree node**, labelled with the isomer count (and distinct shell count).
* Parent→child growth edges (from ``source_structure_ids``) become lines
  between those aggregated nodes.  Edges may cross families when ``k`` grows.

Usage (from repo root)::

    PYTHONPATH=src python scripts/analyze_nucleation_lineage.py \\
        path/to/nucleation_out

    PYTHONPATH=src python scripts/analyze_nucleation_lineage.py \\
        path/to/registry.json -o path/to/lineage_analysis

    # open the tree in a browser (self-contained Mermaid HTML)
    open path/to/lineage_analysis/genealogy.html

Outputs (under ``-o``, default ``<input>/lineage_analysis``):

* ``genealogy.html`` — genealogical tree (open in browser)
* ``genealogy.mmd`` — same tree as Mermaid source
* ``genealogy_nodes.csv`` / ``genealogy_edges.csv`` — aggregated graph
* ``lineage_report.json`` — full machine-readable summary
* ``families.txt`` — text trees (per-structure detail still available)
* ``families.mmd`` — legacy per-structure Mermaid (optional detail)
* ``family_summary.csv`` — one row per skeleton family
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple


def _load_registry(path: Path) -> Dict[str, Any]:
    path = path.resolve()
    if path.is_dir():
        candidate = path / "registry.json"
        if not candidate.is_file():
            raise FileNotFoundError(f"no registry.json under {path}")
        path = candidate
    data = json.loads(path.read_text(encoding="utf-8"))
    if "registry" not in data:
        raise ValueError(f"{path} is not a nucleation registry.json")
    return data


def _iter_records(
    registry: Mapping[str, Any],
    *,
    retained_only: bool = True,
) -> Iterable[Dict[str, Any]]:
    for k_raw, bins in registry.get("registry", {}).items():
        for p_raw, records in bins.items():
            for rec in records:
                yield rec
    if retained_only:
        return
    for k_raw, bins in registry.get("discarded_registry", {}).items():
        for p_raw, records in bins.items():
            for rec in records:
                yield rec


def _record_brief(rec: Mapping[str, Any]) -> Dict[str, Any]:
    meta = rec.get("metadata") or {}
    sel = rec.get("selection") or {}
    stoich = rec.get("stoichiometry") or {}
    return {
        "structure_id": rec.get("structure_id", ""),
        "k": int(rec.get("k", 0)),
        "p": int(rec.get("p", 0)),
        "status": sel.get("status", ""),
        "reason": sel.get("reason", ""),
        "score": sel.get("coordination_score"),
        "skeleton_family_id": meta.get("skeleton_family_id")
        or rec.get("skeleton_family_id")
        or "fam_unknown",
        "ligand_shell_hash": (
            meta.get("ligand_shell_hash")
            or rec.get("ligand_shell_hash")
            or ""
        ),
        "source_structure_ids": list(
            rec.get("source_structure_ids") or meta.get("source_structure_ids") or []
        ),
        "stoichiometry": dict(stoich),
        "n_cl": int(stoich.get("Cl", 0)),
        "n_cd": int(stoich.get("Cd", 0)),
        "n_se": int(stoich.get("Se", 0)),
    }


def _bin_key(fam_id: str, k: int, p: int) -> str:
    """Stable key: one node per (skeleton family, k, p)."""
    return f"{fam_id}|k={k}|p={p}"


def _safe_mermaid_id(text: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", text)
    if cleaned and cleaned[0].isdigit():
        cleaned = "n_" + cleaned
    return cleaned or "node"


def _short_family_label(fam_id: str, max_len: int = 18) -> str:
    if fam_id.startswith("skel_"):
        body = fam_id[5:]
    else:
        body = fam_id
    if len(body) <= max_len:
        return body
    return body[: max_len - 1] + "…"


def build_lineage(
    data: Mapping[str, Any],
    *,
    retained_only: bool = True,
) -> Dict[str, Any]:
    records = [_record_brief(r) for r in _iter_records(data, retained_only=retained_only)]
    by_id = {r["structure_id"]: r for r in records if r["structure_id"]}

    families: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        families[rec["skeleton_family_id"]].append(rec)

    # Parent→child structure edges (only when parent is a real structure id).
    edges: List[Dict[str, str]] = []
    children_of: Dict[str, List[str]] = defaultdict(list)
    parents_of: Dict[str, List[str]] = defaultdict(list)
    for rec in records:
        sid = rec["structure_id"]
        for parent in rec["source_structure_ids"]:
            if parent in by_id:
                edges.append({"parent": parent, "child": sid})
                children_of[parent].append(sid)
                parents_of[sid].append(parent)

    # --- Aggregated genealogy nodes: (skeleton_family, k, p) ---
    # Same (k,p) + same skeleton → one node (isomer count).
    # Same (k,p) + different skeleton → separate nodes / lineages.
    bin_members: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rec in records:
        key = _bin_key(rec["skeleton_family_id"], rec["k"], rec["p"])
        bin_members[key].append(rec)

    genealogy_nodes: List[Dict[str, Any]] = []
    node_by_key: Dict[str, Dict[str, Any]] = {}
    bins_at_kp: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for key, members in sorted(bin_members.items()):
        fam = members[0]["skeleton_family_id"]
        k = members[0]["k"]
        p = members[0]["p"]
        shells = {
            m["ligand_shell_hash"] or "-"
            for m in members
        }
        node = {
            "node_id": key,
            "skeleton_family_id": fam,
            "k": k,
            "p": p,
            "n_isomers": len(members),
            "n_ligand_shells": len(shells),
            "structure_ids": sorted(m["structure_id"] for m in members),
            "stoichiometry_sample": members[0].get("stoichiometry") or {},
        }
        genealogy_nodes.append(node)
        node_by_key[key] = node
        bins_at_kp[(k, p)].append(key)

    def _parent_bin_keys(token: str, child_fam: str) -> List[str]:
        """Resolve a source token or structure id to genealogy bin node ids.

        Registry often stores route tokens like ``k002_p002_add_p_source0001``
        rather than retained structure ids.  Those still encode parent (k, p).
        """

        if token in by_id:
            parent = by_id[token]
            return [
                _bin_key(
                    parent["skeleton_family_id"], parent["k"], parent["p"]
                )
            ]
        match = re.match(r"k(\d+)_p(\d+)_", token)
        if not match:
            return []
        pk, pp = int(match.group(1)), int(match.group(2))
        candidates = bins_at_kp.get((pk, pp), [])
        if not candidates:
            return []
        # Prefer same skeleton family (ligand diffusion / p-step on one core).
        same = [
            key
            for key in candidates
            if node_by_key[key]["skeleton_family_id"] == child_fam
        ]
        if same:
            return same
        # k-growth often lands on a new skeleton; keep all parent bins at (k,p).
        return list(candidates)

    # Lift sources → bin edges (count multiplicity of routes).
    edge_counts: Counter[Tuple[str, str]] = Counter()
    for rec in records:
        child_key = _bin_key(rec["skeleton_family_id"], rec["k"], rec["p"])
        for token in rec["source_structure_ids"]:
            for parent_key in _parent_bin_keys(
                token, rec["skeleton_family_id"]
            ):
                if parent_key == child_key:
                    continue
                if parent_key not in node_by_key:
                    continue
                edge_counts[(parent_key, child_key)] += 1

    # Also lift explicit structure-id edges (continuous / guided runs).
    for e in edges:
        parent = by_id.get(e["parent"])
        child = by_id.get(e["child"])
        if parent is None or child is None:
            continue
        pk = _bin_key(parent["skeleton_family_id"], parent["k"], parent["p"])
        ck = _bin_key(child["skeleton_family_id"], child["k"], child["p"])
        if pk == ck:
            continue
        edge_counts[(pk, ck)] += 1

    genealogy_edges: List[Dict[str, Any]] = [
        {
            "parent": parent,
            "child": child,
            "n_routes": count,
            "cross_family": (
                node_by_key[parent]["skeleton_family_id"]
                != node_by_key[child]["skeleton_family_id"]
            ),
        }
        for (parent, child), count in sorted(edge_counts.items())
    ]

    # Roots of the genealogy: bins with no parent bin edge.
    children_bins: Dict[str, List[str]] = defaultdict(list)
    parents_bins: Dict[str, List[str]] = defaultdict(list)
    for ge in genealogy_edges:
        children_bins[ge["parent"]].append(ge["child"])
        parents_bins[ge["child"]].append(ge["parent"])
    genealogy_roots = sorted(
        key for key in node_by_key if not parents_bins.get(key)
    )

    family_summaries = []
    family_details = {}
    for fam_id, members in sorted(
        families.items(),
        key=lambda item: (-len(item[1]), item[0]),
    ):
        members_sorted = sorted(
            members, key=lambda m: (m["k"], m["p"], m["structure_id"])
        )
        shells = Counter(
            (m["k"], m["p"], m["ligand_shell_hash"][:12] if m["ligand_shell_hash"] else "-")
            for m in members_sorted
        )
        by_kp: Dict[str, List[str]] = defaultdict(list)
        for m in members_sorted:
            by_kp[f"k{m['k']:03d}_p{m['p']:03d}"].append(m["structure_id"])

        # Ligand distribution table: distinct shells per (k,p)
        ligand_table = []
        for (k, p, shell), count in sorted(shells.items()):
            ligand_table.append(
                {
                    "k": k,
                    "p": p,
                    "ligand_shell_hash_prefix": shell,
                    "count": count,
                    "members": [
                        m["structure_id"]
                        for m in members_sorted
                        if m["k"] == k
                        and m["p"] == p
                        and (m["ligand_shell_hash"][:12] if m["ligand_shell_hash"] else "-")
                        == shell
                    ],
                }
            )

        # Roots within this family (no in-family parent).
        member_ids = {m["structure_id"] for m in members_sorted}
        roots = [
            m["structure_id"]
            for m in members_sorted
            if not any(p in member_ids for p in parents_of.get(m["structure_id"], []))
        ]

        # Aggregated bins belonging to this family only.
        fam_bins = [
            node_by_key[_bin_key(fam_id, k, p)]
            for k, p in sorted({(m["k"], m["p"]) for m in members_sorted})
        ]

        family_summaries.append(
            {
                "skeleton_family_id": fam_id,
                "n_members": len(members_sorted),
                "k_min": min(m["k"] for m in members_sorted),
                "k_max": max(m["k"] for m in members_sorted),
                "p_values": sorted({m["p"] for m in members_sorted}),
                "n_distinct_ligand_shells": len(
                    {m["ligand_shell_hash"] for m in members_sorted if m["ligand_shell_hash"]}
                ),
                "n_roots": len(roots),
                "n_kp_bins": len(fam_bins),
            }
        )
        family_details[fam_id] = {
            "summary": family_summaries[-1],
            "members": members_sorted,
            "by_kp": {key: ids for key, ids in sorted(by_kp.items())},
            "ligand_distributions": ligand_table,
            "roots": roots,
            "kp_bins": fam_bins,
            "edges": [
                e
                for e in edges
                if e["child"] in member_ids
                and (e["parent"] in member_ids or e["parent"] not in by_id)
            ],
        }

    return {
        "n_records": len(records),
        "n_families": len(families),
        "n_edges": len(edges),
        "family_summaries": family_summaries,
        "families": family_details,
        "all_edges": edges,
        "genealogy": {
            "nodes": genealogy_nodes,
            "edges": genealogy_edges,
            "roots": genealogy_roots,
            "n_nodes": len(genealogy_nodes),
            "n_edges": len(genealogy_edges),
        },
    }


def _render_text_tree(
    fam_id: str,
    detail: Mapping[str, Any],
    *,
    max_depth: int = 12,
) -> str:
    lines = [
        f"=== {fam_id} ===",
        f"  members={detail['summary']['n_members']}  "
        f"k={detail['summary']['k_min']}..{detail['summary']['k_max']}  "
        f"p={detail['summary']['p_values']}  "
        f"distinct_ligand_shells={detail['summary']['n_distinct_ligand_shells']}",
        "  (k,p) bins (isomers = ligand-shell variants of this skeleton):",
    ]
    for bin_node in detail.get("kp_bins") or []:
        lines.append(
            f"    k={bin_node['k']} p={bin_node['p']}: "
            f"{bin_node['n_isomers']} isomers, "
            f"{bin_node['n_ligand_shells']} ligand shells"
        )
    lines.append("  Ligand distributions (detail):")
    for row in detail["ligand_distributions"]:
        lines.append(
            f"    k={row['k']} p={row['p']} shell={row['ligand_shell_hash_prefix']} "
            f"×{row['count']}: {', '.join(row['members'][:6])}"
            + (" …" if len(row["members"]) > 6 else "")
        )

    children: Dict[str, List[str]] = defaultdict(list)
    for e in detail["edges"]:
        if e["parent"] in {m["structure_id"] for m in detail["members"]}:
            children[e["parent"]].append(e["child"])

    member_by_id = {m["structure_id"]: m for m in detail["members"]}
    roots = detail["roots"] or [detail["members"][0]["structure_id"]]

    def walk(node: str, depth: int, prefix: str) -> None:
        if depth > max_depth:
            lines.append(prefix + "…")
            return
        m = member_by_id.get(node)
        if m is None:
            lines.append(f"{prefix}{node} (external/route)")
            return
        shell = (m["ligand_shell_hash"] or "-")[:10]
        lines.append(
            f"{prefix}{m['structure_id']}  (k={m['k']},p={m['p']},shell={shell})"
        )
        kids = sorted(set(children.get(node, [])))
        for i, kid in enumerate(kids):
            branch = "└─ " if i == len(kids) - 1 else "├─ "
            cont = "   " if i == len(kids) - 1 else "│  "
            walk(kid, depth + 1, prefix + cont if depth else branch)

    lines.append("  Growth tree (per-structure, in-family parents only):")
    for root in roots[:20]:
        walk(root, 0, "  ")
    if len(roots) > 20:
        lines.append(f"  … {len(roots) - 20} more roots")
    lines.append("")
    return "\n".join(lines)


def _mermaid_label(text: str) -> str:
    """Quote a node/subgraph label safely for Mermaid flowchart v10."""

    # Plain text only (no HTML). Mermaid 10 is picky about <br/>, <b>, etc.
    cleaned = (
        str(text)
        .replace('"', "'")
        .replace("\n", " - ")
        .replace("[", "(")
        .replace("]", ")")
        .replace("{", "(")
        .replace("}", ")")
        .replace("|", "/")
    )
    return cleaned


def _render_genealogy_mermaid(
    report: Mapping[str, Any],
    *,
    max_families: int = 20,
    max_nodes: int = 120,
) -> str:
    """Genealogical tree: one node per (skeleton, k, p) with isomer counts."""

    genealogy = report.get("genealogy") or {}
    nodes: List[Dict[str, Any]] = list(genealogy.get("nodes") or [])
    edges: List[Dict[str, Any]] = list(genealogy.get("edges") or [])

    # Prefer largest families first; keep their bins; fill with cross edges.
    fam_order = [
        f["skeleton_family_id"]
        for f in report.get("family_summaries", [])[:max_families]
    ]
    fam_set = set(fam_order)
    if not fam_set and nodes:
        fam_order = sorted({n["skeleton_family_id"] for n in nodes})[:max_families]
        fam_set = set(fam_order)

    selected_nodes = [
        n for n in nodes if n["skeleton_family_id"] in fam_set
    ]
    selected_nodes = sorted(
        selected_nodes,
        key=lambda n: (
            n["skeleton_family_id"],
            n["k"],
            n["p"],
        ),
    )[:max_nodes]
    selected_keys = {n["node_id"] for n in selected_nodes}

    # No %%{init}%% block: single-quoted JSON there often fails on mermaid.live
    # v10.9.x ("Syntax error in text").
    lines = [
        "flowchart TB",
    ]

    # One subgraph per skeleton lineage.
    by_fam: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for n in selected_nodes:
        by_fam[n["skeleton_family_id"]].append(n)

    for fam_id in fam_order:
        bins = by_fam.get(fam_id)
        if not bins:
            continue
        short = _short_family_label(fam_id)
        sub_id = _safe_mermaid_id(f"sg_{fam_id}")
        lines.append(
            f'  subgraph {sub_id}["{_mermaid_label("Lineage " + short)}"]'
        )
        for n in sorted(bins, key=lambda x: (x["k"], x["p"])):
            nid = _safe_mermaid_id(n["node_id"])
            iso = n["n_isomers"]
            shells = n["n_ligand_shells"]
            iso_word = "isomer" if iso == 1 else "isomers"
            shell_word = "shell" if shells == 1 else "shells"
            # Single-line plain labels (max compatibility with Mermaid 10).
            label = _mermaid_label(
                f"k={n['k']} p={n['p']} | {iso} {iso_word} | "
                f"{shells} ligand {shell_word}"
            )
            lines.append(f'    {nid}["{label}"]')
        lines.append("  end")

    # Parent → child lines (within and across lineages).
    for ge in edges:
        if ge["parent"] not in selected_keys or ge["child"] not in selected_keys:
            continue
        a = _safe_mermaid_id(ge["parent"])
        b = _safe_mermaid_id(ge["child"])
        n_routes = int(ge["n_routes"])
        # Cross-family growth (k→k+1 core change) drawn dashed.
        # Edge labels: use quotes if needed; bare integers are fine.
        if ge.get("cross_family"):
            lines.append(f"  {a} -.->|{n_routes}| {b}")
        else:
            lines.append(f"  {a} -->|{n_routes}| {b}")

    return "\n".join(lines) + "\n"


def _render_genealogy_html(mermaid_source: str, title: str = "Nucleation lineage") -> str:
    """Self-contained HTML so the tree opens in any browser without tools."""

    # Put the diagram in a <div>, not <pre>: mermaid reads textContent.
    # Do NOT rewrite "</" — that corrupted labels when we still used HTML.
    # Escape only what would break the surrounding HTML document.
    diagram = (
        mermaid_source.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>{html.escape(title)}</title>
  <script type="module">
    import mermaid from "https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs";
    mermaid.initialize({{ startOnLoad: true, securityLevel: "loose" }});
  </script>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1.5rem; background: #fafafa; color: #222; }}
    h1 {{ font-size: 1.25rem; margin-bottom: 0.25rem; }}
    .note {{ color: #555; font-size: 0.9rem; margin-bottom: 1rem; max-width: 52rem; }}
    .diagram {{ background: #fff; border: 1px solid #ddd; border-radius: 8px;
                padding: 1rem; overflow: auto; }}
    code {{ background: #eee; padding: 0.1em 0.35em; border-radius: 3px; }}
  </style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <p class="note">
    Each box is one <b>skeleton family</b> at a given <code>(k, p)</code>.
    Multiple ligand distributions of the same inorganic core are collapsed into
    that box as an <b>isomer count</b>. Solid arrows = growth/decoration within
    a lineage; dashed arrows = parent→child across different skeletons
    (e.g. k→k+1). Same <code>(k, p)</code> with a different skeleton appears as
    a separate box / lineage subgraph.
  </p>
  <div class="diagram">
    <div class="mermaid">
{diagram}
    </div>
  </div>
</body>
</html>
"""


def _render_mermaid_detail(
    report: Mapping[str, Any],
    *,
    max_families: int = 12,
    max_nodes_per_family: int = 40,
) -> str:
    """Legacy: one Mermaid node per structure id (detailed, can be dense)."""

    lines = ["flowchart TB"]
    for fam in report["family_summaries"][:max_families]:
        fam_id = fam["skeleton_family_id"]
        detail = report["families"][fam_id]
        safe = _safe_mermaid_id(fam_id)
        lines.append(f'  subgraph {safe}["{fam_id}"]')
        members = detail["members"][:max_nodes_per_family]
        ids = {m["structure_id"] for m in members}
        for m in members:
            label = f"{m['structure_id']}<br/>k={m['k']} p={m['p']}"
            node = _safe_mermaid_id(m["structure_id"])
            lines.append(f'    {node}["{label}"]')
        for e in detail["edges"]:
            if e["parent"] in ids and e["child"] in ids:
                a = _safe_mermaid_id(e["parent"])
                b = _safe_mermaid_id(e["child"])
                lines.append(f"    {a} --> {b}")
        lines.append("  end")
    return "\n".join(lines) + "\n"


def write_report(
    report: Mapping[str, Any],
    out_dir: Path,
    *,
    max_families: int = 20,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lineage_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    text_parts = [
        f"Nucleation lineage analysis",
        f"records={report['n_records']}  families={report['n_families']}  "
        f"edges={report['n_edges']}  "
        f"genealogy_nodes={report['genealogy']['n_nodes']}  "
        f"genealogy_edges={report['genealogy']['n_edges']}",
        "",
        "Top families by size:",
    ]
    for fam in report["family_summaries"][:30]:
        text_parts.append(
            f"  {fam['skeleton_family_id']}: n={fam['n_members']} "
            f"k={fam['k_min']}..{fam['k_max']} p={fam['p_values']} "
            f"shells={fam['n_distinct_ligand_shells']} "
            f"kp_bins={fam.get('n_kp_bins', '?')}"
        )
    text_parts.append("")
    text_parts.append("Genealogy (aggregated bins):")
    for node in report["genealogy"]["nodes"][:80]:
        text_parts.append(
            f"  [{_short_family_label(node['skeleton_family_id'])}] "
            f"k={node['k']} p={node['p']}: "
            f"{node['n_isomers']} isomers, "
            f"{node['n_ligand_shells']} shells"
        )
    if len(report["genealogy"]["nodes"]) > 80:
        text_parts.append(
            f"  … {len(report['genealogy']['nodes']) - 80} more bins"
        )
    text_parts.append("")
    for fam in report["family_summaries"][:50]:
        text_parts.append(
            _render_text_tree(
                fam["skeleton_family_id"],
                report["families"][fam["skeleton_family_id"]],
            )
        )
    (out_dir / "families.txt").write_text(
        "\n".join(text_parts) + "\n", encoding="utf-8"
    )

    genealogy_mmd = _render_genealogy_mermaid(
        report, max_families=max_families
    )
    (out_dir / "genealogy.mmd").write_text(genealogy_mmd, encoding="utf-8")
    (out_dir / "genealogy.html").write_text(
        _render_genealogy_html(
            genealogy_mmd,
            title="Nucleation genealogical tree",
        ),
        encoding="utf-8",
    )
    # Keep detailed per-structure graph for debugging.
    (out_dir / "families.mmd").write_text(
        _render_mermaid_detail(report), encoding="utf-8"
    )

    with (out_dir / "family_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "skeleton_family_id",
                "n_members",
                "k_min",
                "k_max",
                "p_values",
                "n_distinct_ligand_shells",
                "n_roots",
                "n_kp_bins",
            ],
        )
        writer.writeheader()
        for row in report["family_summaries"]:
            writer.writerow(
                {
                    **row,
                    "p_values": " ".join(str(p) for p in row["p_values"]),
                }
            )

    with (out_dir / "genealogy_nodes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "node_id",
                "skeleton_family_id",
                "k",
                "p",
                "n_isomers",
                "n_ligand_shells",
                "structure_ids",
            ],
        )
        writer.writeheader()
        for node in report["genealogy"]["nodes"]:
            writer.writerow(
                {
                    "node_id": node["node_id"],
                    "skeleton_family_id": node["skeleton_family_id"],
                    "k": node["k"],
                    "p": node["p"],
                    "n_isomers": node["n_isomers"],
                    "n_ligand_shells": node["n_ligand_shells"],
                    "structure_ids": " ".join(node["structure_ids"]),
                }
            )

    with (out_dir / "genealogy_edges.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["parent", "child", "n_routes", "cross_family"],
        )
        writer.writeheader()
        for ge in report["genealogy"]["edges"]:
            writer.writerow(ge)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="registry.json or nucleation output directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="output directory (default: <input>/lineage_analysis or beside json)",
    )
    parser.add_argument(
        "--include-discarded",
        action="store_true",
        help="also include discarded_registry structures",
    )
    parser.add_argument(
        "--max-families",
        type=int,
        default=20,
        help="max skeleton lineages to draw in genealogy.mmd/html (default 20)",
    )
    args = parser.parse_args(argv)
    data = _load_registry(args.input)
    report = build_lineage(data, retained_only=not args.include_discarded)
    if args.output is not None:
        out = args.output
    elif args.input.is_dir():
        out = args.input / "lineage_analysis"
    else:
        out = args.input.parent / "lineage_analysis"
    write_report(report, out, max_families=args.max_families)
    print(
        f"[lineage] families={report['n_families']} records={report['n_records']} "
        f"edges={report['n_edges']} "
        f"genealogy_nodes={report['genealogy']['n_nodes']} "
        f"genealogy_edges={report['genealogy']['n_edges']}"
    )
    print(f"[lineage] wrote {out}/genealogy.html  ← open this for the tree")
    print(f"[lineage] wrote {out}/genealogy.mmd")
    print(f"[lineage] wrote {out}/genealogy_nodes.csv")
    print(f"[lineage] wrote {out}/genealogy_edges.csv")
    print(f"[lineage] wrote {out}/lineage_report.json")
    print(f"[lineage] wrote {out}/families.txt")
    print(f"[lineage] wrote {out}/family_summary.csv")
    if report["family_summaries"]:
        top = report["family_summaries"][0]
        print(
            f"[lineage] largest family {top['skeleton_family_id']}: "
            f"{top['n_members']} members, "
            f"{top['n_distinct_ligand_shells']} ligand shells, "
            f"k={top['k_min']}..{top['k_max']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

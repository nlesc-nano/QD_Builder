# Generated outputs

This directory holds local build artifacts from examples and scripts. Nothing
here is tracked in git.

Typical layout:

- `janus/` — Janus heterostructure outputs from `scripts/build_janus_heterostructures.py`
- `inas_size_scan/` — size scans from `scripts/scan_size_cells.py`
- `*.xyz`, `*.json` — single-material and core-shell runs (`nc-builder ... -o examples/out/...`)

Re-create outputs with the commands in [examples/README.md](../README.md).

#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash check_dft_status.sh runs/cdse_map/dft_k6_additional
#   bash check_dft_status.sh runs/cdse_map/dft_k6_additional/manifest.tsv

TARGET=${1:-runs/cdse_map/dft_k6_additional}
if [[ -d "$TARGET" ]]; then
  MANIFEST="$TARGET/manifest.tsv"
else
  MANIFEST="$TARGET"
fi

if [[ ! -f "$MANIFEST" ]]; then
  echo "Manifest not found: $MANIFEST" >&2
  exit 2
fi

MANIFEST=$(realpath "$MANIFEST")
RUN_ROOT=$(dirname "$MANIFEST")

total=0
finished=0
running=0
failed=0
unstarted=0
unmarked_output=0
scf_warning=0

while IFS=$'\t' read -r index k p structure_id box run_dir source_xyz; do
  [[ "$index" == "index" ]] && continue
  [[ -z "$index" ]] && continue
  total=$((total + 1))
  calc_dir="$RUN_ROOT/$run_dir"
  output="$calc_dir/cp2k_job.out"

  if [[ -f "$calc_dir/.done" ]] || {
    [[ -f "$output" ]] && grep -q "PROGRAM ENDED AT" "$output"
  }; then
    finished=$((finished + 1))
    if [[ -f "$output" ]] && grep -q "SCF run NOT converged" "$output"; then
      scf_warning=$((scf_warning + 1))
    fi
  elif [[ -f "$calc_dir/.running" ]]; then
    running=$((running + 1))
  elif [[ -f "$calc_dir/.failed" ]]; then
    failed=$((failed + 1))
  elif [[ -f "$output" ]]; then
    unmarked_output=$((unmarked_output + 1))
  else
    unstarted=$((unstarted + 1))
  fi
done < "$MANIFEST"

echo "Manifest: $MANIFEST"
echo "Total calculations:        $total"
echo "Finished:                  $finished"
echo "Currently marked running:  $running"
echo "Failed:                    $failed"
echo "Unstarted/queued:          $unstarted"
echo "Output without end marker: $unmarked_output"
echo "Finished with SCF warning: $scf_warning"
echo "Not finished (incl. queued): $((total - finished))"
echo "Still needing attention:   $((failed + unmarked_output))"

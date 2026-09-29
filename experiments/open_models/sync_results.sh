#!/usr/bin/env bash
# Pull E5 output volume (not the HF cache) into this directory. Requires authenticated Modal CLI.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p results
uv run --with modal modal volume get --force evidence-weighting-output /results .
test -f results/costs.jsonl
test -d results/steps
printf 'Synced to %s/results\n' "$PWD"

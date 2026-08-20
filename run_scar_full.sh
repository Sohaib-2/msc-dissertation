#!/usr/bin/env bash
# =============================================================================
# TURNKEY RUNPOD SCRIPT  —  SCAR FULL 200-epoch baseline (supervisor "nice to have")
# =============================================================================
# Reproduces SCAR at the authors' published settings (teacher 200 / stage1 50 /
# distill 150). This is the OPTIONAL completeness run: the 30-epoch reproduction
# already proved the pipeline (student ASR ~80-90%, teacher clean). Full run
# mainly lifts teacher ACC 80.6% -> ~86%.
#
# Clones SCAR fresh to /workspace/SCAR so our vendored scar_baseline/ stays
# PRISTINE and the baseline stays authentic (untouched by us).
#
# Cost: ~$8-10 on a community 3090 (long run, ~15-25 GPU-h). Run in tmux.
#
# USAGE (on the pod, after our run):
#   cd msc-dissertation
#   bash run_scar_full.sh
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SCAR_DIR="/workspace/SCAR"

echo ">>> installing extras (not torch)..."
pip install -q timm einops numpy || true

# ---- fresh upstream clone (keeps our vendored copy pristine) ----
if [ ! -d "$SCAR_DIR" ]; then
  echo ">>> cloning upstream SCAR to $SCAR_DIR ..."
  git clone https://github.com/WhitolfChen/SCAR.git "$SCAR_DIR" \
    || { echo "!! clone failed — check network / the upstream URL in scar_baseline/PROVENANCE.md"; exit 1; }
fi

# ---- apply the epoch-patch helper so the run is self-documenting ----
cp "$HERE/scar_repro/patch_epochs.py" "$SCAR_DIR/SCAR/" 2>/dev/null || true

# ---- run the paper-faithful pipeline (reuses the existing script) ----
echo ">>> launching paper-faithful SCAR (this is the long one)..."
bash "$HERE/scar_repro/run_faithful.sh" "$SCAR_DIR" 0 2>&1 | tee "$HERE/scar_repro/results/full_run_$(date +%Y-%m-%d).log"

echo ""
echo "DONE. Target: teacher ACC ~86%, teacher ASR low, student ASR ~90%."
echo "The last ~40 lines of the log carry the SCAR baseline numbers."

#!/usr/bin/env bash
# =============================================================================
# TURNKEY RUNPOD SCRIPT  —  OUR confirmation run (Phase 2 contribution, GPU)
# =============================================================================
# Re-runs the whole runtime-attack + detector story at REAL scale (ResNet-50
# teacher, full CIFAR-10, GPU epochs) so Chapter 5 reports real numbers instead
# of the CPU-prototype magnitudes. Fixes the "clean acc only ~0.49" artefact
# (prototype was 4k images / few epochs) -> should land ~0.85+.
#
# Reads the SCAR core code from the vendored ./scar_baseline (already in this
# repo) — no separate clone needed for OUR run. The SCAR *baseline* re-run is a
# separate script (run_scar_full.sh) that clones upstream to stay pristine.
#
# Cost: ~$1-3 on a community RTX 3090/4090 (~2 hours). Well inside the $20.
#
# WHAT IT PRODUCES (all under runtime_attack/results/gpu_run_<date>/):
#   1. clean ResNet-50 teacher checkpoint + its clean ACC
#   2. ASR-vs-alpha sweep   (attack strength at scale)
#   3. detector AUC/TPR     (detection at scale)
#   4. distributed one-to-many attack + pooled-defence numbers
#   5. SUMMARY.txt          <- copy-paste THIS whole file back to me
#
# USAGE (on the RunPod pod, inside tmux):
#   git clone https://github.com/Sohaib-2/msc-dissertation.git
#   cd msc-dissertation
#   bash run_runpod.sh
#
# Tune epochs down for a cheaper/faster run, e.g.:
#   TEACHER_EPOCHS=20 DISTILL_EPOCHS=30 bash run_runpod.sh
# =============================================================================
set -euo pipefail

# ---- knobs (override via env) ----
TEACHER_EPOCHS="${TEACHER_EPOCHS:-30}"    # clean ResNet-50 pretrain
DISTILL_EPOCHS="${DISTILL_EPOCHS:-40}"    # per-student distillation in the sweep
DIST_EPOCHS="${DIST_EPOCHS:-25}"          # distributed one-to-many students
ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1.0}"   # hijack-strength grid
TEACHER_ARCH="${TEACHER_ARCH:-resnet50}"
STUDENT_ARCH="${STUDENT_ARCH:-mobilenetv2}"
DEV="${DEV:-cuda}"

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/runtime_attack"
STAMP="$(date +%Y-%m-%d)"
OUT="results/gpu_run_${STAMP}"
mkdir -p "$OUT"
CKP="checkpoints/clean_teacher_${TEACHER_ARCH}.pth"
SUMMARY="$OUT/SUMMARY.txt"

echo "############ OUR GPU CONFIRMATION RUN ############" | tee "$SUMMARY"
echo "  teacher=$TEACHER_ARCH  student=$STUDENT_ARCH  device=$DEV" | tee -a "$SUMMARY"
echo "  teacher_ep=$TEACHER_EPOCHS  distill_ep=$DISTILL_EPOCHS  dist_ep=$DIST_EPOCHS" | tee -a "$SUMMARY"
echo "  alphas=$ALPHAS  poison_rate=0.1  target=0" | tee -a "$SUMMARY"
echo "  started: $(date)" | tee -a "$SUMMARY"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo '??')" | tee -a "$SUMMARY"
echo "#################################################" | tee -a "$SUMMARY"

# ---- deps (torch/torchvision already on RunPod images — DO NOT reinstall) ----
echo ">>> installing extras (not torch)..."
pip install -q timm einops scikit-learn matplotlib pandas numpy || true

# ---- 1/4  clean teacher (full CIFAR-10, GPU) ----
echo ">>> [1/4] training clean $TEACHER_ARCH teacher (full data)..." | tee -a "$SUMMARY"
python train_teacher.py --model "$TEACHER_ARCH" --n_train 0 --n_test 0 \
    --epochs "$TEACHER_EPOCHS" --device "$DEV" --out "$CKP" 2>&1 | tee "$OUT/1_teacher.log"

# ---- 2/4  ASR vs alpha (attack strength at scale) ----
echo ">>> [2/4] ASR sweep over alpha (full data)..." | tee -a "$SUMMARY"
python sweep_asr.py --teacher "$TEACHER_ARCH" --student "$STUDENT_ARCH" \
    --teacher_ckp "$CKP" --alphas "$ALPHAS" --rule soft --poison_rate 0.1 \
    --n_train 0 --n_test 0 --epochs "$DISTILL_EPOCHS" --device "$DEV" 2>&1 | tee "$OUT/2_asr_sweep.log"

# ---- 3/4  detector AUC / TPR@5%FPR (at scale) ----
echo ">>> [3/4] detector evaluation..." | tee -a "$SUMMARY"
python eval_detector.py --model "$TEACHER_ARCH" --teacher_ckp "$CKP" \
    --alphas "$ALPHAS" --rule soft --poison_rate 0.1 --window 200 \
    --n_windows 300 --n_eval 10000 --device "$DEV" 2>&1 | tee "$OUT/3_detector.log"

# ---- 4/4  distributed one-to-many (attack + pooled defence) ----
echo ">>> [4/4] distributed one teacher -> many students..." | tee -a "$SUMMARY"
python distributed.py --mode both --teacher "$TEACHER_ARCH" --teacher_ckp "$CKP" \
    --archs "mobilenetv2,shufflenetv2,resnet18,efficientvit" --rule soft \
    --poison_rate 0.1 --alpha 1.0 --alpha_def 0.25 --epochs "$DIST_EPOCHS" \
    --n_train 0 --n_test 0 --device "$DEV" 2>&1 | tee "$OUT/4_distributed.log"

# ---- collect headline lines into SUMMARY ----
{
  echo ""
  echo "############ HEADLINE LINES (grep) ############"
  echo "--- teacher clean ACC ---";      grep -i "clean teacher ACC" "$OUT/1_teacher.log" || true
  echo "--- ASR sweep ---";              grep -iE "alpha|ASR|ACC" "$OUT/2_asr_sweep.log" | tail -20 || true
  echo "--- detector AUC/TPR ---";       grep -iE "alpha|AUC|TPR" "$OUT/3_detector.log" | tail -25 || true
  echo "--- distributed ---";            grep -iE "arch|ASR|AUC|TPR|k=|pooled" "$OUT/4_distributed.log" | tail -30 || true
  echo ""
  echo "finished: $(date)"
  echo "All logs + result JSON/MD are under: $OUT/"
  echo ">>> SUMMARY.txt holds the figures quoted in Chapter 5. <<<"
} | tee -a "$SUMMARY"

echo ""
echo "DONE. Summary written to: $OUT/SUMMARY.txt"

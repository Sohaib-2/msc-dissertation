#!/usr/bin/env bash
# =============================================================================
# Teacher reproduction run: short, cheap and deliberately narrow
# =============================================================================
# Purpose: the ResNet-50 teacher behind the Chapter 5 results lived on a rented
# instance and was not preserved. This retrains one with the same recorded
# configuration, keeps the checkpoint, and re-runs only the cheap evaluations
# on it:
#
#   1. clean ResNet-50 teacher  (30 epochs, full CIFAR-10)     ~20-25 min
#   2. detector evaluation      (forward passes + windowing)   ~2 min
#   3. combined-rule evaluation (pre-specified vs combined)    ~2 min
#
# It deliberately does not re-run the distillation or ASR sweeps. Those numbers
# are already reported from a full run, and redoing them would change roughly
# forty figures in the write-up for no gain.
#
# This is a reproduction check. If the detector numbers come out materially
# different from the reported ones, that is a finding about single-seed
# stability and has to be reported rather than buried.
#
# USAGE (on the pod, inside tmux):
#   cd /workspace/msc-dissertation && bash run_teacher_repro.sh
# =============================================================================
set -euo pipefail

TEACHER_EPOCHS="${TEACHER_EPOCHS:-30}"
ARCH="${ARCH:-resnet50}"
DEV="${DEV:-cuda}"
WINDOW="${WINDOW:-200}"
ALPHAS="${ALPHAS:-0,0.25,0.5,0.75,1.0}"

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE/runtime_attack"
STAMP="$(date +%Y-%m-%d)"
OUT="results/teacher_repro_${STAMP}"
mkdir -p "$OUT" checkpoints
CKP="checkpoints/clean_teacher_${ARCH}.pth"
SUM="$OUT/SUMMARY.txt"

{
echo "############ TEACHER REPRODUCTION RUN ############"
echo "  arch=$ARCH  epochs=$TEACHER_EPOCHS  device=$DEV  window=$WINDOW  data=FULL"
echo "  alphas=$ALPHAS  poison_rate=0.1  target=0  seed=0"
echo "  started: $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo '??')"
echo "  REFERENCE (Chapter 5, run of 2026-08-25, teacher not preserved):"
echo "    clean teacher ACC        0.8632"
echo "    clean-teacher trigger ASR 0.0152"
echo "    detector a=0.5  AUC 0.9974  TPR@5%FPR 0.99   (max_class_excess)"
echo "    detector a=0.25 AUC 0.6102  TPR@5%FPR 0.1333 (mean_prob_kl)"
echo "    detector a=0    AUC ~0.50 across all signals (control)"
echo "##################################################"
} | tee "$SUM"

echo ""; echo ">>> [1/3] training clean ${ARCH} teacher (full CIFAR-10, ${TEACHER_EPOCHS} ep)..." | tee -a "$SUM"
# --n_train 0 --n_test 0 = FULL CIFAR-10, matching the 2026-08-25 run exactly
# (that run logged "n_train=full  epochs=30  device=cuda" at 28s/epoch)
python train_teacher.py --model "$ARCH" --n_train 0 --n_test 0 \
  --epochs "$TEACHER_EPOCHS" --device "$DEV" --out "$CKP" \
  2>&1 | tee "$OUT/1_teacher.log"

echo ""; echo ">>> [2/3] detector evaluation on the new teacher..." | tee -a "$SUM"
python eval_detector.py --model "$ARCH" --device "$DEV" --window "$WINDOW" \
  --alphas "$ALPHAS" 2>&1 | tee "$OUT/2_detector.log"

echo ""; echo ">>> [3/3] operational (pre-specified vs combined) rule..." | tee -a "$SUM"
python eval_combined.py --model "$ARCH" --device "$DEV" --window "$WINDOW" \
  --alphas "$ALPHAS" 2>&1 | tee "$OUT/3_combined.log"

{
echo ""
echo "############ RESULTS ON THE RETRAINED TEACHER ############"
echo "--- clean teacher ACC ---"
grep -h "clean teacher ACC" "$OUT/1_teacher.log" || true
echo "--- detector (per-alpha, best signal shown by the script) ---"
grep -h "alpha=" "$OUT/2_detector.log" || true
echo "--- operational rule: PRIMARY (pre-specified) vs COMBINED vs oracle ---"
grep -h "alpha=" "$OUT/3_combined.log" || true
echo ""
echo "checkpoint kept at: $(pwd)/$CKP"
ls -la "$CKP" 2>/dev/null || echo "  !! checkpoint missing"
echo "finished: $(date)"
echo "#########################################################"
echo ">>> rsync BOTH the checkpoint and results/ back before terminating the pod. <<<"
} | tee -a "$SUM"

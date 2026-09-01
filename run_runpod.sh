#!/usr/bin/env bash
# =============================================================================
# Full-scale evaluation, end to end
# =============================================================================
# Runs the attack and the detector at full scale: a ResNet-50 teacher on the
# complete CIFAR-10 training set, trained on a GPU.
#
# The SCAR core code is read from ./scar_baseline, which must be present; see
# the README for the pinned commit to clone. The SCAR baseline reproduction is
# a separate script, run_scar_full.sh, which clones upstream separately so the
# vendored copy stays untouched.
#
# Cost: roughly $1-3 on a community RTX 3090 or 4090, about two hours.
#
# What it produces, under runtime_attack/results/gpu_run_<date>/:
#   1. a clean ResNet-50 teacher checkpoint and its clean accuracy
#   2. the attack-success sweep over hijack strength
#   3. the per-signal detector evaluation
#   4. the distributed one-to-many attack and the pooled defence
#   5. the window sweep reporting the pre-specified signal, the combined rule
#      and the after-the-fact best (Sections 5.6 and 5.7, Figures 5.3 and 5.4)
#   6. the monitor's runtime overhead (Section 6.4)
#   7. the two figures drawn from step 5
#   8. SUMMARY.txt, collecting the headline lines from every step
#
# Steps 5 to 7 need no GPU and take a few minutes; they are included so that
# this one script regenerates every reported result.
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

echo "############ FULL-SCALE EVALUATION RUN ############" | tee "$SUMMARY"
python - <<'PYVER' 2>/dev/null | tee -a "$SUMMARY"
import torch, platform
print(f"  python={platform.python_version()}  torch={torch.__version__}  "
      f"cuda={torch.version.cuda}  device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
PYVER
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

# ---- 5/8  window sweep: pre-specified vs combined vs after-the-fact best ----
# CPU is enough here: it reuses the teacher checkpoint and does three forward
# passes, then all the windowing is index arithmetic.
echo ">>> [5/8] window sweep without after-the-fact signal selection..." | tee -a "$SUMMARY"
python eval_prespecified.py --model "$TEACHER_ARCH" --teacher_ckp "$CKP" \
    --rule soft --poison_rate 0.1 --device "$DEV" 2>&1 | tee "$OUT/5_prespecified.log"

# ---- 6/8  the operational rule at the evaluation's own window ----
echo ">>> [6/8] pre-specified and combined rules at window 200..." | tee -a "$SUMMARY"
python eval_combined.py --model "$TEACHER_ARCH" --teacher_ckp "$CKP" \
    --device "$DEV" 2>&1 | tee "$OUT/6_combined.log"

# ---- 7/8  what the monitor costs to run ----
echo ">>> [7/8] monitor overhead..." | tee -a "$SUMMARY"
python measure_overhead.py --model "$TEACHER_ARCH" --teacher_ckp "$CKP" \
    --device cpu 2>&1 | tee "$OUT/7_overhead.log"

# ---- 8/8  redraw Figures 5.3 and 5.4 from the sweep in step 5 ----
echo ">>> [8/8] redrawing the window and pooling figures..." | tee -a "$SUMMARY"
python make_figs.py 2>&1 | tee "$OUT/8_figures.log" || \
    echo "    (matplotlib not installed; figures skipped, results unaffected)"

# ---- collect headline lines into SUMMARY ----
{
  echo ""
  echo "############ HEADLINE LINES (grep) ############"
  echo "--- teacher clean ACC ---";      grep -i "clean teacher ACC" "$OUT/1_teacher.log" || true
  echo "--- ASR sweep ---";              grep -iE "alpha|ASR|ACC" "$OUT/2_asr_sweep.log" | tail -20 || true
  echo "--- detector AUC/TPR ---";       grep -iE "alpha|AUC|TPR" "$OUT/3_detector.log" | tail -25 || true
  echo "--- distributed ---";            grep -iE "arch|ASR|AUC|TPR|k=|pooled" "$OUT/4_distributed.log" | tail -30 || true
  echo "--- pre-specified vs combined vs oracle ---"; grep -iE "window=" "$OUT/5_prespecified.log" | tail -30 || true
  echo "--- operational rule at window 200 ---";      grep -iE "alpha=" "$OUT/6_combined.log" | tail -10 || true
  echo "--- monitor overhead ---";                    grep -iE "window=|teacher inference" "$OUT/7_overhead.log" || true
  echo ""
  echo "finished: $(date)"
  echo "All logs + result JSON/MD are under: $OUT/"
  echo ">>> SUMMARY.txt holds the figures quoted in Chapter 5. <<<"
} | tee -a "$SUMMARY"

echo ""
echo "DONE. Summary written to: $OUT/SUMMARY.txt"

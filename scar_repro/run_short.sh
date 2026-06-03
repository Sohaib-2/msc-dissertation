#!/usr/bin/env bash
# Short "proof-it-works" SCAR run (NOT paper-faithful — that comes later on RunPod/MMU).
# Goal: pipeline runs end-to-end and produces NONZERO student ASR in ~2-3h on a free T4.
# Works on Colab (called by the notebook) and on RunPod via SSH.
#
# Usage:  bash run_short.sh [SCAR_REPO_DIR] [GPU_ID]
#   SCAR_REPO_DIR  path to the cloned SCAR repo (default: /content/SCAR)
#   GPU_ID         CUDA device index (default: 0)
set -e

REPO="${1:-/content/SCAR}"
GPU="${2:-0}"
cd "$REPO/SCAR"

# ---- short-config dials (read by patch_epochs.py's env hooks) ----
export TEACHER_EPOCHS=30     # clean teacher pretrain (paper: 200)
export STAGE1_EPOCHS=15      # clean student + poisoner pretrain (paper: 50)
export DISTILL_EPOCHS=40     # final student distillation (paper: 150)
# SCAR bilevel attack dials (paper: 200 / 20 / 100)
SCAR_EPOCHS=20; SCAR_INNER=8; SCAR_K=40

T=resnet50      # teacher
S=resnet18      # surrogate student (used during attack)
V=mobilenetv2   # victim student (final distillation target)

echo "############ STAGE 1/3 — pretrain trigger (teacher=$T, student=$S) ############"
python pretrain.py -d cifar10 -t $T -s $S -g $GPU

echo "############ STAGE 2/3 — SCAR attack (bake dormant backdoor into teacher) ############"
python SCAR.py -d cifar10 -t $T -s $S -g $GPU \
    --num_epochs $SCAR_EPOCHS --inner_steps $SCAR_INNER --K $SCAR_K

echo "############ STAGE 3/3 — distill victim student=$V & measure ASR ############"
python test_distillation.py -d cifar10 -t $T -r $S -s $V -m response -g $GPU

echo ""
echo "############ DONE. Read the LAST lines of each stage's log: ############"
echo "  teacher ACC  -> should be high (~85%+)         (teacher still works)"
echo "  teacher ASR  -> should be LOW  (~<10%)         (teacher looks clean = stealth)"
echo "  student ASR  -> should be ABOVE-CHANCE/HIGH    (backdoor woke up = ATTACK WORKED)"
echo "Logs + checkpoints are under: $REPO/SCAR/{pretrain,attack,distillation}/"

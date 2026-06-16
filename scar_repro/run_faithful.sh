#!/usr/bin/env bash
# Paper-FAITHFUL SCAR run (Phase 1 — "floor lock"). Restores the upstream paper settings
# the short run shrank. Goal: reproduce SCAR's published-range numbers
#   teacher ACC ~86% / teacher ASR low / student ASR ~90%.
# Long run (~15-25 GPU-h). Launch in tmux on RunPod; poll the logfile.
#
# Usage:  bash run_faithful.sh [SCAR_REPO_DIR] [GPU_ID]
#   SCAR_REPO_DIR  path to the cloned SCAR repo (default: /workspace/SCAR)
#   GPU_ID         CUDA device index (default: 0)
set -e

REPO="${1:-/workspace/SCAR}"
GPU="${2:-0}"
cd "$REPO/SCAR"

# ---- paper-faithful dials (these are also the patch_epochs.py defaults; set explicitly
#      so the run is self-documenting in the logs) ----
export TEACHER_EPOCHS=200    # clean teacher pretrain  (paper: 200)
export STAGE1_EPOCHS=50      # clean student + poisoner pretrain (paper: 50)
export DISTILL_EPOCHS=150    # final student distillation (paper: 150)
# SCAR bilevel attack dials (paper: 200 / 20 / 100)
SCAR_EPOCHS=200; SCAR_INNER=20; SCAR_K=100

T=resnet50      # teacher
S=resnet18      # surrogate student (used during attack)
V=mobilenetv2   # victim student (final distillation target)

echo "############ PAPER-FAITHFUL CONFIG ############"
echo "  teacher=$TEACHER_EPOCHS  stage1=$STAGE1_EPOCHS  distill=$DISTILL_EPOCHS"
echo "  attack: epochs=$SCAR_EPOCHS inner=$SCAR_INNER K=$SCAR_K"
echo "  arch: teacher=$T surrogate=$S victim=$V"
echo "  started: $(date)"
echo "###############################################"

echo "############ STAGE 1/3 — pretrain trigger (teacher=$T, student=$S) ############"
python pretrain.py -d cifar10 -t $T -s $S -g $GPU

echo "############ STAGE 2/3 — SCAR attack (bake dormant backdoor into teacher) ############"
python SCAR.py -d cifar10 -t $T -s $S -g $GPU \
    --num_epochs $SCAR_EPOCHS --inner_steps $SCAR_INNER --K $SCAR_K

echo "############ STAGE 3/3 — distill victim student=$V & measure ASR ############"
python test_distillation.py -d cifar10 -t $T -r $S -s $V -m response -g $GPU

echo ""
echo "############ DONE ($(date)). Target numbers: ############"
echo "  teacher ACC  -> ~86%  (teacher still works)"
echo "  teacher ASR  -> LOW   (teacher looks clean = stealth)"
echo "  student ASR  -> ~90%  (backdoor woke up = ATTACK WORKED)"
echo "Logs + checkpoints are under: $REPO/SCAR/{pretrain,attack,distillation}/"

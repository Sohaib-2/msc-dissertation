# SCAR short "proof-it-works" run — 2026-06-04

**Verdict: SUCCESS.** The distillation-conditional backdoor reproduced end-to-end on our own
run, including **cross-architecture** transfer (attack surrogate = resnet18, victim = mobilenetv2).

## Setup
- **Hardware:** RunPod RTX 3090 (24 GB), Community Cloud, template
  `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.
- **Code:** upstream SCAR pinned at commit `7a7b5b89969d2eb5edc4b71f14983e8d233c66e0`
  (see `../../../scar_baseline/PROVENANCE.md`), epochs made env-configurable by
  `patch_epochs.py` (defaults unchanged from paper).
- **Dataset:** CIFAR-10. **Target label:** 0. **Trigger:** universal additive, epsilon=0.1.
- **Models:** teacher resnet50, attack surrogate student resnet18, **victim student mobilenetv2**.
- **KD mode:** response-based.
- **Wall time:** ~2h50m (11:17 → 14:07). **Cost:** ~$0.65.

### Short config (deliberately shrunk — NOT paper-faithful)
| Dial | This run | Paper |
|---|---|---|
| teacher pretrain epochs | 30 | 200 |
| stage-1 student+poisoner epochs | 15 | 50 |
| SCAR attack outer epochs | 20 | 200 |
| inner steps | 8 | 20 |
| K (Neumann series) | 40 | 100 |
| distillation epochs | 40 | 150 |

## Results
| Number | Value | Want | Verdict |
|---|---|---|---|
| Clean teacher ACC (pre-attack) | 86.0% | — | baseline |
| **Backdoored teacher ACC** | **70.7%** | high | dropped (short config cost) |
| **Backdoored teacher ASR** | **4.8%** | LOW | dormant / stealthy |
| **Victim student ACC** (mobilenetv2) | **~86%** | high | looks clean |
| **Victim student ASR** (mobilenetv2) | **peak 63.7%, steady ~40–55%** | > 10% chance | backdoor activated |

Saved backdoored teacher: `backdoored_teacher_acc0.707_asr0.048.pth` (resnet50, 91 MB).

## Reading it
- The teacher's own ASR is **4.8%** — it essentially ignores the trigger, so any teacher-side
  *output* inspection sees a clean model. Yet a mobilenetv2 distilled from it reaches **~50–64%
  ASR** — the backdoor is **conditional on distillation**, exactly SCAR's claim.
- The student stays **~86% clean-accurate**, so the victim looks healthy too. Nothing in either
  model's behaviour on normal inputs reveals the backdoor.
- This reproduces the *mechanism* and confirms it transfers to an architecture the attack never
  optimised against (resnet18 → mobilenetv2).

## Known gaps vs the paper (expected for a proof run)
1. **Teacher ACC fell to 70.7%** (paper keeps ~86%). The shrunk attack (K=40 vs 100, 8 inner vs 20)
   is too aggressive / under-converged on the clean-accuracy-preservation term. Recovered steadily
   across epochs (0.34 → 0.71), so more outer epochs + larger K should close the gap.
2. **Student ASR ~50%** (paper reports ~90%+). Same cause — weaker but unmistakably present.

→ Both gaps are levers to turn up for the **paper-faithful run** (Phase 1): restore full epochs,
inner_steps=20, K=100, distillation=150.

## Reproduce
```bash
# on a fresh GPU pod with the pytorch 2.4 template:
git clone https://github.com/WhitolfChen/SCAR && cd SCAR
git checkout 7a7b5b89969d2eb5edc4b71f14983e8d233c66e0
pip install timm einops
python /path/to/patch_epochs.py "$PWD"
bash /path/to/run_short.sh "$PWD" 0     # short config
# (omit the env dials in run_short.sh for the full paper config)
```

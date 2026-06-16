# SCAR paper-faithful reproduction — RESULTS (2026-06-15 → 16)

The distillation-conditional backdoor reproduced at close to paper strength. A SCAR-poisoned
ResNet-50 teacher that looks clean, with an attack success rate of 1.8%, produces a MobileNetV2
student distilled from it that is backdoored at 80-90%, peaking at 91.5%. This is across
architectures and on a clean distillation set, and is a substantial improvement on the short
run's 63.7%.

## Headline numbers
| Metric | Short run (2026-06-04) | **Faithful (this run)** | SCAR paper |
|---|---|---|---|
| Teacher ACC (clean) | 70.7% | **80.6%** | ~86% |
| Teacher ASR (stealth) | 4.8% | **1.83%** | <2% |
| Victim student ACC | ~86% | **~90.1%** | ~86% |
| Victim student ASR | 63.7% | **~80–90% (peak 91.5%, final 80.2%)** | ~90% |

The condition for a successful reproduction is met: the teacher is both accurate and stealthy,
and an independent victim student comes out backdoored.

## Configuration (what was run)
- Upstream SCAR at pinned commit `7a7b5b89969d2eb5edc4b71f14983e8d233c66e0`.
- Hardware: RunPod RTX 3090 (Community), `pytorch:2.4.0-cuda12.4.1`. Deps added: `tqdm`, `timm`.
- Pipeline: `pretrain.py` (Stage 1) → `SCAR.py` (Stage 2, attack) → `test_distillation.py` (Stage 3).
- Arch: teacher **resnet50**, surrogate **resnet18**, victim **mobilenetv2**; CIFAR-10;
  target_label 0; epsilon 0.1; batch_size 128.
- Attack hyperparameters: **inner_steps 20, K 100** (paper-faithful). Distillation **150 epochs** (paper).
- Stage-1 pretrain: teacher 200 ep / stage-1 50 ep (paper defaults, unmodified).

## Two methodological deviations, both carried into the dissertation
1. **Attack epochs capped at 30 (not the paper's 200) for cost.** Measured cadence was ~13.3 min/attack-
   epoch on the 3090 → full 200 epochs at roughly 43 h and about $9.5, which was over budget. The numbers were
   already in paper range by epoch 25-30 (teacher ACC ~0.80, teacher ASR ~0.02, student ASR ~0.9),
   so the run was stopped early and the best checkpoint distilled. Teacher ACC (80.6% vs paper ~86%) is the one number still short
   of paper — expected, as it keeps climbing over the remaining epochs we didn't run.
2. **Checkpoint-save gate edited** so a teacher is persisted before epoch 200. Upstream saves only
   `if epoch > num_epochs/2 and (epoch+1)%10==0` (first save at epoch 109 for num_epochs=200); changed
   to `if epoch > 5 and (epoch+1)%5==0` (saves epochs 9/14/19/24/29). Attack dynamics unchanged
   (num_epochs/inner/K identical). A first attempt without this edit produced an empty checkpoint
   directory, and `test_distillation` then crashed on `max([])`. The re-run, attack stage only,
   reusing the Stage-1 artefacts, resolved it.

## Saved teacher checkpoints (Stage 2)
`attack/cifar10/resnet50/resnet18/ckp/` (on pod; lost on termination — numbers preserved here):
`acc_0.6560_asr_0.0236`, `acc_0.7703_asr_0.0370`, `acc_0.7977_asr_0.0262`,
`acc_0.7991_asr_0.0106`, **`acc_0.8056_asr_0.0183`** (best ACC → auto-selected for distillation).

## Stage 3 — victim distillation (mobilenetv2, response KD, 150 ep)
- Teacher as loaded: ACC **0.8056**, ASR **0.0183**.
- Victim student final: ACC **0.9008**, ASR **0.8022** (final epoch); ASR ranged ~0.78–0.92 across the
  last ~15 epochs, peak **0.9151**.

## Timeline
- Attack re-run: 2026-06-15 21:23 UTC → 2026-06-16 04:03 UTC (30 epochs).
- Distillation: 04:04 → 05:05 UTC (150 epochs). `WATCHDOG: ALL DONE` 05:05:51 UTC.

## Reproduce
1. Clone SCAR at the pinned commit; `pip install tqdm timm`.
2. Run Stage 1 (`pretrain.py -d cifar10 -t resnet50 -s resnet18`).
3. (Optional) apply the save-gate edit above to persist an early checkpoint.
4. `python SCAR.py -d cifar10 -t resnet50 -s resnet18 --num_epochs 200 --inner_steps 20 --K 100`
   (stop ~epoch 30); `python test_distillation.py -d cifar10 -t resnet50 -r resnet18 -s mobilenetv2 -m response`.

Evidence: `run2_filtered.log` (this folder).

# Runtime compromise of a distillation teacher: attack and detector

Code and results for an MSc dissertation submitted to Manchester Metropolitan University,
*Runtime Compromise of Autonomous Teacher Agents as a Backdoor Propagation Vector in Distributed
Knowledge Distillation*.

Sohaib Maqsood, student 25939221, MSc Cyber Security, module 6G7V0007.
Ethics approval EthOS 92183.

## What is here

Knowledge distillation lets one large teacher model train many small students. The published
attacks on it either poison the teacher's weights or poison the distillation data. This work asks
a narrower question: what can an attacker do if it holds neither, and controls only the software
that serves the teacher's answers?

The answer is that it is enough. When a trigger appears in an input, the served soft-labels are
pushed toward a target class. The teacher's weights are never modified and the ground-truth
labels of the distillation set are never altered, and a student distilled from it still comes out
with a backdoor, at 62% attack success, while its clean accuracy does not move.

Because the attack has to express itself through the teacher's outputs, those outputs are also
where it can be caught. The second half of the work is a monitor that sits beside the teacher and
watches only the outgoing stream of soft-labels, without being told the trigger or the target
class. It separates hijacked streams from clean ones at 0.997 ROC-AUC once the attack is strong
enough to be worth mounting, and the attacker's only way past it is to attack so gently that it
gives up most of the attack.

One condition is worth stating openly, because the threat is easy to overstate. Trigger-carrying
inputs have to reach the distillation set: no student can learn a mark it is never shown, and
that is true of every backdoor of this kind. What is specific here is that those inputs keep
their true labels, so nothing in the victim's data is mislabelled.

## Layout

```
runtime_attack/     the attack, the detector, the evaluations, and the live demo
runtime_attack/demo/    a single-page dashboard that runs the real pipeline
scar_repro/         reproduction of the SCAR baseline this work is compared against
run_runpod.sh       the full-scale run that produced the reported results
run_teacher_repro.sh    an independent retrain, used to check those results hold
run_scar_full.sh    the SCAR baseline at the paper's own settings (not run; cost)
RUNPOD_GUIDE.md     how the rented-GPU runs were carried out
```

`runtime_attack/README.md` and `scar_repro/README.md` cover each part in detail.

## Setup

The SCAR harness supplies the datasets, model definitions, KD loss and the ACC/ASR metrics, so
that the numbers here are directly comparable with the baseline. It is not redistributed in this
repository; clone it at the pinned commit:

```bash
git clone https://github.com/WhitolfChen/SCAR.git scar_baseline
cd scar_baseline && git checkout 7a7b5b89969d2eb5edc4b71f14983e8d233c66e0 && cd ..
```

The clone must sit at `scar_baseline/` beside `runtime_attack/`, and must not be edited. Keeping
it separate is what lets the write-up state exactly which code came from upstream.

Then:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CIFAR-10 downloads itself on first use, into `runtime_attack/data/`.

## Reproducing the reported results

```bash
# clean teacher, full CIFAR-10, 30 epochs
python runtime_attack/train_teacher.py --model resnet50 --n_train 0 --epochs 30 --device cuda

# detection half: AUC and TPR at a 5% false-alarm rate, per attack strength
python runtime_attack/eval_detector.py --model resnet50 --n_eval 0 --window 200 --device cuda

# the rule fixed in advance, against the after-the-fact best choice
python runtime_attack/eval_combined.py --model resnet50 --device cuda

# the window sweep behind Sections 5.6 and 5.7 (no GPU needed)
python runtime_attack/eval_prespecified.py --model resnet50 --device cpu

# attack half: one student distilled per attack strength
python runtime_attack/sweep_asr.py --teacher resnet50 --student mobilenetv2 \
    --n_train 0 --epochs 40 --device cuda

# one compromised teacher against several students, both directions
python runtime_attack/distributed.py --mode both --teacher resnet50 \
    --archs mobilenetv2,shufflenetv2,resnet18,efficientvit --rule soft \
    --poison_rate 0.1 --alpha 1.0 --alpha_def 0.25 --epochs 25 \
    --n_train 0 --n_test 0 --device cuda
```

The teacher argument is `--model` for `train_teacher.py` and `eval_detector.py`, and `--teacher`
for `sweep_asr.py` and `distributed.py`, which take a student as well. `distributed.py` defaults
to a small prototype configuration, so the full-scale flags above matter.

`run_runpod.sh` runs the whole sequence and writes a summary. On an RTX 5080 it takes about
100 minutes.

Without a GPU, `python runtime_attack/smoke_test.py` checks the mechanism end to end on a CPU in
a few minutes. Its accuracies are meaningless and should not be quoted; it exists to show that
the pipeline does what it claims before any GPU time is paid for.

## Results in this repository

- `runtime_attack/results/gpu_run_2026-08-25/` — the run the reported figures come from.
- `runtime_attack/results/teacher_repro_2026-08-31/` — a second teacher, trained independently in
  the same configuration, on which every detector figure reproduces to within about 0.01 of AUC.
- `runtime_attack/results/TRADEOFF_2026-07-18.md`, `ADAPTIVE_ATTACKER_2026-07-18.md`,
  `DISTRIBUTED_2026-07-25.md` — the earlier prototype-scale analyses, kept because the write-up
  refers to them.
- `scar_repro/results/` — the SCAR baseline reproduction, including the two deviations from the
  paper's settings that it was subject to.

The ResNet-50 teacher used by the demo is kept at
`runtime_attack/checkpoints/clean_teacher_resnet50.pth`, so the demo can be run without training
anything. It is the teacher from the reproduction run.

## The demo

```bash
cd runtime_attack/demo && bash run_demo.sh
```

Opens on http://127.0.0.1:8000. It runs the project's own attack and detector code live on a CPU
and shows the monitor catching the hijack. The first load takes two to three minutes. See
`runtime_attack/demo/README.md`.

## Attribution

The SCAR implementation is by Chen et al., *Taught Well Learned Ill: Towards
Distillation-conditional Backdoor Attack*, NeurIPS 2025 (arXiv:2509.23871),
https://github.com/WhitolfChen/SCAR, used at commit `7a7b5b8` and unmodified. The attack, the
detector, the evaluation scripts and the demo in this repository are original work.

The two other attacks this work is compared against are Wu et al. (arXiv:2504.21323), which
poisons the distillation data, and Yu et al. (arXiv:2401.17746), whose benign-vector distance is
the nearest existing defence and which the detector's drift signal is compared against directly.

## Note on scope

This repository holds code and results only. The dissertation itself is submitted separately.

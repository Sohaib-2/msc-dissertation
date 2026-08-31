# Runtime output-compromise attack and teacher-side detector

The attack studied in this dissertation. A teacher model whose weights are trained normally and
never modified, and whose distillation set keeps its true labels, is compromised only at its
serving path: when a trigger appears in an input, the soft-labels returned for that input are
pushed toward a target class. A student distilled from that teacher inherits a backdoor.

The defence is a monitor that sits beside the teacher and watches only the outgoing stream of
soft-labels. It is never told the trigger or the target class.

Everything here is built on the SCAR harness in `../scar_baseline`, so the dataset, the models,
the KD loss and the ACC/ASR metrics are the ones the baseline uses and the numbers stay
comparable. The upstream clone is never modified.

## Results at full scale

ResNet-50 teacher, full CIFAR-10, 30 epochs; MobileNetV2 student distilled for 40 epochs;
poison rate 0.1; window of 200 outputs; thresholds at a 5% false-alarm rate.
Logs and result files are in `results/gpu_run_2026-08-25/`.

| alpha | detector AUC | TPR at 5% FPR | student ACC | student ASR |
|---|---|---|---|---|
| 0.0 (control) | 0.501 | 0.050 | 0.8415 | 0.0166 |
| 0.25 | 0.610 | 0.133 | 0.8445 | 0.0828 |
| 0.5 | 0.997 | 0.990 | 0.8386 | 0.1581 |
| 0.75 | 0.998 | 0.997 | 0.8469 | 0.2603 |
| 1.0 | 0.996 | 0.993 | 0.8431 | 0.6190 |

The clean teacher reaches 86.32% accuracy and gives 1.52% attack success on trigger-stamped
images, which is the baseline the attack has to beat.

Two things follow from the table. The student's clean accuracy never moves, so nothing
downstream reveals the attack; the teacher's output stream is the only place the signal exists.
And no operating point is both effective and hard to see. Dropping from alpha = 1.0 to alpha =
0.25 takes detection from 0.99 to 0.13 but takes attack success from 0.62 to 0.08, so hiding
costs the attacker most of the attack.

The detector figures were checked on a second, independently trained teacher of the same
configuration in `results/teacher_repro_2026-08-31/`. It reaches 86.49% clean accuracy and every
detector figure agrees with the original to within about 0.01 of AUC.

## Detection without choosing the signal afterwards

Quoting whichever signal scored highest on each condition is selection after the fact, so
`eval_prespecified.py` reports the signal fixed in advance beside a combined rule that is also
fixed in advance. At the stealthy alpha = 0.25 setting, across window sizes:

| window | pre-specified `max_class_excess` | combined rule | best signal, chosen afterwards |
|---|---|---|---|
| 100 | 0.500 | 0.436 | 0.547 |
| 200 | 0.499 | 0.517 | 0.613 |
| 500 | 0.502 | 0.680 | 0.723 |
| 1000 | 0.505 | 0.859 | 0.873 |
| 2000 | 0.511 | 0.980 | 0.982 |

The pre-specified signal does not improve with window size at all. It reads the histogram of
predicted classes, and a gentle hijack seldom changes which class is predicted, so a longer
window has nothing to accumulate. The combined rule does recover the attack, and it needs no
knowledge of the trigger, the target class or the results, though below a few hundred outputs it
is worse than chance because its standardisation is estimated too noisily at that length. At
alpha = 0.5 and above both saturate at 1.000 from 400 outputs onward.

## What the monitor costs

Scoring runs once per window, not once per query. On a CPU where the teacher answers in 33.6 ms
per query, the scoring pass costs 0.10 ms at a 100-output window and 0.37 ms at 2000, adding
under 0.003% to serving time. Against a teacher answering in 1 ms per query the overhead is
0.018% to 0.103%. A window holds 3.9 to 78 KiB. See `results/overhead_resnet50_cpu.md`.

## Files

- `train_teacher.py` — trains a clean teacher and saves it to `checkpoints/`. Run this first;
  everything else loads its output.
- `runtime_attack.py` — the attack. `FixedTriggerPoisoner` is a self-contained corner-patch
  trigger with the same call signature as SCAR's `Poisoner`, so the baseline's ASR measurement
  works unchanged. `hijack_soft_labels` implements the three rewrite rules: `soft`
  (`p' = (1-alpha)*p + alpha*onehot(target)`, the one used throughout), `hard` (a full override)
  and `logit` (a bias added to the target logit before the softmax).
  `RuntimeAttackDistiller` mirrors SCAR's response-based distiller, changing only the
  soft-labels of the trigger-stamped fraction of each batch.
- `detector.py` — the teacher-side detector. Calibrates on clean outputs, then scores unseen
  streams. Target-agnostic and trigger-agnostic by construction; see the module docstring.
- `eval_detector.py` — the detection half of the sweep: AUC and TPR at 5% FPR for each alpha.
  Needs no training, so it runs in minutes.
- `eval_combined.py` — the operational rule. Compares a signal fixed in advance against a
  standardised combination of all four, and against the after-the-fact best choice as an upper
  bound.
- `sweep_asr.py` — the attack half of the sweep: distils one student per alpha and reports
  accuracy and attack success.
- `distributed.py` — one compromised teacher against several students, in both directions: does
  the backdoor reach every student, and does pooling their query streams help the detector.
- `smoke_test.py` — a small CPU run that checks the mechanism end to end without a GPU.
- `eval_prespecified.py` — the window sweep without after-the-fact signal selection. Reports the
  signal fixed in advance, the deployable combined rule and the best-chosen-afterwards upper
  bound, at every window size and attack strength. The pooled-client result is read off the same
  sweep. Runs on a CPU.
- `measure_overhead.py` — what the monitor costs: scoring time per window against teacher
  inference time, memory per window, and the bytes a pooled deployment would carry.
- `make_figs.py` — regenerates the window-size and pooled-client figures.
- `measure_demo_rates.py` — measures the demo's true alarm rate per walkthrough step.
- `demo/` — a single-page live dashboard that runs the real pipeline. See `demo/README.md`.

## Running it

```bash
# mechanism check on a CPU, downloads CIFAR-10 into ./data
python smoke_test.py

# train the clean teacher (GPU)
python train_teacher.py --model resnet50 --n_train 0 --epochs 30 --device cuda

# detection half (no training)
python eval_detector.py --model resnet50 --n_test 0 --window 200 --device cuda

# attack half (one distillation per alpha)
python sweep_asr.py --model resnet50 --student mobilenetv2 --n_train 0 --epochs 40 --device cuda
```

`--n_train 0` and `--n_test 0` mean the full split rather than a subset. The scripts write both
a JSON and a Markdown table into `results/` after every alpha, so a long run can be inspected or
stopped part-way.

`data/` and any `runtime_attack_out/` are local artefacts and are not tracked.
